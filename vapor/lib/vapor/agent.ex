defmodule Vapor.Agent do
  @moduledoc """
  Immutable agents: an agent is a value (`Vapor.Agent.Spec`), a run is a
  value (`Vapor.Agent.Journal`), and replaying a run is a proof.

      spec = Spec.new(name: "weather", model: %{"kind" => "local", "id" => model_id},
                      instructions: "…", tools: [%{name: "get_weather", effect: "observe", parameters: …}])
      {:ok, run} = Vapor.Agent.run(spec, "Weather in Recife?", backend: backend, impls: %{"get_weather" => &fetch/2})
      Vapor.Agent.replay(spec, run.journal, backend: backend, impls: impls)
      #=> {:ok, %{verified: 3, observed: 1, redacted: 0, mismatches: []}}

  A run alternates model decisions and tool executions, and every one of
  them is an event of the hash-chained journal:

    * `start` — the spec's digest, the input, the run's clock (`now`: the
      only time anything in the run may read; a template's `strftime_now`
      reads it), the run nonce;
    * `model` — the decision (content and tool calls), whether it is
      re-derivable, and its record (tokens and receipt for a local model;
      provider and response id for a remote one);
    * `intent` — before an `act` tool runs: the call, its idempotency key and
      how it was approved (written, and persisted by a store, *before* the
      world is touched); `retry` if a crash interrupted it;
    * `tool` — the call, its effect class, the result or error, and for
      `act` tools the idempotency key the implementation received;
    * `final` / `halt` — the answer, or why the run stopped.

  Determinism makes the journal do more than log:

    * **Replay verifies.** `replay/3` rebuilds the run from the journal:
      every local model decision is *recomputed* and must give the recorded
      tokens bit for bit (vapor's cross-substrate invariance is what makes
      this a meaningful check on another machine); `pure` tools are re-run
      and must agree; `observe` and `act` results — and remote models'
      answers — are read from the record and never re-executed. The rebuilt
      journal must end at the recorded head. Replay never touches the world.
    * **Resume is replay.** A run that crashed resumes from its journal
      (`run(…, journal: prefix)`): the prefix is replayed (verified, no side
      effects repeated), then the run continues live — only if the prefix
      verified (`{:error, {:diverged, mismatches}, _}` otherwise, before any
      new event) — durable execution without a workflow engine. `act` implementations receive an
      idempotency key derived from (run, step, call), the same on every
      attempt, so the outside world can apply a side effect at most once.
    * **Capabilities are part of identity.** An `act` tool runs only if the
      spec grants it (and, with `confirm:`, a human approves — the approval
      or the refusal is recorded). Text a tool reads (a web page, an e-mail) can steer the
      model, but cannot grant a capability: the grants are in the spec's
      digest, fixed before the run started.
    * **Erasure is possible.** With `keys:` and `subject:`, the input, the
      model's texts and tokens, call arguments, tool results and errors are
      sealed per data subject (tool names, effect classes, call ids and
      receipts stay readable for audit);
      shredding the subject's key erases them everywhere while the chain
      stays verifiable (`Vapor.Agent.Journal`).
  """
  alias Vapor.Agent.{Backend, Journal, Spec, Validate}
  alias Vapor.Canonical

  @doc """
  Run an agent on an input. Options: `backend:` (required), `impls:`
  (`%{tool_name => fun(args, ctx)}`, `ctx = %{idempotency_key:, run_id:,
  step:}`, returning `{:ok, json_value}` or `{:error, reason}`), `now:`
  (default: the current UTC time — recorded), `nonce:`, `confirm:`
  (`fun(call) → boolean` for `act` tools), `keys:` and `subject:` (sealing),
  `journal:` (a recorded prefix to resume from), `on_event:` (see below).
  Returns `{:ok, %{answer, journal, run_id, status}}`.

  `on_event:` — `fun(event, journal) → :ok | {:error, reason}`, or a list
  of them — is called for every *new* event, after it is appended and
  before the run goes on (events of a resumed prefix are not new). It is
  the write-ahead point: a `Vapor.Agent.Store` persists there
  (`Vapor.Agent.Store.hook/1`), a UI subscribes there (a Phoenix PubSub
  broadcast for a LiveView). A hook that refuses stops the run with
  `{:error, {:on_event, reason}, state}`, so nothing happens in the world
  that the store did not record first.
  """
  def run(%Spec{} = spec, input, opts) do
    case opts[:journal] do
      %Journal{events: []} -> {:error, :empty_journal, nil}
      %Journal{} = prefix -> drive(prefix_state(spec, prefix, opts), :resume)
      nil -> drive(fresh_state(spec, input, opts), :live)
    end
  catch
    {:on_event, why, st} -> {:error, {:on_event, why}, st}
  end

  @doc """
  Verify a recorded run against its spec (see the moduledoc). Options as
  `run/3` (`backend:` for re-deriving local decisions, `impls:` for pure
  tools, `keys:` to open sealed values). Returns `{:ok, report}` when the
  rebuilt journal matches the record, `{:error, report}` otherwise;
  `report = %{verified, observed, redacted, mismatches, head}`.
  """
  def replay(%Spec{} = spec, %Journal{} = j, opts) do
    with :ok <- Journal.verify(j),
         true <- j.events != [] || {:error, :empty} do
      # verifying records nothing new: no hook can fire
      st = prefix_state(spec, j, Keyword.delete(opts, :on_event))
      {:ok, st} = drive(st, :verify)
      rep = Map.put(st.report, :head, st.journal.head == j.head)
      if rep.mismatches == [] and rep.head, do: {:ok, rep}, else: {:error, rep}
    else
      {:error, why} -> {:error, %{tampered: why}}
    end
  end

  # ------------------------------------------------------------- states --

  defp base(spec, opts) do
    %{spec: spec, digest: Spec.digest(spec), backend: Keyword.get(opts, :backend), impls: Keyword.get(opts, :impls, %{}),
      keys: opts[:keys], subject: opts[:subject], confirm: opts[:confirm], messages: [], step: 0,
      report: %{verified: 0, observed: 0, redacted: 0, mismatches: []}, pending: [], tainted: false,
      on_event: List.wrap(opts[:on_event]), recorded: 0}
  end

  defp fresh_state(spec, input, opts) do
    now = Keyword.get_lazy(opts, :now, fn -> DateTime.utc_now() end) |> iso()
    nonce = Keyword.get_lazy(opts, :nonce, fn -> Base.encode16(:crypto.strong_rand_bytes(8), case: :lower) end)
    st = base(spec, opts)
    run_id = Canonical.hex_digest({:run, st.digest, now, nonce})
    st = Map.merge(st, %{run_id: run_id, now: now, journal: Journal.new(run_id)})
    start = %{"spec" => st.digest, "input" => seal(st, input), "now" => now, "nonce" => nonce}
    # the data subject's key id (pseudonymous), so a replay seals as the run did
    start = if st.keys && st.subject, do: Map.put(start, "subject", st.subject), else: start
    st |> record("start", start) |> begin(input)
  end

  # replaying (or resuming): the recorded events are consumed one by one
  defp prefix_state(spec, %Journal{events: [%{"kind" => "start", "data" => d} | rest]} = j, opts) do
    st = base(spec, opts)
    st = Map.merge(st, %{run_id: j.run_id, now: d["now"], journal: Journal.new(j.run_id), pending: rest,
                         subject: st.subject || d["subject"], recorded: length(j.events)})
    st = record(st, "start", d)

    st =
      if d["spec"] != st.digest,
        do: mismatch(st, 0, :spec, "journal was produced by agent #{String.slice(d["spec"], 0, 12)}…, not #{String.slice(st.digest, 0, 12)}…"),
        else: st

    case open(st, d["input"]) do
      {:ok, input} -> begin(st, input)
      {:redacted, _} -> %{st | report: bump(st.report, :redacted), messages: :redacted}
      {:error, :tampered} -> %{mismatch(st, 0, :sealed, "the input does not open under its subject's key") | messages: :redacted}
    end
  end

  defp begin(st, input) do
    sys = if st.spec.instructions != "", do: [%{"role" => "system", "content" => st.spec.instructions}], else: []
    %{st | messages: sys ++ [%{"role" => "user", "content" => input}]}
  end

  # --------------------------------------------------------------- loop --

  defp drive(st, mode) do
    cond do
      st.pending == [] and mode == :verify -> {:ok, st}
      # a resumed run goes live only from a prefix that verified: a different
      # spec, a recomputed decision that differs, an event out of order — stop
      # before anything new happens
      mode == :resume and st.report.mismatches != [] -> {:error, {:diverged, st.report.mismatches}, st}
      finished?(st) -> {:ok, result(st)}
      # the history cannot be opened (erased, or no keys given): it can be
      # verified as recorded, never continued
      mode != :verify and st.messages == :redacted and st.pending == [] -> {:error, :history_redacted, st}
      match?([%{"kind" => "halt"} | _], st.pending) ->
        [e | rest] = st.pending
        drive(%{record(st, "halt", e["data"]) | pending: rest}, mode)
      st.step >= st.spec.policy["max_steps"] and st.pending == [] ->
        st = record(st, "halt", %{"reason" => "max_steps"})
        {:ok, result(st)}
      true ->
        case model_step(st, mode) do
          {:ok, st} -> drive(st, mode)
          {:error, why, st} -> {:error, why, st}
        end
    end
  end

  defp finished?(st), do: match?([_ | _], st.journal.events) and List.last(st.journal.events)["kind"] in ["final", "halt"] and st.pending == []

  defp result(st) do
    last = List.last(st.journal.events)
    answer =
      with "final" <- last["kind"],
           {:ok, a} <- open(st, last["data"]["answer"]) do
        a
      else
        _ -> nil
      end

    %{answer: answer, journal: st.journal, run_id: st.run_id, status: last["kind"]}
  end

  defp model_step(%{messages: :redacted} = st, _mode) do
    # history unrecoverable: the remaining events are taken as recorded
    st = Enum.reduce(st.pending, st, fn e, st -> %{record(st, e["kind"], e["data"]) | report: bump(st.report, :redacted)} end)
    {:ok, %{st | pending: []}}
  end

  defp model_step(st, mode) do
    step = st.step + 1
    opts = st.spec.policy |> Map.put("seed", seed(st, step)) |> Map.put("now", parse_now(st.now))
    tools = Spec.openai_tools(st.spec)

    {decision, st} =
      case st.pending do
        [%{"kind" => "model", "data" => rec} | rest] ->
          st = %{st | pending: rest}

          cond do
            rec["deterministic"] and st.backend != nil and mode in [:verify, :resume] ->
              case Backend.complete(st.backend, st.messages, tools, opts) do
                {:ok, out} ->
                  fresh = model_data(st, step, out)
                  if fresh == rec,
                    do: {rec, %{st | report: bump(st.report, :verified)}},
                    else: {rec, mismatch(st, step, :model, "recomputed decision differs from the record")}

                {:error, why} ->
                  {rec, mismatch(st, step, :model, "could not recompute: #{inspect(why)}")}
              end

            true ->
              {rec, %{st | report: bump(st.report, :observed)}}
          end

        [] ->
          case Backend.complete(st.backend, st.messages, tools, opts) do
            {:ok, out} -> {model_data(st, step, out), st}
            {:error, why} -> throw({:backend, why})
          end

        [other | _] ->
          {nil, mismatch(st, step, :order, "expected a model event, found #{other["kind"]}")}
      end

    if decision == nil do
      {:ok, %{st | pending: []}}
    else
      st = %{record(st, "model", decision) | step: step}

      with {:ok, content} <- open(st, decision["content"]),
           {:ok, calls} <- open_calls(st, decision["calls"]) do
        after_decision(st, step, decision, content, calls, mode)
      else
        {:redacted, _} -> {:ok, %{st | messages: :redacted, report: bump(st.report, :redacted)}}
        {:error, :tampered} -> {:ok, %{mismatch(st, step, :sealed, "a sealed value does not open under its subject's key") | messages: :redacted}}
      end
    end
  catch
    {:backend, why} -> {:error, {:backend, why}, st}
  end

  # resuming from a prefix that did not verify: stop here, before any tool —
  # drive/2 reports the divergence
  defp after_decision(%{report: %{mismatches: [_ | _]}} = st, _step, _decision, _content, _calls, :resume), do: {:ok, %{st | pending: []}}

  defp after_decision(st, step, decision, content, calls, mode) do
      if calls == [] do
        st = case st.pending do
          [%{"kind" => k} = e | rest] when k in ["final", "halt"] -> %{record(st, k, e["data"]) | pending: rest}
          _ -> record(st, "final", %{"answer" => decision["content"]})
        end

        {:ok, st}
      else
        assistant = %{"role" => "assistant", "content" => content,
                      "tool_calls" => for(c <- calls, do: %{"id" => c["id"], "type" => "function", "function" => %{"name" => c["name"], "arguments" => c["arguments"]}})}

        st = %{st | messages: st.messages ++ [assistant]}
        {:ok, Enum.reduce(Enum.with_index(calls), st, fn {call, i}, st -> tool_step(st, step, i, call, mode) end)}
      end
  end

  # what the model said is the subject's data: its text, the arguments of its
  # calls and the tokens behind them are sealed; the structure (tool names,
  # call ids, receipts) stays in the clear for audit
  defp model_data(st, step, out) do
    %{"step" => step, "content" => seal(st, out.content),
      "calls" => Enum.map(out.calls, fn c -> c |> Map.take(["id", "name", "arguments"]) |> Map.update("arguments", %{}, &seal(st, &1)) end),
      "deterministic" => out.deterministic,
      "record" => Enum.reduce(["tokens", "prompt", "receipt"], out.record, fn k, r -> if is_map_key(r, k), do: Map.update!(r, k, &seal(st, &1)), else: r end)}
  end

  defp open_calls(st, calls) do
    Enum.reduce_while(calls, {:ok, []}, fn c, {:ok, acc} ->
      case open(st, c["arguments"]) do
        {:ok, args} -> {:cont, {:ok, acc ++ [Map.put(c, "arguments", args)]}}
        redacted -> {:halt, redacted}
      end
    end)
  end

  # one tool call: checked against the spec, executed by its effect class.
  # An `act` tool is preceded by an `intent` event — written (and, with a
  # store, persisted) *before* the world is touched: two processes resuming
  # the same run collide on it, so only one of them acts
  defp tool_step(%{report: %{mismatches: [_ | _]}} = st, _step, _i, _call, :resume), do: %{st | pending: []}

  defp tool_step(st, step, i, call, mode) do
    tool = Enum.find(st.spec.tools, &(&1["name"] == call["name"]))
    key = Canonical.hex_digest({:idempotency, st.run_id, step, i, call["name"], call["arguments"]})

    {data, st} =
      case st.pending do
        [%{"kind" => "intent", "data" => irec} | rest] ->
          st = %{st | pending: rest}
          st = if Map.delete(irec, "approval") == intent(call, key), do: st, else: mismatch(st, step, :intent, "recorded intent differs for #{call["name"]}")
          st = record(st, "intent", irec) |> retries(intent(call, key))

          case st.pending do
            [%{"kind" => "tool"} | _] -> recorded_tool(st, tool, call, key, step)
            [] when mode == :verify -> {nil, mismatch(st, step, :order, "journal ends after the intent of #{call["name"]}")}
            # crashed between the intent and its result: the action may have
            # happened. It is attempted again with the same idempotency key,
            # after a `retry` event — which a concurrent resumer collides on
            # (never after a mismatch: a resumed run acts only from a verified prefix)
            [] when mode == :resume and st.report.mismatches != [] -> {nil, %{st | pending: []}}
            [] ->
              st = record(st, "retry", intent(call, key))
              {invoke(st, tool, call, key, step), st}
            [other | _] -> {nil, mismatch(st, step, :order, "expected the result of #{call["name"]}, found #{other["kind"]}")}
          end

        [%{"kind" => "tool"} | _] ->
          recorded_tool(st, tool, call, key, step)

        [] when mode == :verify ->
          {nil, mismatch(st, step, :order, "journal ends before tool call #{call["name"]}")}

        [] ->
          case check(st, tool, call) do
            {:refused, outcome} -> {outcome_data(st, tool, call, key, outcome), st}
            {:go, approval} ->
              st = if tool["effect"] == "act", do: record(st, "intent", Map.put(intent(call, key), "approval", approval)), else: st
              {invoke(st, tool, call, key, step), st}
          end

        [other | _] ->
          {nil, mismatch(st, step, :order, "expected tool call #{call["name"]}, found #{other["kind"]}")}
      end

    if data == nil do
      st
    else
      st = record(st, "tool", data)
      {body, st} = case open(st, data[if(data["status"] == "ok", do: "result", else: "error")]) do
        {:ok, v} -> {v, st}
        {:redacted, _} -> {"[redacted]", st}
        {:error, :tampered} -> {"[unreadable]", mismatch(st, step, :sealed, "the result of #{call["name"]} does not open under its subject's key")}
      end

      content = if data["status"] == "ok", do: Vapor.JSON.encode(body), else: Vapor.JSON.encode(%{"error" => body})
      %{st | messages: st.messages ++ [%{"role" => "tool", "tool_call_id" => call["id"], "name" => call["name"], "content" => content}]}
    end
  end

  defp intent(call, key), do: %{"call_id" => call["id"], "name" => call["name"], "idempotency_key" => key}

  # recorded attempts after a crash, taken as recorded
  defp retries(%{pending: [%{"kind" => "retry", "data" => d} | rest]} = st, expected) do
    st = if d == expected, do: st, else: mismatch(st, 0, :intent, "recorded retry differs for #{expected["name"]}")
    retries(record(%{st | pending: rest}, "retry", d), expected)
  end

  defp retries(st, _expected), do: st

  # a recorded result: pure tools are re-run and must agree; the rest is read
  defp recorded_tool(%{pending: [%{"data" => rec} | rest]} = st, tool, call, key, step) do
    st = %{st | pending: rest}

    if tool && tool["effect"] == "pure" and Map.has_key?(st.impls, tool["name"]) and rec["status"] == "ok" do
      fresh = invoke(st, tool, call, key, step)
      if fresh == rec, do: {rec, %{st | report: bump(st.report, :verified)}}, else: {rec, mismatch(st, step, :tool, "pure tool #{call["name"]} disagrees with the record")}
    else
      {rec, %{st | report: bump(st.report, :observed)}}
    end
  end

  # whether the call may run: `{:go, approval}` or `{:refused, outcome}`
  defp check(st, tool, call) do
    cond do
      tool == nil -> {:refused, {:error, "unknown tool"}}
      (problem = Validate.check(tool["parameters"], call["arguments"])) != :ok -> {:refused, {:error, "invalid arguments: #{problem}"}}
      tool["effect"] == "act" and tool["name"] not in st.spec.grants -> {:refused, {:denied, "capability not granted by the agent's spec"}}
      tool["effect"] == "act" and st.confirm != nil and not st.confirm.(call) -> {:refused, {:denied, "not approved"}}
      not Map.has_key?(st.impls, tool["name"]) -> {:refused, {:error, "no implementation bound"}}
      tool["effect"] == "act" and st.confirm != nil -> {:go, "approved"}
      true -> {:go, "granted"}
    end
  end

  defp invoke(st, tool, call, key, step) do
    outcome =
      try do
        st.impls[tool["name"]].(call["arguments"], %{idempotency_key: key, run_id: st.run_id, step: step})
      rescue
        e -> {:error, Exception.message(e)}
      end

    outcome_data(st, tool, call, key, outcome)
  end

  defp outcome_data(st, tool, call, key, outcome) do
    base = %{"call_id" => call["id"], "name" => call["name"], "arguments" => seal(st, call["arguments"]), "effect" => tool && tool["effect"]}

    case outcome do
      # results and errors may carry the subject's data whatever the effect
      # class (an error message may quote the arguments): sealed alike
      {:ok, v} -> Map.merge(base, %{"status" => "ok", "result" => seal(st, v)} |> put_key(tool, key))
      {:denied, why} -> Map.merge(base, %{"status" => "denied", "error" => why})
      {:error, why} -> Map.merge(base, %{"status" => "error", "error" => seal(st, to_string_safe(why))} |> put_key(tool, key))
      other -> Map.merge(base, %{"status" => "error", "error" => seal(st, "implementation returned #{inspect(other)}")} |> put_key(tool, key))
    end
  end

  defp put_key(m, %{"effect" => "act"}, key), do: Map.put(m, "idempotency_key", key)
  defp put_key(m, _tool, _key), do: m

  defp to_string_safe(why) when is_binary(why), do: why
  defp to_string_safe(why), do: inspect(why)

  # ------------------------------------------------------------ helpers --

  defp record(st, kind, data) do
    st = %{st | journal: Journal.append(st.journal, kind, data)}
    seq = length(st.journal.events) - 1

    # only events beyond the recorded prefix are new
    if seq >= st.recorded do
      event = List.last(st.journal.events)

      for hook <- st.on_event do
        case hook.(event, st.journal) do
          :ok -> :ok
          {:error, why} -> throw({:on_event, why, st})
          other -> throw({:on_event, other, st})
        end
      end
    end

    st
  end

  defp mismatch(st, step, what, why), do: %{st | report: %{st.report | mismatches: st.report.mismatches ++ [%{step: step, what: what, why: why}]}}

  defp bump(rep, k), do: Map.update!(rep, k, &(&1 + 1))

  # the per-step seed: a function of the spec's seed, the run and the step
  defp seed(st, step) do
    <<s::56, _::binary>> = Canonical.digest({:seed, st.spec.policy["seed"], st.run_id, step})
    s
  end

  defp seal(%{keys: nil}, v), do: v
  defp seal(%{subject: nil}, v), do: v
  defp seal(st, v), do: Journal.seal(v, st.subject, st.keys, st.run_id)

  defp open(%{keys: nil}, %{"sealed" => s}), do: {:redacted, s}

  # {:ok, value} | {:redacted, subject} (no key: erased, or not given) |
  # {:error, :tampered} (the key exists and the ciphertext does not open)
  defp open(st, v), do: Journal.unseal(v, st.keys, st.run_id)

  defp iso(%DateTime{} = d), do: DateTime.to_iso8601(d)
  defp iso(%NaiveDateTime{} = d), do: NaiveDateTime.to_iso8601(d)
  defp iso(s) when is_binary(s), do: s

  defp parse_now(s) do
    case DateTime.from_iso8601(s) do
      {:ok, d, _} -> d
      _ -> NaiveDateTime.from_iso8601!(s)
    end
  end
end

defmodule Vapor.Agent.Validate do
  @moduledoc """
  JSON Schema validation of tool arguments (the subset of
  `Vapor.Grammar.JSONSchema`), for decisions that were not constrained —
  remote models may call tools with arguments their schema refuses; the
  agent refuses them too, and says why, before anything runs.
  """

  @doc "`:ok` or a description of the first violation."
  def check(schema, v), do: check(schema, v, "$", defs(schema))

  defp defs(%{} = s), do: Map.merge(s["$defs"] || %{}, s["definitions"] || %{})
  defp defs(_), do: %{}

  defp check(true, _v, _p, _d), do: :ok
  defp check(nil, _v, _p, _d), do: :ok

  defp check(%{"$ref" => "#/" <> ref}, v, p, d), do: check(d[ref |> String.split("/") |> List.last()], v, p, d)
  defp check(%{"const" => c}, v, p, _d), do: if(v == c, do: :ok, else: "#{p}: must be #{inspect(c)}")
  defp check(%{"enum" => e}, v, p, _d), do: if(v in e, do: :ok, else: "#{p}: must be one of #{inspect(e)}")

  defp check(%{"anyOf" => alts}, v, p, d), do: any(alts, v, p, d)
  defp check(%{"oneOf" => alts}, v, p, d), do: any(alts, v, p, d)
  defp check(%{"type" => types} = s, v, p, d) when is_list(types), do: any(Enum.map(types, &Map.put(s, "type", &1)), v, p, d)

  defp check(%{"type" => "object"} = s, v, p, d) when is_map(v) do
    props = s["properties"] || %{}
    missing = Enum.find(s["required"] || [], &(not Map.has_key?(v, &1)))
    extra = if s["additionalProperties"] == false, do: Enum.find(Map.keys(v), &(not Map.has_key?(props, &1)))

    cond do
      missing -> "#{p}: missing #{missing}"
      extra -> "#{p}: unexpected #{extra}"
      true -> Enum.find_value(v, :ok, fn {k, x} -> case check(Map.get(props, k, s["additionalProperties"]), x, "#{p}.#{k}", d) do :ok -> nil; e -> e end end)
    end
  end

  defp check(%{"type" => "array"} = s, v, p, d) when is_list(v) do
    cond do
      length(v) < (s["minItems"] || 0) -> "#{p}: at least #{s["minItems"]} items"
      s["maxItems"] && length(v) > s["maxItems"] -> "#{p}: at most #{s["maxItems"]} items"
      true -> v |> Enum.with_index() |> Enum.find_value(:ok, fn {x, i} -> case check(s["items"], x, "#{p}[#{i}]", d) do :ok -> nil; e -> e end end)
    end
  end

  defp check(%{"type" => "string"} = s, v, p, _d) when is_binary(v) do
    # JSON Schema lengths count code points (not graphemes)
    n = length(String.to_charlist(v))
    cond do
      n < (s["minLength"] || 0) -> "#{p}: at least #{s["minLength"]} characters"
      s["maxLength"] && n > s["maxLength"] -> "#{p}: at most #{s["maxLength"]} characters"
      true -> :ok
    end
  end

  defp check(%{"type" => "integer"}, v, _p, _d) when is_integer(v), do: :ok
  defp check(%{"type" => "number"}, v, _p, _d) when is_number(v), do: :ok
  defp check(%{"type" => "boolean"}, v, _p, _d) when is_boolean(v), do: :ok
  defp check(%{"type" => "null"}, nil, _p, _d), do: :ok
  defp check(%{"type" => t}, v, p, _d) when is_binary(t), do: "#{p}: #{inspect(v)} is not #{t}"
  defp check(%{} = s, v, p, d) when map_size(s) > 0 and is_map_key(s, "properties"), do: check(Map.put(s, "type", "object"), v, p, d)
  defp check(_s, _v, _p, _d), do: :ok

  defp any(alts, v, p, d), do: if(Enum.any?(alts, &(check(&1, v, p, d) == :ok)), do: :ok, else: "#{p}: matches none of the alternatives")
end
