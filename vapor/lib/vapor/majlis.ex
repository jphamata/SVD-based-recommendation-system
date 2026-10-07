defmodule Vapor.Majlis do
  @moduledoc """
  The **majlis** (مجلس, root ج-ل-س *j-l-s*, "to sit together") —
  conversations with models as a content-addressed tree (docs/MAJLIS.md).

  Every message is an immutable node whose hash covers its text, its role,
  its metadata and **its parent's hash** — so a node commits to the whole
  conversation before it, as a git commit does. A *thread* is a mutable
  pointer (`head`) into that tree plus its settings. Everything a chat
  product does with history is then a pointer operation:

    * **edit** a message = a new sibling with the same parent (the old
      branch stays, reachable as "‹ 1/2 ›"); **regenerate** an answer = a new
      assistant sibling; **switch** branches = move the head to the newest
      leaf under a node; **rewind** = move the head back;
    * **fork** (clone a session) at any message = a new thread pointing at
      the same node — O(1), nothing copied, and the two can never disagree
      about the shared past because it is the same bytes;
    * **context** is computed, not remembered: the system prompt, the
      summary of a compacted prefix (which names the hash it summarises),
      pinned messages and the newest turns that fit the token budget — with
      what was dropped listed, never silently;
    * **share** = a computed capability (`Vapor.Khazana.mac/2`) for a
      read-only view; **revoke** = bump the thread's generation, every link
      dies at once;
    * **export/import**: vapor's format (hashes re-verified on import, so a
      tampered export is refused), and the conversation exports of ChatGPT
      (`conversations.json`, a tree) and Claude (a list), whose branches
      become branches here.

  State lives in a `Vapor.Khazana`: every operation is one crash-atomic
  commit. The server serialises writes; generation runs in the caller's
  process (`reply/3`), so a slow model never blocks the other threads.

  A reply comes from any `Vapor.Agent.Backend` (a local vapor engine,
  OpenAI-compatible, Anthropic, or a script). With tools enabled on the
  thread, the reply is an **agent run** (`Vapor.Agent.run/3`) over vapor's
  own tools (`Vapor.Majlis.Tools`: Alembic in its sandbox as the code
  interpreter, the deciders, the solvers); its hash-chained journal is
  stored and named by the answer, so "what did the model do to get this"
  is one lookup, and replayable.
  """
  use GenServer
  alias Vapor.Khazana, as: K

  @prefix "M1"

  # ================================================================ client

  @doc """
  Start a majlis over a store directory. Options: `dir:` (required),
  `name:`, `backends:` (`%{"name" => backend}`), `default:` (backend name),
  `count:` (`fun(text) → tokens`, exact; default ≈ bytes/4, marked
  approximate), `clock:` (`fun() → iso8601`, for tests), `tools:` (a
  `Vapor.Majlis.Tools` registry; default: vapor's own).
  """
  def start_link(opts) do
    gen = if opts[:name], do: [name: opts[:name]], else: []
    GenServer.start_link(__MODULE__, opts, gen)
  end

  def threads(m), do: GenServer.call(m, :threads)
  def thread(m, tid), do: GenServer.call(m, {:thread, tid})
  def new(m, opts \\ []), do: GenServer.call(m, {:new, Map.new(opts)})
  def set(m, tid, fields), do: GenServer.call(m, {:set, tid, Map.new(fields)})
  def delete(m, tid), do: GenServer.call(m, {:delete, tid})
  def say(m, tid, text, meta \\ %{}), do: GenServer.call(m, {:say, tid, text, meta})
  def edit(m, tid, node, text), do: GenServer.call(m, {:edit, tid, node, text})
  def switch(m, tid, node), do: GenServer.call(m, {:switch, tid, node})
  def rewind(m, tid, node), do: GenServer.call(m, {:rewind, tid, node})
  def fork(m, tid, opts \\ []), do: GenServer.call(m, {:fork, tid, Map.new(opts)})
  def pin(m, tid, node, on \\ true), do: GenServer.call(m, {:pin, tid, node, on})
  def path(m, tid), do: GenServer.call(m, {:path, tid})
  def tree(m, tid), do: GenServer.call(m, {:tree, tid})
  def node(m, id), do: GenServer.call(m, {:node, id})
  def context(m, tid, opts \\ []), do: GenServer.call(m, {:context, tid, Map.new(opts)})
  def search(m, query, k \\ 20), do: GenServer.call(m, {:search, query, k})
  def export(m, tid, format \\ :json), do: GenServer.call(m, {:export, tid, format})
  def import(m, data, format \\ :auto), do: GenServer.call(m, {:import, data, format}, 60_000)
  def share(m, tid), do: GenServer.call(m, {:share, tid})
  def revoke(m, tid), do: GenServer.call(m, {:revoke, tid})
  def shared(m, tid, token), do: GenServer.call(m, {:shared, tid, token})
  def journal(m, id), do: GenServer.call(m, {:journal, id})
  def gc(m), do: GenServer.call(m, :gc, 120_000)
  def stats(m), do: GenServer.call(m, :stats)
  def backends(m), do: GenServer.call(m, :backends)

  @doc """
  Generate the assistant's answer at the thread's head (or, with `at:`, as
  a new child of that node — which is what regenerating is). Runs in the
  caller. Options: `backend:` (a name), `max_tokens:`, `temperature:`,
  `seed:`, `tools:` (override the thread's), `budget:`.
  `{:ok, %{node, content, ...}}` or `{:error, why}`.
  """
  def reply(m, tid, opts \\ []) do
    with {:ok, job} <- GenServer.call(m, {:prepare, tid, Map.new(opts)}) do
      t0 = System.monotonic_time(:millisecond)

      result =
        try do
          generate(job)
        rescue
          e -> {:error, Exception.message(e)}
        catch
          :exit, why -> {:error, "the model stopped: #{inspect(why) |> String.slice(0, 200)}"}
        end

      case result do
        {:ok, content, meta, journal} ->
          meta = Map.merge(meta, %{"ms" => System.monotonic_time(:millisecond) - t0, "backend" => job.backend_name,
                                   "context" => %{"tokens" => job.context.tokens, "exact" => job.context.exact, "dropped" => length(job.context.dropped)}})
          GenServer.call(m, {:append_reply, tid, job.parent, content, meta, journal, job.regen})

        {:error, why} ->
          {:error, why}
      end
    end
  end

  @doc "Regenerate an assistant message: a new answer to the same question, as its sibling."
  def regenerate(m, tid, node, opts \\ []) do
    with {:ok, n} <- node(m, node),
         true <- n.role == "assistant" || {:error, "only an assistant message can be regenerated (edit a user message instead)"} do
      reply(m, tid, Keyword.put(opts, :at, n.parent))
    end
  end

  @doc "Say and reply in one call."
  def ask(m, tid, text, opts \\ []) do
    with {:ok, _} <- say(m, tid, text), do: reply(m, tid, opts)
  end

  # the model call, in the caller's process
  defp generate(%{tools: [], backend: b} = job) do
    gen = %{"max_tokens" => job.gen["max_tokens"], "temperature" => job.gen["temperature"] * 1.0, "seed" => job.gen["seed"], "now" => job.now}

    case Vapor.Agent.Backend.complete(b, job.context.messages, [], gen) do
      {:ok, %{content: c} = r} -> {:ok, c || "", %{"record" => Map.get(r, :record, %{}), "deterministic" => Map.get(r, :deterministic, false)}, nil}
      {:error, why} -> {:error, "the model did not answer: #{inspect(why) |> String.slice(0, 300)}"}
    end
  end

  defp generate(job) do
    registry = job.registry
    spec = Vapor.Agent.Spec.new(name: "majlis", model: job.model_descriptor, instructions: "",
                                 tools: Vapor.Majlis.Tools.specs(registry, job.tools),
                                 policy: %{"temperature" => job.gen["temperature"] * 1.0, "max_tokens" => job.gen["max_tokens"], "seed" => job.gen["seed"],
                                           "max_steps" => job.gen["max_steps"]})

    case Vapor.Agent.run(spec, job.context.messages, backend: job.backend, impls: Vapor.Majlis.Tools.impls(registry, job.tools), now: job.now) do
      {:ok, %{answer: answer, journal: j, status: status}} ->
        steps = for %{"kind" => "tool", "data" => d} <- j.events, do: %{"tool" => d["name"], "ok" => d["status"] == "ok"}
        {:ok, answer || "", %{"agent" => %{"status" => to_string(status), "steps" => steps, "run" => j.run_id}}, j}

      {:error, why, _} ->
        {:error, "the agent stopped: #{inspect(why) |> String.slice(0, 300)}"}
    end
  end

  # ================================================================ server

  @impl true
  def init(opts) do
    dir = Keyword.fetch!(opts, :dir)

    with {:ok, k} <- K.open(dir, create: true) do
      st = %{k: k, nodes: %{}, kids: %{}, pos: %{}, owned: %{}, fresh: [], backends: Keyword.get(opts, :backends, %{}), default: opts[:default],
             count: opts[:count], clock: Keyword.get(opts, :clock, fn -> DateTime.utc_now() |> DateTime.truncate(:millisecond) |> DateTime.to_iso8601() end),
             registry: Keyword.get_lazy(opts, :tools, fn -> Vapor.Majlis.Tools.default() end)}
      {:ok, index(st)}
    else
      {:error, why} -> {:stop, why}
    end
  end

  # the message index is rebuilt from the pack: a node is a blob that starts with "M1"
  defp index(st) do
    Enum.reduce(K.hashes(st.k), st, fn h, st ->
      case K.get(st.k, h) do
        {:ok, @prefix <> body} -> add_node(st, K.hex(h), decode!(body), K.position(st.k, h))
        _ -> st
      end
    end)
  end

  defp decode!(body), do: (fn {:ok, t} -> t end).(Vapor.Canonical.decode(body))

  defp add_node(st, id, n, pos) do
    parent = n["parent"] || ""
    st = %{st | nodes: Map.put(st.nodes, id, n), pos: Map.put(st.pos, id, pos), kids: Map.update(st.kids, parent, [id], &(&1 ++ [id])),
                owned: Map.update(st.owned, n["o"], [id], &[id | &1])}
    if pos == nil, do: %{st | fresh: [id | st.fresh]}, else: st
  end

  defp root(st), do: K.root(st.k) || %{"v" => 1, "threads" => %{}, "tseq" => 0}

  defp commit(st, root) do
    {:ok, k} = K.commit(st.k, root)
    # nodes staged by this operation now have positions: the order in which they were written
    pos = Enum.reduce(st.fresh, st.pos, fn id, acc -> Map.put(acc, id, K.position(k, elem(K.unhex(id), 1))) end)
    %{st | k: k, pos: pos, fresh: []}
  end

  # put a message node, written by thread `owner`; returns {id, st} (a node
  # already present is not stored twice). The owner is part of the node, so
  # it is part of its hash: who wrote a message cannot be changed afterwards.
  defp put_node(st, role, content, parent, meta, owner) do
    n = %{"k" => "msg", "role" => role, "content" => content, "parent" => parent, "t" => st.clock.(), "meta" => meta, "o" => owner}
    {h, k} = K.put(st.k, @prefix <> Vapor.Canonical.encode(n))
    id = K.hex(h)
    st = %{st | k: k}
    if Map.has_key?(st.nodes, id), do: {id, st}, else: {id, add_node(st, id, n, nil)}
  end

  @impl true
  def handle_call(msg, _from, st) do
    case op(msg, st, root(st)) do
      {:reply, r, st2} -> {:reply, r, st2}
      {:commit, r, st2, root2} -> {:reply, r, commit(st2, root2)}
    end
  rescue
    e -> {:reply, {:error, Exception.message(e)}, st}
  end

  # ------------------------------------------------------------- threads

  defp op(:threads, st, root) do
    list =
      root["threads"]
      |> Enum.map(fn {id, t} -> summary(st, id, t) end)
      |> Enum.sort_by(&{&1.updated, &1.id}, :desc)

    {:reply, list, st}
  end

  defp op({:thread, tid}, st, root), do: with_thread(st, root, tid, fn t -> {:reply, {:ok, summary(st, tid, t) |> Map.put(:settings, settings(t))}, st} end)

  defp op({:new, o}, st, root) do
    n = root["tseq"] + 1
    tid = "t" <> String.slice(K.mac(st.k, ["thread", n]), 0, 10) |> String.replace(~r/[^A-Za-z0-9]/, "x")
    now = st.clock.()
    budget = o[:budget] || 8192
    _ = normalize_setting("budget", budget)
    # the thread's anchor: an empty root node its first messages hang from, so
    # that editing the first message makes a branch inside this thread's tree
    {anchor, st} = put_node(st, "root", "", nil, %{}, tid)
    t = %{"title" => to_string(o[:title] || "untitled"), "head" => anchor, "anchor" => anchor, "owners" => [tid], "created" => now, "updated" => now,
          "system" => to_string(o[:system] || ""), "model" => o[:model] && to_string(o[:model]), "pins" => [], "summary" => nil, "forked_from" => nil,
          "cap" => 0, "tools" => Enum.map(List.wrap(o[:tools]), &to_string/1), "budget" => budget,
          "gen" => %{"max_tokens" => 1024, "temperature" => 0.7, "seed" => 1, "max_steps" => 6}}
    root = %{root | "tseq" => n, "threads" => Map.put(root["threads"], tid, t)}
    {:commit, {:ok, tid}, st, root}
  end

  defp op({:set, tid, f}, st, root) do
    with_thread(st, root, tid, fn t ->
      allowed = %{title: "title", system: "system", model: "model", tools: "tools", budget: "budget"}

      t =
        Enum.reduce(f, t, fn
          {:gen, g}, t -> Map.update!(t, "gen", &Map.merge(&1, Map.new(g, fn {k, v} -> {to_string(k), v} end)))
          {k, v}, t -> (case allowed[k] do nil -> raise ArgumentError, "unknown setting #{inspect(k)}"; key -> Map.put(t, key, normalize_setting(key, v)) end)
        end)

      {:commit, :ok, st, put_thread(root, tid, touch(t, st))}
    end)
  end

  defp op({:delete, tid}, st, root),
    do: with_thread(st, root, tid, fn _ -> {:commit, :ok, st, %{root | "threads" => Map.delete(root["threads"], tid)}} end)

  # ------------------------------------------------------------ messages

  defp op({:say, tid, text, meta}, st, root) do
    with_thread(st, root, tid, fn t ->
      text = check_text!(text)
      {id, st} = put_node(st, "user", text, t["head"], stringify(meta), tid)
      {:commit, {:ok, id}, st, put_thread(root, tid, touch(%{t | "head" => id}, st))}
    end)
  end

  defp op({:edit, tid, node, text}, st, root) do
    with_thread(st, root, tid, fn t ->
      with {:ok, id} <- resolve(st, node) do
        n = st.nodes[id]
        if n["role"] == "root", do: raise(ArgumentError, "the thread's anchor is not a message")
        text = check_text!(text)
        {new, st} = put_node(st, n["role"], text, n["parent"], Map.put(n["meta"] || %{}, "edited_from", id), tid)
        {:commit, {:ok, new}, st, put_thread(root, tid, touch(%{t | "head" => new}, st))}
      else
        e -> {:reply, e, st}
      end
    end)
  end

  defp op({:switch, tid, node}, st, root) do
    with_thread(st, root, tid, fn t ->
      case resolve(st, node) do
        {:ok, id} -> leaf = newest_leaf(st, id); {:commit, {:ok, leaf}, st, put_thread(root, tid, touch(%{t | "head" => leaf}, st))}
        e -> {:reply, e, st}
      end
    end)
  end

  defp op({:rewind, tid, node}, st, root) do
    with_thread(st, root, tid, fn t ->
      case resolve(st, node) do
        {:ok, id} -> {:commit, {:ok, id}, st, put_thread(root, tid, touch(%{t | "head" => id}, st))}
        e -> {:reply, e, st}
      end
    end)
  end

  defp op({:fork, tid, o}, st, root) do
    with_thread(st, root, tid, fn t ->
      at = if o[:at], do: resolve(st, o[:at]), else: {:ok, t["head"]}

      case at do
        {:ok, id} ->
          n = root["tseq"] + 1
          nid = "t" <> String.slice(K.mac(st.k, ["thread", n]), 0, 10) |> String.replace(~r/[^A-Za-z0-9]/, "x")
          now = st.clock.()
          title = to_string(o[:title] || t["title"] <> " (fork)")
          pins = Enum.filter(t["pins"], &on_path?(st, &1, id))
          summary = if t["summary"] && on_path?(st, t["summary"]["upto"], id), do: t["summary"]
          copy = %{t | "title" => title, "head" => id, "created" => now, "updated" => now, "pins" => pins, "summary" => summary,
                       "forked_from" => %{"thread" => tid, "node" => id}, "cap" => 0, "owners" => [nid]}
          {:commit, {:ok, nid}, st, %{root | "tseq" => n, "threads" => Map.put(root["threads"], nid, copy)}}

        e ->
          {:reply, e, st}
      end
    end)
  end

  defp op({:pin, tid, node, on}, st, root) do
    with_thread(st, root, tid, fn t ->
      case resolve(st, node) do
        {:ok, id} ->
          pins = if on, do: Enum.uniq(t["pins"] ++ [id]), else: List.delete(t["pins"], id)
          {:commit, :ok, st, put_thread(root, tid, touch(%{t | "pins" => pins}, st))}

        e ->
          {:reply, e, st}
      end
    end)
  end

  defp op({:path, tid}, st, root), do: with_thread(st, root, tid, fn t -> {:reply, {:ok, path_view(st, t)}, st} end)

  defp op({:tree, tid}, st, root) do
    with_thread(st, root, tid, fn t ->
      on = t["head"] |> path_ids(st) |> MapSet.new()
      vis = visible(st, t)
      nodes = tree_order(st, t, vis) |> Enum.reject(&(st.nodes[&1]["role"] == "root"))
      {:reply, {:ok, %{head: t["head"], nodes: Enum.map(nodes, &(view(st, &1, vis) |> Map.put(:on_path, MapSet.member?(on, &1))))}}, st}
    end)
  end

  defp op({:node, id}, st, _root) do
    case resolve(st, id) do
      {:ok, full} -> {:reply, {:ok, view(st, full)}, st}
      e -> {:reply, e, st}
    end
  end

  defp op({:context, tid, o}, st, root), do: with_thread(st, root, tid, fn t -> {:reply, {:ok, context_of(st, t, t["head"], o)}, st} end)

  # ------------------------------------------------------------- generation

  defp op({:prepare, tid, o}, st, root) do
    with_thread(st, root, tid, fn t ->
      at =
        case o[:at] do
          nil -> {:ok, t["head"]}
          a -> resolve(st, a)
        end

      with {:ok, parent} <- at,
           true <- (parent != nil and st.nodes[parent]["role"] != "root") || {:error, "the thread is empty: say something first"},
           true <- st.nodes[parent]["role"] in ["user", "tool"] || {:error, "the head is an answer already: say something, or regenerate it"},
           {:ok, name, backend} <- pick_backend(st, o[:backend] || t["model"]) do
        ctx = context_of(st, t, parent, o)
        gen = Map.merge(t["gen"], Map.new(Map.take(o, [:max_tokens, :temperature, :seed]), fn {k, v} -> {to_string(k), v} end))
        tools = Enum.map(List.wrap(o[:tools] || t["tools"]), &to_string/1)

        {:reply,
         {:ok, %{parent: parent, context: ctx, gen: gen, backend: backend, backend_name: name, tools: tools, registry: st.registry,
                 model_descriptor: %{"kind" => "majlis", "name" => name}, now: st.clock.(), regen: o[:at] != nil}}, st}
      else
        e -> {:reply, e, st}
      end
    end)
  end

  defp op({:append_reply, tid, parent, content, meta, journal}, st, root), do: op({:append_reply, tid, parent, content, meta, journal, false}, st, root)

  defp op({:append_reply, tid, parent, content, meta, journal, regen}, st, root) do
    with_thread(st, root, tid, fn t ->
      {meta, st} =
        case journal do
          nil -> {meta, st}
          j -> {h, k} = K.put(st.k, "J1" <> Vapor.Agent.Journal.encode(j)); {Map.put(meta, "journal", K.hex(h)), %{st | k: k}}
        end

      {id, st} = put_node(st, "assistant", to_string(content), parent, stringify(meta), tid)
      # the head follows an answer asked at the head; a regenerated one, when its question is on the
      # path in view. A plain answer whose question the person has typed past keeps the head where it is.
      t = if t["head"] == parent or (regen and on_path?(st, parent, t["head"])), do: %{t | "head" => id}, else: t
      {:commit, {:ok, %{node: id, content: content, meta: meta}}, st, put_thread(root, tid, touch(t, st))}
    end)
  end

  defp op({:journal, id}, st, _root) do
    with {:ok, h} <- K.unhex(id) |> then(fn :error -> {:error, "not a journal id"}; ok -> ok end),
         {:ok, "J1" <> body} <- K.get(st.k, h) |> then(fn :error -> {:error, "no such journal"}; ok -> ok end) do
      {:reply, Vapor.Agent.Journal.decode(body), st}
    else
      e -> {:reply, e, st}
    end
  end

  # ------------------------------------------------------------- compaction

  defp op({:compact, tid, upto, text, by}, st, root) do
    with_thread(st, root, tid, fn t ->
      case resolve(st, upto) do
        {:ok, id} ->
          if on_path?(st, id, t["head"]) do
            s = %{"upto" => id, "text" => text, "by" => by, "covers" => length(msg_ids(id, st)), "t" => st.clock.()}
            {:commit, :ok, st, put_thread(root, tid, touch(%{t | "summary" => s}, st))}
          else
            {:reply, {:error, "that message is not on the thread's current path"}, st}
          end

        e ->
          {:reply, e, st}
      end
    end)
  end

  defp op({:uncompact, tid}, st, root), do: with_thread(st, root, tid, fn t -> {:commit, :ok, st, put_thread(root, tid, %{t | "summary" => nil})} end)

  # ----------------------------------------------------------------- search

  defp op({:search, query, k}, st, root) do
    nodes = for {id, n} <- st.nodes, n["role"] != "root", into: %{}, do: {id, n}
    {:reply, Vapor.Majlis.Search.run(nodes, root["threads"], &msg_ids(&1, st), query, k), st}
  end

  # ----------------------------------------------------- export and import

  defp op({:export, tid, format}, st, root) do
    with_thread(st, root, tid, fn t ->
      {:reply, {:ok, Vapor.Majlis.Exchange.export(format, tid, t, path_view(st, t), tree_nodes(st, t))}, st}
    end)
  end

  defp op({:import, data, format}, st, root) do
    case Vapor.Majlis.Exchange.parse(data, format) do
      {:ok, convs} ->
        {st, root, ids} =
          Enum.reduce(convs, {st, root, []}, fn conv, {st, root, ids} ->
            n = root["tseq"] + 1
            tid = "t" <> String.slice(K.mac(st.k, ["thread", n]), 0, 10) |> String.replace(~r/[^A-Za-z0-9]/, "x")
            now = st.clock.()
            {st, map, anchor, owners} = import_nodes(st, conv.nodes, tid)
            head = map[conv.head] || anchor
            t = %{"title" => conv.title, "head" => head, "anchor" => anchor, "owners" => owners, "created" => conv[:created] || now, "updated" => now,
                  "system" => conv[:system] || "", "model" => nil, "pins" => [], "summary" => nil, "forked_from" => nil, "cap" => 0, "tools" => [],
                  "budget" => 8192, "gen" => %{"max_tokens" => 1024, "temperature" => 0.7, "seed" => 1, "max_steps" => 6}, "imported" => conv.source}
            {st, %{root | "tseq" => n, "threads" => Map.put(root["threads"], tid, t)}, ids ++ [tid]}
          end)

        {:commit, {:ok, ids}, st, root}

      e ->
        {:reply, e, st}
    end
  end

  # -------------------------------------------------------------- sharing

  defp op({:share, tid}, st, root), do: with_thread(st, root, tid, fn t -> {:reply, {:ok, K.mac(st.k, ["share", tid, t["cap"]])}, st} end)

  defp op({:revoke, tid}, st, root), do: with_thread(st, root, tid, fn t -> {:commit, :ok, st, put_thread(root, tid, %{t | "cap" => t["cap"] + 1})} end)

  defp op({:shared, tid, token}, st, root) do
    case root["threads"][tid] do
      %{} = t ->
        if K.mac_ok?(st.k, ["share", tid, t["cap"]], token),
          do: {:reply, {:ok, %{title: t["title"], messages: Enum.map(path_view(st, t).messages, &Map.take(&1, [:id, :role, :content, :t]))}}, st},
          else: {:reply, {:error, :forbidden}, st}

      nil ->
        {:reply, {:error, :forbidden}, st}
    end
  end

  # ------------------------------------------------------------ housekeeping

  defp op(:gc, st, root) do
    live =
      for {_, t} <- root["threads"], id <- visible(st, t), reduce: MapSet.new() do
        acc ->
          acc = MapSet.put(acc, id)
          case st.nodes[id]["meta"]["journal"] do
            j when is_binary(j) -> MapSet.put(acc, j)
            _ -> acc
          end
      end

    hashes = Enum.map(live, &elem(K.unhex(&1), 1))
    {:ok, k, rep} = K.gc(st.k, hashes)
    st2 = index(%{st | k: k, nodes: %{}, kids: %{}, pos: %{}, owned: %{}, fresh: []})
    {:reply, {:ok, rep}, st2}
  end

  defp op(:stats, st, root) do
    msgs = Enum.count(st.nodes, fn {_, n} -> n["role"] != "root" end)
    {:reply, %{threads: map_size(root["threads"]), messages: msgs, seq: st.k.seq, pack_bytes: st.k.len}, st}
  end

  defp op({:backend, name}, st, _root), do: {:reply, pick_backend(st, name), st}

  defp op(:backends, st, _root), do: {:reply, %{names: Map.keys(st.backends), default: st.default}, st}

  @doc "Install a summary of the prefix ending at `upto` (written by a person, or produced by `summarize/4`)."
  def compact(m, tid, upto, text, by \\ "person") when is_binary(text), do: GenServer.call(m, {:compact, tid, upto, text, by})

  @doc "Remove a thread's summary: the full history goes back into the context."
  def uncompact(m, tid), do: GenServer.call(m, {:uncompact, tid})

  @doc """
  Ask the thread's model to summarise the path up to `upto` (default: the
  message before the last user turn), and install it. The summary names the
  hash it covers — which commits to every message before it.
  """
  def summarize(m, tid, upto \\ nil, opts \\ []) do
    with {:ok, p} <- path(m, tid),
         msgs = p.messages,
         true <- length(msgs) >= 3 || {:error, "nothing to compact yet"},
         upto_id = upto || Enum.at(msgs, -3).id,
         {:ok, %{id: id}} <- node(m, upto_id),
         covered = Enum.take_while(msgs, &(&1.id != id)) ++ [Enum.find(msgs, &(&1.id == id))],
         {:ok, t} <- thread(m, tid),
         {:ok, name, backend} <- GenServer.call(m, {:backend, opts[:backend] || t.settings["model"]}) do
      transcript = Enum.map_join(covered, "\n\n", &"#{&1.role}: #{&1.content}")
      prompt = [%{"role" => "system", "content" => "Summarise the conversation below for your own later use: keep every fact, decision, number, name and open question; drop pleasantries. Write in the conversation's language."},
                %{"role" => "user", "content" => transcript}]

      case Vapor.Agent.Backend.complete(backend, prompt, [], %{"max_tokens" => opts[:max_tokens] || 600, "temperature" => 0.0, "seed" => 1}) do
        {:ok, %{content: text}} -> with :ok <- compact(m, tid, id, text, name), do: {:ok, %{upto: id, text: text}}
        {:error, why} -> {:error, "the model did not summarise: #{inspect(why) |> String.slice(0, 200)}"}
      end
    end
  end

  @impl true
  def handle_info(_, st), do: {:noreply, st}

  # ================================================================ helpers

  defp with_thread(st, root, tid, f) do
    case root["threads"][tid] do
      nil -> {:reply, {:error, "no thread #{inspect(tid)}"}, st}
      t -> f.(t)
    end
  end

  defp put_thread(root, tid, t), do: %{root | "threads" => Map.put(root["threads"], tid, t)}
  defp touch(t, st), do: %{t | "updated" => st.clock.()}

  defp normalize_setting("tools", v), do: Enum.map(List.wrap(v), &to_string/1)
  defp normalize_setting("budget", v) when is_integer(v) and v >= 256 and v <= 2_000_000, do: v
  defp normalize_setting("budget", v), do: raise(ArgumentError, "budget: tokens between 256 and 2 000 000, not #{inspect(v)}")
  defp normalize_setting("model", nil), do: nil
  defp normalize_setting(_, v), do: to_string(v)

  defp check_text!(t) when is_binary(t) and byte_size(t) <= 4_000_000 do
    if String.valid?(t), do: t, else: raise(ArgumentError, "text must be UTF-8")
  end

  defp check_text!(_), do: raise(ArgumentError, "text: a string of at most 4 MB")

  defp stringify(m) when is_map(m), do: Map.new(m, fn {k, v} -> {to_string(k), stringify(v)} end)
  defp stringify(l) when is_list(l), do: Enum.map(l, &stringify/1)
  defp stringify(a) when is_atom(a) and a not in [nil, true, false], do: Atom.to_string(a)
  defp stringify(v), do: v

  defp settings(t), do: Map.take(t, ["system", "model", "tools", "budget", "gen", "pins", "summary", "forked_from", "imported"])

  defp summary(st, id, t) do
    p = msg_ids(t["head"], st)
    last = List.last(p)
    preview = if last, do: st.nodes[last]["content"] |> String.slice(0, 120), else: ""
    %{id: id, title: t["title"], head: t["head"], messages: length(p), updated: t["updated"], created: t["created"], preview: preview,
      forked_from: t["forked_from"], shared_generation: t["cap"]}
  end

  @doc false
  def resolve_in(nodes, ref) do
    ref = to_string(ref)

    cond do
      Map.has_key?(nodes, ref) -> {:ok, ref}
      byte_size(ref) < 6 or not Regex.match?(~r/^[0-9a-f]+$/, ref) -> {:error, "no message #{inspect(ref)} (give at least 6 hex digits of its id)"}
      true ->
        case Enum.filter(Map.keys(nodes), &String.starts_with?(&1, ref)) do
          [one] -> {:ok, one}
          [] -> {:error, "no message #{ref}"}
          many -> {:error, "#{ref} is ambiguous (#{length(many)} messages)"}
        end
    end
  end

  defp resolve(st, ref), do: resolve_in(st.nodes, ref)

  defp path_ids(nil, _st), do: []

  defp path_ids(id, st) do
    Stream.unfold(id, fn nil -> nil; i -> {i, st.nodes[i]["parent"]} end) |> Enum.to_list() |> Enum.reverse()
  end

  defp on_path?(st, id, head), do: id in path_ids(head, st)

  # the path without the thread's anchor: the messages
  defp msg_ids(head, st), do: head |> path_ids(st) |> Enum.reject(&(st.nodes[&1]["role"] == "root"))

  # what a thread can see: the nodes its owners wrote, their ancestors (a
  # fork's shared past) and the current path — not a sibling thread's branches
  defp visible(st, t) do
    own = Enum.flat_map(t["owners"] || [], &Map.get(st.owned, &1, []))
    Enum.reduce(own, MapSet.new(path_ids(t["head"], st)), fn id, acc ->
      if MapSet.member?(acc, id), do: acc, else: Enum.reduce(path_ids(id, st), acc, &MapSet.put(&2, &1))
    end)
  end

  # the visible nodes in tree order (parents first, branches in the order they were written)
  defp tree_order(st, t, vis) do
    walk = fn walk, id -> [id | Enum.flat_map(Enum.filter(children(st, id), &MapSet.member?(vis, &1)), &walk.(walk, &1))] end
    case t["anchor"] && MapSet.member?(vis, t["anchor"]) do
      true -> walk.(walk, t["anchor"])
      _ -> []
    end
  end

  defp children(st, id), do: st.kids |> Map.get(id || "", []) |> Enum.sort_by(&(st.pos[&1] || :infinity))

  # the newest leaf under a node: at every step, the most recently written child
  defp newest_leaf(st, id) do
    case children(st, id) do
      [] -> id
      kids -> newest_leaf(st, List.last(kids))
    end
  end

  defp tree_nodes(st, t), do: Enum.map(tree_order(st, t, visible(st, t)), &Map.put(st.nodes[&1], "id", &1))

  defp view(st, id, vis \\ nil) do
    n = st.nodes[id]
    sibs = children(st, n["parent"]) |> then(&if(vis, do: Enum.filter(&1, fn s -> MapSet.member?(vis, s) end), else: &1))
    kids = children(st, id) |> then(&if(vis, do: Enum.filter(&1, fn s -> MapSet.member?(vis, s) end), else: &1))
    parent = if st.nodes[n["parent"]]["role"] == "root", do: nil, else: n["parent"]
    %{id: id, role: n["role"], content: n["content"], parent: parent, t: n["t"], meta: n["meta"] || %{}, thread: n["o"],
      siblings: sibs, index: (Enum.find_index(sibs, &(&1 == id)) || 0) + 1, children: kids}
  end

  defp path_view(st, t) do
    vis = visible(st, t)
    ids = msg_ids(t["head"], st)
    head = if st.nodes[t["head"]]["role"] == "root", do: nil, else: t["head"]
    %{head: head, messages: Enum.map(ids, &(view(st, &1, vis) |> Map.put(:pinned, &1 in t["pins"])))}
  end

  # ---------------------------------------------------------------- context

  defp count(st, text) do
    case st.count do
      nil -> {div(byte_size(text) + 3, 4) + 4, false}
      f -> {f.(text) + 4, true}
    end
  end

  # what the model will see at `head`: system + summary + pins + the newest turns that fit
  defp context_of(st, t, head, o) do
    budget = o[:budget] || t["budget"]
    ids = msg_ids(head, st)

    {summary, ids_after} =
      case t["summary"] do
        %{"upto" => u, "text" => text} = s ->
          if u in ids, do: {s |> Map.put("text", text), Enum.drop_while(ids, &(&1 != u)) |> Enum.drop(1)}, else: {nil, ids}

        _ ->
          {nil, ids}
      end

    summarized = if summary, do: ids -- ids_after, else: []
    sys_text = [t["system"], summary && "Summary of the conversation before this point (it covers #{summary["covers"]} messages, up to #{String.slice(summary["upto"], 0, 12)}):\n" <> summary["text"]]
               |> Enum.reject(&(&1 in [nil, ""])) |> Enum.join("\n\n")
    {sys_tokens, exact} = if sys_text == "", do: {0, true}, else: count(st, sys_text)

    sized = Enum.map(ids_after, fn id -> {n, _} = count(st, st.nodes[id]["content"]); {id, n} end)
    pins = MapSet.new(t["pins"])
    last = List.last(ids_after)

    # the last turn and the pins are required; then the newest of the rest, while they fit
    required = for {id, n} <- sized, id == last or MapSet.member?(pins, id), do: {id, n}
    req_tokens = sys_tokens + Enum.sum(Enum.map(required, &elem(&1, 1)))

    {kept, used} =
      sized
      |> Enum.reverse()
      |> Enum.reject(fn {id, _} -> id == last or MapSet.member?(pins, id) end)
      |> Enum.reduce({MapSet.new(Enum.map(required, &elem(&1, 0))), req_tokens}, fn {id, n}, {keep, used} ->
        if used + n <= budget, do: {MapSet.put(keep, id), used + n}, else: {keep, used}
      end)

    items =
      Enum.map(summarized, &%{id: &1, role: st.nodes[&1]["role"], tokens: 0, status: "summarized"}) ++
        Enum.map(sized, fn {id, n} ->
          status = cond do
            MapSet.member?(pins, id) -> "pinned"
            MapSet.member?(kept, id) -> "sent"
            true -> "dropped"
          end
          %{id: id, role: st.nodes[id]["role"], tokens: n, status: status}
        end)

    msgs =
      (if sys_text != "", do: [%{"role" => "system", "content" => sys_text}], else: []) ++
        for {id, _} <- sized, MapSet.member?(kept, id), do: %{"role" => st.nodes[id]["role"], "content" => st.nodes[id]["content"]}

    %{messages: msgs, items: items, tokens: used, budget: budget, exact: exact and st.count != nil, over: used > budget,
      dropped: for(i <- items, i.status == "dropped", do: i.id), system_tokens: sys_tokens, summary: summary && Map.take(summary, ["upto", "covers", "by"])}
  end

  # ----------------------------------------------------------------- backends

  defp pick_backend(st, name) do
    name = name || st.default || List.first(Enum.sort(Map.keys(st.backends)))

    case st.backends[name] do
      nil -> {:error, if(st.backends == %{}, do: "no model is configured (start the server with a model, or set VAPOR_MIND)", else: "no model #{inspect(name)} (have: #{Enum.join(Map.keys(st.backends), ", ")})")}
      b -> {:ok, name, b}
    end
  end

  # ------------------------------------------------------------------ import

  # nodes arrive as [%{"key", "parent_key", "role", "content", "t", "meta"}] in
  # parent-before-child order. A vapor export brings its own anchor and its
  # nodes verbatim (their hashes are checked, so their owners stay the
  # exporting thread's — which joins the new thread's owners); other
  # products' conversations get an anchor here and are owned by the new thread.
  defp import_nodes(st, nodes, tid) do
    verbatim = Enum.any?(nodes, &Map.has_key?(&1, "verify"))

    {anchor, st} = if verbatim, do: {nil, st}, else: put_node(st, "root", "", nil, %{"imported" => true}, tid)

    {st, map} =
      Enum.reduce(nodes, {st, %{}}, fn n, {st, map} ->
      parent = (n["parent_key"] && map[n["parent_key"]]) || anchor
      meta = Map.merge(n["meta"] || %{}, %{"imported" => true})
      node = %{"k" => "msg", "role" => n["role"], "content" => n["content"], "parent" => parent, "t" => n["t"] || st.clock.(), "meta" => meta, "o" => tid}

      node =
        case n["verify"] do
          nil -> node
          expected ->
            computed = K.hex(:crypto.hash(:sha256, @prefix <> Vapor.Canonical.encode(Map.delete(n["verify_node"], "id"))))
            if computed != expected, do: raise(ArgumentError, "message #{String.slice(expected, 0, 12)} does not match its content: the export was altered")
            Map.delete(n["verify_node"], "id")
        end

      {h, k} = K.put(st.k, @prefix <> Vapor.Canonical.encode(node))
      id = K.hex(h)
      st = %{st | k: k}
      st = if Map.has_key?(st.nodes, id), do: st, else: add_node(st, id, node, nil)
      {st, Map.put(map, n["key"], id)}
      end)

    if verbatim do
      anchor = Enum.find(Map.values(map), &(st.nodes[&1]["role"] == "root"))
      if anchor == nil, do: raise(ArgumentError, "the export has no thread anchor")
      owners = [tid | map |> Map.values() |> Enum.map(&st.nodes[&1]["o"]) |> Enum.uniq()]
      {st, map, anchor, owners}
    else
      {st, map, anchor, [tid]}
    end
  end
end
