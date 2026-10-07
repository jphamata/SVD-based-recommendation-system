defmodule Vapor.Athanor do
  @moduledoc """
  **Athanor** — the alchemist's furnace that keeps a steady fire for as
  long as the work takes. Here: one search engine for any problem written
  in Alembic (docs/ATHANOR.md).

  Every system that "discovers" — sorting routines, matrix-multiplication
  schemes, game strategies, proofs, programs — reduces to the same four
  parts: a **space** of candidates, **proposers** that suggest candidates,
  a **verifier** that scores them independently of who proposed, and a
  **budget**. The Athanor is those four parts with nothing fixed:

    * the space and the verifier are the user's (`Vapor.Athanor.Spec`);
    * the proposers are a portfolio (`Vapor.Athanor.Strategy`) — exhaustive
      enumeration, annealing, evolution and MAP-Elites, CMA-ES, Bayesian
      optimisation, a language model, a person — and a bandit gives the
      next batch of the budget to whichever is improving;
    * every evaluation goes into a **journal** chained by SHA-256, so a run
      is a replayable object, and the result is a **certificate**
      (`Vapor.Athanor.Touchstone`) that anyone re-checks without trusting
      the search;
    * the same budget is spent by **random search** as a control: the
      certificate says how much better than chance the search did, and
      what the probability was that chance alone would get there.

  A run is a pure function of (problem, seed, outside proposals), so it
  can be paused, inspected, steered by a person (`propose/3`, `pin/2`,
  `ban/2`) and resumed (`step/2`); `run/2` does it all at once.
  """
  alias Vapor.Alembic
  alias Vapor.Athanor.{Space, Spec, Strategy}

  @archive 24
  @batch 16

  defstruct spec: nil, strategies: %{}, order: [], cache: %{}, archive: [], elites: %{}, best: nil, evals: 0, invalid: 0, errors: 0,
            dups: 0, round: 0, journal: "", recent: [], observed: [], trace: [], stats: %{}, injected: [], queue: [], pins: MapSet.new(),
            bans: MapSet.new(), sparks: [], status: :running, reason: nil, opts: [], t0: 0, pending: [], exhaustive: false, error_samples: []

  # ================================================================ API

  @doc """
  Parse, search and certify in one call, inside a sandbox (capped heap,
  wall-clock limit). `{:ok, certificate}` or `{:error, message}`.

  Options: `budget:`, `seed:`, `consts:` (override constants), `seconds:`
  (time limit), `control:` (false to skip the random control), `only:`
  (a list of strategy names), `mind:` (a `Vapor.Mind` handle),
  `measure:` (`fn x -> {:ok, number} | {:error, why} end` for measured
  objectives), `heap_mb:`, `on_round:` (called with the run after each round).
  """
  def run(text, opts \\ []) do
    timeout = trunc((Keyword.get(opts, :seconds, 300) + 60) * 1000)
    case Alembic.sandbox(fn -> do_run(text, opts) end, heap_mb: Keyword.get(opts, :heap_mb, 1024), timeout: timeout) do
      {:ok, r} -> r
      {:error, :memory} -> {:error, "the search used more memory than allowed (heap_mb) — make candidates or the verifier smaller"}
      {:error, :timeout} -> {:error, "the search took longer than its time limit"}
      {:error, {:crash, why}} -> {:error, "the search crashed: #{inspect(why) |> String.slice(0, 300)}"}
    end
  end

  defp do_run(text, opts) do
    with {:ok, spec} <- Spec.parse(text, Keyword.take(opts, [:consts, :budget, :seed, :measured, :sense])) do
      r = init(spec, opts)
      # candidates proposed from outside (an agent through MCP, a pipeline): checked, recorded, evaluated like any other
      r = case Keyword.get(opts, :proposals, []) do [] -> r; ps -> propose(r, ps, Keyword.get(opts, :proposals_source, :mind)) end
      r = step(r, :all)
      {:ok, certificate(r)}
    end
  end

  @doc "A fresh run for a parsed spec."
  def init(%Spec{} = spec, opts \\ []) do
    size = spec.space.size
    exhaustive = is_integer(size) and size <= spec.budget and size <= Keyword.get(opts, :max_exhaustive, 5_000_000) and not spec.measured and
                   Keyword.get(opts, :only) in [nil, [:exhaustive]] and spec.start == []
    names = if exhaustive, do: [:exhaustive], else: Strategy.applicable(spec, opts)
    strategies = Map.new(names, &{&1, Strategy.new(&1, spec, spec.seed, opts)})
    r = %__MODULE__{spec: spec, strategies: strategies, order: names, opts: opts, t0: System.monotonic_time(:millisecond),
                    journal: Vapor.Canonical.hex_digest({:athanor, spec.hash, spec.seed}), exhaustive: exhaustive,
                    stats: Map.new(names ++ [:human, :start], &{&1, %{pulls: 0, evals: 0, improved: 0, archived: 0}})}
    if spec.start != [], do: propose(r, spec.start, :start), else: r
  end

  @doc "Run `rounds` rounds (or `:all` until the run stops)."
  def step(%__MODULE__{status: :running} = r, :all) do
    r = one_round(r)
    if r.status == :running, do: step(r, :all), else: r
  end

  def step(r, :all), do: r
  def step(r, 0), do: r
  def step(%__MODULE__{status: :running} = r, n) when is_integer(n), do: r |> one_round() |> step(n - 1)
  def step(r, _), do: r

  @doc "Queue candidates proposed from outside (`source`: `:human`, `:mind`, `:start`); they are checked against the space."
  def propose(r, values, source \\ :human) do
    {ok, bad} =
      values
      |> Enum.map(fn v -> {v, Space.check(r.spec.space, v)} end)
      |> Enum.split_with(fn {_, c} -> match?({:ok, _}, c) end)

    queued = Enum.map(ok, fn {_, {:ok, x}} -> {source, x} end)
    r = %{r | queue: r.queue ++ queued, injected: r.injected ++ Enum.map(queued, fn {s, x} -> %{round: r.round, source: s, candidate: Space.show(r.spec.space, x)} end)}
    r = if bad != [], do: %{r | error_samples: Enum.take(Enum.map(bad, fn {v, {:error, m}} -> "#{inspect_short(v)}: #{m}" end) ++ r.error_samples, 10)}, else: r
    if r.status == :stopped and r.reason in ["budget"], do: r, else: %{r | status: if(r.status == :done and queued != [], do: :running, else: r.status)}
  end

  @doc "Keep a candidate in the archive whatever its score (a person's judgement)."
  def pin(r, key), do: %{r | pins: MapSet.put(r.pins, key)}

  @doc "Never evaluate (or keep) a candidate again."
  def ban(r, key), do: %{r | bans: MapSet.put(r.bans, key), archive: Enum.reject(r.archive, &(&1.key == key))}

  @doc "Stop the run (it can be resumed with `resume/1`)."
  def stop(r), do: %{r | status: :stopped, reason: r.reason || "stopped by the user"}
  def resume(%{status: :stopped} = r), do: %{r | status: :running, reason: nil}
  def resume(r), do: r

  @doc "Give more budget to a run that ran out."
  def extend(r, more) do
    spec = %{r.spec | budget: r.spec.budget + more}
    %{r | spec: spec, status: if(r.reason == "budget", do: :running, else: r.status), reason: if(r.reason == "budget", do: nil, else: r.reason)}
  end

  @doc "Record a measurement for a pending candidate of a measured problem (`key` = its text)."
  def measure(r, key, value) when is_number(value) do
    case Enum.find(r.pending, &(&1.key == key)) do
      nil -> {:error, "no pending candidate #{key}"}
      p ->
        res = finish_measured(r.spec, p, {:ok, value})
        r = %{r | pending: Enum.reject(r.pending, &(&1.key == key))}
        {:ok, record(r, [res], p.strategy)}
    end
  end

  # ================================================================ a round

  defp one_round(r) do
    cond do
      r.pending != [] -> r
      r.evals >= r.spec.budget -> done(r, :stopped, "budget")
      deadline?(r) -> done(r, :stopped, "time")
      true -> do_round(r)
    end
  end

  defp deadline?(r) do
    case Keyword.get(r.opts, :seconds) do
      nil -> false
      s -> System.monotonic_time(:millisecond) - r.t0 > s * 1000
    end
  end

  defp do_round(r) do
    {name, xs, r} = choose(r)
    left = r.spec.budget - r.evals
    space = r.spec.space

    {fresh, r} =
      Enum.reduce(xs, {[], r}, fn x, {acc, r} ->
        x = if name in [:human, :mind, :start] or r.spec.neighbor, do: (case Space.check(space, x) do {:ok, y} -> y; _ -> nil end), else: Space.canon(space, x)
        key = x && Space.show(space, x)
        cond do
          x == nil -> {acc, %{r | invalid: r.invalid + 1}}
          MapSet.member?(r.bans, key) -> {acc, r}
          Map.has_key?(r.cache, key) or Enum.any?(acc, &(&1.key == key)) -> {acc, %{r | dups: r.dups + 1}}
          length(acc) >= left -> {acc, r}
          true -> {[%{x: x, key: key} | acc], r}
        end
      end)

    fresh = Enum.reverse(fresh)
    best_before = r.best && r.best.fitness

    {results, r} =
      if r.spec.measured do
        case Keyword.get(r.opts, :measure) do
          nil -> {[], %{r | pending: Enum.map(fresh, &Map.put(&1, :strategy, name))}}
          f -> {Enum.map(fresh, fn c -> finish_measured(r.spec, c, safe_measure(f, c)) end), r}
        end
      else
        {Enum.map(fresh, &evaluate(r.spec, &1, r.opts)), r}
      end

    r = record(r, results, name)
    st = Strategy.observe(Map.get(r.strategies, name, %{name: name, pulls: 0, reward: 0.0}), results, best_before)
    improved = Enum.count(results, &(&1.status == :ok and (best_before == nil or &1.fitness > best_before)))
    reward = if xs == [], do: 0.0, else: min((improved + 0.1 * Enum.count(results, &(&1[:rank] != nil and in_archive?(r, &1.key)))) / max(length(xs), 1), 1.0)
    st = %{st | pulls: Map.get(st, :pulls, 0) + 1, reward: Map.get(st, :reward, 0.0) + reward}
    r = if Map.has_key?(r.strategies, name), do: %{r | strategies: r.strategies |> discount() |> Map.put(name, st)}, else: r
    r = update_stats(r, name, fn s -> %{s | pulls: s.pulls + 1} end)
    r = %{r | round: r.round + 1}
    if is_function(r.opts[:on_round], 1), do: r.opts[:on_round].(r)
    stop_checks(r, name)
  end

  defp in_archive?(r, key), do: Enum.any?(r.archive, &(&1.key == key))

  defp safe_measure(f, c) do
    try do
      f.(c.x)
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp choose(%{queue: [_ | _] = q} = r) do
    {now, later} = Enum.split(q, @batch)
    src = now |> hd() |> elem(0)
    {same, other} = Enum.split_with(now, &(elem(&1, 0) == src))
    {src, Enum.map(same, &elem(&1, 1)), %{r | queue: other ++ later}}
  end

  defp choose(r) do
    name = pick_strategy(r)
    st = r.strategies[name]
    n = case name do :bayes -> 4; :mind -> 8; _ -> if(r.spec.measured or r.spec.budget <= 200, do: 4, else: @batch) end
    ctx = %{spec: r.spec, archive: r.archive, elites: r.elites, best: r.best, observed: r.observed}
    {xs, st} = Strategy.propose(st, ctx, n)
    r = %{r | strategies: Map.put(r.strategies, name, st)}
    # a strategy with nothing left to propose (exhaustion finished) ends the run
    if xs == [] and name == :exhaustive, do: {name, [], %{r | strategies: Map.put(r.strategies, name, %{st | cont: :done})}}, else: {name, xs, r}
  end

  # discounted UCB (Garivier & Moulines 2011): the past fades, so a strategy that was good early
  # (random sampling, typically) does not keep its share once it stops improving
  @gamma 0.97
  defp discount(strategies), do: Map.new(strategies, fn {k, s} -> {k, %{s | pulls: s.pulls * @gamma, reward: s.reward * @gamma}} end)

  # UCB over the strategies; each is tried once first
  defp pick_strategy(r) do
    names = r.order |> Enum.reject(&(&1 == :mind and (r.strategies[:mind].failures || 0) >= 3))
    untried = Enum.find(names, &(r.strategies[&1].pulls == 0 and r.stats[&1].pulls == 0))
    if untried do
      untried
    else
      total = Enum.sum(Enum.map(names, &r.strategies[&1].pulls))
      Enum.max_by(names, fn n ->
        s = r.strategies[n]
        p = max(s.pulls, 1.0e-3)
        s.reward / p + 0.35 * :math.sqrt(2 * :math.log(max(total, 1.0001)) / p)
      end)
    end
  end

  defp stop_checks(r, name) do
    best = r.best
    cond do
      r.spec.sense == :claim and best != nil and best.refutes -> done(r, :done, "counterexample")
      r.spec.sense == :find and best != nil -> done(r, :done, "found")
      best != nil and target_reached?(r.spec, best.value) -> done(r, :done, "target")
      name == :exhaustive and r.strategies[:exhaustive].cont == :done -> done(r, :done, "exhausted")
      r.evals >= r.spec.budget -> done(r, :stopped, "budget")
      true -> r
    end
  end

  defp target_reached?(%{target: nil}, _), do: false
  defp target_reached?(%{sense: :min, target: t}, v), do: is_number(v) and v <= t
  defp target_reached?(%{sense: :max, target: t}, v), do: is_number(v) and v >= t
  defp target_reached?(_, _), do: false

  defp done(r, status, reason), do: %{r | status: status, reason: reason}

  # ================================================================ evaluation

  @doc "Evaluate one candidate `%{x, key}` against the spec: `%{…, status, value, fitness}`."
  def evaluate(spec, %{x: x} = c, opts \\ []) do
    fuel = Keyword.get(opts, :fuel, Alembic.default_fuel())
    arg = Space.realize(spec.space, x)
    p = spec.prog
    native = Keyword.get(opts, :objective)

    result =
      with :ok <- check_valid(spec, p, arg, fuel) do
        case spec.sense do
          :find -> {:ok, 0.0, 0.0, %{}}
          :claim -> claim(spec, p, arg, fuel)
          sense when is_function(native, 1) ->
            case native.(arg) do
              {:ok, v} when is_number(v) -> {:ok, v, if(sense == :min, do: -v, else: v), %{}}
              {:error, m} -> {:error, to_string(m)}
            end

          sense ->
            if spec.objective == nil do
              {:error, "the objective is measured outside: give a measurement"}
            else
              case Alembic.call(p, spec.objective, [arg], fuel: fuel) do
                {:ok, v} when is_number(v) -> {:ok, v, if(sense == :min, do: -v, else: v), %{}}
                {:ok, v} -> {:error, "#{spec.objective} returned #{Vapor.Alembic.Builtins.type(v)}, not a number"}
                {:error, m} -> {:error, m}
              end
            end
        end
      end

    case result do
      {:ok, v, f, extra} -> Map.merge(c, Map.merge(%{status: :ok, value: v, fitness: f * 1.0, rank: {1, f * 1.0}, refutes: false}, extra))
      :invalid -> infeasible(spec, c, arg, fuel)
      {:error, m} -> Map.merge(c, %{status: :error, value: nil, fitness: nil, error: m})
    end
  end

  # an invalid candidate with a `violation(x)` keeps a rank below every valid one, so the search can climb toward validity
  defp infeasible(%{violation: true} = spec, c, arg, fuel) do
    case Alembic.call(spec.prog, "violation", [arg], fuel: fuel) do
      {:ok, v} when is_number(v) -> Map.merge(c, %{status: :invalid, value: nil, fitness: nil, rank: {0, -v * 1.0}, violation: v})
      _ -> Map.merge(c, %{status: :invalid, value: nil, fitness: nil})
    end
  end

  defp infeasible(_spec, c, _arg, _fuel), do: Map.merge(c, %{status: :invalid, value: nil, fitness: nil})

  defp check_valid(%{valid: false}, _p, _arg, _fuel), do: :ok

  defp check_valid(_spec, p, arg, fuel) do
    if Alembic.defined?(p, "valid", 1) do
      case Alembic.call(p, "valid", [arg], fuel: fuel) do
        {:ok, v} -> if Vapor.Alembic.Compiler.truthy(v), do: :ok, else: :invalid
        {:error, m} -> {:error, "valid: " <> m}
      end
    else
      case Alembic.call(p, "violation", [arg], fuel: fuel) do
        {:ok, v} when is_number(v) -> if v <= 0, do: :ok, else: :invalid
        {:ok, v} -> {:error, "violation returned #{Vapor.Alembic.Builtins.type(v)}, not a number"}
        {:error, m} -> {:error, "violation: " <> m}
      end
    end
  end

  defp claim(spec, p, arg, fuel) do
    case Alembic.call(p, "claim", [arg], fuel: fuel) do
      {:ok, holds} ->
        refutes = not Vapor.Alembic.Compiler.truthy(holds)
        margin =
          if spec.margin do
            case Alembic.call(p, "margin", [arg], fuel: fuel) do
              {:ok, m} when is_number(m) -> m
              _ -> nil
            end
          end
        f = cond do refutes -> 1.0e300; is_number(margin) -> -margin * 1.0; true -> 0.0 end
        {:ok, if(refutes, do: 1, else: 0), f, %{refutes: refutes, margin: margin}}

      {:error, m} -> {:error, "claim: " <> m}
    end
  end

  defp finish_measured(spec, c, {:ok, v}) when is_number(v) do
    with :ok <- check_valid(spec, spec.prog, Space.realize(spec.space, c.x), Alembic.default_fuel()) do
      f = if spec.sense == :max, do: v * 1.0, else: -v * 1.0
      Map.merge(Map.drop(c, [:strategy]), %{status: :ok, value: v, fitness: f, rank: {1, f}, refutes: false})
    else
      :invalid -> Map.merge(Map.drop(c, [:strategy]), %{status: :invalid, value: nil, fitness: nil})
      {:error, m} -> Map.merge(Map.drop(c, [:strategy]), %{status: :error, value: nil, fitness: nil, error: m})
    end
  end

  defp finish_measured(_spec, c, other) do
    m = case other do {:error, m} -> to_string(m); v -> "measurement #{inspect_short(v)} is not a number" end
    Map.merge(Map.drop(c, [:strategy]), %{status: :error, value: nil, fitness: nil, error: m})
  end

  # ================================================================ bookkeeping

  defp record(r, results, name) do
    Enum.reduce(results, r, fn res, r ->
      i = r.evals + 1
      entry = [i, Atom.to_string(name), res.key, Atom.to_string(res.status), if(res.value == nil, do: nil, else: Alembic.show(res.value))]
      journal = :crypto.hash(:sha256, r.journal <> Vapor.Canonical.encode(entry)) |> Base.encode16(case: :lower)
      res = Map.merge(res, %{i: i, by: name})
      r = %{r | evals: i, journal: journal, cache: Map.put(r.cache, res.key, cache_entry(res)), recent: Enum.take([slim(res) | r.recent], 40)}
      r = update_stats(r, name, fn s -> %{s | evals: s.evals + 1} end)

      case res.status do
        :invalid ->
          r = %{r | invalid: r.invalid + 1}
          if res[:rank], do: archive(r, res, name), else: r
        :error -> %{r | errors: r.errors + 1, error_samples: Enum.take(r.error_samples ++ ["#{res.key}: #{res.error}"], 10)}
        :ok ->
          improved = r.best == nil or res.fitness > r.best.fitness
          r = if improved, do: update_stats(%{r | best: res, trace: [[i, res.value, Atom.to_string(name)] | r.trace]}, name, &%{&1 | improved: &1.improved + 1}), else: r
          r = %{r | observed: Enum.take(r.observed ++ [%{x: res.x, fitness: res.fitness, status: :ok}], -200), sparks: spark(r.sparks, [i, res.value, Atom.to_string(name)])}
          r = archive(r, res, name)
          elites(r, res)
      end
    end)
  end

  # every valid evaluation as [index, value, strategy], thinned to at most 4000 (every other one dropped when full)
  defp spark(list, s) when length(list) < 4000, do: [s | list]
  defp spark(list, s), do: [s | list |> Enum.take_every(2)]

  @doc "The sparks (valid evaluations) after index `since`, oldest first."
  def sparks(r, since \\ 0), do: r.sparks |> Enum.take_while(fn [i | _] -> i > since end) |> Enum.reverse()

  defp cache_entry(res), do: Map.take(res, [:status, :value, :fitness, :i])
  defp slim(res), do: %{i: res.i, by: Atom.to_string(res.by), key: res.key, status: Atom.to_string(res.status), value: res.value}

  defp update_stats(r, name, f), do: %{r | stats: Map.update(r.stats, name, f.(%{pulls: 0, evals: 0, improved: 0, archived: 0}), f)}

  defp archive(r, res, name) do
    arch = [res | r.archive] |> Enum.uniq_by(& &1.key) |> Enum.sort_by(& &1.rank, :desc)
    {pinned, rest} = Enum.split_with(arch, &MapSet.member?(r.pins, &1.key))
    arch = (pinned ++ Enum.take(rest, @archive)) |> Enum.sort_by(& &1.rank, :desc)
    r = %{r | archive: arch}
    if Enum.any?(arch, &(&1.key == res.key)), do: update_stats(r, name, &%{&1 | archived: &1.archived + 1}), else: r
  end

  defp elites(%{spec: %{describe: false}} = r, _res), do: r

  defp elites(r, res) do
    case Alembic.call(r.spec.prog, "describe", [Space.realize(r.spec.space, res.x)]) do
      {:ok, d} when is_list(d) ->
        cell = Enum.map(d, fn v when is_number(v) -> round(v); v -> v end)
        case Map.get(r.elites, cell) do
          old when old != nil and old.fitness >= res.fitness -> r
          _ -> if map_size(r.elites) >= 4096 and not Map.has_key?(r.elites, cell), do: r, else: %{r | elites: Map.put(r.elites, cell, Map.put(res, :cell, cell))}
        end
      _ -> r
    end
  end

  # ================================================================ certificate

  @doc """
  The run's certificate: the claim (optimal / target reached / counter-
  example / best found), the best candidate and how it was verified, the
  random control at the same budget, the holdout of the finalists, the
  journal's root and how to replay it.
  """
  def certificate(r, opts \\ []) do
    spec = r.spec
    best = r.best
    control =
      if Keyword.get(r.opts, :control, true) and not spec.measured and r.evals > 0 and Keyword.get(opts, :control, true) and
           not (spec.sense == :claim and r.reason == "exhausted"),
         do: control(r), else: nil

    %{
      kind: "athanor",
      problem: Spec.describe(spec),
      spec_hash: spec.hash,
      sense: Atom.to_string(spec.sense),
      space: spec.space.text,
      space_size: size_out(spec.space.size),
      status: Atom.to_string(r.status),
      verdict: verdict(r),
      reason: r.reason,
      best: best && candidate_out(r, best),
      top: r.archive |> Enum.filter(&(&1.status == :ok)) |> Enum.take(10) |> Enum.map(&candidate_out(r, &1)),
      evaluations: r.evals,
      budget: spec.budget,
      invalid: r.invalid,
      errors: r.errors,
      error_samples: r.error_samples,
      duplicates: r.dups,
      seed: spec.seed,
      strategies: r.stats |> Enum.filter(fn {_, s} -> s.evals > 0 or s.pulls > 0 end) |> Map.new(fn {k, v} -> {Atom.to_string(k), v} end),
      found_by: best && Atom.to_string(best.by),
      trace: Enum.reverse(r.trace),
      control: control,
      holdout: holdout(r),
      closest_invalid: if(r.best == nil, do: closest_invalid(r)),
      elites: map_size(r.elites),
      journal_root: r.journal,
      outside_proposals: r.injected,
      notes: spec.notes,
      ms: System.monotonic_time(:millisecond) - r.t0,
      replay: "vapor athanor run PROBLEM.alb --seed #{spec.seed} --budget #{r.evals}" <> if(r.injected != [], do: " (with the recorded outside proposals)", else: "")
    }
  end

  defp size_out(:infinite), do: "infinite"
  defp size_out(n) when n < 1_000_000_000_000, do: n
  defp size_out(n), do: "≈ 1e#{String.length(Integer.to_string(n)) - 1}"

  defp verdict(r) do
    case {r.reason, r.best} do
      {"counterexample", b} -> "refuted: #{b.key} is a counterexample to claim(x)"
      {"exhausted", nil} -> exhaust_nil(r)
      {"exhausted", b} -> exhausted_text(r, b)
      {"found", b} -> "found: #{b.key} satisfies valid(x)"
      {"target", b} -> "target reached: #{fmt(b.value)} by #{b.key}"
      {_, nil} -> "nothing valid found in #{r.evals} evaluations"
      {_, b} when r.spec.sense == :claim -> "not refuted in #{r.evals} evaluations — this is not a proof (best margin #{fmt(b[:margin])})"
      {_, b} -> "best found: #{fmt(b.value)} by #{b.key} after #{r.evals} evaluations (not proved optimal)"
    end
  end

  defp exhaust_nil(%{spec: %{sense: :claim}} = r), do: "proved over the whole space: claim(x) holds for all #{r.evals} candidates" <> bans_note(r)
  defp exhaust_nil(r), do: "proved: no candidate satisfies valid(x) (all #{r.evals} checked)" <> bans_note(r)

  defp exhausted_text(%{spec: %{sense: :claim}} = r, _b), do: "proved over the whole space: claim(x) holds for all #{r.evals} candidates" <> bans_note(r)
  defp exhausted_text(r, b), do: "optimal: #{fmt(b.value)} by #{b.key} — every one of the #{r.evals} candidates was evaluated" <> bans_note(r)

  defp bans_note(%{bans: b}) do
    if MapSet.size(b) == 0, do: "", else: " (except #{MapSet.size(b)} banned by hand)"
  end

  defp fmt(nil), do: "—"
  defp fmt(v) when is_float(v), do: :erlang.float_to_binary(v, [{:decimals, 6}, :compact])
  defp fmt(v), do: to_string(v)

  defp candidate_out(r, e) do
    shown =
      if r.spec.show do
        case Alembic.call(r.spec.prog, "show", [Space.realize(r.spec.space, e.x)]) do
          {:ok, s} when is_binary(s) -> s
          {:ok, v} -> Alembic.show(v)
          _ -> nil
        end
      end

    %{candidate: e.key, value: e.value, by: e[:by] && Atom.to_string(e.by), at: e[:i], shown: shown, pinned: MapSet.member?(r.pins, e.key), valid: e.status == :ok, violation: e[:violation],
      data: Alembic.to_data(e.x), cell: e[:cell]}
  end

  # the same number of evaluations spent on uniform samples
  defp control(r) do
    n = min(r.evals, 20_000)
    spec = r.spec
    rng = :rand.seed_s(:exsss, {spec.seed, 424_242, 17})
    {xs, _} = Enum.map_reduce(1..n, rng, fn _, g -> Space.sample(spec.space, g) end)
    results = Enum.map(xs, fn x -> evaluate(spec, %{x: Space.canon(spec.space, x), key: nil}, r.opts) end)
    ok = Enum.filter(results, &(&1.status == :ok))
    best = Enum.max_by(ok, & &1.fitness, fn -> nil end)
    reach = if r.best, do: Enum.count(ok, &(&1.fitness >= r.best.fitness)), else: 0
    q_hi = if reach == 0, do: min(3.0 / n, 1.0), else: nil

    %{strategy: "random", evaluations: n, valid: length(ok), best: best && best.value,
      reached_best: reach,
      p_chance: if(reach > 0, do: reach / n, else: nil),
      p_chance_upper95: q_hi,
      says: control_words(r, best, reach, n, q_hi)}
  end

  defp control_words(r, nil, _reach, n, _), do: if(r.best, do: "random search found nothing valid in #{n} samples; the search did", else: "neither found anything valid")

  defp control_words(r, cb, reach, n, q) do
    cond do
      r.best == nil -> "random search did better than the portfolio"
      reach == 0 -> "random search reached #{fmt(cb.value)} at best in #{n} samples and never matched #{fmt(r.best.value)} (chance per sample ≤ #{fmt(q)}, 95 %)"
      true -> "random search matched the best in #{reach} of #{n} samples — the problem is easy at this budget"
    end
  end

  defp holdout(%{spec: %{holdout: false}}), do: nil
  defp holdout(%{best: nil}), do: nil

  defp holdout(r) do
    rows =
      r.archive
      |> Enum.filter(&(&1.status == :ok))
      |> Enum.take(12)
      |> Enum.map(fn e ->
        h = case Alembic.call(r.spec.prog, "holdout", [Space.realize(r.spec.space, e.x)]) do {:ok, v} when is_number(v) -> v; _ -> nil end
        %{candidate: e.key, objective: e.value, holdout: h}
      end)

    good = Enum.filter(rows, &is_number(&1.holdout))
    rho = if length(good) >= 3, do: spearman(Enum.map(good, & &1.objective), Enum.map(good, & &1.holdout)), else: nil
    first = List.first(rows)
    best_h = Enum.max_by(good, fn g -> if r.spec.sense == :min, do: -g.holdout, else: g.holdout end, fn -> nil end)
    rank = best_h && Enum.find_index(good, &(&1.candidate == first.candidate))

    %{finalists: rows, rank_correlation: rho, trials: r.evals,
      noise_max_z: :math.sqrt(2 * :math.log(max(r.evals, 2))),
      says: holdout_words(rho, first, best_h, rank, r.evals)}
  end

  defp holdout_words(rho, first, best_h, _rank, n) do
    base = "the search tried #{n} candidates; the best of #{n} pure-noise scores sits ≈ #{fmt(:math.sqrt(2 * :math.log(max(n, 2))))} standard deviations above their mean — the selection bias a holdout exists to expose"
    cond do
      rho == nil -> base
      rho < 0.2 -> "the finalists' order does not survive the holdout (ρ = #{fmt(rho)}): the search is fitting noise. " <> base
      best_h != nil and best_h.candidate != first.candidate -> "on the holdout another finalist wins (#{best_h.candidate}); ρ = #{fmt(rho)}. " <> base
      true -> "the best candidate stays best on the holdout (ρ = #{fmt(rho)}). " <> base
    end
  end

  @doc "Spearman's rank correlation."
  def spearman(a, b) do
    ra = ranks(a)
    rb = ranks(b)
    n = length(a)
    ma = Enum.sum(ra) / n
    mb = Enum.sum(rb) / n
    cov = Enum.zip_with(ra, rb, &((&1 - ma) * (&2 - mb))) |> Enum.sum()
    sa = :math.sqrt(Enum.sum(Enum.map(ra, &((&1 - ma) ** 2))))
    sb = :math.sqrt(Enum.sum(Enum.map(rb, &((&1 - mb) ** 2))))
    if sa == 0 or sb == 0, do: 0.0, else: cov / (sa * sb)
  end

  defp ranks(xs) do
    sorted = xs |> Enum.with_index() |> Enum.sort_by(&elem(&1, 0))
    groups = Enum.chunk_by(sorted, &elem(&1, 0))
    {pairs, _} =
      Enum.flat_map_reduce(groups, 1, fn g, start ->
        avg = start + (length(g) - 1) / 2
        {Enum.map(g, fn {_, i} -> {i, avg} end), start + length(g)}
      end)
    pairs |> Enum.sort() |> Enum.map(&elem(&1, 1))
  end

  defp closest_invalid(r) do
    case Enum.find(r.archive, &(&1.status == :invalid)) do
      nil -> nil
      e -> candidate_out(r, e)
    end
  end

  defp inspect_short(v), do: v |> inspect() |> String.slice(0, 60)

  @doc "A compact snapshot for live views (console, TUI, sessions)."
  def snapshot(r) do
    %{status: Atom.to_string(r.status), reason: r.reason, evaluations: r.evals, budget: r.spec.budget, round: r.round, sense: Atom.to_string(r.spec.sense),
      space: r.spec.space.text, measured: r.spec.measured,
      best: r.best && candidate_out(r, r.best), top: r.archive |> Enum.take(12) |> Enum.map(&candidate_out(r, &1)), closest_invalid: closest_invalid(r),
      trace: Enum.reverse(r.trace), recent: r.recent, strategies: Map.new(r.stats, fn {k, v} -> {Atom.to_string(k), v} end),
      invalid: r.invalid, errors: r.errors, error_samples: r.error_samples, pending: Enum.map(r.pending, & &1.key), problem: Spec.describe(r.spec),
      queue: length(r.queue), journal_root: r.journal, elites: map_size(r.elites), exhaustive: r.exhaustive}
  end
end
