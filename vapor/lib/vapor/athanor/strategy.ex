defmodule Vapor.Athanor.Strategy do
  @moduledoc """
  The proposers of the Athanor (docs/ATHANOR.md §3). Each strategy is a
  map with a `name` and its own state; `propose/3` returns candidates,
  `observe/3` hands it their results. None of them is trusted: everything
  they propose is evaluated by the same verifier, and the portfolio
  (`Vapor.Athanor`) gives the budget to whichever is improving.

  | strategy | idea | spaces |
  |---|---|---|
  | `exhaustive` | every candidate, in order — a proof when it completes | finite, small enough |
  | `random` | uniform samples — also the control | all |
  | `anneal` | simulated annealing with an adaptive temperature and restarts from the archive | all |
  | `evolve` | steady-state evolution (tournament, crossover, mutation); MAP-Elites when `describe(x)` exists | all |
  | `cmaes` | covariance-matrix adaptation (Hansen) | reals, ints |
  | `bayes` | a Gaussian process with expected improvement — for objectives that are expensive or measured by a person | reals, ints |
  | `mind` | a language model shown the problem and the best candidates | all (needs a model) |
  | `human` | what a person proposes | all |
  """
  alias Vapor.Athanor.{Gauss, Space}
  alias Vapor.Alembic

  @doc "The strategies that apply to a space and spec."
  def applicable(spec, opts) do
    s = spec.space
    vector = Space.vector?(s)
    small_budget = spec.budget <= 400 or spec.measured
    # random sampling is an arm too (rugged spaces at small budgets reward it) — and, separately, the control
    base = [:random, :anneal, :evolve] ++ if(vector and s.n <= 60, do: [:cmaes], else: []) ++ if(vector and s.n <= 12 and small_budget, do: [:bayes], else: [])
    base = if spec.measured, do: Enum.filter(base, &(&1 in [:bayes, :evolve, :random, :anneal])), else: base
    base = if Keyword.get(opts, :mind), do: base ++ [:mind], else: base
    case Keyword.get(opts, :only) do
      nil -> base
      only -> Enum.filter(base ++ [:random], &(&1 in only)) |> Enum.uniq()
    end
  end

  @doc "A fresh strategy."
  def new(name, spec, seed, opts \\ []) do
    rng = :rand.seed_s(:exsss, {seed, :erlang.phash2(name), 7919})
    base = %{name: name, rng: rng, pulls: 0, reward: 0.0}

    case name do
      :exhaustive -> Map.put(base, :cont, resumable(Space.enumerate(spec.space)))
      :anneal -> Map.merge(base, %{cur: nil, cur_f: nil, temp: nil, deltas: [], stall: 0})
      :evolve -> base
      :cmaes -> Map.put(base, :cma, nil)
      :bayes -> base
      :mind -> Map.merge(base, %{mind: Keyword.get(opts, :mind), failures: 0})
      _ -> base
    end
  end

  # ================================================================ propose

  @doc "Propose up to `n` candidates given the run context `ctx` (%{spec, archive, best, cache})."
  def propose(%{name: :random} = st, ctx, n), do: sample_n(st, ctx.spec.space, n)

  def propose(%{name: :exhaustive, cont: cont} = st, _ctx, n) do
    case take(cont, n) do
      {xs, :done} -> {xs, %{st | cont: :done}}
      {xs, c} -> {xs, %{st | cont: c}}
    end
  end

  def propose(%{name: :anneal} = st, ctx, n) do
    st = if st.cur == nil or st.stall > 40, do: restart(st, ctx), else: st
    if st.cur == nil do
      sample_n(st, ctx.spec.space, n)
    else
      {xs, rng} = Enum.map_reduce(1..n, st.rng, fn i, r -> neighbor(ctx, st.cur, r, i) end)
      {xs, %{st | rng: rng}}
    end
  end

  def propose(%{name: :evolve} = st, ctx, n) do
    pop = parents(ctx)
    if length(pop) < 2 do
      sample_n(st, ctx.spec.space, n)
    else
      {xs, rng} =
        Enum.map_reduce(1..n, st.rng, fn i, r ->
          {a, r} = tournament(pop, r)
          {b, r} = tournament(pop, r)
          {c, r} = :rand.uniform_s(r)
          {child, r} = if c < 0.6, do: Space.cross(ctx.spec.space, a, b, r), else: {a, r}
          {m, r} = :rand.uniform_s(r)
          if m < 0.85 or child == a, do: neighbor(ctx, child, r, i), else: {child, r}
        end)
      {xs, %{st | rng: rng}}
    end
  end

  def propose(%{name: :cmaes} = st, ctx, n) do
    s = ctx.spec.space
    cma = st.cma || cma_init(s, ctx)
    {xs, rng} = Enum.map_reduce(1..max(n, 4), st.rng, fn _, r -> cma_sample(cma, s, r) end)
    {xs, %{st | rng: rng, cma: cma}}
  end

  def propose(%{name: :bayes} = st, ctx, n) do
    s = ctx.spec.space
    pts = ctx.observed |> Enum.filter(&(&1.status == :ok)) |> Enum.take(-150)
    if length(pts) < max(2 * s.n, 6) do
      sample_n(st, s, n)
    else
      {xs, rng} = Gauss.suggest(s, pts, n, st.rng)
      {xs, %{st | rng: rng}}
    end
  end

  def propose(%{name: :mind} = st, ctx, n) do
    case Vapor.Mind.propose(st.mind, ctx, n) do
      {:ok, xs} -> {xs, st}
      {:error, _} -> {[], %{st | failures: st.failures + 1}}
    end
  end

  def propose(%{name: :human} = st, _ctx, _n), do: {[], st}

  # ================================================================ observe

  @doc "Hand a strategy the results of its proposals (`results`: %{x, fitness, status}) and the run's best fitness before them."
  def observe(%{name: :anneal} = st, results, best_before) do
    ok = Enum.filter(results, &(&1[:rank] != nil))
    case ok do
      [] -> %{st | stall: st.stall + 1}
      _ ->
        cand = Enum.max_by(ok, & &1.rank)
        {ct, cf} = cand.rank
        cur = st.cur_f
        deltas = case cur do {^ct, f0} -> Enum.take([abs(cf - f0) | st.deltas], 50); _ -> st.deltas end
        temp = case deltas do [] -> 1.0; ds -> max(median(ds) * 0.5, 1.0e-12) end
        temp = temp * :math.pow(0.97, st.pulls)
        {u, rng} = :rand.uniform_s(st.rng)
        accept =
          case cur do
            nil -> true
            {t0, _} when ct > t0 -> true
            {t0, _} when ct < t0 -> false
            {_, f0} -> cf >= f0 or u < :math.exp((cf - f0) / max(temp, 1.0e-300))
          end
        improved = best_before == nil or (cand.status == :ok and cand.fitness > best_before)
        st = %{st | rng: rng, deltas: deltas, temp: temp, stall: if(improved, do: 0, else: st.stall + 1)}
        if accept, do: %{st | cur: cand.x, cur_f: cand.rank}, else: st
    end
  end

  def observe(%{name: :cmaes, cma: cma} = st, results, _best) when cma != nil do
    ok = Enum.filter(results, &(&1.status == :ok))
    if length(ok) >= 2, do: %{st | cma: cma_update(cma, ok)}, else: st
  end

  def observe(st, _results, _best), do: st

  # ================================================================ helpers

  defp sample_n(st, space, n) do
    {xs, rng} = Enum.map_reduce(1..n, st.rng, fn _, r -> Space.sample(space, r) end)
    {xs, %{st | rng: rng}}
  end

  defp restart(st, ctx) do
    case ctx.archive do
      [] -> %{st | cur: nil, cur_f: nil, stall: 0}
      arch ->
        {i, rng} = :rand.uniform_s(min(length(arch), 8), st.rng)
        e = Enum.at(arch, i - 1)
        %{st | cur: e.x, cur_f: e.rank, stall: 0, rng: rng, deltas: []}
    end
  end

  defp neighbor(ctx, x, rng, i) do
    spec = ctx.spec
    if spec.neighbor do
      {k, rng} = :rand.uniform_s(1_000_000_000, rng)
      case Alembic.call(spec.prog, "neighbor", [x, k + i]) do
        {:ok, y} -> {y, rng}
        {:error, _} -> Space.mutate(spec.space, x, rng)
      end
    else
      Space.mutate(spec.space, x, rng)
    end
  end

  defp parents(ctx) do
    case ctx.elites do
      e when map_size(e) >= 4 -> Map.values(e)
      _ -> ctx.archive
    end
  end

  defp tournament(pop, rng) do
    n = length(pop)
    {i, rng} = :rand.uniform_s(n, rng)
    {j, rng} = :rand.uniform_s(n, rng)
    {a, b} = {Enum.at(pop, i - 1), Enum.at(pop, j - 1)}
    {if(a.rank >= b.rank, do: a.x, else: b.x), rng}
  end

  defp median(xs), do: xs |> Enum.sort() |> Enum.at(div(length(xs), 2))

  # resumable enumeration: Enumerable.reduce with :suspend
  defp resumable(enum) do
    fn n -> Enumerable.reduce(enum, {:cont, {n, []}}, &reducer/2) end
  end

  defp reducer(x, {left, acc}) when left <= 1, do: {:suspend, {0, [x | acc]}}
  defp reducer(x, {left, acc}), do: {:cont, {left - 1, [x | acc]}}

  defp take(:done, _n), do: {[], :done}

  defp take(cont, n) when is_function(cont, 1) do
    case cont.(n) do
      {:suspended, {_, acc}, k} -> {Enum.reverse(acc), fn m -> k.({:cont, {m, []}}) end}
      {:done, {_, acc}} -> {Enum.reverse(acc), :done}
      {:halted, {_, acc}} -> {Enum.reverse(acc), :done}
    end
  end

  # ================================================================ CMA-ES
  # Hansen's (μ/μ_w, λ)-CMA-ES in coordinates scaled to [0, 1]; ints are rounded.

  defp cma_init(s, ctx) do
    n = s.n
    mean =
      case ctx.archive do
        [best | _] -> Enum.map(best.x, &((&1 - s.lo) / (s.hi - s.lo)))
        [] -> List.duplicate(0.5, n)
      end
    lambda = 4 + trunc(3 * :math.log(n))
    mu = div(lambda, 2)
    w = for i <- 1..mu, do: :math.log(mu + 0.5) - :math.log(i)
    sw = Enum.sum(w)
    w = Enum.map(w, &(&1 / sw))
    mueff = 1 / Enum.sum(Enum.map(w, &(&1 * &1)))
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, :math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    %{n: n, lo: s.lo * 1.0, hi: s.hi * 1.0, mean: mean, sigma: 0.3, c: identity(n), b: identity(n), d: List.duplicate(1.0, n), pc: List.duplicate(0.0, n), ps: List.duplicate(0.0, n),
      w: w, mu: mu, lambda: lambda, mueff: mueff, cc: cc, cs: cs, c1: c1, cmu: cmu, damps: damps, chin: :math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n)), gen: 0}
  end

  defp cma_sample(cma, s, rng) do
    {z, rng} = Enum.map_reduce(1..cma.n, rng, fn _, r -> :rand.normal_s(r) end)
    dz = Enum.zip_with(cma.d, z, &(&1 * &2))
    y = matvec(cma.b, dz)
    u = Enum.zip_with(cma.mean, y, fn m, yi -> min(max(m + cma.sigma * yi, 0.0), 1.0) end)
    {from_unit(s, u), rng}
  end

  defp from_unit(%{kind: :reals, lo: lo, hi: hi}, u), do: Enum.map(u, &(lo + &1 * (hi - lo)))
  defp from_unit(%{kind: :ints, lo: lo, hi: hi}, u), do: Enum.map(u, &round(lo + &1 * (hi - lo)))

  defp to_unit(_space, x, lo, hi), do: Enum.map(x, &((&1 - lo) / (hi - lo)))

  defp cma_update(cma, ok) do
    # coordinates of the evaluated points, best first
    sorted = ok |> Enum.sort_by(& &1.fitness, :desc) |> Enum.take(cma.mu)
    xs = Enum.map(sorted, &to_unit(nil, &1.x, cma.lo, cma.hi))
    w = Enum.take(cma.w, length(xs))
    sw = Enum.sum(w)
    w = Enum.map(w, &(&1 / sw))
    old = cma.mean
    mean = Enum.reduce(Enum.zip(w, xs), List.duplicate(0.0, cma.n), fn {wi, x}, acc -> Enum.zip_with(acc, x, &(&1 + wi * &2)) end)
    ys = Enum.map(xs, fn x -> Enum.zip_with(x, old, &((&1 - &2) / cma.sigma)) end)
    yw = Enum.zip_with(mean, old, &((&1 - &2) / cma.sigma))
    # C^(-1/2) yw = B D^-1 B' yw
    binv = matvec(transpose(cma.b), yw) |> Enum.zip_with(cma.d, &(&1 / max(&2, 1.0e-300))) |> then(&matvec(cma.b, &1))
    ps = Enum.zip_with(cma.ps, binv, &((1 - cma.cs) * &1 + :math.sqrt(cma.cs * (2 - cma.cs) * cma.mueff) * &2))
    psn = :math.sqrt(Enum.sum(Enum.map(ps, &(&1 * &1))))
    hsig = if psn / :math.sqrt(1 - :math.pow(1 - cma.cs, 2 * (cma.gen + 1))) / cma.chin < 1.4 + 2 / (cma.n + 1), do: 1.0, else: 0.0
    pc = Enum.zip_with(cma.pc, yw, &((1 - cma.cc) * &1 + hsig * :math.sqrt(cma.cc * (2 - cma.cc) * cma.mueff) * &2))
    rank_mu = Enum.reduce(Enum.zip(w, ys), zeros(cma.n), fn {wi, y}, acc -> madd(acc, outer(y, y), wi) end)
    c = cma.c |> mscale(1 - cma.c1 - cma.cmu) |> madd(outer(pc, pc), cma.c1) |> madd(rank_mu, cma.cmu)
    sigma = cma.sigma * :math.exp(cma.cs / cma.damps * (psn / cma.chin - 1)) |> min(1.0) |> max(1.0e-8)
    {b, d2} = jacobi(c)
    d = Enum.map(d2, &:math.sqrt(max(&1, 1.0e-20)))
    %{cma | mean: mean, ps: ps, pc: pc, c: c, sigma: sigma, b: b, d: d, gen: cma.gen + 1}
  end

  defp identity(n), do: for(i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: if(i == j, do: 1.0, else: 0.0)))
  defp zeros(n), do: for(_ <- 1..n, do: List.duplicate(0.0, n))
  defp matvec(m, v), do: Enum.map(m, fn row -> Enum.zip_with(row, v, &(&1 * &2)) |> Enum.sum() end)
  defp transpose(m), do: m |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
  defp outer(a, b), do: Enum.map(a, fn x -> Enum.map(b, &(x * &1)) end)
  defp mscale(m, s), do: Enum.map(m, fn r -> Enum.map(r, &(&1 * s)) end)
  defp madd(a, b, s), do: Enum.zip_with(a, b, fn ra, rb -> Enum.zip_with(ra, rb, &(&1 + s * &2)) end)

  @doc "Eigen-decomposition of a symmetric matrix by cyclic Jacobi: {eigenvectors as columns, eigenvalues}."
  def jacobi(a) do
    n = length(a)
    t = a |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    v = identity(n) |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    {t, v} = jacobi_sweeps(t, v, n, 0)
    {v |> Tuple.to_list() |> Enum.map(&Tuple.to_list/1), for(i <- 0..(n - 1), do: elem(elem(t, i), i))}
  end

  defp jacobi_sweeps(t, v, n, sweep) do
    off = for(i <- 0..(n - 1), j <- 0..(n - 1), i != j, do: elem(elem(t, i), j) ** 2) |> Enum.sum()
    if off < 1.0e-22 or sweep > 50 or n == 1 do
      {t, v}
    else
      {t, v} =
        for p <- 0..(n - 2), q <- (p + 1)..(n - 1), reduce: {t, v} do
          {t, v} ->
            apq = elem(elem(t, p), q)
            if abs(apq) < 1.0e-300 do
              {t, v}
            else
              app = elem(elem(t, p), p)
              aqq = elem(elem(t, q), q)
              theta = (aqq - app) / (2 * apq)
              tt = (if theta >= 0, do: 1, else: -1) / (abs(theta) + :math.sqrt(theta * theta + 1))
              c = 1 / :math.sqrt(tt * tt + 1)
              s = tt * c
              rot(t, v, n, p, q, c, s)
            end
        end
      jacobi_sweeps(t, v, n, sweep + 1)
    end
  end

  defp rot(t, v, n, p, q, c, s) do
    g = fn m, i, j -> elem(elem(m, i), j) end
    set = fn m, i, j, x -> put_elem(m, i, put_elem(elem(m, i), j, x)) end
    # columns p, q of A, then rows p, q
    t =
      Enum.reduce(0..(n - 1), t, fn k, m ->
        akp = g.(m, k, p)
        akq = g.(m, k, q)
        m |> set.(k, p, c * akp - s * akq) |> set.(k, q, s * akp + c * akq)
      end)
    t =
      Enum.reduce(0..(n - 1), t, fn k, m ->
        apk = g.(m, p, k)
        aqk = g.(m, q, k)
        m |> set.(p, k, c * apk - s * aqk) |> set.(q, k, s * apk + c * aqk)
      end)
    v =
      Enum.reduce(0..(n - 1), v, fn k, m ->
        vkp = g.(m, k, p)
        vkq = g.(m, k, q)
        m |> set.(k, p, c * vkp - s * vkq) |> set.(k, q, s * vkp + c * vkq)
      end)
    {t, v}
  end
end
