defmodule Vapor.Finance.Risk do
  @moduledoc """
  Market risk with the tests that say whether the model deserves trust
  (docs/FINANCE.md §6).

  * `var_es/3` — Value at Risk and Expected Shortfall by historical
    simulation, normal, Cornish–Fisher and EWMA (RiskMetrics λ = 0.94).
  * `backtest/3` — the regulator's questions about a VaR model, answered
    with statistics: Kupiec's proportion of failures (LR ~ χ²₁),
    Christoffersen's independence and conditional coverage (χ²₁, χ²₂), and
    the Basel traffic light (zones for 250 days at 99 %).
  * `ledoit_wolf/1` — the shrunk covariance (Ledoit & Wolf 2004, scaled
    identity target): well-conditioned when assets ≈ observations.
  * `min_variance/2` — long-only minimum variance by an active-set
    solver, with the KKT conditions as its certificate.
  * `risk_parity/2` — equal (or budgeted) risk contributions by Newton on
    Spinu's convex formulation; the certificate is the contributions.
  * `hrp/1` — hierarchical risk parity (López de Prado 2016).

  A risk model that never fails its backtest is as suspicious as one that
  always does: `finance_test.exs` and §5i measure the **size** of the
  Kupiec test (it rejects a correct model ≈ 5 % of the time) and its
  **power** (it rejects a normal VaR on fat-tailed returns).
  """
  alias Vapor.Finance.Num
  alias Vapor.Dense

  # ---------------------------------------------------------------- VaR/ES

  @doc """
  VaR and ES of a P&L (or return) series at level `alpha` (0.99), as
  positive losses. Methods: `:historical`, `:normal`, `:cornish_fisher`,
  `:ewma` (λ from `lambda`).
  """
  def var_es(pnl, alpha \\ 0.99, method \\ :historical, opts \\ []) do
    losses = Enum.map(pnl, &(-&1))
    n = length(losses)
    case method do
      :historical ->
        s = Enum.sort(losses)
        var = Num.quantile(s, alpha)
        tail = Enum.filter(s, &(&1 >= var))
        %{var: var, es: Num.mean(tail), method: method, n: n}
      :normal ->
        m = Num.mean(losses); sd = Num.std(losses); z = Num.ninv(alpha)
        %{var: m + sd * z, es: m + sd * Num.npdf(z) / (1 - alpha), method: method, n: n}
      :cornish_fisher ->
        m = Num.mean(losses); sd = Num.std(losses); z = Num.ninv(alpha)
        {sk, ku} = Num.moments(losses); ex = ku - 3
        zc = z + (z * z - 1) * sk / 6 + (z * z * z - 3 * z) * ex / 24 - (2 * z * z * z - 5 * z) * sk * sk / 36
        %{var: m + sd * zc, es: nil, method: method, n: n, skew: sk, excess_kurtosis: ex}
      :ewma ->
        lambda = Keyword.get(opts, :lambda, 0.94)
        v = Enum.reduce(pnl, Num.var(Enum.take(pnl, min(30, n))), fn x, v -> lambda * v + (1 - lambda) * x * x end)
        sd = :math.sqrt(v); z = Num.ninv(alpha)
        %{var: sd * z, es: sd * Num.npdf(z) / (1 - alpha), method: method, n: n, sigma: sd}
    end
  end

  @doc """
  Rolling one-step-ahead VaR forecasts over a series (window `w`): the
  forecast for day t uses days t−w … t−1 only.
  """
  def rolling_var(pnl, w, alpha, method, opts \\ []) do
    arr = List.to_tuple(pnl)
    n = tuple_size(arr)
    for t <- w..(n - 1)//1 do
      win = for i <- (t - w)..(t - 1), do: elem(arr, i)
      {var_es(win, alpha, method, opts).var, elem(arr, t)}
    end
  end

  @doc """
  Backtest of VaR forecasts `[{var, realised_pnl}]` at coverage level
  `alpha`: exceptions, Kupiec POF, Christoffersen independence and
  conditional coverage, Basel zone (scaled to 250 days at 99 %).
  """
  def backtest(pairs, alpha \\ 0.99) do
    hits = Enum.map(pairs, fn {var, pnl} -> if -pnl > var, do: 1, else: 0 end)
    n = length(hits); x = Enum.sum(hits); p = 1 - alpha
    pof = kupiec(n, x, p)
    {ind, cc} = christoffersen(hits, pof.lr)
    # Basel: for 250 observations at 99 % — green 0–4, yellow 5–9, red ≥ 10 (BCBS 1996); otherwise by binomial tail
    zone =
      cond do
        n == 250 and abs(alpha - 0.99) < 1.0e-9 -> cond do x <= 4 -> :green; x <= 9 -> :yellow; true -> :red end
        true ->
          cum = binom_cdf(x, n, p)
          cond do cum < 0.95 -> :green; cum < 0.9999 -> :yellow; true -> :red end
      end
    %{observations: n, exceptions: x, expected: n * p, rate: x / max(n, 1), kupiec: pof, independence: ind, conditional_coverage: cc, zone: zone,
      verdict: cond do
        pof.p_value < 0.05 and x > n * p -> "too many exceptions: the model underestimates risk"
        pof.p_value < 0.05 -> "too few exceptions: the model overestimates risk (capital wasted)"
        ind.p_value < 0.05 -> "exceptions cluster in time: the model reacts too slowly"
        true -> "coverage and independence not rejected at 5 %"
      end}
  end

  @doc "Kupiec's proportion-of-failures likelihood ratio and its χ²₁ p-value."
  def kupiec(n, x, p) do
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ph = x / max(n, 1)
    ll1 = xlogy(n - x, 1 - ph) + xlogy(x, ph)
    lr = max(-2 * (ll0 - ll1), 0.0)
    %{lr: lr, p_value: Num.chi2_sf(lr, 1)}
  end

  defp xlogy(0, _), do: 0.0
  defp xlogy(k, q) when q <= 0, do: (if k == 0, do: 0.0, else: -1.0e300)
  defp xlogy(k, q), do: k * :math.log(q)

  defp christoffersen(hits, lr_pof) do
    {n00, n01, n10, n11} =
      Enum.zip(hits, tl(hits)) |> Enum.reduce({0, 0, 0, 0}, fn
        {0, 0}, {a, b, c, d} -> {a + 1, b, c, d}
        {0, 1}, {a, b, c, d} -> {a, b + 1, c, d}
        {1, 0}, {a, b, c, d} -> {a, b, c + 1, d}
        {1, 1}, {a, b, c, d} -> {a, b, c, d + 1}
      end)
    pi0 = n01 / max(n00 + n01, 1); pi1 = n11 / max(n10 + n11, 1); pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    l1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr_ind = max(-2 * (l0 - l1), 0.0)
    {%{lr: lr_ind, p_value: Num.chi2_sf(lr_ind, 1), transitions: [n00, n01, n10, n11]},
     %{lr: lr_pof + lr_ind, p_value: Num.chi2_sf(lr_pof + lr_ind, 2)}}
  end

  defp binom_cdf(x, n, p) do
    Enum.reduce(0..x, 0.0, fn k, acc -> acc + :math.exp(Num.lgamma(n + 1) - Num.lgamma(k + 1) - Num.lgamma(n - k + 1) + xlogy(k, p) + xlogy(n - k, 1 - p)) end)
  end

  # ------------------------------------------------------------ covariance

  @doc "Sample covariance of a T×N matrix of returns (rows = days)."
  def covariance(rows) do
    t = length(rows); cols = Dense.transpose(rows)
    means = Enum.map(cols, &Num.mean/1)
    centred = Enum.map(cols, fn c -> m = Num.mean(c); Enum.map(c, &(&1 - m)) end)
    {for(a <- centred, do: for(b <- centred, do: Dense.dot(a, b) / (t - 1))), means}
  end

  @doc """
  Ledoit–Wolf shrinkage toward μI (2004): Σ* = δ·μI + (1 − δ)·S with the
  optimal δ estimated from the data (the 1/T normalisation of the paper).
  """
  def ledoit_wolf(rows) do
    t = length(rows); n = length(hd(rows))
    cols = Dense.transpose(rows)
    xs = Enum.map(cols, fn c -> m = Num.mean(c); Enum.map(c, &(&1 - m)) end) |> Dense.transpose()
    s = for i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: Enum.reduce(xs, 0.0, fn r, a -> a + Enum.at(r, i) * Enum.at(r, j) end) / t)
    mu = Enum.sum(for i <- 0..(n - 1), do: Enum.at(Enum.at(s, i), i)) / n
    d2 = Enum.sum(for i <- 0..(n - 1), j <- 0..(n - 1), do: (x = Enum.at(Enum.at(s, i), j) - if(i == j, do: mu, else: 0.0); x * x)) / n
    b2bar = Enum.reduce(xs, 0.0, fn r, acc ->
      outer = for i <- 0..(n - 1), j <- 0..(n - 1), do: (x = Enum.at(r, i) * Enum.at(r, j) - Enum.at(Enum.at(s, i), j); x * x)
      acc + Enum.sum(outer) / n
    end) / (t * t)
    b2 = min(b2bar, d2)
    delta = if d2 > 0, do: b2 / d2, else: 1.0
    shrunk = for i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: delta * if(i == j, do: mu, else: 0.0) + (1 - delta) * Enum.at(Enum.at(s, i), j))
    %{covariance: shrunk, shrinkage: delta, target_mu: mu}
  end

  # ------------------------------------------------------------ portfolios

  @doc """
  Long-only minimum variance: min wᵀΣw s.t. Σw = 1, w ≥ 0 (optionally
  w ≤ cap), by an active-set method on the equality-constrained KKT
  system. Certificate: the KKT residuals (stationarity on the free set,
  multipliers of the bound constraints of the right sign).
  """
  def min_variance(sigma, opts \\ []) do
    n = length(sigma)
    cap = Keyword.get(opts, :cap, 1.0)
    active_loop(sigma, n, cap, MapSet.new(), MapSet.new(), 0)
  end

  defp active_loop(sigma, n, cap, at_zero, at_cap, it) do
    free = Enum.reject(0..(n - 1), &(MapSet.member?(at_zero, &1) or MapSet.member?(at_cap, &1)))
    capped = MapSet.to_list(at_cap)
    budget = 1.0 - cap * length(capped)
    # stationarity on the free set: Σ_FF w_F + Σ_FC w_C = λ·1 ; Σ w_F = budget
    sff = for i <- free, do: (for j <- free, do: at(sigma, i, j))
    rhs_c = for i <- free, do: -Enum.reduce(capped, 0.0, fn j, a -> a + at(sigma, i, j) * cap end)
    m = Enum.map(sff, &(&1 ++ [-1.0])) ++ [List.duplicate(1.0, length(free)) ++ [0.0]]
    case Dense.solve(m, rhs_c ++ [budget]) do
      {:ok, sol} ->
        wf = Enum.take(sol, length(free)); lam = List.last(sol)
        w = Enum.reduce(Enum.zip(free, wf), Map.new(capped, &{&1, cap}), fn {i, x}, acc -> Map.put(acc, i, x) end)
        w = for i <- 0..(n - 1), do: Map.get(w, i, 0.0)
        neg = free |> Enum.filter(&(Enum.at(w, &1) < -1.0e-14))
        over = free |> Enum.filter(&(Enum.at(w, &1) > cap + 1.0e-14))
        grad = Dense.matvec(sigma, w)
        cond do
          it > 4 * n -> {:error, "active set did not settle"}
          neg != [] -> active_loop(sigma, n, cap, MapSet.put(at_zero, Enum.min_by(neg, &Enum.at(w, &1))), at_cap, it + 1)
          over != [] -> active_loop(sigma, n, cap, at_zero, MapSet.put(at_cap, Enum.max_by(over, &Enum.at(w, &1))), it + 1)
          true ->
            # multipliers of the bounds: μ_i = (Σw)_i − λ must be ≥ 0 at zero, ≤ 0 at the cap
            bad0 = Enum.filter(MapSet.to_list(at_zero), &(Enum.at(grad, &1) - lam < -1.0e-12))
            badc = Enum.filter(capped, &(Enum.at(grad, &1) - lam > 1.0e-12))
            cond do
              bad0 != [] -> active_loop(sigma, n, cap, MapSet.delete(at_zero, Enum.min_by(bad0, &(Enum.at(grad, &1) - lam))), at_cap, it + 1)
              badc != [] -> active_loop(sigma, n, cap, at_zero, MapSet.delete(at_cap, Enum.max_by(badc, &(Enum.at(grad, &1) - lam))), it + 1)
              true ->
                var = Dense.dot(w, grad)
                stat = free |> Enum.map(&abs(Enum.at(grad, &1) - lam)) |> Enum.max(fn -> 0.0 end)
                {:ok, %{weights: w, variance: var, volatility: :math.sqrt(max(var, 0.0)), lambda: lam,
                        certificate: %{budget: Enum.sum(w), stationarity: stat, bounds_ok: Enum.all?(w, &(&1 >= -1.0e-14 and &1 <= cap + 1.0e-14)),
                                       multipliers_ok: true, at_zero: Enum.sort(MapSet.to_list(at_zero)), at_cap: Enum.sort(capped)}}}
            end
        end
      {:error, e} -> {:error, "singular KKT system: #{inspect(e)}"}
    end
  end

  defp at(m, i, j), do: m |> Enum.at(i) |> Enum.at(j)

  @doc """
  Risk parity: weights whose risk contributions wᵢ(Σw)ᵢ/σₚ equal the
  budgets bᵢ (default equal), by Newton on Spinu's convex problem
  min ½yᵀΣy − Σ bᵢ ln yᵢ, w = y/Σy. Certificate: the contributions.
  """
  def risk_parity(sigma, budgets \\ nil) do
    n = length(sigma)
    b = budgets || List.duplicate(1.0 / n, n)
    y0 = for i <- 0..(n - 1), do: 1.0 / :math.sqrt(at(sigma, i, i))
    y = Enum.reduce_while(1..100, y0, fn _, y ->
      sy = Dense.matvec(sigma, y)
      g = Enum.zip_with([sy, b, y], fn [a, bi, yi] -> a - bi / yi end)
      h = for i <- 0..(n - 1), do: (for j <- 0..(n - 1), do: at(sigma, i, j) + if(i == j, do: Enum.at(b, i) / (Enum.at(y, i) * Enum.at(y, i)), else: 0.0))
      {:ok, dy} = Dense.solve(h, g)
      # damped so y stays positive
      step = Enum.zip(y, dy) |> Enum.reduce(1.0, fn {yi, d}, s -> if d > 0, do: min(s, 0.95 * yi / d), else: s end)
      y2 = Enum.zip_with(y, dy, &(&1 - step * &2))
      if Dense.norm_inf(g) < 1.0e-15, do: {:halt, y2}, else: {:cont, y2}
    end)
    s = Enum.sum(y); w = Enum.map(y, &(&1 / s))
    sw = Dense.matvec(sigma, w); var = Dense.dot(w, sw)
    rc = Enum.zip_with(w, sw, &(&1 * &2 / var))
    %{weights: w, volatility: :math.sqrt(var), contributions: rc, certificate: %{max_budget_error: Enum.zip_with(rc, b, &abs(&1 - &2)) |> Enum.max()}}
  end

  @doc "Hierarchical risk parity (single linkage on √(½(1 − ρ)), quasi-diagonal order, recursive bisection)."
  def hrp(sigma) do
    n = length(sigma)
    corr = for i <- 0..(n - 1), do: (for j <- 0..(n - 1), do: at(sigma, i, j) / :math.sqrt(at(sigma, i, i) * at(sigma, j, j)))
    dist = fn i, j -> :math.sqrt(max(0.5 * (1 - at(corr, i, j)), 0.0)) end
    # single linkage agglomeration; the order is the leaves of the dendrogram
    clusters = Map.new(0..(n - 1), &{&1, [&1]})
    order = link(clusters, dist)
    w = bisect([order], Map.new(0..(n - 1), &{&1, 1.0}), sigma)
    weights = for i <- 0..(n - 1), do: w[i]
    %{weights: weights, order: order}
  end

  defp link(clusters, _dist) when map_size(clusters) == 1, do: clusters |> Map.values() |> hd()

  defp link(clusters, dist) do
    keys = Map.keys(clusters)
    {a, b, _} = (for a <- keys, b <- keys, a < b, do: {a, b, (for i <- clusters[a], j <- clusters[b], do: dist.(i, j)) |> Enum.min()}) |> Enum.min_by(&elem(&1, 2))
    clusters |> Map.delete(b) |> Map.put(a, clusters[a] ++ clusters[b]) |> link(dist)
  end

  defp bisect([], w, _), do: w
  defp bisect(items, w, sigma) do
    {w, next} =
      Enum.reduce(items, {w, []}, fn c, {w, next} ->
        if length(c) < 2 do
          {w, next}
        else
          {l, r} = Enum.split(c, div(length(c), 2))
          vl = cluster_var(sigma, l); vr = cluster_var(sigma, r)
          a = 1 - vl / (vl + vr)
          w = Enum.reduce(l, w, &Map.update!(&2, &1, fn x -> x * a end))
          w = Enum.reduce(r, w, &Map.update!(&2, &1, fn x -> x * (1 - a) end))
          {w, next ++ [l, r]}
        end
      end)
    bisect(Enum.filter(next, &(length(&1) > 1)), w, sigma)
  end

  defp cluster_var(sigma, idx) do
    sub = for i <- idx, do: (for j <- idx, do: at(sigma, i, j))
    iv = Enum.map(idx, &(1 / at(sigma, &1, &1)))
    s = Enum.sum(iv); w = Enum.map(iv, &(&1 / s))
    Dense.dot(w, Dense.matvec(sub, w))
  end
end
