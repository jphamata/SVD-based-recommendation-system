defmodule Vapor.Finance.Micro do
  @moduledoc """
  Market microstructure models, each with the measurement that says
  whether it fits (docs/FINANCAS.md §12).

  * **Hawkes** (self-exciting arrivals, exponential kernel): simulation by
    Ogata's thinning, maximum likelihood by the O(n) recursion, and the
    **time-rescaling test** — under the fitted model the compensator
    increments are Exp(1), judged by Kolmogorov–Smirnov. The control: a
    Poisson process fitted to the same arrivals fails the test.
  * **Avellaneda–Stoikov** market making: reservation price
    r = s − qγσ²(T − t), spread γσ²(T − t) + (2/γ)ln(1 + γ/k), simulated
    as in the paper against the symmetric strategy with the same spread;
    the inventory strategy must have a much smaller P&L and inventory
    dispersion.
  * **Almgren–Chriss** optimal execution: the closed-form trajectory
    x_j = X sinh(κ(T − t_j))/sinh(κT), certified by solving the same
    mean–variance problem numerically (a tridiagonal system) and
    comparing; the efficient frontier.
  * Estimators on a tape: Roll's spread, Kyle's λ, the realized-variance
    signature plot, the microprice.
  """
  alias Vapor.Finance.Num

  # ================================================================ Hawkes

  @doc "Simulate a Hawkes process on [0, T] by Ogata's thinning: event times."
  def hawkes_simulate(mu, alpha, beta, t_end, seed \\ 1) do
    loop = fn loop, t, excite, acc, k ->
      lam_bar = mu + excite
      w = -:math.log(Num.u01(seed, k)) / lam_bar
      t2 = t + w
      if t2 > t_end or length(acc) > 200_000 do
        Enum.reverse(acc)
      else
        excite2 = excite * :math.exp(-beta * w)
        if Num.u01(seed, k + 1) * lam_bar <= mu + excite2,
          do: loop.(loop, t2, excite2 + alpha, [t2 | acc], k + 2),
          else: loop.(loop, t2, excite2, acc, k + 2)
      end
    end
    loop.(loop, 0.0, 0.0, [], 0)
  end

  @doc "Log-likelihood of event times on [0, T] under (μ, α, β)."
  def hawkes_loglik(times, t_end, mu, alpha, beta) do
    {ll, _, _} =
      Enum.reduce(times, {0.0, 0.0, nil}, fn t, {ll, a, prev} ->
        a = if prev, do: :math.exp(-beta * (t - prev)) * (1 + a), else: 0.0
        {ll + :math.log(mu + alpha * a), a, t}
      end)
    comp = mu * t_end + alpha / beta * Enum.reduce(times, 0.0, fn t, s -> s + (1 - :math.exp(-beta * (t_end - t))) end)
    ll - comp
  end

  @doc """
  Maximum-likelihood fit (Nelder–Mead on log μ, log β and the logit of the
  branching ratio n = α/β < 1), with the time-rescaling KS test, and the
  same test for the best Poisson model.
  """
  def hawkes_fit(times, t_end) do
    n = length(times)
    obj = fn [lm, lb, ln] ->
      mu = :math.exp(lm); beta = :math.exp(lb); br = 1 / (1 + :math.exp(-ln))
      -hawkes_loglik(times, t_end, mu, br * beta, beta)
    end
    start = [:math.log(max(n / t_end * 0.5, 1.0e-6)), :math.log(1.0), 0.0]
    {[lm, lb, ln], nll, _} = Num.nelder_mead(obj, start, step: 0.5, tol: 1.0e-12, maxit: 4000)
    mu = :math.exp(lm); beta = :math.exp(lb); br = 1 / (1 + :math.exp(-ln)); alpha = br * beta
    taus = rescaled(times, mu, alpha, beta)
    ks = Num.ks(taus, fn x -> 1 - :math.exp(-x) end)
    mu_p = n / t_end
    gaps = Enum.zip([0.0 | times], times) |> Enum.map(fn {a, b} -> mu_p * (b - a) end)
    ks_p = Num.ks(gaps, fn x -> 1 - :math.exp(-x) end)
    %{mu: mu, alpha: alpha, beta: beta, branching_ratio: br, loglik: -nll, events: n,
      poisson_loglik: n * :math.log(mu_p) - mu_p * t_end,
      time_rescaling: %{ks: ks.d, p_value: ks.p}, poisson_time_rescaling: %{ks: ks_p.d, p_value: ks_p.p},
      verdict: cond do
        ks.p > 0.05 and ks_p.p < 0.05 -> "self-exciting: the Hawkes model passes the time-rescaling test, Poisson fails it"
        ks.p > 0.05 -> "both models pass: no evidence of self-excitation"
        true -> "the Hawkes model fails the time-rescaling test: another kernel is needed"
      end}
  end

  defp rescaled(times, mu, alpha, beta) do
    {taus, _} =
      Enum.map_reduce(times, {0.0, 0.0, 0.0}, fn t, {prev, a, lam_prev_t} ->
        _ = lam_prev_t
        dt = t - prev
        # Λ(t) − Λ(prev) = μ·dt + (α/β)·a·(1 − e^(−β dt)), a = Σ e^(−β(prev − t_j)) over t_j ≤ prev
        inc = mu * dt + alpha / beta * a * (1 - :math.exp(-beta * dt))
        {inc, {t, a * :math.exp(-beta * dt) + 1, 0.0}}
      end)
    taus
  end

  # ===================================================== Avellaneda–Stoikov

  @doc """
  The simulation of Avellaneda & Stoikov (2008), §4: s₀ = 100, T = 1,
  σ = 2, dt = 0.005, k = 1.5, A = 140, q₀ = 0, `runs` paths; the
  inventory strategy against the symmetric one with the same spread.
  """
  def avellaneda_stoikov(opts \\ []) do
    g = Keyword.get(opts, :gamma, 0.1); sig = Keyword.get(opts, :sigma, 2.0); k = Keyword.get(opts, :k, 1.5); a = Keyword.get(opts, :a, 140.0)
    tt = 1.0; dt = Keyword.get(opts, :dt, 0.005); runs = Keyword.get(opts, :runs, 1000); seed = Keyword.get(opts, :seed, 5)
    steps = round(tt / dt)
    us = Num.uniforms(seed, runs * steps * 3) |> List.to_tuple()
    sim = fn mode ->
      for r <- 0..(runs - 1) do
        {s, q, x, spreads} =
          Enum.reduce(0..(steps - 1), {100.0, 0, 0.0, 0.0}, fn i, {s, q, x, sp} ->
            t = i * dt
            spread = g * sig * sig * (tt - t) + 2 / g * :math.log(1 + g / k)
            r_price = if mode == :inventory, do: s - q * g * sig * sig * (tt - t), else: s
            bid = r_price - spread / 2; ask = r_price + spread / 2
            db = s - bid; da = ask - s
            base = (r * steps + i) * 3
            {q, x} = if elem(us, base) < a * :math.exp(-k * db) * dt, do: {q + 1, x - bid}, else: {q, x}
            {q, x} = if elem(us, base + 1) < a * :math.exp(-k * da) * dt, do: {q - 1, x + ask}, else: {q, x}
            s = s + if(elem(us, base + 2) < 0.5, do: sig * :math.sqrt(dt), else: -sig * :math.sqrt(dt))
            {s, q, x, sp + spread}
          end)
        {x + q * s, q, spreads / steps}
      end
    end
    summary = fn res ->
      pnl = Enum.map(res, &elem(&1, 0)); qs = Enum.map(res, &(elem(&1, 1) * 1.0))
      %{mean_pnl: Num.mean(pnl), std_pnl: Num.std(pnl), mean_q: Num.mean(qs), std_q: Num.std(qs), mean_spread: Num.mean(Enum.map(res, &elem(&1, 2)))}
    end
    inv = summary.(sim.(:inventory)); sym = summary.(sim.(:symmetric))
    %{gamma: g, runs: runs, inventory: inv, symmetric: sym,
      paper_gamma_0_1: %{inventory: %{mean_pnl: 62.94, std_pnl: 5.89, std_q: 2.80}, symmetric: %{mean_pnl: 67.21, std_pnl: 13.43, std_q: 8.66}},
      certificate: %{pnl_dispersion_ratio: inv.std_pnl / sym.std_pnl, inventory_dispersion_ratio: inv.std_q / sym.std_q}}
  end

  # ======================================================== Almgren–Chriss

  @doc """
  Almgren–Chriss (2000), discrete time: sell X over N periods of length τ
  with temporary impact η, permanent γ, fixed cost ε, volatility σ (per
  unit time) and risk aversion λ. Closed form and the independent numeric
  solution of the same quadratic problem.
  """
  def almgren_chriss(o) do
    x0 = o[:x] * 1.0; n = o[:n]; tt = o[:t] * 1.0; sig = o[:sigma] * 1.0; eta = o[:eta] * 1.0; gam = Map.get(o, :gamma, 0.0) * 1.0
    eps = Map.get(o, :epsilon, 0.0) * 1.0; lam = o[:lambda] * 1.0
    tau = tt / n
    eta_t = eta - gam * tau / 2
    kt2 = lam * sig * sig / eta_t
    kappa = if kt2 == 0.0, do: 0.0, else: :math.acosh(1 + kt2 * tau * tau / 2) / tau
    traj = for j <- 0..n, do: (if kappa == 0.0, do: x0 * (1 - j / n), else: x0 * :math.sinh(kappa * (tt - j * tau)) / :math.sinh(kappa * tt))
    # numeric: minimise (η̃/τ)Σ(x_{j−1} − x_j)² + λσ²τ Σ x_j², x₀ = X, x_N = 0 → tridiagonal system for x₁…x_{N−1}
    m = n - 1
    num_traj =
      if m <= 0 do
        [x0, 0.0]
      else
        c = eta_t / tau; d = lam * sig * sig * tau
        diag = List.duplicate(2 * c + d, m)
        off = List.duplicate(-c, m)
        rhs = [c * x0 | List.duplicate(0.0, m - 1)]
        [x0 | Vapor.Dense.tridiag(off, diag, off, rhs)] ++ [0.0]
      end
    trades = Enum.zip(traj, tl(traj)) |> Enum.map(fn {a, b} -> a - b end)
    e = 0.5 * gam * x0 * x0 + eps * Enum.sum(Enum.map(trades, &abs/1)) + eta_t / tau * Enum.sum(Enum.map(trades, &(&1 * &1)))
    v = sig * sig * tau * Enum.sum(Enum.map(tl(traj), &(&1 * &1)))
    %{kappa: kappa, half_life: if(kappa > 0, do: 1 / kappa, else: :infinity), trajectory: traj, numeric: num_traj, trades: trades,
      expected_cost: e, variance: v, objective: e + lam * v,
      certificate: %{max_trajectory_gap: Enum.zip_with(traj, num_traj, &abs(&1 - &2)) |> Enum.max(), relative: (Enum.zip_with(traj, num_traj, &abs(&1 - &2)) |> Enum.max()) / x0}}
  end

  @doc "The efficient frontier (expected cost, standard deviation) over a range of λ."
  def frontier(o, lambdas) do
    for l <- lambdas, do: (r = almgren_chriss(Map.put(o, :lambda, l)); %{lambda: l, expected_cost: r.expected_cost, std: :math.sqrt(r.variance)})
  end

  # ===================================================== tape estimators

  @doc "Roll (1984): the effective spread 2√(−cov(Δpₜ, Δpₜ₋₁)) from trade prices (nil when the covariance is positive)."
  def roll_spread(prices) do
    d = Enum.zip(prices, tl(prices)) |> Enum.map(fn {a, b} -> b - a end)
    xs = Enum.drop(d, -1); ys = tl(d)
    mx = Num.mean(xs); my = Num.mean(ys)
    c = Enum.zip(xs, ys) |> Enum.reduce(0.0, fn {x, y}, a -> a + (x - mx) * (y - my) end) |> Kernel./(max(length(xs) - 1, 1))
    if c < 0, do: 2 * :math.sqrt(-c), else: nil
  end

  @doc "Kyle's λ: the slope of Δmid on signed volume (OLS), with its t-statistic."
  def kyle_lambda(dmid, signed_volume) do
    x = Enum.map(signed_volume, &[1.0, &1 * 1.0])
    case Num.ols(x, dmid) do
      {:ok, %{coef: [_, l], sigma2: s2}} ->
        sxx = Enum.reduce(signed_volume, 0.0, fn v, a -> a + v * v end) - :math.pow(Enum.sum(signed_volume), 2) / length(signed_volume)
        %{lambda: l, t: l / :math.sqrt(s2 / max(sxx, 1.0e-300))}
      _ -> %{lambda: nil, t: nil}
    end
  end

  @doc "Realized variance per unit time at sampling steps 1…kmax: rises at fine steps when quotes bounce."
  def signature(prices, kmax \\ 20) do
    for k <- 1..kmax do
      sub = prices |> Enum.take_every(k)
      rets = Enum.zip(sub, tl(sub)) |> Enum.map(fn {a, b} -> :math.log(b / a) end)
      %{step: k, rv_per_step: Enum.reduce(rets, 0.0, &(&1 * &1 + &2)) / max(length(prices) - 1, 1)}
    end
  end

  @doc "Stoikov's microprice (weighted mid): I·ask + (1 − I)·bid, I = Q_bid/(Q_bid + Q_ask)."
  def microprice(bid, ask, qb, qa), do: (i = qb / max(qb + qa, 1.0e-300); i * ask + (1 - i) * bid)
end
