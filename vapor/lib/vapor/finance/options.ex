defmodule Vapor.Finance.Options do
  @moduledoc """
  Option pricing with the numbers that let a desk judge each price
  (docs/FINANCAS.md §4).

  | function | model | certificate |
  |---|---|---|
  | `bsm/7`, `greeks/7` | Black–Scholes–Merton with dividend yield | put–call parity residual; each Greek against a central difference |
  | `black76/6`, `bachelier/6` | futures (lognormal / normal) | parity |
  | `implied_vol/8` | inverse of BSM by Brent | the no-arbitrage bounds checked **before** solving (outside them: refused, with the bound); the repricing error; the vega and the volatility error one price tick causes |
  | `binomial/10` | CRR and Leisen–Reimer, European and American | convergence to BSM for the European |
  | `heston/7` | Heston (1993) by Lewis' single integral and the "little trap" characteristic function | parity; the ξ → 0 limit equals BSM |
  | `svi_fit/2`, `svi_arbitrage/2` | raw SVI smile | Durrleman's g(k) ≥ 0 on a fine grid (butterfly), total variance non-decreasing in T (calendar); the risk-neutral density |
  | `static_arbitrage/3` | model-free, call quotes across strikes | monotonicity, slope bounds, convexity — each violation returned as the portfolio that exploits it, its payoff checked at every kink |

  Every price is pinned against QuantLib in `financas_test.exs`.
  """
  alias Vapor.Finance.Num

  # ----------------------------------------------------------------- BSM

  @doc "Black–Scholes–Merton price of a European option (`:call` or `:put`)."
  def bsm(type, s, k, t, r, q, sigma) do
    cond do
      t <= 0 or sigma <= 0 ->
        fwd = s * :math.exp((r - q) * max(t, 0)); df = :math.exp(-r * max(t, 0))
        df * max(if(type == :call, do: fwd - k, else: k - fwd), 0.0)
      true ->
        {d1, d2} = d12(s, k, t, r, q, sigma)
        case type do
          :call -> s * :math.exp(-q * t) * Num.ncdf(d1) - k * :math.exp(-r * t) * Num.ncdf(d2)
          :put -> k * :math.exp(-r * t) * Num.ncdf(-d2) - s * :math.exp(-q * t) * Num.ncdf(-d1)
        end
    end
  end

  defp d12(s, k, t, r, q, sigma) do
    v = sigma * :math.sqrt(t)
    d1 = (:math.log(s / k) + (r - q + sigma * sigma / 2) * t) / v
    {d1, d1 - v}
  end

  @doc "Analytic Greeks (theta per year, vega and rho per unit of σ and r)."
  def greeks(type, s, k, t, r, q, sigma) do
    {d1, d2} = d12(s, k, t, r, q, sigma)
    eq = :math.exp(-q * t); er = :math.exp(-r * t); sq = :math.sqrt(t); n1 = Num.npdf(d1)
    gamma = eq * n1 / (s * sigma * sq)
    vega = s * eq * n1 * sq
    case type do
      :call ->
        %{delta: eq * Num.ncdf(d1), gamma: gamma, vega: vega,
          theta: -s * eq * n1 * sigma / (2 * sq) - r * k * er * Num.ncdf(d2) + q * s * eq * Num.ncdf(d1),
          rho: k * t * er * Num.ncdf(d2)}
      :put ->
        %{delta: -eq * Num.ncdf(-d1), gamma: gamma, vega: vega,
          theta: -s * eq * n1 * sigma / (2 * sq) + r * k * er * Num.ncdf(-d2) - q * s * eq * Num.ncdf(-d1),
          rho: -k * t * er * Num.ncdf(-d2)}
    end
  end

  @doc "The Greeks by central differences of the pricer (the independent side of the certificate)."
  def greeks_fd(type, s, k, t, r, q, sigma) do
    p = &bsm(type, &1, k, &2, &3, q, &4)
    hs = s * 1.0e-4; hv = 1.0e-5; ht = 1.0e-5; hr = 1.0e-6
    %{delta: (p.(s + hs, t, r, sigma) - p.(s - hs, t, r, sigma)) / (2 * hs),
      gamma: (p.(s + hs, t, r, sigma) - 2 * p.(s, t, r, sigma) + p.(s - hs, t, r, sigma)) / (hs * hs),
      vega: (p.(s, t, r, sigma + hv) - p.(s, t, r, sigma - hv)) / (2 * hv),
      theta: -(p.(s, t + ht, r, sigma) - p.(s, t - ht, r, sigma)) / (2 * ht),
      rho: (p.(s, t, r + hr, sigma) - p.(s, t, r - hr, sigma)) / (2 * hr)}
  end

  @doc "Black (1976) on a futures price F, discounted at r."
  def black76(type, f, k, t, r, sigma), do: :math.exp(-r * t) * bsm(type, f, k, t, 0.0, 0.0, sigma)

  @doc "Bachelier (normal) model on a forward F with normal volatility σₙ (rates that can be negative)."
  def bachelier(type, f, k, t, r, sn) do
    v = sn * :math.sqrt(t); df = :math.exp(-r * t)
    if v <= 0, do: df * max(if(type == :call, do: f - k, else: k - f), 0.0), else: (
      d = (f - k) / v
      case type do
        :call -> df * ((f - k) * Num.ncdf(d) + v * Num.npdf(d))
        :put -> df * ((k - f) * Num.ncdf(-d) + v * Num.npdf(d))
      end)
  end

  @doc "No-arbitrage bounds of a European option price: {lower, upper}."
  def bounds(:call, s, k, t, r, q), do: {max(s * :math.exp(-q * t) - k * :math.exp(-r * t), 0.0), s * :math.exp(-q * t)}
  def bounds(:put, s, k, t, r, q), do: {max(k * :math.exp(-r * t) - s * :math.exp(-q * t), 0.0), k * :math.exp(-r * t)}

  @doc """
  Implied volatility. The price is first checked against the no-arbitrage
  bounds (outside them no volatility exists, and the answer says which
  bound); then Brent on σ ∈ [10⁻⁶, 10]. `{:ok, %{sigma, certificate}}`.
  """
  def implied_vol(type, price, s, k, t, r, q, tick \\ 0.01) do
    {lo, hi} = bounds(type, s, k, t, r, q)
    cond do
      t <= 0 -> {:error, "expired: no time value, no volatility"}
      price <= lo -> {:error, "price #{fmt(price)} is at or below the lower no-arbitrage bound #{fmt(lo)} (intrinsic value discounted): no volatility reproduces it"}
      price >= hi -> {:error, "price #{fmt(price)} is at or above the upper bound #{fmt(hi)}: no volatility reproduces it"}
      true ->
        f = fn sg -> bsm(type, s, k, t, r, q, sg) - price end
        case Num.brent(f, 1.0e-6, 10.0, 1.0e-15) do
          {:ok, sg} ->
            vega = greeks(type, s, k, t, r, q, sg).vega
            err = abs(f.(sg))
            {:ok, %{sigma: sg, certificate: %{repriced: bsm(type, s, k, t, r, q, sg), abs_error: err, vega: vega,
                                               vol_per_tick: if(vega > 0, do: tick / vega, else: :infinity), bounds: [lo, hi],
                                               well_conditioned: vega > 0 and tick / vega < 0.01}}}
          {:error, _} -> {:error, "no volatility in [10⁻⁶, 10] reproduces the price"}
        end
    end
  end

  defp fmt(x), do: :erlang.float_to_binary(x * 1.0, decimals: 6)

  # ------------------------------------------------------------ lattices

  @doc """
  Binomial lattice: `method` `:crr` (Cox–Ross–Rubinstein) or `:lr`
  (Leisen–Reimer, Peizer–Pratt inversion 2; N made odd), `style`
  `:european` or `:american`.
  """
  def binomial(type, style, s, k, t, r, q, sigma, n, method \\ :lr) do
    n = if method == :lr and rem(n, 2) == 0, do: n + 1, else: n
    dt = t / n
    growth = :math.exp((r - q) * dt)
    {u, d, p} =
      case method do
        :crr ->
          u = :math.exp(sigma * :math.sqrt(dt)); d = 1 / u
          {u, d, (growth - d) / (u - d)}
        :lr ->
          {d1, d2} = d12(s, k, t, r, q, sigma)
          h = fn z -> 0.5 + (if z >= 0, do: 1, else: -1) * 0.5 * :math.sqrt(1 - :math.exp(-:math.pow(z / (n + 1 / 3 + 0.1 / (n + 1)), 2) * (n + 1 / 6))) end
          pp = h.(d1); p = h.(d2)
          u = growth * pp / p
          d = (growth - p * u) / (1 - p)
          {u, d, p}
      end
    disc = :math.exp(-r * dt)
    payoff = fn x -> if type == :call, do: max(x - k, 0.0), else: max(k - x, 0.0) end
    leaves = for j <- 0..n, do: payoff.(s * :math.pow(u, j) * :math.pow(d, n - j))
    vals =
      Enum.reduce((n - 1)..0//-1, leaves, fn i, vs ->
        cont = Enum.zip(vs, tl(vs)) |> Enum.map(fn {dn, up} -> disc * (p * up + (1 - p) * dn) end)
        if style == :american,
          do: cont |> Enum.with_index() |> Enum.map(fn {c, j} -> max(c, payoff.(s * :math.pow(u, j) * :math.pow(d, i - j))) end),
          else: cont
      end)
    hd(vals)
  end

  # --------------------------------------------------------------- Heston

  # complex helpers {re, im}
  defp c(a), do: {a * 1.0, 0.0}
  defp ca({a, b}, {x, y}), do: {a + x, b + y}
  defp cs({a, b}, {x, y}), do: {a - x, b - y}
  defp cm({a, b}, {x, y}), do: {a * x - b * y, a * y + b * x}
  defp cd({a, b}, {x, y}), do: (q = x * x + y * y; {(a * x + b * y) / q, (b * x - a * y) / q})
  defp ce({a, b}), do: (m = :math.exp(a); {m * :math.cos(b), m * :math.sin(b)})
  defp cl({a, b}), do: {:math.log(:math.sqrt(a * a + b * b)), :math.atan2(b, a)}
  defp csq({a, b}) do
    m = :math.sqrt(a * a + b * b)
    re = :math.sqrt((m + a) / 2); im = :math.sqrt(max((m - a) / 2, 0.0))
    {re, if(b < 0, do: -im, else: im)}
  end

  @doc false
  # characteristic function of X_T = ln(S_T/S_0) − (r − q)T under Heston, at complex u (Albrecher et al., "little trap")
  def heston_cf(u, t, %{v0: v0, kappa: kp, theta: th, xi: xi, rho: rho}) do
    iu = cm({0.0, 1.0}, u)
    a = cs(c(kp), cm(c(rho * xi), iu))                       # κ − ρξ iu
    dd = csq(ca(cm(a, a), cm(c(xi * xi), ca(iu, cm(u, u)))))  # √((ρξiu − κ)² + ξ²(iu + u²))
    g = cd(cs(a, dd), ca(a, dd))
    edt = ce(cm(c(-t), dd))
    one = c(1.0)
    cc = cm(c(kp * th / (xi * xi)), cs(cm(cs(a, dd), c(t)), cm(c(2.0), cl(cd(cs(one, cm(g, edt)), cs(one, g))))))
    dc = cm(cd(cs(a, dd), c(xi * xi)), cd(cs(one, edt), cs(one, cm(g, edt))))
    ce(ca(cc, cm(dc, c(v0))))
  end

  @doc """
  Heston price by Lewis' formula:
  C = S e^(−qT) − √(SK) e^(−(r+q)T/2)/π ∫₀^∞ Re[e^(iuk) φ(u − i/2)] / (u² + ¼) du,
  k = ln(S/K) + (r − q)T. Puts by parity.
  """
  def heston(type, s, k, t, r, q, p) do
    kk = :math.log(s / k) + (r - q) * t
    integrand = fn u ->
      phi = heston_cf({u, -0.5}, t, p)
      {re, _} = cm(ce({0.0, u * kk}), phi)
      re / (u * u + 0.25)
    end
    # the integrand decays like e^(−c·u); [0, 200] in 400 panels of 16 points is far past 10⁻¹⁴ for sane parameters
    integral = Num.integrate(integrand, 0.0, 200.0, 400, 16)
    call = s * :math.exp(-q * t) - :math.sqrt(s * k) * :math.exp(-(r + q) * t / 2) / :math.pi() * integral
    case type do
      :call -> call
      :put -> call - s * :math.exp(-q * t) + k * :math.exp(-r * t)
    end
  end

  # ------------------------------------------------------------------ SVI

  @doc "Raw SVI total implied variance w(k) = a + b(ρ(k − m) + √((k − m)² + σ²))."
  def svi_w(%{a: a, b: b, rho: rho, m: m, sigma: sg}, k), do: a + b * (rho * (k - m) + :math.sqrt((k - m) * (k - m) + sg * sg))

  @doc "Durrleman's g(k): the risk-neutral density is non-negative iff g ≥ 0."
  def svi_g(%{b: b, rho: rho, m: m, sigma: sg} = p, k) do
    w = svi_w(p, k)
    x = k - m; rt = :math.sqrt(x * x + sg * sg)
    w1 = b * (rho + x / rt)
    w2 = b * sg * sg / (rt * rt * rt)
    :math.pow(1 - k * w1 / (2 * w), 2) - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2
  end

  @doc "Risk-neutral density of k = ln(K/F) implied by the smile: g(k)/√(2πw) · exp(−d₋²/2)."
  def svi_density(p, k) do
    w = svi_w(p, k)
    dm = -k / :math.sqrt(w) - :math.sqrt(w) / 2
    svi_g(p, k) / :math.sqrt(2 * :math.pi() * w) * :math.exp(-dm * dm / 2)
  end

  @doc """
  Fit raw SVI to quotes [{k, total variance}] (k = ln(K/F)) by Nelder–Mead
  over a reparametrisation that keeps b ≥ 0, |ρ| < 1, σ > 0, from several
  starts; the minimum variance a + bσ√(1 − ρ²) ≥ 0 is enforced.
  """
  def svi_fit(points, _opts \\ []) do
    unpack = fn [a, lb, tr, m, ls] -> %{a: a, b: :math.exp(lb), rho: :math.tanh(tr), m: m, sigma: :math.exp(ls)} end
    obj = fn x ->
      p = unpack.(x)
      minvar = p.a + p.b * p.sigma * :math.sqrt(1 - p.rho * p.rho)
      pen = if minvar < 0, do: 1.0e3 * minvar * minvar, else: 0.0
      Enum.reduce(points, pen, fn {k, w}, acc -> d = svi_w(p, k) - w; acc + d * d end)
    end
    ws = Enum.map(points, &elem(&1, 1)); ks = Enum.map(points, &elem(&1, 0))
    wmin = Enum.min(ws)
    starts = for m0 <- [Enum.min(ks) / 2, 0.0, Enum.max(ks) / 2], r0 <- [-0.5, 0.0, 0.3], do: [wmin * 0.8, :math.log(0.1), :math.atanh(r0), m0, :math.log(0.2)]
    {best, f, _} = starts |> Enum.map(&Num.nelder_mead(obj, &1, step: 0.3, tol: 1.0e-16, maxit: 6000)) |> Enum.min_by(&elem(&1, 1))
    {best, f, _} = Num.nelder_mead(obj, best, step: 0.05, tol: 1.0e-18, maxit: 6000) |> then(fn {b2, f2, i} -> if f2 < f, do: {b2, f2, i}, else: {best, f, i} end)
    p = unpack.(best)
    %{params: p, rmse: :math.sqrt(f / length(points))}
  end

  @doc """
  Butterfly arbitrage of an SVI slice: min g(k) over a grid on [k_lo, k_hi]
  (default ±1.5); `free: true` when g ≥ 0 everywhere on the grid.
  """
  def svi_arbitrage(p, opts \\ []) do
    {lo, hi} = Keyword.get(opts, :range, {-1.5, 1.5})
    n = Keyword.get(opts, :points, 601)
    grid = for i <- 0..(n - 1), do: lo + (hi - lo) * i / (n - 1)
    gs = for k <- grid, do: {k, svi_g(p, k)}
    {kmin, gmin} = Enum.min_by(gs, &elem(&1, 1))
    neg = Enum.filter(gs, fn {_, g} -> g < 0 end)
    %{free: gmin >= 0, g_min: gmin, at_k: kmin, negative_interval: if(neg == [], do: nil, else: [elem(hd(neg), 0), elem(List.last(neg), 0)]),
      min_variance: p.a + p.b * p.sigma * :math.sqrt(1 - p.rho * p.rho)}
  end

  @doc "Calendar arbitrage between two SVI slices (T₁ < T₂): w₂(k) ≥ w₁(k) on the grid."
  def svi_calendar(p1, p2, opts \\ []) do
    {lo, hi} = Keyword.get(opts, :range, {-1.5, 1.5})
    grid = for i <- 0..300, do: lo + (hi - lo) * i / 300
    {k, d} = grid |> Enum.map(&{&1, svi_w(p2, &1) - svi_w(p1, &1)}) |> Enum.min_by(&elem(&1, 1))
    %{free: d >= 0, min_gap: d, at_k: k}
  end

  # ------------------------------------------------- model-free arbitrage

  @doc """
  Static arbitrage in call prices quoted across strikes (one expiry),
  model-free: with D = e^(−rT), calls must be non-increasing in K, with
  slopes in [−D, 0], and convex. Every violation comes back as the
  portfolio that exploits it — a vertical spread or a butterfly — with
  its cost (negative = money received) and its payoff checked at every
  kink (the payoff of such a portfolio is piecewise linear with kinks at
  the strikes, so checking the kinks and the slope beyond the last one
  checks every terminal price).
  """
  def static_arbitrage(quotes, r, t) do
    q = Enum.sort_by(quotes, &elem(&1, 0))
    dfac = :math.exp(-r * t)
    ks = Enum.map(q, &elem(&1, 0))
    pay = fn port, x -> Enum.reduce(port, 0.0, fn {kk, w}, a -> a + w * max(x - kk, 0.0) end) end
    check = fn port ->
      pts = [0.0 | ks] ++ [List.last(ks) * 2 + 1]
      slope_inf = Enum.reduce(port, 0.0, fn {_, w}, a -> a + w end)
      Enum.all?(pts, &(pay.(port, &1) >= -1.0e-12)) and slope_inf >= -1.0e-12
    end
    cost = fn port -> Enum.reduce(port, 0.0, fn {kk, w}, a -> a + w * (Enum.find(q, &(elem(&1, 0) == kk)) |> elem(1)) end) end
    pairs = Enum.zip(q, tl(q))
    mono =
      for {{k1, c1}, {k2, c2}} <- pairs, c2 > c1 + 1.0e-12 do
        port = [{k1, 1.0}, {k2, -1.0}]   # buy the cheaper low strike, sell the dearer high strike
        %{kind: "monotonicity", strikes: [k1, k2], portfolio: port, cost: cost.(port), payoff_nonnegative: check.(port)}
      end
    slope =
      for {{k1, c1}, {k2, c2}} <- pairs, (c1 - c2) / (k2 - k1) > dfac + 1.0e-12 do
        # a call spread worth more than the discounted strike gap: sell it, hold D·(K₂ − K₁) in bonds
        %{kind: "spread above the discounted strike gap", strikes: [k1, k2], portfolio: [{k1, -1.0}, {k2, 1.0}], bond: dfac * (k2 - k1),
          cost: -(c1 - c2) + dfac * (k2 - k1), payoff_nonnegative: true}
      end
    conv =
      for [{k1, c1}, {k2, c2}, {k3, c3}] <- Enum.chunk_every(q, 3, 1, :discard),
          (l = (k3 - k2) / (k3 - k1); c2 > l * c1 + (1 - l) * c3 + 1.0e-12) do
        l = (k3 - k2) / (k3 - k1)
        port = [{k1, l}, {k2, -1.0}, {k3, 1 - l}]   # long the butterfly that should cost ≥ 0
        %{kind: "convexity (butterfly)", strikes: [k1, k2, k3], portfolio: port, cost: cost.(port), payoff_nonnegative: check.(port)}
      end
    all = mono ++ slope ++ conv
    %{arbitrage_free: all == [], violations: all}
  end

  # ------------------------------------------------------------------ text

  @doc """
  The desk's text interface. The first word picks the task:

      price call S=100 K=105 T=0.5 r=5% q=0 vol=25%
      iv put S=100 K=95 T=0.25 r=5% price=1.80
      american put S=36 K=40 T=1 r=6% vol=20% steps=1001
      heston call S=100 K=100 T=1 r=2% v0=0.04 kappa=1.5 theta=0.04 xi=0.5 rho=-0.7
      smile T=0.5 F=100
      80 0.32 · 90 0.27 · 100 0.24 · … (strike and implied vol per line)
  """
  def run(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#") |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    case lines do
      [] -> {:error, "empty"}
      [first | rest] ->
        kv = Regex.scan(~r/([A-Za-z0-9_]+)\s*=\s*([-\d.,eE]+%?)/u, first) |> Map.new(fn [_, k, v] -> {String.downcase(k), parse_v(v)} end)
        words = first |> String.downcase() |> String.split(~r/\s+/)
        type = if "put" in words, do: :put, else: :call
        missing = fn keys -> case Enum.reject(keys, &Map.has_key?(kv, &1)) do [] -> :ok; m -> {:error, "missing #{Enum.join(m, ", ")}"} end end
        case hd(words) do
          w when w in ["price", "preço", "preco", "european", "europeia"] -> with(:ok <- missing.(~w(s k t vol)), do: task_price(type, kv))
          w when w in ["iv", "implied", "implícita", "implicita"] -> with(:ok <- missing.(~w(s k t price)), do: task_iv(type, kv))
          w when w in ["american", "americana"] -> with(:ok <- missing.(~w(s k t vol)), do: task_american(type, kv))
          "heston" -> with(:ok <- missing.(~w(s k t v0 kappa theta xi rho)), do: task_heston(type, kv))
          w when w in ["smile", "sorriso", "svi"] -> with(:ok <- missing.(~w(t f)), do: task_smile(kv, rest))
          w -> {:error, "first word: price, iv, american, heston or smile (got #{inspect(w)})"}
        end
    end
  end

  defp args(kv), do: {Map.get(kv, "s"), Map.get(kv, "k"), Map.get(kv, "t"), Map.get(kv, "r", 0.0), Map.get(kv, "q", 0.0)}

  defp task_price(type, kv) do
    {s, k, t, r, q} = args(kv); v = kv["vol"]
    px = bsm(type, s, k, t, r, q, v)
    other = bsm(if(type == :call, do: :put, else: :call), s, k, t, r, q, v)
    parity = (if type == :call, do: px - other, else: other - px) - (s * :math.exp(-q * t) - k * :math.exp(-r * t))
    ga = greeks(type, s, k, t, r, q, v); gf = greeks_fd(type, s, k, t, r, q, v)
    worst = Enum.max(for kk <- [:delta, :gamma, :vega, :theta, :rho], do: abs(ga[kk] - gf[kk]) / max(abs(ga[kk]), 1.0e-8))
    curve = Enum.map(0..80, fn i -> x = s * (0.5 + i / 80); %{s: x, price: bsm(type, x, k, t, r, q, v), intrinsic: max(if(type == :call, do: x - k, else: k - x), 0.0)} end)
    {:ok, %{task: "price", type: type, price: px, greeks: ga, greeks_fd: gf, payoff_curve: curve,
            lattice_lr_1001: binomial(type, :european, s, k, t, r, q, v, 1001, :lr),
            certificate: %{parity_residual: parity, greeks_max_rel_diff: worst, ok: abs(parity) < 1.0e-10 * max(s, k) and worst < 1.0e-3}}}
  end

  defp task_iv(type, kv) do
    {s, k, t, r, q} = args(kv)
    with {:ok, res} <- implied_vol(type, kv["price"], s, k, t, r, q, Map.get(kv, "tick", 0.01)) do
      top = max(1.0, 2.5 * res.sigma)
      curve = Enum.map(1..80, fn i -> sg = top * i / 80; %{sigma: sg, price: bsm(type, s, k, t, r, q, sg)} end)
      {:ok, Map.merge(%{task: "implied volatility", type: type, quote: kv["price"], curve: curve}, res)}
    end
  end

  defp task_american(type, kv) do
    {s, k, t, r, q} = args(kv); v = kv["vol"]
    n = Map.get(kv, "steps", 801) |> trunc() |> max(11) |> min(5001)
    am = binomial(type, :american, s, k, t, r, q, v, n, :lr)
    eu = binomial(type, :european, s, k, t, r, q, v, n, :lr)
    b = bsm(type, s, k, t, r, q, v)
    conv = [51, 101, 201, 401, 801] |> Enum.filter(&(&1 <= n)) |> Enum.map(fn m -> %{steps: m, american: binomial(type, :american, s, k, t, r, q, v, m, :lr), crr: binomial(type, :american, s, k, t, r, q, v, m, :crr)} end)
    {:ok, %{task: "american", type: type, price: am, european: eu, early_exercise_premium: am - eu, bsm: b, convergence: conv, steps: n,
            certificate: %{european_lattice_vs_bsm: abs(eu - b), premium_nonnegative: am >= eu - 1.0e-12}}}
  end

  defp task_heston(type, kv) do
    {s, k, t, r, q} = args(kv)
    p = %{v0: kv["v0"], kappa: kv["kappa"], theta: kv["theta"], xi: kv["xi"], rho: kv["rho"]}
    if p.xi <= 0 or p.kappa <= 0 or p.theta <= 0 or p.v0 < 0 or abs(p.rho) >= 1 do
      {:error, "Heston needs ξ > 0, κ > 0, θ > 0, v₀ ≥ 0, |ρ| < 1"}
    else
      px = heston(type, s, k, t, r, q, p)
      smile = Enum.flat_map(0..24, fn i -> heston_smile_point(s, k * (0.6 + 0.8 * i / 24), t, r, q, p) end)
      parity = heston(:call, s, k, t, r, q, p) - heston(:put, s, k, t, r, q, p) - (s * :math.exp(-q * t) - k * :math.exp(-r * t))
      {:ok, %{task: "heston", type: type, price: px, smile: smile, feller: 2 * p.kappa * p.theta >= p.xi * p.xi,
              certificate: %{parity_residual: parity, bsm_at_sqrt_v0: bsm(type, s, k, t, r, q, :math.sqrt(p.v0))}}}
    end
  end

  defp heston_smile_point(s, kk, t, r, q, p) do
    c = heston(:call, s, kk, t, r, q, p)
    case implied_vol(:call, c, s, kk, t, r, q) do
      {:ok, iv} -> [%{k: kk, price: c, iv: iv.sigma}]
      _ -> []
    end
  end

  defp task_smile(kv, rest) do
    {t, f} = {kv["t"], kv["f"]}
    pts = rest |> Enum.map(&smile_point(&1, f, t)) |> Enum.reject(&is_nil/1)
    if length(pts) < 5 do
      {:error, "a smile needs at least five quotes: one line per strike — K vol"}
    else
      fit = svi_fit(Enum.map(pts, fn {k, w, _, _} -> {k, w} end))
      arb = svi_arbitrage(fit.params)
      ks = Enum.map(pts, &elem(&1, 0))
      {kmin, kmax} = {Enum.min(ks), Enum.max(ks)}
      curve = Enum.map(0..120, fn i -> k = kmin - 0.3 + (kmax - kmin + 0.6) * i / 120
        %{k: k, strike: f * :math.exp(k), iv: :math.sqrt(max(svi_w(fit.params, k), 0.0) / t), g: svi_g(fit.params, k), density: svi_density(fit.params, k)} end)
      quotes = Enum.map(pts, fn {k, _, kk, v} -> %{k: k, strike: kk, iv: v, fitted_iv: :math.sqrt(max(svi_w(fit.params, k), 0.0) / t)} end)
      {:ok, %{task: "smile", t: t, forward: f, params: fit.params, rmse_total_variance: fit.rmse, quotes: quotes, curve: curve, certificate: arb}}
    end
  end

  defp smile_point(l, f, t) do
    with [a, b | _] <- String.split(String.replace(l, ~r/[;]\s*/, " "), ~r/\s+/),
         {kk, _} <- Float.parse(a), {iv, pct} <- Float.parse(b), true <- kk > 0 and iv > 0 do
      v = if pct == "%" or iv > 3, do: iv / 100, else: iv
      {:math.log(kk / f), v * v * t, kk, v}
    else
      _ -> nil
    end
  end

  defp parse_v(v) do
    v = String.replace(v, ",", ".")
    {pct, v} = if String.ends_with?(v, "%"), do: {true, String.trim_trailing(v, "%")}, else: {false, v}
    {x, _} = Float.parse(if String.contains?(v, ".") or String.contains?(String.downcase(v), "e"), do: v, else: v <> ".0")
    if pct, do: x / 100, else: x
  end
end
