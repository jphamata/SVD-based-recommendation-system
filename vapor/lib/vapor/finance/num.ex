defmodule Vapor.Finance.Num do
  @moduledoc """
  The numerical primitives the finance modules share (docs/FINANCAS.md):
  the normal distribution (Φ by `erfc`, Φ⁻¹ by Acklam's rational
  approximation polished by two Halley steps), Brent's root finder,
  Nelder–Mead, Gauss–Legendre quadrature, moments, the χ² and
  Kolmogorov distributions — each small, each tested against SciPy.
  """

  @sqrt2 :math.sqrt(2.0)
  @inv_sqrt2pi 0.3989422804014327

  def ncdf(x), do: 0.5 * :math.erfc(-x / @sqrt2)
  def npdf(x), do: @inv_sqrt2pi * :math.exp(-0.5 * x * x)

  @doc "Φ⁻¹(p) for 0 < p < 1, to full binary64 accuracy (Acklam + two Halley steps on erfc)."
  def ninv(p) when p > 0.0 and p < 1.0 do
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    horner = fn cs, x -> Enum.reduce(cs, 0.0, &(&2 * x + &1)) end
    pl = 0.02425
    x0 =
      cond do
        p < pl -> (q = :math.sqrt(-2 * :math.log(p)); horner.(c, q) / (horner.(d, q) * q + 1))
        p <= 1 - pl -> (q = p - 0.5; r = q * q; horner.(a, r) * q / (horner.(b, r) * r + 1))
        true -> (q = :math.sqrt(-2 * :math.log(1 - p)); -horner.(c, q) / (horner.(d, q) * q + 1))
      end
    Enum.reduce(1..2, x0, fn _, x ->
      e = ncdf(x) - p
      u = e * :math.sqrt(2 * :math.pi()) * :math.exp(x * x / 2)
      x - u / (1 + x * u / 2)
    end)
  end

  @doc "Brent's method on [a, b] with f(a)·f(b) ≤ 0: `{:ok, x}` or `{:error, :no_bracket}`."
  def brent(f, a, b, tol \\ 1.0e-14, maxit \\ 200) do
    fa = f.(a); fb = f.(b)
    cond do
      fa == 0.0 -> {:ok, a}
      fb == 0.0 -> {:ok, b}
      fa * fb > 0 -> {:error, :no_bracket}
      true -> brent_loop(f, a, b, a, fa, fb, fa, b - a, b - a, tol, maxit)
    end
  end

  defp brent_loop(f, a, b, c, fa, fb, fc, d, e, tol, it) do
    {a, b, c, fa, fb, fc} = if abs(fc) < abs(fb), do: {b, c, b, fb, fc, fb}, else: {a, b, c, fa, fb, fc}
    tol1 = 2 * 2.220446049250313e-16 * abs(b) + 0.5 * tol
    xm = 0.5 * (c - b)
    cond do
      abs(xm) <= tol1 or fb == 0.0 or it == 0 -> {:ok, b}
      true ->
        {d, e} =
          if abs(e) >= tol1 and abs(fa) > abs(fb) do
            s = fb / fa
            {p, q} =
              if a == c do
                {2 * xm * s, 1 - s}
              else
                q = fa / fc; r = fb / fc
                {s * (2 * xm * q * (q - r) - (b - a) * (r - 1)), (q - 1) * (r - 1) * (s - 1)}
              end
            {p, q} = if p > 0, do: {p, -q}, else: {-p, q}
            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)), do: {p / q, d}, else: {xm, d}
          else
            {xm, d}
          end
        a2 = b; fa2 = fb
        b2 = if abs(d) > tol1, do: b + d, else: b + if(xm > 0, do: tol1, else: -tol1)
        fb2 = f.(b2)
        {c2, fc2, d2, e2} = if (fb2 > 0 and fc > 0) or (fb2 < 0 and fc < 0), do: {a2, fa2, b2 - a2, b2 - a2}, else: {c, fc, d, e}
        brent_loop(f, a2, b2, c2, fa2, fb2, fc2, d2, e2, tol, it - 1)
    end
  end

  @doc "Nelder–Mead minimisation from x0 with initial step `step` (a number or a list): `{x, f(x), iterations}`."
  def nelder_mead(f, x0, opts \\ []) do
    n = length(x0)
    step = Keyword.get(opts, :step, 0.1)
    steps = if is_list(step), do: step, else: List.duplicate(step, n)
    maxit = Keyword.get(opts, :maxit, 2000 * n)
    tol = Keyword.get(opts, :tol, 1.0e-12)
    simplex = [x0 | for(i <- 0..(n - 1), do: List.update_at(x0, i, &(&1 + Enum.at(steps, i))))] |> Enum.map(&{&1, f.(&1)})
    nm(f, simplex, n, maxit, tol, 0)
  end

  defp nm(f, simplex, n, maxit, tol, it) do
    [{xb, fb} | _] = s = Enum.sort_by(simplex, &elem(&1, 1))
    {xw, fw} = List.last(s)
    {_, fsw} = Enum.at(s, n - 1)
    spread = abs(fw - fb)
    size = s |> Enum.map(fn {x, _} -> Enum.zip_with(x, xb, &abs(&1 - &2)) |> Enum.max() end) |> Enum.max()
    if it >= maxit or (spread <= tol * (abs(fb) + tol) and size < 1.0e-10) or size < 1.0e-14 do
      {xb, fb, it}
    else
      best = Enum.take(s, n)
      cen = best |> Enum.map(&elem(&1, 0)) |> Enum.zip_with(&(Enum.sum(&1) / n))
      pt = fn t -> Enum.zip_with(cen, xw, &(&1 + t * (&1 - &2))) end
      xr = pt.(1.0); fr = f.(xr)
      s2 =
        cond do
          fr < fb ->
            xe = pt.(2.0); fe = f.(xe)
            best ++ [if(fe < fr, do: {xe, fe}, else: {xr, fr})]
          fr < fsw -> best ++ [{xr, fr}]
          true ->
            {xc, fc} = if fr < fw, do: (x = pt.(0.5); {x, f.(x)}), else: (x = pt.(-0.5); {x, f.(x)})
            if fc < min(fr, fw) do
              best ++ [{xc, fc}]
            else
              [{xb, fb} | Enum.map(tl(s), fn {x, _} -> (y = Enum.zip_with(xb, x, &(&1 + 0.5 * (&2 - &1))); {y, f.(y)}) end)]
            end
        end
      nm(f, s2, n, maxit, tol, it + 1)
    end
  end

  # Gauss–Legendre nodes and weights by Newton on Pₙ (Golub–Welsch not needed at these orders)
  @doc "Nodes and weights of the n-point Gauss–Legendre rule on [−1, 1]."
  def gauss_legendre(n) do
    key = {__MODULE__, :gl, n}
    case Process.get(key) do
      nil ->
        r = for i <- 1..n do
          x0 = :math.cos(:math.pi() * (i - 0.25) / (n + 0.5))
          x = Enum.reduce_while(1..100, x0, fn _, x ->
            {p, dp} = legendre(n, x)
            dx = p / dp
            if abs(dx) < 1.0e-16, do: {:halt, x - dx}, else: {:cont, x - dx}
          end)
          {_, dp} = legendre(n, x)
          {x, 2 / ((1 - x * x) * dp * dp)}
        end
        Process.put(key, r); r
      r -> r
    end
  end

  defp legendre(n, x) do
    {p, pm1} = Enum.reduce(2..n//1, {x, 1.0}, fn k, {p, pm1} -> {((2 * k - 1) * x * p - (k - 1) * pm1) / k, p} end)
    p = if n == 0, do: 1.0, else: p
    dp = n * (x * p - pm1) / (x * x - 1)
    {p, dp}
  end

  @doc "∫ₐᵇ f by composite Gauss–Legendre: `panels` panels of `order` points."
  def integrate(f, a, b, panels \\ 64, order \\ 16) do
    gl = gauss_legendre(order)
    h = (b - a) / panels
    Enum.reduce(0..(panels - 1), 0.0, fn k, acc ->
      lo = a + k * h
      acc + Enum.reduce(gl, 0.0, fn {x, w}, s -> s + w * f.(lo + h * (x + 1) / 2) end) * h / 2
    end)
  end

  # ----------------------------------------------------------- statistics

  def mean([]), do: 0.0
  def mean(xs), do: Enum.sum(xs) / length(xs)

  def var(xs, ddof \\ 1) do
    n = length(xs)
    if n <= ddof, do: 0.0, else: (m = mean(xs); Enum.reduce(xs, 0.0, fn x, a -> a + (x - m) * (x - m) end) / (n - ddof))
  end

  def std(xs, ddof \\ 1), do: :math.sqrt(var(xs, ddof))

  @doc "Sample skewness and (non-excess) kurtosis, population moments (as in Bailey & López de Prado)."
  def moments(xs) do
    n = length(xs); m = mean(xs)
    {m2, m3, m4} = Enum.reduce(xs, {0.0, 0.0, 0.0}, fn x, {a, b, c} -> d = x - m; {a + d * d, b + d * d * d, c + d * d * d * d} end)
    {m2, m3, m4} = {m2 / n, m3 / n, m4 / n}
    if m2 == 0.0, do: {0.0, 3.0}, else: {m3 / :math.pow(m2, 1.5), m4 / (m2 * m2)}
  end

  def quantile(sorted, p) do
    # linear interpolation between order statistics (NumPy's default, "type 7")
    n = length(sorted)
    h = (n - 1) * p
    lo = trunc(:math.floor(h)); hi = min(lo + 1, n - 1)
    a = Enum.at(sorted, lo); b = Enum.at(sorted, hi)
    a + (h - lo) * (b - a)
  end

  def corr(xs, ys) do
    mx = mean(xs); my = mean(ys)
    {sxy, sxx, syy} = Enum.zip(xs, ys) |> Enum.reduce({0.0, 0.0, 0.0}, fn {x, y}, {a, b, c} -> {a + (x - mx) * (y - my), b + (x - mx) * (x - mx), c + (y - my) * (y - my)} end)
    if sxx == 0.0 or syy == 0.0, do: 0.0, else: sxy / :math.sqrt(sxx * syy)
  end

  @doc "Upper tail of χ² with `k` degrees of freedom (regularized Γ(k/2, x/2))."
  def chi2_sf(x, _k) when x <= 0, do: 1.0
  def chi2_sf(x, 1), do: :math.erfc(:math.sqrt(x / 2))
  def chi2_sf(x, 2), do: :math.exp(-x / 2)
  def chi2_sf(x, k), do: gamma_q(k / 2, x / 2)

  # regularized upper incomplete gamma Q(a, x) (Numerical Recipes: series / continued fraction)
  defp gamma_q(a, x) when x < a + 1 do
    {sum, _} = Enum.reduce_while(1..1000, {1 / a, 1 / a}, fn n, {s, del} ->
      del = del * x / (a + n)
      if abs(del) < abs(s) * 1.0e-16, do: {:halt, {s + del, del}}, else: {:cont, {s + del, del}}
    end)
    1 - sum * :math.exp(-x + a * :math.log(x) - lgamma(a))
  end

  defp gamma_q(a, x) do
    tiny = 1.0e-300
    b = x + 1 - a; c = 1 / tiny; d = 1 / b; h = d
    {h, _, _, _} = Enum.reduce_while(1..1000, {h, b, c, d}, fn i, {h, b, c, d} ->
      an = -i * (i - a); b = b + 2
      d = an * d + b; d = if abs(d) < tiny, do: tiny, else: d
      c = b + an / c; c = if abs(c) < tiny, do: tiny, else: c
      d = 1 / d; del = d * c
      if abs(del - 1) < 1.0e-16, do: {:halt, {h * del, b, c, d}}, else: {:cont, {h * del, b, c, d}}
    end)
    :math.exp(-x + a * :math.log(x) - lgamma(a)) * h
  end

  @doc "log Γ(x), x > 0 (Lanczos, g = 7)."
  def lgamma(x) when x < 0.5, do: :math.log(:math.pi() / abs(:math.sin(:math.pi() * x))) - lgamma(1 - x)
  def lgamma(x) do
    g = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    x = x - 1
    a = Enum.reduce(Enum.with_index(tl(g), 1), hd(g), fn {c, i}, acc -> acc + c / (x + i) end)
    t = x + 7.5
    0.5 * :math.log(2 * :math.pi()) + (x + 0.5) * :math.log(t) - t + :math.log(a)
  end

  @doc "Kolmogorov–Smirnov statistic of a sample against a continuous CDF, and its asymptotic p-value."
  def ks(xs, cdf) do
    s = Enum.sort(xs); n = length(s)
    d = s |> Enum.with_index(1) |> Enum.reduce(0.0, fn {x, i}, m -> f = cdf.(x); max(m, max(i / n - f, f - (i - 1) / n)) end)
    # Stephens' correction of the asymptotic Kolmogorov distribution
    l = (:math.sqrt(n) + 0.12 + 0.11 / :math.sqrt(n)) * d
    p = if l < 0.2, do: 1.0, else: min(1.0, max(0.0, 2 * Enum.reduce(1..100, 0.0, fn k, acc -> acc + :math.pow(-1, k - 1) * :math.exp(-2 * k * k * l * l) end)))
    %{d: d, p: p, n: n}
  end

  @doc "Ordinary least squares y ~ X (X rows include the intercept column if wanted): coefficients and residual variance."
  def ols(x, y) do
    case Vapor.Dense.lstsq(x, y) do
      {:ok, b} ->
        res = Enum.zip_with(Vapor.Dense.matvec(x, b), y, &(&2 - &1))
        dof = max(length(y) - length(b), 1)
        {:ok, %{coef: b, sigma2: Enum.reduce(res, 0.0, &(&1 * &1 + &2)) / dof, residuals: res}}
      e -> e
    end
  end

  @doc "Least squares by the normal equations (small p, many rows: one pass over the data, then a p×p solve)."
  def normal_lstsq(x, y) do
    p = length(hd(x))
    {xtx, xty} =
      Enum.zip(x, y) |> Enum.reduce({List.duplicate(List.duplicate(0.0, p), p), List.duplicate(0.0, p)}, fn {row, yy}, {m, v} ->
        {Enum.zip_with(m, row, fn mr, ri -> Enum.zip_with(mr, row, &(&1 + ri * &2)) end), Enum.zip_with(v, row, &(&1 + yy * &2))}
      end)
    Vapor.Dense.solve(xtx, xty)
  end

  @doc "Deterministic uniforms in (0, 1) — splitmix64 counter stream (the same numbers on every host)."
  def uniforms(seed, n), do: (for k <- 0..(n - 1)//1, do: u01(seed, k))

  import Bitwise
  def u01(seed, k) do
    x = band(seed * 0x9E3779B97F4A7C15 + (k + 1) * 0xBF58476D1CE4E5B9, 0xFFFF_FFFF_FFFF_FFFF)
    x = band(bxor(x, x >>> 30) * 0xBF58476D1CE4E5B9, 0xFFFF_FFFF_FFFF_FFFF)
    x = band(bxor(x, x >>> 27) * 0x94D049BB133111EB, 0xFFFF_FFFF_FFFF_FFFF)
    x = bxor(x, x >>> 31)
    ((x >>> 11) + 0.5) / 9_007_199_254_740_992
  end

  @doc "Deterministic standard normals: Φ⁻¹ of the uniform stream (one uniform per normal, no rejection)."
  def normals(seed, n), do: (for k <- 0..(n - 1)//1, do: ninv(u01(seed, k)))
end
