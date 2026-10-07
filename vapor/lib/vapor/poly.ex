defmodule Vapor.Poly do
  @moduledoc """
  Polynomial approximations with a *certified* error bound — what
  homomorphic encryption (CKKS evaluates only `+` and `×`, so every
  non-linearity must become a polynomial) and fixed-point circuits need, and
  what they usually take on faith.

      %{cheb: cs, bound: e, depth: d} = Vapor.Poly.approx(:sigmoid, {-8, 8}, 15)
      # |sigmoid(x) − p(x)| ≤ e for every real x in [−8, 8]; d = multiplicative depth

  The coefficients come from Chebyshev interpolation (in binary64; they are
  whatever they are). The bound is then *proved* in exact rational
  arithmetic, not sampled:

    * on a uniform grid of dyadic points, `p` is evaluated exactly (Clenshaw
      over rationals) and `f` is enclosed using `Vapor.CR`'s correctly
      rounded exponential (true value within one ulp);
    * between two grid points the error `e = f − p` departs from the chord
      through its end values by at most `h²/8 · sup|e''|` (the interpolation
      remainder), and `sup|e''| ≤ sup|f''| + sup|p''|`: `sup|f''|` is a known
      constant of the function (√3/18 < 1/10 for the sigmoid, 4/(3√3) < 4/5
      for tanh, `e^b` for exp on [a, b]), and `sup|p''| ≤ (2/(b−a))² ·
      Σ |c_k|·k²(k²−1)/3` (Markov's inequality for the second derivative of
      each Chebyshev polynomial).

  So `bound = max over the grid + h²/8 · sup|e''|` holds on the whole
  interval. The
  multiplicative depth reported is `⌈log₂(d + 1)⌉`, the least number of
  ciphertext-by-ciphertext levels a degree-`d` evaluation needs — real CKKS
  evaluations (Paterson–Stockmeyer, plaintext constants) often spend one
  more; it is the quantity CKKS parameters are chosen for.
  """
  import Bitwise

  @fns [:sigmoid, :tanh, :exp]

  @doc "Approximate `f` (#{inspect(@fns)}) on `{a, b}` (dyadic endpoints) by a degree-`d` polynomial. Options: `grid:` points − 1, a power of two (default 4096)."
  def approx(f, {a, b}, d, opts \\ []) when f in @fns and b > a and d >= 1 do
    n = d + 1
    nodes = for j <- 0..(n - 1), do: :math.cos(:math.pi() * (j + 0.5) / n)
    xs = Enum.map(nodes, &((&1 * (b - a) + a + b) / 2))
    fx = Enum.map(xs, &float_f(f, &1))

    cheb =
      for k <- 0..d do
        s = Enum.zip(nodes, fx) |> Enum.reduce(0.0, fn {t, y}, acc -> acc + y * :math.cos(k * :math.acos(t)) end)
        if k == 0, do: s / n, else: 2 * s / n
      end

    %{cheb: cheb, interval: {a, b}, degree: d, depth: ceil(:math.log2(d + 1)), bound: certify(f, cheb, {a, b}, Keyword.get(opts, :grid, 4096))}
  end

  @doc "Evaluate the Chebyshev series at a float (for use; the bound is about exact evaluation)."
  def eval(%{cheb: cs, interval: {a, b}}, x) do
    t = (2 * x - a - b) / (b - a)
    {b1, b2} = cs |> Enum.drop(1) |> Enum.reverse() |> Enum.reduce({0.0, 0.0}, fn c, {b1, b2} -> {2 * t * b1 - b2 + c, b1} end)
    t * b1 - b2 + hd(cs)
  end

  @doc "The function itself, in binary64 (reference)."
  def float_f(:sigmoid, x), do: 1 / (1 + :math.exp(-x))
  def float_f(:tanh, x), do: :math.tanh(x)
  def float_f(:exp, x), do: :math.exp(x)

  # ------------------------------------------------------- certification --

  # the bound, as a float rounded up from an exact rational
  defp certify(f, cheb, {a, b}, grid) do
    unless grid == 1 <<< trunc(:math.log2(grid)), do: raise(ArgumentError, "grid must be a power of two")
    cs = Enum.map(cheb, &q/1)
    {qa, qb} = {q(a * 1.0), q(b * 1.0)}
    width = sub(qb, qa)
    h = divq(width, {grid, 1})

    max_grid =
      Enum.reduce(0..grid, {0, 1}, fn i, acc ->
        x = add(qa, mul(h, {i, 1}))
        t = divq(sub(sub(mul({2, 1}, x), qa), qb), width)
        p = clenshaw(cs, t)
        {lo, hi} = enclose(f, x)
        qmax(acc, qmax(qabs(sub(lo, p)), qabs(sub(hi, p))))
      end)

    # between grid points: |e| ≤ max(|e(xᵢ)|, |e(xᵢ₊₁)|) + h²/8 · sup|e''|, with
    # sup|e''| ≤ sup|f''| + sup|p''| and |T_k''| ≤ k²(k² − 1)/3 on [−1, 1]
    s2 = mul(mul(divq({2, 1}, width), divq({2, 1}, width)),
             Enum.reduce(Enum.with_index(cs), {0, 1}, fn {c, k}, acc -> add(acc, mul(qabs(c), {k * k * (k * k - 1), 3})) end))
    e2 = add(f2(f, qb), s2)
    up(add(max_grid, mul(e2, divq(mul(h, h), {8, 1}))))
  end

  # upper bounds of |f''| on the interval: σ'' ≤ √3/18 < 1/10, tanh'' ≤ 4/(3√3) < 4/5, exp'' = exp ≤ e^b
  defp f2(:sigmoid, _b), do: {1, 10}
  defp f2(:tanh, _b), do: {4, 5}
  defp f2(:exp, qb), do: elem(enclose(:exp, qb), 1)

  # an enclosure [lo, hi] of f(x) for a dyadic x
  defp enclose(:exp, x), do: exp_enclosure(x)

  defp enclose(:sigmoid, x) do
    {lo, hi} = exp_enclosure(neg(x))
    # σ is decreasing in e^(−x)
    {divq({1, 1}, add({1, 1}, hi)), divq({1, 1}, add({1, 1}, lo))}
  end

  defp enclose(:tanh, x) do
    {lo, hi} = enclose(:sigmoid, mul({2, 1}, x))
    {sub(mul({2, 1}, lo), {1, 1}), sub(mul({2, 1}, hi), {1, 1})}
  end

  # e^x within one ulp of the correctly rounded binary64 value: |rel| ≤ 2⁻⁵²
  defp exp_enclosure({n, d} = x) do
    xf = to_float(x)
    if q(xf) != {n, d}, do: raise(ArgumentError, "grid point not exactly representable")
    e = q(Vapor.CR.exp_f64(xf))
    {mul(e, {(1 <<< 52) - 1, 1 <<< 52}), mul(e, {(1 <<< 52) + 1, 1 <<< 52})}
  end

  defp clenshaw(cs, t) do
    [c0 | rest] = cs
    {b1, b2} = rest |> Enum.reverse() |> Enum.reduce({{0, 1}, {0, 1}}, fn c, {b1, b2} -> {add(sub(mul(mul({2, 1}, t), b1), b2), c), b1} end)
    add(sub(mul(t, b1), b2), c0)
  end

  # ------------------------------------------------- exact rationals {n, d} --

  # a float as the exact rational it denotes
  defp q(x) when is_float(x) do
    <<s::1, e::11, m::52>> = <<x::float-64>>
    {mant, exp} = if e == 0, do: {m, -1074}, else: {m ||| 1 <<< 52, e - 1075}
    v = if exp >= 0, do: {mant <<< exp, 1}, else: norm({mant, 1 <<< -exp})
    if s == 1, do: neg(v), else: v
  end

  defp q(x) when is_integer(x), do: {x, 1}

  defp norm({n, d}) when d < 0, do: norm({-n, -d})
  defp norm({n, d}), do: (g = Integer.gcd(n, d); {div(n, g), div(d, g)})
  defp add({a, b}, {c, d}), do: norm({a * d + c * b, b * d})
  defp sub(x, y), do: add(x, neg(y))
  defp neg({a, b}), do: {-a, b}
  defp mul({a, b}, {c, d}), do: norm({a * c, b * d})
  defp divq({a, b}, {c, d}), do: norm({a * d, b * c})
  defp qabs({a, b}), do: {abs(a), b}
  defp qmax({a, b} = x, {c, d} = y), do: if(a * d >= c * b, do: x, else: y)

  # nearest-ish float of a rational of any size (64 significant bits, then one rounding)
  defp to_float({0, _}), do: 0.0

  defp to_float({n, d}) do
    k = 70 - (bits(abs(n)) - bits(d))
    qn = if k >= 0, do: div(n <<< k, d), else: div(n, d <<< -k)
    qn * :math.pow(2.0, -k)
  end

  defp bits(x), do: length(Integer.digits(x, 2))

  # a float ≥ the (positive) rational
  defp up({n, d} = r) do
    f = to_float(r)
    if q(f) |> then(fn {a, b} -> a * d >= n * b end), do: f, else: f * (1 + :math.pow(2, -52)) + 5.0e-324
  end
end
