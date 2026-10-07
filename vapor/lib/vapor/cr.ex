defmodule Vapor.CR do
  @moduledoc """
  Correctly rounded elementary functions — host-independent by *definition*.

  Certified programs carry constants computed in the BEAM (RoPE tables,
  frequency scalings). Computing them with `:math.cos/1`, `:math.sin/1` or
  `:math.pow/2` makes the certificate a function of the host's libm: glibc,
  musl and Apple's libm differ in the last bit on some inputs, so two
  independent nodes would compile different constants, produce different
  payloads, and never reach a co-signing quorum (ARCHITECTURE §6).

  The fix is not "a better approximation" but a *mathematical* definition:
  every function here returns the binary32 (or binary64) value nearest to the
  exact real result, ties to even. Any correct implementation anywhere gives
  the same bits, so the result needs no trust in the code that produced it —
  only in the definition.

  Method (Ziv's strategy):

    1. a fast path in binary64, using only IEEE `+ − × ÷` (correctly rounded
       on every conforming machine; the BEAM never fuses or extends), with a
       rigorous, deliberately loose error bound `E`;
    2. if the interval `y ± E` lies strictly between two consecutive
       rounding boundaries of the target format, every real in it — the
       exact result included — rounds to the same value: done;
    3. otherwise the result is recomputed in fixed point on Erlang integers
       with `W` bits (π by Machin's formula, ln 2 by an atanh series, Taylor
       series with truncation counted), and the same test is applied with
       `W = 128, 256, 512, …` until it decides. Transcendental values of
       nonzero rationals are never rounding boundaries (Lindemann–Weierstrass),
       so the loop terminates; exact cases (`cos 0`, `sin 0`, `b^0`, `b^1`)
       are returned directly.

  The fast path decides almost always (measured in `cr_test`); the integer
  path makes the answer correct when it does not. Inputs are binary32 or
  binary64 values given as floats (or integers).
  """
  import Bitwise

  # ------------------------------------------------------------------ API --

  @doc "cos(x) correctly rounded to binary32 (x a float or integer; returned as a float)."
  def cos_f32(x), do: trig(:cos, x, 24)

  @doc "sin(x) correctly rounded to binary32."
  def sin_f32(x), do: trig(:sin, x, 24)

  @doc "cos(x) correctly rounded to binary64."
  def cos_f64(x), do: trig(:cos, x, 53)

  @doc "sin(x) correctly rounded to binary64."
  def sin_f64(x), do: trig(:sin, x, 53)

  @doc "b^e for b > 0, correctly rounded to binary32."
  def pow_f32(b, e), do: pow(b, e, 24)

  @doc "b^e for b > 0, correctly rounded to binary64."
  def pow_f64(b, e), do: pow(b, e, 53)

  @doc "ln(x) for x > 0, correctly rounded to binary64."
  def log_f64(x) when x > 0 do
    {m, e} = dyadic(x)
    if x == 1.0, do: 0.0, else: ziv(53, fn w -> ln_fx(m, e, w) |> with_err(w, 64) end)
  end

  @doc "e^x correctly rounded to binary64 (|x| < 700)."
  def exp_f64(x) when abs(x) < 700 do
    if x == 0, do: 1.0, else: ziv(53, fn w -> {m, e} = dyadic(x); exp_fx(fixed(m, e, w + 64), w + 64) end)
  end

  @doc "The value nearest to `x` in binary32 (ties to even), as a float."
  def to_f32(x) when is_float(x), do: Vapor.F32.to_float(Vapor.F32.from_float(x))

  @doc false
  # the fixed-point path alone (no binary64 fast path): for tests
  def trig_exact(f, x, prec) when f in [:cos, :sin] do
    x = x * 1.0
    if x == 0.0, do: (if f == :cos, do: 1.0, else: x), else: ziv(prec, fn w -> {m, e} = dyadic(x); sincos_fx(f, m, e, w) end)
  end

  # ------------------------------------------------------------- trig --

  defp trig(f, x, prec) do
    x = x * 1.0

    cond do
      x == 0.0 -> if f == :cos, do: 1.0, else: x
      true ->
        case fast_trig(f, x, prec) do
          {:ok, y} -> y
          :undecided -> ziv(prec, fn w -> {m, e} = dyadic(x); sincos_fx(f, m, e, w) end)
        end
    end
  end

  # π/2 = P1 + P2 + P3 + δ: P1, P2 with 32 significant bits (so k·P1 and
  # k·P2 are exact for |k| < 2²¹), P3 rounded to 53 bits; |δ| < 2⁻¹¹⁸
  @fast_max 1_048_576.0

  defp fast_trig(f, x, prec) when abs(x) <= @fast_max do
    {p1, p2, p3} = pio2_split()
    k = Float.round(x * 0.6366197723675814)
    # x − k·P1 is exact (both are multiples of 2⁻³¹ once |x| ≥ π/4, and the
    # difference is below 1); the next two subtractions round once each
    r = x - k * p1 - k * p2 - k * p3
    q = rem(rem(trunc(k), 4) + 4, 4)
    {c, s} = {cos_poly(r), sin_poly(r)}

    y =
      case {f, q} do
        {:cos, 0} -> c
        {:cos, 1} -> -s
        {:cos, 2} -> -c
        {:cos, 3} -> s
        {:sin, 0} -> s
        {:sin, 1} -> c
        {:sin, 2} -> -s
        {:sin, 3} -> -c
      end

    # error of y: reduction ≤ 3 ulp(r) + |k|·2⁻¹¹⁸ (absolute, propagated with
    # slope ≤ 1), polynomial and Horner ≤ 4 ulp(y); bound both generously
    err = abs(y) * 1.0e-15 + abs(r) * 1.0e-15 + 1.0e-30
    decide_float(y, err, prec)
  end

  defp fast_trig(_f, _x, _prec), do: :undecided

  # binary32 target: the decision in binary64 arithmetic alone. The nearest
  # binary32 `c` and its neighbours are exact binary64 values, and so are the
  # midpoints between them; y ± 2·err (each sum rounded once, by less than
  # err) brackets the exact interval y ± err.
  defp decide_float(y, err, 24) when abs(y) > 1.0e-30 and abs(y) < 1.0e30 do
    cb = Vapor.F32.from_float(y)
    c = Vapor.F32.to_float(cb)
    {down, up} = if c > 0, do: {cb - 1, cb + 1}, else: {cb + 1, cb - 1}
    lo_mid = (c + Vapor.F32.to_float(down)) / 2
    hi_mid = (c + Vapor.F32.to_float(up)) / 2
    if lo_mid < y - 2 * err and y + 2 * err < hi_mid, do: {:ok, c}, else: :undecided
  end

  # a binary64 estimate y ± err (a float bound) → decided by the same test,
  # carried out exactly on the dyadic values
  defp decide_float(y, err, prec) do
    {m, e} = dyadic(y)
    {em, ee} = dyadic(err)
    s = min(e, ee)
    i = m <<< (e - s)
    eu = (em <<< (ee - s)) + 1

    case decide_fixed(i, s, eu, prec) do
      {:ok, v} -> {:ok, v}
      :undecided -> :undecided
    end
  end


  # Taylor polynomials on |r| ≤ π/4 (+ a margin), Horner in binary64:
  # truncation < 10⁻²² (sin, to r²¹; cos, to r²⁰)
  defp sin_poly(r) do
    r2 = r * r
    r * Enum.reduce(19..1//-2, 1.0, fn k, acc -> 1.0 - r2 * acc / ((k + 1) * (k + 2)) end)
  end

  defp cos_poly(r) do
    r2 = r * r
    Enum.reduce(19..1//-2, 1.0, fn k, acc -> 1.0 - r2 * acc / (k * (k + 1)) end)
  end

  defp pio2_split do
    case :persistent_term.get({__MODULE__, :pio2}, nil) do
      nil ->
        w = 256
        hp = pi_fx(w) >>> 1
        # top 32 bits of π/2 (π/2 ∈ [1, 2): bit 0 of the integer part is bit w)
        p1i = hp >>> (w - 31) <<< (w - 31)
        rest = hp - p1i
        p2i = rest >>> (w - 63) <<< (w - 63)
        p3i = rest - p2i
        split = {to_float_exact(p1i, -w), to_float_exact(p2i, -w), round_float(p3i, -w, 53)}
        :persistent_term.put({__MODULE__, :pio2}, split)
        split

      s ->
        s
    end
  end

  # cos or sin of x = m·2^e in fixed point with w bits; returns {I, s, err}
  defp sincos_fx(f, m, e, w) do
    big = max(0, bit_len(abs(m)) + e)
    g = w + 32 + big
    xf = fixed(m, e, g)
    hp = pi_fx(g) >>> 1
    k = round_div(xf, hp)
    r = xf - k * hp
    one = 1 <<< g
    r2 = div(r * r, one)

    s = series(r, r2, one, 2)
    c = series(one, r2, one, 1)

    y =
      case {f, rem(rem(k, 4) + 4, 4)} do
        {:cos, 0} -> c
        {:cos, 1} -> -s
        {:cos, 2} -> -c
        {:cos, 3} -> s
        {:sin, 0} -> s
        {:sin, 1} -> c
        {:sin, 2} -> -s
        {:sin, 3} -> -c
      end

    # |k|·(error of π/2 ≤ 2 units) + Taylor truncations (≤ 1 unit per term,
    # ≤ g terms) + the tail: 2^big·4 + 4g units
    {y, -g, (4 <<< big) + 4 * g + 16}
  end

  # Σ_j (−1)^j t_j, t_0 = first, t_{j+1} = t_j·r²/(n(n+1)) for n = start, start + 2, …
  # (sin: first = r, start = 2; cos: first = 1, start = 1); each term truncated
  defp series(first, r2, one, start), do: series(first, r2, one, start, first, -1)

  defp series(t, r2, one, n, acc, sign) do
    t = div(div(t * r2, one), n * (n + 1))
    if t == 0, do: acc, else: series(t, r2, one, n + 2, acc + sign * t, -sign)
  end

  # -------------------------------------------------------------- pow --

  defp pow(b, e, prec) when b > 0 do
    b = b * 1.0
    e = e * 1.0

    cond do
      e == 0.0 -> 1.0
      e == 1.0 -> round_float_value(b, prec)
      b == 1.0 -> 1.0
      true ->
        {mb, eb} = dyadic(b)
        {me, ee} = dyadic(e)

        ziv(prec, fn w ->
          # y = e·ln b with w + 64 + |e|-bits fractional bits
          extra = max(0, bit_len(abs(me)) + ee) + 64
          g = w + extra
          {l, _, lerr} = ln_fx(mb, eb, g + 8)
          y = mul_dyadic(l, me, ee)
          # ln error (lerr units at g+8) scaled by |e|
          yerr = ((lerr * abs(me)) >>> max(0, -ee)) + 2
          {i, s, err} = exp_fx(y >>> 8, g)
          {i, s, err + (div(abs(i) * (yerr + 1), 1 <<< g) + 1) * 4}
        end)
    end
  end

  defp mul_dyadic(l, me, ee) when ee >= 0, do: l * me <<< ee
  defp mul_dyadic(l, me, ee), do: div(l * me, 1 <<< -ee)

  # ---------------------------------------------------- fixed-point core --

  # ln x for x = m·2^e > 0, as {I, −w, err}: value I·2^−w
  defp ln_fx(m, e, w) do
    g = w + 32
    b = bit_len(m)
    k = b - 1 + e
    half = 1 <<< (b - 1)
    # f = m / 2^(b−1) ∈ [1, 2); use f/2 when f > 1.5 so t = (f−1)/(f+1) ∈ [−1/7, 1/5]
    {num, den, k} =
      if 2 * m > 3 * half, do: {m - 2 * half, m + 2 * half, k + 1}, else: {m - half, m + half, k}

    one = 1 <<< g
    t = div(num * one, den)
    t2 = div(t * t, one)
    at = atanh_series(t, t2, one, 1, 0)
    y = k * ln2_fx(g) + 2 * at
    {y >>> 32, -w, abs(k) + 8}
  end

  defp atanh_series(0, _t2, _one, _n, acc), do: acc
  defp atanh_series(t, t2, one, n, acc), do: atanh_series(div(t * t2, one), t2, one, n + 2, acc + div(t, n))

  # e^y for y in fixed point with w fractional bits: {I, s, err}, value I·2^s
  defp exp_fx(y, w) do
    g = w + 48
    yg = y <<< 48
    l2 = ln2_fx(g)
    n = round_div(yg, l2)
    r = yg - n * l2
    # e^r = (e^(r/2¹⁰))^(2¹⁰)
    rr = div(r, 1024)
    one = 1 <<< g
    e = taylor_exp(rr, one, 1, one, one)
    e = Enum.reduce(1..10, e, fn _, acc -> div(acc * acc, one) end)
    {e, n - g, 1 <<< 24}
  end

  defp taylor_exp(rr, one, k, term, acc) do
    term = div(div(term * rr, one), k)
    if term == 0, do: acc, else: taylor_exp(rr, one, k + 1, term, acc + term)
  end

  # constants, memoised per precision
  defp ln2_fx(w), do: memo({:ln2, w}, fn -> (2 * atanh_inv(3, w + 32)) >>> 32 end)

  defp pi_fx(w), do: memo({:pi, w}, fn -> (16 * atan_inv(5, w + 32) - 4 * atan_inv(239, w + 32)) >>> 32 end)

  defp atanh_inv(n, w), do: inv_series(div(1 <<< w, n), n * n, 1, 0, 1)
  defp atan_inv(n, w), do: inv_series(div(1 <<< w, n), n * n, 1, 0, -1)

  defp inv_series(0, _n2, _k, acc, _sign), do: acc
  defp inv_series(p, n2, k, acc, sign), do: inv_series(div(p, n2), n2, k + 2, acc + sign_term(div(p, k), k, sign), sign)

  defp sign_term(t, _k, 1), do: t
  defp sign_term(t, k, -1), do: if(rem(div(k - 1, 2), 2) == 0, do: t, else: -t)

  defp memo(key, f) do
    case :persistent_term.get({__MODULE__, key}, nil) do
      nil ->
        v = f.()
        :persistent_term.put({__MODULE__, key}, v)
        v

      v ->
        v
    end
  end

  # ----------------------------------------------------------- rounding --

  # Ziv's loop over a fixed-point evaluator `f.(w) → {I, s, err}`
  defp ziv(prec, f), do: ziv(prec, f, 128)

  defp ziv(_prec, _f, w) when w > 4096, do: raise(ArithmeticError, "Vapor.CR: undecided at 4096 bits (an exact boundary case)")

  defp ziv(prec, f, w) do
    {i, s, err} = f.(w)

    case decide_fixed(i, s, err, prec) do
      {:ok, y} -> y
      :undecided -> ziv(prec, f, 2 * w)
    end
  end

  defp with_err({i, s, e}, _w, extra), do: {i, s, e + extra}

  # value i·2^s ± err·2^s → the prec-bit rounding, if it is the same at both ends
  defp decide_fixed(i, s, err, prec) do
    sign = if i < 0, do: -1, else: 1
    a = abs(i)

    if a <= err do
      :undecided
    else
      lo = rne(a - err, s, prec)
      hi = rne(a + err, s, prec)
      if lo == hi, do: {:ok, in_range(sign * to_float_exact(elem(lo, 0), elem(lo, 1)), prec)}, else: :undecided
    end
  end

  # binary32 results are normal and finite by contract (no subnormal rounding here)
  defp in_range(y, 24) when abs(y) >= 1.1754943508222875e-38 and abs(y) <= 3.4028234663852886e38, do: y
  defp in_range(_y, 24), do: raise(ArithmeticError, "Vapor.CR: binary32 result outside the normal range")
  defp in_range(y, 53), do: y

  # round a positive integer a·2^s to prec significant bits, ties to even:
  # {mantissa, exponent} normalised (mantissa < 2^prec)
  defp rne(a, s, prec) do
    b = bit_len(a)
    sh = b - prec

    if sh <= 0 do
      {a <<< -sh, s + sh}
    else
      q = a >>> sh
      r = a &&& ((1 <<< sh) - 1)
      half = 1 <<< (sh - 1)

      q =
        cond do
          r > half -> q + 1
          r < half -> q
          true -> q + (q &&& 1)
        end

      if q == 1 <<< prec, do: {q >>> 1, s + sh + 1}, else: {q, s + sh}
    end
  end

  defp round_float(i, s, prec), do: (fn {m, e} -> to_float_exact(m, e) end).(rne(abs(i), s, prec)) * if(i < 0, do: -1, else: 1)

  defp round_float_value(x, 53), do: x
  defp round_float_value(x, 24), do: to_f32(x)

  # ------------------------------------------------------------ helpers --

  @doc false
  # x = m·2^e exactly (binary64 inputs; integers as {n, 0})
  def dyadic(x) when is_integer(x), do: {x, 0}

  def dyadic(x) when is_float(x) do
    <<s::1, ex::11, fr::52>> = <<x::float-64>>
    {m, e} = if ex == 0, do: {fr, -1074}, else: {fr ||| 1 <<< 52, ex - 1075}
    {if(s == 1, do: -m, else: m), e}
  end

  # floor(m·2^(e+w)) — exact when e + w ≥ 0
  defp fixed(m, e, w) when e + w >= 0, do: m <<< (e + w)
  defp fixed(m, e, w), do: m >>> -(e + w)

  defp round_div(a, b) do
    q = div(a, b)
    r = a - q * b
    cond do
      2 * abs(r) > abs(b) -> q + if((a < 0) != (b < 0), do: -1, else: 1)
      true -> q
    end
  end

  defp bit_len(0), do: 0
  defp bit_len(n), do: n |> Integer.digits(2) |> length()

  # m·2^e as a binary64 (m < 2^53, result in the normal range), exactly
  defp to_float_exact(0, _e), do: 0.0

  defp to_float_exact(m, e) do
    b = bit_len(m)
    # normalise to 53 bits: value = (m << (53 − b))·2^(e − (53 − b))
    mm = m <<< (53 - b)
    ee = e - (53 - b) + 52 + 1023

    if ee <= 0 or ee >= 2047, do: raise(ArithmeticError, "Vapor.CR: result outside the binary64 normal range")
    <<f::float-64>> = <<0::1, ee::11, (mm &&& (1 <<< 52) - 1)::52>>
    f
  end
end
