defmodule Vapor.Canon do
  @moduledoc """
  Canonical functions: `exp`, `log`, `rcp`, `rsqrt`, `div`, `sigmoid`, `silu`,
  `max`, `min`, `tanh`, `gelu_tanh`, `gelu` (exact, erf form), `sel` — each defined as a *microprogram*
  over primitives whose results are identical on every substrate:

      add sub mul          correctly rounded binary32 (x86, AArch64, RVV, Vulkan)
      fma                  mul then add (canonical) · fused (:fast)
      neg relu             exact
      sel_lt a b x y       (a < b) ? x : y     ordered compare, NaN → y
      iadd isub iand ixor  32-bit two's-complement lanes (bit patterns)
      shl shr              logical shifts by an immediate

  Division and square root are deliberately *not* primitives: Vulkan allows
  2.5 ulp for `OpFDiv`, and ISAs disagree on `max`/`min` with NaN. Building
  them from the primitives above gives bit-identical results everywhere,
  including GPUs, and the oracle evaluates them by the very same
  microprogram — there is exactly one definition. Division is moreover
  *correctly rounded* (since semantics version 2): with `+ − ×` it makes the
  four basic operations IEEE-754 round-to-nearest on every substrate,
  GPUs included, under the flush-to-zero convention.

  Every function keeps its values finite and normal by construction (inputs
  are clamped into the domain where the recurrence is exact), so devices
  that flush subnormals to zero remain bit-identical.

  A microprogram is a list of `{{:t, k}, op, [operand]}` with operands
  `{:in, i}`, `{:t, k}`, `{:splat, bits}` or `{:imm, n}` (shift counts) —
  the op format of `Vapor.KIR.Kernels.ew/1`, so expansion is inlining.
  """
  import Bitwise
  alias Vapor.F32

  @functions %{exp: 1, log: 1, rcp: 1, rsqrt: 1, div: 2, sigmoid: 1, silu: 1, max: 2, min: 2,
                tanh: 1, gelu_tanh: 1, gelu: 1, sel: 4}
  @primitives %{add: 2, sub: 2, mul: 2, fma: 3, neg: 1, relu: 1, sel_lt: 4,
                iadd: 2, isub: 2, iand: 2, ixor: 2, shl: 2, shr: 2}

  @doc """
  Version of the canonical semantics, recorded in every certificate: a
  program's bits are a function of its terms *and* of this definition.
  1 — through 0.5: `div(a, b) = a·rcp(b)` (within an ulp);
  2 — since 0.6: `div` correctly rounded (IEEE round-to-nearest, DAZ/FTZ).
  """
  def version, do: 2

  def functions, do: @functions
  def primitives, do: @primitives
  def function?(op), do: Map.has_key?(@functions, op)
  def primitive?(op), do: Map.has_key?(@primitives, op)

  # --------------------------------------------------------------- constants --

  defp c(x), do: {:splat, F32.from_float(x)}
  defp bits(b), do: {:splat, b}

  @magic 0x4B40_0000

  # ------------------------------------------------------------- expansion --

  @doc """
  Expand `op` applied to `args` (operands). `k` is the next free temp index.
  Returns `{ops, result_operand, next_k}`; primitives expand to themselves.
  """
  def expand(op, args, k) do
    {ops, {res, k}} = build(op, args, k)
    {ops, res, k}
  end

  defp build(op, args, k) do
    st = %{k: k, ops: []}
    {res, st} = gen(op, args, st)
    {Enum.reverse(st.ops), {res, st.k}}
  end

  # emit one primitive into the builder state
  defp emit(op, args, st) do
    t = {:t, st.k}
    {t, %{st | k: st.k + 1, ops: [{t, op, args} | st.ops]}}
  end

  defp gen(op, args, st) do
    if primitive?(op), do: emit(op, args, st), else: fun(op, args, st)
  end

  # exp(x) = 2^k · p(r),  k = round(x·log₂e),  r = x − k·ln2 (Cody–Waite),
  # p = degree-7 Taylor in Estrin form (depth 3, `Vapor.Estrin` in Lean).
  # x is clamped to [−87, 88]: 2^k stays normal and p·2^k is never
  # subnormal; below −87 the result is +0 by definition.
  defp fun(:exp, [x], st) do
    {lo, hi} = {c(-87.0), c(88.0)}
    {x1, st} = emit(:sel_lt, [x, lo, lo, x], st)
    {xc, st} = emit(:sel_lt, [hi, x1, hi, x1], st)
    {m, st} = emit(:mul, [xc, c(1.4426950408889634)], st)
    {t, st} = emit(:add, [m, bits(@magic)], st)
    {kf, st} = emit(:sub, [t, bits(@magic)], st)
    {h, st} = emit(:mul, [kf, c(0.693359375)], st)
    {r1, st} = emit(:sub, [xc, h], st)
    {l, st} = emit(:mul, [kf, c(-2.1219444005469058e-4)], st)
    {r, st} = emit(:sub, [r1, l], st)
    {p, st} = estrin7(r, Enum.map([1, 1, 2, 6, 24, 120, 720, 5040], &c(1.0 / &1)), st)
    {ki, st} = emit(:isub, [t, bits(@magic)], st)
    {kb, st} = emit(:iadd, [ki, bits(127)], st)
    {scale, st} = emit(:shl, [kb, {:imm, 23}], st)
    {y, st} = emit(:mul, [p, scale], st)
    emit(:sel_lt, [x, lo, c(0.0), y], st)
  end

  # 1/x: Newton–Raphson from the classic bit-level seed on |x| clamped to
  # [2⁻¹²⁵, 2¹²⁵] (seed and iterates stay normal), in residual form
  # y ← y + y·(1 − xy): the residual 1 − xy is exact (Sterbenz) once xy is
  # near 1, so the iteration does not stall one ulp short (it would at
  # y = 0.5⁻ for x = 2 in the y(2 − xy) form). The sign is restored with xor.
  defp fun(:rcp, [x], st) do
    {s, st} = emit(:iand, [x, bits(0x8000_0000)], st)
    {a, st} = emit(:ixor, [x, s], st)
    {a1, st} = emit(:sel_lt, [a, c(:math.pow(2, -125)), c(:math.pow(2, -125)), a], st)
    {ac, st} = emit(:sel_lt, [c(:math.pow(2, 125)), a1, c(:math.pow(2, 125)), a1], st)
    {y0, st} = emit(:isub, [bits(0x7EF3_11C3), ac], st)

    {y, st} =
      Enum.reduce(1..4, {y0, st}, fn _, {y, st} ->
        {xy, st} = emit(:mul, [ac, y], st)
        {e, st} = emit(:sub, [c(1.0), xy], st)
        {ye, st} = emit(:mul, [y, e], st)
        emit(:add, [y, ye], st)
      end)

    emit(:ixor, [y, s], st)
  end

  # 1/√x for x > 0: y ← y + y·(1/2 − (x/2)·y²) from the bit-level seed, x
  # clamped to [2⁻¹²⁵, 2¹²⁴] so x/2 and y² stay normal.
  defp fun(:rsqrt, [x], st) do
    {x1, st} = emit(:sel_lt, [x, c(:math.pow(2, -125)), c(:math.pow(2, -125)), x], st)
    {xc, st} = emit(:sel_lt, [c(:math.pow(2, 124)), x1, c(:math.pow(2, 124)), x1], st)
    {hx, st} = emit(:mul, [xc, c(0.5)], st)
    {sh, st} = emit(:shr, [xc, {:imm, 1}], st)
    {y0, st} = emit(:isub, [bits(0x5F37_5A86), sh], st)

    Enum.reduce(1..3, {y0, st}, fn _, {y, st} ->
      {yy, st} = emit(:mul, [y, y], st)
      {t, st} = emit(:mul, [hx, yy], st)
      {e, st} = emit(:sub, [c(0.5), t], st)
      {ye, st} = emit(:mul, [y, e], st)
      emit(:add, [y, ye], st)
    end)
  end

  # a/b, correctly rounded (IEEE-754 round-to-nearest-even) whenever a, b and
  # the quotient are normal; subnormal inputs are read as zero and subnormal
  # results flushed to zero (DAZ/FTZ, the GPU convention) — so every value
  # this program touches stays normal and the bits agree on every device.
  # Specials as IEEE: x/0 = ±∞ (x ≠ 0), 0/0 = ∞/∞ = NaN, x/∞ = ±0, ∞/x = ±∞;
  # a NaN operand gives the canonical quiet NaN 0x7FC00000.
  #
  # Method (Markstein's correction, without FMA): the significands are
  # brought to a′ ∈ [1, 4), b′ ∈ [1, 2) with q = a′/b′ ∈ [1, 2); q₀ = a′·rcp(b′)
  # is corrected once by its residual, q₁ = q₀ + r₀·y. q₁ is faithful, so its
  # residual r₁ = a′ − q₁b′ is exactly representable and is computed exactly
  # from Dekker's error-free product (Veltkamp splitting by 2¹² + 1). The
  # rounding is then decided exactly: q₁ is correct iff |r₁| ≤ b′·2⁻²⁴ (half
  # an ulp of q ∈ [1, 2), times b′); otherwise it moves one ulp toward r₁.
  # Ties cannot occur (a quotient of 24-bit significands is never a 25-bit
  # midpoint). The exponent is reassembled in integer arithmetic and
  # range-checked by a float comparison on 2²³·1.5 + E.
  defp fun(:div, [a, b], st) do
    {s, st} = emit(:ixor, [a, b], st)
    {s, st} = emit(:iand, [s, bits(0x8000_0000)], st)
    {aa, st} = emit(:iand, [a, bits(0x7FFF_FFFF)], st)
    {ab, st} = emit(:iand, [b, bits(0x7FFF_FFFF)], st)

    # significands in [1, 2) (garbage, but normal, for specials — overridden)
    {ma, st} = emit(:iand, [aa, bits(0x007F_FFFF)], st)
    {ma, st} = emit(:iadd, [ma, bits(0x3F80_0000)], st)
    {mb, st} = emit(:iand, [ab, bits(0x007F_FFFF)], st)
    {mb, st} = emit(:iadd, [mb, bits(0x3F80_0000)], st)

    # a′ < b′ → a′·2 and one less in the exponent (2⁻¹²⁶'s bits = 1 << 23)
    {a2, st} = emit(:add, [ma, ma], st)
    {t, st} = emit(:sel_lt, [ma, mb, a2, ma], st)
    {adj, st} = emit(:sel_lt, [ma, mb, bits(0x0080_0000), bits(0)], st)

    # q₀, one residual correction, then the exact residual of q₁
    {y, st} = fun(:rcp, [mb], st)
    {q0, st} = emit(:mul, [t, y], st)
    {r0, st} = residual(t, q0, mb, st)
    {c0, st} = emit(:mul, [r0, y], st)
    {q1, st} = emit(:add, [q0, c0], st)
    {r1, st} = residual(t, q1, mb, st)

    # round: |r₁| ≤ b′·2⁻²⁴ keeps q₁, else one ulp toward the residual
    {hb, st} = emit(:mul, [mb, c(:math.pow(2, -24))], st)
    {nhb, st} = emit(:neg, [hb], st)
    {up, st} = emit(:iadd, [q1, bits(1)], st)
    {dn, st} = emit(:isub, [q1, bits(1)], st)
    {q, st} = emit(:sel_lt, [r1, nhb, dn, q1], st)
    {q, st} = emit(:sel_lt, [hb, r1, up, q], st)

    # exponent: q's field + (ea − eb) − adj, as integers
    {ea, st} = emit(:iand, [aa, bits(0x7F80_0000)], st)
    {eb, st} = emit(:iand, [ab, bits(0x7F80_0000)], st)
    {de, st} = emit(:isub, [ea, eb], st)
    {de, st} = emit(:isub, [de, adj], st)
    {mag, st} = emit(:iadd, [q, de], st)
    {res, st} = emit(:ixor, [mag, s], st)

    # the biased exponent of the result as a float 1.5·2²³ + E (exact)
    {fq, st} = emit(:shr, [q, {:imm, 23}], st)
    {fe, st} = emit(:shr, [de, {:imm, 23}], st)
    # de >> 23 is the logical shift of a two's-complement multiple of 2²³:
    # bring the 9 remaining bits back to a signed value (± 2⁸ range)
    {fe, st} = emit(:ixor, [fe, bits(0x100)], st)
    {fe, st} = emit(:isub, [fe, bits(0x100)], st)
    {ef, st} = emit(:iadd, [fq, fe], st)
    {ef, st} = emit(:iadd, [ef, bits(@magic)], st)

    # classes of the operands as flags 0 / 2⁻¹²⁶ (integer compare by sign bit)
    {za, st} = int_lt(aa, bits(0x0080_0000), st)
    {zb, st} = int_lt(ab, bits(0x0080_0000), st)
    {ia, st} = int_lt(bits(0x7F7F_FFFF), aa, st)
    {ib, st} = int_lt(bits(0x7F7F_FFFF), ab, st)
    {na, st} = int_lt(bits(0x7F80_0000), aa, st)
    {nb, st} = int_lt(bits(0x7F80_0000), ab, st)

    zero_s = s
    {inf_s, st} = emit(:ixor, [s, bits(0x7F80_0000)], st)

    # result range: E ≤ 0 flushes to ±0, E ≥ 255 overflows to ±∞
    {res, st} = emit(:sel_lt, [ef, c(12_582_913.0), zero_s, res], st)
    {res, st} = emit(:sel_lt, [c(12_583_166.0), ef, inf_s, res], st)

    # specials, in increasing priority: zeros, infinities, NaN
    {zc, st} = emit(:iadd, [za, ib], st)
    {res, st} = emit(:sel_lt, [c(0.0), zc, zero_s, res], st)
    {ic, st} = emit(:iadd, [ia, zb], st)
    {res, st} = emit(:sel_lt, [c(0.0), ic, inf_s, res], st)
    {zz, st} = emit(:iand, [za, zb], st)
    {ii, st} = emit(:iand, [ia, ib], st)
    {nc, st} = emit(:iadd, [na, nb], st)
    {nc, st} = emit(:iadd, [nc, zz], st)
    {nc, st} = emit(:iadd, [nc, ii], st)
    emit(:sel_lt, [c(0.0), nc, bits(0x7FC0_0000), res], st)
  end

  # ln x, IEEE specials: x < 2⁻¹²⁶ (zeros and, read as zero, subnormals) →
  # −∞, x < 0 → NaN, +∞ → +∞, NaN → the canonical NaN. For normal x the
  # method of fdlibm/musl `logf`: x = 2ᵏ·(1 + f) with 1 + f ∈ [√½, √2)
  # (found in integer arithmetic), s = f/(2 + f), and
  #   ln x = k·ln2_hi − ((½f² − (s·(½f² + R(s²)) + k·ln2_lo)) − f)
  # with R the degree-4 minimax in s² — under an ulp (measured: the test
  # suite compares against the correctly rounded logarithm). `s` is
  # f·rcp(2 + f): within an ulp, and it only scales the correction term
  # s·(½f² + R) (at most ~5 % of the result), so a correctly rounded
  # division would buy nothing — and would not fit one kernel's registers.
  # Every intermediate is normal.
  defp fun(:log, [x], st) do
    {y, st} = log_normal(x, st)
    {ax, st} = emit(:iand, [x, bits(0x7FFF_FFFF)], st)
    {neg, st} = emit(:iand, [x, bits(0x8000_0000)], st)
    {zero, st} = int_lt(ax, bits(0x0080_0000), st)
    {inf, st} = int_lt(bits(0x7F7F_FFFF), ax, st)
    {nan, st} = int_lt(bits(0x7F80_0000), ax, st)
    # negative nonzero (normal or infinite) operands: NaN
    {nz, st} = emit(:sel_lt, [c(0.0), zero, bits(0), bits(0x0080_0000)], st)
    {negnz, st} = emit(:shr, [neg, {:imm, 8}], st)
    {negnz, st} = emit(:iand, [negnz, nz], st)
    {y, st} = emit(:sel_lt, [c(0.0), inf, bits(0x7F80_0000), y], st)
    {y, st} = emit(:sel_lt, [c(0.0), zero, bits(0xFF80_0000), y], st)
    {bad, st} = emit(:iadd, [nan, negnz], st)
    emit(:sel_lt, [c(0.0), bad, bits(0x7FC0_0000), y], st)
  end

  defp fun(:sigmoid, [x], st) do
    {nx, st} = emit(:neg, [x], st)
    {e, st} = fun(:exp, [nx], st)
    {d, st} = emit(:add, [c(1.0), e], st)
    fun(:rcp, [d], st)
  end

  defp fun(:silu, [x], st) do
    {s, st} = fun(:sigmoid, [x], st)
    emit(:mul, [x, s], st)
  end

  # tanh(x), total: x is clamped to [−16, 16] (tanh(±16) rounds to ±1).
  # For |x| ≥ 1/4, 2·σ(2x) − 1 (the cancellation costs at most two bits
  # there); below, the odd Taylor polynomial to x⁹ (truncation < 10⁻⁸
  # relative), evaluated on x with large arguments replaced by 0 so that no
  # discarded branch can overflow — small arguments keep their relative
  # accuracy, which logit soft-capping (`cap·tanh(z/cap)`) needs near zero.
  defp fun(:tanh, [x], st) do
    {x1, st} = emit(:sel_lt, [x, c(-16.0), c(-16.0), x], st)
    {xc, st} = emit(:sel_lt, [c(16.0), x1, c(16.0), x1], st)
    {x2, st} = emit(:add, [xc, xc], st)
    {s, st} = fun(:sigmoid, [x2], st)
    {s2, st} = emit(:add, [s, s], st)
    {big, st} = emit(:sub, [s2, c(1.0)], st)
    {nx, st} = emit(:neg, [xc], st)
    {ax, st} = emit(:sel_lt, [xc, nx, nx, xc], st)
    {xs, st} = emit(:sel_lt, [ax, c(0.25), xc, c(0.0)], st)
    {xx, st} = emit(:mul, [xs, xs], st)
    # x·(1 + x²·(−1/3 + x²·(2/15 + x²·(−17/315 + x²·62/2835))))
    {p, st} = emit(:mul, [xx, c(62 / 2835)], st)
    {p, st} = emit(:add, [p, c(-17 / 315)], st)
    {p, st} = emit(:mul, [xx, p], st)
    {p, st} = emit(:add, [p, c(2 / 15)], st)
    {p, st} = emit(:mul, [xx, p], st)
    {p, st} = emit(:add, [p, c(-1 / 3)], st)
    {p, st} = emit(:mul, [xx, p], st)
    {p, st} = emit(:mul, [xs, p], st)
    {small, st} = emit(:add, [xs, p], st)
    emit(:sel_lt, [ax, c(0.25), small, big], st)
  end

  # GELU, tanh form (PyTorch `gelu(approximate="tanh")`, Gemma's
  # `gelu_pytorch_tanh`): ½x(1 + tanh(z)) = x·σ(2z), z = √(2/π)(x + 0.044715x³).
  # The cubic is taken on x clamped to [−16, 16] (beyond, σ is 0 or 1 to
  # binary32 precision), and below −10, where |gelu(x)| < 10⁻³⁶, the result
  # is −0 — as PyTorch's form gives there, by cancellation.
  defp fun(:gelu_tanh, [x], st) do
    {x1, st} = emit(:sel_lt, [x, c(-16.0), c(-16.0), x], st)
    {xc, st} = emit(:sel_lt, [c(16.0), x1, c(16.0), x1], st)
    {x2, st} = emit(:mul, [xc, xc], st)
    {x3, st} = emit(:mul, [x2, xc], st)
    {cx3, st} = emit(:mul, [x3, c(0.044715)], st)
    {u, st} = emit(:add, [xc, cx3], st)
    {z2, st} = emit(:mul, [u, c(1.5957691216057308)], st)
    {s, st} = fun(:sigmoid, [z2], st)
    {y, st} = emit(:mul, [x, s], st)
    emit(:sel_lt, [x, c(-10.0), bits(0x8000_0000), y], st)
  end

  # GELU, exact form (PyTorch `gelu(approximate="none")`, the `gelu` of
  # ViT, BERT, Whisper, CLIP): x·½(1 + erf(x/√2)), in PyTorch's order, so
  # the far negative tail cancels to ±0 as it does there. erf(y) = 1 −
  # erfc(|y|) with the sign restored; erfc is the Chebyshev form of
  # Numerical Recipes (`erfcc`, fractional error < 1.2·10⁻⁷ for all y ≥ 0):
  #   erfc(z) = t·exp(−z² − 1.26551223 + t·P(t)),  t = 1/(1 + z/2).
  # The argument is clamped to [−16, 16]; beyond, erf is ±1 in binary32
  # (and exp's own clamp keeps every intermediate normal).
  defp fun(:gelu, [x], st) do
    {x1, st} = emit(:sel_lt, [x, c(-16.0), c(-16.0), x], st)
    {xc, st} = emit(:sel_lt, [c(16.0), x1, c(16.0), x1], st)
    {y, st} = emit(:mul, [xc, c(0.7071067811865476)], st)
    {s, st} = emit(:iand, [y, bits(0x8000_0000)], st)
    {z, st} = emit(:ixor, [y, s], st)
    {hz, st} = emit(:mul, [z, c(0.5)], st)
    {den, st} = emit(:add, [c(1.0), hz], st)
    {t, st} = fun(:rcp, [den], st)

    coeffs = [0.17087277, -0.82215223, 1.48851587, -1.13520398, 0.27886807, -0.18628806,
              0.09678418, 0.37409196, 1.00002368, -1.26551223]

    {p, st} =
      Enum.reduce(tl(coeffs), {c(hd(coeffs)), st}, fn a, {acc, st} ->
        {m, st} = emit(:mul, [t, acc], st)
        emit(:add, [m, c(a)], st)
      end)

    {zz, st} = emit(:mul, [z, z], st)
    {arg, st} = emit(:sub, [p, zz], st)
    {e, st} = fun(:exp, [arg], st)
    {erfc, st} = emit(:mul, [t, e], st)
    {erf_abs, st} = emit(:sub, [c(1.0), erfc], st)
    {erf, st} = emit(:ixor, [erf_abs, s], st)
    {one_p, st} = emit(:add, [c(1.0), erf], st)
    {hx, st} = emit(:mul, [x, c(0.5)], st)
    emit(:mul, [hx, one_p], st)
  end

  # sel(a, b, x, y) = (a < b) ? x : y — the selection primitive itself,
  # exposed as an operator (ordered comparison: NaN selects y)
  defp fun(:sel, [a, b, x, y], st), do: emit(:sel_lt, [a, b, x, y], st)

  defp fun(:max, [a, b], st), do: emit(:sel_lt, [a, b, b, a], st)
  defp fun(:min, [a, b], st), do: emit(:sel_lt, [b, a, b, a], st)

  # ln x for x normal and positive (other operands give garbage, normal):
  # the reduction of musl's `logf` in integer arithmetic, then its kernel
  defp log_normal(x, st) do
    # x clamped into the normal positive range so every step stays normal
    {xa, st} = emit(:iand, [x, bits(0x7FFF_FFFF)], st)
    {xa, st} = emit(:sel_lt, [xa, c(:math.pow(2, -126)), c(:math.pow(2, -126)), xa], st)
    {xa, st} = emit(:sel_lt, [c(3.4028234663852886e38), xa, c(3.4028234663852886e38), xa], st)
    {ix, st} = emit(:iadd, [xa, bits(0x3F80_0000 - 0x3F35_04F3)], st)
    {kb, st} = emit(:shr, [ix, {:imm, 23}], st)
    # k as a float: (2²³·1.5 + k + 127) − (2²³·1.5 + 127), exact
    {kf, st} = emit(:iadd, [kb, bits(@magic)], st)
    {dk, st} = emit(:sub, [kf, c(12_582_912.0 + 127)], st)
    {m, st} = emit(:iand, [ix, bits(0x007F_FFFF)], st)
    {m, st} = emit(:iadd, [m, bits(0x3F35_04F3)], st)
    {f, st} = emit(:sub, [m, c(1.0)], st)
    {den, st} = emit(:add, [f, c(2.0)], st)
    {rd, st} = fun(:rcp, [den], st)
    {s, st} = emit(:mul, [f, rd], st)
    {z, st} = emit(:mul, [s, s], st)
    {w, st} = emit(:mul, [z, z], st)
    # Lg1..Lg4 of musl (0xaaaaaa.0p-24, 0xccce13.0p-25, 0x91e9ee.0p-25, 0xf89e26.0p-26)
    {a, st} = emit(:mul, [w, bits(0x3E78_9E26)], st)
    {a, st} = emit(:add, [a, bits(0x3ECC_CE13)], st)
    {t1, st} = emit(:mul, [w, a], st)
    {b, st} = emit(:mul, [w, bits(0x3E91_E9EE)], st)
    {b, st} = emit(:add, [b, bits(0x3F2A_AAAA)], st)
    {t2, st} = emit(:mul, [z, b], st)
    {r, st} = emit(:add, [t2, t1], st)
    {hf, st} = emit(:mul, [f, c(0.5)], st)
    {hfsq, st} = emit(:mul, [hf, f], st)
    {e, st} = emit(:add, [hfsq, r], st)
    {e, st} = emit(:mul, [s, e], st)
    {lo, st} = emit(:mul, [dk, bits(0x3717_F7D1)], st)
    {e, st} = emit(:add, [e, lo], st)
    {e, st} = emit(:sub, [e, hfsq], st)
    {e, st} = emit(:add, [e, f], st)
    {hi, st} = emit(:mul, [dk, bits(0x3F31_7180)], st)
    emit(:add, [e, hi], st)
  end

  # x − q·b exactly when the exact value is representable (q faithful):
  # Dekker's product q·b = p + e with Veltkamp halves (2¹² + 1 splitting),
  # x − p exact by Sterbenz, then − e. All operands lie in [2⁻⁵⁰, 4]: normal.
  defp residual(x, q, b, st) do
    {qh, ql, st} = split(q, st)
    {bh, bl, st} = split(b, st)
    {p, st} = emit(:mul, [q, b], st)
    {hh, st} = emit(:mul, [qh, bh], st)
    {e, st} = emit(:sub, [hh, p], st)
    {hl, st} = emit(:mul, [qh, bl], st)
    {e, st} = emit(:add, [e, hl], st)
    {lh, st} = emit(:mul, [ql, bh], st)
    {e, st} = emit(:add, [e, lh], st)
    {ll, st} = emit(:mul, [ql, bl], st)
    {e, st} = emit(:add, [e, ll], st)
    {d, st} = emit(:sub, [x, p], st)
    emit(:sub, [d, e], st)
  end

  defp split(x, st) do
    {cx, st} = emit(:mul, [x, c(4097.0)], st)
    {d, st} = emit(:sub, [cx, x], st)
    {hi, st} = emit(:sub, [cx, d], st)
    {lo, st} = emit(:sub, [x, hi], st)
    {hi, lo, st}
  end

  # x < y for 31-bit magnitudes, as the flag +0 / 2⁻¹²⁶ (the sign bit of
  # x − y, moved to bit 23: a normal float, safe on flush-to-zero devices)
  defp int_lt(x, y, st) do
    {d, st} = emit(:isub, [x, y], st)
    {d, st} = emit(:shr, [d, {:imm, 31}], st)
    emit(:shl, [d, {:imm, 23}], st)
  end

  # p(r) = Σ cᵢ rⁱ, i ≤ 7, in Estrin's scheme (every product rounded separately)
  defp estrin7(r, [c0, c1, c2, c3, c4, c5, c6, c7], st) do
    lin = fn a, b, st ->
      {m, st} = emit(:mul, [b, r], st)
      emit(:add, [a, m], st)
    end

    {p01, st} = lin.(c0, c1, st)
    {p23, st} = lin.(c2, c3, st)
    {p45, st} = lin.(c4, c5, st)
    {p67, st} = lin.(c6, c7, st)
    {r2, st} = emit(:mul, [r, r], st)
    {r4, st} = emit(:mul, [r2, r2], st)
    {m0, st} = emit(:mul, [r2, p23], st)
    {q0, st} = emit(:add, [p01, m0], st)
    {m1, st} = emit(:mul, [r2, p67], st)
    {q1, st} = emit(:add, [p45, m1], st)
    {m2, st} = emit(:mul, [r4, q1], st)
    emit(:add, [q0, m2], st)
  end

  # ------------------------------------------------------------- semantics --

  @doc "Evaluate `op` on binary32 bit patterns by running its microprogram."
  def eval(op, xs, policy \\ :canonical) do
    {ops, res, _} = expand(op, Enum.with_index(xs, fn _, i -> {:in, i} end), 0)
    ins = List.to_tuple(xs)

    env =
      Enum.reduce(ops, %{}, fn {t, p, args}, env ->
        Map.put(env, t, prim(p, Enum.map(args, &operand(&1, ins, env)), policy))
      end)

    operand(res, ins, env)
  end

  @doc "Pre-expanded evaluator for `op`: a function from operand bit patterns to bits."
  def compile(op, policy \\ :canonical) do
    {ops, res, _} = expand(op, Enum.map(0..(Map.fetch!(@functions, op) - 1), &{:in, &1}), 0)

    fn xs ->
      ins = List.to_tuple(xs)

      env =
        Enum.reduce(ops, %{}, fn {t, p, args}, env ->
          Map.put(env, t, prim(p, Enum.map(args, &operand(&1, ins, env)), policy))
        end)

      operand(res, ins, env)
    end
  end

  defp operand({:in, i}, ins, _env), do: elem(ins, i)
  defp operand({:t, _} = t, _ins, env), do: Map.fetch!(env, t)
  defp operand({:splat, b}, _ins, _env), do: b
  defp operand({:imm, n}, _ins, _env), do: {:imm, n}

  @doc "One primitive on bit patterns."
  def prim(:add, [a, b], _), do: F32.add(a, b)
  def prim(:sub, [a, b], _), do: F32.sub(a, b)
  def prim(:mul, [a, b], _), do: F32.mul(a, b)
  def prim(:fma, [a, b, c], :canonical), do: F32.add(F32.mul(a, b), c)
  def prim(:fma, [a, b, c], :fast), do: F32.fma(a, b, c)
  def prim(:neg, [a], _), do: F32.neg(a)
  def prim(:relu, [a], _), do: F32.relu(a)
  def prim(:sel_lt, [a, b, x, y], _), do: if(F32.lt?(a, b), do: x, else: y)
  def prim(:iadd, [a, b], _), do: a + b &&& 0xFFFF_FFFF
  def prim(:isub, [a, b], _), do: a - b &&& 0xFFFF_FFFF
  def prim(:iand, [a, b], _), do: a &&& b
  def prim(:ixor, [a, b], _), do: bxor(a, b)
  def prim(:shl, [a, {:imm, n}], _), do: a <<< n &&& 0xFFFF_FFFF
  def prim(:shr, [a, {:imm, n}], _), do: a >>> n
end
