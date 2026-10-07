defmodule Vapor.F32 do
  @moduledoc """
  Exact IEEE-754 binary32 arithmetic on the BEAM — the numeric ground truth
  that every substrate (x86, ARM64, RISC-V, SPIR-V) is compared against.

  Values are carried as raw 32-bit patterns, never as "approximately f32"
  floats, so `-0.0`, subnormals and the exact rounding of every operation are
  observable and digestible.

    * `add/2`, `sub/2`, `mul/2` evaluate in binary64 and round once to
      binary32. This is *correctly rounded*: double rounding through a format
      with p' ≥ 2p + 2 bits of precision is innocuous (Figueroa, 1995;
      53 ≥ 2·24 + 2).
    * `fma/3` is computed in exact dyadic arithmetic (arbitrary-precision
      integers) and rounded exactly once — no double-rounding argument needed.
    * `round_dyadic/2` is the single rounding primitive (round-to-nearest,
      ties-to-even, gradual underflow, overflow to ±∞).

  Non-finite operands raise: a certificate only covers finite executions.
  """
  import Bitwise

  @type bits :: 0..0xFFFF_FFFF
  @sign 0x8000_0000

  # ---------------------------------------------------------------- codecs --

  @doc "Decode a little-endian binary of binary32 values into bit patterns."
  @spec decode(binary) :: [bits]
  def decode(bin) when rem(byte_size(bin), 4) == 0, do: for(<<b::32-little <- bin>>, do: b)

  @doc "Encode bit patterns as a little-endian binary."
  @spec encode([bits]) :: binary
  def encode(list), do: for(b <- list, into: <<>>, do: <<b::32-little>>)

  @doc "Round an Elixir float (binary64) to the nearest binary32 pattern."
  @spec from_float(float | integer) :: bits
  def from_float(x) when is_number(x) do
    <<b::32>> = <<x * 1.0::float-32>>
    b
  end

  @doc "Exact binary32 → binary64 widening (raises on ±∞/NaN)."
  @spec to_float(bits) :: float
  def to_float(b) do
    case <<b::32>> do
      <<x::float-32>> -> x
      _ -> raise ArithmeticError, "non-finite binary32 0x#{Integer.to_string(b, 16)}"
    end
  end

  def finite?(b), do: (b >>> 23 &&& 0xFF) != 0xFF
  def zero?(b), do: (b &&& 0x7FFF_FFFF) == 0
  def subnormal?(b), do: (b >>> 23 &&& 0xFF) == 0 and (b &&& 0x7F_FFFF) != 0
  def negative?(b), do: (b &&& @sign) != 0

  # ------------------------------------------------------------ arithmetic --

  @spec add(bits, bits) :: bits
  def add(a, b), do: from_float(to_float(a) + to_float(b))

  @spec sub(bits, bits) :: bits
  def sub(a, b), do: from_float(to_float(a) - to_float(b))

  @spec mul(bits, bits) :: bits
  def mul(a, b), do: from_float(to_float(a) * to_float(b))

  @doc "Negation is a sign-bit flip (exact, total)."
  @spec neg(bits) :: bits
  def neg(a), do: bxor(a, @sign)

  @doc "relu(x) = x > 0 ? x : +0 — the canonical definition all backends realize."
  @spec relu(bits) :: bits
  def relu(a), do: if(not negative?(a) and not zero?(a) and not nan?(a), do: a, else: 0)

  def nan?(b), do: (b >>> 23 &&& 0xFF) == 0xFF and (b &&& 0x7F_FFFF) != 0

  @doc "Ordered less-than (IEEE `compareQuietLess`): false if either is NaN; −0 = +0."
  @spec lt?(bits, bits) :: boolean
  def lt?(a, b) do
    cond do
      nan?(a) or nan?(b) -> false
      zero?(a) and zero?(b) -> false
      true -> key(a) < key(b)
    end
  end

  # total order key on non-NaN patterns (±∞ included)
  defp key(b), do: if(negative?(b), do: -(b &&& 0x7FFF_FFFF), else: b)

  @doc "Fused multiply-add a·b + c with a single rounding."
  @spec fma(bits, bits, bits) :: bits
  def fma(a, b, c) do
    {ma, ea} = to_dyadic(a)
    {mb, eb} = to_dyadic(b)
    {mc, ec} = to_dyadic(c)
    {mp, ep} = {ma * mb, ea + eb}
    {m, e} = dyadic_add({mp, ep}, {mc, ec})

    cond do
      m != 0 ->
        round_dyadic(m, e)

      # exact zero: IEEE 754 §6.3 — −0 only when both addends are −0
      mp == 0 and mc == 0 ->
        if (bxor(a, b) &&& @sign) != 0 and negative?(c), do: @sign, else: 0

      true ->
        0
    end
  end

  # ------------------------------------------------------ dyadic rationals --

  @doc "Exact value of a finite binary32 as `{m, e}` with value m·2^e."
  @spec to_dyadic(bits) :: {integer, integer}
  def to_dyadic(b) do
    exp = b >>> 23 &&& 0xFF
    frac = b &&& 0x7F_FFFF
    s = if negative?(b), do: -1, else: 1

    cond do
      exp == 0xFF -> raise ArithmeticError, "non-finite binary32"
      exp == 0 -> {s * frac, -149}
      true -> {s * (frac ||| 0x80_0000), exp - 150}
    end
  end

  @doc "Exact sum of two dyadic rationals."
  def dyadic_add({m1, e1}, {m2, e2}) when e1 <= e2, do: {m1 + (m2 <<< (e2 - e1)), e1}
  def dyadic_add(x, y), do: dyadic_add(y, x)

  @doc """
  Round m·2^e to binary32: round-to-nearest-even, gradual underflow,
  overflow to ±∞. The one rounding primitive of the numeric ground truth.
  """
  @spec round_dyadic(integer, integer) :: bits
  def round_dyadic(0, _e), do: 0

  def round_dyadic(m, e) do
    s = if m < 0, do: @sign, else: 0
    a = abs(m)
    len = bitlen(a)
    # quantum exponent: 24 significant bits, never finer than 2^-149
    q = max(len - 1 + e - 23, -149)
    shift = q - e
    mant = if shift <= 0, do: a <<< -shift, else: rne(a, shift)
    {mant, q} = if mant == 1 <<< 24, do: {1 <<< 23, q + 1}, else: {mant, q}

    cond do
      mant < 1 <<< 23 -> s ||| mant
      q + 150 >= 255 -> s ||| 0x7F80_0000
      true -> s ||| (q + 150) <<< 23 ||| (mant - (1 <<< 23))
    end
  end

  defp rne(a, shift) do
    t = a >>> shift
    low = a &&& (1 <<< shift) - 1
    half = 1 <<< (shift - 1)
    if low > half or (low == half and (t &&& 1) == 1), do: t + 1, else: t
  end

  defp bitlen(a), do: bitlen(a, 0)
  defp bitlen(0, n), do: n
  defp bitlen(a, n) when a >= 1 <<< 64, do: bitlen(a >>> 64, n + 64)
  defp bitlen(a, n), do: bitlen(a >>> 1, n + 1)
end
