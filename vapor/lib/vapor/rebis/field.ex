defmodule Vapor.Rebis.Field do
  @moduledoc """
  Binary extension fields GF(2ⁿ) = GF(2)[x]/(p): elements are integers
  whose bits are polynomial coefficients; addition is XOR; multiplication
  is the **carry-less** product reduced modulo `p` — what `PCLMULQDQ`
  (x86), `PMULL` (ARM) and `vclmul` (RISC-V Zvbc) compute in hardware.

  Two fields are named because they carry the world's encrypted traffic:
  GF(2⁸) mod `x⁸+x⁴+x³+x+1` (AES's S-box is the inverse there, followed by
  an affine map — derived here, not tabulated) and GF(2¹²⁸) mod
  `x¹²⁸+x⁷+x²+x+1` (GCM's authenticator, GHASH).
  """
  import Bitwise

  @aes 0x11B
  @gcm (1 <<< 128) ||| 0x87

  @doc "The AES field polynomial `x⁸+x⁴+x³+x+1`."
  def aes_poly, do: @aes
  @doc "The GCM field polynomial `x¹²⁸+x⁷+x²+x+1`."
  def gcm_poly, do: @gcm

  @doc "Carry-less product of two non-negative integers (polynomials over GF(2))."
  def clmul(a, b) when a >= 0 and b >= 0, do: clmul(a, b, 0)
  defp clmul(_a, 0, acc), do: acc
  defp clmul(a, b, acc), do: clmul(a <<< 1, b >>> 1, if((b &&& 1) == 1, do: bxor(acc, a), else: acc))

  @doc "Degree of a non-zero polynomial (−1 for 0)."
  def degree(0), do: -1
  def degree(a), do: deg(a, -1)
  defp deg(0, d), do: d
  defp deg(a, d) when a >= 1 <<< 64, do: deg(a >>> 64, d + 64)
  defp deg(a, d), do: deg(a >>> 1, d + 1)

  @doc "Remainder of polynomial division `a mod p`."
  def reduce(a, p) do
    dp = degree(p)
    do_reduce(a, p, dp)
  end

  defp do_reduce(a, p, dp) do
    da = degree(a)
    if da < dp, do: a, else: do_reduce(bxor(a, p <<< (da - dp)), p, dp)
  end

  @doc "Product in GF(2)[x]/(p)."
  def mul(a, b, p), do: reduce(clmul(a, b), p)

  @doc "Power by squaring in GF(2)[x]/(p)."
  def pow(_a, 0, _p), do: 1
  def pow(a, e, p) when rem(e, 2) == 0, do: pow(mul(a, a, p), div(e, 2), p)
  def pow(a, e, p), do: mul(a, pow(a, e - 1, p), p)

  @doc """
  Inverse in GF(2)[x]/(p) by the extended Euclidean algorithm over GF(2)[x]
  (`p` irreducible); `inv(0)` is `0` by the AES convention.
  """
  def inv(0, _p), do: 0

  def inv(a, p) do
    {g, s} = egcd(p, reduce(a, p), 0, 1)
    if g != 1, do: raise(ArithmeticError, "#{a} has no inverse modulo #{p} (not irreducible?)")
    reduce(s, p)
  end

  # invariant: r_i ≡ s_i · a (mod p)
  defp egcd(r0, 0, s0, _s1), do: {r0, s0}

  defp egcd(r0, r1, s0, s1) do
    {q, r} = divmod(r0, r1)
    egcd(r1, r, s1, bxor(s0, clmul(q, s1)))
  end

  @doc "Polynomial division with remainder over GF(2)."
  def divmod(a, b) when b != 0 do
    db = degree(b)
    do_div(a, b, db, 0)
  end

  defp do_div(a, b, db, q) do
    da = degree(a)
    if da < db, do: {q, a}, else: do_div(bxor(a, b <<< (da - db)), b, db, q ||| 1 <<< (da - db))
  end

  @doc "Whether `p` (degree n) is irreducible over GF(2) (Rabin's test)."
  def irreducible?(p) do
    n = degree(p)
    if n < 1, do: false, else: rabin(p, n)
  end

  defp rabin(p, n) do
    # x^(2^n) ≡ x (mod p), and gcd(x^(2^(n/q)) − x, p) = 1 for each prime q | n
    xpow = fn k -> Enum.reduce(1..k//1, 2, fn _, acc -> mul(acc, acc, p) end) end
    primes = for q <- 2..n, rem(n, q) == 0, Enum.all?(2..max(2, q - 1)//1, &(&1 == q or rem(q, &1) != 0)), do: q
    reduce(bxor(xpow.(n), 2), p) == 0 and Enum.all?(primes, fn q -> pgcd(p, bxor(xpow.(div(n, q)), 2)) == 1 end)
  end

  defp pgcd(a, 0), do: a
  defp pgcd(a, b), do: pgcd(b, elem(divmod(a, b), 1))

  # ------------------------------------------------------------ AES S-box

  @doc "AES S-box from first principles: `affine(x⁻¹)` in GF(2⁸) (FIPS-197 §5.1.1)."
  def sbox(x) when x in 0..255 do
    b = inv(x, @aes)
    # bᵢ' = bᵢ ⊕ bᵢ₊₄ ⊕ bᵢ₊₅ ⊕ bᵢ₊₆ ⊕ bᵢ₊₇ ⊕ cᵢ, c = 0x63
    bxor(bxor(bxor(bxor(bxor(b, rotl8(b, 1)), rotl8(b, 2)), rotl8(b, 3)), rotl8(b, 4)), 0x63)
  end

  @doc "The inverse S-box (the affine map inverted, then the field inverse)."
  def inv_sbox(y) when y in 0..255 do
    b = bxor(bxor(bxor(rotl8(y, 1), rotl8(y, 3)), rotl8(y, 6)), 0x05)
    inv(b, @aes)
  end

  defp rotl8(b, k), do: (b <<< k ||| b >>> (8 - k)) &&& 0xFF

  # ------------------------------------------------------------- GHASH

  @doc """
  Multiplication in GCM's GF(2¹²⁸) exactly as NIST SP 800-38D Algorithm 1
  states it: 128-bit blocks, bit 0 the *most* significant ("reflected"),
  right shifts with `R = 11100001 ‖ 0¹²⁰`.
  """
  def gcm_mul_spec(x, y) do
    r = 0xE1 <<< 120

    {z, _} =
      Enum.reduce(0..127, {0, y}, fn i, {z, v} ->
        z = if (x >>> (127 - i) &&& 1) == 1, do: bxor(z, v), else: z
        v = if (v &&& 1) == 1, do: bxor(v >>> 1, r), else: v >>> 1
        {z, v}
      end)

    z
  end

  @doc """
  The same product by the other road: reflect both blocks into ordinary
  polynomial order, carry-less multiply, reduce mod `x¹²⁸+x⁷+x²+x+1`,
  reflect back. Agreeing with `gcm_mul_spec/2` is a theorem checked by test.
  """
  def gcm_mul(x, y), do: x |> reflect128() |> mul(reflect128(y), @gcm) |> reflect128()

  @doc "Bit reversal of a 128-bit block."
  def reflect128(x) do
    {v, _} = for <<b::1 <- <<x::128>> >>, reduce: {0, 0}, do: ({acc, i} -> {acc ||| b <<< i, i + 1})
    v
  end

  @doc "GHASH_H over a binary whose length is a multiple of 16 bytes."
  def ghash(h, bin, mul \\ &gcm_mul/2) when rem(byte_size(bin), 16) == 0 do
    for <<blk::128 <- bin>>, reduce: 0 do
      y -> mul.(bxor(y, blk), h)
    end
  end
end
