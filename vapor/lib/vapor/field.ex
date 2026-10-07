defmodule Vapor.Field do
  @moduledoc """
  Prime fields for proof systems and lattice cryptography — exact arithmetic
  on BEAM integers, the reference (oracle) every future kernel of this kind
  is checked against, as `Vapor.Runtime.Oracle` is for floating point.

  | field | p | 2-adicity | used by |
  |---|---|---|---|
  | `:babybear` | 2³¹ − 2²⁷ + 1 | 27 | STARK provers (RISC Zero, SP1) |
  | `:goldilocks` | 2⁶⁴ − 2³² + 1 | 32 | Plonky2/3-style provers |
  | `:bn254` | r of BN254 (254 bits) | 28 | Groth16/PLONK (circom, snarkjs), Ethereum precompiles |
  | `:ntt30` | 998 244 353 = 119·2²³ + 1 | 23 | an NTT-friendly modulus (negacyclic rings up to N = 2²²) |

  A field is a map `%{name, p, g, s}`: modulus, a multiplicative generator,
  and the 2-adicity `s` (2ˢ ∥ p − 1), so `root_of_unity(f, 2ᵏ)` exists for
  every `k ≤ s`. Integers enter by the signed embedding (`from_int/2`): a
  negative `z` is `p + z`, and `to_int/2` reads values above `p/2` back as
  negatives — injective on `|z| < p/2`, which is what makes an integer
  computation and its image in the field the same computation (see
  `Vapor.ZK`).
  """
  import Bitwise

  @fields %{
    babybear: %{name: :babybear, p: 2_013_265_921, g: 31, s: 27},
    goldilocks: %{name: :goldilocks, p: 18_446_744_069_414_584_321, g: 7, s: 32},
    bn254: %{name: :bn254, p: 21_888_242_871_839_275_222_246_405_745_257_275_088_548_364_400_416_034_343_698_204_186_575_808_495_617, g: 5, s: 28},
    ntt30: %{name: :ntt30, p: 998_244_353, g: 3, s: 23}
  }

  @doc "A named field (`:babybear | :goldilocks | :bn254 | :ntt30`)."
  def get(name), do: Map.fetch!(@fields, name)

  @doc "The fields by name."
  def all, do: @fields

  def add(%{p: p}, a, b), do: rem(a + b, p)
  def sub(%{p: p}, a, b), do: rem(a - b + p, p)
  def neg(%{p: p}, a), do: rem(p - a, p)
  def mul(%{p: p}, a, b), do: rem(a * b, p)

  @doc "aᵉ mod p (square-and-multiply)."
  def pow(%{p: p}, a, e) when e >= 0, do: powmod(rem(a, p), e, p, 1)

  defp powmod(_b, 0, _p, acc), do: acc
  defp powmod(b, e, p, acc), do: powmod(rem(b * b, p), e >>> 1, p, if((e &&& 1) == 1, do: rem(acc * b, p), else: acc))

  @doc "a⁻¹ (Fermat); raises on 0."
  def inv(%{p: p} = f, a) do
    if rem(a, p) == 0, do: raise(ArithmeticError, "0 has no inverse in #{f.name}")
    pow(f, a, p - 2)
  end

  @doc "The signed embedding of an integer (requires `|z| < p/2` to be invertible)."
  def from_int(%{p: p}, z) when is_integer(z), do: Integer.mod(z, p)

  @doc "The integer a field element stands for under the signed embedding."
  def to_int(%{p: p}, a), do: if(a > div(p, 2), do: a - p, else: a)

  @doc "A primitive n-th root of unity, n a power of two dividing p − 1."
  def root_of_unity(%{p: p, g: g, s: s} = f, n) do
    k = trunc(:math.log2(n))
    unless n == 1 <<< k and k <= s, do: raise(ArgumentError, "no #{n}-th root of unity in #{f.name} (2-adicity #{s})")
    pow(f, g, div(p - 1, n))
  end

  @doc "Little-endian bytes of a field element (`n8` bytes)."
  def to_bytes(%{p: p}, a, n8 \\ nil), do: <<a::little-size(8 * (n8 || bytes(p)))>>

  @doc "Bytes per element: the modulus rounded up to 8-byte words (as circom/snarkjs store them)."
  def bytes(p), do: div(bits_of(p) + 63, 64) * 8

  defp bits_of(p), do: p |> Integer.digits(2) |> length()
end

defmodule Vapor.NTT do
  @moduledoc """
  The number-theoretic transform over a `Vapor.Field` — the kernel at the
  heart of both STARK provers (low-degree extension, polynomial commitment)
  and lattice cryptography (products in `Z_q[X]/(Xᴺ + 1)` for BFV, CKKS,
  ML-KEM). Exact reference: iterative radix-2 Cooley–Tukey with bit
  reversal; the negacyclic variant twists by a 2N-th root ψ.

  This module is the oracle a certified NTT kernel will be checked against
  bit for bit (docs/ZK_FHE.md); it is not itself fast.
  """
  import Bitwise
  alias Vapor.Field

  @doc "Forward NTT of a list whose length is a power of two."
  def forward(f, xs), do: transform(f, xs, Field.root_of_unity(f, length(xs)))

  @doc "Inverse NTT."
  def inverse(f, xs) do
    n = length(xs)
    w = Field.inv(f, Field.root_of_unity(f, n))
    ninv = Field.inv(f, n)
    f |> transform(xs, w) |> Enum.map(&Field.mul(f, &1, ninv))
  end

  @doc "Cyclic convolution: a·b mod (Xⁿ − 1)."
  def cyclic(f, a, b), do: inverse(f, Enum.zip_with(forward(f, a), forward(f, b), &Field.mul(f, &1, &2)))

  @doc "Negacyclic convolution: a·b mod (Xⁿ + 1) — the ring of BFV/CKKS/ML-KEM."
  def negacyclic(f, a, b) do
    n = length(a)
    psi = Field.root_of_unity(f, 2 * n)
    pinv = Field.inv(f, psi)
    tw = powers(f, psi, n)
    ut = powers(f, pinv, n)
    twist = fn xs -> Enum.zip_with(xs, tw, &Field.mul(f, &1, &2)) end
    c = cyclic(f, twist.(a), twist.(b))
    Enum.zip_with(c, ut, &Field.mul(f, &1, &2))
  end

  @doc "Schoolbook products (the oracle of the oracle): cyclic or negacyclic."
  def naive(f, a, b, sign) do
    n = length(a)
    at = List.to_tuple(a)
    bt = List.to_tuple(b)

    for k <- 0..(n - 1) do
      Enum.reduce(0..(n - 1), 0, fn i, acc ->
        j = k - i
        {j, s} = if j < 0, do: {j + n, sign}, else: {j, 1}
        term = Field.mul(f, elem(at, i), elem(bt, j))
        if s == 1, do: Field.add(f, acc, term), else: Field.sub(f, acc, term)
      end)
    end
  end

  # [1, x, x², …, xⁿ⁻¹]
  defp powers(f, x, n), do: Enum.map_reduce(1..n, 1, fn _, acc -> {acc, Field.mul(f, acc, x)} end) |> elem(0)

  defp transform(f, xs, w) do
    n = length(xs)
    bits = trunc(:math.log2(n))
    unless n == 1 <<< bits, do: raise(ArgumentError, "length #{n} is not a power of two")
    a = xs |> Enum.with_index() |> Enum.sort_by(fn {_, i} -> rev(i, bits) end) |> Enum.map(&elem(&1, 0)) |> List.to_tuple()
    stages(f, a, w, n, 2) |> Tuple.to_list()
  end

  defp stages(_f, a, _w, n, len) when len > n, do: a

  defp stages(f, a, w, n, len) do
    wl = Field.pow(f, w, div(n, len))
    half = div(len, 2)
    tw = Enum.map_reduce(1..half, 1, fn _, acc -> {acc, Field.mul(f, acc, wl)} end) |> elem(0) |> List.to_tuple()

    a =
      Enum.reduce(0..(n - 1)//len, a, fn start, a ->
        Enum.reduce(0..(half - 1), a, fn j, a ->
          u = elem(a, start + j)
          v = Field.mul(f, elem(a, start + j + half), elem(tw, j))
          a |> put_elem(start + j, Field.add(f, u, v)) |> put_elem(start + j + half, Field.sub(f, u, v))
        end)
      end)

    stages(f, a, w, n, len * 2)
  end

  defp rev(i, bits), do: Enum.reduce(0..(bits - 1)//1, 0, fn b, acc -> acc <<< 1 ||| (i >>> b &&& 1) end)
end
