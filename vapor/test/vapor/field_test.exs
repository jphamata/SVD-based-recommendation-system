defmodule Vapor.FieldTest do
  @moduledoc """
  Prime fields and the NTT (`Vapor.Field`, `Vapor.NTT`): the moduli are
  prime, the generators generate the 2-power subgroups, the transform
  inverts, and NTT products equal schoolbook products — cyclic and
  negacyclic (the ring of lattice cryptography).
  """
  use ExUnit.Case, async: true
  import Bitwise
  alias Vapor.{Field, NTT}

  # Miller–Rabin with 40 fixed bases (deterministic below 2⁶⁴; for BN254 an
  # error probability below 4⁻⁴⁰, and the constant is the published one)
  defp prime?(n) do
    {d, s} = split(n - 1, 0)
    f = %{p: n}

    Enum.all?(Enum.take([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] ++ Enum.to_list(41..400//7), 40), fn a ->
      a = rem(a, n)
      x = Field.pow(f, a, d)
      a == 0 or x == 1 or x == n - 1 or Enum.any?(1..(s - 1)//1, fn r -> Field.pow(f, x, 1 <<< r) == n - 1 end)
    end)
  end

  defp split(d, s) when rem(d, 2) == 0, do: split(div(d, 2), s + 1)
  defp split(d, s), do: {d, s}

  test "moduli, 2-adicity, generators" do
    assert Field.get(:babybear).p == (1 <<< 31) - (1 <<< 27) + 1
    assert Field.get(:goldilocks).p == (1 <<< 64) - (1 <<< 32) + 1

    for {_, f} <- Field.all() do
      assert prime?(f.p), "#{f.name}"
      {_, s} = split(f.p - 1, 0)
      assert s == f.s, "#{f.name}: 2-adicity"
      # g is a non-residue, so the 2ˢ-th root it yields has order exactly 2ˢ
      assert Field.pow(f, f.g, div(f.p - 1, 2)) == f.p - 1
      w = Field.root_of_unity(f, 1 <<< f.s)
      assert Field.pow(f, w, 1 <<< f.s) == 1 and Field.pow(f, w, 1 <<< (f.s - 1)) == f.p - 1
    end
  end

  test "signed embedding: injective below p/2" do
    f = Field.get(:babybear)
    for z <- [0, 1, -1, 127 * 127 * 4096, -(127 * 128 * 4096), div(f.p, 2), -div(f.p, 2)] do
      assert Field.to_int(f, Field.from_int(f, z)) == z
    end
    # past the bound, two integers collide: why the admissibility bound must shrink for small fields
    assert Field.from_int(f, div(f.p, 2) + 1) == Field.from_int(f, div(f.p, 2) + 1 - f.p)
  end

  test "NTT inverts, and products equal schoolbook products (cyclic and negacyclic)" do
    :rand.seed(:exsss, {9, 9, 9})

    for {_, f} <- Field.all(), n <- [1, 2, 8, 64] do
      a = for _ <- 1..n, do: :rand.uniform(f.p) - 1
      b = for _ <- 1..n, do: :rand.uniform(f.p) - 1
      assert NTT.inverse(f, NTT.forward(f, a)) == a
      assert NTT.cyclic(f, a, b) == NTT.naive(f, a, b, 1)
      assert NTT.negacyclic(f, a, b) == NTT.naive(f, a, b, -1)
    end

    # X^(n−1) · X = −1 in Z_q[X]/(Xⁿ + 1)
    f = Field.get(:ntt30)
    x1 = [0, 1] ++ List.duplicate(0, 6)
    xn = List.duplicate(0, 7) ++ [1]
    assert NTT.negacyclic(f, x1, xn) == [f.p - 1 | List.duplicate(0, 7)]
  end
end
