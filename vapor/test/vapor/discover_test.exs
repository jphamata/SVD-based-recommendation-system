defmodule Vapor.DiscoverTest do
  @moduledoc """
  Discovery with certificates (docs/DESCOBERTA.md): every found algorithm
  is checked by something that does not trust the search — the 0-1
  principle, the exact matrix-multiplication tensor over the integers,
  exhaustive evaluation on a word size — and each claim has its control.
  """
  use ExUnit.Case, async: true
  import Bitwise
  alias Vapor.Discover, as: D

  @moduletag timeout: 600_000

  test "sorting networks: the known minimal sizes for n ≤ 8, certified by the 0-1 principle; random-and-pruned is worse" do
    for n <- 3..8 do
      r = D.network(n)
      assert r.sorts
      assert r.size == r.known.size, "n=#{n}: #{r.size} vs known #{r.known.size}"
    end

    # the control: a random network, every redundant comparator removed, is larger
    c = D.random_network(8, 1)
    assert c.sorts and c.size > 19
    # and the certificate is not vacuous: drop any comparator of the found network and it no longer sorts
    %{net: net} = D.network(8)
    assert Enum.all?(0..(length(net) - 1), fn k -> not D.sorts?(8, List.delete_at(net, k)) end)
  end

  test "a found network is a vapor program: oracle bits = a sort, on random floats" do
    %{net: net} = D.network(6)
    b = 64
    p = D.network_program(6, net, b)
    {:ok, c} = Vapor.Compile.Lower.lower(p)
    cols = for i <- 0..5, do: Vapor.Tensor.random(:f32, [b], 100 + i)
    env = Map.new(Enum.with_index(cols), fn {t, i} -> {:"w#{i}", t} end)
    {:ok, out} = Vapor.Runtime.Native.run_oracle(c, env, [])
    got = for i <- 0..5, do: Vapor.Tensor.to_floats(out.outputs[:"s#{i}"])
    want = cols |> Enum.map(&Vapor.Tensor.to_floats/1) |> Enum.zip_with(&Enum.sort/1)
    assert Enum.zip_with(got, & &1) == want
  end

  test "synthesis: Hacker's Delight tricks rediscovered, minimal on the domain, verified beyond it; the naive formula fails" do
    {:ok, a} = D.synthesize("average")
    assert a.ops == 4 and a.verified == %{w8: true, w16: true, w32: true}
    assert a.text in ["((x & y) + ((x ^ y) >> 1))", "(((x ^ y) >> 1) + (x & y))"]
    # nothing with three operations computes it (exhaustive on 4-bit words)
    assert {:error, {:none_up_to, 3, _}} = D.synthesize("average", max_ops: 3)

    for {name, ops} <- [{"clear_lowest_one", 2}, {"lowest_one", 2}] do
      {:ok, r} = D.synthesize(name)
      assert r.ops == ops and r.verified.w8 and r.verified.w32
    end

    # the control: (x + y) >> 1 overflows — the verifier catches it
    naive = {:shr1, {:add, {:var, 0}, {:var, 1}}}
    refute D.verify(naive, D.specs()["average"], 8, :all)
  end

  test "matrix multiplication: a 7-product 2×2 algorithm found and verified exactly; rounding alone is not a proof" do
    {:ok, s} = D.matmul(2, 7, tries: 30, seed: 1)
    assert s.mults == 7 and D.bilinear_ok?(2, s)

    # it multiplies integer matrices exactly
    for k <- 1..20 do
      a = for i <- 0..1, do: for(j <- 0..1, do: rem(k * 7 + i * 3 + j * 11, 19) - 9)
      b = for i <- 0..1, do: for(j <- 0..1, do: rem(k * 5 + i * 13 + j * 2, 17) - 8)
      want = for i <- 0..1, do: for(j <- 0..1, do: Enum.at(Enum.at(a, i), 0) * Enum.at(Enum.at(b, 0), j) + Enum.at(Enum.at(a, i), 1) * Enum.at(Enum.at(b, 1), j))
      assert D.bilinear_apply(s, a, b) == want
    end

    # the control: the tries before success were rounded too — and wrong; the exact check is what decides
    assert s.tries > 1
    # and rank 6 is impossible (Winograd 1971): no candidate passes
    assert {:error, {:not_found, 10}} = D.matmul(2, 6, tries: 10, seed: 1)
  end

  test "complexity: the class of exact operation counts — Strassen recursion n^2.807, the schoolbook n^3" do
    {:ok, s} = D.matmul(2, 7, tries: 30, seed: 1)
    counts = D.recursive_counts(s, 7)
    assert D.complexity(for(%{n: n, mults: m} <- counts, n >= 2, do: {n, m})).class == "n^2.807"
    # the control: schoolbook counts
    assert D.complexity(for(k <- 1..7, n = 1 <<< k, do: {n, n * n * n})).class == "n^3"
    assert D.complexity(for(n <- [10, 20, 40, 80, 160, 320], do: {n, n * :math.log2(n) * 3 + 5})).class == "n log n"
  end
end
