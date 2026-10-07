defmodule VaporNxTest do
  @moduledoc """
  `defn` compiled by `Vapor.Nx.Compiler`: arithmetic gives the same bits as
  Nx's reference evaluator; transcendental functions stay within their
  certified bounds; every substrate of the machine gives the same bits;
  operations outside the certified fragment are refused by name.
  """
  use ExUnit.Case, async: false
  import Nx.Defn

  @moduletag timeout: 600_000

  defn arith(a, b, c, d), do: (a * b + c) * d - a
  defn quotient(a, b), do: a / b
  defn layer(x, w, b), do: Nx.sigmoid(Nx.dot(x, [1], w, [1]) + b)
  defn softmaxish(x), do: Nx.exp(x - Nx.reduce_max(x, axes: [1], keep_axes: true))
  defn gate(x, y), do: Nx.select(Nx.less(x, y), Nx.tanh(x), y * 0.5)
  defn row_sums(x, w), do: Nx.sum(Nx.dot(x, w) * 2.0, axes: [1])
  defn trig(x), do: Nx.cos(x)

  defp rand(shape, key, scale \\ 1.0) do
    {t, _} = Nx.Random.normal(Nx.Random.key(key), 0.0, scale, shape: shape, type: :f32)
    Nx.backend_transfer(t, Nx.BinaryBackend)
  end

  defp vapor(fun), do: Nx.Defn.jit(fun, compiler: Vapor.Nx.Compiler)
  defp reference(fun), do: Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator)

  defp bits(t), do: Nx.to_binary(t)

  # largest distance in units in the last place (f32, same sign region)
  defp max_ulp(a, b) do
    ord = fn <<x::signed-32-little>> -> if x < 0, do: -0x80000000 - x, else: x end
    Enum.zip(for(<<x::binary-4 <- bits(a)>>, do: ord.(x)), for(<<y::binary-4 <- bits(b)>>, do: ord.(y)))
    |> Enum.map(fn {x, y} -> abs(x - y) end)
    |> Enum.max()
  end

  test "tensors convert both ways, byte for byte" do
    t = rand({3, 16}, 1)
    v = Vapor.Nx.from_nx(t)
    assert v.shape == [3, 16] and v.dtype == :f32
    assert bits(Vapor.Nx.to_nx(v)) == bits(t) and Vapor.Nx.to_nx(v).shape == {3, 16}
  end

  test "+ − × with broadcasting: the same bits as Nx's evaluator; ÷ within 1 ulp" do
    a = rand({8, 32}, 2)
    b = rand({32}, 3)
    c = rand({8, 1}, 4)
    d = Nx.add(Nx.abs(rand({8, 32}, 5)), 0.5)
    got = vapor(&arith/4).(a, b, c, d)
    assert got.shape == {8, 32}
    assert bits(got) == bits(reference(&arith/4).(a, b, c, d))

    # vapor's canonical division is a microprogram over + − × (the same bits
    # everywhere, not IEEE's correctly rounded quotient): at most 1 ulp away
    q = vapor(&quotient/2).(a, d)
    assert max_ulp(q, reference(&quotient/2).(a, d)) <= 1
  end

  test "dot, sigmoid, exp, reductions, select: within certified bounds of the evaluator" do
    x = rand({4, 32}, 6)
    w = rand({16, 32}, 7, 0.2)
    b = rand({16}, 8)
    assert max_ulp(vapor(&layer/3).(x, w, b), reference(&layer/3).(x, w, b)) <= 64

    s = rand({4, 32}, 9, 3.0)
    assert max_ulp(vapor(&softmaxish/1).(s), reference(&softmaxish/1).(s)) <= 8

    y = rand({4, 32}, 10)
    assert max_ulp(vapor(&gate/2).(s, y), reference(&gate/2).(s, y)) <= 8

    # dot against a [k, n] operand (transposed), a sum that drops its axis
    wt = rand({32, 16}, 11, 0.2)
    got = vapor(&row_sums/2).(x, wt)
    want = reference(&row_sums/2).(x, wt)
    assert got.shape == {4}
    assert Nx.all_close(got, want, rtol: 1.0e-5, atol: 1.0e-5) |> Nx.to_number() == 1
  end

  test "certified: every substrate of this machine computes the same bits" do
    x = rand({4, 32}, 12)
    w = rand({16, 32}, 13, 0.2)
    b = rand({16}, 14)
    {:ok, compiled} = Vapor.Nx.certify(&layer/3, [x, w, b])
    parity = compiled.certificate.payload.parity
    assert parity.bit_identical == :all_outputs and length(parity.substrates) >= 2

    runs =
      for sub <- Vapor.Runtime.Substrates.list() do
        {:ok, r, _} = Vapor.run(compiled, %{:"p0@4x32" => Vapor.Nx.from_nx(x), :"p1@16x32" => Vapor.Nx.from_nx(w), :"p2@1x16" => %{Vapor.Nx.from_nx(b) | shape: [1, 16]}},
                                substrates: [sub])
        {sub.id, r.outputs.out0.data}
      end

    assert runs |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> length() == 1
    IO.puts("\n  same bits on #{inspect(Enum.map(runs, &elem(&1, 0)))}")
  end

  test "outside the fragment: refused, by name" do
    assert_raise ArgumentError, ~r/operation :cos/, fn -> vapor(&trig/1).(rand({16}, 15)) end
    assert_raise ArgumentError, ~r/type \{:s, 32\}/, fn -> vapor(&arith/4).(Nx.iota({16}), Nx.iota({16}), Nx.iota({16}), Nx.iota({16})) end
  end
end
