defmodule Vapor.RewriteTest do
  use ExUnit.Case, async: true
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Rewrite
  alias Vapor.Runtime.Oracle

  defp rw(t), do: Program.new(o: t) |> Rewrite.rewrite() |> then(fn {p, n} -> {p.outputs[:o], n} end)

  setup do
    {:ok, x: T.input(:x, :f32, [4])}
  end

  test "exact identities fire", %{x: x} do
    assert {^x, 1} = rw(T.neg(T.neg(x)))
    assert {^x, 1} = rw(T.mul(x, T.splat(1.0)))
    assert {^x, 1} = rw(T.ew(:add, [x, {:splat, 0x8000_0000}]))
    assert {^x, 1} = rw(T.sub(x, T.splat(0.0)))
  end

  test "x + (+0.0) is NOT an identity (−0 + +0 = +0) and is not rewritten", %{x: x} do
    t = T.add(x, T.splat(0.0))
    assert {^t, 0} = rw(t)
    negzero = Tensor.from_list(:f32, [1], [{:bits, 0x8000_0000}])
    [r] = Oracle.eval(T.add(T.input(:z, :f32, [1]), T.splat(0.0)), %{z: negzero}) |> Tensor.to_list()
    assert r == 0, "so the rule would have changed −0 into +0"
  end

  test "constant folding evaluates the declared semantics under the policy" do
    c = T.const(Tensor.from_list(:f32, [2], [1.5, -2.0]))
    {{:const, t}, _} = rw(T.fma(c, c, T.splat(0.25)))
    assert Tensor.to_floats(t) == [2.5, 4.25]
  end
end
