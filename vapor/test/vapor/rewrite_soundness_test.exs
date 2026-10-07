defmodule Vapor.RewriteSoundnessTest do
  @moduledoc """
  The rewriter's identities (`Vapor.Compile.Rewrite`) are proved in Lean
  (`proofs/Vapor/Binary32.lean`); this file ties the proof to the code:

    * the Lean model's reading of a bit pattern (`units`, extracted) is
      `Vapor.F32`'s, on every exponent and on random patterns;
    * the rewriter applies exactly the proved rules and refuses the refuted
      one (`x + (+0)`), and the rules hold bit for bit on the oracle and on
      the native worker for ±0, subnormals and the extremes;
    * the scope is what the docs say: NaN is refused by the oracle (0.7's
      "including NaN" was false — a signalling NaN times one is quieted on
      x86, so its bits change).
  """
  use ExUnit.Case, async: true
  alias Vapor.{Extracted, F32, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Rewrite
  alias Vapor.Runtime.{Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @edge [0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x0080_0000, 0x7F7F_FFFF, 0xFF7F_FFFF,
         0x3F80_0000, 0xBF80_0000, 0x3EAA_AAAB, 0x4049_0FDB]

  defp exact(b), do: trunc(F32.to_float(b) * :math.pow(2, 149))

  test "the Lean model reads every bit pattern as Vapor.F32 does (every exponent, both signs, random significands)" do
    :rand.seed(:exsss, {1, 2, 3})

    for e <- 0..254, s <- [0, 1], _ <- 1..3 do
      b = s * 0x8000_0000 + e * 0x80_0000 + :rand.uniform(0x80_0000) - 1
      assert Extracted.units(b) == exact(b), "0x#{Integer.to_string(b, 16)}"
    end

    for b <- @edge, do: assert(Extracted.units(b) == exact(b))
  end

  test "the rewriter applies the proved rules, refuses the refuted one, and the results keep every bit" do
    x = T.input(:x, :f32, [1, 16])
    rules = [mul_one: T.mul(x, T.splat(1.0)), one_mul: T.mul(T.splat(1.0), x), add_negzero: T.add(x, {:splat, 0x8000_0000}),
             sub_poszero: T.sub(x, T.splat(0.0)), negneg: T.neg(T.neg(x))]

    for {name, t} <- rules do
      {p, n} = Rewrite.rewrite(Program.new(y: t))
      assert n >= 1 and p.outputs[:y] == x, "#{name} not applied"
    end

    # refuted: x + (+0) is left alone, because (−0) + (+0) = +0
    {_, 0} = Rewrite.rewrite(Program.new(y: T.add(x, T.splat(0.0))))
    {_, 0} = Rewrite.rewrite(Program.new(y: T.sub(x, {:splat, 0x8000_0000})))
    neg0 = Tensor.new(:f32, [1, 16], :binary.copy(<<0x8000_0000::32-little>>, 16))
    assert Oracle.eval_program(Program.new(y: T.add(x, T.splat(0.0))), %{x: neg0}).y.data == :binary.copy(<<0::32>>, 16)

    # every rule on the edge patterns (±0, subnormals, extremes): the unrewritten program gives x's bits back,
    # on the oracle and, when present, on the native worker
    xs = Tensor.new(:f32, [1, 16], for(b <- @edge ++ [0x0000_0002, 0x8000_0010, 0x0040_0000, 0x7F00_0000], into: <<>>, do: <<b::32-little>>))
    w = if Substrates.binary("vapor-worker", "native"), do: elem(Worker.start_link(exec: worker_exec(:host)), 1)

    for {name, t} <- rules do
      p = Program.new(y: t)
      assert Oracle.eval_program(p, %{x: xs}).y.data == xs.data, "#{name} (oracle)"

      if w do
        {:ok, c} = Vapor.Compile.Lower.lower(p)
        {:ok, r} = Native.run(w, c, %{x: xs}, isa: Substrates.host_isa(), mode: :native)
        assert r.outputs.y.data == xs.data, "#{name} (native)"
      end
    end
  end

  test "NaN is outside the claim: the oracle refuses it" do
    for nan <- [0x7FC0_0000, 0x7FA0_0000, 0xFFC0_0001] do
      assert_raise ArithmeticError, fn -> F32.mul(nan, 0x3F80_0000) end
    end

    refute File.read!("lib/vapor/compile/rewrite.ex") =~ "including ±0 and NaN"
  end
end
