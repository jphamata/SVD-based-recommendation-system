defmodule Vapor.EnvelopeTest do
  use ExUnit.Case, async: true
  alias Vapor.{F32, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.Native
  alias Vapor.Verify.{Dyadic, Envelope}
  import Vapor.TestHelpers

  test "γₙ is an upward dyadic bound of n·u/(1 − n·u)" do
    for n <- [1, 7, 295, 100_000] do
      g = Dyadic.gamma(n)
      exact = n / (16_777_216 - n)
      assert Dyadic.to_float(g) >= exact and Dyadic.to_float(g) - exact < 1.0e-15
    end
  end

  test "oracle outputs of both policies lie inside the envelope; a displaced value does not" do
    for prog <- [ew_chain(8), ssm_block(256, 256)], pol <- [:canonical, :fast] do
      {:ok, c} = Lower.lower(prog, policy: pol)

      {env, opts} =
        if c.program.state == [],
          do: {%{x: Tensor.random(:f32, [37], 1), y: Tensor.random(:f32, [37], 2)}, []},
          else: {ssm_env(3, 256, 256), [iterations: 3, sequence: [:x]]}

      {:ok, run} = Native.run_oracle(c, env, opts)
      bounds = Envelope.bounds(c, env, opts)
      steps = if run.steps == [], do: [run.outputs], else: run.steps

      for {outs, bs} <- Enum.zip(steps, bounds), {name, t} <- outs do
        assert :ok = Envelope.check(bs[name], t)
        # push element 0 just past twice its bound: must be caught
        [b0 | _] = bs[name]
        [x0 | rest] = Tensor.to_list(t)
        e = Dyadic.to_float(Envelope.max_error([b0]))
        moved = F32.from_float(F32.to_float(x0) + 2 * e + 1.0e-3)
        bad = %{t | data: F32.encode([moved | rest])}
        assert {:error, %{index: 0}} = Envelope.check(bs[name], bad)
      end
    end
  end

  test "integer contractions are certified exactly (zero envelope)" do
    {:ok, c} = Lower.lower(gemm_program())
    env = %{a: Tensor.random(:s8, [5, 203], 3)}
    {:ok, run} = Native.run_oracle(c, env)
    [bs] = Envelope.bounds(c, env)
    assert :ok = Envelope.check(bs[:c], run.outputs[:c])
    assert Dyadic.to_float(Envelope.max_error(bs[:c])) == 0.0
  end

  test "the Wilkinson decision is the Lean-extracted one" do
    x = T.input(:x, :f32, [512])
    w = T.const(Vapor.Quant.Sb4.quantize(Tensor.random(:f32, [4, 512], 9)))
    {:ok, c} = Lower.lower(Program.new(y: T.qgemv(w, x)))
    [bs] = Envelope.bounds(c, %{x: Tensor.random(:f32, [512], 4)})
    assert [{:wilkinson, _, _, _, n} | _] = bs[:y]
    assert n == 512 + 16 + 31
  end
end
