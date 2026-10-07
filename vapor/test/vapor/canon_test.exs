defmodule Vapor.CanonTest do
  @moduledoc """
  Canonical functions, broadcasting, row reductions and dense linear maps
  (phase P1): one microprogram definition, bit-identical on every substrate.
  """
  use ExUnit.Case, async: false
  import Bitwise
  alias Vapor.{Canon, F32, Program}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Dispatch, Native, Substrates, Worker}
  import Vapor.TestHelpers

  # --------------------------------------------------------------- programs --

  @edge [-87.0001, -87.0, -86.9, -50.0, -1.0e-30, 0.0, 1.0e-30, 1.0e-7, 0.5, 1.0, 2.0,
         20.0, 87.9, 88.0, 88.1, 1.0e18, -1.0e18, 1.2e-38, 1.0e-45]
  @extreme [3.4e38, -3.4e38, 1.0e30, -1.0e30]

  defp programs, do: canon_programs()

  defp compiled(prog, pol) do
    {:ok, c} = Lower.lower(prog, policy: pol)
    c
  end

  # ------------------------------------------------------------ semantics --

  test "every canonical function is a microprogram over portable primitives only" do
    for {f, arity} <- Canon.functions() do
      {ops, {:t, _}, _} = Canon.expand(f, Enum.map(0..(arity - 1), &{:in, &1}), 0)
      assert Enum.all?(ops, fn {_, p, _} -> Canon.primitive?(p) end)
      # no division, square root or native min/max anywhere
      refute Enum.any?(ops, fn {_, p, _} -> p in [:div, :sqrt, :max, :min, :fma] end)
    end
  end

  test "accuracy against binary64 references (measured, published in the docs)" do
    :rand.seed(:exsss, {7, 7, 7})
    ulp = fn got, ref -> abs(key(got) - key(F32.from_float(ref))) end

    cases = [
      {:exp, fn -> -87 + :rand.uniform() * 175 end, &:math.exp/1, 2},
      {:rcp, fn -> sign() * :math.pow(2, -100 + :rand.uniform() * 200) end, &(1 / &1), 1},
      {:rsqrt, fn -> :math.pow(2, -120 + :rand.uniform() * 240) end, &(1 / :math.sqrt(&1)), 1},
      {:sigmoid, fn -> -40 + :rand.uniform() * 80 end, &(1 / (1 + :math.exp(-&1))), 3},
      {:silu, fn -> -40 + :rand.uniform() * 80 end, &(&1 / (1 + :math.exp(-&1))), 3},
      {:tanh, fn -> sign() * :math.pow(2, -20 + :rand.uniform() * 25) end, &:math.tanh/1, 6},
      {:gelu_tanh, fn -> -3 + :rand.uniform() * 6 end,
       &(&1 / (1 + :math.exp(-1.5957691216057308 * (&1 + 0.044715 * &1 * &1 * &1)))), 16},
      # the far negative tail is ill-conditioned (|z·σ'(z)/σ(z)| ≈ 80 at x = −9):
      # one rounding of z costs ~80 ulp there — PyTorch's ½x(1 + tanh z)
      # cancels to 0 instead
      {:gelu_tanh, fn -> -9.5 + :rand.uniform() * 21.5 end,
       &(&1 / (1 + :math.exp(-1.5957691216057308 * (&1 + 0.044715 * &1 * &1 * &1)))), 160}
    ]

    for {f, gen, ref, bound} <- cases do
      run = Canon.compile(f)

      worst =
        Enum.reduce(1..3000, 0, fn _, w ->
          xb = F32.from_float(gen.())
          max(w, ulp.(run.([xb]), ref.(F32.to_float(xb))))
        end)

      assert worst <= bound, "#{f}: #{worst} ulp > #{bound}"
    end
  end

  test "totality at the edges: finite, never subnormal, exact special values" do
    exp = Canon.compile(:exp)
    for x <- @edge ++ @extreme, xb = F32.from_float(x), y = exp.([xb]) do
      assert F32.finite?(y)
      refute F32.subnormal?(y)
    end

    assert exp.([F32.from_float(0.0)]) == F32.from_float(1.0)
    assert exp.([F32.from_float(-88.0)]) == 0
    assert exp.([F32.from_float(1000.0)]) == exp.([F32.from_float(88.0)])
    rcp = Canon.compile(:rcp)
    assert rcp.([F32.from_float(2.0)]) == F32.from_float(0.5)
    assert rcp.([F32.from_float(-4.0)]) == F32.from_float(-0.25)
    rsqrt = Canon.compile(:rsqrt)
    assert rsqrt.([F32.from_float(4.0)]) == F32.from_float(0.5)
    assert F32.finite?(rcp.([0]))
    mx = Canon.compile(:max)
    assert mx.([0x8000_0000, 0]) == 0x8000_0000
    assert mx.([0, 0x8000_0000]) == 0
  end

  test "broadcasting outside the four iteration classes is a Rung-1/2 rejection, not a miscompile" do
    a = T.input(:a, :f32, [3, 1, 16])
    b = T.input(:b, :f32, [1, 4, 16])
    assert {:error, %Vapor.Rejection{}} = Lower.lower(Program.new(y: T.add(a, b)))
    assert {:error, %Vapor.Rejection{}} = Program.check(Program.new(y: T.reduce(:sum, T.input(:z, :f32, [3, 20]))))
    assert {:error, %Vapor.Rejection{}} = Program.check(Program.new(y: T.add(T.input(:p, :f32, [3]), T.input(:q, :f32, [3, 1]))))
  end

  test "the ladder certifies every P1 program under :canonical and refuses :fast for canonical functions" do
    for {name, prog, _env} <- programs() do
      assert {:ok, c} = Vapor.compile(prog), name
      assert c.certificate.payload.parity.bit_identical == :all_outputs
    end

    {_, fprog, _} = hd(programs())
    assert {:error, %Vapor.Rejection{bound: "no a-priori envelope" <> _}} = Vapor.compile(fprog, policy: :fast)

    # max/min are exact selections: they keep an analytic envelope under :fast
    m = T.input(:m, :f32, [4, 32])
    assert {:ok, c} = Vapor.compile(Program.new(y: T.max(T.reduce(:max, m), T.reduce(:sum, m))), policy: :fast)
    assert is_float(c.certificate.payload.parity.envelope_max_abs_error.y)
  end

  # ------------------------------------------------------------ substrates --

  @tag :native
  test "host x86 and the RVV interpreter (poisoned, VLEN 128–512) are bit-identical to the oracle" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))

    for {name, prog, env} <- programs(), pol <- [:canonical, :fast] do
      c = compiled(prog, pol)
      {:ok, ref} = Native.run_oracle(c, env)
      for isa <- Vapor.Runtime.Substrates.host_isas() do
        {:ok, got} = Native.run(w, c, env, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, "#{name} (#{pol}) on the host (#{isa})"
      end

      for vlen <- [128, 256, 512] do
        {:ok, emu} = Native.run(w, c, env, isa: :riscv64, mode: :emulate, poison: true, vlen: vlen)
        assert emu.outputs == ref.outputs, "#{name} (#{pol}) on the RVV interpreter, VLEN #{vlen}"
      end
    end
  end

  @tag :qemu
  @tag timeout: 900_000
  test "AArch64 NEON and RVV under QEMU are bit-identical to the oracle" do
    for {target, isa} <- [{:aarch64, :aarch64}, {{:riscv64, 128}, :riscv64}, {{:riscv64, 512}, :riscv64}] do
      {:ok, w} = Worker.start_link(exec: worker_exec(target))

      for {name, prog, env} <- programs() do
        c = compiled(prog, :canonical)
        {:ok, ref} = Native.run_oracle(c, env)
        {:ok, got} = Native.run(w, c, env, isa: isa, mode: :native)
        assert got.outputs == ref.outputs, "#{name} on #{inspect(target)}"
      end
    end
  end

  @tag :vulkan
  test "the Vulkan fabric is bit-identical to the oracle" do
    fabric = Enum.find(Substrates.list(), &(&1.kind == :fabric))

    for {name, prog, env} <- programs() do
      c = compiled(prog, :canonical)
      {:ok, ref} = Native.run_oracle(c, env)
      {:ok, got} = Dispatch.run_on(fabric, c, env, [])
      assert got.outputs == ref.outputs, "#{name} on the fabric"
    end
  end

  defp key(b), do: if((b &&& 0x8000_0000) != 0, do: -(b &&& 0x7FFF_FFFF), else: b)
  defp sign, do: if(:rand.uniform() < 0.5, do: -1, else: 1)
end
