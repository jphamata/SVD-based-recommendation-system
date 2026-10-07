defmodule Vapor.CrossTest do
  @moduledoc """
  The same machine code on real (emulated) silicon semantics: the worker is
  cross-compiled for AArch64 and RISC-V and runs under QEMU user mode, so
  NEON and RVV 1.0 instructions execute on QEMU's implementations — an
  oracle independent of both vapor's encoders and its own RVV interpreter.
  """
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Native, Worker}
  import Vapor.TestHelpers

  @moduletag :qemu
  @moduletag timeout: 600_000

  defp programs do
    [
      {ew_chain(12), %{x: Tensor.random(:f32, [1003], 1), y: Tensor.random(:f32, [1003], 2)}, []},
      {gemm_program(), %{a: Tensor.random(:s8, [9, 203], 3)}, []},
      {ssm_block(), ssm_env(4), [iterations: 4, sequence: [:x]]}
    ]
  end

  for {name, target, isa} <- [{"AArch64 NEON", :aarch64, :aarch64},
                               {"RISC-V RVV VLEN=128", {:riscv64, 128}, :riscv64},
                               {"RISC-V RVV VLEN=256", {:riscv64, 256}, :riscv64},
                               {"RISC-V RVV VLEN=512", {:riscv64, 512}, :riscv64}] do
    test "#{name} under QEMU is bit-identical to the oracle and to the in-tree interpreter" do
      {:ok, w} = Worker.start_link(exec: worker_exec(unquote(Macro.escape(target))))
      {:ok, host} = Worker.start_link(exec: worker_exec(:host))

      for {prog, env, opts} <- programs(), pol <- [:canonical, :fast] do
        {:ok, c} = Lower.lower(prog, policy: pol)
        {:ok, ref} = Native.run_oracle(c, env, opts)
        {:ok, got} = Native.run(w, c, env, [isa: unquote(isa), mode: :native] ++ opts)
        assert got.outputs == ref.outputs

        if unquote(isa) == :riscv64 do
          {:ok, emu} = Native.run(host, c, env, [isa: :riscv64, mode: :emulate, poison: true] ++ opts)
          assert emu.outputs == got.outputs, "interpreter vs QEMU"
        end
      end
    end
  end
end
