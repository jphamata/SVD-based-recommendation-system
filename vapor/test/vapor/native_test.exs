defmodule Vapor.NativeTest do
  @moduledoc "Substrate I: host execution, the RVV interpreter, and fault containment."
  use ExUnit.Case, async: false
  alias Vapor.{Tensor}
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Native, Plan, Worker}
  import Vapor.TestHelpers

  @moduletag :native

  setup_all do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, worker: w}
  end

  defp programs do
    [
      {ew_chain(12), %{x: Tensor.random(:f32, [1003], 1), y: Tensor.random(:f32, [1003], 2)}, []},
      {gemm_program(), %{a: Tensor.random(:s8, [9, 203], 3)}, []},
      {ssm_block(), ssm_env(5), [iterations: 5, sequence: [:x]]}
    ]
  end

  test "the worker is sandboxed and reports the host ISA", %{worker: w} do
    info = Worker.info(w)
    assert info.arch == Vapor.Runtime.Substrates.host_isa()
    assert info.sandbox in [:seccomp, :unsupported]
  end

  test "host machine code (every host ISA) is bit-identical to the oracle (both policies)", %{worker: w} do
    for isa <- Vapor.Runtime.Substrates.host_isas(), {prog, env, opts} <- programs(), pol <- [:canonical, :fast] do
      {:ok, c} = Lower.lower(prog, policy: pol)
      {:ok, ref} = Native.run_oracle(c, env, opts)
      {:ok, got} = Native.run(w, c, env, [isa: isa, mode: :native] ++ opts)
      assert got.outputs == ref.outputs, inspect(isa)
      assert got.steps == if(opts == [], do: [], else: ref.steps)
    end
  end

  test "RVV code runs bit-exactly in the in-tree interpreter at VLEN 128/256/512 with tail poisoning",
       %{worker: w} do
    for {prog, env, opts} <- programs() do
      {:ok, c} = Lower.lower(prog)
      {:ok, ref} = Native.run_oracle(c, env, opts)

      for vlen <- [128, 256, 512] do
        {:ok, got} = Native.run(w, c, env, [isa: :riscv64, mode: :emulate, vlen: vlen, poison: true] ++ opts)
        assert got.outputs == ref.outputs, "VLEN=#{vlen}"
        assert got.retired > 0
      end
    end
  end

  defp raw(code, opts \\ []) do
    %Plan{code: code, buffers: [%{kind: :zero, len: 64, writable: true}],
          calls: [%{entry: 0, args: [{:buf, 0, 0}]}], returns: [0]}
    |> then(&{&1, opts})
  end

  test "emulator faults are values, not signals: the worker survives", %{worker: w} do
    before = Worker.info(w).restarts
    # an all-zero word is reserved-illegal in RISC-V
    {plan, _} = raw(<<0::32>>)
    assert {:error, {:unit_fault, %{code: :illegal_instruction, pc: 0}}} = Worker.run(w, plan, mode: :emulate)
    # ld a0, 0(zero): address 0 is outside every bound buffer
    {plan, _} = raw(<<0x00003503::32-little>>)
    assert {:error, {:unit_fault, %{code: :memory_fault}}} = Worker.run(w, plan, mode: :emulate)
    # jal x0, 0: spins until the fuel budget is exhausted
    {plan, _} = raw(<<0x0000006F::32-little>>)
    assert {:error, {:unit_fault, %{code: :fuel_exhausted}}} = Worker.run(w, plan, mode: :emulate, fuel: 10_000)
    assert Worker.info(w).restarts == before
  end

  @tag :x86_64_host
  test "native faults kill only the worker; OTP respawns it and the next unit runs", %{worker: w} do
    if Vapor.Runtime.Substrates.host_isa() != :x86_64, do: :ok, else: do_native_faults(w)
  end

  defp do_native_faults(w) do
    cases = [
      {<<0x0F, 0x0B>>, :sigill, []},
      # mov dword [0x10], 0
      {<<0xC7, 0x04, 0x25, 0x10, 0, 0, 0, 0, 0, 0, 0>>, :sigsegv, []},
      # jmp $ — only the in-worker watchdog can stop it
      {<<0xEB, 0xFE>>, :sigalrm, [timeout: 300]},
      # mov eax, 59 (execve); syscall — refused by the seccomp allowlist
      {<<0xB8, 59, 0, 0, 0, 0x0F, 0x05>>, if(Worker.info(w).sandbox == :seccomp, do: :sigsys, else: :any), []}
    ]

    for {code, sig, opts} <- cases do
      before = Worker.info(w).restarts
      {plan, _} = raw(code)

      case Worker.run(w, plan, [mode: :native] ++ opts) do
        {:error, {:worker_crashed, {:signal, got}}} -> if sig != :any, do: assert(got == sig)
        other -> flunk("expected a contained #{sig}, got #{inspect(other)}")
      end

      assert Worker.info(w).restarts == before + 1
      # the respawned worker is healthy: a `ret` runs fine
      {plan, _} = raw(<<0xC3>>)
      assert {:ok, _} = Worker.run(w, plan, mode: :native)
    end
  end
end
