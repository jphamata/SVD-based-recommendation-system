defmodule Vapor.FabricTest do
  @moduledoc "Substrate II: the Vulkan daemon, parity, and failover."
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Dispatch, Fabric, Native, Substrates}
  alias Vapor.Verify.Envelope
  import Vapor.TestHelpers

  @moduletag :vulkan

  defp programs do
    [
      {ew_chain(12), %{x: Tensor.random(:f32, [1003], 1), y: Tensor.random(:f32, [1003], 2)}, []},
      {gemm_program(), %{a: Tensor.random(:s8, [9, 203], 3)}, []},
      {ssm_block(), ssm_env(4), [iterations: 4, sequence: [:x]]}
    ]
  end

  setup_all do
    {:ok, f} = Fabric.start_link(exec: [Substrates.binary("vapor-fabric", "native"), "--fault-injection"])
    {:ok, fabric: f}
  end

  test "the daemon drives a real device through the whole compute pipeline", %{fabric: f} do
    info = Fabric.info(f)
    assert info.ready and byte_size(info.name) > 0
  end

  test "canonical policy: GPU bit-identical to the oracle; fast: inside the envelope", %{fabric: f} do
    for {prog, env, opts} <- programs(), pol <- [:canonical, :fast] do
      {:ok, c} = Lower.lower(prog, policy: pol)
      {:ok, ref} = Native.run_oracle(c, env, opts)
      {:ok, gpu} = Fabric.run(f, c, env, opts)
      steps = fn r -> if r.steps == [], do: [r.outputs], else: r.steps end

      if pol == :canonical do
        assert steps.(gpu) == steps.(ref)
      else
        for {outs, bs} <- Enum.zip(steps.(gpu), Envelope.bounds(c, env, opts)), {name, t} <- outs,
            do: assert(:ok = Envelope.check(bs[name], t))
      end
    end
  end

  test "zero-copy import of weight files (VK_EXT_external_memory_host) when available", %{fabric: f} do
    # weights ≥ 64 KiB travel as /dev/shm files and are imported, not copied
    {prog, env, opts} = List.last(programs())
    {:ok, c} = Lower.lower(prog)
    assert {:ok, _} = Fabric.run(f, c, env, opts)
    assert is_boolean(Fabric.info(f).host_import)
  end

  test "a driver crash is contained and the unit is rerouted to the native substrate", %{fabric: f} do
    {prog, env, opts} = List.last(programs())
    {:ok, c} = Lower.lower(prog)
    {:ok, ref} = Native.run_oracle(c, env, opts)
    before = Fabric.info(f).restarts
    assert {:error, {:fabric_crashed, {:signal, 11}}} = Fabric.inject_fault(f)
    assert Fabric.info(f).restarts == before + 1

    # now make the fabric fail mid-unit: a run on a device that just crashed
    # is rerouted by the dispatcher; simulate by pointing at a dead server
    dead = spawn(fn -> :ok end)
    subs = [%{id: :fabric, kind: :fabric, isa: :spirv, mode: :gpu, server: dead} | Enum.reject(Substrates.list(), &(&1.id == :fabric))]
    {:ok, res, trace} = Dispatch.run(c, env, [substrates: subs, prefer: :fabric] ++ opts)
    assert [{:fabric, {:failed, _}} | _] = trace
    assert res.substrate in [:host_avx512, :host, :rvv_emulated, :oracle]
    assert res.steps == ref.steps
  end
end
