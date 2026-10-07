defmodule Vapor.MetalTest do
  @moduledoc """
  Metal without a Mac: the kernel library translated to MSL
  (`Vapor.Emit.MSL`) and executed through `vapor-metal-sim` — the Metal
  daemon's own protocol, plans and sessions, with the device replaced by
  shared objects clang built from the exact MSL text — must give the
  oracle's bits; the daemon itself cross-compiles for macOS.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Emit.MSL
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Fabric, Native, Session, Substrates, Worker}
  alias Vapor.Verify.Envelope
  import Vapor.TestHelpers

  @moduletag :msl_shim
  @moduletag :native

  setup_all do
    {:ok, f} = Vapor.MSLShim.start(:conforming, args: ["--fault-injection"])
    {:ok, metal: f}
  end

  defp run(f, prog, env, opts \\ [], policy \\ :canonical) do
    {:ok, c} = Lower.lower(prog, policy: policy)
    Vapor.MSLShim.ensure(c, :conforming)
    {:ok, want} = Native.run_oracle(c, env, opts)
    {:ok, got} = Fabric.run(f, c, env, opts)
    {c, want, got}
  end

  test "every kernel of the GPU library has an MSL form (the cooperative-matrix variant excepted)" do
    keys = [{:reduce, :sum}, {:reduce, :max}, :gemv_f32, :gemv_bf16, {:gemv_masked, :f32}, {:gemv_masked, :bf16},
            {:gemv_grouped, :f32}, {:gemv_grouped, :bf16}, :gemv_sb4, :gemv_sb4_masked, :sb_sums, :gemm_i8, :gather_row,
            :gather_row_bf16, :rope, :sample, :transpose, {:kv_write, :copy}, {:kv_write, :inplace}]

    for k <- keys, pol <- [:canonical, :fast] do
      %{src: src, nbind: nb} = MSL.compile(k, pol)
      assert src =~ "kernel void vapor_"
      assert length(Regex.scan(~r/\[\[buffer\(\d+\)\]\]/, src)) == nb + if(src =~ "constant uint* pc", do: 1, else: 0)
      # no goto in MSL: control flow is rebuilt structurally
      refute src =~ "goto"
      if pol == :fast and k in [:gemv_f32, {:reduce, :sum}] == false, do: :ok
    end

    assert MSL.compile(:gemm_i8_coop, :canonical) == nil
  end

  test "the canonical programs on the Metal daemon = the oracle, bit for bit", %{metal: f} do
    for {name, prog, env} <- canon_programs() do
      {_c, want, got} = run(f, prog, env)
      assert got.outputs == want.outputs, name
    end
  end

  test "windows, emits and state feedback across iterations; int8 GEMM; the attention block", %{metal: f} do
    {_c, want, got} = run(f, ssm_block(), ssm_env(4), iterations: 4, sequence: [:x])
    assert got.steps == want.steps and length(got.steps) == 4

    {_c, want, got} = run(f, gemm_program(), %{a: Tensor.random(:s8, [9, 203], 3)})
    assert got.outputs == want.outputs

    {_c, want, got} = run(f, attention_block(t: 6), attention_env(6))
    assert got.outputs == want.outputs
  end

  test "the fast policy (fused multiply-add) stays inside the rigorous envelope", %{metal: f} do
    for {prog, env, opts} <- [{ew_chain(12), %{x: Tensor.random(:f32, [1003], 1), y: Tensor.random(:f32, [1003], 2)}, []},
                              {ssm_block(), ssm_env(3), [iterations: 3, sequence: [:x]]}] do
      {c, _want, got} = run(f, prog, env, opts, :fast)
      steps = if got.steps == [], do: [got.outputs], else: got.steps
      for {outs, bs} <- Enum.zip(steps, Envelope.bounds(c, env, opts)), {name, t} <- outs, do: assert(:ok = Envelope.check(bs[name], t))
    end
  end

  test "a Metal session = the CPU session; the engine serves on Metal with the CPU's tokens", %{metal: f} do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, cfg} = Config.from_map(tiny_config("qwen2"))
    {:ok, p} = Llama.program(cfg, tiny_weights(cfg), max_seq: 16)
    {:ok, comp} = Lower.lower(p)
    Vapor.MSLShim.ensure(comp, :conforming)
    ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end
    drive = fn s ->
      {:ok, %{logits: a}, _} = Session.step(s, %{tok: ids.([3, 1, 4]), pos: ids.([0, 1, 2])}, [:logits])
      {:ok, %{logits: b}, _} = Session.step(s, %{tok: ids.([1]), pos: ids.([3])}, [:logits])
      {a, b}
    end

    {:ok, cpu} = Session.open(w, comp, isa: Substrates.host_isa())
    {:ok, gpu} = Session.open(f, comp, isa: :msl)
    assert %{kind: :fabric, staged: false, device_local: true} = Session.info(gpu)
    assert drive.(gpu) == drive.(cpu)
    :ok = Session.close(gpu)

    {:ok, c2} = Config.from_map(tiny_config("llama", %{"vocab_size" => 128}))
    base = [config: c2, weights: tiny_weights(c2, 3), max_seq: 32, page: 8, sequences: 2, step_tokens: 16]
    {:ok, prep} = Vapor.Engine.prepare(base)
    Vapor.MSLShim.ensure(prep.comp, :conforming)
    serve = fn e ->
      for {pr, o} <- [{[5, 9, 2, 7], [max_tokens: 6, temperature: 0.0]}, {[3, 8], [max_tokens: 5, temperature: 0.8, seed: 2]}] do
        {:ok, ref} = Vapor.Engine.generate(e, pr, o)
        Vapor.Engine.collect(ref, [], 120_000)
      end
    end

    {:ok, cpu_e} = Vapor.Engine.start_link(base)
    {:ok, gpu_e} = Vapor.Engine.start_link(base ++ [isa: :msl, fabric: f])
    assert serve.(gpu_e) == serve.(cpu_e)
    GenServer.stop(gpu_e)
    GenServer.stop(cpu_e)
  end

  test "a crash of the Metal daemon is contained and respawned", %{metal: f} do
    before = Fabric.info(f).restarts
    assert {:error, {:fabric_crashed, {:signal, 11}}} = Fabric.inject_fault(f)
    assert Fabric.info(f).restarts == before + 1
    {_c, want, got} = run(f, ew_chain(4), %{x: Tensor.random(:f32, [100], 1), y: Tensor.random(:f32, [100], 2)})
    assert got.outputs == want.outputs
  end

  test "a module the device never compiled is a typed error, not a crash" do
    {:ok, f} = Fabric.start_link(exec: [Substrates.binary("vapor-metal-sim", "native"), "--cache", System.tmp_dir!()])
    prog = Program.new(y: T.mul(T.input(:x, :f32, [16]), T.splat(3.0)))
    {:ok, c} = Lower.lower(prog)
    assert {:error, {:unit_fault, %{message: msg}}} = Fabric.run(f, c, %{x: Tensor.random(:f32, [16], 1)})
    assert msg =~ "ModuleNotCompiled" and msg =~ "module not built for this device"
  end

  @tag :zig
  @tag timeout: 600_000
  test "the Metal daemon and the CPU worker cross-compile for Apple Silicon and Intel Macs, linking only libSystem" do
    native = Path.expand("../../native", __DIR__)

    for {target, cpu} <- [{"aarch64-macos", 0x0100000C}, {"x86_64-macos", 0x01000007}] do
      out = Path.join(System.tmp_dir!(), "vapor-#{target}")
      {log, rc} = System.cmd("zig", ["build", "-Dtarget=#{target}", "-Doptimize=ReleaseSafe", "--prefix", out], cd: native, stderr_to_stdout: true)
      assert rc == 0, log
      for exe <- ["vapor-metal", "vapor-worker"] do
        bin = File.read!(Path.join([out, "bin", exe]))
        assert <<0xFEEDFACF::32-little, ^cpu::32-little, _sub::32, 2::32-little, ncmds::32-little, _::32, _::32, _::32, rest::binary>> = bin
        dylibs = load_dylibs(rest, ncmds)
        assert dylibs == ["/usr/lib/libSystem.B.dylib"], "#{exe}: #{inspect(dylibs)}"
      end

      # generated code on Apple Silicon: MAP_JIT pages toggled per thread, I-cache invalidated
      worker = File.read!(Path.join([out, "bin", "vapor-worker"]))
      if target == "aarch64-macos" do
        assert worker =~ "_pthread_jit_write_protect_np" and worker =~ "_sys_icache_invalidate"
      end
    end
  end

  # LC_LOAD_DYLIB (0xC) names, from a 64-bit Mach-O's load commands
  defp load_dylibs(cmds, n) do
    {names, _} =
      Enum.reduce(1..n, {[], cmds}, fn _, {acc, <<cmd::32-little, size::32-little, _::binary>> = b} ->
        <<lc::binary-size(size), rest::binary>> = b
        acc = if cmd == 0xC, do: (<<_::64, off::32-little, _::binary>> = lc; [lc |> binary_part(off, size - off) |> String.trim_trailing(<<0>>) |> String.split(<<0>>) |> hd() | acc]), else: acc
        {acc, rest}
      end)

    Enum.reverse(names)
  end
end
