defmodule Vapor.StableHLOTest do
  @moduledoc """
  The portable route to accelerators vapor does not drive itself
  (Tenstorrent through tt-xla, TPUs, GPUs): programs exported as StableHLO
  must compute what vapor computes, the untranslatable must be refused by
  name, and the admission kit must measure the device that runs it — here
  XLA's CPU through PJRT, whose arithmetic turns out not to be canonical.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Substrate, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Export.StableHLO
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.Native
  alias Vapor.Verify.Envelope
  import Vapor.TestHelpers

  defp run_xla(prog, env) do
    {:ok, ex} = StableHLO.export(prog, env: env)
    d = Path.join(System.tmp_dir!(), "vapor-shlo-#{System.unique_integer([:positive])}")
    File.mkdir_p!(d)
    File.write!(Path.join(d, "module.mlir"), ex.mlir)
    ins = for {n, dt, s} <- ex.inputs, do: (File.write!(Path.join(d, "in_#{n}.bin"), env[n].data); %{name: n, dtype: dt, shape: s, file: "in_#{n}.bin"})
    File.write!(Path.join(d, "manifest.json"), Vapor.JSON.encode(%{inputs: ins, outputs: Enum.map(ex.outputs, fn {n, dt, s} -> %{name: n, dtype: dt, shape: s} end)}))
    {out, rc} = System.cmd(python(), [Path.expand("../python/stablehlo_run.py", __DIR__), d], stderr_to_stdout: true)
    assert rc == 0, out
    Map.new(ex.outputs, fn {n, dt, s} -> {n, Tensor.new(dt, s, File.read!(Path.join(d, "out_#{n}.bin")))} end)
  end

  defp elems(%{elems: e}), do: e
  defp elems(l), do: l

  defp oracle(prog, env) do
    {:ok, c} = Lower.lower(prog)
    {:ok, r} = Native.run_oracle(c, env)
    {c, r.outputs}
  end

  test "untranslatable operators are refused by name, with the repair" do
    x = T.input(:x, :f32, [2, 16])
    {:error, r} = StableHLO.export(Program.new(t: T.sample(x, T.input(:p, :f32, [2, 2]))))
    assert r.bound =~ "sample" and r.repair =~ "decode loop"
    {:error, r} = StableHLO.export(Program.new(y: T.qgemv(T.const(Vapor.Quant.Sb4.quantize(Tensor.random(:f32, [16, 256], 1))), T.input(:v, :f32, [256]))))
    assert r.bound =~ "qgemv" and r.repair =~ "dequantized"
    {:error, r} = StableHLO.export(Program.new(y: T.gelu(x)))
    assert r.bound =~ "erf" and r.repair =~ "gelu_tanh"
  end

  test "the module is plain StableHLO: one public main, typed arguments, exact constants" do
    w = Tensor.random(:f32, [3, 16], 1)
    {:ok, ex} = StableHLO.export(Program.new(y: T.linear(T.input(:x, :f32, [2, 16]), T.const(w))))
    assert ex.inputs == [{:x, :f32, [2, 16]}] and ex.outputs == [{:y, :f32, [2, 3]}]
    assert ex.mlir =~ "func.func public @main(%arg0: tensor<2x16xf32>"
    # constants travel as their bytes (hex), never as rounded decimals
    assert ex.mlir =~ Base.encode16(w.data)
    refute ex.mlir =~ "custom_call"
  end

  @tag :jax
  test "on XLA (PJRT): exact operators give vapor's bits, contractions stay inside the rigorous envelope" do
    for {name, prog, env} <- canon_programs(), match?({:ok, _}, StableHLO.export(prog, env: env)) do
      got = run_xla(prog, env)
      {c, want} = oracle(prog, env)
      [bounds] = Envelope.bounds(c, env)

      for {n, t} <- got do
        cond do
          t.data == want[n].data -> :ok
          Envelope.analytic?(elems(bounds[n])) -> assert :ok == Envelope.check(bounds[n], t), "#{name}/#{n}"
          true -> :not_analytic
        end
      end
    end

    # selections, gathers and relabellings are exact everywhere
    m = Tensor.random(:f32, [7, 48], 3, scale: 3.0)
    idx = Tensor.from_list(:s32, [5], [0, 8, 3, -1, 6])
    tab = Tensor.random(:f32, [9, 32], 4)
    prog = Program.new(r: T.relu(T.input(:m, :f32, [7, 48])), s: T.sel(T.input(:m, :f32, [7, 48]), T.splat(0.5), T.input(:m, :f32, [7, 48]), T.splat(-1.0)),
                       g: T.gather_row(T.const(tab), T.input(:i, :s32, [5])), t: T.transpose(T.input(:m, :f32, [7, 48])))
    env = %{m: m, i: idx}
    assert run_xla(prog, env) == elem(oracle(prog, env), 1)
  end

  @tag :jax
  test "a whole decoder step (gather, RMSNorm, RoPE, KV write, GQA attention, SwiGLU) runs on XLA with vapor's answer" do
    for arch <- ["llama", "qwen2"] do
      {:ok, c} = Config.from_map(tiny_config(arch))
      {:ok, p} = Llama.program(c, tiny_weights(c), max_seq: 16)
      ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end
      caches = for {:input, n, dt, [s, w]} <- Program.inputs(p), dt == :f32, into: %{}, do: {n, Tensor.random(:f32, [s, w], :erlang.phash2(n), scale: 0.5)}
      env = Map.merge(caches, %{tok: ids.([3, 1, 4, 1]), pos: ids.([5, 6, 7, 8])})
      got = run_xla(p, env)
      {_c, want} = oracle(p, env)
      g = Tensor.to_floats(got.logits)
      w = Tensor.to_floats(want.logits)
      scale = w |> Enum.map(&abs/1) |> Enum.max()
      assert Enum.zip(g, w) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max() < 1.0e-5 * max(scale, 1.0), arch
      argmax = fn xs -> xs |> Enum.chunk_every(c.vocab) |> Enum.map(fn r -> Enum.find_index(r, &(&1 == Enum.max(r))) end) end
      assert argmax.(g) == argmax.(w)
      # the cache update is exact selection: rows not written keep their bits,
      # the written ones (positions 5…8) hold the new keys and values
      for {n, old} <- caches, nx = :"#{n}_next", Map.has_key?(got, nx) do
        for r <- 0..15, r not in 5..8, do: assert(Tensor.row(got[nx], r) == Tensor.row(old, r))
        for r <- 5..8, do: refute(Tensor.row(got[nx], r) == Tensor.row(old, r))
      end
    end
  end

  @tag :jax
  @tag timeout: 600_000
  test "the admission kit measures the PJRT device that runs it: XLA's CPU is envelope-bound, not canonical" do
    dir = Path.join(System.tmp_dir!(), "vapor-kit-#{System.unique_integer([:positive])}")
    %{exported: ex, skipped: sk} = Substrate.Kit.write(dir)
    assert :linear_f32 in ex and :attention in ex and :contraction in ex
    assert Keyword.has_key?(sk, :sample) and Keyword.has_key?(sk, :qgemv_sb4)
    assert File.exists?(Path.join(dir, "run_kit.py")) and File.exists?(Path.join(dir, "stablehlo_run.py"))

    {out, 0} = System.cmd(python(), [Path.join(dir, "run_kit.py"), dir, "--platform", "cpu"], stderr_to_stdout: true)
    refute out =~ "FAILED"
    a = Substrate.Kit.judge(dir)
    assert a.device =~ "PJRT"
    # measured, not assumed: XLA fuses a·b+c and flushes subnormals, even
    # though the module asks for a multiply and an add
    assert a.verdict == :envelope
    assert a.fingerprint.contraction == true
    assert a.fingerprint.flush_to_zero == true and a.fingerprint.significand_bits == 24
    assert a.probes.gemm_i8.equal and a.probes.nan_select.equal
  end
end
