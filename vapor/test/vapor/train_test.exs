defmodule Vapor.TrainTest do
  @moduledoc """
  Phase P7 — LoRA distillation as a recurrent program. A teacher that
  differs from the base model by a rank-16 change of its last MLP is
  distilled into the base model's LoRA adapters: the KL divergence falls,
  every step runs inside the worker (one crossing for the whole run), and
  the run is bit-for-bit reproducible — oracle = host, any thread count.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor, Train}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Native, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :native
  @moduletag timeout: 900_000
  @t 16

  setup_all do
    {:ok, c} = Config.from_map(tiny_config("llama", %{"vocab_size" => 128}))
    ws = tiny_weights(c, 5)
    l = c.layers - 1
    pre = "model.layers.#{l}."
    frozen = %{ln2: ws[pre <> "post_attention_layernorm.weight"], wg: ws[pre <> "mlp.gate_proj.weight"],
               wu: ws[pre <> "mlp.up_proj.weight"], wd: ws[pre <> "mlp.down_proj.weight"],
               norm: ws["model.norm.weight"], head: ws["lm_head.weight"]}

    # features: the base model's input to its last MLP block, for 4 batches
    {:ok, p} = Llama.program(c, ws, max_seq: 16)
    name = :"layers.#{l}.attn_out"
    {:input, _, _, shape} = T.ref(name, Program.bound(p)[name])
    feat = Program.new([h: T.input(name, :f32, shape)], lets: p.lets)
    {:ok, fc} = Vapor.Compile.Lower.lower(feat)
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))

    hs =
      for b <- 1..4 do
        toks = Tensor.random(:s32, [@t], 100 + b, max: 128)
        env = %{tok: toks, pos: Tensor.from_list(:s32, [@t], Enum.to_list(0..(@t - 1)))}
        {:ok, r} = Native.run(w, fc, Map.merge(env, Llama.empty_caches(c, 16)), isa: Substrates.host_isa(), mode: :native)
        r.outputs.h
      end

    {:ok, comp} = Vapor.Compile.Lower.lower(Train.program(c, frozen, rank: 16, alpha: 32, tokens: @t, lr: 3.0e-3))

    # the teacher: the same block with adapters set to a rank-16 change
    shapes = %{gate_b: [c.intermediate, 16], up_b: [c.intermediate, 16], down_b: [c.hidden, 16]}
    teacher = Map.merge(Train.init(c, seed: 77), Map.new(shapes, fn {k, sh} -> {k, Tensor.random(:f32, sh, 200 + map_size(%{k => 1}) + length(sh) * 7 + hd(sh), scale: 0.15)} end))

    pts =
      for h <- hs do
        env = Map.merge(teacher, %{h: Tensor.new(:f32, [1 | h.shape], h.data), p_teacher: zero([1, @t, c.vocab])})
              |> Map.merge(Train.schedule(1))
        {:ok, r} = Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native, iterations: 1,
                              sequence: [:h, :p_teacher, :c1, :c2])
        softmax_rows(r.outputs.logits, c.vocab)
      end

    {:ok, c: c, comp: comp, hs: hs, pts: pts, worker: w}
  end

  defp zero(shape), do: Tensor.new(:f32, shape, :binary.copy(<<0::32>>, Enum.product(shape)))

  defp softmax_rows(z, v) do
    rows = z |> Tensor.to_floats() |> Enum.chunk_every(v)
    Tensor.from_list(:f32, z.shape, Enum.flat_map(rows, fn r ->
      m = Enum.max(r)
      e = Enum.map(r, &:math.exp(&1 - m))
      s = Enum.sum(e)
      Enum.map(e, &(&1 / s))
    end))
  end

  defp stack(ts), do: Tensor.new(hd(ts).dtype, [length(ts) | hd(ts).shape], IO.iodata_to_binary(Enum.map(ts, & &1.data)))

  defp run_env(ctx, steps) do
    idx = for s <- 0..(steps - 1), do: rem(s, 4)
    Train.init(ctx.c)
    |> Map.merge(%{h: stack(Enum.map(idx, &Enum.at(ctx.hs, &1))), p_teacher: stack(Enum.map(idx, &Enum.at(ctx.pts, &1)))})
    |> Map.merge(Train.schedule(steps))
    |> then(&{&1, idx})
  end

  @seq [:h, :p_teacher, :c1, :c2]

  test "distillation lowers the KL divergence, all steps behind one crossing", ctx do
    steps = 120
    {env, idx} = run_env(ctx, steps)
    {:ok, r} = Native.run(ctx.worker, ctx.comp, env, isa: Substrates.host_isa(), mode: :native, iterations: steps, sequence: @seq)
    kls = Enum.zip_with(r.steps, idx, fn s, i -> Train.kl(Enum.at(ctx.pts, i), s.logits) end)
    first = Enum.take(kls, 4) |> Enum.sum()
    last = Enum.take(kls, -4) |> Enum.sum()
    IO.puts("\n  KL over 4 batches: first #{Float.round(first / 4, 5)}, last #{Float.round(last / 4, 5)} (#{steps} steps, #{Float.round(r.elapsed_ns / 1.0e6, 1)} ms in the worker)")
    assert last < 0.25 * first
  end

  test "a training run is bit-for-bit reproducible: oracle = host, 1 = 2 threads", ctx do
    {env, _} = run_env(ctx, 2)
    run = [iterations: 2, sequence: @seq]
    {:ok, ref} = Native.run_oracle(ctx.comp, env, run)

    for threads <- [1, 2] do
      {:ok, w} = Worker.start_link(exec: worker_exec(:host), threads: threads)
      {:ok, got} = Native.run(w, ctx.comp, env, [isa: Substrates.host_isa(), mode: :native] ++ run)
      assert got.outputs == ref.outputs
    end
  end
end
