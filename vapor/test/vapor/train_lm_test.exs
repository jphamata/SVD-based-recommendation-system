defmodule Vapor.TrainLMTest do
  @moduledoc """
  Pre-training from scratch (`Vapor.Train.LM`): gradients against
  PyTorch's autograd, bits that do not depend on how many workers there
  are (nor on which computes what, nor on one of them dying), resumption
  equal to an uninterrupted run, and a checkpoint that the inference stack
  and `transformers` load.
  """
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Train.LM
  alias Vapor.Runtime.Worker
  import Vapor.TestHelpers

  @corpus "Um texto pequeno para treinar, repetido com variações: a eclusa admite, o compilador emite, o worker executa. " <>
            "A small text to train on, repeated with variations: the airlock admits, the compiler emits, the worker runs. "

  setup_all do
    c = LM.new(d: 32, layers: 2, heads: 2, ff: 64, seq: 16, seqs: 2, vocab: 256)
    corpus = String.duplicate(@corpus, 6)
    run = LM.start(c, corpus, micro: 4, chunk: 2, steps: 20, seed: 7, lr: 1.0e-2, warmup: 2)
    {:ok, c: c, corpus: corpus, run: run}
  end

  defp workers(n), do: for(_ <- 1..n, do: elem(Worker.start_link(exec: worker_exec(:host)), 1))

  @tag :torch
  test "gradients = PyTorch autograd in binary64, every parameter", %{c: c, corpus: corpus} do
    ps = LM.init(c, 3)
    b = LM.batch(c, corpus, 1, 1, 0)
    {:ok, comp} = Vapor.Compile.Lower.lower(LM.grad_program(c))
    {:ok, r} = Vapor.Runtime.Native.run_oracle(comp, Map.merge(Map.new(ps, fn {n, t} -> {:"p.#{n}", t} end), %{x: b.x, y: b.y}))
    d = Path.join(System.tmp_dir!(), "vapor-lmref-#{System.unique_integer([:positive])}")
    File.mkdir_p!(d)
    File.write!(Path.join(d, "config.json"), Vapor.JSON.encode(Map.from_struct(c)))
    :ok = Vapor.Ingest.Safetensors.write(Path.join(d, "params.safetensors"), ps)
    ids = fn t -> t |> Tensor.to_floats() |> Enum.chunk_every(c.vocab) |> Enum.map(fn row -> Enum.find_index(row, &(&1 == 1.0)) end) end
    File.write!(Path.join(d, "batch.json"), Vapor.JSON.encode(%{x: ids.(b.x), y: ids.(b.y)}))
    {out, 0} = System.cmd(python(), [Path.expand("../python/lm_reference.py", __DIR__), d], stderr_to_stdout: true)
    _ = out
    {:ok, ref} = Vapor.JSON.decode(File.read!(Path.join(d, "loss.json")))
    assert abs(hd(Tensor.to_floats(r.outputs.loss)) - ref["loss"]) < 1.0e-5 * ref["loss"]
    {:ok, gs} = Vapor.Ingest.Safetensors.read(Path.join(d, "grads.safetensors"))

    for {n, _} <- LM.shapes(c) do
      g = Tensor.to_floats(r.outputs[:"g.#{n}"])
      want = Tensor.to_floats(gs[n])
      scale = want |> Enum.map(&abs/1) |> Enum.max()
      err = Enum.zip(g, want) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()
      assert err <= 2.0e-5 * scale, "#{n}: #{err} vs scale #{scale}"
    end

    # a control: the gradient of another batch is far from this reference
    b2 = LM.batch(c, corpus, 1, 1, 1)
    {:ok, r2} = Vapor.Runtime.Native.run_oracle(comp, Map.merge(Map.new(ps, fn {n, t} -> {:"p.#{n}", t} end), %{x: b2.x, y: b2.y}))
    far = Tensor.to_floats(r2.outputs[:"g.head"]) |> Enum.zip(Tensor.to_floats(gs["head"])) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()
    assert far > 1.0e-2 * (Tensor.to_floats(gs["head"]) |> Enum.map(&abs/1) |> Enum.max())
  end

  @tag :native
  test "the same bits on 1 or 2 workers, any assignment, a worker lost mid-run, and the exact oracle", %{corpus: corpus, run: run} do
    one = LM.train(run, corpus, workers(1), 3)
    two = LM.train(run, corpus, workers(2), 3)
    perverse = LM.train(run, corpus, workers(2), 3, assign: fn k, j -> k + j + 1 end)
    assert LM.digest(one) == LM.digest(two)
    assert LM.digest(one) == LM.digest(perverse)
    assert one.losses == two.losses

    # one of the two workers dies after the first step: its chunks are recomputed elsewhere
    [w1, w2] = workers(2)
    r = LM.train(run, corpus, [w1, w2], 1)
    Process.unlink(w2)
    Process.exit(w2, :kill)
    r = LM.train(r, corpus, [w1, w2], 2)
    assert LM.digest(r) == LM.digest(one)

    # the exact oracle (no worker at all) agrees bit for bit
    assert LM.digest(LM.train(run, corpus, [], 1)) == LM.digest(LM.train(run, corpus, workers(1), 1))
  end

  @tag :native
  test "the tree shape is part of the definition: another chunk size is another run (the test can fail)", %{c: c, corpus: corpus, run: run} do
    # chunk 2 of 4 is the balanced tree itself ((0 + a) + b = a + b); a fold of 4 is another shape
    other = LM.start(c, corpus, micro: 4, chunk: 4, steps: 20, seed: 7, lr: 1.0e-2, warmup: 2)
    ws = workers(2)
    refute LM.digest(LM.train(run, corpus, ws, 2)) == LM.digest(LM.train(other, corpus, ws, 2))
  end

  @tag :native
  test "checkpoint and resume continue with the bits of an uninterrupted run", %{corpus: corpus, run: run} do
    ws = workers(2)
    straight = LM.train(run, corpus, ws, 4)
    half = LM.train(run, corpus, ws, 2)
    path = Path.join(System.tmp_dir!(), "vapor-lm-#{System.unique_integer([:positive])}.safetensors")
    :ok = LM.checkpoint(half, path)
    resumed = run |> LM.resume(path) |> LM.train(corpus, ws, 2)
    assert resumed.step == 4 and LM.digest(resumed) == LM.digest(straight)
  end

  @tag :native
  test "it learns: the loss falls and the held-out bits per byte beat the byte frequencies", %{corpus: corpus, run: run} do
    ws = workers(2)
    {before, _} = LM.bits_per_byte(run, @corpus, ws)
    trained = LM.train(run, corpus, ws, 20)
    {after_, _} = LM.bits_per_byte(trained, @corpus, ws)
    {_, l_first} = List.last(trained.losses)
    {_, l_last} = hd(trained.losses)
    assert l_last < 0.6 * l_first
    unigram = Vapor.Quality.Text.unigram_bits(@corpus, Vapor.Quality.Text.profile(corpus))
    assert before > 7.0 and after_ < unigram, "untrained #{before}, trained #{after_}, unigram #{unigram}"
  end

  @tag :native
  test "the exported checkpoint is a Llama: the inference stack computes the training model's logits", %{c: c, corpus: corpus, run: run} do
    trained = LM.train(run, corpus, workers(1), 2)
    dir = Path.join(System.tmp_dir!(), "vapor-lm-export-#{System.unique_integer([:positive])}")
    {:ok, _} = LM.export(trained, dir)
    {:ok, %{program: p}} = Vapor.Model.load(dir, max_seq: c.seq)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    toks = :binary.bin_to_list(binary_part(corpus, 5, c.seq))
    ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end
    {:ok, inf} = Vapor.Runtime.Native.run_oracle(comp, Map.merge(zero_caches(p), %{tok: ids.(toks), pos: ids.(Enum.to_list(0..(c.seq - 1)))}))
    {:ok, tr} = Vapor.Compile.Lower.lower(LM.eval_program(c))
    x = LM.onehot(toks ++ List.duplicate(32, c.seq), c.vocab)
    {:ok, ev} = Vapor.Runtime.Native.run_oracle(tr, Map.merge(Map.new(trained.params, fn {n, t} -> {:"p.#{n}", t} end), %{x: x}))
    a = inf.outputs.logits |> Tensor.to_floats()
    b = ev.outputs.logits |> Tensor.to_floats() |> Enum.take(c.seq * c.vocab)
    assert Enum.zip(a, b) |> Enum.map(fn {u, v} -> abs(u - v) end) |> Enum.max() < 1.0e-4
  end

  @tag :torch
  @tag :native
  test "transformers loads the exported checkpoint and computes the same logits", %{c: c, corpus: corpus, run: run} do
    trained = LM.train(run, corpus, workers(1), 2)
    dir = Path.join(System.tmp_dir!(), "vapor-lm-hf-#{System.unique_integer([:positive])}")
    {:ok, _} = LM.export(trained, dir)
    toks = :binary.bin_to_list(binary_part(corpus, 9, c.seq))
    File.write!(Path.join(dir, "toks.json"), Vapor.JSON.encode(toks))
    out = Path.join(dir, "hf.safetensors")
    {log, rc} = System.cmd(python(), [Path.expand("../python/hf_logits.py", __DIR__), dir, Path.join(dir, "toks.json"), out], stderr_to_stdout: true)
    assert rc == 0, log
    {:ok, %{"logits" => hf}} = Vapor.Ingest.Safetensors.read(out)
    {:ok, tr} = Vapor.Compile.Lower.lower(LM.eval_program(c))
    x = LM.onehot(toks ++ List.duplicate(32, c.seq), c.vocab)
    {:ok, ev} = Vapor.Runtime.Native.run_oracle(tr, Map.merge(Map.new(trained.params, fn {n, t} -> {:"p.#{n}", t} end), %{x: x}))
    a = Tensor.to_floats(hf)
    b = ev.outputs.logits |> Tensor.to_floats() |> Enum.take(c.seq * c.vocab)
    assert Enum.zip(a, b) |> Enum.map(fn {u, v} -> abs(u - v) end) |> Enum.max() < 1.0e-4
  end

  defp zero_caches(p) do
    for {:input, n, :f32, [s, w]} <- Vapor.Program.inputs(p), into: %{}, do: {n, Tensor.new(:f32, [s, w], :binary.copy(<<0::32>>, s * w))}
  end
end
