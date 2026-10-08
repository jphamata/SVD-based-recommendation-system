defmodule Vapor.HFParityTest do
  @moduledoc """
  Phase P3, the model oracle: small random checkpoints written by Hugging
  Face `transformers` (PyTorch, float32) are loaded through vapor's airlocks
  and executed on the native substrate. The two implementations sum in
  different orders, so equality is not expected; the declared tolerance is

      max |logit_vapor − logit_hf|  ≤  1e-5 · max |logit_hf|

  (measured: ≤ 6e-7 on every variant — see docs/ECOSYSTEM.md), and greedy
  decoding must produce the same tokens, except after a step whose top-two
  logits are closer than the tolerance (a tie that either order may break).

  Variants cover each family and each RoPE form: Llama with q/k/v/o biases
  and an untied head, Llama 3 frequency scaling, linear scaling with plain
  multi-head attention, Mistral, and Qwen2 with a tied head and a single KV
  head.

  4-bit weights (`quantize: :sb4`) change the model, so they are held to a
  different standard: on a width-256 Llama, the error of vapor's `:sb4`
  logits against transformers' float32 logits must be no worse than 1.25×
  that of the common 4-bit baseline with the same group size (affine,
  32-weight groups, "Q4_1"), fake-quantized in PyTorch — in relative RMSE
  and in mean KL divergence of the next-token distributions.
  """
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Ingest.Safetensors
  alias Vapor.Model.Decoder
  alias Vapor.Runtime.{Native, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :torch
  @moduletag timeout: 600_000
  @variants ~w(llama llama3-rope linear-rope mistral qwen2)
  @s 32
  @tol 1.0e-5

  setup_all do
    root = Path.join(System.tmp_dir!(), "vapor-hf-#{System.unique_integer([:positive])}")
    script = Path.expand("../python/hf_reference.py", __DIR__)

    for v <- @variants ++ ["llama-q"] do
      File.mkdir_p!(Path.join(root, v))
      py!(File.read!(script), [Path.join(root, v), v, "7"])
    end

    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    on_exit(fn -> File.rm_rf!(root) end)
    {:ok, root: root, worker: w}
  end

  defp load(root, v, opts \\ []) do
    {:ok, ref} = Safetensors.read(Path.join([root, v, "reference.safetensors"]))
    {:ok, %{config: c, program: p}} = Vapor.Model.load(Path.join(root, v), [max_seq: @s] ++ opts)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {c, comp, ref}
  end

  defp step(w, comp, caches, toks, p0) do
    n = length(toks)
    env = Map.merge(caches, %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(p0..(p0 + n - 1)))})
    {:ok, got} = Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native)
    next = Map.new(caches, fn {k, _} -> {k, got.outputs[:"#{k}_next"]} end)
    {got.outputs.logits, next}
  end

  defp last_row(%Tensor{shape: [n, v]} = t), do: t |> Tensor.row(n - 1) |> then(&Tensor.new(:f32, [v], &1)) |> Tensor.to_floats()

  # first index of the maximum (torch.argmax), and the gap to the runner-up
  defp argmax(xs) do
    {best, i} = xs |> Enum.with_index() |> Enum.max_by(&elem(&1, 0), fn a, b -> a > b end)
    second = xs |> List.delete_at(i) |> Enum.max()
    {i, best - second}
  end

  for v <- @variants do
    test "#{v}: prefill logits within tolerance of transformers", %{root: root, worker: w} do
      {c, comp, ref} = load(root, unquote(v))
      {logits, _} = step(w, comp, Decoder.empty_caches(c, @s), Tensor.to_list(ref["prompt"]), 0)

      got = Tensor.to_floats(logits)
      want = Tensor.to_floats(ref["logits"])
      scale = want |> Enum.map(&abs/1) |> Enum.max()
      err = Enum.zip_with(got, want, &abs(&1 - &2)) |> Enum.max()
      assert logits.shape == ref["logits"].shape
      assert err <= @tol * scale, "max |Δ| = #{err}, allowed #{@tol * scale}"
    end

    test "#{v}: greedy decoding (prefill + cached single-token steps) matches transformers", %{root: root, worker: w} do
      {c, comp, ref} = load(root, unquote(v))
      prompt = Tensor.to_list(ref["prompt"])
      want = ref["greedy"] |> Tensor.to_list() |> Enum.drop(length(prompt))

      {logits, caches} = step(w, comp, Decoder.empty_caches(c, @s), prompt, 0)

      {got, _} =
        Enum.map_reduce(0..(length(want) - 1), {logits, caches}, fn i, {logits, caches} ->
          row = last_row(logits)
          {tok, gap} = argmax(row)
          {lg, cs} = if i < length(want) - 1, do: step(w, comp, caches, [tok], length(prompt) + i), else: {nil, caches}
          {{tok, gap, Enum.max(Enum.map(row, &abs/1))}, {lg, cs}}
        end)

      # identical up to the first near-tie, which only a tie may explain
      mismatch = Enum.zip(got, want) |> Enum.find_index(fn {{t, _, _}, u} -> t != u end)

      if mismatch do
        {_, gap, scale} = Enum.at(got, mismatch)
        assert gap <= 2 * @tol * scale, "diverged at step #{mismatch} with a clear margin #{gap}"
      end

      assert mismatch == nil or mismatch > 0
      if mismatch, do: IO.puts("#{unquote(v)}: near-tie at step #{mismatch}")
    end
  end

  # vapor's export, read back by transformers: F32 must give *the same*
  # logits as the original checkpoint (same weights, same configuration);
  # BF16 the logits vapor computes on the same rounded weights
  for v <- @variants do
    test "#{v}: exported by vapor as a safetensors directory, transformers reproduces the checkpoint", %{root: root, worker: w} do
      {:ok, m} = Vapor.Model.open(Path.join(root, unquote(v)))
      {:ok, ref} = Safetensors.read(Path.join([root, unquote(v), "reference.safetensors"]))
      prompt = Tensor.to_list(ref["prompt"])

      for dtype <- ["F32", "BF16"] do
        dir = Path.join(root, "#{unquote(v)}-export-#{dtype}")
        :ok = Vapor.Model.write(dir, m.config, m.weights, dtype: dtype, shard_bytes: 40_000)

        out =
          py!("""
          import sys, json, torch
          from transformers import AutoModelForCausalLM
          m = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype=torch.float32).eval()
          with torch.no_grad():
              print(m(torch.tensor([json.loads(sys.argv[2])])).logits[0].contiguous().numpy().tobytes().hex())
          """, [dir, Vapor.JSON.encode(prompt)])

        theirs = Base.decode16!(String.trim(out), case: :lower)

        if dtype == "F32" do
          assert theirs == ref["logits"].data
        else
          {:ok, %{config: c, program: p}} = Vapor.Model.load(dir, max_seq: @s)
          {:ok, comp} = Vapor.Compile.Lower.lower(p)
          {logits, _} = step(w, comp, Decoder.empty_caches(c, @s), prompt, 0)
          want = for <<x::float-32-little <- theirs>>, do: x
          scale = want |> Enum.map(&abs/1) |> Enum.max()
          assert Enum.zip_with(Tensor.to_floats(logits), want, &abs(&1 - &2)) |> Enum.max() <= @tol * scale
        end
      end
    end
  end

  # relative RMSE and mean KL(p_ref ‖ p) over the prompt positions
  defp quality(%Tensor{} = got, %Tensor{shape: [_, v]} = ref) do
    rows = &(&1 |> Tensor.to_floats() |> Enum.chunk_every(v))
    {a, b} = {rows.(got), rows.(ref)}
    {fa, fb} = {List.flatten(a), List.flatten(b)}
    rmse = :math.sqrt(Enum.sum(Enum.zip_with(fa, fb, &((&1 - &2) ** 2))) / Enum.sum(Enum.map(fb, &(&1 * &1))))

    softmax = fn r ->
      m = Enum.max(r)
      e = Enum.map(r, &:math.exp(&1 - m))
      s = Enum.sum(e)
      Enum.map(e, &(&1 / s))
    end

    kl =
      Enum.zip_with(a, b, fn x, y ->
        Enum.zip_with(softmax.(y), softmax.(x), fn p, q -> p * :math.log(p / q) end) |> Enum.sum()
      end)

    {rmse, Enum.sum(kl) / length(kl)}
  end

  test "llama (width 256): f32 within tolerance; 4-bit :sb4 no worse than 1.25× the Q4_1 baseline", %{root: root, worker: w} do
    {c, comp, ref} = load(root, "llama-q")
    prompt = Tensor.to_list(ref["prompt"])
    {f32, _} = step(w, comp, Decoder.empty_caches(c, @s), prompt, 0)
    {rmse32, _} = quality(f32, ref["logits"])
    assert rmse32 < @tol

    {_, qcomp, _} = load(root, "llama-q", quantize: :sb4)
    {q, _} = step(w, qcomp, Decoder.empty_caches(c, @s), prompt, 0)
    {rmse, kl} = quality(q, ref["logits"])
    {rmse41, kl41} = quality(ref["logits_q4_1"], ref["logits"])
    {rmse40, kl40} = quality(ref["logits_q4_0"], ref["logits"])

    IO.puts("\n  4-bit, width 256, #{length(prompt)} positions — rel. RMSE / mean KL: " <>
              "sb4 #{fmt(rmse)} / #{fmt(kl)}, Q4_1 #{fmt(rmse41)} / #{fmt(kl41)}, Q4_0 #{fmt(rmse40)} / #{fmt(kl40)}")

    assert rmse <= 1.25 * rmse41
    assert kl <= 1.25 * kl41
  end

  defp fmt(x), do: :erlang.float_to_binary(x, decimals: 4)
end
