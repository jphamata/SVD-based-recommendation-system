defmodule Vapor.MergeAlignTest do
  use ExUnit.Case, async: true
  alias Vapor.{Lock, Merge, Tensor}
  alias Vapor.Merge.Align
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Runtime.Oracle
  import Vapor.TestHelpers

  defp model(seed) do
    map = tiny_config("qwen2")
    {:ok, c} = Config.from_map(map)
    {:ok, spec, ws} = Lock.from_map(map, tiny_weights(c, seed))
    {%{spec: spec, weights: ws}, c}
  end

  # the same network with its SwiGLU hidden units shuffled by a seeded permutation
  defp shuffled(%{weights: ws} = m, seed) do
    :rand.seed(:exsss, {seed, 1, 2})

    ws =
      ws
      |> Enum.filter(fn {k, _} -> is_binary(k) and String.ends_with?(k, ".mlp.gate_proj.weight") end)
      |> Enum.reduce(ws, fn {k, g}, acc ->
        b = String.replace_suffix(k, ".mlp.gate_proj.weight", "")
        perm = Enum.shuffle(0..(hd(g.shape) - 1))
        permute(acc, b, perm)
      end)

    %{m | weights: ws}
  end

  defp permute(ws, b, perm) do
    g = ws[b <> ".mlp.gate_proj.weight"]
    [n, k] = g.shape
    rows = fn t -> rs = t |> Tensor.to_floats() |> Enum.chunk_every(k) |> List.to_tuple(); Tensor.from_list(:f32, t.shape, Enum.flat_map(perm, &elem(rs, &1))) end
    d = ws[b <> ".mlp.down_proj.weight"]
    cols = fn t -> t |> Tensor.to_floats() |> Enum.chunk_every(n) |> Enum.flat_map(fn r -> rt = List.to_tuple(r); Enum.map(perm, &elem(rt, &1)) end) |> then(&Tensor.from_list(:f32, t.shape, &1)) end
    ws |> Map.put(b <> ".mlp.gate_proj.weight", rows.(g)) |> Map.put(b <> ".mlp.up_proj.weight", rows.(ws[b <> ".mlp.up_proj.weight"])) |> Map.put(b <> ".mlp.down_proj.weight", cols.(d))
  end

  defp logits(c, ws) do
    toks = [3, 50, 7, 81, 12]
    n = length(toks)
    {:ok, p} = Decoder.program(c, ws, max_seq: 16)
    env = Map.merge(Decoder.empty_caches(c, 16), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
    Oracle.eval_program(p, env).logits |> Tensor.to_floats()
  end

  defp max_diff(a, b), do: Enum.zip_reduce(a, b, 0.0, fn x, y, m -> max(m, abs(x - y)) end)

  test "the Hungarian method finds the optimal assignment (against all permutations, n ≤ 7)" do
    :rand.seed(:exsss, {1, 1, 1})

    for n <- 1..7, _ <- 1..6 do
      sim = for _ <- 1..n, do: for(_ <- 1..n, do: :rand.uniform() * 2 - 1)
      p = Align.hungarian_max(sim)
      assert Enum.sort(p) == Enum.to_list(0..(n - 1))
      val = fn perm -> Enum.zip(sim, perm) |> Enum.reduce(0.0, fn {row, j}, s -> s + Enum.at(row, j) end) end
      best = perms(Enum.to_list(0..(n - 1))) |> Enum.map(val) |> Enum.max()
      assert_in_delta val.(p), best, 1.0e-12
    end
  end

  test "a planted shuffle is undone exactly, and the shuffled network was the same function all along" do
    {a, c} = model(1)
    b = shuffled(a, 7)
    refute b.weights == a.weights
    assert {:ok, back, rep} = Align.align(a, b)
    assert back.weights == a.weights
    assert Enum.all?(rep.blocks, &(&1.moved > 0 and &1.similarity_after >= &1.similarity_before))
    # the permutation is a symmetry: the logits agree up to the order of down's contraction
    assert max_diff(logits(c, a.weights), logits(c, b.weights)) < 1.0e-4
  end

  test "fusing a network with its shuffled copy: without alignment it is damaged, with alignment it is itself" do
    {a, c} = model(2)
    b = shuffled(a, 11)
    ref = logits(c, a.weights)
    {:ok, naive} = Merge.merge([a, b], method: :linear)
    {:ok, fixed} = Merge.merge([a, b], method: :linear, align: true)
    assert fixed.weights == a.weights |> Map.merge(Map.take(fixed.weights, Enum.reject(Map.keys(fixed.weights), &is_binary/1)))
    assert max_diff(ref, logits(c, fixed.weights)) == 0.0
    assert max_diff(ref, logits(c, naive.weights)) > 1.0e-3
    assert [d] = fixed.receipt.payload.params.aligned
    assert byte_size(d) == 64
  end

  test "two independently initialised networks: alignment raises the matched similarity of every block" do
    {a, _} = model(3)
    {b, _} = model(4)
    {:ok, _, rep} = Align.align(a, b)
    assert Enum.all?(rep.blocks, &(&1.similarity_after > &1.similarity_before))
  end

  test "refusals: nothing to align; blocks that do not correspond" do
    {a, _} = model(5)
    no_mlp = %{a | weights: Map.reject(a.weights, fn {k, _} -> is_binary(k) and String.contains?(k, ".mlp.") end)}
    assert {:error, _} = Align.align(no_mlp, no_mlp)
    assert {:error, _} = Align.align(a, no_mlp)
  end

  defp perms([]), do: [[]]
  defp perms(xs), do: for(x <- xs, rest <- perms(xs -- [x]), do: [x | rest])
end
