defmodule Vapor.Merge.Align do
  @moduledoc """
  **Permutation alignment** before fusion — what `Vapor.Merge.diagnose/2`
  asked for when it called two networks "unrelated" (docs/MERGING.md §8).

  A network is the same function under any permutation of its hidden
  units: in a SwiGLU block `down(silu(gate·x) ⊙ up·x)`, reordering the rows
  of `gate` and `up` and the columns of `down` by one permutation changes
  nothing (each hidden unit is computed and consumed on its own). Two
  networks trained apart land in different orders, so a weight average
  mixes unrelated units and destroys both. **Weight matching** (Ainsworth,
  Hayase & Srinivasa, *Git Re-Basin*, 2023) picks, per block, the
  permutation of the second model that maximises the total inner product
  with the first — a linear assignment problem, solved exactly here by the
  shortest-augmenting-path Hungarian method (`O(n³)` in the block's width).

  Scope, stated: the SwiGLU blocks (`*.mlp.gate_proj/up_proj/down_proj`)
  of the Llama-family layout, which is where the width is. Attention heads
  and the residual stream also have symmetries (head order within a
  key/value group; a permutation of the hidden dimension threaded through
  every layer) — not aligned here. The permuted model is the *same
  function* as the original up to the rounding of `down`'s contraction in
  another order (checked by test). Width above #{2048} is refused: the
  cubic assignment belongs on a native substrate, not the BEAM.
  """
  alias Vapor.Tensor

  @max_width 2048

  @doc """
  Permute `model`'s SwiGLU hidden units to match `reference` (both
  `%{spec, weights}`). Returns `{:ok, aligned_model, report}`; the report
  has, per block, the similarity before and after and how many units moved,
  and a SHA-256 of every permutation (what a receipt records).
  """
  def align(%{weights: ref} = _reference, %{weights: w} = model) do
    blocks = blocks(ref)
    width = fn b -> hd(ref[b <> ".mlp.gate_proj.weight"].shape) end
    wide = Enum.find(blocks, &(width.(&1) > @max_width))

    cond do
      blocks == [] ->
        {:error, "no SwiGLU blocks (*.mlp.gate_proj/up_proj/down_proj) to align"}

      Enum.any?(blocks, fn b -> not Map.has_key?(w, b <> ".mlp.gate_proj.weight") end) ->
        {:error, "the models do not have the same blocks"}

      wide != nil ->
        {:error, "#{wide}: width #{width.(wide)} above #{@max_width}"}

      true ->
        {weights, reports} =
          Enum.map_reduce(blocks, w, fn b, acc ->
            {perm, rep} = match_block(ref, acc, b)
            {rep, permute_block(acc, b, perm)}
          end)
          |> then(fn {reps, ws} -> {ws, reps} end)

        {:ok, %{model | weights: weights}, %{blocks: reports, digest: digest(reports)}}
    end
  end

  defp blocks(ws) do
    for({k, _} <- ws, is_binary(k), m = Regex.run(~r/^(.*)\.mlp\.gate_proj\.weight$/, k), m != nil,
        p = Enum.at(m, 1), Map.has_key?(ws, p <> ".mlp.up_proj.weight") and Map.has_key?(ws, p <> ".mlp.down_proj.weight"),
        do: p)
    |> Enum.sort()
  end

  # S[i][j] = gateᴬᵢ·gateᴮⱼ + upᴬᵢ·upᴮⱼ + downᴬ[:,i]·downᴮ[:,j]
  defp match_block(ref, w, b) do
    {ga, ua, da} = parts(ref, b)
    {gb, ub, db} = parts(w, b)
    n = length(ga)
    feats_a = Enum.zip_with([ga, ua, da], fn [g, u, d] -> g ++ u ++ d end)
    feats_b = Enum.zip_with([gb, ub, db], fn [g, u, d] -> g ++ u ++ d end)
    sim = for fa <- feats_a, do: for(fb <- feats_b, do: dot(fa, fb))
    perm = hungarian_max(sim)
    before = Enum.reduce(0..(n - 1), 0.0, fn i, s -> s + (sim |> Enum.at(i) |> Enum.at(i)) end)
    after_ = Enum.zip(sim, perm) |> Enum.reduce(0.0, fn {row, j}, s -> s + Enum.at(row, j) end)
    moved = perm |> Enum.with_index() |> Enum.count(fn {j, i} -> i != j end)
    {perm, %{block: b, width: n, similarity_before: before, similarity_after: after_, moved: moved,
             permutation_sha256: :crypto.hash(:sha256, :erlang.term_to_binary(perm)) |> Base.encode16(case: :lower)}}
  end

  # rows of gate and up, and the columns of down, as lists of floats per hidden unit
  defp parts(ws, b) do
    g = ws[b <> ".mlp.gate_proj.weight"]
    u = ws[b <> ".mlp.up_proj.weight"]
    d = ws[b <> ".mlp.down_proj.weight"]
    [n, k] = g.shape
    [dm, ^n] = d.shape
    rows = fn t -> t |> Tensor.to_floats() |> Enum.chunk_every(k) end
    cols = d |> Tensor.to_floats() |> Enum.chunk_every(n) |> Enum.zip_with(& &1)
    _ = dm
    {rows.(g), rows.(u), cols}
  end

  defp dot(a, b), do: Enum.zip_reduce(a, b, 0.0, fn x, y, s -> s + x * y end)

  # new unit i is the old unit perm[i]
  defp permute_block(ws, b, perm) do
    g = ws[b <> ".mlp.gate_proj.weight"]
    u = ws[b <> ".mlp.up_proj.weight"]
    d = ws[b <> ".mlp.down_proj.weight"]
    [_n, k] = g.shape
    [_dm, n] = d.shape
    pt = List.to_tuple(perm)
    rows = fn %Tensor{data: data} = t ->
      chunks = for <<r::binary-size(k * 4) <- data>>, do: r
      ct = List.to_tuple(chunks)
      %{t | data: IO.iodata_to_binary(for i <- 0..(n - 1), do: elem(ct, elem(pt, i)))}
    end

    cols = fn %Tensor{data: data} = t ->
      out =
        for <<row::binary-size(n * 4) <- data>> do
          vals = for <<v::binary-4 <- row>>, do: v
          vt = List.to_tuple(vals)
          for i <- 0..(n - 1), do: elem(vt, elem(pt, i))
        end

      %{t | data: IO.iodata_to_binary(out)}
    end

    ws
    |> Map.put(b <> ".mlp.gate_proj.weight", rows.(Tensor.widen(g)) |> retype(g))
    |> Map.put(b <> ".mlp.up_proj.weight", rows.(Tensor.widen(u)) |> retype(u))
    |> Map.put(b <> ".mlp.down_proj.weight", cols.(Tensor.widen(d)) |> retype(d))
  end

  defp retype(t, %Tensor{dtype: :bf16}), do: Tensor.to_bf16(t)
  defp retype(t, _), do: t

  defp digest(reports), do: :crypto.hash(:sha256, Enum.map(reports, & &1.permutation_sha256)) |> Base.encode16(case: :lower)

  # ------------------------------------------------------------ assignment

  @doc """
  The permutation `p` maximising `Σᵢ sim[i][p[i]]` (a square list of lists
  of numbers): the Hungarian method by shortest augmenting paths with
  potentials, `O(n³)`, exact for the given numbers; ties go to the lowest
  column, so the answer is deterministic.
  """
  def hungarian_max(sim) do
    n = length(sim)
    if Enum.any?(sim, &(length(&1) != n)), do: raise(ArgumentError, "a square matrix")
    big = sim |> List.flatten() |> Enum.map(&abs/1) |> Enum.max(fn -> 0.0 end)
    # minimise cost = −sim (shifted to be non-negative; the optimum does not move)
    cost = sim |> Enum.map(fn row -> row |> Enum.map(&(big - &1)) |> List.to_tuple() end) |> List.to_tuple()
    hungarian_min(cost, n)
  end

  # e-maxx's O(n³) Hungarian: rows 1..n assigned to columns 1..n; 0 is the sentinel
  defp hungarian_min(cost, n) do
    inf = :infinity
    u = :array.new(n + 1, default: 0.0)
    v = :array.new(n + 1, default: 0.0)
    p = :array.new(n + 1, default: 0)

    {_u, _v, p} =
      Enum.reduce(1..n//1, {u, v, p}, fn i, {u, v, p} ->
        p = :array.set(0, i, p)
        minv = :array.new(n + 1, default: inf)
        used = :array.new(n + 1, default: false)
        way = :array.new(n + 1, default: 0)
        augment(cost, n, u, v, p, minv, used, way, 0)
      end)

    # p[j] = row assigned to column j  →  perm[row] = column
    assign = for j <- 1..n, into: %{}, do: {:array.get(j, p) - 1, j - 1}
    for i <- 0..(n - 1), do: assign[i]
  end

  defp augment(cost, n, u, v, p, minv, used, way, j0) do
    used = :array.set(j0, true, used)
    i0 = :array.get(j0, p)
    row = elem(cost, i0 - 1)
    ui0 = :array.get(i0, u)

    {minv, way, delta, j1} =
      Enum.reduce(1..n, {minv, way, :infinity, 0}, fn j, {mv, wy, delta, j1} ->
        if :array.get(j, used) do
          {mv, wy, delta, j1}
        else
          cur = elem(row, j - 1) - ui0 - :array.get(j, v)
          {mv, wy} = if lt(cur, :array.get(j, mv)), do: {:array.set(j, cur, mv), :array.set(j, j0, wy)}, else: {mv, wy}
          mj = :array.get(j, mv)
          if lt(mj, delta), do: {mv, wy, mj, j}, else: {mv, wy, delta, j1}
        end
      end)

    {u, v, minv} =
      Enum.reduce(0..n, {u, v, minv}, fn j, {u, v, mv} ->
        if :array.get(j, used) do
          pj = :array.get(j, p)
          {:array.set(pj, :array.get(pj, u) + delta, u), :array.set(j, :array.get(j, v) - delta, v), mv}
        else
          {u, v, :array.set(j, :array.get(j, mv) - delta, mv)}
        end
      end)

    if :array.get(j1, p) == 0 do
      {u, v, unwind(p, way, j1)}
    else
      augment(cost, n, u, v, p, minv, used, way, j1)
    end
  end

  defp unwind(p, _way, 0), do: p

  defp unwind(p, way, j1) do
    j0 = :array.get(j1, way)
    unwind(:array.set(j1, :array.get(j0, p), p), way, j0)
  end

  defp lt(_a, :infinity), do: true
  defp lt(:infinity, _b), do: false
  defp lt(a, b), do: a < b
end
