defmodule Vapor.Bio.Align do
  @moduledoc """
  Pairwise sequence alignment (docs/PROTEINS.md §4): Gotoh's dynamic
  programme with affine gaps (open + extend), global (Needleman–Wunsch)
  or local (Smith–Waterman), scored by BLOSUM62 for proteins or by
  match/mismatch for nucleotides. Exact: the score is the optimum, and
  the traceback is an alignment that attains it (both checked against
  Biopython's `PairwiseAligner` in the tests).
  """

  @blosum_path Path.join([:code.priv_dir(:vapor) |> to_string(), "quality", "protein", "BLOSUM62"])

  @doc "BLOSUM62 as a map `{a, b} => score`."
  def blosum62 do
    case :persistent_term.get({__MODULE__, :blosum}, nil) do
      nil ->
        lines = File.read!(@blosum_path) |> String.split("\n") |> Enum.reject(&(String.starts_with?(&1, "#") or String.trim(&1) == ""))
        [head | rows] = lines
        cols = String.split(head)
        m = for r <- rows, [a | vals] = String.split(r), {b, v} <- Enum.zip(cols, vals), into: %{}, do: {{a, b}, String.to_integer(v)}
        :persistent_term.put({__MODULE__, :blosum}, m)
        m
      m -> m
    end
  end

  @doc """
  Align `a` and `b`. Options: `mode: :global | :local`, `matrix: :blosum62 |
  {match, mismatch}`, `open: 11`, `extend: 1` (gap of length k costs
  open + (k − 1)·extend). `%{score, a, b, identity, mode}`.
  """
  def align(a, b, opts \\ []) do
    mode = Keyword.get(opts, :mode, :global)
    open = Keyword.get(opts, :open, 11)
    ext = Keyword.get(opts, :extend, 1)
    s = case Keyword.get(opts, :matrix, :blosum62) do
      :blosum62 -> (m = blosum62(); fn x, y -> Map.get(m, {x, y}, -4) end)
      {ma, mi} -> fn x, y -> if x == y, do: ma, else: mi end
    end
    at = a |> String.upcase() |> String.graphemes() |> List.to_tuple()
    bt = b |> String.upcase() |> String.graphemes() |> List.to_tuple()
    {n, m} = {tuple_size(at), tuple_size(bt)}
    ninf = -1_000_000_000
    local = mode == :local

    # rows of {M, X (gap in b: a consumed), Y (gap in a)} with traceback pointers
    row0 = for j <- 0..m, do: (if j == 0, do: {0, ninf, ninf}, else: {ninf, ninf, if(local, do: ninf, else: -open - (j - 1) * ext)})
    {rows, _} =
      Enum.map_reduce(1..n//1, row0, fn i, prev ->
        first = {ninf, if(local, do: ninf, else: -open - (i - 1) * ext), ninf}
        {row, _} =
          Enum.map_reduce(1..m//1, first, fn j, left ->
            {pm, px, py} = Enum.at(prev, j - 1)
            {upm, ux, upy} = Enum.at(prev, j)
            sc = s.(elem(at, i - 1), elem(bt, j - 1))
            mm = Enum.max([pm, px, py]) + sc
            mm = if local, do: max(mm, sc), else: mm
            xx = max(upm - open, max(ux - ext, upy - open))
            {lm, lx, ly} = left
            yy = max(lm - open, max(ly - ext, lx - open))
            cell = {mm, xx, yy}
            {cell, cell}
          end)
        r = [first | row]
        {r, r}
      end)
    table = [row0 | rows] |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    {si, sj, score} =
      if local do
        for i <- 0..n, j <- 0..m, reduce: {0, 0, 0} do
          {bi, bj, bs} -> (v = elem(elem(elem(table, i), j), 0); if v > bs, do: {i, j, v}, else: {bi, bj, bs})
        end
      else
        {mm, xx, yy} = elem(elem(table, n), m)
        {n, m, Enum.max([mm, xx, yy])}
      end
    start = if local, do: 0, else: (({mm, xx, _yy} = elem(elem(table, n), m)); cond do mm == score -> 0; xx == score -> 1; true -> 2 end)
    {ra, rb} = trace(table, at, bt, si, sj, start, s, open, ext, local, [], [])
    ident = Enum.zip(ra, rb) |> Enum.count(fn {x, y} -> x == y and x != "-" end)
    %{score: score, a: Enum.join(ra), b: Enum.join(rb), identity: if(ra == [], do: 0.0, else: ident / length(ra)), mode: mode}
  end

  defp v(table, i, j, k), do: elem(elem(elem(table, i), j), k)

  defp trace(_t, _a, _b, 0, 0, _st, _s, _o, _e, _local, ra, rb), do: {ra, rb}
  defp trace(t, a, b, i, j, st, s, o, e, local, ra, rb) do
    cond do
      local and st == 0 and v(t, i, j, 0) == s.(elem(a, i - 1), elem(b, j - 1)) and i >= 1 and j >= 1 and
          (i == 1 or j == 1 or Enum.max([v(t, i - 1, j - 1, 0), v(t, i - 1, j - 1, 1), v(t, i - 1, j - 1, 2)]) <= 0) ->
        {[elem(a, i - 1) | ra], [elem(b, j - 1) | rb]}
      i == 0 -> trace(t, a, b, 0, j - 1, 2, s, o, e, local, ["-" | ra], [elem(b, j - 1) | rb])
      j == 0 -> trace(t, a, b, i - 1, 0, 1, s, o, e, local, [elem(a, i - 1) | ra], ["-" | rb])
      st == 0 ->
        sc = s.(elem(a, i - 1), elem(b, j - 1))
        cur = v(t, i, j, 0)
        prev = Enum.find(0..2, fn k -> v(t, i - 1, j - 1, k) + sc == cur end) || 0
        trace(t, a, b, i - 1, j - 1, prev, s, o, e, local, [elem(a, i - 1) | ra], [elem(b, j - 1) | rb])
      st == 1 ->
        cur = v(t, i, j, 1)
        prev = cond do v(t, i - 1, j, 0) - o == cur -> 0; v(t, i - 1, j, 1) - e == cur -> 1; true -> 2 end
        trace(t, a, b, i - 1, j, prev, s, o, e, local, [elem(a, i - 1) | ra], ["-" | rb])
      true ->
        cur = v(t, i, j, 2)
        prev = cond do v(t, i, j - 1, 0) - o == cur -> 0; v(t, i, j - 1, 2) - e == cur -> 2; true -> 1 end
        trace(t, a, b, i, j - 1, prev, s, o, e, local, ["-" | ra], [elem(b, j - 1) | rb])
    end
  end

  @doc "The score of a given alignment (two gapped strings of equal length) — the check of `align/3`'s traceback."
  def score_of(ga, gb, opts \\ []) do
    open = Keyword.get(opts, :open, 11)
    ext = Keyword.get(opts, :extend, 1)
    s = case Keyword.get(opts, :matrix, :blosum62) do
      :blosum62 -> (m = blosum62(); fn x, y -> Map.get(m, {x, y}, -4) end)
      {ma, mi} -> fn x, y -> if x == y, do: ma, else: mi end
    end
    Enum.zip(String.graphemes(ga), String.graphemes(gb))
    |> Enum.reduce({0, nil}, fn {x, y}, {acc, gap} ->
      cond do
        x == "-" -> {acc - if(gap == :a, do: ext, else: open), :a}
        y == "-" -> {acc - if(gap == :b, do: ext, else: open), :b}
        true -> {acc + s.(x, y), nil}
      end
    end)
    |> elem(0)
  end
end
