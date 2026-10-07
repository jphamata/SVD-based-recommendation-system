defmodule Vapor.Logic.Problems do
  @moduledoc """
  Finite combinatorics as CNF (docs/LOGICA.md §2): the encodings behind
  the computer-assisted theorems of Ramsey theory — each a function from
  the parameters to `%{vars, clauses, decode}` for `Vapor.Logic.SAT`.

    * `schur(n, k)` — colour 1…n with k colours, no monochromatic
      x + y = z. Schur numbers S(2) = 4, S(3) = 13 (and S(4) = 44, S(5) =
      160 — the last settled by a two-petabyte SAT proof in 2017).
    * `vdw(k, r, n)` — r colours on 1…n, no monochromatic arithmetic
      progression of length k. W(3; 2) = 9, W(3; 3) = 27, W(4; 2) = 35.
    * `ramsey(s, t, n)` — two-colour the edges of Kₙ with no red Kₛ and
      no blue Kₜ. R(3, 3) = 6, R(3, 4) = 9.
    * `pigeonhole(p, h)`, `queens(n)`, `coloring(edges, k)`,
      `sudoku(grid)`.

  The number itself is a pair of facts: a model at n − 1 (checked by
  evaluation) and a refutation at n (checked by DRUP).
  """

  defp at_least_one(vs), do: [vs]
  defp at_most_one(vs), do: for(a <- vs, b <- vs, a < b, do: [-a, -b])

  @doc "Schur: {1…n} in k sum-free colour classes. The colour of 1 is fixed (symmetry)."
  def schur(n, k) do
    v = fn i, c -> (i - 1) * k + c end
    cl =
      (for i <- 1..n, do: for(c <- 1..k, do: v.(i, c))) ++
      (for i <- 1..n, c1 <- 1..k, c2 <- 1..k, c1 < c2, do: [-v.(i, c1), -v.(i, c2)]) ++
      (for x <- 1..n, y <- x..n, x + y <= n, c <- 1..k, do: Enum.uniq([-v.(x, c), -v.(y, c), -v.(x + y, c)])) ++
      [[v.(1, 1)]]
    %{vars: n * k, clauses: cl, decode: fn m -> for i <- 1..n, do: Enum.find(1..k, &m[v.(i, &1)]) end, name: "Schur S(#{k}) ≥ #{n}?"}
  end

  @doc "Check a Schur colouring directly: no x + y = z inside one class."
  def schur_ok?(colours) do
    t = List.to_tuple(colours)
    n = tuple_size(t)
    Enum.all?(for x <- 1..n, y <- x..n, x + y <= n, do: not (elem(t, x - 1) == elem(t, y - 1) and elem(t, y - 1) == elem(t, x + y - 1)))
  end

  @doc "van der Waerden: r colours on 1…n, no monochromatic k-term progression. The colour of 1 is fixed."
  def vdw(k, r, n) do
    v = fn i, c -> (i - 1) * r + c end
    aps = for a <- 1..n, d <- 1..div(n - 1, max(k - 1, 1))//1, a + (k - 1) * d <= n, do: for(j <- 0..(k - 1), do: a + j * d)
    cl =
      (for i <- 1..n, do: for(c <- 1..r, do: v.(i, c))) ++
      (for i <- 1..n, c1 <- 1..r, c2 <- 1..r, c1 < c2, do: [-v.(i, c1), -v.(i, c2)]) ++
      (for ap <- aps, c <- 1..r, do: Enum.map(ap, &(-v.(&1, c)))) ++ [[v.(1, 1)]]
    %{vars: n * r, clauses: cl, decode: fn m -> for i <- 1..n, do: Enum.find(1..r, &m[v.(i, &1)]) end, name: "W(#{k}; #{r}) > #{n}?"}
  end

  @doc "Check a colouring for monochromatic k-term progressions."
  def vdw_ok?(colours, k) do
    t = List.to_tuple(colours)
    n = tuple_size(t)
    not Enum.any?(for a <- 1..n, d <- 1..max(div(n - 1, max(k - 1, 1)), 1), a + (k - 1) * d <= n, do: Enum.map(0..(k - 1), &elem(t, a + &1 * d - 1)) |> Enum.uniq() |> length() == 1)
  end

  @doc "Ramsey: edges of Kₙ red (true) or blue, with no red Kₛ and no blue Kₜ."
  def ramsey(s, t, n) do
    pairs = for i <- 1..n, j <- (i + 1)..n//1, do: {i, j}
    id = pairs |> Enum.with_index(1) |> Map.new()
    e = fn a, b -> id[{min(a, b), max(a, b)}] end
    cliques = fn k -> combinations(Enum.to_list(1..n), k) end
    cl =
      (for c <- cliques.(s), do: for([a, b] <- combinations(c, 2), do: -e.(a, b))) ++
      (for c <- cliques.(t), do: for([a, b] <- combinations(c, 2), do: e.(a, b)))
    %{vars: length(pairs), clauses: cl, decode: fn m -> for {{a, b}, i} <- id, m[i], do: [a, b] end, name: "R(#{s}, #{t}) > #{n}?"}
  end

  @doc "Check a red edge set: no red Kₛ, no blue Kₜ."
  def ramsey_ok?(red, s, t, n) do
    r = MapSet.new(Enum.map(red, fn [a, b] -> {min(a, b), max(a, b)} end))
    redp = fn a, b -> MapSet.member?(r, {min(a, b), max(a, b)}) end
    not Enum.any?(combinations(Enum.to_list(1..n), s), fn c -> Enum.all?(combinations(c, 2), fn [a, b] -> redp.(a, b) end) end) and
      not Enum.any?(combinations(Enum.to_list(1..n), t), fn c -> Enum.all?(combinations(c, 2), fn [a, b] -> not redp.(a, b) end) end)
  end

  @doc "Pigeonhole: p pigeons in h holes, one per hole (unsatisfiable when p > h; hard for resolution)."
  def pigeonhole(p, h) do
    v = fn i, j -> (i - 1) * h + j end
    cl = (for i <- 1..p, do: for(j <- 1..h, do: v.(i, j))) ++ (for j <- 1..h, i1 <- 1..p, i2 <- 1..p, i1 < i2, do: [-v.(i1, j), -v.(i2, j)])
    %{vars: p * h, clauses: cl, decode: fn m -> for i <- 1..p, do: Enum.find(1..h, &m[v.(i, &1)]) end, name: "#{p} pigeons in #{h} holes?"}
  end

  @doc "N queens."
  def queens(n) do
    v = fn r, c -> (r - 1) * n + c end
    sq = for r <- 1..n, c <- 1..n, do: {r, c}
    attack = fn {r1, c1}, {r2, c2} -> r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2) end
    cl = (for r <- 1..n, do: for(c <- 1..n, do: v.(r, c))) ++
         (for a <- sq, b <- sq, a < b, attack.(a, b), do: [-v.(elem(a, 0), elem(a, 1)), -v.(elem(b, 0), elem(b, 1))])
    %{vars: n * n, clauses: cl, decode: fn m -> for r <- 1..n, do: Enum.find(1..n, &m[v.(r, &1)]) end, name: "#{n} queens"}
  end

  @doc "k-colouring of a graph given as edges `[[a, b]]` over vertices 1…n."
  def coloring(edges, k) do
    n = edges |> List.flatten() |> Enum.max(fn -> 1 end)
    v = fn i, c -> (i - 1) * k + c end
    cl = (for i <- 1..n, do: for(c <- 1..k, do: v.(i, c))) ++ Enum.flat_map(1..n, fn i -> at_most_one(for c <- 1..k, do: v.(i, c)) end) ++
         (for [a, b] <- edges, c <- 1..k, do: [-v.(a, c), -v.(b, c)])
    %{vars: n * k, clauses: cl, decode: fn m -> for i <- 1..n, do: Enum.find(1..k, &m[v.(i, &1)]) end, name: "#{k}-colouring"}
  end

  @doc "Sudoku (9×9, 0 = empty) as CNF."
  def sudoku(grid) do
    v = fn r, c, d -> 81 * r + 9 * c + d end
    cells = for r <- 0..8, c <- 0..8, do: {r, c}
    units = (for r <- 0..8, do: for(c <- 0..8, do: {r, c})) ++ (for c <- 0..8, do: for(r <- 0..8, do: {r, c})) ++
            (for br <- 0..2, bc <- 0..2, do: for(r <- 0..2, c <- 0..2, do: {3 * br + r, 3 * bc + c}))
    cl = Enum.flat_map(cells, fn {r, c} -> at_least_one(for d <- 1..9, do: v.(r, c, d)) ++ at_most_one(for d <- 1..9, do: v.(r, c, d)) end) ++
         Enum.flat_map(units, fn u -> Enum.flat_map(1..9, fn d -> at_least_one(for {r, c} <- u, do: v.(r, c, d)) end) end) ++
         (for {row, r} <- Enum.with_index(grid), {d, c} <- Enum.with_index(row), d in 1..9, do: [v.(r, c, d)])
    %{vars: 729, clauses: cl, decode: fn m -> for r <- 0..8, do: for(c <- 0..8, do: Enum.find(1..9, &m[v.(r, c, &1)])) end, name: "sudoku"}
  end

  @doc false
  def combinations(_, 0), do: [[]]
  def combinations([], _), do: []
  def combinations([h | t], k), do: Enum.map(combinations(t, k - 1), &[h | &1]) ++ combinations(t, k)
end
