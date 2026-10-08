defmodule Vapor.Science.Biology do
  @moduledoc """
  Evolution, genomes and folding, each against an exact or published
  reference (docs/SCIENCE.md):

    * **population genetics** — a mutant's fixation in a Wright–Fisher
      population, simulated replicate by replicate with counter-based
      draws, against the **exact** probability of the Markov chain (solved
      here) and Kimura's diffusion formula (1962);
    * **phylogenetics** — sequences evolved down a known tree by the
      Jukes–Cantor model, distances corrected for multiple hits, the tree
      rebuilt by neighbour joining (Saitou & Nei 1987) and compared split
      by split (Robinson–Foulds); the control destroys homology by
      shuffling each sequence's sites;
    * **protein folding, the HP model** (Dill 1985) on the square lattice —
      exact ground states by enumeration of self-avoiding walks for short
      chains, and a seeded Monte Carlo search (pull moves are not needed at
      these lengths: crankshaft/end/corner moves with annealing) that
      reaches the published optimum of the classic 20-mer benchmark
      (E = −9, Unger & Moult 1993). The control: random self-avoiding
      walks.

  Real protein structure — its metrics (TM-score, GDT, lDDT), folding
  from contacts by distance geometry and contacts read from co-evolution —
  is `Vapor.Bio` (docs/PROTEINS.md); the HP model here is the physics toy
  that makes the search problem exact.
  """
  alias Vapor.Sampler

  # ----------------------------------------------------- Wright–Fisher

  defp binom_pmf(n, p) do
    q = 1 - p

    cond do
      p == 0.0 -> [1.0 | List.duplicate(0.0, n)]
      p == 1.0 -> List.duplicate(0.0, n) ++ [1.0]
      true -> for k <- 0..n, do: :math.exp(lnchoose(n, k) + k * :math.log(p) + (n - k) * :math.log(q))
    end
  end

  defp lnchoose(n, k), do: lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
  defp lgamma(x), do: Enum.reduce(1..(x - 1)//1, 0.0, fn i, s -> s + :math.log(i) end)

  defp next_p(i, n, s), do: i * (1 + s) / (i * (1 + s) + n - i)

  @doc "The exact fixation probability of one mutant (haploid Wright–Fisher, N, selection s): the chain's absorption probability."
  def fixation_exact(n, s) do
    rows = for i <- 0..n, do: binom_pmf(n, next_p(i, n, s))
    u0 = for i <- 0..n, do: i / n

    Enum.reduce_while(1..100_000, u0, fn _, u ->
      u2 = for {row, i} <- Enum.with_index(rows), do: (if i in [0, n], do: (if i == n, do: 1.0, else: 0.0), else: Enum.sum(Enum.zip_with(row, u, &*/2)))
      if Enum.zip(u, u2) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max() < 1.0e-14, do: {:halt, u2}, else: {:cont, u2}
    end)
    |> Enum.at(1)
  end

  @doc "Kimura's diffusion approximation for one mutant: (1 − e^{−2s}) / (1 − e^{−2Ns}) (haploid)."
  def kimura(n, s) when s == 0, do: 1 / n
  def kimura(n, s), do: (1 - :math.exp(-2 * s)) / (1 - :math.exp(-2 * n * s))

  @doc "Simulated fixation frequency over `reps` replicates (seeded)."
  def fixation_sim(n, s, reps, seed \\ 1) do
    fixed =
      Enum.count(1..reps, fn r ->
        Enum.reduce_while(1..100_000, 1, fn g, i ->
          p = next_p(i, n, s)
          # binomial draw by inversion of the cdf
          u = Sampler.uniform(seed * 1_000_003 + r, g)
          k = invert(binom_pmf(n, p), u)
          cond do
            k == 0 -> {:halt, false}
            k == n -> {:halt, true}
            true -> {:cont, k}
          end
        end) == true
      end)

    fixed / reps
  end

  defp invert(pmf, u), do: invert(pmf, u, 0, 0.0)
  defp invert([p], _u, k, _acc), do: if(p >= 0, do: k, else: k)
  defp invert([p | rest], u, k, acc), do: if(acc + p >= u, do: k, else: invert(rest, u, k + 1, acc + p))

  # ------------------------------------------------------- phylogenetics

  @doc """
  The reference tree (6 taxa, branch lengths in substitutions per site),
  as `{:node, [{child, length}]}` / `{:leaf, name}`.
  """
  def tree do
    {:node,
     [{{:node, [{{:leaf, "A"}, 0.1}, {{:leaf, "B"}, 0.15}]}, 0.1},
      {{:node, [{{:leaf, "C"}, 0.2}, {{:node, [{{:leaf, "D"}, 0.05}, {{:leaf, "E"}, 0.1}]}, 0.15}]}, 0.05},
      {{:leaf, "F"}, 0.3}]}
  end

  @doc "Evolve sequences of length `len` down `tree` by Jukes–Cantor: `%{name => [0..3]}`."
  def evolve(tree, len, seed \\ 1) do
    root = for i <- 0..(len - 1), do: trunc(Sampler.uniform(seed, i) * 4) |> min(3)
    walk(tree, root, seed * 7919, %{}) |> elem(1)
  end

  defp walk({:leaf, name}, seq, salt, acc), do: {salt, Map.put(acc, name, seq)}

  defp walk({:node, kids}, seq, salt, acc) do
    Enum.reduce(kids, {salt, acc}, fn {child, t}, {salt, acc} ->
      p = 0.75 * (1 - :math.exp(-4 / 3 * t))
      salt = salt + 1
      mutated = for {b, i} <- Enum.with_index(seq), do: (u = Sampler.uniform(salt, 2 * i); if(u < p, do: rem(b + 1 + min(trunc(Sampler.uniform(salt, 2 * i + 1) * 3), 2), 4), else: b))
      walk(child, mutated, salt * 31, acc)
    end)
  end

  @doc "Jukes–Cantor distance between two sequences (p-distance corrected for multiple hits)."
  def jc_distance(a, b) do
    p = Enum.zip(a, b) |> Enum.count(fn {x, y} -> x != y end) |> Kernel./(length(a))
    if p >= 0.75, do: 10.0, else: -0.75 * :math.log(1 - 4 / 3 * p)
  end

  @doc "Neighbour joining on a distance map `%{{a, b} => d}` over `names`: the unrooted tree as a list of splits (sets of leaf names on one side)."
  def neighbour_joining(names, dist) do
    d = for a <- names, b <- names, into: %{}, do: {{a, b}, if(a == b, do: 0.0, else: Map.get(dist, {a, b}) || Map.fetch!(dist, {b, a}))}
    nj(Enum.map(names, &{&1, MapSet.new([&1])}), d, [])
  end

  # clusters: [{id, leafset}]; returns the splits made
  defp nj(clusters, _d, splits) when length(clusters) <= 3, do: splits

  defp nj(clusters, d, splits) do
    n = length(clusters)
    ids = Enum.map(clusters, &elem(&1, 0))
    r = Map.new(ids, fn i -> {i, Enum.sum(for j <- ids, do: d[{i, j}])} end)
    {i, j} = (for a <- ids, b <- ids, a < b, do: {a, b}) |> Enum.min_by(fn {a, b} -> (n - 2) * d[{a, b}] - r[a] - r[b] end)
    set = MapSet.union(elem(List.keyfind(clusters, i, 0), 1), elem(List.keyfind(clusters, j, 0), 1))
    u = {:u, i, j}
    rest = Enum.reject(ids, &(&1 in [i, j]))
    d = Enum.reduce(rest, d, fn k, d -> v = (d[{i, k}] + d[{j, k}] - d[{i, j}]) / 2; d |> Map.put({u, k}, v) |> Map.put({k, u}, v) end) |> Map.put({u, u}, 0.0)
    nj([{u, set} | Enum.reject(clusters, fn {id, _} -> id in [i, j] end)], d, [set | splits])
  end

  @doc "The non-trivial splits of a rooted reference tree (each as the leaf set below an internal edge)."
  def splits(tree) do
    {_, s} = collect(tree)
    all = leaves(tree)
    s |> Enum.reject(&(MapSet.size(&1) < 2 or MapSet.size(&1) > MapSet.size(all) - 2))
  end

  defp collect({:leaf, n}), do: {MapSet.new([n]), []}
  defp collect({:node, kids}) do
    {sets, acc} = Enum.reduce(kids, {[], []}, fn {c, _}, {sets, acc} -> {s, a} = collect(c); {[s | sets], acc ++ [s | a]} end)
    {Enum.reduce(sets, MapSet.new(), &MapSet.union/2), acc}
  end

  defp leaves(t), do: elem(collect(t), 0)

  @doc "Robinson–Foulds distance between split lists over `all` leaves (a split and its complement are the same)."
  def robinson_foulds(s1, s2, all) do
    canon = fn s -> MapSet.new(s, fn x -> if MapSet.member?(x, Enum.min(all)), do: x, else: MapSet.difference(MapSet.new(all), x) end) end
    {a, b} = {canon.(s1), canon.(s2)}
    MapSet.size(MapSet.difference(a, b)) + MapSet.size(MapSet.difference(b, a))
  end

  @doc """
  Simulate, measure, rebuild: `%{rf, rf_shuffled, distance_error}` — the
  RF distance of the NJ tree to the truth, the control's (each sequence's
  sites shuffled), and the mean relative error of the corrected distances
  against the true path lengths.
  """
  def phylogeny(len \\ 2000, seed \\ 1) do
    t = tree()
    seqs = evolve(t, len, seed)
    names = Enum.sort(Map.keys(seqs))
    dist = for a <- names, b <- names, a < b, into: %{}, do: {{a, b}, jc_distance(seqs[a], seqs[b])}
    truth = splits(t)
    nj = neighbour_joining(names, dist)
    shuffled = Map.new(seqs, fn {k, s} -> {k, Vapor.Modal.Rng.permute(s, :erlang.phash2(k))} end)
    dist_s = for a <- names, b <- names, a < b, into: %{}, do: {{a, b}, jc_distance(shuffled[a], shuffled[b])}
    true_d = path_lengths(t)
    err = (for {{a, b}, d} <- dist, do: abs(d - true_d[{a, b}]) / true_d[{a, b}]) |> then(&(Enum.sum(&1) / length(&1)))
    %{rf: robinson_foulds(nj, truth, names), rf_shuffled: robinson_foulds(neighbour_joining(names, dist_s), truth, names), distance_error: err}
  end

  defp path_lengths(t) do
    depths = depth_map(t, 0.0, [], %{})
    names = Map.keys(depths)

    for a <- names, b <- names, a < b, into: %{} do
      {da, pa} = depths[a]
      {db, pb} = depths[b]
      {{_, dc}, _} = Enum.zip(pa, pb) |> Enum.take_while(fn {x, y} -> x == y end) |> List.last()
      {{a, b}, da + db - 2 * dc}
    end
  end

  # leaf → {depth, ancestors root first as {id, depth}}
  defp depth_map({:leaf, n}, d, path, acc), do: Map.put(acc, n, {d, path})

  defp depth_map({:node, kids}, d, path, acc) do
    path = path ++ [{make_ref(), d}]
    Enum.reduce(kids, acc, fn {c, l}, acc -> depth_map(c, d + l, path, acc) end)
  end

  # -------------------------------------------------------- HP folding

  @doc "The classic 20-mer (Unger & Moult 1993): optimum −9 on the square lattice."
  def benchmark20, do: "HPHPPHHPHPPHPHHPPHPH"

  @doc "Energy: −1 per non-bonded H–H lattice contact."
  def energy(seq, coords) do
    hs = for {c, i} <- Enum.with_index(String.graphemes(seq)), c == "H", do: i
    pos = coords |> Enum.with_index() |> Map.new(fn {p, i} -> {p, i} end)
    cs = List.to_tuple(coords)
    hset = MapSet.new(hs)

    -Enum.count(for i <- hs, {dx, dy} <- [{1, 0}, {0, 1}], {x, y} = elem(cs, i), j = pos[{x + dx, y + dy}], j != nil, MapSet.member?(hset, j), abs(i - j) > 1, do: 1)
  end

  @doc "Exact ground state by enumeration of self-avoiding walks (first step fixed, first turn up): for short chains."
  def ground_state(seq) do
    n = String.length(seq)
    {best, conf} = saw(n, [{1, 0}, {0, 0}], MapSet.new([{0, 0}, {1, 0}]), false, seq, {1, nil})
    %{energy: best, coords: conf}
  end

  defp saw(n, path, _seen, _turned, seq, {best, bc}) when length(path) == n do
    coords = Enum.reverse(path)
    e = energy(seq, coords)
    if e < best, do: {e, coords}, else: {best, bc}
  end

  defp saw(n, [{x, y} | _] = path, seen, turned, seq, acc) do
    moves = if turned, do: [{1, 0}, {-1, 0}, {0, 1}, {0, -1}], else: [{1, 0}, {0, 1}]

    Enum.reduce(moves, acc, fn {dx, dy}, acc ->
      p = {x + dx, y + dy}
      if MapSet.member?(seen, p), do: acc, else: saw(n, [p | path], MapSet.put(seen, p), turned or dy != 0, seq, acc)
    end)
  end

  @doc """
  Monte Carlo search (seeded): pivot moves on a self-avoiding chain with
  simulated annealing; `%{energy, coords, steps}`.
  """
  def fold(seq, opts \\ []) do
    n = String.length(seq)
    steps = Keyword.get(opts, :steps, 300_000)
    seed = Keyword.get(opts, :seed, 1)
    start = for i <- 0..(n - 1), do: {i, 0}

    {_, best, bc} =
      Enum.reduce(1..steps, {start, 0, start}, fn s, {cur, best, bc} ->
        temp = Keyword.get(opts, :t0, 1.2) * :math.pow(Keyword.get(opts, :t1, 0.12) / Keyword.get(opts, :t0, 1.2), s / steps)
        k = 1 + trunc(Sampler.uniform(seed, 3 * s) * (n - 2))
        op = trunc(Sampler.uniform(seed, 3 * s + 1) * 6)
        # half pivots (global), half local moves (corner flips, end moves)
        cand = if op < 3, do: pivot(cur, k, op), else: local(cur, trunc(Sampler.uniform(seed, 3 * s) * n) |> min(n - 1), op)

        if cand && self_avoiding?(cand) do
          {e0, e1} = {energy(seq, cur), energy(seq, cand)}
          accept = e1 <= e0 or Sampler.uniform(seed, 3 * s + 2) < :math.exp(-(e1 - e0) / temp)
          cur = if accept, do: cand, else: cur
          e = if accept, do: e1, else: e0
          if e < best, do: {cur, e, cur}, else: {cur, best, bc}
        else
          {cur, best, bc}
        end
      end)

    %{energy: best, coords: bc, steps: steps}
  end

  # rotate (90°, 180°, 270°) the tail after site k about it
  defp pivot(coords, k, op) do
    {head, tail} = Enum.split(coords, k + 1)
    {cx, cy} = List.last(head)
    rot = fn {x, y} ->
      {dx, dy} = {x - cx, y - cy}
      case op do
        0 -> {cx - dy, cy + dx}
        1 -> {cx - dx, cy - dy}
        _ -> {cx + dy, cy - dx}
      end
    end
    head ++ Enum.map(tail, rot)
  end

  # a corner flip (a residue whose neighbours are diagonal moves to the other corner) or an end move
  defp local(coords, i, op) do
    n = length(coords)
    t = List.to_tuple(coords)

    cond do
      i == 0 or i == n - 1 ->
        {x, y} = elem(t, if(i == 0, do: 1, else: n - 2))
        opts = for {dx, dy} <- [{1, 0}, {-1, 0}, {0, 1}, {0, -1}], q = {x + dx, y + dy}, q not in coords, do: q
        if opts == [], do: nil, else: List.replace_at(coords, i, Enum.at(opts, rem(op, length(opts))))

      true ->
        {{x0, y0}, {x1, y1}, {x2, y2}} = {elem(t, i - 1), elem(t, i), elem(t, i + 1)}
        if abs(x0 - x2) == 1 and abs(y0 - y2) == 1, do: List.replace_at(coords, i, {x0 + x2 - x1, y0 + y2 - y1}), else: nil
    end
  end

  defp self_avoiding?(coords), do: length(Enum.uniq(coords)) == length(coords)

  @doc "The control: the mean energy of random self-avoiding conformations (grown at random, restarts on dead ends)."
  def random_energy(seq, samples \\ 200, seed \\ 1) do
    n = String.length(seq)
    es = for s <- 1..samples, c = grow(n, seed * 1000 + s), c != nil, do: energy(seq, c)
    Enum.sum(es) / max(length(es), 1)
  end

  defp grow(n, salt) do
    Enum.reduce_while(1..50, nil, fn attempt, _ ->
      path = Enum.reduce_while(1..(n - 1), [{0, 0}], fn i, [{x, y} | _] = p ->
        free = for {dx, dy} <- [{1, 0}, {-1, 0}, {0, 1}, {0, -1}], q = {x + dx, y + dy}, q not in p, do: q
        if free == [], do: {:halt, nil}, else: {:cont, [Enum.at(free, trunc(Sampler.uniform(salt * 61 + attempt, i) * length(free)) |> min(length(free) - 1)) | p]}
      end)
      if path, do: {:halt, Enum.reverse(path)}, else: {:cont, nil}
    end)
  end
end
