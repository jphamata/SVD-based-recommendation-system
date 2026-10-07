defmodule Vapor.Graph do
  @moduledoc """
  **Complex networks, with the statistics that keep claims honest.**

  Network science has two reproducibility problems. The first is
  numerical: community detection, sampling and bootstraps are randomised
  and order-dependent, so a reported modularity or a "scale-free" verdict
  moves from run to run and machine to machine. The second is
  statistical: a straight-ish line on a log–log degree plot was for years
  taken as a power law, and when the tests of Clauset, Shalizi & Newman
  (2009) were applied to a thousand real networks, strongly scale-free
  ones turned out to be rare (Broido & Clauset 2019).

  Here every random choice is a counter-based draw of `(seed, counter)`
  (`Vapor.Sampler`) and the logarithms that turn draws into skips are
  correctly rounded (`Vapor.CR`), so a generator, a null model, a
  bootstrap sample or an epidemic is a function of its seed — on any
  machine. And every structural claim
  comes with its control:

    * `power_law/2` — the discrete power law fitted by maximum likelihood
      with the cutoff `x_min` chosen by the Kolmogorov–Smirnov distance,
      a goodness-of-fit p-value by semi-parametric bootstrap, and a
      likelihood ratio against the exponential (Vuong's test): the
      Barabási–Albert degrees pass, the Erdős–Rényi ones do not;
    * `rewire/3` — the configuration-model null (degree-preserving double
      edge swaps): a clustering coefficient or a modularity means
      something only as a z-score against it (`zscore/4`);
    * `communities/2` — Louvain (local moves, then aggregation) in a fixed
      node order with ties to the lowest index, scored by modularity;
      recovered on a planted partition, nothing on its null (label
      propagation was tried first and floods a planted partition into one
      community when the visiting order is not random);
    * `sir/3`, `percolation/3`, `giant/1` — epidemics and robustness,
      against the theory: the epidemic threshold of heterogeneous
      mean-field theory `T_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩)`, the Erdős–Rényi giant
      component `S = 1 − e^{−cS}`, Albert–Jeong–Barabási's robustness of
      scale-free networks to failures and fragility to attacks;
    * `pagerank/2` — power iteration in binary64 in a fixed order, and
      `pagerank_program/2`, the same iteration as a vapor program (dense,
      `f32`): bit-identical on every substrate, its ranking equal to the
      host's.

  Graphs are simple and undirected unless built `directed: true`; nodes
  are `0..n−1`. Sizes up to a few thousand nodes are the intent (all
  pairs shortest paths, betweenness and the dense PageRank program are
  quadratic or worse).
  """
  alias Vapor.{CR, Program, Sampler, Tensor}
  alias Vapor.Algebra.Term, as: T

  defstruct n: 0, adj: {}, directed: false

  @type t :: %__MODULE__{n: non_neg_integer, adj: tuple, directed: boolean}

  # ---------------------------------------------------------------- build --

  @doc "A graph from an edge list (self-loops and duplicates dropped). Option `directed`."
  def from_edges(n, edges, opts \\ []) do
    directed = Keyword.get(opts, :directed, false)

    adj =
      edges
      |> Enum.reject(fn {a, b} -> a == b end)
      |> Enum.flat_map(fn {a, b} -> if directed, do: [{a, b}], else: [{a, b}, {b, a}] end)
      |> Enum.uniq()
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))

    %__MODULE__{n: n, directed: directed, adj: List.to_tuple(for(i <- 0..(n - 1)//1, do: Enum.sort(Map.get(adj, i, []))))}
  end

  @doc "Neighbours (out-neighbours if directed) of node `i`, sorted."
  def neighbours(%__MODULE__{adj: a}, i), do: elem(a, i)

  @doc "The edges `{a, b}` (a < b for undirected graphs), sorted."
  def edges(%__MODULE__{} = g) do
    for i <- 0..(g.n - 1)//1, j <- neighbours(g, i), g.directed or i < j, do: {i, j}
  end

  def degrees(%__MODULE__{} = g), do: for(i <- 0..(g.n - 1)//1, do: length(neighbours(g, i)))
  def edge_count(%__MODULE__{} = g), do: length(edges(g))

  # ----------------------------------------------------------- generators --

  @doc "G(n, p): each pair an edge with probability p (geometric skipping, Batagelj & Brandes 2005)."
  def erdos_renyi(n, p, seed) do
    lp = CR.log_f64(1.0 - p)

    {edges, _} =
      Stream.unfold({1, -1, 0}, fn {v, w, k} ->
        r = Sampler.uniform(seed, k)
        w = w + 1 + trunc(CR.log_f64(1.0 - r) / lp)
        {v, w} = advance_row(v, w)
        if v < n, do: {{v, w}, {v, w, k + 1}}, else: nil
      end)
      |> Enum.to_list()
      |> then(&{&1, nil})

    from_edges(n, edges)
  end

  defp advance_row(v, w) when w >= v, do: advance_row(v + 1, w - v)
  defp advance_row(v, w), do: {v, w}

  @doc "Barabási–Albert: each new node attaches to `m` distinct nodes chosen in proportion to degree."
  def barabasi_albert(n, m, seed) do
    # the repeated-nodes list: a node appears once per edge end
    init = for i <- 0..(m - 1), j <- (i + 1)..m//1, do: {i, j}
    targets = Enum.flat_map(init, fn {a, b} -> [a, b] end)

    {edges, _, _} =
      Enum.reduce((m + 1)..(n - 1)//1, {init, :array.from_list(targets), length(targets)}, fn v, {es, arr, len} ->
        chosen = pick_distinct(arr, len, m, seed, v, MapSet.new(), 0)
        new = Enum.map(chosen, &{&1, v})
        arr = Enum.reduce(chosen, arr, fn t, a -> a = :array.set(:array.size(a), t, a); :array.set(:array.size(a), v, a) end)
        {new ++ es, arr, len + 2 * m}
      end)

    from_edges(n, edges)
  end

  defp pick_distinct(_arr, _len, m, _seed, _v, set, _k) when map_size(set.map) == m, do: Enum.sort(MapSet.to_list(set))

  defp pick_distinct(arr, len, m, seed, v, set, k) do
    t = :array.get(trunc(Sampler.uniform(seed, v * 1000 + k) * len), arr)
    pick_distinct(arr, len, m, seed, v, MapSet.put(set, t), k + 1)
  end

  @doc "Watts–Strogatz: a ring of n nodes each joined to its k nearest (k even), each edge rewired with probability beta."
  def watts_strogatz(n, k, beta, seed) do
    ring = for i <- 0..(n - 1), j <- 1..div(k, 2), do: {i, rem(i + j, n)}

    {edges, _} =
      ring
      |> Enum.with_index()
      |> Enum.map_reduce(MapSet.new(Enum.map(ring, fn {a, b} -> {min(a, b), max(a, b)} end)), fn {{a, b}, idx}, set ->
        if Sampler.uniform(seed, 2 * idx) < beta do
          c = Enum.find(Stream.iterate(0, &(&1 + 1)) |> Stream.map(&trunc(Sampler.uniform(seed + 1, idx * 64 + &1) * n)) |> Enum.take(64),
                        fn c -> c != a and not MapSet.member?(set, {min(a, c), max(a, c)}) end)

          if c, do: {{a, c}, set |> MapSet.delete({min(a, b), max(a, b)}) |> MapSet.put({min(a, c), max(a, c)})}, else: {{a, b}, set}
        else
          {{a, b}, set}
        end
      end)

    from_edges(n, edges)
  end

  @doc "A planted partition: `groups` blocks of `size` nodes, edges with probability `p_in` inside a block, `p_out` across."
  def planted(groups, size, p_in, p_out, seed) do
    n = groups * size

    edges =
      for i <- 0..(n - 1), j <- (i + 1)..(n - 1)//1, Sampler.uniform(seed, i * n + j) < if(div(i, size) == div(j, size), do: p_in, else: p_out), do: {i, j}

    {from_edges(n, edges), for(i <- 0..(n - 1), do: div(i, size))}
  end

  # --------------------------------------------------------------- measures --

  @doc "Local clustering coefficient of every node (0 for degree < 2)."
  def clustering(%__MODULE__{} = g) do
    sets = for i <- 0..(g.n - 1)//1, do: MapSet.new(neighbours(g, i))
    st = List.to_tuple(sets)

    for i <- 0..(g.n - 1)//1 do
      ns = neighbours(g, i)
      k = length(ns)

      if k < 2,
        do: 0.0,
        else: 2.0 * Enum.sum(for(a <- ns, b <- ns, a < b, MapSet.member?(elem(st, a), b), do: 1)) / (k * (k - 1))
    end
  end

  @doc "Average clustering coefficient (Watts & Strogatz)."
  def avg_clustering(g), do: g |> clustering() |> mean()

  @doc "Degree assortativity (Newman 2002): the Pearson correlation of the degrees at either end of an edge."
  def assortativity(%__MODULE__{} = g) do
    d = List.to_tuple(degrees(g))
    pairs = for {a, b} <- edges(g), p <- [{elem(d, a), elem(d, b)}, {elem(d, b), elem(d, a)}], do: p
    {xs, ys} = Enum.unzip(pairs)
    {mx, my} = {mean(xs), mean(ys)}
    cov = Enum.zip_with(xs, ys, &((&1 - mx) * (&2 - my))) |> mean()
    cov / :math.sqrt((xs |> Enum.map(&((&1 - mx) ** 2)) |> mean()) * (ys |> Enum.map(&((&1 - my) ** 2)) |> mean()))
  end

  @doc "Connected components, largest first (lists of nodes)."
  def components(%__MODULE__{} = g, removed \\ MapSet.new()) do
    {comps, _} =
      Enum.reduce(0..(g.n - 1)//1, {[], removed}, fn i, {acc, seen} ->
        if MapSet.member?(seen, i), do: {acc, seen}, else: (({c, seen} = bfs_comp(g, [i], MapSet.put(seen, i), [i])); {[c | acc], seen})
      end)

    Enum.sort_by(comps, &(-length(&1)))
  end

  defp bfs_comp(_g, [], seen, acc), do: {acc, seen}

  defp bfs_comp(g, [v | rest], seen, acc) do
    new = Enum.reject(neighbours(g, v), &MapSet.member?(seen, &1))
    bfs_comp(g, rest ++ new, Enum.into(new, seen), new ++ acc)
  end

  @doc "The fraction of nodes in the largest component (after removing `removed`)."
  def giant(%__MODULE__{} = g, removed \\ MapSet.new()) do
    case components(g, removed) do
      [] -> 0.0
      [c | _] -> length(c) / g.n
    end
  end

  @doc "Shortest-path distances from `s` (BFS): a map node → hops."
  def distances(%__MODULE__{} = g, s), do: bfs_dist(g, [s], %{s => 0})

  defp bfs_dist(_g, [], d), do: d

  defp bfs_dist(g, frontier, d) do
    {next, d} =
      Enum.reduce(frontier, {[], d}, fn v, {nx, d} ->
        Enum.reduce(neighbours(g, v), {nx, d}, fn u, {nx, d} -> if Map.has_key?(d, u), do: {nx, d}, else: {[u | nx], Map.put(d, u, d[v] + 1)} end)
      end)

    bfs_dist(g, Enum.reverse(next), d)
  end

  @doc "Average shortest path length within the largest component (exact: a BFS from every node)."
  def avg_path_length(%__MODULE__{} = g) do
    [c | _] = components(g)
    ds = Enum.flat_map(c, fn s -> g |> distances(s) |> Map.values() |> Enum.reject(&(&1 == 0)) end)
    mean(ds)
  end

  @doc "Betweenness centrality (Brandes 2001), normalised by (n−1)(n−2)/2 for undirected graphs."
  def betweenness(%__MODULE__{} = g) do
    cb =
      Enum.reduce(0..(g.n - 1)//1, %{}, fn s, cb ->
        {order, preds, sigma} = brandes_bfs(g, s)

        {_, cb} =
          Enum.reduce(order, {%{}, cb}, fn w, {delta, cb} ->
            dw = Map.get(delta, w, 0.0)

            delta =
              Enum.reduce(Map.get(preds, w, []), delta, fn v, dl ->
                Map.update(dl, v, sigma[v] / sigma[w] * (1 + dw), &(&1 + sigma[v] / sigma[w] * (1 + dw)))
              end)

            {delta, if(w != s, do: Map.update(cb, w, dw, &(&1 + dw)), else: cb)}
          end)

        cb
      end)

    norm = if g.n > 2, do: (g.n - 1) * (g.n - 2), else: 1
    for i <- 0..(g.n - 1)//1, do: Map.get(cb, i, 0.0) / norm
  end

  # BFS from s: nodes in non-increasing distance order, predecessors, path counts
  defp brandes_bfs(g, s) do
    loop = fn loop, frontier, dist, sigma, preds, order ->
      if frontier == [] do
        {order, preds, sigma}
      else
        {next, dist, sigma, preds} =
          Enum.reduce(frontier, {[], dist, sigma, preds}, fn v, acc ->
            Enum.reduce(neighbours(g, v), acc, fn w, {nx, dist, sigma, preds} ->
              cond do
                not Map.has_key?(dist, w) -> {[w | nx], Map.put(dist, w, dist[v] + 1), Map.put(sigma, w, sigma[v]), Map.put(preds, w, [v])}
                dist[w] == dist[v] + 1 -> {nx, dist, Map.update!(sigma, w, &(&1 + sigma[v])), Map.update!(preds, w, &[v | &1])}
                true -> {nx, dist, sigma, preds}
              end
            end)
          end)

        next = Enum.reverse(next)
        loop.(loop, next, dist, sigma, preds, Enum.reverse(next) ++ order)
      end
    end

    loop.(loop, [s], %{s => 0}, %{s => 1.0}, %{}, [s])
  end

  # ---------------------------------------------------------------- pagerank --

  @doc """
  PageRank by power iteration (damping `d`, 0.85; dangling nodes spread
  uniformly), binary64, sums in node order: `[score]`. Options `iters`
  (100), `tol` (1e-12, L1).
  """
  def pagerank(%__MODULE__{} = g, opts \\ []) do
    d = Keyword.get(opts, :damping, 0.85)
    n = g.n
    out = List.to_tuple(degrees(g))
    inn = in_lists(g)
    x0 = List.duplicate(1.0 / n, n)

    Enum.reduce_while(1..Keyword.get(opts, :iters, 100), x0, fn _, x ->
      xt = List.to_tuple(x)
      dangling = Enum.sum(for i <- 0..(n - 1), elem(out, i) == 0, do: elem(xt, i))

      nx =
        for i <- 0..(n - 1) do
          (1 - d) / n + d * (dangling / n + Enum.sum(for j <- elem(inn, i), do: elem(xt, j) / elem(out, j)))
        end

      if Enum.zip_with(x, nx, &abs(&1 - &2)) |> Enum.sum() < Keyword.get(opts, :tol, 1.0e-12), do: {:halt, nx}, else: {:cont, nx}
    end)
  end

  defp in_lists(%__MODULE__{directed: false} = g), do: g.adj

  defp in_lists(%__MODULE__{} = g) do
    m = Enum.group_by(edges(g), &elem(&1, 1), &elem(&1, 0))
    List.to_tuple(for i <- 0..(g.n - 1), do: Enum.sort(Map.get(m, i, [])))
  end

  @doc """
  `iters` PageRank iterations as a vapor program: the transition matrix
  `M` (f32, `[N, N]`, N padded to 16, column-stochastic with the dangling
  columns uniform) as a constant, `x ← (1 − d)/n + d·M·x` unrolled, output
  `rank` (`f32[1, N]`). Canonical semantics: the same bits on every
  substrate.
  """
  def pagerank_program(%__MODULE__{} = g, iters \\ 50, d \\ 0.85) do
    n = g.n
    np = max(16, div(n + 15, 16) * 16)
    out = List.to_tuple(degrees(g))
    inn = in_lists(g) |> Tuple.to_list() |> Enum.map(&MapSet.new/1) |> List.to_tuple()

    m =
      for i <- 0..(np - 1), j <- 0..(np - 1) do
        cond do
          i >= n or j >= n -> 0.0
          elem(out, j) == 0 -> 1.0 / n
          MapSet.member?(elem(inn, i), j) -> 1.0 / elem(out, j)
          true -> 0.0
        end
      end

    mt = T.const(Tensor.from_list(:f32, [np, np], m))
    teleport = T.const(Tensor.from_list(:f32, [1, np], for(i <- 0..(np - 1), do: if(i < n, do: (1 - d) / n, else: 0.0))))
    x0 = T.const(Tensor.from_list(:f32, [1, np], for(i <- 0..(np - 1), do: if(i < n, do: 1.0 / n, else: 0.0))))

    {lets, x} =
      Enum.reduce(1..iters, {[], x0}, fn k, {lets, x} ->
        term = T.add(teleport, T.mul(T.linear(x, mt), T.splat(d)))
        {[{:"x#{k}", term} | lets], T.ref(:"x#{k}", term)}
      end)

    Program.new([rank: x], lets: Enum.reverse(lets))
  end

  # ------------------------------------------------------------- communities --

  @doc """
  Communities by the Louvain method (Blondel et al. 2008), deterministic:
  nodes visited in index order, each moved to the neighbouring community
  of greatest modularity gain (ties to the smallest community), sweeps
  until nothing moves; then communities become nodes of a weighted graph
  and the same again, until a level changes nothing. Returns `[label]`
  (0.. in order of first appearance). (Label propagation, tried first,
  floods a planted partition into one community without a random visiting
  order — found in 0.10.)
  """
  def communities(%__MODULE__{} = g, _opts \\ []) do
    w = for {a, b} <- edges(g), reduce: %{} do
      acc -> acc |> add_w(a, b, 1.0) |> add_w(b, a, 1.0)
    end

    w = Enum.reduce(0..(g.n - 1), w, fn i, acc -> Map.put_new(acc, i, %{}) end)
    louvain(w, Enum.to_list(0..(g.n - 1)), Map.new(0..(g.n - 1), &{&1, &1}))
  end

  defp add_w(m, a, b, x), do: Map.update(m, a, %{b => x}, &Map.update(&1, b, x, fn y -> y + x end))

  # w: node → %{neighbour → weight} (a self-loop holds twice the inside weight);
  # members: original node → current node
  defp louvain(w, original, members) do
    nodes = w |> Map.keys() |> Enum.sort()
    m2 = w |> Enum.map(fn {_, nb} -> nb |> Map.values() |> Enum.sum() end) |> Enum.sum()
    k = Map.new(w, fn {i, nb} -> {i, nb |> Map.values() |> Enum.sum()} end)
    comm0 = Map.new(nodes, &{&1, &1})
    tot0 = Map.new(nodes, &{&1, k[&1]})

    {comm, _, moved} =
      Enum.reduce_while(1..100, {comm0, tot0, false}, fn _, {comm, tot, any} ->
        {comm, tot, moved} =
          Enum.reduce(nodes, {comm, tot, false}, fn i, {comm, tot, moved} ->
            ci = comm[i]
            tot = Map.update!(tot, ci, &(&1 - k[i]))
            links = Enum.reduce(w[i], %{}, fn {j, x}, acc -> if j == i, do: acc, else: Map.update(acc, comm[j], x, &(&1 + x)) end)
            gain = fn c -> Map.get(links, c, 0.0) - Map.get(tot, c, 0.0) * k[i] / m2 end
            best = [ci | Map.keys(links)] |> Enum.uniq() |> Enum.sort() |> Enum.max_by(&gain.(&1))
            best = if gain.(best) > gain.(ci) + 1.0e-12, do: best, else: ci
            {Map.put(comm, i, best), Map.update(tot, best, k[i], &(&1 + k[i])), moved or best != ci}
          end)

        if moved, do: {:cont, {comm, tot, true}}, else: {:halt, {comm, tot, any}}
      end)

    members = Map.new(members, fn {o, c} -> {o, comm[c]} end)

    if moved do
      agg = for {i, nb} <- w, {j, x} <- nb, reduce: %{} do
        acc -> add_w(acc, comm[i], comm[j], x)
      end

      louvain(agg, original, members)
    else
      original |> Enum.map(&members[&1]) |> relabel()
    end
  end

  defp relabel(ls) do
    {out, _} = Enum.map_reduce(ls, %{}, fn l, m -> case m do %{^l => k} -> {k, m}; _ -> {map_size(m), Map.put(m, l, map_size(m))} end end)
    out
  end

  @doc "Newman's modularity Q of a partition (`[label]`)."
  def modularity(%__MODULE__{} = g, labels) do
    lt = List.to_tuple(labels)
    deg = List.to_tuple(degrees(g))
    m2 = Enum.sum(Tuple.to_list(deg))
    inside = Enum.count(edges(g), fn {a, b} -> elem(lt, a) == elem(lt, b) end) * 2
    tot = Enum.reduce(0..(g.n - 1), %{}, fn i, acc -> Map.update(acc, elem(lt, i), elem(deg, i), &(&1 + elem(deg, i))) end)
    inside / m2 - Enum.sum(for {_, s} <- tot, do: (s / m2) ** 2)
  end

  @doc "Normalised mutual information of two partitions (arithmetic normalisation)."
  def nmi(a, b) do
    n = length(a)
    pa = Enum.frequencies(a)
    pb = Enum.frequencies(b)
    pab = Enum.frequencies(Enum.zip(a, b))
    h = fn p -> -Enum.sum(for {_, c} <- p, do: c / n * CR.log_f64(c / n)) end
    mi = Enum.sum(for {{x, y}, c} <- pab, do: c / n * CR.log_f64(c * n / (pa[x] * pb[y])))
    den = (h.(pa) + h.(pb)) / 2
    if den == 0, do: 1.0, else: mi / den
  end

  # ------------------------------------------------------------ null models --

  @doc """
  The configuration-model null of `g`: `swaps` degree-preserving double
  edge swaps (a–b, c–d → a–d, c–b, refused when they would make a loop or
  a multi-edge), every draw from `(seed, k)`. Same degree of every node.
  """
  def rewire(%__MODULE__{directed: false} = g, swaps, seed) do
    es = edges(g)
    set = MapSet.new(es)
    arr = :array.from_list(es)
    m = length(es)

    {arr, _} =
      Enum.reduce(0..(swaps - 1)//1, {arr, set}, fn k, {arr, set} ->
        i = trunc(Sampler.uniform(seed, 2 * k) * m)
        j = trunc(Sampler.uniform(seed, 2 * k + 1) * m)
        {a, b} = :array.get(i, arr)
        # the second edge's orientation drawn too: always pairing the lower
        # ends with the upper ones explores only part of the null (found in
        # 0.10: an Erdős–Rényi graph scored z = 20 against its own null)
        {c, d} = case :array.get(j, arr) do {c, d} -> if Sampler.uniform(seed + 1, k) < 0.5, do: {c, d}, else: {d, c} end
        e1 = {min(a, d), max(a, d)}
        e2 = {min(c, b), max(c, b)}

        if i == j or a == d or c == b or e1 == e2 or MapSet.member?(set, e1) or MapSet.member?(set, e2) do
          {arr, set}
        else
          set = set |> MapSet.delete({a, b}) |> MapSet.delete({min(c, d), max(c, d)}) |> MapSet.put(e1) |> MapSet.put(e2)
          {:array.set(j, e2, :array.set(i, e1, arr)), set}
        end
      end)

    from_edges(g.n, :array.to_list(arr))
  end

  @doc """
  The z-score of `stat` (a function of a graph) against `samples`
  configuration-model nulls of `g` (each `10·m` swaps): `%{value, null_mean,
  null_sd, z}`.
  """
  def zscore(%__MODULE__{} = g, stat, samples, seed) do
    v = stat.(g)
    m = edge_count(g)
    nulls = for s <- 1..samples, do: stat.(rewire(g, 10 * m, seed * 1000 + s))
    mu = mean(nulls)
    sd = :math.sqrt(mean(Enum.map(nulls, &((&1 - mu) ** 2))))
    %{value: v, null_mean: mu, null_sd: sd, z: (v - mu) / max(sd, 1.0e-12)}
  end

  # -------------------------------------------------------------- power laws --

  @doc """
  The discrete power law of a sample of positive integers (degrees),
  Clauset–Shalizi–Newman: for each candidate `x_min` the MLE
  `α = 1 + n·[Σ ln(x/(x_min − ½))]⁻¹`, the `x_min` minimising the KS
  distance between the data above it and the fitted law (Hurwitz zeta
  normalisation, summed to convergence); then a goodness-of-fit p-value
  from `boot` synthetic samples (the semi-parametric bootstrap: below
  `x_min` resampled from the data, above it drawn from the fit, each refit
  in full), and the log-likelihood ratio R against a discrete exponential
  on the same tail with Vuong's normalised p-value (R > 0: the power law
  is the better of the two). Returns `%{alpha, xmin, n_tail, ks, p, lr,
  lr_p, verdict}`, the verdict `:rejected` (p < 0.1), `:power_law` (not
  rejected and significantly better than the exponential),
  `:exponential` or `:inconclusive`. Options: `boot` (100), `seed`.

  The draws are platform-independent; the statistics use the host's
  `pow` and `exp` (identical on one platform, within an ulp across
  platforms — a bootstrap comparison could flip only on an exact tie).
  """
  def power_law(xs, opts \\ []) do
    xs = xs |> Enum.filter(&(&1 > 0)) |> Enum.sort()
    fit = fit_power(xs)
    boot = Keyword.get(opts, :boot, 100)
    seed = Keyword.get(opts, :seed, 1)
    n = length(xs)
    below = Enum.filter(xs, &(&1 < fit.xmin))
    ntail = fit.n_tail
    cdf = tail_cdf(fit.alpha, fit.xmin)

    exceed =
      Enum.count(1..boot//1, fn b ->
        synth =
          for k <- 0..(n - 1) do
            if Sampler.uniform(seed * 7919 + b, 2 * k) < ntail / n or below == [],
              do: draw(cdf, Sampler.uniform(seed * 7919 + b, 2 * k + 1), fit.xmin),
              else: Enum.at(below, trunc(Sampler.uniform(seed * 7919 + b, 2 * k + 1) * length(below)))
          end

        fit_power(Enum.sort(synth)).ks >= fit.ks
      end)

    {lr, lr_p} = vs_exponential(Enum.filter(xs, &(&1 >= fit.xmin)), fit)
    p = if boot > 0, do: exceed / boot, else: nil

    # the verdict of Clauset, Shalizi & Newman: plausible only if the
    # bootstrap does not reject it (p ≥ 0.1), and preferred only if it
    # beats the alternative significantly
    verdict =
      cond do
        p != nil and p < 0.1 -> :rejected
        lr > 0 and lr_p < 0.1 -> :power_law
        lr < 0 and lr_p < 0.1 -> :exponential
        true -> :inconclusive
      end

    Map.merge(fit, %{p: p, lr: lr, lr_p: lr_p, verdict: verdict})
  end

  defp fit_power(xs) do
    cands = xs |> Enum.uniq() |> Enum.filter(fn x -> Enum.count(xs, &(&1 >= x)) >= 10 end)
    cands = if cands == [], do: [hd(xs)], else: cands

    cands
    |> Enum.map(fn xmin ->
      tail = Enum.filter(xs, &(&1 >= xmin))
      nt = length(tail)
      s = Enum.reduce(tail, 0.0, fn x, a -> a + CR.log_f64(x / (xmin - 0.5)) end)
      alpha = if s > 0, do: 1 + nt / s, else: 99.0
      %{alpha: alpha, xmin: xmin, n_tail: nt, ks: ks(tail, alpha, xmin)}
    end)
    |> Enum.min_by(&{&1.ks, &1.xmin})
  end

  # KS distance between the empirical tail and the discrete power law's CDF
  defp ks(tail, alpha, xmin) do
    nt = length(tail)
    cdf = tail_cdf(alpha, xmin)
    freq = Enum.frequencies(tail)

    {d, _} =
      freq
      |> Enum.sort()
      |> Enum.reduce({0.0, 0}, fn {x, c}, {d, cum} ->
        cum = cum + c
        {max(d, abs(cum / nt - cdf.(x))), cum}
      end)

    d
  end

  # P(X ≤ x) for the discrete power law on x ≥ xmin: 1 − ζ(α, x+1)/ζ(α, xmin)
  defp tail_cdf(alpha, xmin) do
    z0 = hurwitz(alpha, xmin)
    fn x -> 1.0 - hurwitz(alpha, x + 1) / z0 end
  end

  # Hurwitz zeta ζ(s, q) = Σ_{k≥0} (k+q)^−s, by direct sum plus the Euler–Maclaurin tail
  defp hurwitz(s, q) do
    nn = 30
    head = Enum.reduce(0..(nn - 1), 0.0, fn k, acc -> acc + :math.pow(k + q, -s) end)
    a = nn + q
    head + :math.pow(a, 1 - s) / (s - 1) + 0.5 * :math.pow(a, -s) + s * :math.pow(a, -s - 1) / 12
  end

  defp draw(cdf, u, xmin), do: draw_from(cdf, u, xmin)
  defp draw_from(cdf, u, x), do: if(cdf.(x) >= u or x > 100_000, do: x, else: draw_from(cdf, u, x + 1))

  # log-likelihood ratio of the power law against the discrete exponential
  # P(x) = (1 − e^−λ) e^−λ(x − xmin) on the tail; Vuong's p-value
  defp vs_exponential(tail, fit) do
    n = length(tail)
    m = mean(tail) - fit.xmin
    lam = CR.log_f64(1 + 1 / max(m, 1.0e-9))
    z0 = hurwitz(fit.alpha, fit.xmin)
    lp = Enum.map(tail, fn x -> -fit.alpha * CR.log_f64(x) - CR.log_f64(z0) end)
    le = Enum.map(tail, fn x -> CR.log_f64(1 - :math.exp(-lam)) - lam * (x - fit.xmin) end)
    d = Enum.zip_with(lp, le, &(&1 - &2))
    r = Enum.sum(d)
    mu = r / n
    sd = :math.sqrt(Enum.sum(Enum.map(d, &((&1 - mu) ** 2))) / n)
    p = if sd == 0, do: 0.0, else: erfc(abs(r) / (sd * :math.sqrt(2 * n)))
    {r, p}
  end

  # complementary error function (Numerical Recipes' Chebyshev fit, |rel err| < 1.2e-7)
  defp erfc(x) do
    z = abs(x)
    t = 1 / (1 + 0.5 * z)

    r =
      t * :math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
        t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))

    if x >= 0, do: r, else: 2 - r
  end

  # ---------------------------------------------------------- epidemics, robustness --

  @doc """
  A discrete-time SIR epidemic: every step each infected node infects
  each susceptible neighbour with probability `beta`, then recovers with
  probability `gamma`. Starts from `seeds` infected nodes (chosen by the
  seed). Returns `%{final_size, peak, curve: [infected per step]}` (sizes
  as fractions of n). Every draw is `(seed, step, node, neighbour)`.
  """
  def sir(%__MODULE__{} = g, {beta, gamma}, opts \\ []) do
    seed = Keyword.get(opts, :seed, 1)
    k0 = Keyword.get(opts, :seeds, 1)
    start = for(j <- 0..(k0 - 1), do: trunc(Sampler.uniform(seed, 1_000_000 + j) * g.n)) |> Enum.uniq()
    state = Map.new(start, &{&1, :i})

    loop = fn loop, state, t, curve ->
      inf = for {v, :i} <- state, do: v

      if inf == [] or t > 10_000 do
        rec = Enum.count(state, fn {_, s} -> s == :r end)
        %{final_size: rec / g.n, peak: Enum.max(curve) / g.n, curve: Enum.reverse(curve) |> Enum.map(&(&1 / g.n))}
      else
        newly =
          for v <- Enum.sort(inf), u <- neighbours(g, v), not Map.has_key?(state, u), Sampler.uniform(seed + t * 7919, v * g.n + u) < beta, uniq: true, do: u

        state = Enum.reduce(inf, state, fn v, s -> if Sampler.uniform(seed + t * 7919 + 1, v) < gamma, do: Map.put(s, v, :r), else: s end)
        state = Enum.reduce(newly, state, &Map.put(&2, &1, :i))
        loop.(loop, state, t + 1, [length(inf) | curve])
      end
    end

    loop.(loop, state, 0, [])
  end

  @doc """
  The epidemic threshold of heterogeneous mean-field theory, as a
  transmissibility: `T_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩)` — an outbreak can become
  large only when T = β/(1 − (1 − β)(1 − γ)) exceeds it.
  """
  def threshold(%__MODULE__{} = g) do
    ks = degrees(g)
    k1 = mean(ks)
    k2 = mean(Enum.map(ks, &(&1 * &1)))
    k1 / (k2 - k1)
  end

  @doc "The transmissibility of the discrete-time SIR: the probability that an infected node infects a given neighbour before recovering."
  def transmissibility(beta, gamma), do: beta / (1 - (1 - beta) * (1 - gamma))

  @doc """
  Robustness (Albert, Jeong & Barabási 2000): the giant component's
  fraction after removing a fraction `f` of the nodes, either at random
  (`:failure`, by the seed) or the highest-degree first (`:attack`).
  """
  def percolation(%__MODULE__{} = g, f, mode, seed \\ 1) do
    k = round(f * g.n)

    removed =
      case mode do
        :failure -> 0..(g.n - 1) |> Enum.to_list() |> Vapor.Modal.Rng.permute(seed) |> Enum.take(k)
        :attack -> g |> degrees() |> Enum.with_index() |> Enum.sort_by(fn {d, i} -> {-d, i} end) |> Enum.take(k) |> Enum.map(&elem(&1, 1))
      end

    giant(g, MapSet.new(removed)) * g.n / max(g.n - k, 1)
  end

  @doc "The Erdős–Rényi giant component for mean degree c: the root of S = 1 − e^{−cS} (0 for c ≤ 1)."
  def er_giant(c) when c <= 1, do: 0.0
  def er_giant(c), do: Enum.reduce(1..200, 0.5, fn _, s -> 1 - :math.exp(-c * s) end)

  defp mean([]), do: 0.0
  defp mean(xs), do: Enum.sum(xs) / length(xs)
end
