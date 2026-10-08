defmodule Vapor.Logic.Causal do
  @moduledoc """
  Causal claims decided on a causal diagram (docs/LOGIC.md §7): Pearl's
  interventional queries `P(y | do(x))` over a semi-Markovian model, that
  is, an acyclic graph of observed variables with directed edges `a -> b`
  and bidirected edges `a <-> b`, each standing for a hidden common cause.

  Three decisions, each with an object a reader can check:

    * **identifiability** by the ID algorithm (Shpitser & Pearl, 2006),
      which is *complete*: if it fails, the query is not identifiable from
      the observed joint `P(v)` in any model with this diagram. Success
      returns the **estimand**, an expression in `P(v)` only (sums,
      products, conditionals, ratios). Failure returns a **hedge**: two
      C-forests `F′ ⊂ F` sharing their roots that witness the failure,
      checked structurally by `hedge?/5`;
    * **back-door adjustment** (Pearl, 1993): `Z` admits
      `P(y | do(x)) = Σ_z P(y | x, z) P(z)` when no element of `Z`
      descends from `X` and `Z` d-separates `X` from `Y` once the edges out
      of `X` are cut. A refusal names the open path;
    * **d-separation** by the moralised ancestral graph (Lauritzen et al.,
      1990), with each bidirected edge made an explicit hidden parent.

  What these decisions do **not** do: they do not find the diagram. Every
  verdict is conditional on the diagram the person states. A causal claim
  is only as good as its assumptions, which this module makes explicit and
  never infers from data.

  Soundness is tested, not argued: estimands are evaluated in **exact
  rationals** on random structural causal models with binary variables and
  explicit hidden parents, and compared with the true intervention computed
  by cutting the model (`test/vapor/causal_test.exs`). The naive `P(y | x)`
  is the control: it disagrees wherever there is confounding.
  """

  defmodule Graph do
    @moduledoc "A semi-Markovian diagram: `nodes` (sorted), `dir` (set of {a, b} for a → b), `bi` (set of sorted {a, b})."
    defstruct nodes: [], dir: MapSet.new(), bi: MapSet.new()
  end

  @max_nodes 64

  # ------------------------------------------------------------- the graph

  @doc """
  A diagram from edges `[{:dir, a, b} | {:bi, a, b}]` and optional isolated
  nodes. Refuses cycles, self-loops and more than #{@max_nodes} nodes.
  """
  def graph(edges, extra_nodes \\ []) do
    nodes = Enum.uniq(Enum.flat_map(edges, fn {_, a, b} -> [a, b] end) ++ extra_nodes) |> Enum.sort()
    dir = for({:dir, a, b} <- edges, into: MapSet.new(), do: {a, b})
    bi = for({:bi, a, b} <- edges, into: MapSet.new(), do: if(a <= b, do: {a, b}, else: {b, a}))
    g = %Graph{nodes: nodes, dir: dir, bi: bi}

    cond do
      length(nodes) > @max_nodes -> {:error, "at most #{@max_nodes} variables"}
      Enum.any?(edges, fn {_, a, b} -> a == b end) -> {:error, "an edge from a variable to itself"}
      topo(g) == nil -> {:error, "the directed edges form a cycle: a causal diagram is acyclic"}
      true -> {:ok, g}
    end
  end

  @doc "Parents of a node."
  def parents(%Graph{dir: d}, v), do: for({a, ^v} <- d, do: a) |> Enum.sort()
  @doc "Children of a node."
  def children(%Graph{dir: d}, v), do: for({^v, b} <- d, do: b) |> Enum.sort()

  @doc "Ancestors of a set (the set included)."
  def ancestors(g, set), do: closure(g, set, &parents/2)
  @doc "Descendants of a set (the set included)."
  def descendants(g, set), do: closure(g, set, &children/2)

  defp closure(g, set, next) do
    Stream.iterate({MapSet.new(set), set}, fn {seen, frontier} ->
      new = frontier |> Enum.flat_map(&next.(g, &1)) |> Enum.reject(&MapSet.member?(seen, &1)) |> Enum.uniq()
      {MapSet.union(seen, MapSet.new(new)), new}
    end)
    |> Enum.find(fn {_, f} -> f == [] end)
    |> elem(0)
    |> MapSet.to_list()
    |> Enum.sort()
  end

  @doc "A topological order (Kahn, ties by name), or nil if the directed part has a cycle."
  def topo(%Graph{nodes: ns} = g) do
    indeg = Map.new(ns, &{&1, length(parents(g, &1))})
    go(g, Enum.sort(for({v, 0} <- indeg, do: v)), indeg, [])
  end

  defp go(%Graph{nodes: ns}, [], _indeg, acc), do: if(length(acc) == length(ns), do: Enum.reverse(acc), else: nil)

  defp go(g, [v | ready], indeg, acc) do
    {indeg, newly} =
      Enum.reduce(children(g, v), {indeg, []}, fn c, {m, nw} ->
        k = m[c] - 1
        {Map.put(m, c, k), if(k == 0, do: [c | nw], else: nw)}
      end)

    go(g, Enum.sort(ready ++ newly), indeg, [v | acc])
  end

  @doc "The subgraph induced by a set of nodes."
  def induced(%Graph{} = g, set) do
    s = MapSet.new(set)
    %Graph{nodes: Enum.filter(g.nodes, &MapSet.member?(s, &1)),
           dir: MapSet.filter(g.dir, fn {a, b} -> MapSet.member?(s, a) and MapSet.member?(s, b) end),
           bi: MapSet.filter(g.bi, fn {a, b} -> MapSet.member?(s, a) and MapSet.member?(s, b) end)}
  end

  @doc "The graph with the edges *into* `xs` removed (directed and bidirected): G with X overlined."
  def cut_incoming(%Graph{} = g, xs) do
    s = MapSet.new(xs)
    %{g | dir: MapSet.reject(g.dir, fn {_, b} -> MapSet.member?(s, b) end),
          bi: MapSet.reject(g.bi, fn {a, b} -> MapSet.member?(s, a) or MapSet.member?(s, b) end)}
  end

  @doc "The graph with the directed edges *out of* `xs` removed: G with X underlined."
  def cut_outgoing(%Graph{} = g, xs) do
    s = MapSet.new(xs)
    %{g | dir: MapSet.reject(g.dir, fn {a, _} -> MapSet.member?(s, a) end)}
  end

  @doc "The c-components (districts): classes of the bidirected connectivity, each sorted, in order of their first node."
  def c_components(%Graph{nodes: ns, bi: bi}) do
    adj = Enum.reduce(bi, Map.new(ns, &{&1, []}), fn {a, b}, m -> m |> Map.update!(a, &[b | &1]) |> Map.update!(b, &[a | &1]) end)

    {comps, _} =
      Enum.reduce(ns, {[], MapSet.new()}, fn v, {acc, seen} ->
        if MapSet.member?(seen, v) do
          {acc, seen}
        else
          comp = flood(adj, [v], MapSet.new([v]))
          {acc ++ [Enum.sort(MapSet.to_list(comp))], MapSet.union(seen, comp)}
        end
      end)

    comps
  end

  defp flood(_adj, [], seen), do: seen

  defp flood(adj, [v | rest], seen) do
    new = Enum.reject(adj[v], &MapSet.member?(seen, &1))
    flood(adj, rest ++ new, MapSet.union(seen, MapSet.new(new)))
  end

  # ------------------------------------------------------------ d-separation

  @doc """
  `true` when `xs` and `ys` are d-separated by `zs`: in the moral graph of
  the ancestral set of `xs ∪ ys ∪ zs` (bidirected edges as hidden parents),
  removing `zs` leaves no path between them.
  """
  def dsep?(%Graph{} = g, xs, ys, zs) do
    if Enum.any?(xs, &(&1 in ys)), do: false, else: dsep_path(g, xs, ys, zs) == nil
  end

  @doc "A path that d-connects `xs` and `ys` given `zs` in the moral ancestral graph (a list of nodes), or nil."
  def dsep_path(%Graph{} = g, xs, ys, zs) do
    anc = MapSet.new(ancestors(g, xs ++ ys ++ zs))
    sub = induced(g, MapSet.to_list(anc))
    # hidden parents made explicit: one node per bidirected edge
    hidden = for {a, b} <- sub.bi, do: {"⟨#{a}↔#{b}⟩", a, b}
    pars = fn v -> parents(sub, v) ++ for({h, a, b} <- hidden, v in [a, b], do: h) end
    nodes = sub.nodes ++ Enum.map(hidden, &elem(&1, 0))

    edges =
      Enum.flat_map(sub.nodes, fn v ->
        ps = pars.(v)
        # parent–child edges, and the "marriages" between parents of a common child
        for(p <- ps, do: {p, v}) ++ for(p <- ps, q <- ps, p < q, do: {p, q})
      end)

    blocked = MapSet.new(zs)
    adj = Enum.reduce(edges, Map.new(nodes, &{&1, []}), fn {a, b}, m -> m |> Map.update!(a, &[b | &1]) |> Map.update!(b, &[a | &1]) end)
    targets = MapSet.new(ys)
    bfs(adj, Enum.map(xs, &{&1, [&1]}), MapSet.new(xs), blocked, targets)
  end

  defp bfs(_adj, [], _seen, _blocked, _targets), do: nil

  defp bfs(adj, [{v, path} | rest], seen, blocked, targets) do
    if MapSet.member?(targets, v) do
      path |> Enum.reverse() |> Enum.reject(&String.starts_with?(&1, "⟨"))
    else
      next = adj |> Map.get(v, []) |> Enum.sort() |> Enum.reject(&(MapSet.member?(seen, &1) or MapSet.member?(blocked, &1)))
      bfs(adj, rest ++ Enum.map(next, &{&1, [&1 | path]}), MapSet.union(seen, MapSet.new(next)), blocked, targets)
    end
  end

  # ------------------------------------------------------------- back-door

  @doc """
  The back-door criterion for `zs` relative to `(xs, ys)`: `:ok` or
  `{:refuted, why}` (a descendant of X in Z, or the open back-door path).
  """
  def backdoor(%Graph{} = g, xs, ys, zs) do
    desc = MapSet.new(descendants(g, xs))

    case Enum.find(zs, &MapSet.member?(desc, &1)) do
      nil ->
        case dsep_path(cut_outgoing(g, xs), xs, ys, zs) do
          nil -> :ok
          path -> {:refuted, "a back-door path stays open: #{Enum.join(path, " – ")}"}
        end

      z ->
        {:refuted, "#{z} descends from #{Enum.join(xs, ", ")}: adjusting for it would block or bias the effect"}
    end
  end

  @doc "The adjustment estimand for a valid back-door set: Σ_z P(y | x, z) P(z)."
  def adjustment_estimand(xs, ys, []), do: {:p, ys, xs}
  def adjustment_estimand(xs, ys, zs), do: {:sum, zs, {:prod, [{:p, ys, xs ++ zs}, {:p, zs, []}]}}

  # -------------------------------------------------------- identification

  @doc """
  Identify `P(ys | do(xs))` on the diagram. `{:ok, estimand}` (an expression
  over the observed joint), or `{:fail, hedge}` where `hedge` is
  `%{f, f_prime, ys, xs}`: the node sets `F ⊃ F′` at which the recursion
  failed, for the sub-query `P(ys | do(xs))` it was answering there.
  """
  def identify(%Graph{} = g, ys, xs) do
    {:ok, simplify(id(Enum.sort(ys), Enum.sort(xs), {:p, g.nodes, []}, g, topo(g)))}
  catch
    {:hedge, f, f2, ys2, xs2} -> {:fail, %{f: f, f_prime: f2, ys: ys2, xs: xs2}}
  end

  # Shpitser & Pearl (2006), Figure 3; `p` is the current distribution over g's nodes as an expression
  defp id(ys, xs, p, g, order) do
    v = g.nodes
    anc_y = ancestors(g, ys)

    cond do
      # 1. no intervention: marginalise
      xs == [] ->
        marginal(p, v, ys)

      # 2. drop what is not an ancestor of Y
      length(anc_y) < length(v) ->
        id(ys, xs -- (xs -- anc_y), marginal(p, v, anc_y), induced(g, anc_y), Enum.filter(order, &(&1 in anc_y)))

      # 3. intervene also on what cannot affect Y once X is fixed
      (w = ((v -- xs) -- ancestors(cut_incoming(g, xs), ys))) != [] ->
        id(ys, Enum.sort(xs ++ w), p, g, order)

      true ->
        rest = v -- xs
        comps = c_components(induced(g, rest))

        case comps do
          # 4. several districts once X is removed: a product of their own identifications
          [_, _ | _] ->
            terms = for s <- comps, do: id(s, v -- s, p, g, order)
            sum_out(rest -- ys, {:prod, terms})

          [s] ->
            gcomps = c_components(g)

            cond do
              # 5. the whole graph is one district: the hedge
              gcomps == [v] ->
                throw({:hedge, v, s, ys, xs})

              # 6. S is a district of G: the product of its conditionals
              s in gcomps ->
                terms = for vi <- s, do: conditional(p, v, vi, predecessors(order, vi))
                sum_out(s -- ys, {:prod, terms})

              # 7. S sits inside a larger district S′: recurse on S′
              true ->
                s2 = Enum.find(gcomps, fn c -> Enum.all?(s, &(&1 in c)) end)

                p2 =
                  {:prod,
                   for vi <- s2 do
                     prev = predecessors(order, vi)
                     # condition on the predecessors inside S′ (as variables) and outside S′ (as values)
                     conditional(p, v, vi, prev)
                   end}

                id(ys, Enum.filter(xs, &(&1 in s2)), {:dist, s2, p2}, induced(g, s2), Enum.filter(order, &(&1 in s2)))
            end
        end
    end
  end

  defp predecessors(order, v), do: Enum.take_while(order, &(&1 != v))

  # Σ over v \ keep of a distribution
  defp marginal({:p, vs, []}, _v, keep), do: {:p, Enum.filter(vs, &(&1 in keep)), []}
  defp marginal({:dist, vs, e}, _v, keep), do: {:dist, Enum.filter(vs, &(&1 in keep)), sum_out(vs -- keep, e)}
  defp marginal(p, v, keep), do: sum_out(v -- keep, p)

  # P(vi | prev) from the current distribution
  defp conditional({:p, _vs, []}, _v, vi, prev), do: {:p, [vi], prev}

  defp conditional({:dist, vs, e}, _v, vi, prev) do
    prev = Enum.filter(prev, &(&1 in vs))
    num = sum_out(vs -- [vi | prev], e)
    den = sum_out(vs -- prev, e)
    {:frac, num, den}
  end

  defp sum_out([], e), do: e
  defp sum_out(vars, e), do: {:sum, Enum.sort(vars), e}

  # expressions: {:p, vars, given} | {:sum, vars, e} | {:prod, [e]} | {:frac, n, d} | {:dist, vars, e} (a distribution over vars)
  defp simplify({:dist, _vs, e}), do: simplify(e)
  defp simplify({:sum, [], e}), do: simplify(e)
  defp simplify({:sum, vs, {:sum, ws, e}}), do: simplify({:sum, Enum.sort(vs ++ ws), e})
  defp simplify({:sum, vs, e}), do: {:sum, vs, simplify(e)}
  defp simplify({:prod, [e]}), do: simplify(e)

  defp simplify({:prod, es}) do
    es |> Enum.map(&simplify/1) |> Enum.flat_map(fn {:prod, xs} -> xs; x -> [x] end) |> then(&{:prod, &1})
  end

  defp simplify({:frac, n, d}), do: {:frac, simplify(n), simplify(d)}
  defp simplify(e), do: e

  # ----------------------------------------------------------------- hedges

  @doc """
  Check a hedge for `P(ys | do(xs))` (Shpitser & Pearl, 2006): node sets
  `f2 ⊊ f`, `f` meets X and `f2` does not, each bidirected-connected in
  its induced graph, and both contain, as edge subgraphs, C-forests with
  one root set `R ⊆ An(Y)` (in G with the edges into X cut). Such forests
  exist exactly when every node of `f` reaches `R` by a directed path
  inside `f`, and every node of `f2` by one inside `f2`, with `R` taken
  as `f2 ∩ An(Y)`.
  """
  def hedge?(%Graph{} = g, ys, xs, f, f2) do
    c_comp? = fn set -> set != [] and length(c_components(induced(g, set))) == 1 end
    r = Enum.filter(f2, &(&1 in ancestors(cut_incoming(g, xs), ys)))

    reaches? = fn set ->
      sub = induced(g, set)
      Enum.all?(set, fn v -> Enum.any?(descendants(sub, [v]), &(&1 in r)) end)
    end

    Enum.all?(f2, &(&1 in f)) and length(f2) < length(f) and Enum.any?(f, &(&1 in xs)) and not Enum.any?(f2, &(&1 in xs)) and
      c_comp?.(f) and c_comp?.(f2) and r != [] and reaches?.(f) and reaches?.(f2)
  end

  # ------------------------------------------------------------- presentation

  @doc "An estimand as text: `Σ_{w} P(y | x, w) P(w)`."
  def to_text({:p, vs, []}), do: "P(#{Enum.join(vs, ", ")})"
  def to_text({:p, vs, given}), do: "P(#{Enum.join(vs, ", ")} | #{Enum.join(given, ", ")})"
  def to_text({:sum, vs, e}), do: "Σ_{#{Enum.join(vs, ", ")}} " <> paren(e)
  def to_text({:prod, es}), do: Enum.map_join(es, " ", &paren/1)
  def to_text({:frac, n, d}), do: "[#{to_text(n)}] / [#{to_text(d)}]"

  defp paren({:sum, _, _} = e), do: "(" <> to_text(e) <> ")"
  defp paren(e), do: to_text(e)

  @doc """
  The free variables of an estimand: those not summed out. Beyond the
  query's own `y` and `x`, an estimand may keep others free (the napkin's
  `z`): its value is then the same at every value of them, and a check
  must hold at each.
  """
  def free_vars({:p, vs, given}), do: Enum.sort(Enum.uniq(vs ++ given))
  def free_vars({:sum, vs, e}), do: free_vars(e) -- vs
  def free_vars({:prod, es}), do: es |> Enum.flat_map(&free_vars/1) |> Enum.uniq() |> Enum.sort()
  def free_vars({:frac, n, d}), do: Enum.sort(Enum.uniq(free_vars(n) ++ free_vars(d)))

  # ----------------------------------------------------------- evaluation

  @doc """
  Evaluate an estimand on a joint distribution over binary observed
  variables (`joint`: a map from `%{var => 0 | 1}` assignments of all
  observed variables to rationals `{n, d}`), at `point`, which must give a
  value to every variable of `free_vars/1`. Exact; used to test soundness.
  """
  def eval(e, joint, point) do
    missing = free_vars(e) -- Map.keys(point)
    if missing != [], do: raise(ArgumentError, "the point leaves #{Enum.join(missing, ", ")} unassigned")
    vars = joint |> Map.keys() |> hd() |> Map.keys()
    ev(e, joint, point, vars)
  end

  alias Vapor.Logic.LP

  defp ev({:p, vs, given}, joint, pt, _vars) do
    num = marg(joint, Map.take(pt, vs ++ given))
    den = marg(joint, Map.take(pt, given))
    if LP.qzero?(den), do: {0, 1}, else: LP.qdiv(num, den)
  end

  defp ev({:sum, vs, e}, joint, pt, vars) do
    Enum.reduce(assignments(vs), {0, 1}, fn a, s -> LP.qadd(s, ev(e, joint, Map.merge(pt, a), vars)) end)
  end

  defp ev({:prod, es}, joint, pt, vars), do: Enum.reduce(es, {1, 1}, fn e, s -> LP.qmul(s, ev(e, joint, pt, vars)) end)

  defp ev({:frac, n, d}, joint, pt, vars) do
    den = ev(d, joint, pt, vars)
    if LP.qzero?(den), do: {0, 1}, else: LP.qdiv(ev(n, joint, pt, vars), den)
  end

  defp marg(joint, fixed) do
    Enum.reduce(joint, {0, 1}, fn {a, pr}, s -> if Enum.all?(fixed, fn {k, v} -> a[k] == v end), do: LP.qadd(s, pr), else: s end)
  end

  @doc "Every 0/1 assignment of a list of variables."
  def assignments(vs), do: Enum.reduce(vs, [%{}], fn v, acc -> for a <- acc, b <- [0, 1], do: Map.put(a, v, b) end)

  # ------------------------------------------------------------ text form

  @doc """
  The desk's text form:

      causal
      x -> m
      m -> y
      x <-> y
      identify y | do(x)
      backdoor x -> y | z
      dsep a, b | c

  Returns `{:ok, %{graph, queries}}`.
  """
  def parse(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#") |> hd() |> String.trim())) |> Enum.reject(&(&1 in ["", "causal", "causal:"]))
    names = fn s -> s |> String.split(~r/[\s,]+/, trim: true) end

    {edges, queries, errors} =
      Enum.reduce(lines, {[], [], []}, fn l, {es, qs, errs} ->
        cond do
          m = Regex.run(~r/^identify\s+(.+?)\s*\|\s*do\((.*)\)$/i, l) -> {es, qs ++ [{:identify, names.(Enum.at(m, 1)), names.(Enum.at(m, 2))}], errs}
          m = Regex.run(~r/^backdoor\s+(.+?)\s*->\s*(.+?)\s*\|\s*(.*)$/i, l) -> {es, qs ++ [{:backdoor, names.(Enum.at(m, 1)), names.(Enum.at(m, 2)), names.(Enum.at(m, 3))}], errs}
          m = Regex.run(~r/^dsep\s+(.+?)\s*;\s*(.+?)(?:\s*\|\s*(.*))?$/i, l) -> {es, qs ++ [{:dsep, names.(Enum.at(m, 1)), names.(Enum.at(m, 2)), names.(Enum.at(m, 3) || "")}], errs}
          m = Regex.run(~r/^([A-Za-z_][\w]*)\s*<->\s*([A-Za-z_][\w]*)$/, l) -> {es ++ [{:bi, Enum.at(m, 1), Enum.at(m, 2)}], qs, errs}
          m = Regex.run(~r/^([A-Za-z_][\w]*)\s*->\s*([A-Za-z_][\w]*)$/, l) -> {es ++ [{:dir, Enum.at(m, 1), Enum.at(m, 2)}], qs, errs}
          m = Regex.run(~r/^node\s+(.+)$/i, l) -> {es ++ Enum.map(names.(Enum.at(m, 1)), &{:node, &1, &1}), qs, errs}
          true -> {es, qs, errs ++ [l]}
        end
      end)

    {isolated, edges} = Enum.split_with(edges, &match?({:node, _, _}, &1))

    with [] <- errors || [],
         true <- queries != [] || {:error, "no query: add `identify y | do(x)`, `backdoor x -> y | z` or `dsep a; b | c`"},
         {:ok, g} <- graph(edges, Enum.map(isolated, &elem(&1, 1))) do
      unknown = queries |> Enum.flat_map(fn q -> q |> Tuple.to_list() |> tl() |> List.flatten() end) |> Enum.reject(&(&1 in g.nodes))
      if unknown == [], do: {:ok, %{graph: g, queries: queries}}, else: {:error, "not in the diagram: #{Enum.join(Enum.uniq(unknown), ", ")}"}
    else
      [bad | _] when is_binary(bad) -> {:error, "cannot read #{inspect(bad)} (edges a -> b, a <-> b; queries identify / backdoor / dsep)"}
      {:error, _} = e -> e
    end
  end

  @doc "Decide every query of the text form; each verdict with its object."
  def run(text) do
    with {:ok, %{graph: g, queries: qs}} <- parse(text) do
      {:ok, %{kind: "causal", variables: g.nodes, results: Enum.map(qs, &decide(g, &1))}}
    end
  end

  @doc false
  def decide(g, {:identify, ys, xs}) do
    case identify(g, ys, xs) do
      {:ok, e} -> %{query: "P(#{Enum.join(ys, ", ")} | do(#{Enum.join(xs, ", ")}))", verdict: "identifiable", estimand: to_text(e)}
      {:fail, h} -> %{query: "P(#{Enum.join(ys, ", ")} | do(#{Enum.join(xs, ", ")}))", verdict: "not identifiable", hedge: h, hedge_checked: hedge?(g, h.ys, h.xs, h.f, h.f_prime)}
    end
  end

  def decide(g, {:backdoor, xs, ys, zs}) do
    case backdoor(g, xs, ys, zs) do
      :ok -> %{query: "back-door {#{Enum.join(zs, ", ")}} for #{Enum.join(xs, ", ")} → #{Enum.join(ys, ", ")}", verdict: "valid", estimand: to_text(adjustment_estimand(xs, ys, zs))}
      {:refuted, why} -> %{query: "back-door {#{Enum.join(zs, ", ")}} for #{Enum.join(xs, ", ")} → #{Enum.join(ys, ", ")}", verdict: "invalid", reason: why}
    end
  end

  def decide(g, {:dsep, xs, ys, zs}) do
    case dsep_path(g, xs, ys, zs) do
      nil -> %{query: "#{Enum.join(xs, ", ")} ⫫ #{Enum.join(ys, ", ")} | #{Enum.join(zs, ", ")}", verdict: "d-separated"}
      path -> %{query: "#{Enum.join(xs, ", ")} ⫫ #{Enum.join(ys, ", ")} | #{Enum.join(zs, ", ")}", verdict: "d-connected", path: Enum.join(path, " – ")}
    end
  end
end
