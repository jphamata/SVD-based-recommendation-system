defmodule Vapor.Discover do
  @moduledoc """
  Algorithm discovery by search, each result with a certificate that does
  not trust the search (docs/DESCOBERTA.md):

    * **sorting networks** (`network/2`) — beam search over the set of 0-1
      vectors a comparator prefix can still produce; correctness by the
      **0-1 principle** (Knuth 5.3.4: a network sorts every input iff it
      sorts all 2ⁿ zero-one inputs), checked bit-parallel (`sorts?/2`).
      Sizes and depths are compared with the known optima; the control is
      a random network pruned of every redundant comparator;
    * **matrix multiplication** (`matmul/3`) — a rank-R bilinear algorithm
      for n×n products, found by alternating least squares driven to
      integer coefficients, then **verified exactly over the integers**
      (`bilinear_ok?/2`): every coefficient of the n²×n²×n² tensor. For 2×2
      at rank 7 it rediscovers a Strassen-class algorithm; the control is
      rank 6, which no search may find — Winograd (1971) proved 7 minimal;
    * **bit-trick synthesis** (`synthesize/2`) — the shortest straight-line
      program over `add sub and or xor not neg shr1 sar` for a spec on
      machine words, by bottom-up enumeration with **observational
      equivalence over the whole input space** of 5-bit words (sound on a
      finite domain: two terms equal on every input are interchangeable),
      so the length found is **minimal for that operation set** — then
      verified exhaustively on 8-bit words and on 10⁵ random 16- and
      32-bit inputs;
    * **complexity** (`complexity/1`) — the asymptotic class of measured
      operation counts: for each of n, n log n, n², n^log₂7, n³, 2ⁿ the
      least-squares fit a·g(n) + b, ranked by relative residual (every
      class has the same two parameters, so no penalty is needed).

  Every search is a function of its seed (`Vapor.Sampler`): a discovery
  is replayable (`Vapor.Archive`).
  """
  import Bitwise
  alias Vapor.Sampler

  # ------------------------------------------------------ sorting networks --

  @doc "Known minimal sizes and depths of sorting networks (Knuth; Codish, Cruz-Filipe, Frank & Schneider-Kamp 2014–2016)."
  def known(n), do: %{size: Enum.at([0, 0, 1, 3, 5, 9, 12, 16, 19, 25, 29], n), depth: Enum.at([0, 0, 1, 3, 3, 5, 5, 6, 6, 7, 7], n)}

  @doc """
  The 0-1 principle, bit-parallel: wire i is an integer whose bit k is the
  wire's value on the k-th of the 2ⁿ zero-one inputs; a comparator (i, j)
  is `(wᵢ ∧ wⱼ, wᵢ ∨ wⱼ)`. The network sorts iff, at the end, no input has
  a 1 on a wire above a 0.
  """
  def sorts?(n, net) do
    wires = for i <- 0..(n - 1), do: Enum.reduce(0..((1 <<< n) - 1), 0, fn k, acc -> if (k >>> i &&& 1) == 1, do: acc ||| 1 <<< k, else: acc end)
    w = Enum.reduce(net, List.to_tuple(wires), fn {i, j}, t ->
      {a, b} = {elem(t, i), elem(t, j)}
      t |> put_elem(i, a &&& b) |> put_elem(j, a ||| b)
    end)

    Enum.all?(0..(n - 2), fn i -> (elem(w, i) &&& bnot(elem(w, i + 1))) == 0 end)
  end

  @doc "Depth: comparators layered as early as their wires allow."
  def depth(net) do
    {_, d} = Enum.reduce(net, {%{}, 0}, fn {i, j}, {ready, d} ->
      l = max(Map.get(ready, i, 0), Map.get(ready, j, 0)) + 1
      {ready |> Map.put(i, l) |> Map.put(j, l), max(d, l)}
    end)

    d
  end

  @doc """
  Search an n-input sorting network: `%{net, size, depth, sorts, known}`.
  Beam search (`beam:`, default 64) over comparator sequences scored by how
  many distinct 0-1 vectors remain possible (a sorted network leaves n+1);
  the first layer is fixed to the maximal matching (Parberry's lemma: some
  optimal network starts that way). Redundant comparators are pruned at
  the end. `seed:` breaks ties.
  """
  def network(n, opts \\ []) do
    beam = Keyword.get(opts, :beam, 64)
    seed = Keyword.get(opts, :seed, 1)
    first = for i <- 0..(div(n, 2) - 1), do: {2 * i, 2 * i + 1}
    all = MapSet.new(0..((1 <<< n) - 1))
    start = Enum.reduce(first, all, &apply_cmp(&2, &1))
    comps = for i <- 0..(n - 2), j <- (i + 1)..(n - 1), do: {i, j}
    goal = n + 1

    # a beam can die out (no comparator merges two vectors): reported, not looped on
    found =
      try do
        beam_search([{start, Enum.reverse(first)}], comps, goal, beam, seed, 0)
      catch
        :stuck -> :stuck
      end

    case found do
      :stuck ->
        %{n: n, net: [], size: 0, depth: 0, sorts: false, known: known(n), stuck: true}

      net ->
        net = net |> Enum.reverse() |> prune(n)
        %{n: n, net: net, size: length(net), depth: depth(net), sorts: sorts?(n, net), known: known(n)}
    end
  end

  defp beam_search(states, comps, goal, beam, seed, round) do
    case Enum.find(states, fn {set, _} -> MapSet.size(set) == goal end) do
      {_, path} ->
        path

      nil ->
        next =
          for {set, path} <- states, c <- comps, s2 = apply_cmp(set, c), MapSet.size(s2) < MapSet.size(set), do: {s2, [c | path]}

        if next == [], do: throw(:stuck)

        next =
          next
          |> Enum.uniq_by(fn {s, _} -> s end)
          |> Enum.sort_by(fn {s, path} -> {MapSet.size(s), Sampler.uniform(seed, round * 1_000_003 + :erlang.phash2(path, 1_000_000))} end)
          |> Enum.take(beam)

        beam_search(next, comps, goal, beam, seed, round + 1)
    end
  end

  # a comparator on the set of 0-1 vectors (bit i = wire i; sorted = ones on the high wires)
  defp apply_cmp(set, {i, j}) do
    MapSet.new(set, fn v -> if (v >>> i &&& 1) == 1 and (v >>> j &&& 1) == 0, do: v - (1 <<< i) + (1 <<< j), else: v end)
  end

  # drop every comparator whose removal keeps the network sorting (last first)
  defp prune(net, n) do
    Enum.reduce((length(net) - 1)..0//-1, net, fn k, acc ->
      if k < length(acc) do
        cand = List.delete_at(acc, k)
        if sorts?(n, cand), do: cand, else: acc
      else
        acc
      end
    end)
  end

  @doc "The control: random comparators until the network sorts, then every redundant one pruned."
  def random_network(n, seed) do
    comps = for i <- 0..(n - 2), j <- (i + 1)..(n - 1), do: {i, j}
    net = grow_random(n, comps, seed, 0, [])
    net = prune(net, n)
    %{n: n, net: net, size: length(net), depth: depth(net), sorts: sorts?(n, net)}
  end

  defp grow_random(n, comps, seed, k, acc) do
    if k > 0 and rem(k, n) == 0 and sorts?(n, Enum.reverse(acc)) do
      Enum.reverse(acc)
    else
      c = Enum.at(comps, trunc(Sampler.uniform(seed, k) * length(comps)))
      grow_random(n, comps, seed, k + 1, [c | acc])
    end
  end

  @doc """
  A network as a vapor program: wire inputs `w0 … w(n-1)` (f32[b]), each
  comparator a `min`/`max` pair — sorting b vectors at once, the same bits
  on every substrate.
  """
  def network_program(n, net, b) do
    alias Vapor.Algebra.Term, as: T
    wires = for i <- 0..(n - 1), do: T.input(:"w#{i}", :f32, [b])

    t = Enum.reduce(net, List.to_tuple(wires), fn {i, j}, t ->
      {x, y} = {elem(t, i), elem(t, j)}
      t |> put_elem(i, T.min(x, y)) |> put_elem(j, T.max(x, y))
    end)

    Vapor.Program.new(for(i <- 0..(n - 1), do: {:"s#{i}", elem(t, i)}))
  end

  # ------------------------------------------------- matrix multiplication --

  @doc "The n×n matrix-multiplication tensor: T[a][b][c] = 1 iff a = (i,j), b = (j,k), c = (i,k)."
  def matmul_tensor(n) do
    for i <- 0..(n - 1), j <- 0..(n - 1), k <- 0..(n - 1), into: MapSet.new(), do: {i * n + j, j * n + k, i * n + k}
  end

  @doc "Exact check over the integers: Σ_r U[a][r]·V[b][r]·W[c][r] = T[a][b][c] for every (a, b, c)."
  def bilinear_ok?(n, %{u: u, v: v, w: w}) do
    t = matmul_tensor(n)
    m = n * n
    r = length(hd(u))
    {ut, vt, wt} = {List.to_tuple(Enum.map(u, &List.to_tuple/1)), List.to_tuple(Enum.map(v, &List.to_tuple/1)), List.to_tuple(Enum.map(w, &List.to_tuple/1))}

    Enum.all?(for(a <- 0..(m - 1), b <- 0..(m - 1), c <- 0..(m - 1), do: {a, b, c}), fn {a, b, c} ->
      s = Enum.reduce(0..(r - 1), 0, fn q, acc -> acc + elem(elem(ut, a), q) * elem(elem(vt, b), q) * elem(elem(wt, c), q) end)
      s == if(MapSet.member?(t, {a, b, c}), do: 1, else: 0)
    end)
  end

  @doc """
  Search a rank-`r` bilinear algorithm for n×n products: `{:ok, %{u, v, w,
  rank, mults, adds, tries}}` with integer coefficients verified exactly,
  or `{:error, {:not_found, tries}}`. Alternating least squares from
  random starts (`tries:`, `seed:`), regularised, with a growing pull of
  every coefficient towards the nearest of −1, 0, 1, then rounding and the
  exact check.
  """
  def matmul(n, r, opts \\ []) do
    tries = Keyword.get(opts, :tries, 40)
    seed = Keyword.get(opts, :seed, 1)
    iters = Keyword.get(opts, :iters, 400)
    m = n * n
    t = matmul_tensor(n)
    dense = for a <- 0..(m - 1), do: (for b <- 0..(m - 1), do: (for c <- 0..(m - 1), do: if(MapSet.member?(t, {a, b, c}), do: 1.0, else: 0.0)))

    Enum.reduce_while(1..tries, {:error, {:not_found, tries}}, fn k, acc ->
      rnd = fn i -> (Sampler.uniform(seed * 7919 + k, i) - 0.5) * 2.0 end
      init = fn off -> for a <- 0..(m - 1), do: (for q <- 0..(r - 1), do: rnd.(off + a * r + q)) end
      {u, v, w} = als(dense, init.(0), init.(10_000), init.(20_000), m, r, iters)
      cand = %{u: round3(u), v: round3(v), w: round3(w)}

      if bilinear_ok?(n, cand) do
        nz = fn f -> f |> List.flatten() |> Enum.count(&(&1 != 0)) end
        # additions: per product, (nonzeros of its u and v) − 2; per output, (nonzeros in its row of W) − 1
        adds = Enum.sum(for q <- 0..(r - 1), do: max(col_nz(cand.u, q) - 1, 0) + max(col_nz(cand.v, q) - 1, 0)) + Enum.sum(for row <- cand.w, do: max(Enum.count(row, &(&1 != 0)) - 1, 0))
        {:halt, {:ok, Map.merge(cand, %{n: n, rank: r, mults: r, adds: adds, nonzeros: nz.(cand.u) + nz.(cand.v) + nz.(cand.w), tries: k})}}
      else
        {:cont, acc}
      end
    end)
  end

  defp col_nz(f, q), do: Enum.count(f, fn row -> Enum.at(row, q) != 0 end)
  defp round3(f), do: Enum.map(f, fn row -> Enum.map(row, fn x -> x |> round() |> max(-1) |> min(1) end) end)

  # T as [a][b][c]; mode products for the three factors
  defp als(t, u, v, w, m, r, iters) do
    tb = transpose3(t, :b)
    tc = transpose3(t, :c)

    Enum.reduce(1..iters, {u, v, w}, fn it, {u, v, w} ->
      # the pull towards integers grows over the second half
      pull = if it < div(iters, 2), do: 0.0, else: 0.02 * (it - div(iters, 2)) / max(div(iters, 2), 1) * 10
      lam = 1.0e-3
      w = solve_factor(tc, u, v, m, r, lam, pull, w)
      u = solve_factor(t, v, w, m, r, lam, pull, u)
      v = solve_factor(tb, w, u, m, r, lam, pull, v)
      {u, v, w}
    end)
  end

  # tensor indexed so that the free index is first: x[f][p][q] ≈ Σ_r A[p][r]·B[q][r]·X[f][r]
  defp transpose3(t, :b), do: for(b <- 0..(length(t) - 1), do: for(c <- 0..(length(t) - 1), do: for(a <- 0..(length(t) - 1), do: t |> Enum.at(a) |> Enum.at(b) |> Enum.at(c))))
  defp transpose3(t, :c), do: for(c <- 0..(length(t) - 1), do: for(a <- 0..(length(t) - 1), do: for(b <- 0..(length(t) - 1), do: t |> Enum.at(a) |> Enum.at(b) |> Enum.at(c))))

  defp solve_factor(x, a, b, m, r, lam, pull, old) do
    # design rows: (p, q) → a[p][r]·b[q][r]
    rows = for p <- 0..(m - 1), q <- 0..(m - 1), do: Enum.zip_with(Enum.at(a, p), Enum.at(b, q), &(&1 * &2))
    ata = for i <- 0..(r - 1), do: (for j <- 0..(r - 1), do: Enum.reduce(rows, 0.0, fn row, s -> s + Enum.at(row, i) * Enum.at(row, j) end) + if(i == j, do: lam + pull, else: 0.0))

    for {f, k} <- Enum.with_index(x) do
      tv = List.flatten(f)
      prior = Enum.at(old, k) |> Enum.map(&(pull * min(1.0, max(-1.0, Float.round(&1)))))
      atb = for i <- 0..(r - 1), do: Enum.reduce(Enum.zip(rows, tv), 0.0, fn {row, y}, s -> s + Enum.at(row, i) * y end) + Enum.at(prior, i)
      gauss(ata, atb)
    end
  end

  # Gaussian elimination with partial pivoting (small dense systems)
  defp gauss(a, b) do
    n = length(b)
    m = Enum.zip_with(a, b, fn row, y -> List.to_tuple(row ++ [y]) end) |> List.to_tuple()

    m =
      Enum.reduce(0..(n - 1), m, fn col, m ->
        piv = Enum.max_by(col..(n - 1), fn r -> abs(elem(elem(m, r), col)) end)
        m = if piv != col, do: m |> put_elem(col, elem(m, piv)) |> put_elem(piv, elem(m, col)), else: m
        prow = elem(m, col)
        d = elem(prow, col)
        d = if abs(d) < 1.0e-12, do: 1.0e-12, else: d

        Enum.reduce((col + 1)..(n - 1)//1, m, fn r, m ->
          row = elem(m, r)
          f = elem(row, col) / d
          put_elem(m, r, List.to_tuple(Enum.zip_with(Tuple.to_list(row), Tuple.to_list(prow), fn x, y -> x - f * y end)))
        end)
      end)

    Enum.reduce((n - 1)..0//-1, %{}, fn i, sol ->
      row = elem(m, i)
      s = Enum.reduce((i + 1)..(n - 1)//1, elem(row, n), fn j, s -> s - elem(row, j) * sol[j] end)
      d = elem(row, i)
      Map.put(sol, i, s / if(abs(d) < 1.0e-12, do: 1.0e-12, else: d))
    end)
    |> then(fn sol -> for i <- 0..(n - 1), do: sol[i] end)
  end

  @doc """
  Apply a bilinear algorithm to integer matrices (lists of rows) — the
  products `(u_r·a)(v_r·b)` then `c = Σ_r w_r·p_r` — exactly, with bignums.
  """
  def bilinear_apply(%{u: u, v: v, w: w}, a, b) do
    av = List.flatten(a)
    bv = List.flatten(b)
    r = length(hd(u))
    n = length(a)
    dot = fn f, x, q -> Enum.sum(Enum.zip_with(f, x, fn row, y -> Enum.at(row, q) * y end)) end
    p = for q <- 0..(r - 1), do: dot.(u, av, q) * dot.(v, bv, q)
    c = for row <- w, do: Enum.sum(Enum.zip_with(row, p, &(&1 * &2)))
    Enum.chunk_every(c, n)
  end

  @doc """
  Exact operation counts of the recursive algorithm on N = n^k matrices
  (one level: r products of half-size blocks, `adds` block additions): the
  multiplications are rᵏ — exponent log_n r (Strassen: log₂7 ≈ 2.807).
  """
  def recursive_counts(%{n: n, rank: r, adds: adds}, levels) do
    for k <- 0..levels do
      size = Integer.pow(n, k)
      {mults, add} = rec(k, n, r, adds)
      %{n: size, mults: mults, adds: add}
    end
  end

  defp rec(0, _n, _r, _a), do: {1, 0}

  defp rec(k, n, r, a) do
    {m, ad} = rec(k - 1, n, r, a)
    half = Integer.pow(n, k - 1)
    {r * m, r * ad + a * half * half}
  end

  # ------------------------------------------------------------- synthesis --

  @unary ~w(not neg shr1 sar)a
  @binary ~w(add sub and or xor)a

  @doc "The specs with names, as functions of word width: `fn w -> fn args -> value end end`."
  def specs do
    %{
      "average" => %{arity: 2, f: fn w -> fn [x, y] -> div(x + y, 2) &&& mask(w) end end, doc: "⌊(x + y)/2⌋ without overflow"},
      "average_ceil" => %{arity: 2, f: fn w -> fn [x, y] -> div(x + y + 1, 2) &&& mask(w) end end, doc: "⌈(x + y)/2⌉ without overflow"},
      "abs" => %{arity: 1, f: fn w -> fn [x] -> abs(signed(x, w)) &&& mask(w) end end, doc: "|x| in two's complement"},
      "clear_lowest_one" => %{arity: 1, f: fn w -> fn [x] -> x &&& (x - 1) &&& mask(w) end end, doc: "x with its lowest set bit cleared"},
      "lowest_one" => %{arity: 1, f: fn w -> fn [x] -> x &&& (-x &&& mask(w)) end end, doc: "the lowest set bit of x alone"}
    }
  end

  defp mask(w), do: (1 <<< w) - 1
  defp signed(x, w), do: if(x >= 1 <<< (w - 1), do: x - (1 <<< w), else: x)

  @doc """
  The shortest program for a spec (`specs/0` name, or `%{arity, f}`):
  `{:ok, %{expr, text, ops, width, verified: %{w8, w16, w32}, explored}}` or
  `{:error, {:none_up_to, max, explored}}`. Enumeration is bottom-up by
  operation count (a tree: a shared subterm counts each time it appears)
  over `width`-bit words (5 for one input, 4 for two), keeping one term
  per **function** — its values on every input of the domain: sound and
  complete there, so no shorter program exists over this operation set
  and the constant 1 on that domain. The result is then checked on 8-bit
  words exhaustively and on random 16- and 32-bit inputs.
  """
  def synthesize(spec, opts \\ []) do
    spec = if is_binary(spec), do: Map.fetch!(specs(), spec), else: spec
    max = Keyword.get(opts, :max_ops, 4)
    w = Keyword.get(opts, :width, if(spec.arity == 1, do: 5, else: 4))
    inputs = all_inputs(spec.arity, w)
    target = inputs |> Enum.map(spec.f.(w)) |> :erlang.list_to_binary()

    leaves =
      for(i <- 0..(spec.arity - 1), do: {{:var, i}, inputs |> Enum.map(&Enum.at(&1, i)) |> :erlang.list_to_binary()}) ++
        [{{:const, 1}, :binary.copy(<<1>>, length(inputs))}]

    {lv0, seen} =
      Enum.reduce(leaves, {[], %{}}, fn {e, vals}, {lv, seen} ->
        if Map.has_key?(seen, vals), do: {lv, seen}, else: {[{e, vals} | lv], Map.put(seen, vals, e)}
      end)

    case Map.get(seen, target) do
      nil -> grow(1, max, %{0 => Enum.reverse(lv0)}, seen, w, target, spec)
      e -> finish(e, 0, spec, map_size(seen), w)
    end
  end

  defp grow(k, max, _levels, seen, _w, _target, _spec) when k > max, do: {:error, {:none_up_to, max, map_size(seen)}}

  defp grow(k, max, levels, seen, w, target, spec) do
    m = mask(w)
    un = Stream.flat_map(Map.fetch!(levels, k - 1), fn {e, v} -> Stream.map(@unary, fn op -> {{op, e}, map1(v, &op1(op, &1, w, m))} end) end)

    pairs =
      Stream.flat_map(0..(k - 1), fn i ->
        j = k - 1 - i
        Stream.flat_map(Map.fetch!(levels, i), fn {ea, va} ->
          Stream.flat_map(Map.fetch!(levels, j), fn {eb, vb} ->
            # commutative ops once per unordered pair; sub both ways
            for op <- @binary, op == :sub or i < j or (i == j and ea <= eb), do: {{op, ea, eb}, map2(va, vb, &op2(op, &1, &2, m))}
          end)
        end)
      end)

    result =
      Enum.reduce_while(Stream.concat(un, pairs), {[], seen}, fn {e, vals}, {acc, seen} ->
        cond do
          vals == target -> {:halt, {:found, e, map_size(seen)}}
          Map.has_key?(seen, vals) -> {:cont, {acc, seen}}
          true -> {:cont, {[{e, vals} | acc], Map.put(seen, vals, e)}}
        end
      end)

    case result do
      {:found, e, n} -> finish(e, k, spec, n, w)
      {new, seen} -> grow(k + 1, max, Map.put(levels, k, new), seen, w, target, spec)
    end
  end

  defp map1(v, f), do: for(<<x <- v>>, into: <<>>, do: <<f.(x)>>)
  defp map2(a, b, f), do: :erlang.list_to_binary(Enum.zip_with(:binary.bin_to_list(a), :binary.bin_to_list(b), f))

  defp finish(e, k, spec, explored, w) do
    {:ok, %{expr: e, text: show(e), ops: k, width: w, explored: explored, verified: %{w8: verify(e, spec, 8, :all), w16: verify(e, spec, 16, 100_000), w32: verify(e, spec, 32, 100_000)}}}
  end

  @doc "Check a program against a spec on w-bit words: on every input (`:all`) or on `count` seeded random inputs."
  def verify(e, spec, w, how) do
    f = spec.f.(w)
    ins = if how == :all, do: all_inputs(spec.arity, w), else: (for k <- 1..how, do: (for i <- 0..(spec.arity - 1), do: trunc(Sampler.uniform(w * 31 + i, k) * (1 <<< w)) |> min(mask(w))))
    edge = for v <- [0, 1, mask(w), 1 <<< (w - 1), (1 <<< (w - 1)) - 1], do: List.duplicate(v, spec.arity)
    Enum.all?(ins ++ edge, fn args -> eval(e, args, w) == f.(args) end)
  end

  def eval({:var, i}, args, _w), do: Enum.at(args, i)
  def eval({:const, c}, _args, _w), do: c
  def eval({op, a}, args, w), do: op1(op, eval(a, args, w), w, mask(w))
  def eval({op, a, b}, args, w), do: op2(op, eval(a, args, w), eval(b, args, w), mask(w))

  defp op1(:not, x, _w, m), do: bxor(x, m)
  defp op1(:neg, x, _w, m), do: -x &&& m
  defp op1(:shr1, x, _w, _m), do: x >>> 1
  defp op1(:sar, x, w, m), do: if((x >>> (w - 1) &&& 1) == 1, do: m, else: 0)

  defp op2(:add, x, y, m), do: x + y &&& m
  defp op2(:sub, x, y, m), do: x - y &&& m
  defp op2(:and, x, y, _m), do: x &&& y
  defp op2(:or, x, y, _m), do: x ||| y
  defp op2(:xor, x, y, _m), do: bxor(x, y)

  defp all_inputs(1, w), do: for(x <- 0..mask(w), do: [x])
  defp all_inputs(2, w), do: for(x <- 0..mask(w), y <- 0..mask(w), do: [x, y])

  @doc "A program as text (x, y the inputs)."
  def show({:var, i}), do: Enum.at(["x", "y", "z"], i)
  def show({:const, c}), do: Integer.to_string(c)
  def show({:not, a}), do: "~" <> show(a)
  def show({:neg, a}), do: "-" <> show(a)
  def show({:shr1, a}), do: "(" <> show(a) <> " >> 1)"
  def show({:sar, a}), do: "sign(" <> show(a) <> ")"
  def show({op, a, b}), do: "(" <> show(a) <> " " <> %{add: "+", sub: "-", and: "&", or: "|", xor: "^"}[op] <> " " <> show(b) <> ")"

  # ------------------------------------------------------------ complexity --

  @classes [{"n", &Function.identity/1}, {"n log n", &__MODULE__.nlogn/1}, {"n^2", &__MODULE__.sq/1}, {"n^2.807", &__MODULE__.strassen/1}, {"n^3", &__MODULE__.cube/1}, {"2^n", &__MODULE__.exp2/1}]
  @doc false
  def nlogn(n), do: n * :math.log2(max(n, 2))
  @doc false
  def sq(n), do: n * n
  @doc false
  def strassen(n), do: :math.pow(n, :math.log2(7))
  @doc false
  def cube(n), do: n * n * n
  @doc false
  def exp2(n), do: :math.pow(2, n)

  @doc """
  The asymptotic class of measured costs `[{n, cost}]`: for each class g,
  the best `cost ≈ a·g(n) + b` (least squares), scored by the log of the
  relative residual; returns `%{class, fits}` (sorted). The candidates
  differ in growth, so on enough sizes the right one wins clearly.
  """
  def complexity(points) do
    fits =
      for {name, g} <- @classes do
        xs = Enum.map(points, fn {n, _} -> g.(n) end)
        ys = Enum.map(points, &elem(&1, 1))
        {a, b} = linfit(xs, ys)
        rel = Enum.zip(xs, ys) |> Enum.map(fn {x, y} -> ((a * x + b - y) / max(abs(y), 1.0e-12)) ** 2 end) |> Enum.sum()
        %{class: name, a: a, b: b, rel_err: :math.sqrt(rel / length(points))}
      end
      |> Enum.sort_by(& &1.rel_err)

    %{class: hd(fits).class, fits: fits}
  end

  defp linfit(xs, ys) do
    n = length(xs)
    {mx, my} = {Enum.sum(xs) / n, Enum.sum(ys) / n}
    sxx = Enum.reduce(xs, 0.0, &((&1 - mx) ** 2 + &2))
    sxy = Enum.zip(xs, ys) |> Enum.reduce(0.0, fn {x, y}, s -> s + (x - mx) * (y - my) end)
    a = if sxx == 0, do: 0.0, else: sxy / sxx
    {a, my - a * mx}
  end

  # --------------------------------------------------------------- replay --

  @doc false
  # A recipe comes from an archive or a request: every parameter is bounded
  # here, so replaying one costs seconds, never the machine.
  def replay("discover.sorting_network", %{"n" => n} = r) do
    with {:ok, n} <- bounded(n, 2..10), {:ok, beam} <- bounded(r["beam"] || 64, 1..256), {:ok, seed} <- bounded(r["seed"] || 1, 0..1_000_000_000) do
      {:ok, network(n, beam: beam, seed: seed)}
    end
  end

  def replay("discover.matmul", %{"n" => n, "rank" => k} = r) do
    with {:ok, n} <- bounded(n, 2..2), {:ok, k} <- bounded(k, 1..8), {:ok, tries} <- bounded(r["tries"] || 40, 1..60), {:ok, seed} <- bounded(r["seed"] || 1, 0..1_000_000_000) do
      matmul(n, k, tries: tries, seed: seed)
    end
  end

  def replay(_, _), do: {:error, :bad_recipe}

  @doc false
  def bounded(x, a..b) when is_integer(x) and x >= a and x <= b, do: {:ok, x}
  def bounded(x, a..b), do: {:error, {:out_of_bounds, x, [a, b]}}
end
