defmodule Vapor.Athanor.Space do
  @moduledoc """
  The search spaces of the Athanor (docs/ATHANOR.md §2). A space knows how
  to **sample** a candidate, **mutate** and **cross** candidates, **enumerate**
  itself in a fixed order when it is finite, **check** that a value proposed
  from outside (a person, a model) belongs to it, and put a candidate in a
  **canonical** form so that equal candidates are evaluated once.

  | constructor | candidates | size |
  |---|---|---|
  | `bits(n)` | lists of n zeros and ones | 2ⁿ |
  | `ints(n, lo, hi)` | lists of n integers in [lo, hi] | (hi−lo+1)ⁿ |
  | `reals(n, lo, hi)` | lists of n floats in [lo, hi] | ∞ |
  | `perm(n)` · `perm(xs)` | orderings of 0…n−1 (or of xs) | n! |
  | `subset(xs, k)` | sorted k-element subsets of xs | C(\\|xs\\|, k) |
  | `subsets(xs)` | sorted subsets of xs, any size | 2^\\|xs\\| |
  | `seq(n, alphabet)` | lists of n symbols | \\|A\\|ⁿ |
  | `graph(n)` | simple graphs on n vertices, as sorted edge lists | 2^(n(n−1)/2) |
  | `partition(n, k)` | labels 0…k−1 for n items, canonically relabelled | Σ S(n, j) |
  | `program(vars, ops, leaves, max_size)` | expressions over vars, given to the objective as a function | ∞ (enumerable by size) |

  Randomness is an explicit `:rand` state (`exsss`): a run is a function
  of its seed.
  """
  alias Vapor.Alembic

  defstruct kind: nil, n: 0, lo: 0, hi: 0, items: [], k: 0, vars: [], ops: [], leaves: [], max_size: 9, size: :infinite, text: ""

  @unary ~w(neg sin cos exp log sqrt abs sq tanh inv)
  @binary ~w(+ - * / ^ min max)

  # ============================================================ building

  @doc "A space from a reserved `space = kind(args…)` definition, its arguments evaluated in the program."
  def from_ast(prog, {:call, {:var, kind, _}, args, _}) do
    vals =
      Enum.reduce_while(args, {:ok, []}, fn a, {:ok, acc} ->
        case Alembic.eval_ast(prog, a) do
          {:ok, v} -> {:cont, {:ok, acc ++ [v]}}
          {:error, m} -> {:halt, {:error, "space: #{m}"}}
        end
      end)

    with {:ok, vs} <- vals, do: build(kind, vs)
  end

  def from_ast(_prog, _), do: {:error, "space must be one of: bits(n), ints(n, lo, hi), reals(n, lo, hi), perm(n), subset(xs, k), subsets(xs), seq(n, alphabet), graph(n), partition(n, k), program(vars, ops, leaves, max_size)"}

  @doc "Build a space from its kind and argument values."
  def build("bits", [n]) when is_integer(n) and n in 1..4096, do: ok(%__MODULE__{kind: :bits, n: n, size: Integer.pow(2, n)}, "bits(#{n})")
  def build("ints", [n, lo, hi]) when is_integer(n) and n in 1..4096 and is_integer(lo) and is_integer(hi) and hi >= lo,
    do: ok(%__MODULE__{kind: :ints, n: n, lo: lo, hi: hi, size: Integer.pow(hi - lo + 1, n)}, "ints(#{n}, #{lo}, #{hi})")
  def build("reals", [n, lo, hi]) when is_integer(n) and n in 1..1000 and is_number(lo) and is_number(hi) and hi > lo,
    do: ok(%__MODULE__{kind: :reals, n: n, lo: lo * 1.0, hi: hi * 1.0}, "reals(#{n}, #{lo}, #{hi})")
  def build("perm", [n]) when is_integer(n) and n in 1..4096, do: build("perm", [Enum.to_list(0..(n - 1))])
  def build("perm", [xs]) when is_list(xs) and length(xs) in 1..4096,
    do: ok(%__MODULE__{kind: :perm, n: length(xs), items: xs, size: fact(length(xs))}, "perm(#{length(xs)})")
  def build("subset", [xs, k]) when is_list(xs) and is_integer(k) and k >= 0 and k <= length(xs) and length(xs) <= 100_000 do
    xs = Enum.uniq(xs) |> Enum.sort()
    ok(%__MODULE__{kind: :subset, items: xs, n: length(xs), k: k, size: binom(length(xs), k)}, "subset(#{length(xs)} items, #{k})")
  end
  def build("subsets", [xs]) when is_list(xs) and length(xs) in 1..4096 do
    xs = Enum.uniq(xs) |> Enum.sort()
    ok(%__MODULE__{kind: :subsets, items: xs, n: length(xs), size: Integer.pow(2, length(xs))}, "subsets(#{length(xs)} items)")
  end
  def build("seq", [n, alpha]) when is_integer(n) and n in 1..4096 and is_list(alpha) and alpha != [],
    do: ok(%__MODULE__{kind: :seq, n: n, items: Enum.uniq(alpha), size: Integer.pow(length(Enum.uniq(alpha)), n)}, "seq(#{n}, #{length(Enum.uniq(alpha))} symbols)")
  def build("seq", [n, s]) when is_binary(s), do: build("seq", [n, String.graphemes(s)])
  def build("graph", [n]) when is_integer(n) and n in 2..200,
    do: ok(%__MODULE__{kind: :graph, n: n, items: for(i <- 0..(n - 2), j <- (i + 1)..(n - 1), do: {i, j}), size: Integer.pow(2, div(n * (n - 1), 2))}, "graph(#{n})")
  def build("partition", [n, k]) when is_integer(n) and n in 1..4096 and is_integer(k) and k in 1..n,
    do: ok(%__MODULE__{kind: :partition, n: n, k: k, size: Enum.reduce(1..k, 0, &(&2 + stirling2(n, &1)))}, "partition(#{n}, #{k})")
  def build("program", [vars, ops, leaves, max_size]) when is_list(vars) and is_list(ops) and is_list(leaves) and is_integer(max_size) and max_size in 1..60 do
    bad = Enum.reject(ops, &(&1 in @unary or &1 in @binary))
    cond do
      vars == [] or not Enum.all?(vars, &is_binary/1) -> {:error, "program: vars must be a non-empty list of names, like [\"x\"]"}
      bad != [] -> {:error, "program: unknown operators #{inspect(bad)} (known: #{Enum.join(@unary ++ @binary, " ")})"}
      not Enum.all?(leaves, &(is_number(&1) or (is_binary(&1) and &1 in vars))) -> {:error, "program: leaves are numbers or the vars"}
      true ->
        leaves = Enum.uniq(vars ++ leaves)
        size = program_count(length(leaves), Enum.count(ops, &(&1 in @unary)), Enum.count(ops, &(&1 in @binary)), max_size)
        ok(%__MODULE__{kind: :program, vars: vars, ops: ops, leaves: leaves, max_size: max_size, size: size}, "program(#{Enum.join(vars, ", ")}; #{Enum.join(ops, " ")}; size ≤ #{max_size})")
    end
  end
  def build("program", [vars, ops, leaves]), do: build("program", [vars, ops, leaves, 9])
  def build("program", [vars, ops]), do: build("program", [vars, ops, [1, 2], 9])
  def build(kind, args), do: {:error, "space #{kind}(#{Enum.map_join(args, ", ", &Alembic.show/1)}) is not one I know or its arguments are out of range"}

  defp ok(s, text), do: {:ok, %{s | text: text}}

  def unary_ops, do: @unary
  def binary_ops, do: @binary

  # ============================================================ sampling

  defp u(rng), do: :rand.uniform_s(rng)
  defp ui(n, rng), do: :rand.uniform_s(n, rng)

  defp pick(xs, rng) do
    {i, rng} = ui(length(xs), rng)
    {Enum.at(xs, i - 1), rng}
  end

  @doc "A uniformly sampled candidate."
  def sample(%{kind: :bits, n: n}, rng), do: map_n(n, rng, fn r -> ui(2, r) |> then(fn {b, r} -> {b - 1, r} end) end)
  def sample(%{kind: :ints, n: n, lo: lo, hi: hi}, rng), do: map_n(n, rng, fn r -> ui(hi - lo + 1, r) |> then(fn {v, r} -> {lo + v - 1, r} end) end)
  def sample(%{kind: :reals, n: n, lo: lo, hi: hi}, rng), do: map_n(n, rng, fn r -> u(r) |> then(fn {v, r} -> {lo + v * (hi - lo), r} end) end)
  def sample(%{kind: :perm, items: xs}, rng), do: shuffle(xs, rng)
  def sample(%{kind: :subset, items: xs, k: k}, rng), do: shuffle(xs, rng) |> then(fn {s, r} -> {s |> Enum.take(k) |> Enum.sort(), r} end)
  def sample(%{kind: :subsets, items: xs}, rng) do
    {keep, rng} = map_n(length(xs), rng, fn r -> ui(2, r) end)
    {xs |> Enum.zip(keep) |> Enum.filter(fn {_, b} -> b == 2 end) |> Enum.map(&elem(&1, 0)), rng}
  end
  def sample(%{kind: :seq, n: n, items: a}, rng), do: map_n(n, rng, &pick(a, &1))
  def sample(%{kind: :graph, items: es}, rng) do
    {keep, rng} = map_n(length(es), rng, fn r -> ui(2, r) end)
    {es |> Enum.zip(keep) |> Enum.filter(fn {_, b} -> b == 2 end) |> Enum.map(&elem(&1, 0)), rng}
  end
  def sample(%{kind: :partition, n: n, k: k}, rng) do
    {ls, rng} = map_n(n, rng, fn r -> ui(k, r) |> then(fn {v, r} -> {v - 1, r} end) end)
    {relabel(ls), rng}
  end
  def sample(%{kind: :program} = s, rng) do
    {size, rng} = ui(s.max_size, rng)
    random_tree(s, size, rng)
  end

  defp map_n(n, rng, f) do
    {xs, rng} = Enum.reduce(1..n, {[], rng}, fn _, {acc, r} -> {x, r} = f.(r); {[x | acc], r} end)
    {Enum.reverse(xs), rng}
  end

  defp shuffle(xs, rng) do
    {keyed, rng} = Enum.reduce(xs, {[], rng}, fn x, {acc, r} -> {v, r} = u(r); {[{v, x} | acc], r} end)
    {keyed |> Enum.sort() |> Enum.map(&elem(&1, 1)), rng}
  end

  # ============================================================ moves

  @doc "A neighbouring candidate (one local move)."
  def mutate(%{kind: :bits, n: n}, x, rng) do
    {flips, rng} = ui(min(3, n), rng)
    Enum.reduce(1..flips, {x, rng}, fn _, {y, r} -> {i, r} = ui(n, r); {List.update_at(y, i - 1, &(1 - &1)), r} end)
  end

  def mutate(%{kind: :ints, n: n, lo: lo, hi: hi}, x, rng) do
    {i, rng} = ui(n, rng)
    {how, rng} = u(rng)
    {v, rng} =
      if how < 0.7 do
        span = max(div(hi - lo, 8), 1)
        {d, rng} = ui(2 * span, rng)
        d = if d <= span, do: -d, else: d - span
        {Enum.at(x, i - 1) + d, rng}
      else
        {v, rng} = ui(hi - lo + 1, rng)
        {lo + v - 1, rng}
      end
    {List.replace_at(x, i - 1, v |> max(lo) |> min(hi)), rng}
  end

  def mutate(%{kind: :reals, n: n, lo: lo, hi: hi}, x, rng) do
    {all?, rng} = u(rng)
    {scale, rng} = u(rng)
    sigma = (hi - lo) * :math.pow(10, -3 * scale)
    if all? < 0.3 do
      map_n(n, rng, fn r -> {z, r} = :rand.normal_s(r); {z, r} end)
      |> then(fn {zs, r} -> {Enum.zip_with(x, zs, &clampf(&1 + sigma * &2, lo, hi)), r} end)
    else
      {i, rng} = ui(n, rng)
      {z, rng} = :rand.normal_s(rng)
      {List.update_at(x, i - 1, &clampf(&1 + sigma * z, lo, hi)), rng}
    end
  end

  def mutate(%{kind: :perm, n: n}, x, rng) when n < 2, do: {x, rng}

  def mutate(%{kind: :perm, n: n}, x, rng) do
    {how, rng} = ui(3, rng)
    {i, rng} = ui(n, rng)
    {j, rng} = ui(n, rng)
    {a, b} = {min(i, j) - 1, max(i, j) - 1}
    case how do
      1 -> {x |> List.replace_at(a, Enum.at(x, b)) |> List.replace_at(b, Enum.at(x, a)), rng}
      2 -> {Enum.slice(x, 0, a) ++ Enum.reverse(Enum.slice(x, a, b - a + 1)) ++ Enum.drop(x, b + 1), rng}
      3 -> (v = Enum.at(x, i - 1); {List.insert_at(List.delete_at(x, i - 1), j - 1, v), rng})
    end
  end

  def mutate(%{kind: :subset, items: xs, k: k}, x, rng) when k == 0 or k == length(xs), do: {x, rng}

  def mutate(%{kind: :subset, items: xs}, x, rng) do
    out = xs -- x
    {i, rng} = ui(length(x), rng)
    {v, rng} = pick(out, rng)
    {x |> List.replace_at(i - 1, v) |> Enum.sort(), rng}
  end

  def mutate(%{kind: :subsets, items: xs}, x, rng) do
    {v, rng} = pick(xs, rng)
    {if(v in x, do: List.delete(x, v), else: Enum.sort([v | x])), rng}
  end

  def mutate(%{kind: :seq, n: n, items: a}, x, rng) do
    {i, rng} = ui(n, rng)
    {v, rng} = pick(a, rng)
    {List.replace_at(x, i - 1, v), rng}
  end

  def mutate(%{kind: :graph, items: es}, x, rng) do
    {e, rng} = pick(es, rng)
    {if(e in x, do: List.delete(x, e), else: Enum.sort([e | x])), rng}
  end

  def mutate(%{kind: :partition, n: n, k: k}, x, rng) do
    {i, rng} = ui(n, rng)
    {g, rng} = ui(k, rng)
    {relabel(List.replace_at(x, i - 1, g - 1)), rng}
  end

  def mutate(%{kind: :program} = s, x, rng) do
    {how, rng} = u(rng)
    paths = tree_paths(x)
    {p, rng} = pick(paths, rng)
    cond do
      how < 0.5 ->
        budget = max(s.max_size - tree_size(x) + tree_size(get_at(x, p)), 1)
        {sz, rng} = ui(min(budget, 5), rng)
        {t, rng} = random_tree(s, sz, rng)
        {put_at(x, p, t), rng}

      how < 0.85 ->
        {t, rng} = point_mutation(s, get_at(x, p), rng)
        {put_at(x, p, t), rng}

      true ->
        # hoist: a subtree replaces the whole
        {get_at(x, p), rng}
    end
  end

  @doc "A child of two parents."
  def cross(%{kind: k}, a, b, rng) when k in [:bits, :ints, :reals, :seq] do
    {mask, rng} = map_n(length(a), rng, fn r -> ui(2, r) end)
    {Enum.zip_with([a, b, mask], fn [x, y, m] -> if m == 1, do: x, else: y end), rng}
  end

  def cross(%{kind: :perm, n: n}, a, b, rng) do
    # order crossover (OX): a slice of a, the rest in b's order
    {i, rng} = ui(n, rng)
    {j, rng} = ui(n, rng)
    {lo, hi} = {min(i, j) - 1, max(i, j) - 1}
    slice = Enum.slice(a, lo, hi - lo + 1)
    rest = Enum.reject(b, &(&1 in slice))
    {Enum.take(rest, lo) ++ slice ++ Enum.drop(rest, lo), rng}
  end

  def cross(%{kind: :subset, k: k}, a, b, rng) do
    {s, rng} = shuffle(Enum.uniq(a ++ b), rng)
    {s |> Enum.take(k) |> Enum.sort(), rng}
  end

  def cross(%{kind: kind}, a, b, rng) when kind in [:subsets, :graph] do
    both = MapSet.intersection(MapSet.new(a), MapSet.new(b))
    either = Enum.uniq(a ++ b) |> Enum.reject(&MapSet.member?(both, &1))
    {keep, rng} = map_n(max(length(either), 1), rng, fn r -> ui(2, r) end)
    picked = either |> Enum.zip(keep) |> Enum.filter(fn {_, m} -> m == 1 end) |> Enum.map(&elem(&1, 0))
    {Enum.sort(MapSet.to_list(both) ++ picked), rng}
  end

  def cross(%{kind: :partition}, a, b, rng) do
    {mask, rng} = map_n(length(a), rng, fn r -> ui(2, r) end)
    {relabel(Enum.zip_with([a, b, mask], fn [x, y, m] -> if m == 1, do: x, else: y end)), rng}
  end

  def cross(%{kind: :program} = s, a, b, rng) do
    {pa, rng} = pick(tree_paths(a), rng)
    {pb, rng} = pick(tree_paths(b), rng)
    child = put_at(a, pa, get_at(b, pb))
    if tree_size(child) <= s.max_size, do: {child, rng}, else: {a, rng}
  end

  defp clampf(v, lo, hi), do: v |> max(lo) |> min(hi)

  # ============================================================ canon / check

  @doc "The canonical form of a candidate (equal candidates, equal forms)."
  def canon(%{kind: k}, x) when k in [:subset, :subsets, :graph], do: Enum.sort(x)
  def canon(%{kind: :partition}, x), do: relabel(x)
  def canon(%{kind: :reals}, x), do: Enum.map(x, &(&1 * 1.0))
  def canon(%{kind: :program}, x), do: simplify_tree(x)
  def canon(_, x), do: x

  @doc "Does a value proposed from outside belong to the space? `{:ok, canonical}` or `{:error, why}`."
  def check(%{kind: :bits, n: n} = s, x) when is_list(x), do: (if length(x) == n and Enum.all?(x, &(&1 in [0, 1])), do: {:ok, x}, else: no(s))
  def check(%{kind: :ints, n: n, lo: lo, hi: hi} = s, x) when is_list(x), do: (if length(x) == n and Enum.all?(x, &(is_integer(&1) and &1 >= lo and &1 <= hi)), do: {:ok, x}, else: no(s))
  def check(%{kind: :reals, n: n, lo: lo, hi: hi} = s, x) when is_list(x), do: (if length(x) == n and Enum.all?(x, &(is_number(&1) and &1 >= lo and &1 <= hi)), do: {:ok, canon(s, x)}, else: no(s))
  def check(%{kind: :perm, items: xs} = s, x) when is_list(x), do: (if Enum.sort(x) == Enum.sort(xs), do: {:ok, x}, else: no(s))
  def check(%{kind: :subset, items: xs, k: k} = s, x) when is_list(x), do: (if length(Enum.uniq(x)) == k and length(x) == k and Enum.all?(x, &(&1 in xs)), do: {:ok, Enum.sort(x)}, else: no(s))
  def check(%{kind: :subsets, items: xs} = s, x) when is_list(x), do: (if length(Enum.uniq(x)) == length(x) and Enum.all?(x, &(&1 in xs)), do: {:ok, Enum.sort(x)}, else: no(s))
  def check(%{kind: :seq, n: n, items: a} = s, x) when is_list(x), do: (if length(x) == n and Enum.all?(x, &(&1 in a)), do: {:ok, x}, else: no(s))
  def check(%{kind: :graph, items: es} = s, x) when is_list(x) do
    x = Enum.map(x, fn [a, b] -> {min(a, b), max(a, b)}; {a, b} -> {min(a, b), max(a, b)}; e -> e end)
    if Enum.all?(x, &(&1 in es)) and length(Enum.uniq(x)) == length(x), do: {:ok, Enum.sort(x)}, else: no(s)
  end
  def check(%{kind: :partition, n: n, k: k} = s, x) when is_list(x), do: (if length(x) == n and Enum.all?(x, &(is_integer(&1) and &1 >= 0 and &1 < k)), do: {:ok, relabel(x)}, else: no(s))
  def check(%{kind: :program} = s, x) when is_binary(x), do: parse_program(s, x)
  def check(%{kind: :program} = s, x) do
    if valid_tree?(s, x) and tree_size(x) <= s.max_size, do: {:ok, canon(s, x)}, else: no(s)
  end
  def check(s, _), do: no(s)

  defp no(s), do: {:error, "not a member of #{s.text}"}

  @doc "Relabel a partition so groups are numbered by first appearance."
  def relabel(ls) do
    {out, _} = Enum.map_reduce(ls, %{}, fn l, m ->
      case Map.fetch(m, l) do
        {:ok, v} -> {v, m}
        :error -> v = map_size(m); {v, Map.put(m, l, v)}
      end
    end)
    out
  end

  # ============================================================ enumeration

  @doc "Every candidate of a finite space, in a fixed order (a stream). Programs: by size, up to max_size."
  def enumerate(%{kind: :bits, n: n}), do: Stream.map(0..(Integer.pow(2, n) - 1), fn v -> for i <- 0..(n - 1), do: Bitwise.band(Bitwise.bsr(v, i), 1) end)
  def enumerate(%{kind: :ints, n: n, lo: lo, hi: hi}), do: radix(n, hi - lo + 1) |> Stream.map(fn ds -> Enum.map(ds, &(&1 + lo)) end)
  def enumerate(%{kind: :seq, n: n, items: a}), do: (t = List.to_tuple(a); radix(n, length(a)) |> Stream.map(fn ds -> Enum.map(ds, &elem(t, &1)) end))
  def enumerate(%{kind: :perm, items: xs}), do: Stream.unfold(Enum.to_list(0..(length(xs) - 1)), fn nil -> nil; p -> {p, next_perm(p)} end) |> Stream.map(fn p -> (t = List.to_tuple(xs); Enum.map(p, &elem(t, &1))) end)
  def enumerate(%{kind: :subset, items: xs, k: k}), do: combos_stream(xs, k)
  def enumerate(%{kind: :subsets, items: xs}), do: Stream.flat_map(0..length(xs), &combos_stream(xs, &1))
  def enumerate(%{kind: :graph, items: es}), do: Stream.flat_map(0..length(es), &combos_stream(es, &1))
  def enumerate(%{kind: :partition, n: n, k: k}), do: rgs(n, k)
  def enumerate(%{kind: :program} = s), do: Stream.flat_map(1..s.max_size, &trees_of_size(s, &1))
  def enumerate(%{kind: :reals}), do: []

  defp radix(n, b) do
    Stream.unfold(List.duplicate(0, n), fn
      nil -> nil
      ds -> {ds, inc(ds, b)}
    end)
  end

  defp inc([], _b), do: nil
  defp inc([d | rest], b) when d + 1 < b, do: [d + 1 | rest]
  defp inc([_ | rest], b), do: (case inc(rest, b) do nil -> nil; r -> [0 | r] end)

  defp next_perm(p) do
    t = List.to_tuple(p)
    n = tuple_size(t)
    case Enum.find((n - 2)..0//-1, fn i -> elem(t, i) < elem(t, i + 1) end) do
      nil -> nil
      i ->
        j = Enum.find((n - 1)..(i + 1)//-1, fn j -> elem(t, j) > elem(t, i) end)
        t = t |> put_elem(i, elem(t, j)) |> put_elem(j, elem(t, i))
        l = Tuple.to_list(t)
        Enum.take(l, i + 1) ++ Enum.reverse(Enum.drop(l, i + 1))
    end
  end

  defp combos_stream(xs, k) do
    t = List.to_tuple(xs)
    n = tuple_size(t)
    if k > n do
      []
    else
      Stream.unfold(Enum.to_list(0..(k - 1)//1), fn
        nil -> nil
        idx -> {Enum.map(idx, &elem(t, &1)), next_combo(idx, n, k)}
      end)
    end
  end

  defp next_combo([], _n, _k), do: nil

  defp next_combo(idx, n, k) do
    t = List.to_tuple(idx)
    case Enum.find((k - 1)..0//-1, fn i -> elem(t, i) < n - k + i end) do
      nil -> nil
      i ->
        v = elem(t, i) + 1
        Enum.take(idx, i) ++ Enum.map(0..(k - 1 - i), &(v + &1))
    end
  end

  # restricted growth strings: every partition of n items into ≤ k groups exactly once
  defp rgs(n, k) do
    Stream.unfold([0 | List.duplicate(0, n - 1)], fn
      nil -> nil
      s -> {s, next_rgs(s, k)}
    end)
  end

  defp next_rgs(s, k) do
    t = List.to_tuple(s)
    n = tuple_size(t)
    prefix_max = fn i -> Enum.max(Enum.map(0..(i - 1)//1, &elem(t, &1)), fn -> -1 end) end
    case Enum.find((n - 1)..1//-1, fn i -> elem(t, i) < k - 1 and elem(t, i) <= prefix_max.(i) end) do
      nil -> nil
      i -> Enum.take(s, i) ++ [elem(t, i) + 1] ++ List.duplicate(0, n - i - 1)
    end
  end

  # ============================================================ programs

  @doc "A program candidate (a tree) as an Alembic function of its variables."
  def realize(%{kind: :program, vars: vars}, tree) do
    f = tree_fun(tree, Map.new(Enum.with_index(vars)))
    {:fn, show(%{kind: :program, vars: vars}, tree), length(vars), fn args -> f.(List.to_tuple(args)) end}
  end

  def realize(_space, x), do: x

  defp tree_fun(v, idx) when is_binary(v), do: (i = Map.fetch!(idx, v); fn a -> num(elem(a, i)) end)
  defp tree_fun(c, _idx) when is_number(c), do: fn _ -> c end

  defp tree_fun([op, a], idx) do
    fa = tree_fun(a, idx)
    g = unary_fun(op)
    fn args -> g.(fa.(args)) end
  end

  defp tree_fun([op, a, b], idx) do
    {fa, fb} = {tree_fun(a, idx), tree_fun(b, idx)}
    g = binary_fun(op)
    fn args -> g.(fa.(args), fb.(args)) end
  end

  defp num(v) when is_number(v), do: v
  defp num(v), do: Vapor.Alembic.Compiler.fail("a program's input must be a number, got #{Vapor.Alembic.Builtins.type(v)}")

  defp dom(name, f), do: fn x -> (try do f.(x) rescue ArithmeticError -> Vapor.Alembic.Compiler.fail("#{name}(#{x}) is undefined") end) end

  defp unary_fun("neg"), do: &(-&1)
  defp unary_fun("sin"), do: &:math.sin/1
  defp unary_fun("cos"), do: &:math.cos/1
  defp unary_fun("exp"), do: dom("exp", &:math.exp/1)
  defp unary_fun("log"), do: dom("log", &:math.log/1)
  defp unary_fun("sqrt"), do: dom("sqrt", &:math.sqrt/1)
  defp unary_fun("abs"), do: &abs/1
  defp unary_fun("sq"), do: dom("sq", &(&1 * &1))
  defp unary_fun("tanh"), do: &:math.tanh/1
  defp unary_fun("inv"), do: dom("inv", &(1 / &1))

  defp binary_fun("+"), do: fn a, b -> a + b end
  defp binary_fun("-"), do: fn a, b -> a - b end
  defp binary_fun("*"), do: fn a, b -> a * b end
  defp binary_fun("/"), do: fn a, b -> if b == 0, do: Vapor.Alembic.Compiler.fail("division by zero"), else: a / b end
  defp binary_fun("^"), do: fn a, b -> (try do :math.pow(a * 1.0, b * 1.0) rescue ArithmeticError -> Vapor.Alembic.Compiler.fail("#{a}^#{b} is undefined") end) end
  defp binary_fun("min"), do: &min/2
  defp binary_fun("max"), do: &max/2

  @doc "A candidate as text (programs in infix)."
  def show(%{kind: :program}, t), do: infix(t, 0)
  def show(_s, x), do: Alembic.show(x)

  defp infix(v, _) when is_binary(v), do: v
  defp infix(c, _) when is_number(c), do: Alembic.show(c)
  defp infix(["neg", a], _), do: "-" <> infix(a, 4)
  defp infix([f, a], _), do: f <> "(" <> infix(a, 0) <> ")"
  defp infix([op, a, b], _) when op in ["min", "max"], do: op <> "(" <> infix(a, 0) <> ", " <> infix(b, 0) <> ")"
  defp infix([op, a, b], p) do
    q = %{"+" => 1, "-" => 1, "*" => 2, "/" => 2, "^" => 3}[op]
    l = infix(a, if(op == "^", do: q + 1, else: q))
    r = infix(b, if(op in ["-", "/"], do: q + 1, else: q))
    s = l <> " " <> op <> " " <> r
    if q < p, do: "(" <> s <> ")", else: s
  end

  @doc "Parse an expression written in infix into a program tree of the space."
  def parse_program(s, text) do
    case Vapor.Alembic.Parser.expression(text) do
      {:ok, ast} ->
        case to_tree(ast, s) do
          {:ok, t} -> if tree_size(t) <= s.max_size * 2, do: {:ok, canon(s, t)}, else: {:error, "program larger than the space allows"}
          e -> e
        end
      {:error, e} -> {:error, Alembic.format_error(e)}
    end
  end

  defp to_tree({:lit, c}, _s) when is_number(c), do: {:ok, c}
  defp to_tree({:var, v, _}, s), do: if(v in s.vars, do: {:ok, v}, else: {:error, "unknown variable #{v}"})
  defp to_tree({:neg, a, _}, s) do
    with {:ok, t} <- to_tree(a, s) do
      cond do
        is_number(t) -> {:ok, -t}
        "neg" in s.ops -> {:ok, ["neg", t]}
        "-" in s.ops -> {:ok, ["-", 0, t]}
        "*" in s.ops -> {:ok, ["*", -1, t]}
        true -> {:error, "negation is not available in this space"}
      end
    end
  end
  defp to_tree({:bin, op, a, b, _}, s) when op in @binary, do: if(op in s.ops, do: tree2(op, a, b, s), else: {:error, "#{op} is not an operator of this space (#{Enum.join(s.ops, " ")})"})
  defp to_tree({:call, {:var, f, _}, args, _}, s) when f in @unary or f in ["min", "max"] do
    cond do
      f not in s.ops -> {:error, "#{f} is not an operator of this space (#{Enum.join(s.ops, " ")})"}
      match?([_], args) and f in @unary -> with({:ok, t} <- to_tree(hd(args), s), do: {:ok, [f, t]})
      match?([_, _], args) -> tree2(f, hd(args), List.last(args), s)
      true -> {:error, "#{f}: wrong number of arguments"}
    end
  end
  defp to_tree(_, _), do: {:error, "a program uses only numbers, its variables and the operators of the space"}

  defp tree2(op, a, b, s), do: with({:ok, x} <- to_tree(a, s), {:ok, y} <- to_tree(b, s), do: {:ok, [op, x, y]})

  defp valid_tree?(s, v) when is_binary(v), do: v in s.vars
  defp valid_tree?(_s, c) when is_number(c), do: true
  defp valid_tree?(s, [op, a]), do: op in @unary and valid_tree?(s, a)
  defp valid_tree?(s, [op, a, b]), do: op in @binary and valid_tree?(s, a) and valid_tree?(s, b)
  defp valid_tree?(_, _), do: false

  @doc "Number of nodes."
  def tree_size([_, a]), do: 1 + tree_size(a)
  def tree_size([_, a, b]), do: 1 + tree_size(a) + tree_size(b)
  def tree_size(_), do: 1

  defp tree_paths(t), do: tree_paths(t, [])
  defp tree_paths([_, a] = _t, p), do: [Enum.reverse(p) | tree_paths(a, [1 | p])]
  defp tree_paths([_, a, b], p), do: [Enum.reverse(p) | tree_paths(a, [1 | p]) ++ tree_paths(b, [2 | p])]
  defp tree_paths(_, p), do: [Enum.reverse(p)]

  defp get_at(t, []), do: t
  defp get_at(t, [i | rest]), do: get_at(Enum.at(t, i), rest)
  defp put_at(_t, [], v), do: v
  defp put_at(t, [i | rest], v), do: List.replace_at(t, i, put_at(Enum.at(t, i), rest, v))

  defp point_mutation(s, t, rng) do
    unary = Enum.filter(s.ops, &(&1 in @unary))
    binary = Enum.filter(s.ops, &(&1 in @binary))
    case t do
      [_op, a] when unary != [] -> {op, rng} = pick(unary, rng); {[op, a], rng}
      [_op, a, b] when binary != [] -> {op, rng} = pick(binary, rng); {[op, a, b], rng}
      [_ | _] -> {t, rng}
      _ ->
        {how, rng} = u(rng)
        if how < 0.5 and Enum.any?(s.leaves, &is_number/1) and is_number(t) do
          {z, rng} = :rand.normal_s(rng)
          {Float.round(t * (1 + 0.1 * z) * 1.0, 6), rng}
        else
          pick(s.leaves, rng)
        end
    end
  end

  @doc "A random tree of exactly (or about) `size` nodes."
  def random_tree(s, size, rng) do
    unary = Enum.filter(s.ops, &(&1 in @unary))
    binary = Enum.filter(s.ops, &(&1 in @binary))
    cond do
      size <= 1 or (unary == [] and binary == []) -> pick(s.leaves, rng)
      size == 2 and unary != [] ->
        {op, rng} = pick(unary, rng)
        {a, rng} = pick(s.leaves, rng)
        {[op, a], rng}
      binary == [] ->
        {op, rng} = pick(unary, rng)
        {a, rng} = random_tree(s, size - 1, rng)
        {[op, a], rng}
      true ->
        {r, rng} = u(rng)
        if unary != [] and r < 0.25 do
          {op, rng} = pick(unary, rng)
          {a, rng} = random_tree(s, size - 1, rng)
          {[op, a], rng}
        else
          {op, rng} = pick(binary, rng)
          {l, rng} = ui(max(size - 2, 1), rng)
          {a, rng} = random_tree(s, l, rng)
          {b, rng} = random_tree(s, max(size - 1 - l, 1), rng)
          {[op, a, b], rng}
        end
    end
  end

  @doc "Every tree of exactly `size` nodes (a stream)."
  def trees_of_size(s, 1), do: s.leaves
  def trees_of_size(s, size) do
    unary = Enum.filter(s.ops, &(&1 in @unary))
    binary = Enum.filter(s.ops, &(&1 in @binary))
    us = Stream.flat_map(unary, fn op -> Stream.map(trees_of_size(s, size - 1), &[op, &1]) end)
    bs =
      Stream.flat_map(binary, fn op ->
        Stream.flat_map(1..max(size - 2, 1)//1, fn l ->
          r = size - 1 - l
          if r < 1, do: [], else: Stream.flat_map(trees_of_size(s, l), fn a -> Stream.map(trees_of_size(s, r), &[op, a, &1]) end)
        end)
      end)
    Stream.concat(us, bs)
  end

  # constant folding and commutative ordering: equal programs, equal trees
  @doc false
  def simplify_tree([op, a]) do
    a = simplify_tree(a)
    if is_number(a) and op in ["neg", "abs", "sq"], do: unary_fun(op).(a), else: [op, a]
  end

  def simplify_tree([op, a, b]) do
    {a, b} = {simplify_tree(a), simplify_tree(b)}
    {a, b} = if op in ["+", "*", "min", "max"] and Vapor.Alembic.Builtins.compare(b, a) == :lt, do: {b, a}, else: {a, b}
    cond do
      is_number(a) and is_number(b) and op in ["+", "-", "*"] -> binary_fun(op).(a, b)
      is_number(a) and is_number(b) and op == "/" and b != 0 -> a / b
      op == "*" and a == 1 -> b
      op == "*" and b == 1 -> a
      op in ["*"] and (a == 0 or b == 0) -> 0
      op == "+" and a == 0 -> b
      op == "+" and b == 0 -> a
      op == "-" and b == 0 -> a
      op == "/" and b == 1 -> a
      op == "-" and a == b -> 0
      true -> [op, a, b]
    end
  end

  def simplify_tree(t), do: t

  # ============================================================ counting

  # trees with L leaves, U unary and B binary operators, of every size up to max
  defp program_count(l, u, b, max) do
    t =
      Enum.reduce(2..max//1, %{1 => l}, fn s, t ->
        bin = Enum.reduce(1..(s - 2)//1, 0, fn k, acc -> acc + t[k] * Map.get(t, s - 1 - k, 0) end)
        Map.put(t, s, u * t[s - 1] + b * bin)
      end)
    t |> Map.values() |> Enum.sum()
  end

  defp fact(n), do: Enum.reduce(1..max(n, 1), 1, &*/2)
  defp binom(n, k) when k < 0 or k > n, do: 0
  defp binom(n, k), do: Enum.reduce(1..min(k, n - k)//1, 1, fn i, acc -> div(acc * (n - min(k, n - k) + i), i) end)
  defp stirling2(n, k), do: div(Enum.reduce(0..k, 0, fn j, acc -> acc + Integer.pow(-1, k - j) * binom(k, j) * Integer.pow(j, n) end), fact(k))

  @doc "Coordinates of a candidate for the continuous strategies (reals and ints only)."
  def vector?(%{kind: k}), do: k in [:reals, :ints]
end
