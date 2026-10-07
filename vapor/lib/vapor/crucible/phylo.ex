defmodule Vapor.Crucible.Phylo do
  @moduledoc """
  A tree from the user's own aligned sequences (docs/CRUCIBLE.md §7):
  FASTA (DNA or protein), distances corrected for multiple hits
  (Jukes–Cantor for DNA, Poisson for proteins), **neighbour joining** with
  branch lengths, the tree in Newick, and the evidence a reader needs to
  trust a clade:

    * **bootstrap** support for every split (columns resampled with
      replacement, `bootstrap` replicates, seeded);
    * **saturation**: pairs too divergent for the correction (p ≥ 0.75 for
      DNA) are named — their distance is a floor, not a measurement;
    * **additivity**: the four-point condition's worst violation, relative
      — how far the distances are from any tree at all;
    * the **control**: each sequence's columns shuffled independently
      (homology destroyed, composition kept) — its splits' support must
      collapse; if it does not, the "signal" is composition.
  """

  def run(text) do
    with {:ok, seqs} <- fasta(text) do
      names = Enum.map(seqs, &elem(&1, 0))
      kind = if Enum.all?(seqs, fn {_, s} -> Enum.all?(s, &(&1 in ~w(A C G T U N - ?))) end), do: :dna, else: :protein
      opts = options(text)
      reps = opts |> Map.get("bootstrap", 100) |> min(1000)
      cols = seqs |> Enum.map(&elem(&1, 1)) |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
      cols = Enum.reject(cols, fn c -> Enum.any?(c, &(&1 in ["-", "?", "N"])) end)

      if length(cols) < 10 do
        {:error, "fewer than 10 gap-free columns: align the sequences first"}
      else
        {d, sat} = distances(names, cols, kind)
        {newick, splits} = nj(names, d)
        boot = bootstrap(names, cols, kind, splits, reps, 1)
        control_cols = shuffle_rows(names, cols)
        {_, csplits} = nj(names, elem(distances(names, control_cols, kind), 0))
        cboot = bootstrap(names, control_cols, kind, csplits, reps, 2)
        mean = fn m -> if map_size(m) == 0, do: 0.0, else: Enum.sum(Map.values(m)) / map_size(m) end
        add = four_point(names, d)

        {:ok, %{kind: "phylogeny", alphabet: Atom.to_string(kind), taxa: names, columns: length(cols), newick: newick,
                splits: Enum.map(splits, fn sp -> %{taxa: Enum.sort(MapSet.to_list(sp)), support: Map.get(boot, sp, 0.0)} end) |> Enum.sort_by(&(-&1.support)),
                distances: for(a <- names, do: for(b <- names, do: d[{a, b}])), saturated: sat, additivity: add,
                control: %{mean_support: mean.(cboot), splits: map_size(cboot)},
                evidence: [
                  %{check: "bootstrap", ok: true, detail: "mean support #{pct(mean.(boot))} over #{map_size(boot)} splits (#{reps} replicates)"},
                  %{check: "shuffled control", ok: mean.(cboot) < mean.(boot), detail: "with homology destroyed the mean support falls to #{pct(mean.(cboot))}"},
                  %{check: "saturation", ok: sat == [], detail: if(sat == [], do: "no pair beyond the correction's range", else: "#{length(sat)} pair(s) saturated: #{sat |> Enum.take(4) |> Enum.map_join(", ", fn [a, b] -> a <> "–" <> b end)}")},
                  %{check: "additivity", ok: add < 0.1, detail: "worst four-point violation #{Float.round(add * 100, 1)} % of the quartet's scale"}
                ],
                says: "#{length(names)} taxa, #{length(cols)} columns; neighbour joining with #{reps} bootstrap replicates"}}
      end
    end
  end

  defp options(text) do
    Regex.scan(~r/^\s*(bootstrap|seed)\s*=\s*(\d+)\s*$/m, text) |> Map.new(fn [_, k, v] -> {k, String.to_integer(v)} end)
  end

  defp pct(x), do: "#{round(x * 100)} %"

  @doc "Parse FASTA into [{name, [residue]}]."
  def fasta(text) do
    blocks = text |> String.split(~r/^>/m, trim: true) |> Enum.filter(&String.contains?(&1, "\n"))
    seqs =
      for b <- blocks do
        [head | lines] = String.split(b, "\n")
        name = head |> String.split() |> List.first() |> to_string()
        seq = lines |> Enum.reject(&Regex.match?(~r/^\s*\w+\s*=/, &1)) |> Enum.join() |> String.replace(~r/\s/, "") |> String.upcase() |> String.graphemes()
        {name, seq}
      end
      |> Enum.reject(fn {n, s} -> n == "" or s == [] end)

    lens = seqs |> Enum.map(&length(elem(&1, 1))) |> Enum.uniq()
    cond do
      length(seqs) < 4 -> {:error, "at least 4 sequences in FASTA (>name, then the sequence)"}
      length(seqs) > 200 -> {:error, "at most 200 sequences"}
      length(lens) > 1 -> {:error, "the sequences must be aligned (same length); lengths found: #{Enum.join(lens, ", ")}"}
      length(Enum.uniq(Enum.map(seqs, &elem(&1, 0)))) < length(seqs) -> {:error, "sequence names must be unique"}
      true -> {:ok, seqs}
    end
  end

  defp distances(names, cols, kind) do
    rows = for i <- 0..(length(names) - 1), do: Enum.map(cols, &Enum.at(&1, i))
    named = Enum.zip(names, rows)
    {d, sat} =
      for {a, ra} <- named, {b, rb} <- named, reduce: {%{}, []} do
        {d, sat} ->
          if a == b do
            {Map.put(d, {a, b}, 0.0), sat}
          else
            p = Enum.zip(ra, rb) |> Enum.count(fn {x, y} -> x != y end) |> Kernel./(length(ra))
            {dist, s?} = correct(p, kind)
            {Map.put(d, {a, b}, dist), if(s? and a < b, do: [[a, b] | sat], else: sat)}
          end
      end
    {d, Enum.reverse(sat)}
  end

  defp correct(p, :dna) when p >= 0.74, do: {3.0, true}
  defp correct(p, :dna), do: {-0.75 * :math.log(1 - 4 / 3 * p), false}
  defp correct(p, :protein) when p >= 0.95, do: {3.0, true}
  defp correct(p, :protein), do: {-:math.log(1 - p), false}

  # neighbour joining with branch lengths → {newick, splits as leaf sets}
  defp nj(names, d) do
    clusters = Enum.map(names, &{&1, &1, MapSet.new([&1])})
    join(clusters, d, [])
  end

  defp join(clusters, d, splits) when length(clusters) == 3 do
    [{a, ta, _}, {b, tb, _}, {c, tc, _}] = clusters
    la = max((d[{a, b}] + d[{a, c}] - d[{b, c}]) / 2, 0.0)
    lb = max((d[{a, b}] + d[{b, c}] - d[{a, c}]) / 2, 0.0)
    lc = max((d[{a, c}] + d[{b, c}] - d[{a, b}]) / 2, 0.0)
    {"(#{ta}:#{r(la)},#{tb}:#{r(lb)},#{tc}:#{r(lc)});", splits}
  end

  defp join(clusters, d, splits) do
    n = length(clusters)
    ids = Enum.map(clusters, &elem(&1, 0))
    rs = Map.new(ids, fn i -> {i, Enum.sum(for j <- ids, j != i, do: d[{i, j}])} end)
    {i, j} = (for a <- ids, b <- ids, a < b, do: {a, b}) |> Enum.min_by(fn {a, b} -> (n - 2) * d[{a, b}] - rs[a] - rs[b] end)
    {_, ti, si} = List.keyfind(clusters, i, 0)
    {_, tj, sj} = List.keyfind(clusters, j, 0)
    li = max(d[{i, j}] / 2 + (rs[i] - rs[j]) / (2 * (n - 2)), 0.0)
    lj = max(d[{i, j}] - li, 0.0)
    u = {:u, i, j}
    rest = Enum.reject(ids, &(&1 in [i, j]))
    d = Enum.reduce(rest, d, fn k, d -> v = (d[{i, k}] + d[{j, k}] - d[{i, j}]) / 2; d |> Map.put({u, k}, v) |> Map.put({k, u}, v) end) |> Map.put({u, u}, 0.0)
    set = MapSet.union(si, sj)
    join([{u, "(#{ti}:#{r(li)},#{tj}:#{r(lj)})", set} | Enum.reject(clusters, fn {id, _, _} -> id in [i, j] end)], d, if(MapSet.size(set) >= 2, do: [set | splits], else: splits))
  end

  defp r(x), do: :erlang.float_to_binary(x, [{:decimals, 5}, :compact])

  defp canon(sp, names) do
    first = hd(Enum.sort(names))
    if MapSet.member?(sp, first), do: MapSet.difference(MapSet.new(names), sp), else: sp
  end

  defp bootstrap(names, cols, kind, splits, reps, salt) do
    n = length(cols)
    t = List.to_tuple(cols)
    target = MapSet.new(Enum.map(splits, &canon(&1, names)))
    counts =
      Enum.reduce(1..reps, %{}, fn rep, acc ->
        sample = for k <- 1..n, do: elem(t, trunc(Vapor.Alembic.Builtins.hash01([salt, rep, k]) * n))
        {_, sp} = nj(names, elem(distances(names, sample, kind), 0))
        Enum.reduce(sp, acc, fn s, a -> c = canon(s, names); if MapSet.member?(target, c), do: Map.update(a, c, 1, &(&1 + 1)), else: a end)
      end)
    Map.new(splits, fn s -> {s, Map.get(counts, canon(s, names), 0) / reps} end)
  end

  defp shuffle_rows(names, cols) do
    rows = for i <- 0..(length(names) - 1), do: Enum.map(cols, &Enum.at(&1, i))
    shuffled = rows |> Enum.with_index() |> Enum.map(fn {row, i} -> row |> Enum.with_index() |> Enum.sort_by(fn {_, k} -> Vapor.Alembic.Builtins.hash01([:shuffle, i, k]) end) |> Enum.map(&elem(&1, 0)) end)
    shuffled |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
  end

  # worst violation of the four-point condition, relative to the quartet's largest sum
  defp four_point(names, d) when length(names) > 24, do: four_point(Enum.take(names, 24), d)

  defp four_point(names, d) do
    quartets = for a <- names, b <- names, c <- names, e <- names, a < b and b < c and c < e, do: {a, b, c, e}
    quartets
    |> Enum.map(fn {a, b, c, e} ->
      s = Enum.sort([d[{a, b}] + d[{c, e}], d[{a, c}] + d[{b, e}], d[{a, e}] + d[{b, c}]], :desc)
      [s1, s2, _] = s
      (s1 - s2) / max(s1, 1.0e-12)
    end)
    |> Enum.max(fn -> 0.0 end)
  end
end
