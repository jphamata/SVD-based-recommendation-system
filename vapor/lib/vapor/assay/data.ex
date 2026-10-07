defmodule Vapor.Assay.Data do
  @moduledoc """
  Data hygiene for evaluation and training (docs/ASSAY.md §4–5):
  **contamination** of test items by a training corpus, and
  **near-duplicate** detection in a corpus.
  """
  alias Vapor.Assay.Stats
  import Stats, only: [f: 1]

  @doc "Words, lowercased, punctuation dropped (the GPT-3 contamination study's normalisation)."
  def words(text), do: text |> String.downcase() |> String.replace(~r/[^\p{L}\p{N}\s]/u, " ") |> String.split()

  defp grams(ws, n) when length(ws) < n, do: []
  defp grams(ws, n), do: ws |> Enum.chunk_every(n, 1, :discard) |> Enum.map(&:erlang.phash2(&1, 4_294_967_296))

  # ================================================================ contamination

  @doc """
  `{"train": [texts] | text, "test": [texts], "scores": [numbers]?}` →
  per-item overlap with the corpus: contaminated when any `n`-gram (13 by
  default; 8 for short items) of the item appears in training, plus the
  fraction of the item's n-grams found. With scores: the clean-vs-dirty
  difference and its permutation test.
  """
  def contamination(text, o) do
    with {:ok, m} <- decode(text) do
      train = m["train"] |> List.wrap() |> Enum.map(&to_string/1)
      test = m["test"] |> List.wrap() |> Enum.map(&to_string/1)
      scores = m["scores"]
      n = Keyword.get(o, :ngram, m["ngram"] || 13)
      cond do
        train == [] or test == [] -> {:error, "give both train (texts) and test (items)"}
        true ->
          index8 = train |> Enum.flat_map(&grams(words(&1), 8)) |> MapSet.new()
          index_n = train |> Enum.flat_map(&grams(words(&1), n)) |> MapSet.new()
          items =
            test
            |> Enum.with_index()
            |> Enum.map(fn {t, i} ->
              ws = words(t)
              {gs, k} = if length(ws) >= n, do: {grams(ws, n), n}, else: {grams(ws, 8), 8}
              idx = if k == n, do: index_n, else: index8
              hits = Enum.count(gs, &MapSet.member?(idx, &1))
              %{index: i, words: length(ws), ngram: k, overlap: if(gs == [], do: 0.0, else: hits / length(gs)), contaminated: hits > 0, score: scores && Enum.at(scores, i)}
            end)
          dirty = Enum.filter(items, & &1.contaminated)
          clean = Enum.reject(items, & &1.contaminated)
          effect = if is_list(scores) and dirty != [] and clean != [], do: score_effect(clean, dirty, Keyword.get(o, :seed, 1))
          short = Enum.count(items, &(&1.words < 8))

          {:ok, %{items: items, contaminated: length(dirty), clean: length(clean), clean_indices: Enum.map(clean, & &1.index), ngram: n, train_ngrams: MapSet.size(index_n), too_short: short,
                  score_effect: effect,
                  says: "#{length(dirty)} of #{length(items)} test items share a #{n}-gram (8-gram for short items) with the training corpus" <>
                    if(effect, do: "; score on contaminated items #{f(effect.dirty_mean)} vs clean #{f(effect.clean_mean)} (p = #{f(effect.p)})", else: "") <>
                    if(short > 0, do: "; #{short} item(s) shorter than 8 words cannot be tested this way", else: ""),
                  evidence: [%{check: "clean subset", ok: clean != [], detail: "#{length(clean)} items remain for an uncontaminated score"}]}}
      end
    end
  end

  defp decode(text) do
    case Vapor.JSON.decode(String.trim(text)) do
      {:ok, %{} = m} -> {:ok, m}
      _ -> {:error, ~s(contamination takes JSON: {"train": [texts], "test": [texts], "scores": [numbers]})}
    end
  end

  defp score_effect(clean, dirty, seed) do
    cs = Enum.map(clean, &Stats.num(&1.score)) |> Enum.reject(&is_nil/1)
    ds = Enum.map(dirty, &Stats.num(&1.score)) |> Enum.reject(&is_nil/1)
    if cs == [] or ds == [] do
      nil
    else
      obs = Stats.mean(ds) - Stats.mean(cs)
      all = cs ++ ds
      nd = length(ds)
      {hits, _} =
        Enum.reduce(1..2000, {0, Stats.rng(seed)}, fn _, {h, r} ->
          {keyed, r} = Enum.map_reduce(all, r, fn x, rr -> {u, rr} = :rand.uniform_s(rr); {{u, x}, rr} end)
          perm = keyed |> Enum.sort() |> Enum.map(&elem(&1, 1))
          {d, c} = Enum.split(perm, nd)
          {if(abs(Stats.mean(d) - Stats.mean(c)) >= abs(obs) - 1.0e-12, do: h + 1, else: h), r}
        end)
      %{clean_mean: Stats.mean(cs), dirty_mean: Stats.mean(ds), difference: obs, p: (hits + 1) / 2001}
    end
  end

  # ================================================================ dedup

  @doc """
  Near-duplicate detection: MinHash signatures (128 hashes over word
  5-shingles), locality-sensitive hashing (16 bands × 8 rows) for
  candidates, every candidate pair **verified by exact Jaccard** ≥
  `threshold` (0.8). Input: one document per line, or JSON lines with a
  `text` field. Returns clusters and the indices to keep (the first of each).
  """
  def dedup(text, o) do
    docs = docs(text)
    threshold = Keyword.get(o, :threshold, 0.8)
    k = Keyword.get(o, :shingle, 5)
    n = length(docs)

    if n < 2 do
      {:error, "need at least two documents (one per line)"}
    else
      shingles = Enum.map(docs, fn d -> ws = words(d); (if length(ws) < k, do: [ws], else: Enum.chunk_every(ws, k, 1, :discard)) |> Enum.map(&:erlang.phash2/1) |> MapSet.new() end)
      seeds = for i <- 1..128, do: :erlang.phash2({:minhash, i}, 4_294_967_296)
      sigs = Enum.map(shingles, fn s -> Enum.map(seeds, fn sd -> s |> Enum.map(&:erlang.phash2({sd, &1}, 4_294_967_296)) |> Enum.min(fn -> 0 end) end) end)
      buckets =
        sigs
        |> Enum.with_index()
        |> Enum.flat_map(fn {sig, i} -> sig |> Enum.chunk_every(8) |> Enum.with_index() |> Enum.map(fn {band, b} -> {{b, :erlang.phash2(band)}, i} end) end)
        |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
      cands = buckets |> Map.values() |> Enum.flat_map(fn ids -> ids = Enum.uniq(ids); for a <- ids, b <- ids, a < b, do: {a, b} end) |> Enum.uniq()
      st = List.to_tuple(shingles)
      sgt = List.to_tuple(sigs)
      checked =
        Enum.map(cands, fn {a, b} ->
          sa = elem(st, a)
          sb = elem(st, b)
          inter = MapSet.size(MapSet.intersection(sa, sb))
          union = MapSet.size(MapSet.union(sa, sb))
          est = Enum.zip(elem(sgt, a), elem(sgt, b)) |> Enum.count(fn {x, y} -> x == y end) |> Kernel./(128)
          %{a: a, b: b, jaccard: if(union == 0, do: 1.0, else: inter / union), estimate: est}
        end)
      dup = Enum.filter(checked, &(&1.jaccard >= threshold))
      clusters = components(n, Enum.map(dup, &{&1.a, &1.b}))
      keep = clusters |> Enum.map(&Enum.min/1) |> Enum.sort()
      err = if checked == [], do: 0.0, else: checked |> Enum.map(&abs(&1.jaccard - &1.estimate)) |> Stats.mean()

      {:ok, %{documents: n, candidates: length(cands), duplicates: length(dup), clusters: Enum.filter(clusters, &(length(&1) > 1)), keep: keep, removed: n - length(keep),
              threshold: threshold, minhash_error: err,
              says: "#{n - length(keep)} of #{n} documents are near-duplicates (Jaccard ≥ #{threshold} on #{k}-word shingles, verified exactly); #{length(keep)} kept",
              evidence: [%{check: "MinHash estimate", ok: err < 0.1, detail: "mean |estimate − exact Jaccard| over #{length(checked)} candidate pairs: #{f(err)}"}]}}
    end
  end

  @doc "Documents from text: JSON lines with `text`, else one per line."
  def docs(text) do
    lines = text |> String.split("\n") |> Enum.reject(&(String.trim(&1) == ""))
    Enum.map(lines, fn l ->
      case String.trim(l) do
        "{" <> _ = j -> case Vapor.JSON.decode(j) do {:ok, %{"text" => t}} -> to_string(t); _ -> l end
        _ -> l
      end
    end)
  end

  defp components(n, edges) do
    parent = Enum.reduce(edges, Map.new(0..(n - 1), &{&1, &1}), fn {a, b}, p -> union(p, a, b) end)
    0..(n - 1) |> Enum.group_by(&find(parent, &1)) |> Map.values() |> Enum.map(&Enum.sort/1) |> Enum.sort()
  end

  defp find(p, x), do: (if p[x] == x, do: x, else: find(p, p[x]))
  defp union(p, a, b), do: (ra = find(p, a); rb = find(p, b); if(ra == rb, do: p, else: Map.put(p, max(ra, rb), min(ra, rb))))
end
