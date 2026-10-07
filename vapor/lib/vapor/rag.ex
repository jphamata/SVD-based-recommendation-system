defmodule Vapor.RAG do
  @moduledoc """
  Retrieval-augmented generation whose every step can be checked.

  What breaks RAG in practice is not retrieval quality alone: it is that a
  result cannot be reproduced (indexes rebuilt on other hardware rank
  differently; embeddings drift with batch size and kernels), that nobody
  can prove which corpus an answer came from, and that models invent
  quotations. Each is addressed at its root:

    * **A corpus is a value.** Text is normalised (NFC), split by a fixed
      procedure into chunks with byte offsets, and each chunk is a leaf of a
      Merkle tree (`Vapor.Merkle`, RFC 6962 hashing). The root names the
      exact corpus; every retrieved chunk carries an inclusion proof.
    * **Retrieval is a function.** BM25 uses binary64 `+ − × ÷` in a fixed
      order and a correctly rounded logarithm (`Vapor.CR`); dense scores are
      a certified program (`Term.linear` of the query against the embedding
      matrix, canonical policy) run on the worker — the same bits on every
      substrate, as are the embeddings themselves (`Vapor.Embed`); hybrid
      fusion is reciprocal-rank fusion in exact rational arithmetic; ties
      break by chunk id. A retrieval's receipt is the canonical digest of
      (corpus root, method, parameters, query, embedder, results), and
      `verify/2` recomputes it.
    * **Quotations are verbatim by construction.** `citation_constraint/3`
      lets the model write freely but, inside `<quote src="i">…</quote>`,
      only bytes that continue a substring of source `i` (a suffix automaton
      per chunk, `Vapor.Grammar`). `check_citations/2` re-verifies quotes and
      inclusion proofs after the fact, for answers produced anywhere.
  """
  alias Vapor.{Canonical, CR, Merkle, Tensor}
  alias Vapor.Algebra.Term, as: T

  defstruct chunks: {}, root: nil, leaves: [], bm25: nil, dense: nil, params: %{}

  @type t :: %__MODULE__{}

  # ------------------------------------------------------------ corpus --

  @doc """
  Build a corpus from `[{doc_id, text}]`. Options: `:max_chars` (800, a
  chunk's target size), `:overlap` (1 sentence carried into the next
  chunk), `:embedder` (a `Vapor.Embed`: also build the dense index).
  """
  def corpus(docs, opts \\ []) do
    params = %{max_chars: Keyword.get(opts, :max_chars, 800), overlap: Keyword.get(opts, :overlap, 1), unicode: Vapor.Unicode.version()}
    chunks = Enum.flat_map(docs, fn {doc, text} -> chunk(doc, text, params) end)
    leaves = Enum.map(chunks, &Merkle.leaf(Canonical.encode({&1.doc, &1.start, &1.stop, &1.text})))
    rag = %__MODULE__{chunks: List.to_tuple(chunks), leaves: leaves, root: Merkle.root(leaves), params: params, bm25: bm25_index(chunks)}

    case opts[:embedder] do
      nil -> rag
      e -> dense_index(rag, e)
    end
  end

  @doc "The corpus root as hex (what an answer cites)."
  def root_hex(%__MODULE__{root: r}), do: Base.encode16(r, case: :lower)

  @doc false
  # paragraphs → sentences → chunks of at most max_chars (a longer sentence is its own chunk)
  def chunk(doc, text, %{max_chars: max, overlap: ov}) do
    text = Vapor.Unicode.nfc(text)

    sentences =
      Regex.scan(~r/[^\n.!?]*(?:[.!?]+(?=\s|$)|\n+|$)\s*/u, text, return: :index)
      |> Enum.map(fn [{at, len}] -> {at, len} end)
      |> Enum.reject(fn {_, len} -> len == 0 end)

    sentences
    |> pack(max, ov, [], [])
    |> Enum.map(fn sents ->
      {s0, _} = hd(sents)
      {sl, ll} = List.last(sents)
      stop = sl + ll
      body = binary_part(text, s0, stop - s0)
      %{doc: doc, start: s0, stop: stop, text: body, id: Canonical.hex_digest({doc, s0, body}) |> binary_part(0, 16)}
    end)
    |> Enum.reject(&(String.trim(&1.text) == ""))
  end

  defp pack([], _max, _ov, [], acc), do: Enum.reverse(acc)
  defp pack([], _max, _ov, cur, acc), do: Enum.reverse([Enum.reverse(cur) | acc])

  defp pack([s | rest], max, ov, cur, acc) do
    len = fn l -> Enum.reduce(l, 0, fn {_, n}, a -> a + n end) end

    if cur != [] and len.(cur) + elem(s, 1) > max do
      done = Enum.reverse(cur)
      # the last `ov` sentences open the next chunk, if they leave room
      carry = Enum.take(done, -ov)
      carry = if len.(carry) + elem(s, 1) > max, do: [], else: carry
      pack(rest, max, ov, [s | Enum.reverse(carry)], [done | acc])
    else
      pack(rest, max, ov, [s | cur], acc)
    end
  end

  @doc "Chunk `i` (0-based) with its Merkle inclusion proof."
  def chunk_with_proof(%__MODULE__{} = r, i), do: Map.put(elem(r.chunks, i), :proof, Merkle.proof(r.leaves, i))

  @doc "Check that a chunk (with `:proof`) belongs to the corpus with root `root`."
  def member?(chunk, root) do
    Merkle.verify(Merkle.leaf(Canonical.encode({chunk.doc, chunk.start, chunk.stop, chunk.text})), chunk.proof, root)
  end

  # --------------------------------------------------------------- BM25 --

  @k1 1.2
  @b 0.75

  @doc false
  def terms(text), do: Regex.scan(~r/[\p{L}\p{N}]+/u, String.downcase(text)) |> List.flatten()

  defp bm25_index(chunks) do
    tfs = Enum.map(chunks, fn c -> Enum.frequencies(terms(c.text)) end)
    lens = Enum.map(tfs, fn tf -> tf |> Map.values() |> Enum.sum() end)
    n = length(chunks)
    df = Enum.reduce(tfs, %{}, fn tf, acc -> Enum.reduce(Map.keys(tf), acc, &Map.update(&2, &1, 1, fn x -> x + 1 end)) end)
    avg = if n == 0, do: 0.0, else: Enum.sum(lens) / n
    %{tfs: List.to_tuple(tfs), lens: List.to_tuple(lens), df: df, n: n, avg: avg}
  end

  @doc "BM25 (k₁ = 1.2, b = 0.75): `[{chunk_index, score}]`, best first."
  def bm25(%__MODULE__{bm25: ix}, query, k) do
    qs = query |> terms() |> Enum.uniq() |> Enum.sort()
    idf = Map.new(qs, fn t -> nt = Map.get(ix.df, t, 0); {t, CR.log_f64((ix.n - nt + 0.5) / (nt + 0.5) + 1.0)} end)

    0..(ix.n - 1)//1
    |> Enum.map(fn i ->
      tf = elem(ix.tfs, i)
      dl = elem(ix.lens, i)

      score =
        Enum.reduce(qs, 0.0, fn t, acc ->
          case Map.get(tf, t, 0) do
            0 -> acc
            f -> acc + idf[t] * (f * (@k1 + 1)) / (f + @k1 * (1 - @b + @b * dl / ix.avg))
          end
        end)

      {i, score}
    end)
    |> Enum.filter(fn {_, s} -> s > 0 end)
    |> top(k)
  end

  # ------------------------------------------------------------- dense --

  defp dense_index(%__MODULE__{} = r, e) do
    texts = r.chunks |> Tuple.to_list() |> Enum.map(& &1.text)
    {:ok, vecs} = Vapor.Embed.embed(e, texts)
    d = e.spec.width
    n = length(vecs)
    m = Tensor.from_list(:f32, [n, d], List.flatten(vecs))
    # scores = q·Eᵀ: one certified linear map, canonical policy
    prog = Vapor.Program.new(scores: T.linear(T.input(:q, :f32, [1, d]), T.const(m)))
    {:ok, comp} = Vapor.Compile.Lower.lower(prog)
    %{r | dense: %{embedder: e, comp: comp, n: n, d: d, matrix_digest: Canonical.hex_digest(m.data)}}
  end

  @doc "Dense retrieval: `[{chunk_index, score}]` by the certified dot products."
  def dense(%__MODULE__{dense: nil}, _q, _k), do: {:error, :no_dense_index}

  def dense(%__MODULE__{dense: dx}, query, k) do
    {:ok, [q]} = Vapor.Embed.embed(dx.embedder, [query])

    {:ok, got} =
      Vapor.Runtime.Native.run(dx.embedder.worker, dx.comp, %{q: Tensor.from_list(:f32, [1, dx.d], q)},
                               isa: dx.embedder.isa, mode: :native)

    got.outputs.scores |> Tensor.to_floats() |> Enum.with_index(fn s, i -> {i, s} end) |> top(k)
  end

  # best first; equal scores by chunk index (the corpus order)
  defp top(scored, k), do: scored |> Enum.sort(fn {i, a}, {j, b} -> a > b or (a == b and i < j) end) |> Enum.take(k)

  # ------------------------------------------------------------ hybrid --

  @doc """
  Reciprocal-rank fusion, `Σ 1/(60 + rank)`, in exact rationals: no
  rounding can reorder two chunks.
  """
  def rrf(lists, k, c \\ 60) do
    lists
    |> Enum.reduce(%{}, fn list, acc ->
      list
      |> Enum.with_index(1)
      |> Enum.reduce(acc, fn {{i, _}, rank}, acc -> Map.update(acc, i, {1, c + rank}, &frac_add(&1, {1, c + rank})) end)
    end)
    |> Enum.sort(fn {i, {a, b}}, {j, {x, y}} -> a * y > x * b or (a * y == x * b and i < j) end)
    |> Enum.take(k)
  end

  defp frac_add({a, b}, {c, d}) do
    n = a * d + c * b
    m = b * d
    g = Integer.gcd(n, m)
    {div(n, g), div(m, g)}
  end

  # --------------------------------------------------------- retrieval --

  @doc """
  Retrieve: `method` in `:bm25 | :dense | :hybrid` (default: hybrid when
  there is a dense index). Returns `%{hits, root, receipt, method, query, k}`;
  each hit is a chunk with its rank, score and inclusion proof.
  """
  def retrieve(%__MODULE__{} = r, query, opts \\ []) do
    k = Keyword.get(opts, :k, 4)
    method = Keyword.get(opts, :method, if(r.dense, do: :hybrid, else: :bm25))
    wide = max(4 * k, 20)

    ranked =
      case method do
        :bm25 -> bm25(r, query, k)
        :dense -> dense(r, query, k)
        :hybrid -> rrf([bm25(r, query, wide), dense(r, query, wide)], k)
      end

    hits =
      ranked
      |> Enum.with_index(1)
      |> Enum.map(fn {{i, score}, rank} -> chunk_with_proof(r, i) |> Map.merge(%{rank: rank, score: score, index: i}) end)

    emb = if r.dense, do: r.dense.embedder.digest, else: nil
    receipt = Canonical.hex_digest({:retrieval, r.root, method, k, query, emb, Enum.map(hits, & &1.id)})
    %{hits: hits, root: r.root, receipt: receipt, method: method, query: query, k: k}
  end

  @doc """
  Re-derive a retrieval against a corpus: same root, same hits, valid
  proofs, same receipt. `:ok` or `{:error, why}`.
  """
  def verify(%__MODULE__{} = r, %{query: q, method: m, k: k} = result) do
    again = retrieve(r, q, method: m, k: k)

    cond do
      result.root != r.root -> {:error, :different_corpus}
      Enum.map(again.hits, & &1.id) != Enum.map(result.hits, & &1.id) -> {:error, :different_hits}
      not Enum.all?(result.hits, &member?(&1, r.root)) -> {:error, :bad_proof}
      again.receipt != result.receipt -> {:error, :different_receipt}
      true -> :ok
    end
  end

  # --------------------------------------------------------- citations --

  @doc """
  The sources block for a prompt: each hit numbered as the model must cite it.
  """
  def context(hits) do
    hits
    |> Enum.with_index(1)
    |> Enum.map_join("\n\n", fn {h, i} -> "[#{i}] (#{h.doc})\n#{h.text}" end)
  end

  @doc """
  A lazy constraint for answers with verbatim quotations: the text is free;
  after `<quote src="` the model must write the number of a source, `">`,
  a non-empty substring of that source, and `</quote>`.
  """
  def citation_constraint(hits, vocab, eos_ids) do
    alts =
      for i <- 1..length(hits) do
        {:seq, [{:lit, "#{i}\">"}, {:substr, {:source, i}}, {:lit, "</quote>"}]}
      end

    defs = for {h, i} <- Enum.with_index(hits, 1), into: %{}, do: {{:source, i}, Vapor.Grammar.suffix_automaton(h.text)}
    g = Vapor.Grammar.new({:alt, alts}, defs)
    Vapor.Grammar.Constraint.new(g, vocab, eos_ids, {:lazy, "<quote src=\""})
  end

  @doc """
  Every `<quote src="i">…</quote>` of an answer, checked: the quote occurs
  verbatim in source `i`, and source `i` proves membership in `root`.
  `[%{src, quote, verbatim, member}]`.
  """
  def check_citations(answer, hits, root) do
    tup = List.to_tuple(hits)

    Regex.scan(~r/<quote src="(\d+)">(.*?)<\/quote>/s, answer)
    |> Enum.map(fn [_, src, quote] ->
      i = String.to_integer(src)
      h = if i >= 1 and i <= tuple_size(tup), do: elem(tup, i - 1)
      %{src: i, quote: quote, verbatim: h != nil and quote != "" and :binary.match(h.text, quote) != :nomatch,
        member: h != nil and member?(h, root)}
    end)
  end
end
