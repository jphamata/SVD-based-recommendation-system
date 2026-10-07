defmodule Vapor.Docs.Library do
  @moduledoc """
  A searchable library of files — the RAG of `Vapor.RAG` fed by the
  document airlock (`Vapor.Docs`), with two indices and one root.

    * **Text**: every passage of every file (PDF pages, sheets, slides,
      archive members…) chunked into a `Vapor.RAG` corpus whose Merkle root
      names it; BM25 by default, hybrid when an embedder is given.
    * **Pictures**: a deterministic visual descriptor per decoded image —
      an 8×8 area-averaged thumbnail per channel and a 3×8-bin colour
      histogram, centred and L2-normalised — compared by one certified
      contraction (`Vapor.Modal.Runner`). It finds *visually similar*
      pictures (layout and colour), not semantically similar ones: there is
      no CLIP here, and the slot for one is the encoder adapter
      (`Vapor.Lock.Adapters.Encoder`) plus a projector.
    * **Root**: the canonical digest of (text corpus root, manifest of every
      file met with its SHA-256). A search receipt binds the query, the
      hits and this root; `verify/2` recomputes everything — so an answer
      can prove which *files*, not only which chunks, it came from.

  Adding the same file twice (by SHA-256) is a no-op.
  """
  alias Vapor.{Canonical, RAG, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Modal.{Image, Runner}

  defstruct files: [], passages: [], images: [], warnings: [], rag: nil, visual: nil, opts: []

  def new(opts \\ []), do: rebuild(%__MODULE__{opts: opts})

  @doc "Ingest a path or `{name, bytes}` and re-index: `{:ok, library, report}`."
  def add(%__MODULE__{} = lib, src, opts \\ []) do
    with {:ok, r} <- Vapor.Docs.ingest(src, Keyword.merge(lib.opts, opts)) do
      top = hd(r.files)

      if Enum.any?(lib.files, &(&1.sha256 == top.sha256 and &1.path == top.path)) do
        {:ok, lib, %{added: 0, duplicate: top.path, warnings: []}}
      else
        lib = %{lib | files: lib.files ++ r.files, passages: lib.passages ++ r.passages, images: lib.images ++ r.images,
                      warnings: lib.warnings ++ r.warnings}

        {:ok, rebuild(lib), %{added: length(r.passages), images: length(r.images), files: length(r.files), warnings: r.warnings, file: top}}
      end
    end
  end

  defp rebuild(lib) do
    docs = Enum.map(lib.passages, &{&1.doc, &1.text})
    rag = RAG.corpus(docs, Keyword.take(lib.opts, [:max_chars, :overlap, :embedder]))
    %{lib | rag: rag, visual: visual_index(lib.images)}
  end

  @doc "The library root (hex): text corpus root + manifest of files."
  def root(%__MODULE__{} = lib),
    do: Canonical.hex_digest({:vapor_library, 1, lib.rag.root, Enum.map(lib.files, &{&1.path, &1.sha256})})

  @doc "Counts, for a status line."
  def stats(%__MODULE__{} = lib),
    do: %{files: length(lib.files), passages: length(lib.passages), chunks: tuple_size(lib.rag.chunks), images: length(lib.images),
          warnings: length(lib.warnings), root: root(lib)}

  @doc """
  Search the text: the `Vapor.RAG.retrieve/3` result plus, per hit, the
  file it came from (outermost container) and its SHA-256, the library
  root, and a receipt over all of it.
  """
  def search(%__MODULE__{} = lib, query, opts \\ []) do
    r = RAG.retrieve(lib.rag, query, opts)
    sha = Map.new(lib.files, &{&1.path, &1.sha256})

    hits =
      Enum.map(r.hits, fn h ->
        file = h.doc |> String.split("#") |> hd()
        outer = file |> String.split("!/") |> hd()
        Map.merge(h, %{file: file, file_sha256: sha[file], container: outer, container_sha256: sha[outer]})
      end)

    root = root(lib)
    Map.merge(r, %{hits: hits, library_root: root, library_receipt: Canonical.hex_digest({:library_search, root, r.receipt})})
  end

  @doc "Recompute a search against this library: `:ok` or `{:error, why}`."
  def verify(%__MODULE__{} = lib, result) do
    cond do
      result.library_root != root(lib) -> {:error, :different_library}
      Canonical.hex_digest({:library_search, result.library_root, result.receipt}) != result.library_receipt -> {:error, :bad_receipt}
      true -> RAG.verify(lib.rag, result)
    end
  end

  # ------------------------------------------------------------- pictures --

  @dims 224

  @doc "The visual descriptor of an image: `@dims` floats, centred, unit length."
  def descriptor(%Image{w: w, h: h, c: c, px: px}) do
    # one pass: per 8×8 cell and channel a sum and a count, and the histogram
    {sums, counts, hist} =
      Enum.reduce(0..(w * h - 1), {%{}, %{}, %{}}, fn i, {sm, ct, hs} ->
        {x, y} = {rem(i, w), div(i, w)}
        cell = div(y * 8, h) * 8 + div(x * 8, w)
        vals = if c == 3, do: [elem(px, i * 3), elem(px, i * 3 + 1), elem(px, i * 3 + 2)], else: List.duplicate(elem(px, i), 3)

        {sm, hs} =
          vals
          |> Enum.with_index()
          |> Enum.reduce({sm, hs}, fn {v, ch}, {sm, hs} ->
            {Map.update(sm, {ch, cell}, v, &(&1 + v)), Map.update(hs, {ch, min(trunc(v * 8), 7) |> max(0)}, 1, &(&1 + 1))}
          end)

        {sm, Map.update(ct, cell, 1, &(&1 + 1)), hs}
      end)

    n = w * h
    thumb = for ch <- 0..2, cell <- 0..63, do: Map.get(sums, {ch, cell}, 0.0) / max(Map.get(counts, cell, 0), 1)
    hist = for ch <- 0..2, b <- 0..7, do: Map.get(hist, {ch, b}, 0) / n

    v = thumb ++ hist
    mean = Enum.sum(v) / 216
    v = Enum.map(v, &(&1 - mean)) ++ List.duplicate(0.0, @dims - 216)
    norm = :math.sqrt(Enum.reduce(v, 0.0, &(&1 * &1 + &2)))
    Enum.map(v, &if(norm > 0, do: &1 / norm, else: 0.0))
  end

  defp visual_index([]), do: nil

  defp visual_index(images) do
    m = Tensor.from_list(:f32, [length(images), @dims], images |> Enum.map(&descriptor(&1.image)) |> List.flatten())
    p = Vapor.Program.new([scores: T.linear(T.input(:q, :f32, [1, @dims]), T.ref(:pictures, T.const(m)))], lets: [pictures: T.const(m)])
    %{program: p, docs: Enum.map(images, & &1.doc)}
  end

  @doc "Pictures most similar to `image` (a `Vapor.Modal.Image`): `[%{doc, score}]`."
  def search_image(%__MODULE__{visual: nil}, _img, _opts), do: []

  def search_image(%__MODULE__{visual: v}, %Image{} = img, opts) do
    q = Tensor.from_list(:f32, [1, @dims], descriptor(img))
    scores = Runner.run(v.program, %{q: q}, Keyword.take(opts, [:worker])).scores |> Tensor.to_floats()

    v.docs
    |> Enum.zip(scores)
    |> Enum.with_index()
    |> Enum.sort(fn {{_, a}, i}, {{_, b}, j} -> a > b or (a == b and i < j) end)
    |> Enum.take(Keyword.get(opts, :k, 4))
    |> Enum.map(fn {{doc, s}, _} -> %{doc: doc, score: s} end)
  end
end
