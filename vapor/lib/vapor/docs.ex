defmodule Vapor.Docs do
  @moduledoc """
  **A eclusa de documentos** — files in, provenance-carrying passages out.

  The model airlock (`Vapor.Lock`) keeps model families out of the core;
  this one keeps file formats out of retrieval. A reader *claims* a file by
  its bytes (magic numbers and container structure), never by its name, and
  turns it into passages:

      %{doc: "relatorio.zip!/anexos/contrato.pdf#p3", text: "…", kind: :pdf,
        sha256: "…", meta: %{page: 3, …}}

  `doc` is a path through containers (`!/` enters an archive, `#p3` names a
  page, `#slide2`, `#sheet:Modelos`), so a retrieved chunk says exactly
  where it came from — down to the page — and its bytes are bound to the
  file's SHA-256 in the library's manifest (`Vapor.Docs.Library`), which is
  part of every retrieval receipt.

  | reader | formats |
  |---|---|
  | `Vapor.Docs.Zip` | zip (recursive), and through it EPUB, DOCX/XLSX/PPTX, ODT/ODS/ODP |
  | `Vapor.Docs.PDF` | PDF 1.0–2.0 text: Flate/ASCIIHex/ASCII85, object streams, xref streams, ToUnicode CMaps, WinAnsi; per page |
  | `Vapor.Docs.Office` | Word, Excel (shared strings, numbers, per sheet), PowerPoint (per slide), OpenDocument |
  | `Vapor.Docs.Markup` | HTML/XHTML (scripts and styles dropped, blocks → lines, entities), XML |
  | `Vapor.Docs.Pictures` | PNG (decoded: every colour type and depth, Adam7; tEXt/iTXt/zTXt), JPEG (decoded, baseline and progressive, = libjpeg bit for bit; EXIF, comments), PPM/PGM, GIF/WebP (size and embedded text) |
  | plain | UTF-8 text, Markdown, CSV, JSON (and Latin-1 text, transcoded) |

  Text that exists only as pixels — a scanned PDF page (its JPEG, CCITT
  fax, Flate/LZW or 1-bit image), a photographed page, a screenshot — is read by
  `Vapor.Vision.OCR`, a model admitted through the airlock (`ocr: :auto`,
  the default, reads when a native worker exists); an OCR passage says so
  in its metadata and carries its confidence. What cannot become text is
  *said*, not guessed: an encrypted PDF, a Huffman-coded JBIG2
  scan, a JPX image, an arithmetic-coded JPEG, a binary blob — each becomes a `warning` in the
  report, never silent garbage in the index. An image also contributes
  its embedded text and, in the library, a visual descriptor for
  query-by-image.

  Limits (defaults, all options): `max_bytes` per expanded file (64 MiB),
  `max_total` expanded bytes per ingest (256 MiB), `max_entries` per
  archive (10 000), `max_depth` of nested archives (4), `max_ratio` of
  compression for members over 1 MiB (200×: text compresses 3–10×, a
  bomb of zeros ~1000×). Every ingest runs under `Vapor.Hermetic.seal/2`:
  `heap_mb` (2048, heap and binaries together) and `timeout` (600 s) bound
  what a file that slips past the declared limits can cost. A zip's declared sizes and ratios are checked
  **before** anything is inflated, and inflation is capped at the declared
  size (a lying header is a rejection) — a zip bomb is a rejection, not an
  outage. Archive member names are never used as file
  system paths.
  """
  alias Vapor.Rejection
  alias Vapor.Docs.{Markup, Office, PDF, Pictures, Zip}

  @defaults [max_bytes: 64 * 1024 * 1024, max_total: 256 * 1024 * 1024, max_entries: 10_000, max_depth: 4, max_ratio: 200, ocr: :auto,
             heap_mb: 2048, timeout: 600_000]

  @doc """
  Ingest a file (path) or `{name, bytes}`: `{:ok, %{passages, images,
  files, warnings}}` — `files` is the manifest (one entry per file met,
  containers included: path, kind, bytes, sha256, reader), `images` the
  decoded pictures (for the visual index).
  """
  def ingest(src, opts \\ []) do
    opts = Keyword.merge(@defaults, opts)

    with {:ok, name, bytes} <- read(src) do
      budget = :counters.new(1, [])
      acc = %{passages: [], images: [], files: [], warnings: []}
      limits = [heap_mb: opts[:heap_mb], timeout: opts[:timeout]]

      # every reader runs sealed: a malformed or hostile file costs at most the seal's memory and time
      case Vapor.Hermetic.seal(fn -> visit(name, bytes, 0, opts, budget, acc) end, limits) do
        {:ok, {:ok, acc}} ->
          {:ok, %{passages: Enum.reverse(acc.passages), images: Enum.reverse(acc.images),
                  files: Enum.reverse(acc.files), warnings: Enum.reverse(acc.warnings)}}

        {:ok, {:error, _} = e} ->
          e

        {:error, failure} ->
          {:error, Rejection.new({:docs, name}, "reading #{Vapor.Hermetic.describe(failure, limits)}",
                                 "raise heap_mb or timeout if the file is genuine, or split it")}
      end
    end
  end

  defp read({name, bytes}) when is_binary(name) and is_binary(bytes), do: {:ok, name, bytes}

  defp read(path) when is_binary(path) do
    case File.read(path) do
      {:ok, b} -> {:ok, Path.basename(path), b}
      {:error, why} -> {:error, Rejection.new({:file, path}, "readable (#{inspect(why)})", "check the path")}
    end
  end

  @doc false
  # one file: sniff, extract, recurse into containers
  def visit(name, bytes, depth, opts, budget, acc) do
    :counters.add(budget, 1, byte_size(bytes))

    cond do
      byte_size(bytes) > opts[:max_bytes] ->
        {:error, Rejection.new({:docs, name}, "at most #{opts[:max_bytes]} bytes per file (#{byte_size(bytes)})", "raise max_bytes or split the file")}

      :counters.get(budget, 1) > opts[:max_total] ->
        {:error, Rejection.new({:docs, name}, "at most #{opts[:max_total]} expanded bytes per ingest", "raise max_total or ingest in parts")}

      true ->
        kind = sniff(name, bytes)
        acc = %{acc | files: [%{path: name, kind: kind, bytes: byte_size(bytes), sha256: sha(bytes)} | acc.files]}
        extract(kind, name, bytes, depth, opts, budget, acc)
    end
  end

  defp sha(b), do: :crypto.hash(:sha256, b) |> Base.encode16(case: :lower)

  @doc "The kind of a file, from its bytes (the name only breaks ties between text formats)."
  def sniff(name, bytes) do
    ext = name |> Path.extname() |> String.downcase()

    cond do
      match?(<<"%PDF-", _::binary>>, bytes) -> :pdf
      match?(<<0x89, "PNG\r\n", 0x1A, "\n", _::binary>>, bytes) -> :png
      match?(<<0xFF, 0xD8, 0xFF, _::binary>>, bytes) -> :jpeg
      match?(<<"GIF8", _::binary>>, bytes) -> :gif
      match?(<<"RIFF", _::32, "WEBP", _::binary>>, bytes) -> :webp
      match?(<<"P5", _::binary>>, bytes) or match?(<<"P6", _::binary>>, bytes) -> :pnm
      match?(<<"PK", 3, 4, _::binary>>, bytes) or match?(<<"PK", 5, 6, _::binary>>, bytes) -> Zip.flavour(bytes)
      not text?(bytes) -> :binary
      ext in ~w(.html .htm .xhtml) or html?(bytes) -> :html
      ext in ~w(.xml .svg) or match?(<<"<?xml", _::binary>>, bytes) -> :xml
      ext == ".csv" -> :csv
      ext == ".json" -> :json
      ext in ~w(.md .markdown) -> :markdown
      true -> :text
    end
  end

  defp html?(bytes) do
    head = bytes |> binary_part(0, min(512, byte_size(bytes))) |> String.downcase()
    String.contains?(head, "<!doctype html") or String.contains?(head, "<html")
  end

  # text: valid UTF-8 (or Latin-1 without control bytes) with few NULs
  defp text?(bytes) do
    sample = binary_part(bytes, 0, min(8192, byte_size(bytes)))
    not String.contains?(sample, <<0>>) and (String.valid?(Pictures.utf8_prefix(sample)) or latin1?(sample))
  end

  defp latin1?(b), do: Enum.all?(:binary.bin_to_list(b), &(&1 in [9, 10, 13] or &1 >= 32))

  defp extract(kind, name, bytes, depth, opts, budget, acc) when kind in [:zip, :docx, :xlsx, :pptx, :odf, :epub] do
    if depth >= opts[:max_depth] do
      {:error, Rejection.new({:docs, name}, "archives nested at most #{opts[:max_depth]} deep", "raise max_depth")}
    else
      Zip.extract(kind, name, bytes, depth, opts, budget, acc)
    end
  end

  defp extract(:pdf, name, bytes, _d, opts, _b, acc) do
    case PDF.pages(bytes) do
      {:ok, pages, warns} ->
        passages = for {text, i} <- Enum.with_index(pages, 1), String.trim(text) != "",
                       do: %{doc: "#{name}#p#{i}", text: text, kind: :pdf, meta: %{page: i, pages: length(pages)}}

        empty = for {text, i} <- Enum.with_index(pages, 1), String.trim(text) == "", do: i
        {ocr_passages, ocr_warns} = ocr_pages(name, bytes, empty, length(pages), opts)
        {:ok, add(acc, passages ++ ocr_passages, Enum.map(warns, &"#{name}: #{&1}") ++ ocr_warns)}

      {:error, %Rejection{bound: b}} ->
        {:ok, add(acc, [], ["#{name}: refused — #{b}"])}
    end
  end

  defp extract(kind, name, bytes, _d, opts, _b, acc) when kind in [:png, :jpeg, :gif, :webp, :pnm] do
    case Pictures.read(kind, bytes) do
      {:ok, pic} ->
        text = Pictures.describe(name, pic)
        {ocr, ocr_warns} = if pic[:image], do: ocr_picture(name, pic.image, opts), else: {[], []}
        acc = add(acc, [%{doc: name, text: text, kind: :image, meta: Map.drop(pic, [:image])} | ocr],
                  (pic[:warnings] |> List.wrap() |> Enum.map(&"#{name}: #{&1}")) ++ ocr_warns)
        {:ok, if(pic[:image], do: %{acc | images: [%{doc: name, image: pic.image} | acc.images]}, else: acc)}

      {:error, %Rejection{bound: b}} ->
        {:ok, add(acc, [], ["#{name}: refused — #{b}"])}
    end
  end

  defp extract(:html, name, bytes, _d, _o, _b, acc), do: {:ok, add(acc, [%{doc: name, text: Markup.html_text(utf8(bytes)), kind: :html, meta: %{}}], [])}
  defp extract(:xml, name, bytes, _d, _o, _b, acc), do: {:ok, add(acc, [%{doc: name, text: Markup.xml_text(utf8(bytes)), kind: :xml, meta: %{}}], [])}
  defp extract(:binary, name, _bytes, _d, _o, _b, acc), do: {:ok, add(acc, [], ["#{name}: binary, not indexed"])}
  defp extract(kind, name, bytes, _d, _o, _b, acc), do: {:ok, add(acc, [%{doc: name, text: utf8(bytes), kind: kind, meta: %{}}], [])}

  # ------------------------------------------------------------------ OCR --
  #
  # Text that exists only as pixels — a scanned page, a photographed
  # whiteboard, a screenshot — read by Vapor.Vision.OCR, a model admitted
  # through the airlock. `ocr: :auto` reads when the model loads and a
  # native worker exists; `true` insists (oracle if need be); `false` skips.
  # An OCR passage says so (`meta.ocr`) and carries its confidence.

  defp ocr_ready(opts) do
    case opts[:ocr] do
      false -> {:off, nil}
      mode ->
        w = Vapor.Vision.OCR.worker()
        cond do
          mode == :auto and w == nil -> {:off, "no native worker for OCR (ocr: true reads on the oracle, slowly)"}
          match?({:ok, _}, Vapor.Vision.OCR.default()) -> {:on, w}
          true -> {:off, "the OCR model did not load"}
        end
    end
  end

  defp ocr_pages(_name, _bytes, [], _n, _opts), do: {[], []}

  defp ocr_pages(name, bytes, empty, n, opts) do
    no_text = fn i, why -> "#{name}#p#{i}: no text layer#{why}" end

    case ocr_ready(opts) do
      {:off, why} ->
        {[], Enum.map(empty, &no_text.(&1, if(why, do: " (scanned? #{why})", else: " (scanned? OCR is off)")))}

      {:on, w} ->
        case PDF.images(bytes, empty) do
          {:ok, imgs, warns} ->
            {ps, ws} =
              Enum.map_reduce(empty, [], fn i, ws ->
                case Map.get(imgs, i, []) |> Enum.max_by(&(&1.w * &1.h), fn -> nil end) do
                  nil -> {[], [no_text.(i, " and no image OCR can read") | ws]}
                  img -> ocr_passage("#{name}#p#{i}", img, w, %{page: i, pages: n}, :pdf, ws)
                end
              end)

            {List.flatten(ps), Enum.reverse(ws) ++ Enum.map(warns, &"#{name}: #{&1}")}

          {:error, %Rejection{bound: b}} ->
            {[], Enum.map(empty, &no_text.(&1, " (images unreadable: #{b})"))}
        end
    end
  end

  defp ocr_picture(name, img, opts) do
    case ocr_ready(opts) do
      {:on, w} ->
        {ps, ws} = ocr_passage(name <> "#ocr", img, w, %{}, :image_text, [], quiet: true)
        {List.flatten(ps), ws}

      {:off, _} ->
        {[], []}
    end
  end

  defp ocr_passage(doc, img, w, meta, kind, ws, opts \\ []) do
    # a picture that is not a page (a photo, a chart) is read quietly: its
    # OCR counts only when confident; a scanned page always says what it got
    quiet = Keyword.get(opts, :quiet, false)

    # a photo keeps only lines that hold a word and that the reader is sure of
    keep? = fn l -> not quiet or (l.confidence >= 0.8 and Vapor.Vision.OCR.word?(l.text)) end

    case Vapor.Vision.OCR.read(img, worker: w) do
      {:ok, %{lines: ls} = r} when ls != [] or r.tables != [] ->
        # each table is a passage of its own, as Markdown (rows stay rows
        # in the index), its structure and HTML in the meta
        tables =
          for {t, i} <- Enum.with_index(r.tables, 1), not quiet or t.confidence >= 0.8 do
            %{doc: "#{doc}#table#{i}", text: t.markdown, kind: :table,
              meta: Map.merge(meta, %{ocr: %{confidence: Float.round(t.confidence, 4)},
                                      table: %{kind: t.kind, rows: t.rows, cols: t.cols, header_rows: t.header_rows,
                                               html: Vapor.Vision.Table.to_html(t), csv: Vapor.Vision.Table.to_csv(t)}})}
          end

        case Enum.filter(ls, keep?) do
          [] ->
            {tables, ws}

          ls ->
            c = Enum.reduce(ls, 0.0, &(&1.confidence + &2)) / length(ls)
            low = if c < 0.8, do: ["#{doc}: OCR confidence #{Float.round(c, 2)} — check the text"], else: []
            text = Enum.map_join(ls, "\n", & &1.text)
            {[%{doc: doc, text: text, kind: kind, meta: Map.merge(meta, %{ocr: %{confidence: Float.round(c, 4), lines: length(ls)}})} | tables], low ++ ws}
        end

      {:ok, _} ->
        {[], if(quiet, do: ws, else: ["#{doc}: no text layer, and OCR found no text" | ws])}

      {:error, %Rejection{bound: b}} ->
        {[], ["#{doc}: OCR refused — #{b}" | ws]}
    end
  end

  @doc false
  def add(acc, passages, warns) do
    passages = Enum.reject(passages, &(String.trim(&1.text) == ""))
    %{acc | passages: Enum.reverse(passages) ++ acc.passages, warnings: Enum.reverse(warns) ++ acc.warnings}
  end

  @doc "Bytes as UTF-8: as they are when valid, else read as Latin-1."
  def utf8(bytes) do
    if String.valid?(bytes), do: bytes, else: :unicode.characters_to_binary(bytes, :latin1)
  end

  @doc "Office readers, for the zip reader."
  def office(kind, name, files, acc), do: Office.extract(kind, name, files, acc)
end
