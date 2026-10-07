defmodule Mix.Tasks.Vapor.Ocr do
  @shortdoc "Read text in pictures and scans (OCR through the model airlock); build datasets; measure error rates"
  @moduledoc """
      mix vapor.ocr read FILE… [--greedy] [--no-columns] [--no-tables] [--table-format md|html|csv]
                                                          # PNG, JPEG, PPM or PDF (pages without a text layer)
      mix vapor.ocr tables [DIR]                          # structure F1 and cell CER of scanned tables, with controls
      mix vapor.ocr eval DIR [--tesseract] [--limit N] [--model DIR]
      mix vapor.ocr page IMAGE TRUTH.json [--tesseract] [--model DIR]
      mix vapor.ocr dataset DIR OUT

  `read` prints the text with each line's confidence, and every table found
  (ruled grids, rule-only tables; `Vapor.Vision.Table`) as Markdown, HTML or
  CSV. `tables` scores the table fixtures (`test/python/table_render.py`)
  with the ICDAR-2013 adjacency metric against the controls of
  `docs/OCR.md §3e`. `eval` reads every
  line of a rendered set (`test/python/ocr_render.py`: `DIR/labels.json`)
  and prints character and word error rates, overall and per font; with
  `--tesseract` Tesseract's frozen readings of the same images
  (`DIR/tesseract.json`, from `test/python/ocr_tesseract.py`) are scored
  beside vapor's — vapor never runs an external tool. `page` scores a full page
  against its true lines (`{"lines": […]}`), line by line in order.
  `dataset` writes the normalised line bitmaps training reads
  (`Vapor.Vision.OCR.dataset/2`).

  Every command reads with the shipped character language model and the
  reading order of columns; `--greedy` decodes without the model and
  `--no-columns` reads the page as one block — the controls of
  `docs/OCR.md` §3b–§3c, from the command line.
  """
  use Mix.Task
  alias Vapor.Vision.OCR

  @impl true
  def run(argv) do
    {o, args, _} = OptionParser.parse(argv, strict: [tesseract: :boolean, limit: :integer, model: :string, greedy: :boolean, columns: :boolean,
                                                    tables: :boolean, table_format: :string])
    Mix.Task.run("app.start")
    model = if o[:model], do: ok!(OCR.load(o[:model])), else: ok!(OCR.default())
    :persistent_term.put({__MODULE__, :opts}, Enum.reject([lm: if(o[:greedy], do: false), columns: o[:columns], tables: o[:tables]], fn {_, v} -> v == nil end))
    :persistent_term.put({__MODULE__, :format}, o[:table_format] || "md")

    case args do
      ["read" | files] -> Enum.each(files, &read(&1, model))
      ["eval", dir] -> eval(dir, model, o)
      ["page", img, truth] -> page(img, truth, model, o)
      ["dataset", dir, out] -> Mix.shell().info(inspect(OCR.dataset(dir, out)))
      ["tables" | rest] -> tables(List.first(rest) || Path.join(to_string(:code.priv_dir(:vapor)), "quality/tables"))
      _ -> Mix.raise("usage: mix vapor.ocr read FILE… | eval DIR | page IMAGE TRUTH.json | dataset DIR OUT")
    end
  end

  defp ropts, do: :persistent_term.get({__MODULE__, :opts}, [])

  defp read(file, model) do
    bytes = File.read!(file)

    pictures =
      case Vapor.Docs.sniff(Path.basename(file), bytes) do
        :pdf ->
          {:ok, pages, _} = Vapor.Docs.PDF.pages(bytes)
          empty = for {t, i} <- Enum.with_index(pages, 1), String.trim(t) == "", do: i
          {:ok, imgs, _} = Vapor.Docs.PDF.images(bytes, empty)
          for {i, list} <- Enum.sort(imgs), img <- list, do: {"page #{i}", img}

        kind when kind in [:png, :jpeg, :pnm] ->
          {:ok, pic} = Vapor.Docs.Pictures.read(kind, bytes)
          [{Path.basename(file), pic.image}]

        other ->
          Mix.raise("#{file}: #{other} is not a picture or a PDF")
      end

    for {label, img} <- pictures do
      {:ok, r} = OCR.read(img, [model: model] ++ ropts())
      Mix.shell().info("── #{label} (#{img.w}×#{img.h}, confidence #{Float.round(r.confidence, 3)})")
      for l <- r.lines, do: Mix.shell().info("#{String.pad_leading(Float.to_string(Float.round(l.confidence, 2)), 5)}  #{l.text}")

      for {t, i} <- Enum.with_index(r.tables, 1) do
        Mix.shell().info("── table #{i}: #{t.kind}, #{t.rows}×#{t.cols}, #{t.header_rows} header row(s), confidence #{Float.round(t.confidence, 3)}")

        Mix.shell().info(
          case :persistent_term.get({__MODULE__, :format}, "md") do
            "html" -> Vapor.Vision.Table.to_html(t)
            "csv" -> Vapor.Vision.Table.to_csv(t)
            _ -> Vapor.Vision.Table.to_markdown(t)
          end)
      end
    end
  end

  defp tables(dir) do
    e = Vapor.Quality.Round08.evaluate_tables(dir, worker: OCR.worker())
    pct = fn nil -> "—"; x -> "#{Float.round(x * 100, 1)} %" end
    Mix.shell().info("table            style     dims   F1     exact  CER     free CER  lines F1  no-span F1  Tesseract CER*")

    for r <- e.rows do
      Mix.shell().info(Enum.join([String.pad_trailing(r.name, 16), String.pad_trailing(r.style, 9), String.pad_trailing(inspect(r.dims), 6),
                                  Float.round(r.f1, 3), Float.round(r.exact, 2), pct.(r.cer), pct.(r.free_cer), Float.round(r.lines_f1, 3),
                                  Float.round(r.nospan_f1, 3), pct.(r.tesseract_cer)], "  "))
    end

    m = e.mean
    Mix.shell().info("mean: F1 #{Float.round(m.f1, 3)} (0.7 reading #{Float.round(m.lines_f1, 3)}), cell CER #{pct.(m.cer)} (free #{pct.(m.free_cer)}; " <>
                       "Tesseract on perfectly cropped cells #{pct.(m.tesseract_cer)})\n* Tesseract given the true cell boxes")
  end

  defp eval(dir, model, o) do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))
    items = meta["lines"] |> Enum.sort() |> Enum.take(o[:limit] || 1_000_000)
    tess = o[:tesseract] && File.exists?(Path.join(dir, "tesseract.json"))
    if o[:tesseract] && !tess, do: Mix.shell().info("no #{Path.join(dir, "tesseract.json")} (test/python/ocr_tesseract.py writes it): vapor only")

    rows =
      for {file, %{"text" => text, "font" => font}} <- items do
        path = Path.join(dir, file)
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(path))
        {:ok, r} = OCR.read(pic.image, [model: model] ++ ropts())
        ref = String.trim(Regex.replace(~r/ +/, text, " "))
        t = if tess, do: OCR.tesseract(path) || "", else: nil
        %{font: font, ref: ref, vapor: r.text, tess: t}
      end

    report = fn label, rs ->
      line = fn key ->
        {e, n, we, wn} =
          Enum.reduce(rs, {0, 0, 0, 0}, fn r, {e, n, we, wn} ->
            h = Map.fetch!(r, key)
            {e + OCR.levenshtein(String.graphemes(h), String.graphemes(r.ref)), n + String.length(r.ref),
             we + OCR.levenshtein(String.split(h), String.split(r.ref)), wn + length(String.split(r.ref))}
          end)

        "CER #{pct(e / max(n, 1))}  WER #{pct(we / max(wn, 1))}"
      end

      Mix.shell().info("#{String.pad_trailing(label, 24)} vapor #{line.(:vapor)}" <> if(tess, do: "   tesseract #{line.(:tess)}", else: "") <> "   (#{length(rs)} lines)")
    end

    report.("all", rows)
    rows |> Enum.group_by(& &1.font) |> Enum.sort() |> Enum.each(fn {f, rs} -> report.(f, rs) end)
  end

  defp page(img, truth, model, o) do
    {:ok, pic} = Vapor.Docs.Pictures.read(Vapor.Docs.sniff(img, File.read!(img)), File.read!(img))
    {:ok, gt} = Vapor.JSON.decode(File.read!(truth))
    ref = Enum.join(gt["lines"], "\n")
    {:ok, r} = OCR.read(pic.image, [model: model] ++ ropts())
    hyp = r.lines |> Enum.take(length(gt["lines"])) |> Enum.map_join("\n", & &1.text)
    Mix.shell().info("vapor     CER #{pct(OCR.cer(hyp, ref))}  WER #{pct(OCR.wer(hyp, ref))}\n#{hyp}")

    with true <- o[:tesseract] == true, t when is_binary(t) <- OCR.tesseract(img) do
      t = t |> String.split("\n", trim: true) |> Enum.take(length(gt["lines"])) |> Enum.join("\n")
      Mix.shell().info("tesseract CER #{pct(OCR.cer(t, ref))}  WER #{pct(OCR.wer(t, ref))}\n#{t}")
    end
  end

  defp pct(x), do: "#{:erlang.float_to_binary(x * 100, decimals: 2)} %"

  defp ok!({:ok, v}), do: v
  defp ok!({:error, %Vapor.Rejection{} = r}), do: Mix.raise("refused at #{inspect(r.node)}: #{r.bound} — #{r.repair}")
end
