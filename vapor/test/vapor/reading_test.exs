defmodule Vapor.ReadingTest do
  @moduledoc """
  Reading a real office scan: layout (the reading order of columns), the
  character language model, and the CTC beam search that uses it — each
  against the control that would fail without it.

    * columns: two- and three-column scanned PDFs read in order (and read
      line-across-columns, CER > 30 %, with the cut turned off); a
      one-column page is untouched by the cut;
    * the language model is a probability distribution, never zero, and
      means something (a shuffled corpus does not predict held-out text);
    * the beam search fixes what the frames leave ambiguous, and abstains
      on a line with no language in it;
    * on the held-out lines, the decoder with the model is measurably
      better than greedy — and on random strings it is *identical*.
  """
  use ExUnit.Case, async: false
  alias Vapor.Docs.PDF
  alias Vapor.Vision.{CharLM, OCR, Segment}

  @moduletag timeout: 900_000
  @scans Path.expand("../../priv/quality/scans", __DIR__)
  @held_out Path.expand("../../priv/quality/ocr", __DIR__)
  @controls Path.expand("../../priv/quality/ocr_controls", __DIR__)

  defp corpus, do: File.read!(Path.expand("../../priv/quality/pt_reference.txt", __DIR__)) <> " " <> File.read!(Path.expand("../../priv/quality/en_reference.txt", __DIR__))
  defp alphabet, do: (Enum.map(33..126, &<<&1>>) -- ["\""]) ++ String.graphemes("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÇ")

  setup_all do
    %{lm: CharLM.build(corpus(), alphabet(), 5)}
  end

  describe "the character language model" do
    test "a distribution over the alphabet and the space, for any history; nothing has probability zero", %{lm: lm} do
      for h <- ["", "a", "contrat", "the quick", "zzzzqx#", "R$ 12."] do
        ps = for c <- [" " | alphabet()], do: :math.exp(CharLM.log_prob(lm, h, c))
        assert_in_delta Enum.sum(ps), 1.0, 1.0e-9
        assert Enum.min(ps) > 0
      end

      # it knows the languages it was counted from
      assert CharLM.log_prob(lm, "informaçã", "o") > CharLM.log_prob(lm, "informaçã", "x")
      assert CharLM.log_prob(lm, "the transl", "a") > CharLM.log_prob(lm, "the transl", "q")
    end

    test "deterministic; and a shuffled corpus (same letters, no language) predicts held-out text worse", %{lm: lm} do
      assert CharLM.build(corpus(), alphabet(), 5).table == lm.table
      held = File.read!(Path.expand("../../priv/quality/pt_holdout.txt", __DIR__)) |> String.slice(0, 4000)
      real = CharLM.bits_per_char(lm, held)
      shuffled = CharLM.bits_per_char(CharLM.shuffled(corpus(), alphabet(), 5), held)
      assert real < 3.5 and shuffled > real + 1.0
    end

    test "the beam search fixes what the frames leave ambiguous — and abstains on a line with no language", %{lm: lm} do
      {:ok, m} = OCR.default()
      labels = m.labels
      idx = labels |> Tuple.to_list() |> Enum.with_index() |> Map.new()
      n = tuple_size(labels)

      # frames: each character one confident frame and a blank, except the
      # ambiguous ones, split between two readings (0.54 for the wrong one, 0.44 the right)
      frames = fn spec ->
        Enum.flat_map(spec, fn
          {wrong, right} -> [dist(n, [{idx[wrong], 0.54}, {idx[right], 0.44}]), dist(n, [{0, 0.99}])]
          c -> [dist(n, [{idx[c], 0.97}]), dist(n, [{0, 0.99}])]
        end)
      end

      cfg = %{lm: lm, weight: 0.8, bonus: 3.0, beam: 8, gate: 8.0}
      text = fn ks -> Enum.map_join(ks, &elem(labels, &1)) end
      line = frames.(String.graphemes("a c") ++ [{"1", "l"}] ++ String.graphemes("áusula do contrato"))
      assert text.(OCR.greedy_labels(line)) == "a c1áusula do contrato"
      assert text.(OCR.ctc_beam(line, labels, cfg)) == "a cláusula do contrato"

      # the alignment gives the chosen character its frames and probability
      chars = OCR.align(line, OCR.ctc_beam(line, labels, cfg), labels)
      l = Enum.find(chars, &(&1.char == "l"))
      assert l.frames == [6]
      assert_in_delta l.p, 0.44, 0.02

      # random characters: the reading costs > gate bits per character: the frames decide
      noise = frames.(String.graphemes("xQ7#") ++ [{"1", "l"}] ++ String.graphemes("zK9&w"))
      assert text.(OCR.ctc_beam(noise, labels, cfg)) == text.(OCR.greedy_labels(noise))
    end
  end

  defp dist(n, peaks) do
    rest = (1.0 - Enum.reduce(peaks, 0.0, fn {_, p}, a -> a + p end)) / (n - length(peaks))
    m = Map.new(peaks)
    for(i <- 0..(n - 1), do: :math.log(Map.get(m, i, rest))) |> List.to_tuple()
  end

  # ------------------------------------------------------------ on pages --

  defp page(name) do
    {:ok, truth} = Vapor.JSON.decode(File.read!(Path.join(@scans, "truth.json")))
    {:ok, %{1 => [img]}, []} = PDF.images(File.read!(Path.join(@scans, name <> ".pdf")), [1])
    {img, truth[name]["blocks"] |> List.flatten() |> Enum.join("\n")}
  end

  test "the layout of every scanned page: as many blocks as the typesetter made, the title not cut at a word space" do
    {:ok, truth} = Vapor.JSON.decode(File.read!(Path.join(@scans, "truth.json")))

    for {name, t} <- truth do
      {img, _} = page(name)
      blocks = img |> Segment.gray() |> Segment.ink() |> Segment.components() |> Segment.blocks()
      assert length(blocks) == length(t["blocks"]), name
    end
  end

  @tag :native
  test "a line with no ascender keeps its accents: the tilde of \"não\", the cedilla of \"ação\"" do
    {img, _} = page("p4_two_col_title")
    {:ok, r} = OCR.read(img, lm: false)
    line = Enum.find(r.lines, &String.starts_with?(&1.text, "mesm"))
    # before 0.7 this line read "mesmã eeeução rão anuneiãm ã mesnã ãção": its accents were dropped by the line finder
    assert OCR.cer(line.text, "mesma execução não anunciam a mesma ação") < 0.1
  end

  @tag :native
  test "scanned two-column pages (CCITT in a PDF) read in order; with the cut off, lines cross the columns" do
    for name <- ["p2_two_col", "p4_two_col_title"] do
      {img, ref} = page(name)
      {:ok, r} = OCR.read(img)
      assert OCR.cer(r.text, ref) < 0.03, name
      # the control: one block, lines across both columns
      {:ok, flat} = OCR.read(img, columns: false)
      assert OCR.cer(flat.text, ref) > 0.30, name
    end
  end

  @tag :native
  test "the whole path: a scanned CCITT PDF ingested as a document becomes a searchable page passage" do
    {:ok, r} = Vapor.Docs.ingest(Path.join(@scans, "p6_legal_two_col.pdf"))
    assert [%{doc: "p6_legal_two_col.pdf#p1", kind: :pdf, meta: %{ocr: %{confidence: c}}} = p] = r.passages
    assert c > 0.9
    {_img, ref} = page("p6_legal_two_col")
    assert OCR.cer(p.text, ref) < 0.03
    # before 0.7 the same file gave no passage, only "an image in CCITTFaxDecode (not decoded here)"
    refute Enum.any?(r.warnings, &(&1 =~ "CCITT"))
  end

  @tag :native
  test "a one-column page: the cut finds one block and changes nothing" do
    {img, ref} = page("p7_legal_one_col")
    g = Segment.gray(img)
    comps = g |> Segment.ink() |> Segment.components()
    assert [_one] = Segment.blocks(comps)
    {:ok, a} = OCR.read(img)
    {:ok, b} = OCR.read(img, columns: false)
    assert a.text == b.text and OCR.cer(a.text, ref) < 0.02
  end

  defp lines(dir) do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(dir, "labels.json")))
    {:ok, m} = OCR.default()
    w = OCR.worker()

    for {file, %{"text" => text}} <- Enum.sort(meta["lines"]),
        {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(dir, file))),
        [lps] <- [OCR.line_logprobs(pic.image, m, w)], lps != nil,
        do: {String.trim(Regex.replace(~r/ +/, text, " ")), lps}
  end

  defp cer_of(items, dec, labels) do
    {e, n} =
      Enum.reduce(items, {0, 0}, fn {ref, lps}, {e, n} ->
        hyp = dec.(lps) |> Enum.map_join(&elem(labels, &1)) |> String.trim()
        {e + OCR.levenshtein(String.graphemes(hyp), String.graphemes(ref)), n + String.length(ref)}
      end)

    e / n
  end

  @tag :native
  test "held-out lines: the model cuts the character errors by a quarter or more; random strings: identical to greedy" do
    {:ok, m} = OCR.default()
    lm = OCR.language_model(m, [])
    beam = &OCR.ctc_beam(&1, m.labels, lm)

    held = lines(@held_out)
    g = cer_of(held, &OCR.greedy_labels/1, m.labels)
    b = cer_of(held, beam, m.labels)
    assert b < 0.75 * g, "greedy #{g}, beam #{b}"

    rand = lines(Path.join(@controls, "random"))
    assert Enum.all?(rand, fn {_, lps} -> beam.(lps) == OCR.greedy_labels(lps) end)

    codes = lines(Path.join(@controls, "codes"))
    assert cer_of(codes, beam, m.labels) <= cer_of(codes, &OCR.greedy_labels/1, m.labels)
  end
end
