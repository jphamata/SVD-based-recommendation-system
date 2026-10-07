defmodule Vapor.ScriptsTest do
  @moduledoc """
  The non-Latin readers shipped in 0.10, each on lines set in typefaces it
  was never trained on (`priv/quality/<script>`, rendered by
  `test/python/ocr_render_script.py` and `ocr_render_cjk.py` with seeds and
  vocabulary halves the training never saw), and each against a control:

    * **Arabic** — read in visual order and returned to logical order by
      `Vapor.Vision.Bidi`, whose inverse must equal python-bidi's on every
      line; the Latin reader on the same lines is the control (it must fail);
    * **Cyrillic** — the same, left to right; the control is again the
      Latin reader, which shares a few letter shapes and nothing else;
    * **CJK** — Chinese, Japanese and Korean, with and without the
      character language model; on random characters (no language to
      model) the model must change nothing;
    * **cursive Latin** — no reader is shipped, and asking for one must be
      refused, not answered with the print reader's guesses.
  """
  use ExUnit.Case, async: false
  alias Vapor.Docs.Pictures
  alias Vapor.Vision.{Bidi, CJK, OCR}

  @moduletag timeout: 900_000
  @moduletag :native
  @q Path.expand("../../priv/quality", __DIR__)

  defp lines(dir), do: dir |> Path.join("labels.json") |> File.read!() |> Vapor.JSON.decode() |> elem(1) |> Map.fetch!("lines") |> Enum.sort()
  defp image(dir, f), do: elem(Pictures.read(:png, File.read!(Path.join(dir, f))), 1).image
  defp norm(s), do: String.trim(Regex.replace(~r/ +/, s, " "))

  defp cer(model, dir, take, key, w) do
    {e, n} =
      dir
      |> lines()
      |> Enum.take(take)
      |> Task.async_stream(fn {f, l} ->
        {:ok, r} = OCR.read(image(dir, f), model: model, worker: w, lm: false, figures: false, tables: false)
        ref = norm(l[key])
        {OCR.levenshtein(String.graphemes(r.text), String.graphemes(ref)), String.length(ref)}
      end, timeout: :infinity, max_concurrency: 2)
      |> Enum.reduce({0, 0}, fn {:ok, {e, n}}, {a, b} -> {a + e, b + n} end)

    e / n
  end

  setup_all do
    {:ok, w: OCR.worker()}
  end

  test "Arabic: the inverse bidi of every label is python-bidi's; unseen typefaces read, the Latin reader fails", %{w: w} do
    dir = Path.join(@q, "arabic/test")
    for {f, l} <- lines(dir), do: assert(Bidi.logical(l["text"]) == l["logical"], f)

    {:ok, ar} = OCR.default(:arabic)
    {:ok, la} = OCR.default(:latin)
    assert ar.direction == :rtl
    v = cer(ar, dir, 30, "logical", w)
    c = cer(la, dir, 10, "logical", w)
    assert v < 0.25, "Arabic CER #{v}"
    assert c > 0.8, "control (Latin reader) CER #{c}"
  end

  test "Arabic: a line read through OCR.read comes back in logical order, with the visual order alongside", %{w: w} do
    dir = Path.join(@q, "arabic/test")
    {f, l} = dir |> lines() |> Enum.min_by(fn {_, l} -> String.length(l["text"]) end)
    {:ok, ar} = OCR.default(:arabic)
    {:ok, r} = OCR.read(image(dir, f), model: ar, worker: w, lm: false, figures: false, tables: false)
    [line] = r.lines
    assert line.direction == :rtl
    assert Bidi.logical(line.visual) == line.text
    assert OCR.levenshtein(String.graphemes(line.text), String.graphemes(norm(l["logical"]))) <= 0.3 * String.length(l["logical"])
  end

  test "Cyrillic: unseen typefaces read; the Latin reader on the same lines fails", %{w: w} do
    dir = Path.join(@q, "cyrillic/test")
    {:ok, cy} = OCR.default(:cyrillic)
    {:ok, la} = OCR.default(:latin)
    v = cer(cy, dir, 30, "text", w)
    c = cer(la, dir, 10, "text", w)
    assert v < 0.1, "Cyrillic CER #{v}"
    assert c > 0.6, "control (Latin reader) CER #{c}"
  end

  test "cursive Latin: no reader is shipped, and asking for one is refused with the reason" do
    assert {:error, %Vapor.Rejection{} = r} = OCR.default(:cursive)
    assert r.repair =~ "train one on real handwriting"
  end

  test "CJK: unseen typefaces within bounds; the language model helps text and does nothing to random characters", %{w: w} do
    for {lang, bound} <- [zh: 0.15, ja: 0.08, ko: 0.2] do
      {:ok, pack} = CJK.default(lang)

      cer = fn split, lm ->
        dir = Path.join([@q, "cjk", Atom.to_string(lang), split])

        {e, n} =
          Enum.reduce(lines(dir), {0, 0}, fn {f, %{"text" => t}}, {e, n} ->
            {:ok, r} = CJK.read(pack, image(dir, f), worker: w, lm: lm)
            {e + OCR.levenshtein(String.graphemes(r.text), String.graphemes(t)), n + String.length(t)}
          end)

        e / n
      end

      {t0, t1} = {cer.("test", false), cer.("test", true)}
      {r0, r1} = {cer.("random", false), cer.("random", true)}
      assert min(t0, t1) <= bound, "#{lang}: #{t0} → #{t1}"
      # where the docs say the model helps (ja, ko), it must; zh: measured, no gain claimed
      if lang in [:ja, :ko], do: assert(t1 < t0 - 0.01, "#{lang}: the language model must help: #{t0} → #{t1}")
      assert r1 - r0 <= 0.005, "#{lang}: random #{r0} → #{r1}"
    end
  end
end
