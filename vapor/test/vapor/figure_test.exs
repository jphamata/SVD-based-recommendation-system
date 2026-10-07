defmodule Vapor.FigureTest do
  @moduledoc """
  Figures measured against the truth of charts and pages they were never
  tuned on (`priv/quality/figures`, rendered by `test/python/chart_render.py`
  with seeds the development set did not use):

    * `charts` — matplotlib's default style: the data within the stated
      error on most charts;
    * `test` — another typeface (a serif), grid lines, log axes, JPEG: the
      harder set, where what matters most is that a chart is read right or
      refused, never read wrong;
    * `shuffled` — the control: real charts with their tick labels permuted.
      Every one must be refused: a digitizer that reports numbers here
      reports numbers it cannot know;
    * `pages` — prose with a chart or a photograph and its caption.
  """
  use ExUnit.Case, async: false
  alias Vapor.Docs.Pictures
  alias Vapor.Quality.Figures, as: Q
  alias Vapor.Vision.{Figure, OCR, Segment}

  @moduletag timeout: 900_000
  @moduletag :native
  @dir Path.expand("../../priv/quality/figures", __DIR__)

  defp picture(path), do: elem(Pictures.read(:png, File.read!(path)), 1).image

  defp run(split) do
    {:ok, truth} = Vapor.JSON.decode(File.read!(Path.join([@dir, split, "truth.json"])))
    w = OCR.worker()

    truth
    |> Enum.sort()
    |> Task.async_stream(fn {f, t} -> r = Figure.digitize(picture(Path.join([@dir, split, f])), worker: w); {f, t, r, Q.chart(r, t)} end,
      timeout: :infinity, max_concurrency: 2)
    |> Enum.map(fn {:ok, x} -> x end)
  end

  # an accepted reading that is grossly wrong: a series missed or off by more than 5 % of the span
  defp gross?({_, _, {:ok, _}, s}) do
    Enum.any?(s.series, fn r ->
      Map.get(r, :missed, false) or (r[:median] || r[:max] || 0) > 0.05 or (r[:recall] || 1.0) < 0.5 or not is_integer(Map.get(r, :expected, 0))
    end)
  end

  defp gross?(_), do: false

  test "default-style charts: the data of most within 1 % of the axis span; none read grossly wrong" do
    res = run("charts")
    ok = Enum.count(res, fn {_, _, _, s} -> s.ok end)
    assert ok >= 24, "#{ok}/30 within tolerance"
    assert Enum.filter(res, &gross?/1) == []
  end

  test "the control: permuted tick labels are refused, every one" do
    res = run("shuffled")
    assert Enum.all?(res, fn {_, _, r, _} -> match?({:error, {kind, :y}} when kind in [:inconsistent_scale, :implausible_ticks, :unreadable_ticks], r) end),
           inspect(for {f, _, r, _} <- res, not match?({:error, _}, r), do: f)
  end

  test "the harder set (serif, grids, log axes, JPEG): read right or refused, never grossly wrong" do
    res = run("test")
    ok = Enum.count(res, fn {_, _, _, s} -> s.ok end)
    assert ok >= 15, "#{ok}/30"
    assert Enum.filter(res, &gross?/1) == []
  end

  test "pages: every figure found (IoU ≥ 0.9), charts told from photographs, captions read" do
    {:ok, truth} = Vapor.JSON.decode(File.read!(Path.join([@dir, "pages", "truth.json"])))
    w = OCR.worker()

    rows =
      for {f, t} <- Enum.sort(truth) do
        img = picture(Path.join([@dir, "pages", f]))
        comps = img |> Segment.gray() |> Segment.ink() |> Segment.components()
        {figs, _} = Figure.detect(comps, picture: img, worker: w)
        [a, b, c, d] = t["figure"]
        {f, t, figs, Enum.map(figs, &Q.iou(&1.box, {a, b, c, d}))}
      end

    for {f, t, figs, ious} <- rows do
      assert length(figs) == 1, "#{f}: #{length(figs)} figures"
      assert hd(ious) >= 0.9, "#{f}: IoU #{hd(ious)}"
      assert hd(figs).kind == if(t["kind"] == "chart", do: :chart, else: :picture)
    end

    {errs, total} =
      Enum.reduce(rows, {0, 0}, fn {_, t, [fig], _}, {e, n} ->
        ref = t["caption"]["text"]
        got = (fig.caption && fig.caption.text) || ""
        {e + OCR.levenshtein(String.graphemes(got), String.graphemes(ref)), n + String.length(ref)}
      end)

    assert errs / total <= 0.1, "caption CER #{errs / total}"
  end

  test "OCR.read sets figures apart: the caption is text, the tick labels are not" do
    {:ok, truth} = Vapor.JSON.decode(File.read!(Path.join([@dir, "pages", "truth.json"])))
    {f, t} = truth |> Enum.sort() |> Enum.find(fn {_, t} -> t["kind"] == "chart" end)
    img = picture(Path.join([@dir, "pages", f]))
    {:ok, with} = OCR.read(img, lm: false)
    {:ok, without} = OCR.read(img, lm: false, figures: false)
    assert [%{kind: :chart, caption: %{text: cap}}] = with.figures
    assert String.starts_with?(cap, String.slice(t["caption"]["text"], 0, 4))
    # the control: without figure detection, the chart's marks become "text" lines
    assert length(without.lines) > length(with.lines)
  end
end
