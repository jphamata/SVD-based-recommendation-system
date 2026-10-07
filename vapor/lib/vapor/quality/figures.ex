defmodule Vapor.Quality.Figures do
  @moduledoc """
  Scoring `Vapor.Vision.Figure` against the truth of rendered charts
  (`test/python/chart_render.py`: every series' points, the axis limits
  and scale) and pages (the figure's and the caption's boxes).

  Errors are in units of the axis span (in decades on a log axis), so a
  number means the same on every chart:

    * line — for every column read, |y − y_true(x)| where y_true is the
      polyline matplotlib drew (interpolated in the axis' own scale); a
      series passes when the median column is within 1 % (a thick line's
      centre departs from the polyline at its sharp corners: the mean and
      the 95th percentile are reported too);
    * bar — |value − true value| per bar, the count of bars must match;
    * scatter — recall and precision of points within 2 % of both spans.

  A series is matched to the truth by colour (nearest, within 60 RGB
  units); a true series with no match counts as missed.
  """

  @doc "Score one digitized chart: `%{ok, series: [...], missed}`."
  def chart({:error, why}, _truth), do: %{ok: false, refused: why, series: [], missed: :all}

  def chart({:ok, d}, t) do
    [y0, y1] = t["ylim"]
    [x0, x1] = t["xlim"]
    log = t["log_y"] == true
    yspan = if log, do: :math.log10(y1) - :math.log10(y0), else: y1 - y0
    xspan = x1 - x0

    scored =
      for ts <- t["series"] do
        case Enum.min_by(d.series, &cdist(&1.color, ts["color"]), fn -> nil end) do
          nil -> %{color: ts["color"], missed: true}
          s -> if cdist(s.color, ts["color"]) > 60, do: %{color: ts["color"], missed: true}, else: score(t["kind"], s, ts, {xspan, yspan, log})
        end
      end

    ok = scored != [] and Enum.all?(scored, &(not Map.get(&1, :missed, false) and &1.pass))
    %{ok: ok, series: scored, missed: Enum.count(scored, &Map.get(&1, :missed, false))}
  end

  defp score("line", %{kind: :line, points: ps}, ts, {_xspan, yspan, log}) do
    xs = ts["x"]
    ys = ts["y"]
    tr = Enum.zip(xs, if(log, do: Enum.map(ys, &:math.log10/1), else: ys))
    {lo, hi} = {List.first(xs), List.last(xs)}

    errs =
      for {x, y} <- ps, is_number(x) and is_number(y), x >= lo and x <= hi, (not log) or y > 0 do
        yv = if log, do: :math.log10(y), else: y
        abs(yv - interp(tr, x)) / yspan
      end

    mean = if errs == [], do: 1.0, else: Enum.sum(errs) / length(errs)
    sorted = Enum.sort(errs)
    p95 = if errs == [], do: 1.0, else: Enum.at(sorted, round(0.95 * (length(errs) - 1)))
    median = if errs == [], do: 1.0, else: Enum.at(sorted, div(length(errs), 2))
    %{kind: :line, mean: mean, median: median, p95: p95, n: length(errs), pass: errs != [] and median <= 0.01}
  end

  defp score("bar", %{kind: :bar, bars: bars}, ts, {_, yspan, _}) do
    truth = ts["y"]

    if length(bars) != length(truth) do
      %{kind: :bar, n: length(bars), expected: length(truth), max: 1.0, pass: false}
    else
      errs = Enum.zip_with(bars, truth, fn b, v -> abs(b.value - v) / yspan end)
      %{kind: :bar, n: length(bars), max: Enum.max(errs), mean: Enum.sum(errs) / length(errs), pass: Enum.max(errs) <= 0.02}
    end
  end

  defp score("scatter", %{kind: :scatter, points: ps}, ts, {xspan, yspan, _}) do
    truth = Enum.zip(ts["x"], ts["y"])
    near = fn {a, b}, set -> Enum.any?(set, fn {c, e} -> abs(a - c) / xspan <= 0.02 and abs(b - e) / yspan <= 0.02 end) end
    recall = Enum.count(truth, &near.(&1, ps)) / max(length(truth), 1)
    precision = Enum.count(ps, &near.(&1, truth)) / max(length(ps), 1)
    %{kind: :scatter, recall: recall, precision: precision, n: length(ps), expected: length(truth), pass: recall >= 0.9 and precision >= 0.9}
  end

  defp score(kind, s, _ts, _), do: %{kind: s.kind, expected: kind, pass: false}

  defp interp([{x0, y0} | _], x) when x <= x0, do: y0
  defp interp([{x0, y0}, {x1, y1} | _], x) when x <= x1, do: y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1.0e-12)
  defp interp([_ | rest], x) when length(rest) >= 2, do: interp(rest, x)
  defp interp([_, {_, y1}], _x), do: y1
  defp interp([{_, y}], _x), do: y

  defp cdist("#" <> a, "#" <> b) do
    <<r1, g1, b1>> = Base.decode16!(a, case: :mixed)
    <<r2, g2, b2>> = Base.decode16!(b, case: :mixed)
    :math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
  end

  @doc "Intersection over union of two boxes `{x0, y0, x1, y1}` (inclusive)."
  def iou({a0, b0, a1, b1}, {c0, d0, c1, d1}) do
    iw = min(a1, c1) - max(a0, c0) + 1
    ih = min(b1, d1) - max(b0, d0) + 1
    inter = if iw > 0 and ih > 0, do: iw * ih, else: 0
    inter / ((a1 - a0 + 1) * (b1 - b0 + 1) + (c1 - c0 + 1) * (d1 - d0 + 1) - inter)
  end
end
