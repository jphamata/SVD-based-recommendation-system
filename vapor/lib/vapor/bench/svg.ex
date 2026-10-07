defmodule Vapor.Bench.SVG do
  @moduledoc """
  Two static chart forms for the benchmark report, as self-contained SVG:
  a log–log roofline (points + roof lines) and a line chart. Colors are
  roles (CSS custom properties) with light and dark values; every mark
  carries a `<title>` (the hover tooltip of an SVG viewer); series are
  direct-labelled and listed in a legend, so identity is never color alone.
  """

  @w 720
  @h 420
  @m %{l: 70, r: 150, t: 48, b: 56}

  @style """
  <style>
    svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df; --roof:#8a8984;
          --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; font-family: system-ui, sans-serif; }
    @media (prefers-color-scheme: dark) {
      svg { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --grid:#33322f; --roof:#8f8e88;
            --s1:#3987e5; --s2:#d95926; --s3:#199e70; }
    }
    .bg { fill: var(--surface); } .t { fill: var(--ink); font-size: 15px; font-weight: 600; }
    .lab { fill: var(--ink2); font-size: 12px; } .val { fill: var(--ink); font-size: 12px; }
    .grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--ink2); stroke-width: 1; }
    .roof { stroke: var(--roof); stroke-width: 2; fill: none; stroke-dasharray: 6 4; }
  </style>
  """

  defp frame(title, body),
    do: ~s[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 #{@w} #{@h}" role="img" aria-label="#{esc(title)}">#{@style}] <>
          ~s[<rect class="bg" width="#{@w}" height="#{@h}"/><text class="t" x="#{@m.l}" y="28">#{esc(title)}</text>] <> body <> "</svg>\n"

  @doc """
  Roofline: `points` = [{label, intensity (FLOP/B), GFLOP/s}], roofs from
  `bandwidth` (GB/s, measured) and `peak` (GFLOP/s).
  """
  def roofline(title, points, bandwidth, peak) do
    {x0, x1} = {0.01, 100.0}
    ys = Enum.map(points, &elem(&1, 2))
    {y0, y1} = {min(0.05, Enum.min(ys) / 2), max(peak * 2, Enum.max(ys) * 2)}
    px = fn x -> @m.l + (:math.log10(x) - :math.log10(x0)) / (:math.log10(x1) - :math.log10(x0)) * (@w - @m.l - @m.r) end
    py = fn y -> @h - @m.b - (:math.log10(y) - :math.log10(y0)) / (:math.log10(y1) - :math.log10(y0)) * (@h - @m.t - @m.b) end

    grid =
      (for e <- -2..2, x = :math.pow(10, e), do: ~s[<line class="grid" x1="#{px.(x)}" y1="#{@m.t}" x2="#{px.(x)}" y2="#{@h - @m.b}"/><text class="lab" x="#{px.(x)}" y="#{@h - @m.b + 18}" text-anchor="middle">#{fmt(x)}</text>]) ++
        (for e <- floor(:math.log10(y0))..ceil(:math.log10(y1)), y = :math.pow(10, e), y >= y0 and y <= y1,
             do: ~s[<line class="grid" x1="#{@m.l}" y1="#{py.(y)}" x2="#{@w - @m.r}" y2="#{py.(y)}"/><text class="lab" x="#{@m.l - 8}" y="#{py.(y) + 4}" text-anchor="end">#{fmt(y)}</text>])

    knee = peak / bandwidth
    roof_pts = [{x0, bandwidth * x0}, {knee, peak}, {x1, peak}] |> Enum.map(fn {x, y} -> "#{px.(x)},#{py.(y)}" end) |> Enum.join(" ")

    roofs =
      ~s[<polyline class="roof" points="#{roof_pts}"><title>roofs: #{fmt(bandwidth)} GB/s measured, #{fmt(peak)} GFLOP/s peak</title></polyline>] <>
        ~s[<text class="lab" x="#{px.(x1) + 6}" y="#{py.(peak) + 4}">#{fmt(peak)} GFLOP/s</text>] <>
        ~s[<text class="lab" x="#{px.(0.02)}" y="#{py.(bandwidth * 0.02) - 8}">#{fmt(bandwidth)} GB/s</text>]

    marks =
      points
      |> Enum.with_index()
      |> Enum.map(fn {{label, x, y}, i} ->
        ~s[<circle cx="#{px.(x)}" cy="#{py.(y)}" r="5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"><title>#{esc(label)}: #{fmt(x)} FLOP/B, #{fmt(y)} GFLOP/s</title></circle>] <>
          ~s[<text class="val" x="#{px.(x) + 9}" y="#{py.(y) + 4 + rem(i, 2) * 12 - 6}">#{esc(label)}</text>]
      end)

    axes =
      ~s[<text class="lab" x="#{(@m.l + @w - @m.r) / 2}" y="#{@h - 14}" text-anchor="middle">arithmetic intensity (FLOP per byte, log)</text>] <>
        ~s[<text class="lab" transform="translate(18 #{(@m.t + @h - @m.b) / 2}) rotate(-90)" text-anchor="middle">GFLOP/s (log)</text>]

    frame(title, Enum.join(grid) <> roofs <> Enum.join(marks) <> axes)
  end

  @doc "Line chart: `series` = [{name, [{x, y}]}] (≤ 3), categorical x positions `xs`."
  def lines(title, xlabel, ylabel, xs, series) do
    ymax = series |> Enum.flat_map(&elem(&1, 1)) |> Enum.map(&elem(&1, 1)) |> Enum.max() |> nice()
    n = length(xs)
    px = fn i -> @m.l + i / max(n - 1, 1) * (@w - @m.l - @m.r) end
    py = fn y -> @h - @m.b - y / ymax * (@h - @m.t - @m.b) end
    idx = xs |> Enum.with_index() |> Map.new()

    grid =
      (for k <- 0..4, y = ymax * k / 4,
           do: ~s[<line class="grid" x1="#{@m.l}" y1="#{py.(y)}" x2="#{@w - @m.r}" y2="#{py.(y)}"/><text class="lab" x="#{@m.l - 8}" y="#{py.(y) + 4}" text-anchor="end">#{fmt(y)}</text>]) ++
        (for {x, i} <- idx, do: ~s[<text class="lab" x="#{px.(i)}" y="#{@h - @m.b + 18}" text-anchor="middle">#{x}</text>])

    body =
      series
      |> Enum.with_index(1)
      |> Enum.map(fn {{name, pts}, s} ->
        color = "var(--s#{s})"
        path = pts |> Enum.map(fn {x, y} -> "#{px.(idx[x])},#{py.(y)}" end) |> Enum.join(" ")
        {lx, ly} = List.last(pts)

        ~s[<polyline points="#{path}" fill="none" stroke="#{color}" stroke-width="2"/>] <>
          Enum.map_join(pts, fn {x, y} ->
            ~s[<circle cx="#{px.(idx[x])}" cy="#{py.(y)}" r="4" fill="#{color}" stroke="var(--surface)" stroke-width="2"><title>#{esc(name)}, #{x}: #{fmt(y)}</title></circle>]
          end) <>
          ~s[<text class="val" x="#{px.(idx[lx]) + 10}" y="#{py.(ly) + 4}">#{esc(name)}</text>]
      end)

    legend =
      series
      |> Enum.with_index(1)
      |> Enum.map(fn {{name, _}, s} ->
        y = @m.t + 4 + (s - 1) * 18
        ~s[<rect x="#{@w - @m.r + 16}" y="#{y - 9}" width="12" height="3" rx="1" fill="var(--s#{s})"/><text class="lab" x="#{@w - @m.r + 34}" y="#{y - 4}">#{esc(name)}</text>]
      end)

    axes =
      ~s[<text class="lab" x="#{(@m.l + @w - @m.r) / 2}" y="#{@h - 14}" text-anchor="middle">#{esc(xlabel)}</text>] <>
        ~s[<text class="lab" transform="translate(18 #{(@m.t + @h - @m.b) / 2}) rotate(-90)" text-anchor="middle">#{esc(ylabel)}</text>]

    frame(title, Enum.join(grid) <> Enum.join(body) <> Enum.join(legend) <> axes)
  end

  defp nice(y) do
    e = :math.pow(10, floor(:math.log10(y)))
    Enum.find([1, 2, 2.5, 5, 10], &(&1 * e >= y)) * e
  end

  @doc false
  def fmt(x) when x >= 100, do: Integer.to_string(round(x))
  def fmt(x) when x >= 10, do: :erlang.float_to_binary(x * 1.0, decimals: 1)
  def fmt(x) when x >= 1, do: :erlang.float_to_binary(x * 1.0, decimals: 2)
  def fmt(x), do: :erlang.float_to_binary(x * 1.0, [{:decimals, 3}, :compact])

  defp esc(s), do: s |> to_string() |> String.replace("&", "&amp;") |> String.replace("<", "&lt;") |> String.replace(">", "&gt;")
end
