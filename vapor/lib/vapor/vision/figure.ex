defmodule Vapor.Vision.Figure do
  @moduledoc """
  **Figures on a page, and the numbers inside a chart.**

  Two problems, both answered from the geometry of print rather than from a
  trained detector:

  1. **Where are the figures, and what are their captions?** (`detect/2`)
     Text is made of marks no taller than about one and a half text
     heights; a chart's frame, a curve, a bar or a photograph's blobs are
     not. A mark spanning six text heights or more seeds a figure; the
     figure grows over every mark that touches it (tick labels, axis
     titles, a legend), never across the gap that separates it from a
     paragraph. Its caption is the nearest line below (or above) that
     *says* it is one — "Figure 3", "Fig. 2.", "Figura 1 —", "Gráfico",
     "图 2", "図", "그림", "شكل" — read by the OCR.
  2. **What are the data?** (`digitize/2`) A chart is a frame (two spines
     meeting at a corner), tick marks along the spines, tick labels read
     by the OCR, and marks of colour inside. The labels give the axis
     scale — but only if they agree with one: a linear or a logarithmic
     map from pixel to value is fitted to *every* label, and if more than
     one disagrees the chart is **refused** (`{:error,
     {:inconsistent_scale, axis}}`) rather than read with an invented
     scale. Inside the frame each colour is a series, classified by the
     shape of its marks: filled rectangles on a common base are bars,
     many small round blobs are a scatter, anything else is a line read
     column by column.

  Measured on matplotlib charts never used to tune it (another typeface,
  other DPIs, grid lines, log axes, JPEG), with the control of charts whose
  tick labels are permuted (which must be refused) and of photographs
  (which have no axes): `docs/OCR.md §3h`.
  """
  alias Vapor.Modal.Image
  alias Vapor.Vision.{OCR, Segment}

  # ---------------------------------------------------------------- pixels --

  # 8-bit RGB of an image, row-major, 3 bytes per pixel
  defp rgb(%Image{w: w, h: h, c: c, px: px}) do
    vals = Tuple.to_list(px)

    data =
      case c do
        3 -> for v <- vals, into: <<>>, do: <<clamp8(v * 255)>>
        1 -> for v <- vals, into: <<>>, do: (b = clamp8(v * 255); <<b, b, b>>)
        _ -> for vs <- Enum.chunk_every(vals, c), into: <<>>, do: (for v <- Enum.take(vs, 3), into: <<>>, do: <<clamp8(v * 255)>>)
      end

    %{w: w, h: h, data: data}
  end

  defp clamp8(v), do: v |> round() |> max(0) |> min(255)

  defp px(%{w: w, data: d}, x, y) do
    o = (y * w + x) * 3
    <<_::binary-size(o), r, g, b, _::binary>> = d
    {r, g, b}
  end

  defp lum({r, g, b}), do: (299 * r + 587 * g + 114 * b) / 1000
  defp sat({r, g, b}), do: max(r, max(g, b)) - min(r, min(g, b))

  # dark and grey: the ink of frames, ticks and labels
  defp dark?(p), do: lum(p) < 150 and sat(p) < 80

  @doc "Crop an image to `{x0, y0, x1, y1}` (inclusive)."
  def crop(%Image{w: w, c: c, px: px}, {x0, y0, x1, y1}) do
    vals = for y <- y0..y1, x <- x0..x1, ch <- 0..(c - 1), do: elem(px, (y * w + x) * c + ch)
    %Image{w: x1 - x0 + 1, h: y1 - y0 + 1, c: c, px: List.to_tuple(vals)}
  end

  # ------------------------------------------------------------------ axes --

  @doc """
  The frame of a chart: `{:ok, %{box: {left, top, right, bottom}, spines,
  x_ticks, y_ticks}}` (box on the inner edges of the spines; ticks as pixel
  centres) or `{:error, :no_axes}`.
  """
  def axes(%Image{} = img), do: axes_rgb(rgb(img))

  defp axes_rgb(%{w: w, h: h} = im) do
    dark = for y <- 0..(h - 1), into: %{}, do: {y, for(x <- 0..(w - 1), dark?(px(im, x, y)), into: MapSet.new(), do: x)}
    at = fn x, y -> y >= 0 and y < h and x >= 0 and x < w and MapSet.member?(dark[y], x) end

    rows = for y <- 0..(h - 1), {a, b} = longest_run(Enum.sort(MapSet.to_list(dark[y]))), b - a + 1 >= 0.35 * w, do: {y, a, b}
    cols = for x <- 0..(w - 1), {a, b} = longest_run(for(y <- 0..(h - 1), at.(x, y), do: y)), b - a + 1 >= 0.3 * h, do: {x, a, b}

    with [_ | _] <- rows, [_ | _] <- cols do
      bottom = rows |> groups() |> List.last()
      left = cols |> groups() |> hd()
      {yb0, yb1, bx0, bx1} = bottom
      {xl0, xl1, ly0, ly1} = left

      # an L: the left spine reaches the bottom one, the bottom one starts
      # at the left one (a tick mark may prolong either by its length)
      if ly1 >= yb0 - 3 and ly1 <= yb1 + 12 and bx0 >= xl0 - 12 and bx0 <= xl1 + 3 do
        top = rows |> groups() |> Enum.find(fn {y0, _, a, b} -> y0 < yb0 - 0.2 * h and a >= xl0 - 12 and a <= xl1 + 3 and abs(b - bx1) <= 12 end)
        right = cols |> groups() |> Enum.filter(fn {x0, _, _, b} -> x0 > xl1 + 0.3 * (bx1 - bx0) and b >= yb0 - 3 and b <= yb1 + 12 end) |> List.last()
        {yt0, yt1} = if top && elem(top, 0) < yb0, do: {elem(top, 0), elem(top, 1)}, else: {ly0, ly0 - 1}
        {xr0, xr1} = if right, do: {elem(right, 0), elem(right, 1)}, else: {bx1 + 1, bx1}
        box = {xl1 + 1, yt1 + 1, xr0 - 1, yb0 - 1}

        x_ticks = ticks(for(x <- xl0..max(xr1, xl0), at.(x, yb1 + 1) and at.(x, yb1 + 2), do: x), fn x -> run_len(at, x, yb1 + 1, 0, 1) end)
        y_ticks = ticks(for(y <- min(yt0, ly0)..yb1, at.(xl0 - 1, y) and at.(xl0 - 2, y), do: y), fn y -> run_len(at, xl0 - 1, y, -1, 0) end)

        {:ok, %{box: box, spines: %{bottom: {yb0, yb1}, left: {xl0, xl1}, top: top && {yt0, yt1}, right: right && {xr0, xr1}},
                x_ticks: x_ticks, y_ticks: y_ticks, size: {w, h}}}
      else
        {:error, :no_axes}
      end
    else
      _ -> {:error, :no_axes}
    end
  end

  # longest run of consecutive integers (gaps of one allowed: antialiasing)
  defp longest_run([]), do: {0, -1}

  defp longest_run([x | xs]) do
    {best, cur} =
      Enum.reduce(xs, {{x, x}, {x, x}}, fn v, {best, {a, b}} ->
        cur = if v - b <= 2, do: {a, v}, else: {v, v}
        {if(span(cur) > span(best), do: cur, else: best), cur}
      end)

    if span(cur) > span(best), do: cur, else: best
  end

  defp span({a, b}), do: b - a

  # adjacent lines of the same spine, as {first, last, run start, run end}
  defp groups(lines) do
    lines
    |> Enum.chunk_while([], fn {i, _, _} = l, acc ->
      case acc do
        [{j, _, _} | _] when i == j + 1 -> {:cont, [l | acc]}
        [] -> {:cont, [l]}
        _ -> {:cont, Enum.reverse(acc), [l]}
      end
    end, fn acc -> {:cont, Enum.reverse(acc), []} end)
    |> Enum.reject(&(&1 == []))
    |> Enum.map(fn ls -> {elem(hd(ls), 0), elem(List.last(ls), 0), ls |> Enum.map(&elem(&1, 1)) |> Enum.min(), ls |> Enum.map(&elem(&1, 2)) |> Enum.max()} end)
  end

  defp run_len(at, x, y, dx, dy), do: if(at.(x, y), do: 1 + run_len(at, x + dx, y + dy, dx, dy), else: 0)

  # tick marks: groups of adjacent positions, their centres (a tick is short: at most a few pixels long)
  defp ticks(pos, len) do
    pos
    |> Enum.filter(&(len.(&1) <= 12))
    |> Enum.chunk_while([], fn p, acc ->
      case acc do
        [q | _] when p == q + 1 -> {:cont, [p | acc]}
        [] -> {:cont, [p]}
        _ -> {:cont, Enum.reverse(acc), [p]}
      end
    end, fn acc -> {:cont, Enum.reverse(acc), []} end)
    |> Enum.reject(&(&1 == [] or length(&1) > 4))
    |> Enum.map(fn g -> Enum.sum(g) / length(g) end)
  end

  # ------------------------------------------------------------ digitizing --

  @doc """
  Read the data of a chart image: `{:ok, %{axes, x, y, series}}` where `x`
  and `y` are `%{scale: :linear | :log | :categorical, ticks: [%{at, text,
  value}], rejected}` and each series is `%{color, kind: :line | :bar |
  :scatter, points: [{x, y}]}` (bars: `%{x, label, value}` in `bars`).
  Refusals: `{:error, :no_axes}`, `{:error, {:unreadable_ticks, axis}}`,
  `{:error, {:inconsistent_scale, axis}}`, `{:error, :no_series}`.
  Options: `model` (the OCR reader, default `OCR.default/0`), `worker`.
  """
  def digitize(%Image{} = img, opts \\ []) do
    im = rgb(img)

    with {:ok, ax} <- axes_rgb(im),
         {:ok, model} <- (if opts[:model], do: {:ok, opts[:model]}, else: OCR.default()),
         w = Keyword.get_lazy(opts, :worker, &OCR.worker/0),
         labels = tick_labels(img, ax, model, w),
         {:numeric, yl} <- values(labels.y) |> then(fn {:numeric, _} = v -> v; _ -> {:error, {:unreadable_ticks, :y}} end),
         {:ok, ys} <- scale(:y, yl, ax.y_ticks),
         {:ok, xs} <- x_scale(labels.x, ax.x_ticks),
         [_ | _] = series <- series(im, ax, xs, ys) || {:error, :no_series} do
      {:ok, %{axes: ax.box, x: xs, y: ys, series: series}}
    else
      [] -> {:error, :no_series}
      err -> err
    end
  end

  @doc false
  # the tick labels as read (inspection, tests)
  def read_ticks(%Image{} = img, opts \\ []) do
    with {:ok, ax} <- axes(img), {:ok, model} <- OCR.default() do
      tick_labels(img, ax, model, Keyword.get_lazy(opts, :worker, &OCR.worker/0)) |> Map.put(:axes, ax)
    end
  end

  @numeric MapSet.new(String.graphemes("0123456789.-"))

  # tick labels: the text marks just below the bottom spine (x) and just
  # left of the left spine (y), grouped into labels and read
  defp tick_labels(img, ax, model, w) do
    g = Segment.gray(img)
    mask = Segment.ink(g)
    comps = Segment.components(mask) |> Enum.filter(&(&1.area >= 2))
    {left, top, right, bottom} = ax.box
    {xl0, _} = ax.spines.left
    {_, yb1} = ax.spines.bottom
    {iw, ih} = ax.size

    # y labels: rows of marks left of the spine whose middle is within the
    # axis (the first x label may also start left of the spine, lower down)
    beside =
      comps
      |> Enum.filter(fn %{box: {_, y0, x1, y1}} -> x1 < xl0 and y1 >= top - 0.06 * ih and y0 <= bottom + 0.06 * ih end)
      |> rows()
      |> Enum.filter(fn cs -> {_, a, _, b} = hull(cs); (a + b) / 2 <= yb1 + 3 end)
      |> Enum.concat()
    # (the lowest y label hangs below the spine: its marks are not x labels)
    below = Enum.filter(comps -- beside, fn %{box: {x0, y0, x1, _}} -> y0 > yb1 and x1 >= left - 0.08 * iw and x0 <= right + 0.08 * iw end)

    x =
      case below do
        [] -> []
        _ ->
          t = below |> Enum.map(fn %{box: {_, y0, _, _}} -> y0 end) |> Enum.min()
          first = Enum.filter(below, fn %{box: {_, y0, _, _}} -> y0 <= t + 3 end)
          hh = first |> Enum.map(fn %{box: {_, y0, _, y1}} -> y1 - y0 + 1 end) |> Enum.max()
          band = Enum.filter(below, fn %{box: {_, y0, _, _}} -> y0 <= t + 0.5 * hh end)
          band |> split_by(fn %{box: {x0, _, x1, _}} -> {x0, x1} end, 0.6 * hh) |> Enum.map(&read_label(&1, g, model, w, :x)) |> read_axis(g, model, w)
      end

    y =
      case rows(beside) do
        [] -> []
        rs ->
          runs = Enum.map(rs, fn cs ->
            hh = cs |> Enum.map(fn %{box: {_, y0, _, y1}} -> y1 - y0 + 1 end) |> Enum.max()
            # the label next to the spine: the rightmost run of marks in the row
            cs |> split_by(fn %{box: {x0, _, x1, _}} -> {x0, x1} end, 0.6 * hh) |> List.last()
          end)

          edge = runs |> Enum.map(&right_edge/1) |> Enum.max()
          hh = runs |> Enum.map(fn cs -> cs |> Enum.map(fn %{box: {_, y0, _, y1}} -> y1 - y0 + 1 end) |> Enum.max() end) |> Enum.sort() |> then(&Enum.at(&1, div(length(&1), 2)))
          # labels are aligned on the spine: an axis title (often rotated) lies further out
          runs |> Enum.filter(&(right_edge(&1) >= edge - 1.5 * hh)) |> Enum.map(&read_label(&1, g, model, w, :y)) |> Enum.sort_by(& &1.at) |> read_axis(g, model, w)
      end

    %{x: x, y: y}
  end

  # rows of marks: chains of marks overlapping vertically, top to bottom
  defp rows(cs) do
    cs
    |> Enum.sort_by(fn %{box: {_, y0, _, _}} -> y0 end)
    |> Enum.chunk_while({nil, []}, fn %{box: {_, y0, _, y1}} = c, {bottom, acc} ->
      if bottom == nil or y0 <= bottom, do: {:cont, {max(bottom || y1, y1), [c | acc]}}, else: {:cont, Enum.reverse(acc), {y1, [c]}}
    end, fn {_, acc} -> {:cont, Enum.reverse(acc), nil} end)
    |> Enum.reject(&(&1 == []))
  end

  defp hull(cs), do: cs |> Enum.map(& &1.box) |> Enum.reduce(&union/2)

  defp right_edge(cs), do: cs |> Enum.map(fn %{box: {_, _, x1, _}} -> x1 end) |> Enum.max()

  defp split_by(cs, span, gap) do
    cs
    |> Enum.sort_by(&elem(span.(&1), 0))
    |> Enum.chunk_while([], fn c, acc ->
      case acc do
        [] -> {:cont, [c]}
        _ ->
          right = acc |> Enum.map(&elem(span.(&1), 1)) |> Enum.max()
          if elem(span.(c), 0) - right - 1 > gap, do: {:cont, Enum.reverse(acc), [c]}, else: {:cont, [c | acc]}
      end
    end, fn acc -> {:cont, Enum.reverse(acc), []} end)
    |> Enum.reject(&(&1 == []))
  end

  # A tick label is small (8–12 px digits at 72–100 dpi): it is read from
  # its own patch of the grey image enlarged to a 24 px line, where the
  # reader is at home, not from the page's ink map.
  defp read_label(cs, g, model, w, axis) do
    {x0, y0, x1, y1} = cs |> Enum.map(& &1.box) |> Enum.reduce(fn {a0, b0, a1, b1}, {c0, d0, c1, d1} -> {min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)} end)
    k = max(1, min(4, round(24 / max(y1 - y0 + 1, 1))))
    pad = 3
    patch = g |> gray_crop({max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, g.w - 1), min(y1 + pad, g.h - 1)}) |> enlarge(k)
    pcs = patch |> Segment.ink() |> Segment.components() |> Enum.filter(&(&1.area >= 2 * k))
    free = OCR.read_components(pcs, model, w, nil, marks: true)
    digits = OCR.read_components(pcs, model, w, nil, marks: true, allowed: @numeric)

    glyphs =
      pcs
      |> column_glyphs()
      |> Enum.map(&%{box: &1, feat: glyph_feature(patch, &1)})
      |> superscripts()

    %{at: if(axis == :x, do: (x0 + x1) / 2, else: (y0 + y1) / 2), text: String.trim(free.text), number: parse(digits.text),
      digits: String.replace(digits.text, " ", ""), confidence: digits.confidence, box: {x0, y0, x1, y1}, glyphs: glyphs}
  end

  # An axis read as one sentence: the labels' enlarged patches side by
  # side, two heights apart, read as a single line — a lone "4" is a line
  # the reader never met in training; "1  2  3  4  5  6" is an ordinary
  # one, with a baseline and a text height measured over every label. Each
  # character goes back to the label under it. Where the sentence cannot
  # be formed or read (more than one line, too long), the labels keep
  # their own readings.
  defp read_axis([], _g, _model, _w), do: []

  defp read_axis(labels, g, model, w) do
    hs = Enum.map(labels, fn %{box: {_, y0, _, y1}} -> y1 - y0 + 1 end)
    hmax = Enum.max(hs)
    k = max(1, min(4, round(24 / hmax)))
    patches = Enum.map(labels, fn %{box: {x0, y0, x1, y1}} -> g |> gray_crop({max(x0 - 2, 0), max(y0 - 2, 0), min(x1 + 2, g.w - 1), min(y1 + 2, g.h - 1)}) |> enlarge(k) end)
    gap = 2 * (hmax + 4) * k
    ch = Enum.max(Enum.map(patches, & &1.h))
    cw = Enum.sum(Enum.map(patches, & &1.w)) + gap * (length(patches) + 1)

    {spans, _} = Enum.map_reduce(patches, gap, fn p, x -> {{x, x + p.w - 1}, x + p.w + gap} end)

    if cw * 12 / max(hmax * k, 1) > 1000 and length(labels) > 1 do
      # too long for one line of the reader: two sentences
      {a, b} = Enum.split(labels, div(length(labels), 2))
      read_axis(a, g, model, w) ++ read_axis(b, g, model, w)
    else
      canvas = compose(patches, spans, cw, ch)

      case canvas |> Segment.ink() |> Segment.components() |> Segment.lines() do
        [line] ->
          free = per_label(OCR.read_line(line, model, w, nil), line, spans)
          digits = per_label(OCR.read_line(line, model, w, nil, allowed: @numeric), line, spans)

          labels
          |> Enum.with_index()
          |> Enum.map(fn {l, i} ->
            # a second opinion, weighed by `consensus/1` with the label's own reading
            Map.merge(l, %{text2: Map.get(free, i, ""), digits2: Map.get(digits, i, "")})
          end)

        _ ->
          labels
      end
    end
  end

  # patches bottom-aligned on a white canvas
  defp compose(patches, spans, cw, ch) do
    rows =
      for y <- 0..(ch - 1) do
        Enum.zip(patches, spans)
        |> Enum.reduce({<<>>, 0}, fn {p, {x0, _}}, {acc, x} ->
          pad = :binary.copy(<<255>>, x0 - x)
          yy = y - (ch - p.h)
          row = if yy >= 0, do: binary_part(p.gray, yy * p.w, p.w), else: :binary.copy(<<255>>, p.w)
          {acc <> pad <> row, x0 + p.w}
        end)
        |> then(fn {acc, x} -> acc <> :binary.copy(<<255>>, cw - x) end)
      end

    %{w: cw, h: ch, gray: IO.iodata_to_binary(rows)}
  end

  # the characters of a reading, by the label their frames fall on
  defp per_label(read, line, spans) do
    {x0, _, _, _} = line.box
    s = 12 / line.ref

    read.chars
    |> Enum.reject(&(&1.char == " " or &1.frames == []))
    |> Enum.group_by(
      fn c ->
        x = x0 + (2 * Enum.sum(c.frames) / length(c.frames) + 4 - 2) / s
        spans |> Enum.with_index() |> Enum.min_by(fn {{a, b}, _} -> if x < a, do: a - x, else: if(x > b, do: x - b, else: 0) end) |> elem(1)
      end,
      & &1.char
    )
    |> Map.new(fn {i, cs} -> {i, Enum.join(cs)} end)
  end

  defp gray_crop(%{w: w, gray: g}, {x0, y0, x1, y1}) do
    cw = x1 - x0 + 1
    data = for y <- y0..y1, into: <<>>, do: binary_part(g, y * w + x0, cw)
    %{w: cw, h: y1 - y0 + 1, gray: data}
  end

  # bilinear enlargement by an integer factor
  defp enlarge(img, 1), do: img

  defp enlarge(%{w: w, h: h, gray: g}, k) do
    at = fn x, y -> :binary.at(g, min(y, h - 1) * w + min(x, w - 1)) end

    data =
      for yy <- 0..(h * k - 1), into: <<>> do
        fy = max((yy + 0.5) / k - 0.5, 0.0)
        y = trunc(fy)
        ty = fy - y

        for xx <- 0..(w * k - 1), into: <<>> do
          fx = max((xx + 0.5) / k - 0.5, 0.0)
          x = trunc(fx)
          tx = fx - x
          v = (at.(x, y) * (1 - tx) + at.(x + 1, y) * tx) * (1 - ty) + (at.(x, y + 1) * (1 - tx) + at.(x + 1, y + 1) * tx) * ty
          <<round(v)>>
        end
      end

    %{w: w * k, h: h * k, gray: data}
  end

  # the glyphs of a label: runs of inked columns (a thin 0 broken in two
  # arcs by the binarisation is still one glyph; digits of a tick label do
  # not touch), each with the rows its marks cover
  defp column_glyphs(cs) do
    cs
    |> Enum.sort_by(fn %{box: {a, _, _, _}} -> a end)
    |> Enum.reduce([], fn %{box: b}, acc ->
      case acc do
        [{a0, b0, a1, b1} | rest] ->
          {c0, d0, c1, d1} = b
          if c0 <= a1, do: [{a0, min(b0, d0), max(a1, c1), max(b1, d1)} | rest], else: [b | acc]

        [] ->
          [b]
      end
    end)
    |> Enum.reverse()
  end

  # a glyph's darkness on a 10×14 grid of its box, unit length (with its size relative to the patch, so "." ≠ "0")
  defp glyph_feature(%{w: w, h: h, gray: g}, {x0, y0, x1, y1}) do
    {gw, gh} = {10, 14}
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1

    cells =
      for cy <- 0..(gh - 1), cx <- 0..(gw - 1) do
        ya = y0 + div(cy * bh, gh)
        yb = max(ya, y0 + div((cy + 1) * bh, gh) - 1)
        xa = x0 + div(cx * bw, gw)
        xb = max(xa, x0 + div((cx + 1) * bw, gw) - 1)
        vs = for y <- ya..min(yb, h - 1)//1, x <- xa..min(xb, w - 1)//1, do: 255 - :binary.at(g, y * w + x)
        if vs == [], do: 0.0, else: Enum.sum(vs) / length(vs)
      end

    n = :math.sqrt(Enum.reduce(cells, 0.0, &(&1 * &1 + &2)))
    {if(n > 0, do: Enum.map(cells, &(&1 / n)), else: cells), {bw / h, bh / h}}
  end

  # Tick labels of one axis are set in one typeface: the same digit is the
  # same picture every time it occurs. Glyphs are clustered by picture, and
  # each cluster takes the character the reader most often gives it where
  # a label's reading has as many characters as the label has glyphs; a
  # label whose reading dropped or doubled a character is then re-read
  # glyph by glyph. (A reader that drops the narrow "1" of "1990" in one
  # label reads it in "2010": the cluster knows.)
  defp consensus(labels) do
    glyphs = for {l, i} <- Enum.with_index(labels), {gl, j} <- Enum.with_index(l.glyphs), do: {{i, j}, gl.feat}
    cluster = cluster_glyphs(glyphs)
    # the label's own reading weighs two, the sentence's one
    readings = fn l -> Enum.reject([{l.digits, 2}, {Map.get(l, :digits2), 1}], fn {r, _} -> r in [nil, ""] end) end

    # punctuation is told by its shape and place: a point is a small mark
    # on the baseline, a minus a short bar at mid-height — a reader that
    # drops the point of "32.5" would otherwise make every label ten times
    # larger and the scale would still fit
    punct =
      for {l, i} <- Enum.with_index(labels), {gl, j} <- Enum.with_index(l.glyphs), p = gl.punct, into: %{}, do: {cluster[{i, j}], p}

    votes =
      for {l, i} <- Enum.with_index(labels), {r, wt} <- readings.(l), String.length(r) == length(l.glyphs), {ch, j} <- Enum.with_index(String.graphemes(r)), reduce: %{} do
        acc -> Map.update(acc, cluster[{i, j}], %{ch => wt}, &Map.update(&1, ch, wt, fn n -> n + wt end))
      end

    char = votes |> Map.new(fn {c, vs} -> {c, vs |> Enum.max_by(&elem(&1, 1)) |> elem(0)} end) |> Map.merge(punct)

    # readings of another length: aligned with the glyphs (known glyphs
    # must match their character, unknown ones take the character aligned
    # with them), which names the unknown clusters
    votes2 =
      for {l, i} <- Enum.with_index(labels), {r, wt} <- readings.(l), String.length(r) != length(l.glyphs), l.glyphs != [], reduce: %{} do
        acc ->
          known = for j <- 0..(length(l.glyphs) - 1), do: {cluster[{i, j}], char[cluster[{i, j}]]}
          Enum.reduce(align(known, String.graphemes(r)), acc, fn {c, ch}, a -> Map.update(a, c, %{ch => wt}, &Map.update(&1, ch, wt, fn n -> n + wt end)) end)
      end

    char = Map.merge(Map.new(votes2, fn {c, vs} -> {c, vs |> Enum.max_by(&elem(&1, 1)) |> elem(0)} end), char)

    labels
    |> Enum.with_index()
    |> Enum.map(fn {l, i} ->
      read = for j <- 0..(length(l.glyphs) - 1)//1, do: char[cluster[{i, j}]]

      if l.glyphs != [] and Enum.all?(read, & &1) do
        t = Enum.join(read)
        %{l | number: number(t, Enum.map(l.glyphs, & &1.sup)), digits: t}
      else
        l
      end
    end)
  end

  # a label's value from its glyphs' characters: "10" followed by raised
  # glyphs is a power of ten (matplotlib's log axes: 10⁻¹, 10⁰, 10¹)
  defp number(t, sups) do
    chars = String.graphemes(t)

    case Enum.split_while(Enum.zip(chars, sups), fn {_, sup} -> not sup end) do
      {[_ | _] = base, [_ | _] = exp} ->
        b = parse(Enum.map_join(base, &elem(&1, 0)))
        e = parse(Enum.map_join(exp, &elem(&1, 0)))
        if b && e && Enum.all?(exp, &elem(&1, 1)), do: :math.pow(b, e)

      _ ->
        parse(t)
    end
  end

  # raised glyphs: bottom well above the line's baseline, and smaller
  defp superscripts(glyphs) do
    case glyphs do
      [] -> []
      _ ->
        base = glyphs |> Enum.map(fn %{box: {_, _, _, y1}} -> y1 end) |> Enum.max()
        hmax = glyphs |> Enum.map(fn %{box: {_, y0, _, y1}} -> y1 - y0 + 1 end) |> Enum.max()
        top = glyphs |> Enum.map(fn %{box: {_, y0, _, _}} -> y0 end) |> Enum.min()

        Enum.map(glyphs, fn %{box: {x0, y0, x1, y1}} = g ->
          {gw, gh} = {x1 - x0 + 1, y1 - y0 + 1}
          mid = (y0 + y1) / 2

          punct =
            cond do
              gh <= 0.3 * hmax and gw <= 0.45 * hmax and base - y1 <= 0.2 * hmax -> "."
              gh <= 0.3 * hmax and gw >= 0.3 * hmax and mid > top + 0.25 * hmax and mid < base - 0.15 * hmax -> "-"
              true -> nil
            end

          g |> Map.put(:sup, punct == nil and base - y1 > 0.3 * hmax and gh < 0.85 * hmax) |> Map.put(:punct, punct)
        end)
    end
  end

  # edit-distance alignment of glyphs (cluster, known char or nil) with a
  # reading; returns {cluster, char} for the unknown glyphs matched to a char
  defp align(glyphs, chars) do
    g = List.to_tuple(glyphs)
    c = List.to_tuple(chars)
    {n, m} = {tuple_size(g), tuple_size(c)}
    sub = fn i, j -> (case elem(g, i - 1) do {_, nil} -> 0; {_, k} -> if(k == elem(c, j - 1), do: 0, else: 1) end) end

    table =
      for i <- 0..n, j <- 0..m, reduce: %{} do
        t ->
          v =
            cond do
              i == 0 -> {j, :ins}
              j == 0 -> {i, :del}
              true -> Enum.min_by([{elem(t[{i - 1, j - 1}], 0) + sub.(i, j), :sub}, {elem(t[{i - 1, j}], 0) + 1, :del}, {elem(t[{i, j - 1}], 0) + 1, :ins}], &elem(&1, 0))
            end

          Map.put(t, {i, j}, v)
      end

    Stream.unfold({n, m}, fn
      {0, 0} -> nil
      {i, j} ->
        case elem(table[{i, j}], 1) do
          :sub -> {(case elem(g, i - 1) do {k, nil} -> {k, elem(c, j - 1)}; _ -> nil end), {i - 1, j - 1}}
          :del -> {nil, {i - 1, j}}
          :ins -> {nil, {i, j - 1}}
        end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp cluster_glyphs(glyphs) do
    {_, assign} =
      Enum.reduce(glyphs, {[], %{}}, fn {key, {f, {rw, rh}}}, {centres, assign} ->
        match =
          centres
          |> Enum.with_index()
          |> Enum.find(fn {{cf, {cw, ch}}, _} -> cos(cf, f) >= 0.93 and abs(cw - rw) <= 0.12 and abs(ch - rh) <= 0.12 end)

        case match do
          {_, k} -> {centres, Map.put(assign, key, k)}
          nil -> {centres ++ [{f, {rw, rh}}], Map.put(assign, key, length(centres))}
        end
      end)

    assign
  end

  defp cos(a, b), do: Enum.zip_with(a, b, &(&1 * &2)) |> Enum.sum()

  # An axis is numeric when most of its labels hold a digit in the free
  # reading; its values are then the readings restricted to digits, sign
  # and point (the free reader takes a small 0 for an O or a parenthesis).
  defp values(labels) do
    numeric = Enum.count(labels, &Regex.match?(~r/[0-9]/, &1.text <> Map.get(&1, :text2, "")))

    if labels != [] and numeric >= 0.5 * length(labels),
      do: {:numeric, labels |> consensus() |> Enum.map(&Map.put(&1, :value, &1.number))},
      else: {:categorical, Enum.map(labels, &Map.put(&1, :value, nil))}
  end

  defp parse(t) do
    s = t |> String.replace(" ", "") |> String.replace(~r/^[-−–—]+/u, "-")

    case Float.parse(s) do
      {v, ""} -> v
      _ -> nil
    end
  end

  # the x axis may be categorical (bar charts): labels that are not numbers
  defp x_scale(labels, ticks) do
    case values(labels) do
      {:categorical, ls} -> {:ok, %{scale: :categorical, ticks: Enum.map(ls, &Map.put(&1, :at, snap(&1.at, ticks))), rejected: []}}
      {:numeric, ls} -> scale(:x, ls, ticks)
    end
  end

  # a label sits on its tick: use the tick's position when one is near
  defp snap(at, []), do: at

  defp snap(at, ticks) do
    t = Enum.min_by(ticks, &abs(&1 - at))
    if abs(t - at) <= 6, do: t, else: at
  end

  @doc false
  # Fit value = a + b·pixel (linear) or log10(value) = a + b·pixel (log) to
  # every read label; at most one label may disagree (an OCR slip), else
  # the axis is refused.
  def scale(axis, labels, ticks) do
    pts = for %{value: v, at: at} = l <- labels, v != nil, do: {snap(at, ticks), v, l}

    cond do
      length(pts) == 2 ->
        two_decades(axis, pts, ticks)

      length(pts) < 3 ->
        {:error, {:unreadable_ticks, axis}}

      true ->
      fits =
        [:linear, :log]
        |> Enum.flat_map(fn kind ->
          ps = for {p, v, l} <- pts, kind == :linear or v > 0, do: {p, if(kind == :log, do: :math.log10(v), else: v), l}
          if length(ps) >= 3, do: [fit(kind, ps)], else: []
        end)
        |> Enum.filter(& &1)

      need = max(3, length(pts) - 1)

      # linear unless the logarithmic scale explains more labels (equally spaced
      # powers of ten do not fit a line; a tie is never a reason for a log axis)
      case fits |> Enum.filter(&(length(&1.inliers) >= need)) |> Enum.sort_by(&{-length(&1.inliers), &1.kind != :linear}) do
        [best | _] ->
          if nice?(best),
            do: {:ok, %{scale: best.kind, a: best.a, b: best.b, ticks: Enum.map(best.inliers, &elem(&1, 2)), rejected: Enum.map(best.outliers, &elem(&1, 2))}},
            else: {:error, {:implausible_ticks, axis}}

        [] ->
          {:error, {:inconsistent_scale, axis}}
      end
    end
  end

  # A log axis spanning one decade shows two labels (10¹, 10²) — too few to
  # check a scale. The unlabeled minor ticks check it: on a true log axis
  # they fall at log10(2..9) of the way between the two powers of ten. At
  # least three of them must be where the scale puts them (±1.5 px), or the
  # axis is refused.
  defp two_decades(axis, [{p1, v1, l1}, {p2, v2, l2}], ticks) do
    {e1, e2} = {:math.log10(abs(v1) + 1.0e-300), :math.log10(abs(v2) + 1.0e-300)}

    if v1 > 0 and v2 > 0 and abs(e1 - round(e1)) < 1.0e-9 and abs(e2 - round(e2)) < 1.0e-9 and abs(round(e2) - round(e1)) == 1 do
      b = (e2 - e1) / (p2 - p1)
      a = e1 - b * p1
      lo = min(e1, e2)
      minor = for m <- 2..9, do: (lo + :math.log10(m) - a) / b
      hits = Enum.count(minor, fn q -> Enum.any?(ticks, &(abs(&1 - q) <= 1.5)) end)

      if hits >= 3,
        do: {:ok, %{scale: :log, a: a, b: b, ticks: [l1, l2], rejected: [], minor_ticks_checked: hits}},
        else: {:error, {:unreadable_ticks, axis}}
    else
      {:error, {:unreadable_ticks, axis}}
    end
  end

  # A plotting library puts linear ticks at multiples of their step (0, 20,
  # 40 — never 1, 21, 41): a set of labels that fits a line but not that
  # rule is a reading shifted by one consistent misread glyph (a 0 taken
  # for a 1 everywhere), which the line fit alone would accept. Log axes:
  # each label a power of ten times 1, 2 or 5.
  defp nice?(%{kind: :linear, inliers: inl}) do
    vs = inl |> Enum.map(&elem(&1, 1)) |> Enum.sort()
    steps = vs |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> b - a end) |> Enum.sort()
    step = Enum.at(steps, div(length(steps), 2))
    step > 0 and Enum.all?(vs, fn v -> q = v / step; abs(q - round(q)) <= 0.01 end)
  end

  defp nice?(%{kind: :log, inliers: inl}) do
    Enum.all?(inl, fn {_, lv, _} ->
      m = :math.pow(10, lv - Float.floor(lv))
      Enum.any?([1.0, 2.0, 5.0, 10.0], &(abs(m - &1) <= 0.02 * &1))
    end)
  end

  # the pair of labels whose line explains the most labels (within 1.5 px), refitted by least squares on its inliers
  defp fit(kind, ps) do
    cands =
      for {p1, v1, _} <- ps, {p2, v2, _} <- ps, p2 > p1 + 2, v2 != v1 do
        b = (v2 - v1) / (p2 - p1)
        a = v1 - b * p1
        {inl, outl} = Enum.split_with(ps, fn {p, v, _} -> abs((v - a) / b - p) <= 1.5 end)
        {inl, outl}
      end

    case Enum.max_by(cands, fn {inl, _} -> length(inl) end, fn -> nil end) do
      nil -> nil
      {inl, outl} ->
        {a, b} = least_squares(Enum.map(inl, fn {p, v, _} -> {p, v} end))
        res = inl |> Enum.map(fn {p, v, _} -> abs((v - a) / b - p) end) |> Enum.max()
        %{kind: kind, a: a, b: b, inliers: inl, outliers: outl, residual: res}
    end
  end

  defp least_squares(pts) do
    n = length(pts)
    {sx, sy} = Enum.reduce(pts, {0.0, 0.0}, fn {x, y}, {a, b} -> {a + x, b + y} end)
    {mx, my} = {sx / n, sy / n}
    sxx = Enum.reduce(pts, 0.0, fn {x, _}, a -> a + (x - mx) * (x - mx) end)
    sxy = Enum.reduce(pts, 0.0, fn {x, y}, a -> a + (x - mx) * (y - my) end)
    b = sxy / sxx
    {my - b * mx, b}
  end

  @doc "The value at pixel `p` of a fitted axis."
  def value(%{scale: :linear, a: a, b: b}, p), do: a + b * p
  def value(%{scale: :log, a: a, b: b}, p), do: :math.pow(10, a + b * p)

  def value(%{scale: :categorical, ticks: ts}, p) do
    t = Enum.min_by(ts, &abs(&1.at - p), fn -> nil end)
    t && t.text
  end

  # ---------------------------------------------------------------- series --

  @grey {0.5773502691896258, 0.5773502691896258, 0.5773502691896258}

  # Series by colour. A mark's edge is its colour blended with the white
  # paper, p = c + t·(W − c), so W − p = (1 − t)·(W − c): the *direction*
  # of W − p is the colour's, whatever the blend. Pixels are grouped by
  # that direction (grey and black, on the diagonal, are frames, grids and
  # text), so a one-pixel antialiased line whose every pixel is a blend is
  # still one series, and its pixels weigh by how far from white they are.
  defp series(im, ax, xs, ys) do
    {l, t, r, b} = ax.box
    {l, t, r, b} = {l + 1, t + 1, r - 1, b - 1}

    colored =
      for y <- t..b//1, x <- l..r//1, {pr, pg, pb} = px(im, x, y), v = {255 - pr, 255 - pg, 255 - pb}, n = norm(v), n >= 30,
          d = scale3(v, 1 / n), dot(d, @grey) < 0.985, do: {x, y, d, n, {pr, pg, pb}}

    seeds = seeds(colored)

    colored
    |> Enum.group_by(fn {_, _, d, _, _} ->
      case seeds |> Enum.with_index() |> Enum.map(fn {s, i} -> {dot(s, d), i} end) |> Enum.max_by(&elem(&1, 0), fn -> nil end) do
        {c, i} when c >= 0.985 -> i
        _ -> nil
      end
    end)
    |> Map.delete(nil)
    |> Enum.sort()
    |> Enum.map(fn {_, pxs} ->
      # the series' colour: its least blended pixels
      core = pxs |> Enum.sort_by(fn {_, _, _, n, _} -> -n end) |> Enum.take(max(1, div(length(pxs), 10))) |> Enum.map(&elem(&1, 4)) |> mean_color()
      one_series(core, Enum.map(pxs, fn {x, y, _, n, _} -> {x, y, n} end), ax, xs, ys)
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp norm({a, b, c}), do: :math.sqrt(a * a + b * b + c * c)
  defp scale3({a, b, c}, k), do: {a * k, b * k, c * k}
  defp dot({a, b, c}, {d, e, f}), do: a * d + b * e + c * f

  defp seeds(colored) do
    total = length(colored)

    colored
    |> Enum.group_by(fn {_, _, {a, b, c}, _, _} -> {round(a * 40), round(b * 40), round(c * 40)} end, &elem(&1, 2))
    |> Enum.map(fn {_, ds} -> {length(ds), ds} end)
    |> Enum.filter(fn {n, _} -> n >= max(10, 0.01 * total) end)
    |> Enum.sort_by(fn {n, _} -> -n end)
    |> Enum.reduce([], fn {_, ds}, acc ->
      d = ds |> Enum.reduce({0.0, 0.0, 0.0}, fn {a, b, c}, {x, y, z} -> {x + a, y + b, z + c} end) |> then(&scale3(&1, 1 / norm(&1)))
      if Enum.any?(acc, &(dot(&1, d) >= 0.993)), do: acc, else: acc ++ [d]
    end)
  end

  defp mean_color(ps) do
    n = length(ps)
    {r, g, b} = Enum.reduce(ps, {0, 0, 0}, fn {r, g, b}, {a, c, d} -> {a + r, c + g, d + b} end)
    {round(r / n), round(g / n), round(b / n)}
  end


  defp hex({r, g, b}), do: "#" <> Base.encode16(<<r, g, b>>, case: :lower)

  defp one_series(color, weighted, ax, xs, ys) do
    {l, t, r, b} = ax.box
    pixels = Enum.map(weighted, fn {x, y, _} -> {x, y} end)
    set = pixels |> MapSet.new() |> close_gaps()
    comps = blobs(set)
    big = Enum.filter(comps, &(&1.area >= 20))
    side = min(r - l, b - t)

    bars = Enum.filter(big, fn c -> {x0, y0, x1, y1} = c.box; c.area >= 0.85 * (x1 - x0 + 1) * (y1 - y0 + 1) and x1 - x0 >= 3 end)
    bar_area = bars |> Enum.map(& &1.area) |> Enum.sum()
    bottoms = Enum.map(bars, fn %{box: {_, _, _, y1}} -> y1 end)

    # markers: compact blobs (several overlapping markers make one larger
    # compact blob, split below); a line is long and thin, or sparse in its box
    small =
      Enum.filter(comps, fn %{box: {x0, y0, x1, y1}, area: a} ->
        {bw, bh} = {x1 - x0 + 1, y1 - y0 + 1}
        bw >= 3 and bh >= 3 and max(bw, bh) <= 0.15 * side and max(bw, bh) <= 2.5 * min(bw, bh) and a >= 0.4 * bw * bh
      end)

    small_area = small |> Enum.map(& &1.area) |> Enum.sum()

    cond do
      length(pixels) < 15 ->
        nil

      bars != [] and bar_area >= 0.8 * length(pixels) and Enum.max(bottoms) - Enum.min(bottoms) <= 3 ->
        items =
          bars
          |> Enum.sort_by(fn %{box: {x0, _, _, _}} -> x0 end)
          |> Enum.map(fn %{box: {x0, y0, x1, _}} ->
            cx = (x0 + x1) / 2
            %{x: if(xs.scale == :categorical, do: cat_index(xs, cx), else: value(xs, cx)), label: if(xs.scale == :categorical, do: value(xs, cx)), value: value(ys, y0 - 0.5)}
          end)

        %{color: hex(color), kind: :bar, bars: items, points: Enum.map(items, &{&1.x, &1.value})}

      xs.scale == :categorical ->
        nil

      length(small) >= 3 and small_area >= 0.75 * length(pixels) ->
        # one marker's area: the lower quartile of the blobs (most blobs are single markers)
        areas = small |> Enum.map(& &1.area) |> Enum.sort()
        unit = Enum.at(areas, div(length(areas), 4))

        pts =
          small
          |> Enum.flat_map(fn %{pixels: ps, area: a} -> kmeans(ps, max(1, round(a / unit))) end)
          |> Enum.map(fn {x, y} -> {value(xs, x), value(ys, y)} end)
          |> Enum.sort()

        %{color: hex(color), kind: :scatter, points: pts}

      true ->
        # per column, the darkness-weighted centre of the line: sub-pixel
        pts =
          weighted
          |> Enum.group_by(&elem(&1, 0), fn {_, y, n} -> {y, n} end)
          |> Enum.sort()
          |> Enum.map(fn {x, yn} -> {value(xs, x * 1.0), value(ys, Enum.sum(Enum.map(yn, fn {y, n} -> y * n end)) / Enum.sum(Enum.map(yn, &elem(&1, 1))))} end)

        %{color: hex(color), kind: :line, points: pts}
    end
  end

  # Grid lines drawn over a bar cut it into pieces: gaps of up to two
  # pixels between marks of the same colour, along a row or a column, are
  # filled (a closing). A line or a marker is unchanged by it.
  defp close_gaps(set) do
    fill =
      for {x, y} <- set, {dx, dy} <- [{0, 2}, {0, 3}, {2, 0}, {3, 0}], MapSet.member?(set, {x + dx, y + dy}),
          k <- 1..(max(dx, dy) - 1), q = {x + div(dx * k, max(dx, dy)), y + div(dy * k, max(dx, dy))}, not MapSet.member?(set, q), do: q

    Enum.into(fill, set)
  end

  # k centres of a blob's pixels (k overlapping markers): seeded along the
  # blob's longer side, ten Lloyd iterations
  defp kmeans(ps, 1), do: [centroid(ps)]

  defp kmeans(ps, k) do
    {xs, ys} = Enum.unzip(ps)
    {x0, x1, y0, y1} = {Enum.min(xs), Enum.max(xs), Enum.min(ys), Enum.max(ys)}
    init = for i <- 0..(k - 1), do: (f = (i + 0.5) / k; if(x1 - x0 >= y1 - y0, do: {x0 + f * (x1 - x0), (y0 + y1) / 2}, else: {(x0 + x1) / 2, y0 + f * (y1 - y0)}))

    Enum.reduce(1..10, init, fn _, cs ->
      groups = Enum.group_by(ps, fn {x, y} -> cs |> Enum.with_index() |> Enum.min_by(fn {{cx, cy}, _} -> (x - cx) ** 2 + (y - cy) ** 2 end) |> elem(1) end)
      for {c, i} <- Enum.with_index(cs), do: (case groups[i] do nil -> c; g -> centroid(g) end)
    end)
  end

  defp centroid(ps), do: {Enum.sum(Enum.map(ps, &elem(&1, 0))) / length(ps), Enum.sum(Enum.map(ps, &elem(&1, 1))) / length(ps)}

  defp cat_index(%{ticks: ts}, p), do: ts |> Enum.with_index() |> Enum.min_by(fn {t, _} -> abs(t.at - p) end) |> elem(1)

  # connected components of a pixel set (8-neighbourhood)
  defp blobs(set) do
    {comps, _} =
      Enum.reduce(set, {[], MapSet.new()}, fn p, {acc, seen} ->
        if MapSet.member?(seen, p) do
          {acc, seen}
        else
          {pxs, seen} = flood([p], set, MapSet.put(seen, p), [])
          {xs, ys} = Enum.unzip(pxs)
          {[%{box: {Enum.min(xs), Enum.min(ys), Enum.max(xs), Enum.max(ys)}, area: length(pxs), pixels: pxs} | acc], seen}
        end
      end)

    comps
  end

  defp flood([], _set, seen, out), do: {out, seen}

  defp flood([{x, y} = p | rest], set, seen, out) do
    {rest, seen} =
      for dx <- -1..1, dy <- -1..1, q = {x + dx, y + dy}, MapSet.member?(set, q), not MapSet.member?(seen, q), reduce: {rest, seen} do
        {st, sn} -> {[q | st], MapSet.put(sn, q)}
      end

    flood(rest, set, seen, [p | out])
  end

  # ------------------------------------------------------------- detection --

  # numbered (arabic, roman, Arabic-Indic digits) or lettered ("Figure A:", in an appendix)
  @caption ~r/^\s*(?i:fig(ure|ura)?\.?|figs?\.|gr[aá]fico|chart|plate|imagem|image|图|圖|図|그림|شكل)\s*([0-9IVX٠-٩]+|[A-Z](?=[.:\s—–-]))/u

  @doc """
  Figures of a page: `[%{box, kind: :chart | :picture, caption: %{box,
  text} | nil, comps}]` and the components left for the text
  (`{figures, rest}`). Options: `ocr` (a model for reading the captions;
  default `OCR.default/0`), `worker`, `digitize: true` (each chart's data
  under `data`, its refusal under `refused`), `picture` (the page image,
  needed for `kind` and `digitize`).
  """
  def detect(comps, opts \\ []) do
    hs = comps |> Enum.filter(&(&1.area >= 4)) |> Enum.map(&height/1) |> Enum.sort()

    if hs == [] do
      {[], comps}
    else
      h0 = Enum.at(hs, div(length(hs), 2))
      textlike = Enum.filter(comps, fn c -> height(c) >= 0.6 * h0 and height(c) <= 1.6 * h0 end)

      # a seed is a large mark, large both ways (a run of dashes in a line of
      # text is long but one text height tall) — but not a frame drawn around text (a box
      # holding lines of text is a framed paragraph, not a figure; the
      # markers of a scatter plot are text-sized too, but not in lines)
      seeds =
        Enum.filter(comps, fn c ->
          max(height(c), width(c)) >= 6 * h0 and min(height(c), width(c)) >= 1.5 * h0 and
            not framed_text?(Enum.filter(textlike, &inside?(&1.box, c.box)), h0)
        end)

      regions = seeds |> Enum.map(& &1.box) |> merge_boxes(round(h0))
      {regions, rest} = grow(regions, comps, h0)

      figures =
        regions
        |> Enum.map(fn {box, cs} -> %{box: refine(box, opts[:picture]), comps: cs} end)
        |> Enum.map(&caption(&1, rest, h0, opts))
        |> Enum.map(&kind(&1, opts))

      {figures, rest}
    end
  end

  # A photograph's binarised ink is scattered blobs, so its marks cover
  # only part of it; on the page itself the picture is a rectangle of
  # non-paper pixels. Each side moves out while the row (column) just
  # beyond it is mostly not paper. A chart, ink on white, does not move.
  defp refine(box, %Image{} = img) do
    g = Segment.gray(img)
    paper? = fn x, y -> :binary.at(g.gray, y * g.w + x) >= 235 end

    busy = fn
      {:row, y, x0, x1} -> y >= 0 and y < g.h and Enum.count(x0..x1, &(not paper?.(&1, y))) >= 0.6 * (x1 - x0 + 1)
      {:col, x, y0, y1} -> x >= 0 and x < g.w and Enum.count(y0..y1, &(not paper?.(x, &1))) >= 0.6 * (y1 - y0 + 1)
    end

    Stream.iterate({box, true}, fn {{x0, y0, x1, y1}, _} ->
      nb = {if(busy.({:col, x0 - 1, y0, y1}), do: x0 - 1, else: x0), if(busy.({:row, y0 - 1, x0, x1}), do: y0 - 1, else: y0),
            if(busy.({:col, x1 + 1, y0, y1}), do: x1 + 1, else: x1), if(busy.({:row, y1 + 1, x0, x1}), do: y1 + 1, else: y1)}
      {nb, nb != {x0, y0, x1, y1}}
    end)
    |> Enum.find(fn {_, moved} -> not moved end)
    |> elem(0)
  end

  defp refine(box, _), do: box

  # three or more lines of eight marks set close together, as letters are
  defp framed_text?(marks, h0) when length(marks) < 24, do: (_ = h0; false)

  defp framed_text?(marks, h0) do
    marks
    |> Segment.lines()
    |> Enum.count(fn %{glyphs: gs} ->
      gaps = gs |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [%{box: {_, _, a1, _}}, %{box: {b0, _, _, _}}] -> b0 - a1 - 1 end) |> Enum.sort()
      length(gs) >= 8 and Enum.at(gaps, div(length(gaps), 2)) <= 0.6 * h0
    end)
    |> Kernel.>=(3)
  end

  defp height(%{box: {_, y0, _, y1}}), do: y1 - y0 + 1
  defp width(%{box: {x0, _, x1, _}}), do: x1 - x0 + 1
  defp inside?({a0, b0, a1, b1}, {c0, d0, c1, d1}), do: a0 > c0 and b0 > d0 and a1 < c1 and b1 < d1

  defp merge_boxes(boxes, gap) do
    merged =
      Enum.reduce(boxes, [], fn b, acc ->
        {near, far} = Enum.split_with(acc, &close?(&1, b, gap))
        [Enum.reduce(near, b, &union/2) | far]
      end)

    if length(merged) == length(boxes), do: merged, else: merge_boxes(merged, gap)
  end

  defp close?({a0, b0, a1, b1}, {c0, d0, c1, d1}, g), do: c0 <= a1 + g and a0 <= c1 + g and d0 <= b1 + g and b0 <= d1 + g
  defp union({a0, b0, a1, b1}, {c0, d0, c1, d1}), do: {min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)}

  # every mark inside a figure's box, or within 1.5 text heights of it, joins
  # it (tick labels, axis titles, legends) — repeatedly, so a label beside
  # a label joins too, but never across a paragraph's gap
  defp grow(regions, comps, h0) do
    g = round(1.5 * h0)

    # every mark of a text line of eight marks or more → that line's horizontal extent
    lines =
      for %{glyphs: gs, box: {lx0, _, lx1, _}} <- Segment.lines(comps), length(gs) >= 8, %{pixels: ps} <- gs, p <- ps, into: %{}, do: {p, {lx0, lx1}}

    {final, rest} =
      Enum.reduce(1..6, {Enum.map(regions, &{&1, []}), comps}, fn _, {rs, left} ->
        Enum.reduce(rs, {[], left}, fn {box, cs}, {acc, left} ->
          {take, keep} = Enum.split_with(left, &close?(box, &1.box, g))
          # a line of text running well past the figure's sides is the
          # page's (a caption set close above it, a paragraph), not the figure's
          {bx0, _, bx1, _} = box
          {page, take} = Enum.split_with(take, fn c -> case lines[hd(c.pixels)] do {lx0, lx1} -> lx0 < bx0 - 3 * h0 or lx1 > bx1 + 3 * h0; nil -> false end end)
          keep = page ++ keep
          box = Enum.reduce(take, box, &union(&1.box, &2))
          {acc ++ [{box, cs ++ take}], keep}
        end)
      end)

    {final |> Enum.map(fn {b, cs} -> {b, cs} end) |> merge_regions(g), rest}
  end

  defp merge_regions(rs, g) do
    Enum.reduce(rs, [], fn {b, cs}, acc ->
      case Enum.split_with(acc, fn {b2, _} -> close?(b2, b, g) end) do
        {[], _} -> acc ++ [{b, cs}]
        {near, far} -> far ++ [Enum.reduce(near, {b, cs}, fn {b2, c2}, {bb, cc} -> {union(b2, bb), c2 ++ cc} end)]
      end
    end)
    |> Enum.sort_by(fn {{_, y0, _, _}, _} -> y0 end)
  end

  # the caption: the nearest text line below (else above) within three
  # text heights that reads as one
  defp caption(%{box: {x0, y0, x1, y1}} = f, rest, h0, opts) do
    # (a caption may start at the margin, left of a narrow figure: the band is
    # the page's width; 8 text heights deep, past an axis title the figure
    # did not take in)
    near = Enum.filter(rest, fn %{box: {_, b0, _, b1}} -> b0 > y1 and b0 - y1 <= 8 * h0 or b1 < y0 and y0 - b1 <= 8 * h0 end)
    lines = near |> Segment.lines() |> Enum.filter(fn %{box: {a0, _, a1, _}} -> a1 >= x0 and a0 <= x1 end)

    # the nearest lines on either side, nearest first: a caption above a
    # figure with body text right below it (or an axis title left outside
    # the box) is still among them
    ordered = Enum.sort_by(lines, fn %{box: {_, ly0, _, ly1}} -> if ly0 > y1, do: ly0 - y1, else: y0 - ly1 end)

    model = case opts[:ocr] do nil -> elem(OCR.default(), 1); m -> m end
    w = Keyword.get_lazy(opts, :worker, &OCR.worker/0)

    cap =
      Enum.find_value(Enum.take(ordered, 4), fn line ->
        r = read_text_line(line, model, w, opts[:picture])
        if Regex.match?(@caption, r.text), do: %{box: line.box, text: r.text, position: if(elem(line.box, 1) > y1, do: :below, else: :above)}
      end)

    Map.put(f, :caption, cap)
  end

  # a text line, read from its own grey patch enlarged when the type is
  # small (a caption is often set two points smaller than the body)
  defp read_text_line(%{box: {x0, y0, x1, y1}} = line, model, w, picture) do
    h = y1 - y0 + 1

    case picture do
      %Image{} = img when h < 22 ->
        g = Segment.gray(img)
        k = min(3, max(2, round(26 / h)))
        patch = g |> gray_crop({max(x0 - 3, 0), max(y0 - 3, 0), min(x1 + 3, g.w - 1), min(y1 + 3, g.h - 1)}) |> enlarge(k)
        OCR.read_components(patch |> Segment.ink() |> Segment.components(), model, w, nil, marks: true)

      _ ->
        OCR.read_line(line, model, w, nil)
    end
  end

  defp kind(%{box: box} = f, opts) do
    case opts[:picture] do
      %Image{} = img ->
        sub = crop(img, box)

        case axes(sub) do
          {:ok, _} ->
            f = Map.put(f, :kind, :chart)

            if opts[:digitize] do
              case digitize(sub, Keyword.take(opts, [:worker]) ++ [model: opts[:ocr]] |> Enum.reject(fn {_, v} -> is_nil(v) end)) do
                {:ok, d} -> Map.put(f, :data, d)
                {:error, why} -> Map.put(f, :refused, why)
              end
            else
              f
            end

          _ ->
            Map.put(f, :kind, :picture)
        end

      _ ->
        Map.put(f, :kind, :figure)
    end
  end
end
