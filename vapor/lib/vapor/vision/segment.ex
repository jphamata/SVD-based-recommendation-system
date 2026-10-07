defmodule Vapor.Vision.Segment do
  @moduledoc """
  From a picture to glyphs, by first principles and no model: the half of
  OCR that is geometry. `Vapor.Vision.OCR` gives the glyphs to a classifier
  admitted by the model airlock; this module only finds them.

    1. **Ink** — Sauvola's adaptive threshold over integral images
       (`T = m·(1 + k·(s/R − 1))` in a window around each pixel), so an
       unevenly lit photograph of a page binarises as well as a clean scan.
    2. **Components** — 8-connected regions of ink (labels in `:atomics`).
    3. **Blocks** — the page cut into its reading order by a recursive
       XY-cut over the components (`blocks/2`): a column gutter (an empty
       vertical strip across the whole region) splits before a horizontal
       gap does, so a title or a footer that spans the columns is cut off
       first and two columns are read one after the other — not line by
       line across both.
    4. **Lines** (within each block) — the vertical *cores* of the components (their middle
       half) projected on the rows: ascenders and descenders of neighbouring
       lines no longer bridge them. Every component joins the line its
       extent overlaps most.
    5. **Glyphs** — components of a line that share their columns (the dot
       of an *i*, an accent over *é*, the halves of *:* or *=*) merge.
    6. **Spaces** — a gap wider than a fraction of the line's median glyph
       height.
    7. **Features** — each glyph as a 20×20 coverage map of its own ink
       (aspect kept, centred) plus 16 numbers of geometry relative to its
       line (size, position against the baseline, aspect, density, parts):
       416 values, the classifier's input row.

  Pictures are `Vapor.Modal.Image` or `%{w, h, gray}` (a binary of 8-bit
  luminance). Everything here is integer or binary64 arithmetic on the
  BEAM: deterministic.
  """
  @grid 20
  @feat 16
  @doc "Width of a glyph's feature row (20×20 coverage + 16 geometry values)."
  def width, do: @grid * @grid + @feat

  # ---------------------------------------------------------------- gray --

  @doc "8-bit luminance `%{w, h, gray}` of a `Vapor.Modal.Image` (values in [0, 1]) or of an existing gray map."
  def gray(%{w: _, h: _, gray: _} = g), do: g

  def gray(%Vapor.Modal.Image{w: w, h: h, c: c, px: px}) do
    vals = Tuple.to_list(px)

    bytes =
      case c do
        1 -> for v <- vals, into: <<>>, do: <<clamp8(v * 255)>>
        3 -> for [r, g, b] <- Enum.chunk_every(vals, 3), into: <<>>, do: <<clamp8((0.299 * r + 0.587 * g + 0.114 * b) * 255)>>
        _ -> for vs <- Enum.chunk_every(vals, c), into: <<>>, do: <<clamp8(Enum.sum(Enum.take(vs, 3)) / 3 * 255)>>
      end

    %{w: w, h: h, gray: bytes}
  end

  defp clamp8(v), do: v |> round() |> max(0) |> min(255)

  # ---------------------------------------------------------------- ink --

  @doc """
  Sauvola binarisation: a tuple of row tuples of 0/1 (1 = ink). Options:
  `radius` (window half-size; default `max(7, min(w, h) ÷ 40)`), `k` (0.2),
  `r` (128).
  """
  def ink(%{w: w, h: h, gray: g}, opts \\ []) do
    rad = Keyword.get(opts, :radius, max(7, div(min(w, h), 40)))
    k = Keyword.get(opts, :k, 0.2)
    rr = Keyword.get(opts, :r, 128.0)
    {s1, s2} = integral(g, w, h)

    for y <- 0..(h - 1) do
      {ya, yb} = {max(y - rad, 0), min(y + rad, h - 1)}
      {a1, b1, a2, b2} = {elem(s1, ya), elem(s1, yb + 1), elem(s2, ya), elem(s2, yb + 1)}
      row = binary_part(g, y * w, w)

      for x <- 0..(w - 1) do
        {xa, xb} = {max(x - rad, 0), min(x + rad, w - 1)}
        n = (yb - ya + 1) * (xb - xa + 1)
        sum = elem(b1, xb + 1) - elem(b1, xa) - elem(a1, xb + 1) + elem(a1, xa)
        sq = elem(b2, xb + 1) - elem(b2, xa) - elem(a2, xb + 1) + elem(a2, xa)
        m = sum / n
        s = :math.sqrt(max(sq / n - m * m, 0.0))
        if :binary.at(row, x) < m * (1 + k * (s / rr - 1)), do: 1, else: 0
      end
      |> List.to_tuple()
    end
    |> List.to_tuple()
  end

  # integral images of g and g², (h + 1) × (w + 1)
  defp integral(g, w, h) do
    zero = Tuple.duplicate(0, w + 1)

    {r1, r2} =
      Enum.reduce(0..(h - 1), {[zero], [zero]}, fn y, {[p1 | _] = acc1, [p2 | _] = acc2} ->
        row = binary_part(g, y * w, w)

        {c1, c2, _, _} =
          for(<<v <- row>>, reduce: {[0], [0], 0, 0}, do: ({l1, l2, s1, s2} -> {[s1 + v | l1], [s2 + v * v | l2], s1 + v, s2 + v * v}))

        n1 = c1 |> Enum.reverse() |> List.to_tuple()
        n2 = c2 |> Enum.reverse() |> List.to_tuple()
        {[add_rows(n1, p1, w) | acc1], [add_rows(n2, p2, w) | acc2]}
      end)

    {r1 |> Enum.reverse() |> List.to_tuple(), r2 |> Enum.reverse() |> List.to_tuple()}
  end

  defp add_rows(a, b, w), do: for(i <- 0..w, do: elem(a, i) + elem(b, i)) |> List.to_tuple()

  # --------------------------------------------------------- components --

  @doc """
  8-connected components of an ink map: `[%{box: {x0, y0, x1, y1}, area,
  pixels}]` (components of fewer than `min_area` pixels are dropped).
  """
  def components(mask, opts \\ []) do
    h = tuple_size(mask)
    w = if h > 0, do: tuple_size(elem(mask, 0)), else: 0
    min_area = Keyword.get(opts, :min_area, 2)
    seen = :atomics.new(max(w * h, 1), [])
    at = fn x, y -> elem(elem(mask, y), x) end

    for y <- 0..(h - 1)//1, x <- 0..(w - 1)//1, at.(x, y) == 1, :atomics.get(seen, y * w + x + 1) == 0, reduce: [] do
      acc ->
        :atomics.put(seen, y * w + x + 1, 1)
        px = fill([{x, y}], [], mask, seen, w, h)
        if length(px) >= min_area, do: [component(px) | acc], else: acc
    end
    |> Enum.reverse()
  end

  defp fill([], out, _mask, _seen, _w, _h), do: out

  defp fill([{x, y} = p | stack], out, mask, seen, w, h) do
    stack =
      for dy <- -1..1, dx <- -1..1, {nx, ny} = {x + dx, y + dy}, nx >= 0 and ny >= 0 and nx < w and ny < h,
          elem(elem(mask, ny), nx) == 1, :atomics.get(seen, ny * w + nx + 1) == 0, reduce: stack do
        st ->
          :atomics.put(seen, ny * w + nx + 1, 1)
          [{nx, ny} | st]
      end

    fill(stack, [p | out], mask, seen, w, h)
  end

  defp component(px) do
    {xs, ys} = Enum.unzip(px)
    %{box: {Enum.min(xs), Enum.min(ys), Enum.max(xs), Enum.max(ys)}, area: length(px), pixels: px}
  end

  # --------------------------------------------------------------- blocks --

  @doc """
  The reading order of a page: its components cut into blocks by a
  recursive XY-cut, `[%{box, comps}]` in the order they are read.

  At every region, with `h` the median height of its text-sized components
  (never less than the page's: a title set larger has wider word spaces):

    * a **column gutter** — an empty vertical strip across the whole region,
      at least `gutter` × `h` wide (default 1.5; a word space is ≈ 0.5 `h`) —
      splits it into columns, read left to right;
    * otherwise a **horizontal gap** wider than both `gap` × `h` (default
      1.5) and 1.5 × the region's median gap between rows of text (so a
      double-spaced body is not cut line by line) splits it into bands, read
      top to bottom;
    * otherwise it is a block.

  Gutters are tried first, so a title or a footer spanning the columns is
  cut away (it blocks every gutter), and then each band's columns are read
  one after the other. Specks of dust, rules and frames (as in `lines/2`)
  do not take part in the cuts; a speck inside a gutter is dropped, any
  other small component joins the block that contains its centre.
  `columns: false` returns the whole page as one block.
  """
  def blocks(comps, opts \\ []) do
    big = Enum.filter(comps, &(&1.area >= 4))
    hs = big |> Enum.map(&height/1) |> Enum.sort()

    cond do
      hs == [] -> []
      Keyword.get(opts, :columns, true) == false -> [%{box: hull(comps), comps: comps}]
      true ->
        h0 = median(hs)
        solid = Enum.filter(comps, fn c -> layout?(c, h0) end)
        leaves = cut(solid, h0, opts, 0)
        assign(comps, leaves, h0)
    end
  end

  # text-sized: not a speck, not a rule, not a frame
  defp layout?(c, h0) do
    (height(c) >= 0.25 * h0 or width(c) >= 0.25 * h0) and c.area >= 4 and
      not (width(c) > 12 * max(height(c), 1) and height(c) <= 0.35 * h0) and height(c) <= 10 * h0
  end

  defp cut([], _h0, _opts, _depth), do: []
  defp cut([_] = cs, _h0, _opts, _depth), do: [cs]
  defp cut(cs, _h0, _opts, depth) when depth > 24, do: [cs]

  defp cut(cs, h0, opts, depth) do
    # thresholds in the region's own text height: a title set larger than
    # the body has wider word spaces, which must not read as a gutter
    h0 = max(h0, cs |> Enum.map(&height/1) |> Enum.sort() |> median())
    gutter = Keyword.get(opts, :gutter, 1.5) * h0
    xs = gaps(cs, fn %{box: {x0, _, x1, _}} -> {x0, x1} end)

    case Enum.filter(xs, fn {a, b} -> b - a - 1 >= gutter end) do
      [_ | _] = gs ->
        # columns in reading order: right to left for a right-to-left script
        split(cs, gs, fn %{box: {x0, _, x1, _}} -> (x0 + x1) / 2 end)
        |> then(fn cols -> if opts[:direction] == :rtl, do: Enum.reverse(cols), else: cols end)
        |> Enum.flat_map(&cut(&1, h0, opts, depth + 1))

      [] ->
        ys = gaps(cs, fn %{box: {_, y0, _, y1}} -> {y0, y1} end)
        widths = ys |> Enum.map(fn {a, b} -> b - a - 1 end) |> Enum.sort()
        typical = if widths == [], do: 0, else: median(widths)
        th = max(Keyword.get(opts, :gap, 1.5) * h0, 1.5 * typical)

        case Enum.filter(ys, fn {a, b} -> b - a - 1 > th end) do
          [] -> [cs]
          gs -> split(cs, gs, fn %{box: {_, y0, _, y1}} -> (y0 + y1) / 2 end) |> Enum.flat_map(&cut(&1, h0, opts, depth + 1))
        end
    end
  end

  # empty intervals {last covered, next covered} of the projection of the boxes
  defp gaps(cs, span) do
    cs
    |> Enum.map(span)
    |> Enum.sort()
    |> Enum.reduce({[], nil}, fn
      {_a, b}, {acc, nil} -> {acc, b}
      {a, b}, {acc, e} when a > e + 1 -> {[{e, a} | acc], b}
      {_a, b}, {acc, e} -> {acc, max(b, e)}
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  defp split(cs, gs, centre) do
    cuts = Enum.map(gs, fn {a, b} -> (a + b) / 2 end)

    cs
    |> Enum.group_by(fn c -> Enum.count(cuts, &(centre.(c) > &1)) end)
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map(&elem(&1, 1))
  end

  # every component to the block whose box (grown by half a text height) holds its centre
  defp assign(comps, leaves, h0) do
    m = round(h0 / 2)
    boxes = leaves |> Enum.map(&hull/1) |> Enum.with_index()

    groups =
      Enum.group_by(comps, fn %{box: {x0, y0, x1, y1}} ->
        {cx, cy} = {(x0 + x1) / 2, (y0 + y1) / 2}

        Enum.find_value(boxes, fn {{a0, b0, a1, b1}, i} ->
          if cx >= a0 - m and cx <= a1 + m and cy >= b0 - m and cy <= b1 + m, do: i
        end)
      end)

    for {box, i} <- boxes, cs = Map.get(groups, i, []), cs != [], do: %{box: box, comps: cs}
  end

  defp hull(cs) do
    cs |> Enum.map(& &1.box) |> Enum.reduce(fn {a0, b0, a1, b1}, {c0, d0, c1, d1} -> {min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)} end)
  end

  # ---------------------------------------------------------------- lines --

  @doc """
  Group components into lines (top to bottom) of glyphs (left to right):
  `[%{box, glyphs: [%{box, pixels, parts}], spaces: [glyph index after
  which a space falls], ref, baseline}]`.
  """
  def lines(comps, opts \\ []) do
    big = Enum.filter(comps, &(&1.area >= 4))
    hs = big |> Enum.map(&height/1) |> Enum.sort()

    if hs == [] do
      []
    else
      h0 = median(hs)
      # rules and frames: very long and thin, or taller than ten text lines
      comps = Enum.reject(comps, fn c -> width(c) > 12 * max(height(c), 1) and height(c) <= 0.35 * h0 or height(c) > 10 * h0 end)
      body = Enum.filter(comps, &(height(&1) >= 0.5 * h0))
      bands = bands(body)

      # an accent or a cedilla off a line with no ascender or descender lies
      # outside its line's core band by a fraction of the text height: it
      # still belongs to it (a fixed 3 px dropped the tilde of every "não"
      # on such lines — found in 0.7, docs/OCR.md §6)
      reach = max(3, round(0.6 * h0))

      comps
      |> Enum.group_by(fn c -> best_band(c, bands, reach) end)
      |> Enum.reject(fn {b, _} -> b == nil end)
      |> Enum.sort_by(fn {b, _} -> b end)
      |> Enum.map(&elem(&1, 1))
      |> satellites()
      |> Enum.map(&line(&1, opts))
      |> Enum.reject(&(&1.glyphs == []))
    end
  end

  # A group of marks too short to be a line of its own, lying over the
  # columns of a taller neighbour and touching or nearly touching it, is
  # part of that neighbour: the dots and strokes of Arabic (ب ت ث ن ي ة,
  # the bar of ك), whose median height is that of a dot, form bands of
  # their own above and below the letter bodies. Without this, 1 Arabic
  # line in 4 came out as two or three (found in 0.10, docs/OCR.md §3g).
  # A real line of text is never under half its neighbour's height, so
  # Latin pages are untouched (the 0.7 quality suite still holds).
  defp satellites(groups) do
    boxes = Enum.map(groups, &group_box/1)
    indexed = Enum.zip(groups, boxes) |> Enum.with_index()

    host = fn {{_, {x0, y0, x1, y1}}, i} ->
      h = y1 - y0 + 1

      indexed
      |> Enum.filter(fn {{_, {a0, b0, a1, b1}}, j} ->
        hh = b1 - b0 + 1
        gap = max(b0 - y1, y0 - b1)
        j != i and h < 0.5 * hh and x0 >= a0 - 2 and x1 <= a1 + 2 and gap <= max(3, round(0.35 * hh))
      end)
      |> Enum.max_by(fn {{_, {_, b0, _, b1}}, _} -> b1 - b0 end, fn -> nil end)
      |> case do
        nil -> nil
        {_, j} -> j
      end
    end

    hosts = Map.new(indexed, fn {_, i} = e -> {i, host.(e)} end)
    # follow a satellite to its final host (a host is never itself a satellite of a smaller group)
    root = fn root, i -> case hosts[i] do nil -> i; j -> root.(root, j) end end

    indexed
    |> Enum.group_by(fn {_, i} -> root.(root, i) end, fn {{cs, _}, _} -> cs end)
    |> Enum.sort_by(fn {i, _} -> i end)
    |> Enum.map(fn {_, css} -> Enum.concat(css) end)
  end

  defp group_box(cs) do
    cs
    |> Enum.map(& &1.box)
    |> Enum.reduce(fn {a0, b0, a1, b1}, {c0, d0, c1, d1} -> {min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)} end)
  end

  defp height(%{box: {_, y0, _, y1}}), do: y1 - y0 + 1
  defp width(%{box: {x0, _, x1, _}}), do: x1 - x0 + 1
  defp median(sorted), do: Enum.at(sorted, div(length(sorted), 2))

  # row bands where the middle halves of body components lie
  defp bands(body) do
    prof =
      Enum.reduce(body, %{}, fn %{box: {_, y0, _, y1}}, acc ->
        q = div(y1 - y0 + 1, 4)
        Enum.reduce((y0 + q)..(y1 - q)//1, acc, fn y, a -> Map.update(a, y, 1, &(&1 + 1)) end)
      end)

    rows = prof |> Map.keys() |> Enum.sort()

    rows
    |> Enum.chunk_while([], fn y, acc ->
      case acc do
        [] -> {:cont, [y]}
        [prev | _] when y == prev + 1 -> {:cont, [y | acc]}
        _ -> {:cont, Enum.reverse(acc), [y]}
      end
    end, fn acc -> {:cont, Enum.reverse(acc), []} end)
    |> Enum.reject(&(&1 == []))
    |> Enum.map(fn ys -> {hd(ys), List.last(ys)} end)
  end

  defp best_band(%{box: {_, y0, _, y1}}, bands, reach) do
    bands
    |> Enum.with_index()
    |> Enum.map(fn {{a, b}, i} ->
      ov = min(y1, b) - max(y0, a) + 1
      dist = if ov > 0, do: 0, else: min(abs(y0 - b), abs(y1 - a))
      {i, ov, dist}
    end)
    |> Enum.sort_by(fn {_, ov, dist} -> {-ov, dist} end)
    |> case do
      [{i, ov, dist} | _] when ov > 0 or dist <= reach -> i
      _ -> nil
    end
  end

  defp line(cs, opts) do
    glyphs = cs |> Enum.sort_by(fn %{box: {x0, _, _, _}} -> x0 end) |> merge_columns([])
    hs = glyphs |> Enum.map(&height/1) |> Enum.sort()
    ref = max(median(hs), 1)
    baseline = glyphs |> Enum.map(fn %{box: {_, _, _, y1}} -> y1 end) |> Enum.sort() |> median()
    space = Keyword.get(opts, :space, 0.26) * ref

    spaces =
      glyphs
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.filter(fn {[%{box: {_, _, x1, _}}, %{box: {nx0, _, _, _}}], _} -> nx0 - x1 - 1 > space end)
      |> Enum.map(&elem(&1, 1))

    box = Enum.reduce(glyphs, fn %{box: {a0, b0, a1, b1}}, %{box: {c0, d0, c1, d1}} = g -> %{g | box: {min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)}} end).box
    %{box: box, glyphs: glyphs, spaces: spaces, ref: ref, baseline: baseline}
  end

  # components left to right; one sharing at least half of the narrower's
  # columns with the glyph being built joins it
  defp merge_columns([], acc), do: Enum.reverse(acc)

  defp merge_columns([c | rest], []), do: merge_columns(rest, [glyph(c)])

  defp merge_columns([c | rest], [g | acc]) do
    {cx0, _, cx1, _} = c.box
    {gx0, _, gx1, _} = g.box
    ov = min(cx1, gx1) - max(cx0, gx0) + 1

    if ov >= 0.5 * min(cx1 - cx0 + 1, gx1 - gx0 + 1) do
      merge_columns(rest, [join(g, c) | acc])
    else
      merge_columns(rest, [glyph(c), g | acc])
    end
  end

  defp glyph(c), do: %{box: c.box, pixels: c.pixels, parts: 1, area: c.area}

  defp join(g, c) do
    {a0, b0, a1, b1} = g.box
    {c0, d0, c1, d1} = c.box
    %{box: {min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)}, pixels: c.pixels ++ g.pixels, parts: g.parts + 1, area: g.area + c.area}
  end

  # ------------------------------------------------------------- features --

  @doc "The 416-value feature row of every glyph of a line (a list of lists)."
  def features(%{glyphs: gs, ref: ref, baseline: bl}) do
    Enum.map(gs, &glyph_features(&1, ref, bl))
  end

  defp glyph_features(%{box: {x0, y0, x1, y1}, pixels: px, parts: parts, area: area}, ref, bl) do
    {w, h} = {x1 - x0 + 1, y1 - y0 + 1}
    s = (@grid - 2) / max(w, h)
    {ox, oy} = {(@grid - w * s) / 2, (@grid - h * s) / 2}
    cov = :counters.new(@grid * @grid, [:write_concurrency])
    # coverage in 1/256ths of a cell, accumulated as integers
    for {x, y} <- px do
      {fx0, fy0} = {ox + (x - x0) * s, oy + (y - y0) * s}
      {fx1, fy1} = {fx0 + s, fy0 + s}

      for cy <- trunc(fy0)..min(ceil_i(fy1) - 1, @grid - 1)//1, cx <- trunc(fx0)..min(ceil_i(fx1) - 1, @grid - 1)//1 do
        a = (min(fx1, cx + 1) - max(fx0, cx)) * (min(fy1, cy + 1) - max(fy0, cy))
        if a > 0, do: :counters.add(cov, cy * @grid + cx + 1, round(a * 256))
      end
    end

    grid = for i <- 1..(@grid * @grid), do: min(:counters.get(cov, i) / 256, 1.0)
    r = ref * 1.0

    geo = [w / r, h / r, (y0 - bl) / r, (y1 - bl) / r, min(w / h, 4.0) / 4, area / (w * h), min(parts, 3) / 3,
           (if h <= 0.35 * r, do: 1.0, else: 0.0), (if y1 < bl - 0.3 * r, do: 1.0, else: 0.0), (if y1 > bl + 0.15 * r, do: 1.0, else: 0.0)]

    grid ++ geo ++ List.duplicate(0.0, @feat - length(geo))
  end

  defp ceil_i(v) do
    t = trunc(v)
    if v > t, do: t + 1, else: t
  end

  # ------------------------------------------------------- line bitmaps --

  @line_h 32
  @base_row 22
  @ref_px 12
  @win 8
  @stride 2

  @doc """
  A line normalised for the sequence reader: its own ink (the pixels of its
  glyphs only — a neighbour's descender never enters), scaled so that the
  line's median glyph height is #{@ref_px} px and the baseline sits on row
  #{@base_row} of #{@line_h}, as coverage in 0–255. Returns `%{w, h:
  #{@line_h}, data}` (row-major bytes).
  """
  def line_bitmap(%{glyphs: gs, ref: ref, baseline: bl, box: {x0, _, x1, _}}) do
    s = @ref_px / ref
    w = ceil_i((x1 - x0 + 1) * s) + 4
    cov = :counters.new(@line_h * w, [])

    for %{pixels: px} <- gs, {x, y} <- px do
      {fx0, fy0} = {(x - x0) * s + 2, (y - (bl + 1)) * s + @base_row}
      {fx1, fy1} = {fx0 + s, fy0 + s}

      for cy <- max(trunc_floor(fy0), 0)..min(ceil_i(fy1) - 1, @line_h - 1)//1, cx <- max(trunc_floor(fx0), 0)..min(ceil_i(fx1) - 1, w - 1)//1 do
        a = (min(fx1, cx + 1) - max(fx0, cx)) * (min(fy1, cy + 1) - max(fy0, cy))
        if a > 0, do: :counters.add(cov, cy * w + cx + 1, round(a * 1024))
      end
    end

    data = for i <- 1..(@line_h * w), into: <<>>, do: <<min(div(:counters.get(cov, i) * 255 + 512, 1024), 255)>>
    %{w: w, h: @line_h, data: data}
  end

  defp trunc_floor(v) do
    t = trunc(v)
    if v < t, do: t - 1, else: t
  end

  @doc """
  The frames of a line bitmap: windows of #{@win} columns every #{@stride}
  (a convolution whose stride is smaller than its kernel is a gather of
  overlapping windows), each the #{@line_h}×#{@win} values row-major in
  [0, 1] — `f32[T, #{@line_h * @win}]`, T = (w − #{@win}) ÷ #{@stride} + 1.
  """
  def frames(%{w: w, h: @line_h, data: data}) do
    {w, data} = if w < @win, do: {@win, pad_cols(data, w, @win)}, else: {w, data}
    t = div(w - @win, @stride) + 1

    vals =
      for f <- 0..(t - 1), y <- 0..(@line_h - 1), <<v <- binary_part(data, y * w + f * @stride, @win)>>, do: v / 255

    Vapor.Tensor.from_list(:f32, [t, @line_h * @win], vals)
  end

  defp pad_cols(data, w, to) do
    for y <- 0..(@line_h - 1), into: <<>>, do: binary_part(data, y * w, w) <> :binary.copy(<<0>>, to - w)
  end

  @doc "Frame geometry: `%{height, window, stride, row_width}`."
  def frame_geometry, do: %{height: @line_h, window: @win, stride: @stride, row_width: @line_h * @win, base_row: @base_row, ref_px: @ref_px}

  @doc "Every step at once: picture → `%{lines, mask_size}` with features (see `lines/2`)."
  def analyse(picture, opts \\ []) do
    g = gray(picture)
    mask = ink(g, opts)
    ls = mask |> components(opts) |> lines(opts)
    %{w: g.w, h: g.h, lines: Enum.map(ls, &Map.put(&1, :features, features(&1)))}
  end
end
