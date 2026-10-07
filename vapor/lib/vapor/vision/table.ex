defmodule Vapor.Vision.Table do
  @moduledoc """
  **Tables on a scanned page, cell by cell** — the structure that every
  reader of invoices, statements, price lists and scientific papers needs,
  and that a text reader destroys: read in blocks, a table comes out column
  by column (0.7: "Código, DQ-20430, WF-62266, …, Descrição, lado, …"),
  with its rules read as underscores.

  First principles, no model — a table is declared by its **rules**:

    1. **Rules** are the long thin runs of ink. Every component that could
       hold one (long and thin, or large and sparse like a grid) is cut into
       maximal horizontal and vertical runs; runs longer than a few text
       heights, stacked over their thickness, are rule segments. Text never
       contributes: its runs are a stroke long.
    2. **Ruled tables** (`kind: :ruled`) — horizontal and vertical segments
       that cross form a cluster; its distinct positions are the grid. A
       table without an outer frame (lines only *between* cells) gets its
       missing edges where the crossing lines end. Two neighbouring grid
       positions are **one cell** exactly when the line between them is
       absent over their common span: merged cells (a header over two
       columns, a label over two rows) fall out of the geometry.
    3. **Rule-only tables** (`kind: :rules`) — horizontal rules of a common
       width and no vertical line: the three rules of a scientific table, or
       a rule under every row of a statement. Rows are the text lines
       between them; **columns are gutters** — vertical strips empty in
       *every* body row, wider than a word space. A header word that
       crosses a gutter makes its cell span the columns it covers. A set of
       rules whose body has no gutter is not a table (a separator above and
       below a paragraph stays prose).
    4. **Cells are read** by the OCR model, each on its own ink — the rules
       are removed, so a vertical line is never read as `|` or `l`.

  Output (`read/4`): `%{kind, box, rows, cols, header_rows, cells: [%{row,
  col, rowspan, colspan, box, text, confidence}]}` and the renderings
  `to_markdown/1` (GFM; a multi-row header is flattened "Group / Sub"),
  `to_html/1` (with `rowspan`/`colspan`), `to_csv/1` (RFC 4180),
  `to_rows/1` (lists of strings).

  Measured with the ICDAR 2013 table-competition metric (adjacency
  relations between neighbouring cells) on scanned pages in fonts the
  reader never saw, against two controls — the 0.7 reading (one line per
  row) and the grid without merged-cell detection: `docs/OCR.md §3e`,
  `mix vapor.quality`.

  Out (declared): tables with **no rule at all** (alignment alone cannot
  tell a table from columns of prose without a model of what cells hold),
  a cell's wrapped lines in a rule-only table (each line is a row there),
  rotated tables.
  """
  alias Vapor.Vision.{OCR, Segment}

  # --------------------------------------------------------------- detect --

  @doc """
  Find the tables among a page's components. Returns `{tables, rest}`:
  each table `%{kind, box, rows, cols, header_rows, cells: [%{row, col,
  rowspan, colspan, box, comps}]}` (the cells' ink, rules removed);
  `rest` the components outside every table (for the page's reading
  order). Options: `tables: false` (none), `spans: false` (no merged cells:
  the ablation control).
  """
  def detect(comps, opts \\ []) do
    if Keyword.get(opts, :tables, true) == false or comps == [] do
      {[], comps}
    else
      h0 = text_height(comps)
      t = max(4, round(0.3 * h0))
      {rule_comps, text} = Enum.split_with(comps, &rule_like?(&1, h0))
      pixels = Enum.flat_map(rule_comps, & &1.pixels)
      hs = h_segments(pixels, max(3 * h0, 40), t, h0)
      vs = v_segments(pixels, max(round(1.5 * h0), 24), t, h0)

      ruled = ruled_tables(hs, vs, t, Keyword.get(opts, :spans, true))
      used = ruled |> Enum.flat_map(& &1.rules) |> MapSet.new()
      open = rule_tables(Enum.reject(hs, &MapSet.member?(used, &1)), text, h0, t)

      tables =
        (ruled ++ open)
        |> Enum.map(fn tb -> fill(tb, text, t, h0) end)
        |> Enum.filter(&(&1.rows >= 2 and &1.cols >= 2 and Enum.any?(&1.cells, fn c -> c.comps != [] end)))
        |> Enum.sort_by(fn %{box: {_, y0, _, _}} -> y0 end)

      inside = fn %{box: {x0, y0, x1, y1}} ->
        {cx, cy} = {div(x0 + x1, 2), div(y0 + y1, 2)}
        Enum.any?(tables, fn %{box: {a0, b0, a1, b1}} -> cx >= a0 - t and cx <= a1 + t and cy >= b0 - t and cy <= b1 + t end)
      end

      # rule components of a table go with it; others (an underline, a lone
      # separator) stay on the page, where the line finder drops them
      table_rule? = fn c -> Enum.any?(tables, fn %{box: {a0, b0, a1, b1}} ->
        {x0, y0, x1, y1} = c.box
        x0 >= a0 - t and x1 <= a1 + t and y0 >= b0 - t and y1 <= b1 + t
      end) end

      rest = Enum.reject(text, inside) ++ Enum.reject(rule_comps, table_rule?)
      {Enum.map(tables, &Map.delete(&1, :rules)), Enum.sort_by(rest, fn %{box: {x0, y0, _, _}} -> {y0, x0} end)}
    end
  end

  defp text_height(comps) do
    hs = comps |> Enum.filter(&(&1.area >= 4)) |> Enum.map(&height/1) |> Enum.sort()
    if hs == [], do: 12, else: max(Enum.at(hs, div(length(hs), 2)), 6)
  end

  defp height(%{box: {_, y0, _, y1}}), do: y1 - y0 + 1
  defp width(%{box: {x0, _, x1, _}}), do: x1 - x0 + 1

  # long and thin either way, or large and sparse (a grid, a frame)
  defp rule_like?(c, h0) do
    {w, h} = {width(c), height(c)}
    thin = max(6, 0.5 * h0)

    (w >= max(3 * h0, 40) and h <= thin) or (h >= max(1.5 * h0, 24) and w <= thin) or
      (w >= 3 * h0 and h >= 1.5 * h0 and c.area <= 0.3 * w * h)
  end

  # maximal runs along one axis (gaps of ≤ 2 px bridged), stacked over the
  # rule's thickness; thicker than a rule is a filled box, not a line
  defp h_segments(pixels, min_len, t, h0) do
    pixels
    |> Enum.group_by(&elem(&1, 1), &elem(&1, 0))
    |> Enum.flat_map(fn {y, xs} -> for {a, b} <- runs(Enum.sort(xs)), b - a + 1 >= min_len, do: {y, a, b} end)
    |> stack(t)
    |> Enum.filter(fn s -> s.b1 - s.b0 + 1 <= max(8, 0.6 * h0) end)
    |> Enum.map(fn s -> %{dir: :h, y: div(s.b0 + s.b1, 2), x0: s.a0, x1: s.a1, thick: s.b1 - s.b0 + 1} end)
  end

  defp v_segments(pixels, min_len, t, h0) do
    pixels
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.flat_map(fn {x, ys} -> for {a, b} <- runs(Enum.sort(ys)), b - a + 1 >= min_len, do: {x, a, b} end)
    |> stack(t)
    |> Enum.filter(fn s -> s.b1 - s.b0 + 1 <= max(8, 0.6 * h0) end)
    |> Enum.map(fn s -> %{dir: :v, x: div(s.b0 + s.b1, 2), y0: s.a0, y1: s.a1, thick: s.b1 - s.b0 + 1} end)
  end

  defp runs([]), do: []

  defp runs([v | rest]) do
    {acc, a, b} =
      Enum.reduce(rest, {[], v, v}, fn x, {acc, a, b} -> if x - b <= 3, do: {acc, a, x}, else: {[{a, b} | acc], x, x} end)

    Enum.reverse([{a, b} | acc])
  end

  # runs `{across, along0, along1}` sorted by `across`: a run joins a
  # segment whose last row is adjacent and which it overlaps by half
  defp stack(runs, _t) do
    runs
    |> Enum.sort()
    |> Enum.reduce([], fn {c, a, b}, segs ->
      {hit, others} =
        Enum.split_with(segs, fn s ->
          c - s.b1 <= 2 and min(b, s.a1) - max(a, s.a0) >= 0.5 * min(b - a, s.a1 - s.a0)
        end)

      case hit do
        [] -> [%{b0: c, b1: c, a0: a, a1: b} | segs]
        [s | more] -> [%{s | b1: c, a0: min(a, s.a0), a1: max(b, s.a1)} | more ++ others]
      end
    end)
  end

  # ------------------------------------------------------------ ruled grids --

  defp ruled_tables(hs, vs, t, spans?) do
    nodes = List.to_tuple(hs ++ vs)
    n = tuple_size(nodes)
    cross? = fn h, v -> v.x >= h.x0 - t and v.x <= h.x1 + t and h.y >= v.y0 - t and h.y <= v.y1 + t end

    parent =
      for i <- 0..(n - 1)//1, j <- (i + 1)..(n - 1)//1, reduce: Map.new(0..(n - 1)//1, &{&1, &1}) do
        par ->
          {a, b} = {elem(nodes, i), elem(nodes, j)}

          linked =
            case {a.dir, b.dir} do
              {:h, :v} -> cross?.(a, b)
              {:v, :h} -> cross?.(b, a)
              _ -> false
            end

          if linked, do: union(par, i, j), else: par
      end

    0..(n - 1)//1
    |> Enum.group_by(&find(parent, &1))
    |> Map.values()
    |> Enum.map(fn idx -> Enum.map(idx, &elem(nodes, &1)) end)
    |> Enum.flat_map(fn segs ->
      {h, v} = Enum.split_with(segs, &(&1.dir == :h))
      if h != [] and v != [], do: List.wrap(grid(h, v, t, spans?)), else: []
    end)
  end

  defp find(par, i), do: if(par[i] == i, do: i, else: find(par, par[i]))
  defp union(par, i, j), do: Map.put(par, find(par, i), find(par, j))

  defp grid(h, v, t, spans?) do
    xs = positions(Enum.map(v, & &1.x), t)
    ys = positions(Enum.map(h, & &1.y), t)
    {hx0, hx1} = {h |> Enum.map(& &1.x0) |> Enum.min(), h |> Enum.map(& &1.x1) |> Enum.max()}
    {vy0, vy1} = {v |> Enum.map(& &1.y0) |> Enum.min(), v |> Enum.map(& &1.y1) |> Enum.max()}
    # a missing outer edge (lines only between cells) is where the crossing lines end
    xs = if(hx0 < hd(xs) - t, do: [hx0 | xs], else: xs)
    xs = if(hx1 > List.last(xs) + t, do: xs ++ [hx1], else: xs)
    ys = if(vy0 < hd(ys) - t, do: [vy0 | ys], else: ys)
    ys = if(vy1 > List.last(ys) + t, do: ys ++ [vy1], else: ys)
    {nr, nc} = {length(ys) - 1, length(xs) - 1}

    if nr < 1 or nc < 1 do
      nil
    else
      {xt, yt} = {List.to_tuple(xs), List.to_tuple(ys)}

      covered = fn segs, pos, a, b, key ->
        len =
          segs
          |> Enum.filter(&(abs(Map.fetch!(&1, key) - pos) <= t))
          |> Enum.map(fn s ->
            {s0, s1} = if key == :x, do: {s.y0, s.y1}, else: {s.x0, s.x1}
            max(0, min(s1, b) - max(s0, a))
          end)
          |> Enum.sum()

        len >= 0.5 * max(b - a, 1)
      end

      # neighbours with no line between them over their common span are one cell
      par = Map.new(for(r <- 0..(nr - 1), c <- 0..(nc - 1), do: {{r, c}, {r, c}}))

      # (`spans: false`, the ablation control: every grid position its own cell)
      covered = if spans?, do: covered, else: fn _, _, _, _, _ -> true end

      par =
        for r <- 0..(nr - 1), c <- 0..(nc - 2)//1, reduce: par do
          p -> if covered.(v, elem(xt, c + 1), elem(yt, r) + t, elem(yt, r + 1) - t, :x), do: p, else: union(p, {r, c}, {r, c + 1})
        end

      par =
        for r <- 0..(nr - 2)//1, c <- 0..(nc - 1), reduce: par do
          p -> if covered.(h, elem(yt, r + 1), elem(xt, c) + t, elem(xt, c + 1) - t, :y), do: p, else: union(p, {r, c}, {r + 1, c})
        end

      cells =
        par
        |> Map.keys()
        |> Enum.group_by(&find(par, &1))
        |> Map.values()
        |> Enum.flat_map(fn pos ->
          {rs, cs} = {Enum.map(pos, &elem(&1, 0)), Enum.map(pos, &elem(&1, 1))}
          {r0, r1, c0, c1} = {Enum.min(rs), Enum.max(rs), Enum.min(cs), Enum.max(cs)}
          # a merged region that is not a rectangle is read as its grid cells
          if length(pos) == (r1 - r0 + 1) * (c1 - c0 + 1),
            do: [cell(r0, c0, r1 - r0 + 1, c1 - c0 + 1, xt, yt)],
            else: Enum.map(pos, fn {r, c} -> cell(r, c, 1, 1, xt, yt) end)
        end)
        |> Enum.sort_by(&{&1.row, &1.col})

      header = cells |> Enum.filter(&(&1.row == 0)) |> Enum.map(& &1.rowspan) |> Enum.max(fn -> 1 end)

      %{kind: :ruled, box: {hd(xs), hd(ys), List.last(xs), List.last(ys)}, rows: nr, cols: nc,
        header_rows: if(nr > header, do: header, else: 0), cells: cells, rules: h ++ v}
    end
  end

  defp cell(r, c, rs, cs, xt, yt),
    do: %{row: r, col: c, rowspan: rs, colspan: cs, box: {elem(xt, c), elem(yt, r), elem(xt, c + cs), elem(yt, r + rs)}}

  # distinct positions: values within t of each other are one line
  defp positions(vals, t) do
    vals
    |> Enum.sort()
    |> Enum.chunk_while([], fn v, acc -> if acc == [] or v - hd(acc) <= t, do: {:cont, [v | acc]}, else: {:cont, Enum.reverse(acc), [v]} end,
      fn acc -> {:cont, Enum.reverse(acc), []} end)
    |> Enum.reject(&(&1 == []))
    |> Enum.map(&div(Enum.sum(&1), length(&1)))
  end

  # ------------------------------------------------------ rule-only tables --

  defp rule_tables(hs, text, h0, t) do
    hs
    |> Enum.sort_by(& &1.y)
    |> chains(h0)
    |> Enum.flat_map(fn chain -> List.wrap(open_table(chain, text, h0, t)) end)
  end

  # consecutive rules of a common extent (both ends within a tolerance);
  # shorter rules inside (a rule under a group header) do not break a chain
  defp chains(hs, h0) do
    tol = fn w -> max(2 * h0, 0.03 * w) end

    hs
    |> Enum.reduce([], fn s, chains ->
      w = s.x1 - s.x0

      case Enum.find_index(chains, fn [f | _] -> abs(f.x0 - s.x0) <= tol.(w) and abs(f.x1 - s.x1) <= tol.(w) end) do
        nil -> [[s] | chains]
        i -> List.update_at(chains, i, &[s | &1])
      end
    end)
    |> Enum.map(&Enum.reverse/1)
    |> Enum.filter(&(length(&1) >= 2))
  end

  defp open_table(rules, text, h0, t) do
    x0 = rules |> Enum.map(& &1.x0) |> Enum.min()
    x1 = rules |> Enum.map(& &1.x1) |> Enum.max()
    {y0, y1} = {hd(rules).y, List.last(rules).y}

    comps =
      Enum.filter(text, fn %{box: {a0, b0, a1, b1}} ->
        {cx, cy} = {div(a0 + a1, 2), div(b0 + b1, 2)}
        cx > x0 and cx < x1 and cy > y0 + t and cy < y1 - t
      end)

    lines = Segment.lines(comps)

    # the header: the lines above the second rule (three rules or more), else the first line
    {head, body} =
      if length(rules) >= 3 do
        y2 = Enum.at(rules, 1).y
        Enum.split_with(lines, fn %{box: {_, a, _, b}} -> div(a + b, 2) < y2 end)
      else
        Enum.split(lines, 1)
      end

    gaps = gutters(body, x0, x1, h0)

    if length(body) < 2 or gaps == [] do
      nil
    else
      bounds = [x0 | Enum.map(gaps, fn {a, b} -> div(a + b, 2) end)] ++ [x1]
      bt = List.to_tuple(bounds)
      nc = length(bounds) - 1
      all = head ++ body
      rows_y = row_bounds(all, y0, y1)

      cells =
        all
        |> Enum.with_index()
        |> Enum.flat_map(fn {line, r} ->
          {ya, yb} = Enum.at(rows_y, r)

          line
          |> words()
          |> Enum.map(fn {wa, wb} -> {cols_of(wa, wb, bt, nc), {wa, wb}} end)
          |> group_words()
          |> Enum.map(fn {c0, c1} -> %{row: r, col: c0, rowspan: 1, colspan: c1 - c0 + 1, box: {elem(bt, c0), ya, elem(bt, c1 + 1), yb}} end)
        end)

      %{kind: :rules, box: {x0, y0, x1, y1}, rows: length(all), cols: nc, header_rows: length(head),
        header: if(length(rules) >= 3, do: :ruled, else: :assumed), cells: cells, rules: rules}
      |> drop_empty_columns(length(head))
    end
  end

  # a column no body cell occupies is a gap mistaken for two gutters: its
  # neighbours' bounds absorb it
  defp drop_empty_columns(t, hr) do
    used = for c <- t.cells, c.row >= hr, k <- c.col..(c.col + c.colspan - 1), into: MapSet.new(), do: k
    empty = Enum.reject(0..(t.cols - 1), &MapSet.member?(used, &1))

    if empty == [] or MapSet.size(used) < 2 do
      t
    else
      shift = fn k -> k - Enum.count(empty, &(&1 < k)) end

      cells =
        Enum.map(t.cells, fn c ->
          last = c.col + c.colspan - 1
          keep = Enum.reject(c.col..last, &(&1 in empty))
          {c0, c1} = if keep == [], do: {c.col, c.col}, else: {hd(keep), List.last(keep)}
          %{c | col: shift.(c0), colspan: max(shift.(c1) - shift.(c0) + 1, 1)}
        end)

      %{t | cols: t.cols - length(empty), cells: cells}
    end
  end

  # empty vertical strips across every body line, wider than a word space
  defp gutters(lines, x0, x1, h0) do
    min_gap = max(h0, 12)

    # a speck of dust inside a gutter is not a column
    occupied =
      lines
      |> Enum.flat_map(fn l -> for %{box: {a, y0, b, y1}} = g <- l.glyphs, g.area >= 6 and y1 - y0 + 1 >= 0.3 * h0, do: {a, b} end)
      |> Enum.sort()
      |> Enum.reduce([], fn {a, b}, acc ->
        case acc do
          [{pa, pb} | rest] when a <= pb + 1 -> [{pa, max(b, pb)} | rest]
          _ -> [{a, b} | acc]
        end
      end)
      |> Enum.reverse()

    case occupied do
      [] -> []
      _ ->
        occupied
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [{_, b}, {a, _}] -> {b + 1, a - 1} end)
        |> Enum.filter(fn {a, b} -> b - a + 1 >= min_gap and a > x0 and b < x1 end)
    end
  end

  defp row_bounds(lines, y0, y1) do
    boxes = Enum.map(lines, fn %{box: {_, a, _, b}} -> {a, b} end)
    mids = boxes |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [{_, b}, {a, _}] -> div(b + a, 2) end)
    Enum.zip([y0 | mids], mids ++ [y1])
  end

  # words of a line: glyph runs between its spaces
  defp words(%{glyphs: gs, spaces: sp}) do
    cuts = MapSet.new(sp)

    gs
    |> Enum.with_index()
    |> Enum.chunk_while([], fn {g, i}, acc ->
      acc = [g | acc]
      if MapSet.member?(cuts, i), do: {:cont, Enum.reverse(acc), []}, else: {:cont, acc}
    end, fn acc -> {:cont, Enum.reverse(acc), []} end)
    |> Enum.reject(&(&1 == []))
    |> Enum.map(fn ws -> {ws |> Enum.map(fn %{box: {a, _, _, _}} -> a end) |> Enum.min(), ws |> Enum.map(fn %{box: {_, _, b, _}} -> b end) |> Enum.max()} end)
  end

  defp cols_of(a, b, bt, nc) do
    cs = for c <- 0..(nc - 1), min(b, elem(bt, c + 1)) - max(a, elem(bt, c)) > 0, do: c
    if cs == [], do: {0, 0}, else: {Enum.min(cs), Enum.max(cs)}
  end

  # words whose column ranges touch form one cell
  defp group_words(ws) do
    ws
    |> Enum.sort_by(fn {{c0, _}, {a, _}} -> {c0, a} end)
    |> Enum.reduce([], fn {{c0, c1}, _}, acc ->
      case acc do
        [{p0, p1} | rest] when c0 <= p1 -> [{p0, max(p1, c1)} | rest]
        _ -> [{c0, c1} | acc]
      end
    end)
    |> Enum.reverse()
  end

  # --------------------------------------------------------------- content --

  # each cell's ink: the text components whose centre lies inside it
  defp fill(tb, text, t, h0) do
    {a0, b0, a1, b1} = tb.box

    mine =
      Enum.filter(text, fn %{box: {x0, y0, x1, y1}} ->
        {cx, cy} = {div(x0 + x1, 2), div(y0 + y1, 2)}
        cx >= a0 - t and cx <= a1 + t and cy >= b0 - t and cy <= b1 + t
      end)

    cells =
      Enum.map(tb.cells, fn %{box: {x0, y0, x1, y1}} = c ->
        comps = Enum.filter(mine, fn %{box: {p0, q0, p1, q1}} ->
          {cx, cy} = {div(p0 + p1, 2), div(q0 + q1, 2)}
          cx >= x0 and cx < x1 and cy >= y0 and cy < y1
        end)

        Map.put(c, :comps, despeckle(comps, h0))
      end)

    %{tb | cells: cells}
  end

  # a speck of dust in a cell is not a character: a tiny component far from
  # every glyph-sized one is dropped (the dot of an i, an accent, a comma
  # sit next to their letters and stay)
  defp despeckle(comps, h0) do
    {big, small} = Enum.split_with(comps, &(&1.area >= 8 and height(&1) >= 0.25 * h0))
    reach = 0.6 * h0

    near? = fn %{box: {x0, y0, x1, y1}} ->
      Enum.any?(big, fn %{box: {a0, b0, a1, b1}} ->
        dx = max(0, max(a0 - x1, x0 - a1))
        dy = max(0, max(b0 - y1, y0 - b1))
        dx <= reach and dy <= reach
      end)
    end

    big ++ Enum.filter(small, near?)
  end

  # ------------------------------------------------------------------ read --

  @doc """
  Read every cell of a detected table with the OCR model: the cells gain
  `text`, `confidence` and `type` (their ink is dropped from the result).

  **A column is a type, and a column agrees with itself.** Every cell is
  first read freely (the frames' own reading, with the language model).
  Then, on the frames already computed — no second pass of the encoder:

    1. **Type.** A cell is *numeric-compatible* when its frames can be read
       with only digits and the column's own symbols (those most of its
       cells share: `R`, `$`, `.`, `,`, `%`, `/`, `-`…) at a cost of at most
       `type_gate` nats over their free best path. A column whose body cells
       are ≥ 70 % compatible is numeric, and its cells take the restricted
       reading — a `5` seen as `õ` becomes the best digit at those frames.
    2. **Spacing.** A space between two digits is either the column's
       convention ("1 234,56") or the reader's noise ("$5 1,295.29"): the
       hypothesis whose cells agree on a shape more often wins.
    3. **Shape.** When most cells share one shape (`AA-99999`,
       `99/99/9999`, `R$ 99.999,99` — `Vapor.Vision.Template`), every cell is
       decoded again *inside* that shape (a CTC Viterbi search over its
       automaton; letters limited to the column's alphabet) and the result
       kept when it costs the frames at most `shape_gate` nats.

  Text columns keep their free reading. The gates were chosen on a
  validation set of other tables (`docs/OCR.md §3e`); `typed: false` skips
  all three steps (the control).
  """
  def read(table, model, worker, lm, opts \\ []) do
    # a cell's ink is content: an unsure short token ("T1") is kept, not dropped as a mark
    first =
      Enum.map(table.cells, fn c ->
        r =
          if c.comps == [], do: %{text: "", confidence: 1.0, lines: []},
                            else: OCR.read_components(c.comps, model, worker, lm, Keyword.merge(opts, logprobs: true, marks: true))

        Map.merge(c, %{text: r.text, confidence: r.confidence, lines: Enum.map(r.lines, &Map.delete(&1, :chars)), type: :text})
      end)

    cells = if Keyword.get(opts, :typed, true), do: typed(first, table, model.labels, opts), else: first
    finish(%{table | cells: cells}, Keyword.get(opts, :keep_frames, false))
  end

  defp finish(table, keep_frames) do
    strip = fn ls -> if keep_frames, do: ls, else: Enum.map(ls, &Map.delete(&1, :lps)) end
    cells = Enum.map(table.cells, &(&1 |> Map.delete(:comps) |> Map.update(:lines, [], strip)))
    filled = Enum.reject(cells, &(&1.text == ""))
    conf = if filled == [], do: 0.0, else: Enum.sum(Enum.map(filled, & &1.confidence)) / length(filled)
    table |> Map.put(:cells, cells) |> Map.put(:confidence, conf)
  end

  @type_gate 2.0
  @shape_gate 1.5
  @numeric_chars MapSet.new(String.graphemes("0123456789.,%$R+-/:()€£ "))
  @digits MapSet.new(String.graphemes("0123456789"))
  @ascii_letters MapSet.new(String.graphemes("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))

  alias Vapor.Vision.Template

  defp typed(cells, table, labels, opts) do
    hr = table.header_rows
    {tg, sg} = {Keyword.get(opts, :type_gate, @type_gate), Keyword.get(opts, :shape_gate, @shape_gate)}
    body? = fn c -> c.row >= hr and c.colspan == 1 and c.text != "" end
    frames = fn c -> case c.lines do
      [%{lps: lps}] when is_list(lps) -> lps
      _ -> nil
    end end

    Enum.reduce(0..(table.cols - 1), cells, fn col, cells ->
      mine = Enum.filter(cells, &(&1.col == col and body?.(&1)))

      if mine == [] do
        cells
      else
        # 1. type: the column's symbols, then each cell's restricted reading and its cost
        symbols =
          mine
          |> Enum.flat_map(&(&1.text |> String.graphemes() |> Enum.uniq()))
          |> Enum.frequencies()
          |> Enum.filter(fn {ch, n} -> n >= 0.4 * length(mine) and MapSet.member?(@numeric_chars, ch) end)
          |> MapSet.new(&elem(&1, 0))

        allowed = @digits |> MapSet.union(symbols) |> MapSet.put(" ")

        restricted =
          Map.new(mine, fn c ->
            case frames.(c) do
              nil -> {{c.row, c.col}, nil}
              lps ->
                {text, score} = Template.restricted(lps, labels, allowed)
                ok = (Template.free_score(lps) - score) / max(String.length(text), 1) <= tg and digit_share(text) >= 0.5
                {{c.row, c.col}, if(ok, do: text)}
            end
          end)

        numeric? = Enum.count(restricted, fn {_, t} -> t != nil end) >= 0.7 * length(mine)

        cells =
          if numeric? do
            Enum.map(cells, fn c ->
              case Map.get(restricted, {c.row, c.col}) do
                t when is_binary(t) -> %{c | text: t, type: :numeric} |> Map.put(:free, c.text)
                _ -> c
              end
            end)
          else
            cells
          end

        # 2. spacing: noise or convention — whichever gives the stronger shape consensus
        mine = Enum.filter(cells, &(&1.col == col and body?.(&1)))
        squeeze = &Regex.replace(~r/(?<=[0-9.,\-\/:])\s+(?=[0-9.,%\-\/:])/u, &1, "")
        spaced = Enum.map(mine, & &1.text)
        tight = Enum.map(spaced, squeeze)

        {texts, squeezed?} =
          if numeric? and share(tight) > share(spaced), do: {tight, true}, else: {spaced, false}

        cells =
          if squeezed? do
            Enum.map(cells, fn c -> if c.col == col and body?.(c) and c.type == :numeric, do: %{c | text: squeeze.(c.text)}, else: c end)
          else
            cells
          end

        # 3. shape: decode every cell inside the column's consensus shape
        case Template.consensus(texts) do
          {:ok, shape} ->
            letters = column_letters(texts)

            Enum.map(cells, fn c ->
              with true <- c.col == col and body?.(c),
                   lps when is_list(lps) <- frames.(c),
                   {:ok, ks, score} <- Template.decode(lps, labels, shape, letters: letters),
                   text = Enum.map_join(ks, &elem(labels, &1)),
                   true <- (Template.free_score(lps) - score) / max(String.length(text), 1) <= sg do
                if text == c.text, do: Map.put(c, :shape, :kept), else: c |> Map.put_new(:free, c.text) |> Map.merge(%{text: text, shape: :decoded})
              else
                _ -> c
              end
            end)

          :none ->
            cells
        end
      end
    end)
  end

  defp digit_share(text) do
    g = text |> String.replace(" ", "") |> String.graphemes()
    if g == [], do: 0.0, else: Enum.count(g, &MapSet.member?(@digits, &1)) / length(g)
  end

  defp share(texts) do
    case texts |> Enum.map(&Template.of/1) |> Enum.frequencies() |> Map.values() do
      [] -> 0.0
      ns -> Enum.max(ns) / length(texts)
    end
  end

  # letters a shape's classes may take: unaccented ASCII, plus any other
  # letter most of the column's cells hold (an accent the column really uses)
  defp column_letters(texts) do
    extra =
      texts
      |> Enum.flat_map(&(&1 |> String.graphemes() |> Enum.uniq()))
      |> Enum.frequencies()
      |> Enum.filter(fn {g, n} -> n >= 0.4 * length(texts) and String.upcase(g) != String.downcase(g) end)
      |> MapSet.new(&elem(&1, 0))

    MapSet.union(@ascii_letters, extra)
  end

  # -------------------------------------------------------------- renderings --

  @doc "The table as rows of strings (a spanning cell's text in its first position, the rest empty)."
  def to_rows(%{rows: nr, cols: nc, cells: cells}) do
    m = Map.new(cells, &{{&1.row, &1.col}, &1.text})
    for r <- 0..(nr - 1), do: for(c <- 0..(nc - 1), do: Map.get(m, {r, c}, ""))
  end

  @doc """
  The column headings: a multi-row header flattened top to bottom ("Group /
  Sub"), a spanning heading repeated over the columns it covers.
  """
  def headings(%{header_rows: 0, cols: nc}), do: List.duplicate("", nc)

  def headings(%{header_rows: hr, cols: nc, cells: cells}) do
    for c <- 0..(nc - 1) do
      cells
      |> Enum.filter(fn x -> x.row < hr and c >= x.col and c < x.col + x.colspan and x.text != "" end)
      |> Enum.sort_by(& &1.row)
      |> Enum.map(& &1.text)
      |> Enum.uniq()
      |> Enum.join(" / ")
    end
  end

  defp body_rows(%{header_rows: hr} = t), do: t |> to_rows() |> Enum.drop(hr)

  @doc "GitHub-flavoured Markdown (pipes in cells escaped; a multi-row header flattened)."
  def to_markdown(t) do
    esc = fn s -> s |> String.replace("|", "\\|") |> String.replace("\n", " ") end
    head = headings(t)
    nc = length(head)
    line = fn row -> "| " <> Enum.map_join(row, " | ", esc) <> " |" end
    Enum.join([line.(head), "|" <> String.duplicate(" --- |", nc) | Enum.map(body_rows(t), line)], "\n")
  end

  @doc """
  HTML `<table>` with the structure as found: `rowspan`, `colspan`, `<th>`
  for header rows, an empty cell where the grid has no cell.
  """
  def to_html(%{rows: nr, cols: nc, cells: cells, header_rows: hr}) do
    esc = fn s -> s |> String.replace("&", "&amp;") |> String.replace("<", "&lt;") |> String.replace(">", "&gt;") end
    at = Map.new(cells, &{{&1.row, &1.col}, &1})
    covered = for c <- cells, r <- c.row..(c.row + c.rowspan - 1), k <- c.col..(c.col + c.colspan - 1), {r, k} != {c.row, c.col}, into: MapSet.new(), do: {r, k}

    tr = fn r ->
      tag = if r < hr, do: "th", else: "td"

      tds =
        for k <- 0..(nc - 1), not MapSet.member?(covered, {r, k}), into: "" do
          case at[{r, k}] do
            nil -> "<#{tag}></#{tag}>"
            c ->
              span = (if c.rowspan > 1, do: ~s( rowspan="#{c.rowspan}"), else: "") <> if(c.colspan > 1, do: ~s( colspan="#{c.colspan}"), else: "")
              "<#{tag}#{span}>#{esc.(c.text)}</#{tag}>"
          end
        end

      "<tr>" <> tds <> "</tr>"
    end

    head = if hr > 0, do: "<thead>" <> Enum.map_join(0..(hr - 1), tr) <> "</thead>", else: ""
    body = if nr > hr, do: "<tbody>" <> Enum.map_join(hr..(nr - 1), tr) <> "</tbody>", else: ""
    "<table>" <> head <> body <> "</table>"
  end

  @doc "CSV (RFC 4180): the flattened headings, then the body rows."
  def to_csv(t) do
    q = fn s -> if String.contains?(s, [",", "\"", "\n", "\r"]), do: "\"" <> String.replace(s, "\"", "\"\"") <> "\"", else: s end
    rows = if t.header_rows > 0, do: [headings(t) | body_rows(t)], else: to_rows(t)
    Enum.map_join(rows, "\r\n", fn row -> Enum.map_join(row, ",", q) end) <> "\r\n"
  end

  # -------------------------------------------------------------- measure --

  @doc """
  Score a reading against a ground truth with the ICDAR 2013 table metric:
  the **adjacency relations** of non-empty cells (each cell and its nearest
  non-empty neighbour to the right and below, over every row/column it
  spans). Predicted cells are matched to truth cells by containment of the
  truth cell's centre. Returns `%{precision, recall, f1, exact, cer}` —
  `exact` the share of truth cells found with the same row, column and
  spans, `cer` the mean character error of the cells' text (a missed cell
  counts 1).

  Truth: `%{rows, cols, cells: [%{row, col, rowspan, colspan, text, box}]}`
  (boxes in page pixels).
  """
  def score(pred_cells, truth_cells) do
    truth = truth_cells |> Enum.with_index() |> Enum.map(fn {c, i} -> Map.put(c, :id, i) end)
    centre = fn {x0, y0, x1, y1} -> {(x0 + x1) / 2, (y0 + y1) / 2} end

    # each predicted cell → the truth cells whose centre it contains
    owner =
      Map.new(truth, fn tc ->
        {cx, cy} = centre.(tc.box)

        p =
          Enum.find_index(pred_cells, fn %{box: {x0, y0, x1, y1}} -> cx >= x0 and cx < x1 and cy >= y0 and cy < y1 end)

        {tc.id, p}
      end)

    # a predicted cell owning several truth cells merged them: no match for any
    counts = owner |> Map.values() |> Enum.reject(&is_nil/1) |> Enum.frequencies()
    matched = Map.new(owner, fn {tid, p} -> {tid, if(p != nil and counts[p] == 1, do: p, else: nil)} end)

    rel_truth = relations(truth, & &1.id)
    pred_ided = pred_cells |> Enum.with_index() |> Enum.map(fn {c, i} -> Map.put(c, :pid, i) end) |> Enum.reject(&(Map.get(&1, :text, "x") == ""))
    pid_to_tid = for {tid, p} <- matched, p != nil, into: %{}, do: {p, tid}
    rel_pred = relations(pred_ided, & &1.pid) |> Enum.map(fn {a, b, d} -> {pid_to_tid[a], pid_to_tid[b], d} end)
    {rt, rp} = {MapSet.new(rel_truth), rel_pred}
    hit = Enum.count(rp, &MapSet.member?(rt, &1))
    precision = if rp == [], do: 0.0, else: hit / length(rp)
    recall = if MapSet.size(rt) == 0, do: 0.0, else: hit / MapSet.size(rt)
    f1 = if precision + recall == 0, do: 0.0, else: 2 * precision * recall / (precision + recall)

    exact =
      Enum.count(truth, fn tc ->
        case matched[tc.id] do
          nil -> false
          p -> pc = Enum.at(pred_cells, p); {pc.rowspan, pc.colspan} == {tc.rowspan, tc.colspan}
        end
      end) / max(length(truth), 1)

    cer =
      Enum.map(truth, fn tc ->
        case matched[tc.id] do
          nil -> 1.0
          p -> min(OCR.cer(Map.get(Enum.at(pred_cells, p), :text, ""), tc.text), 1.0)
        end
      end)
      |> then(&(Enum.sum(&1) / max(length(&1), 1)))

    %{precision: precision, recall: recall, f1: f1, exact: exact, cer: cer}
  end

  # ICDAR 2013: for every non-empty cell and every row (column) it spans,
  # the nearest non-empty cell to the right (below)
  defp relations(cells, id) do
    occupied =
      for c <- cells, r <- c.row..(c.row + c.rowspan - 1), k <- c.col..(c.col + c.colspan - 1), into: %{}, do: {{r, k}, c}

    maxc = cells |> Enum.map(&(&1.col + &1.colspan)) |> Enum.max(fn -> 0 end)
    maxr = cells |> Enum.map(&(&1.row + &1.rowspan)) |> Enum.max(fn -> 0 end)

    right =
      for c <- cells, r <- c.row..(c.row + c.rowspan - 1),
          n = Enum.find_value((c.col + c.colspan)..maxc//1, fn k -> occupied[{r, k}] end), n != nil,
          do: {id.(c), id.(n), :right}

    down =
      for c <- cells, k <- c.col..(c.col + c.colspan - 1),
          n = Enum.find_value((c.row + c.rowspan)..maxr//1, fn r -> occupied[{r, k}] end), n != nil,
          do: {id.(c), id.(n), :down}

    Enum.uniq(right ++ down)
  end
end
