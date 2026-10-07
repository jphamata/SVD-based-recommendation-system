defmodule Vapor.TableTest do
  @moduledoc """
  Tables on scanned pages (`Vapor.Vision.Table`, docs/OCR.md §3e): the
  geometry on synthetic ink (no model), the renderings, the ICDAR-2013
  metric, column shapes (`Vapor.Vision.Template`), and — with the native
  worker — real scanned tables in unseen fonts, against the 0.7 reading.
  """
  use ExUnit.Case, async: true
  alias Vapor.Vision.{Table, Template}

  # ink as a tuple of row tuples, drawn with rectangles
  defp canvas(w, h, rects) do
    on = MapSet.new(for {x0, y0, x1, y1} <- rects, y <- y0..y1, x <- x0..x1, do: {x, y})
    for(y <- 0..(h - 1), do: for(x <- 0..(w - 1), do: if(MapSet.member?(on, {x, y}), do: 1, else: 0)) |> List.to_tuple()) |> List.to_tuple()
  end

  # a "word": a few glyph-sized blobs (14 px tall) starting at x
  defp word(x, y, n), do: for(i <- 0..(n - 1), do: {x + i * 12, y, x + i * 12 + 8, y + 13})

  defp comps(mask), do: Vapor.Vision.Segment.components(mask)

  test "a ruled grid with a header spanning two columns and a label spanning two rows" do
    # 4 columns × 4 rows; (0,0) spans rows 0–1; (0,2) spans columns 2–3
    xs = [20, 120, 220, 320, 420]
    ys = [20, 60, 100, 140, 180]
    t = 2
    vlines =
      for {x, i} <- Enum.with_index(xs), r <- 0..3, not (i == 3 and r == 0), not (i == 0 and false),
          do: {x, Enum.at(ys, r), x + t, Enum.at(ys, r + 1)}
    hlines =
      for {y, j} <- Enum.with_index(ys), c <- 0..3, not (j == 1 and c in [0, 1]),
          do: {Enum.at(xs, c), y, Enum.at(xs, c + 1) + t, y + t}

    texts = [word(40, 70, 3), word(240, 33, 4), word(240, 73, 2), word(340, 73, 2)] ++
              for(r <- 2..3, c <- 0..3, do: word(Enum.at(xs, c) + 15, Enum.at(ys, r) + 13, 3))

    mask = canvas(460, 200, vlines ++ hlines ++ List.flatten(texts))
    {[tb], rest} = Table.detect(comps(mask))
    assert tb.kind == :ruled and {tb.rows, tb.cols} == {4, 4}
    assert rest == []
    spans = Map.new(tb.cells, &{{&1.row, &1.col}, {&1.rowspan, &1.colspan}})
    assert spans[{0, 0}] == {2, 1}
    assert spans[{0, 2}] == {1, 2}
    assert spans[{1, 2}] == {1, 1} and spans[{3, 3}] == {1, 1}
    assert tb.header_rows == 2
    # every word landed in its cell, the rules in none
    assert Enum.count(tb.cells, &(&1.comps != [])) == 12

    # the ablation control: no merged cells
    {[flat], _} = Table.detect(comps(mask), spans: false)
    assert length(flat.cells) == 16
  end

  test "lines only between cells (no frame): the missing edges come from where the lines end" do
    # 3 columns, 3 rows; two inner verticals across the table, two inner horizontals across it
    vlines = [{120, 20, 121, 140}, {220, 20, 221, 140}]
    hlines = [{20, 60, 320, 61}, {20, 100, 320, 101}]
    texts = for r <- 0..2, c <- 0..2, do: word(30 + c * 100, 30 + r * 40, 3)
    {[tb], _} = Table.detect(comps(canvas(340, 160, vlines ++ hlines ++ List.flatten(texts))))
    assert {tb.rows, tb.cols} == {3, 3}
    assert Enum.all?(tb.cells, &(&1.comps != []))
  end

  test "three rules and no vertical line: columns are gutters; a header word over a gutter spans" do
    rules = [{20, 20, 520, 22}, {20, 80, 520, 82}, {20, 220, 520, 222}]
    # header: two rows — a group heading over columns 1–2, then three headings
    group = word(250, 30, 8)
    heads = [word(30, 56, 4), word(220, 56, 3), word(400, 56, 3)]
    body = for r <- 0..3, c <- 0..2, do: word(30 + c * 190, 92 + r * 32, 4)
    {[tb], _} = Table.detect(comps(canvas(540, 240, rules ++ List.flatten([group | heads ++ body]))))
    assert tb.kind == :rules and tb.cols == 3 and tb.rows == 6 and tb.header_rows == 2
    g = Enum.find(tb.cells, &(&1.row == 0))
    assert {g.col, g.colspan} == {1, 2}
  end

  test "two rules around a paragraph are not a table (no gutter in the body)" do
    rules = [{20, 20, 520, 22}, {20, 160, 520, 162}]
    # prose: words at irregular offsets — every column of the strip is inked in some line
    lines = for r <- 0..3, do: Enum.flat_map(0..9, fn k -> word(30 + k * 48 + rem(r * 17 + k * 7, 23), 40 + r * 28, 3) end)
    assert {[], _} = Table.detect(comps(canvas(540, 180, rules ++ List.flatten(lines))))
  end

  defp sample do
    %{rows: 4, cols: 3, header_rows: 2,
      cells: [%{row: 0, col: 0, rowspan: 2, colspan: 1, text: "Item"}, %{row: 0, col: 1, rowspan: 1, colspan: 2, text: "Quarter"},
              %{row: 1, col: 1, rowspan: 1, colspan: 1, text: "Q1"}, %{row: 1, col: 2, rowspan: 1, colspan: 1, text: "Q2"},
              %{row: 2, col: 0, rowspan: 1, colspan: 1, text: "a|b"}, %{row: 2, col: 1, rowspan: 1, colspan: 1, text: "1,5"},
              %{row: 2, col: 2, rowspan: 1, colspan: 1, text: "2"}, %{row: 3, col: 0, rowspan: 1, colspan: 1, text: "<c>"},
              %{row: 3, col: 1, rowspan: 1, colspan: 1, text: "3"}, %{row: 3, col: 2, rowspan: 1, colspan: 1, text: "4 \"x\""}]}
  end

  test "renderings: Markdown flattens the header, HTML keeps the spans, CSV quotes per RFC 4180" do
    t = sample()
    assert Table.headings(t) == ["Item", "Quarter / Q1", "Quarter / Q2"]

    assert Table.to_markdown(t) ==
             "| Item | Quarter / Q1 | Quarter / Q2 |\n| --- | --- | --- |\n| a\\|b | 1,5 | 2 |\n| <c> | 3 | 4 \"x\" |"

    html = Table.to_html(t)
    assert html =~ ~s(<thead><tr><th rowspan="2">Item</th><th colspan="2">Quarter</th></tr><tr><th>Q1</th><th>Q2</th></tr></thead>)
    assert html =~ "<td>&lt;c&gt;</td>"
    # an unoccupied grid position is an empty cell, so a span stays over its own columns
    open = %{rows: 2, cols: 3, header_rows: 1, cells: [%{row: 0, col: 1, rowspan: 1, colspan: 2, text: "Q"}] ++
             for(k <- 0..2, do: %{row: 1, col: k, rowspan: 1, colspan: 1, text: "#{k}"})}
    assert Table.to_html(open) =~ ~s(<thead><tr><th></th><th colspan="2">Q</th></tr></thead>)
    assert Table.to_csv(t) == "Item,Quarter / Q1,Quarter / Q2\r\na|b,\"1,5\",2\r\n<c>,3,\"4 \"\"x\"\"\"\r\n"
  end

  test "the ICDAR-2013 metric: perfect = 1; merging two cells loses their relations; a wrong span is not exact" do
    box = fn r, c, rs, cs -> {c * 100, r * 40, (c + cs) * 100, (r + rs) * 40} end
    truth = for c <- sample().cells, do: Map.put(c, :box, box.(c.row, c.col, c.rowspan, c.colspan))
    assert %{f1: 1.0, exact: 1.0, cer: 0.0} = Table.score(truth, truth)

    # two body cells read as one (as a gutter missed would)
    merged =
      truth
      |> Enum.reject(&({&1.row, &1.col} in [{3, 1}, {3, 2}]))
      |> Kernel.++([%{row: 3, col: 1, rowspan: 1, colspan: 2, text: "3 4", box: box.(3, 1, 1, 2)}])

    s = Table.score(merged, truth)
    assert s.f1 < 0.9 and s.exact < 1.0 and s.cer > 0.0
  end

  test "column shapes: learned by consensus (digits required), decoded inside the shape's automaton" do
    assert Template.of("WF-62266") == [{:class, :upper}, {:lit, "-"}, {:class, :digit}]
    assert Template.of("R$ 54.104,43") == [{:class, :upper}, {:lit, "$"}, {:lit, " "}, {:class, :digit}, {:lit, "."}, {:class, :digit}, {:lit, ","}, {:class, :digit}]
    # a class the whole column spells alike is a constant of the column
    assert {:ok, [{:lit, "R"}, {:lit, "$"}, {:lit, " "}, {:class, :digit}, {:lit, "."}, {:class, :digit}, {:lit, ","}, {:class, :digit}]} =
             Template.consensus(["R$ 54.104,43", "R$ 9.881,95", "R$ 30.045,02", "R$ õ0.6,47"])
    assert {:ok, [{:class, :upper}, {:lit, "-"}, {:class, :digit}]} = Template.consensus(["AB-123", "CD-45", "E F-6", "GH-789"])
    assert :none = Template.consensus(["alpha", "beta", "gamma"])
    assert :none = Template.consensus(["AB-1", "x"])

    # frames that say "W", blank, " ", "F", "-", "6": inside the shape the space cannot be read
    labels = List.to_tuple(["", " ", "-", "6", "F", "W", "õ"])
    lp = fn probs -> probs |> Enum.map(&:math.log/1) |> List.to_tuple() end
    frames = [
      lp.([0.01, 0.01, 0.01, 0.01, 0.01, 0.94, 0.01]),
      lp.([0.6, 0.35, 0.01, 0.01, 0.01, 0.01, 0.01]),
      lp.([0.3, 0.55, 0.01, 0.01, 0.11, 0.01, 0.01]),
      lp.([0.01, 0.01, 0.01, 0.01, 0.94, 0.01, 0.01]),
      lp.([0.01, 0.01, 0.94, 0.01, 0.01, 0.01, 0.01]),
      lp.([0.01, 0.01, 0.01, 0.40, 0.01, 0.01, 0.55])
    ]

    {free, _} = Template.restricted(frames, labels, MapSet.new(["W", "F", "-", "6", "õ", " "]))
    assert free == "W F-õ"
    {:ok, ks, score} = Template.decode(frames, labels, [{:class, :upper}, {:lit, "-"}, {:class, :digit}])
    assert Enum.map_join(ks, &elem(labels, &1)) == "WF-6"
    assert score <= Template.free_score(frames)
    # a shape the frames cannot spell (more characters than frames) is refused
    assert :none = Template.decode(Enum.take(frames, 2), labels, [{:class, :upper}, {:lit, "-"}, {:class, :digit}])
  end

  @tag :native
  @tag timeout: 900_000
  test "scanned tables in unseen fonts: structure F1, typed cell reading, against the 0.7 reading" do
    w = Vapor.Vision.OCR.worker()
    e = Vapor.Quality.Round08.evaluate_tables(Path.expand("priv/quality/tables"), worker: w, names: ["t01_grid", "t05_booktabs", "t07_hrules", "t11_inner"])

    for r <- e.rows do
      assert r.dims == r.truth_dims, r.name
      assert r.f1 >= 0.95, "#{r.name}: F1 #{r.f1}"
      assert r.lines_f1 < r.f1
    end

    assert e.mean.cer < 0.08 and e.mean.cer < e.mean.free_cer
    assert e.mean.grid_nospan_f1 < e.mean.grid_f1
  end

  @tag :native
  test "a scanned PDF with a table: the library gets the table as a Markdown passage with its structure" do
    {:ok, r} = Vapor.Docs.ingest(Path.expand("priv/quality/tables/t01_grid.pdf"))
    [t] = Enum.filter(r.passages, &(&1.kind == :table))
    assert t.doc =~ "#table1"
    assert t.text =~ ~r/^\| Código \| Descrição \| .+ \| .+ \|\n\| --- \| --- \| --- \| --- \|/u
    assert t.text =~ "| DQ-20430 | lado |"
    assert t.meta.table.rows == 7 and t.meta.table.cols == 4
    assert t.meta.table.html =~ ~s(colspan="2")
    # the prose around the table is a passage of its own, without the cells
    [page] = Enum.filter(r.passages, &(&1.kind == :pdf))
    refute page.text =~ "DQ-20430"
  end
end
