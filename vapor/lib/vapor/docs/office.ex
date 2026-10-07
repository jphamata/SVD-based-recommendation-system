defmodule Vapor.Docs.Office do
  @moduledoc """
  Text of Office Open XML and OpenDocument files, from their XML parts
  (`Vapor.Docs.Markup` tokens — no entity expansion, no DTD):

    * **Word** — `word/document.xml` (+ footnotes, endnotes): paragraphs
      (`w:p`) on lines, runs (`w:t`), tabs, breaks, table cells separated by
      tabs and rows by lines;
    * **Excel** — every sheet in workbook order, named (`#sheet:Nome`): shared
      strings, inline strings, numbers, booleans; a formula without a cached
      value is written as its formula (`=B1*2`) — it is not evaluated;
    * **PowerPoint** — every slide in presentation order (`#slide3`);
    * **OpenDocument** — `content.xml`: paragraphs, headings, tabs, spaces,
      table cells.
  """
  alias Vapor.Docs
  alias Vapor.Docs.Markup

  def extract(:docx, name, x, acc) do
    parts = ["word/document.xml", "word/footnotes.xml", "word/endnotes.xml"] |> Enum.filter(&Map.has_key?(x, &1))
    text = parts |> Enum.map_join("\n", &lines(x[&1], text: ["w:t"], para: ["w:p", "w:tr"], tab: ["w:tab", "w:tc"], br: ["w:br", "w:cr"], cell: ["w:tc"]))
    Docs.add(acc, [%{doc: name, text: text, kind: :docx, meta: %{}}], [])
  end

  def extract(:xlsx, name, x, acc) do
    shared = shared_strings(x["xl/sharedStrings.xml"])
    rels = rels(x["xl/_rels/workbook.xml.rels"])

    sheets =
      for {:empty, "sheet", attrs} <- Markup.tokens(x["xl/workbook.xml"] || ""), do: {attr(attrs, "name"), attr(attrs, "r:id")}

    passages =
      for {sheet, rid} <- sheets, target = rels[rid], path = "xl/" <> String.trim_leading(target, "/xl/"), xml = x[path], xml != nil do
        %{doc: "#{name}#sheet:#{sheet}", text: cells(xml, shared), kind: :xlsx, meta: %{sheet: sheet}}
      end

    Docs.add(acc, passages, [])
  end

  def extract(:pptx, name, x, acc) do
    rels = rels(x["ppt/_rels/presentation.xml.rels"])
    order = for {:empty, "p:sldid", attrs} <- Markup.tokens(x["ppt/presentation.xml"] || ""), do: rels[attr(attrs, "r:id")]

    passages =
      for {target, i} <- Enum.with_index(order, 1), xml = x["ppt/" <> (target || "")], xml != nil do
        %{doc: "#{name}#slide#{i}", text: lines(xml, text: ["a:t"], para: ["a:p"], tab: [], br: ["a:br"]), kind: :pptx, meta: %{slide: i}}
      end

    Docs.add(acc, passages, [])
  end

  def extract(:odf, name, x, acc) do
    text = lines(x["content.xml"] || "", text: :all, para: ["text:p", "text:h", "table:table-row"], tab: ["text:tab", "table:table-cell"],
                 br: ["text:line-break"], space: ["text:s"], cell: ["table:table-cell"])
    Docs.add(acc, [%{doc: name, text: text, kind: :odf, meta: %{}}], [])
  end

  # text runs inside `text` tags (or everywhere with :all), paragraphs on lines
  defp lines(xml, rules) do
    text_tags = rules[:text]
    {para, tab, br, space, cell} = {rules[:para], rules[:tab], rules[:br], rules[:space] || [], rules[:cell] || []}

    # inside a table cell a paragraph ends with a space: the cell ends with a tab, the row with a line
    {out, _} =
      xml
      |> Markup.tokens()
      |> Enum.reduce({[], 0}, fn
        # d counts open text tags; +1000 per open table cell
        {:open, n, _}, {out, d} ->
          d = if(text_tags == :all or n in text_tags, do: d + 1, else: d)
          {out, if(n in cell, do: d + 1000, else: d)}
        {:close, n}, {out, d} ->
          {d, in_cell} = if n in cell, do: {d - 1000, false}, else: {d, d >= 1000}
          d = if(text_tags != :all and n in text_tags, do: max(d - 1, 0), else: d)
          cond do
            n in para and in_cell -> {[" " | out], d}
            n in para -> {["\n" | out], d}
            n in tab -> {["\t" | out], d}
            true -> {out, d}
          end
        {:empty, n, _}, {out, d} ->
          cond do
            n in br -> {["\n" | out], d}
            n in tab -> {["\t" | out], d}
            n in space -> {[" " | out], d}
            n in para -> {["\n" | out], d}
            true -> {out, d}
          end
        {:text, t}, {out, d} when rem(d, 1000) > 0 or text_tags == :all -> {[Markup.entities(t) | out], d}
        {:text, t, :raw}, {out, d} when rem(d, 1000) > 0 or text_tags == :all -> {[t | out], d}
        _, acc -> acc
      end)

    out
    |> Enum.reverse()
    |> IO.iodata_to_binary()
    |> String.split("\n")
    |> Enum.map(&(&1 |> String.replace(~r/ *\t */, "\t") |> String.trim(" ") |> String.trim_trailing("\t")))
    |> Enum.reject(&(&1 == ""))
    |> Enum.join("\n")
  end

  defp shared_strings(nil), do: {}

  defp shared_strings(xml) do
    {items, cur, _} =
      xml
      |> Markup.tokens()
      |> Enum.reduce({[], nil, false}, fn
        {:open, "si", _}, {items, _, _} -> {items, [], false}
        {:close, "si"}, {items, cur, _} -> {[cur |> Enum.reverse() |> IO.iodata_to_binary() | items], nil, false}
        {:open, "t", _}, {items, cur, _} -> {items, cur, true}
        {:close, "t"}, {items, cur, _} -> {items, cur, false}
        {:text, t}, {items, cur, true} when is_list(cur) -> {items, [Markup.entities(t) | cur], true}
        _, acc -> acc
      end)

    _ = cur
    items |> Enum.reverse() |> List.to_tuple()
  end

  # rows of a sheet, cells separated by tabs
  defp cells(xml, shared) do
    {rows, _} =
      xml
      |> Markup.tokens()
      |> Enum.reduce({[], nil}, fn
        {:open, "row", _}, {rows, _} -> {[[] | rows], nil}
        {:open, "c", attrs}, {rows, _} -> {rows, %{t: attr(attrs, "t"), v: nil, f: nil, is: [], in: nil}}
        {:empty, "c", _}, acc -> acc
        {:open, tag, _}, {rows, %{} = c} when tag in ["v", "f", "t"] -> {rows, %{c | in: tag}}
        {:close, tag}, {rows, %{} = c} when tag in ["v", "f", "t"] -> {rows, %{c | in: nil}}
        {:text, x}, {rows, %{in: "v"} = c} -> {rows, %{c | v: Markup.entities(x)}}
        {:text, x}, {rows, %{in: "f"} = c} -> {rows, %{c | f: Markup.entities(x)}}
        {:text, x}, {rows, %{in: "t"} = c} -> {rows, %{c | is: [Markup.entities(x) | c.is]}}
        {:close, "c"}, {[row | rows], %{} = c} -> {[[value(c, shared) | row] | rows], nil}
        _, acc -> acc
      end)

    rows |> Enum.reverse() |> Enum.map(&(&1 |> Enum.reverse() |> Enum.join("\t"))) |> Enum.reject(&(String.trim(&1) == "")) |> Enum.join("\n")
  end

  defp value(%{t: "s", v: v}, shared) when is_binary(v), do: elem(shared, String.to_integer(String.trim(v)))
  defp value(%{t: "inlineStr", is: is}, _), do: is |> Enum.reverse() |> Enum.join()
  defp value(%{t: "b", v: v}, _), do: if(v == "1", do: "TRUE", else: "FALSE")
  defp value(%{v: nil, f: f}, _) when is_binary(f), do: "=" <> f
  defp value(%{v: v}, _) when is_binary(v), do: v
  defp value(_, _), do: ""

  defp rels(nil), do: %{}

  defp rels(xml) do
    for {kind, "relationship", attrs} <- Markup.tokens(xml), kind in [:empty, :open], into: %{}, do: {attr(attrs, "Id"), attr(attrs, "Target")}
  end

  defp attr(attrs, name) do
    case Regex.run(~r/(?:^|\s)#{Regex.escape(name)}\s*=\s*"([^"]*)"/i, attrs) do
      [_, v] -> Markup.entities(v)
      nil -> nil
    end
  end
end
