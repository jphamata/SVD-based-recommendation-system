defmodule Vapor.Docs.Markup do
  @moduledoc """
  Tags and text, without an XML parser: a tokenizer that never expands an
  entity it was not given (no DTDs, no external entities, no billion
  laughs), enough for the text of HTML, XHTML and the XML parts of Office
  and OpenDocument files.
  """

  @doc "`[{:open, name, attrs_binary} | {:close, name} | {:empty, name, attrs} | {:text, binary}]`."
  def tokens(bin), do: tokens(bin, [])

  defp tokens(<<>>, acc), do: Enum.reverse(acc)
  defp tokens(<<"<!--", rest::binary>>, acc), do: tokens(after_marker(rest, "-->"), acc)
  defp tokens(<<"<![CDATA[", rest::binary>>, acc) do
    case :binary.split(rest, "]]>") do
      [t, r] -> tokens(r, [{:text, t, :raw} | acc])
      [t] -> tokens("", [{:text, t, :raw} | acc])
    end
  end
  defp tokens(<<"<?", rest::binary>>, acc), do: tokens(after_marker(rest, "?>"), acc)
  defp tokens(<<"<!", rest::binary>>, acc), do: tokens(after_marker(rest, ">"), acc)

  defp tokens(<<"<", rest::binary>> = all, acc) do
    case :binary.split(rest, ">") do
      [tag, r] ->
        tok =
          case tag do
            "/" <> name -> {:close, lname(name)}
            _ ->
              {name, attrs} = split_name(tag)
              if String.ends_with?(tag, "/"), do: {:empty, lname(name), attrs}, else: {:open, lname(name), attrs}
          end

        tokens(r, [tok | acc])

      [_] ->
        tokens("", [{:text, all} | acc])
    end
  end

  defp tokens(bin, acc) do
    case :binary.match(bin, "<") do
      {i, _} -> tokens(binary_part(bin, i, byte_size(bin) - i), [{:text, binary_part(bin, 0, i)} | acc])
      :nomatch -> tokens(<<>>, [{:text, bin} | acc])
    end
  end

  defp after_marker(bin, m) do
    case :binary.split(bin, m) do
      [_, r] -> r
      [_] -> ""
    end
  end

  defp split_name(tag) do
    tag = String.trim_trailing(tag, "/")
    case String.split(tag, ~r/\s/, parts: 2) do
      [n, a] -> {n, a}
      [n] -> {n, ""}
    end
  end

  defp lname(n), do: n |> String.trim() |> String.downcase()

  @doc "Decode the five XML entities and numeric references (and common HTML ones)."
  def entities(t) do
    Regex.replace(~r/&(#x[0-9a-fA-F]+|#[0-9]+|[a-zA-Z]+);/, t, fn whole, ref ->
      case ref do
        "#x" <> h -> cp(String.to_integer(h, 16), whole)
        "#" <> d -> cp(String.to_integer(d), whole)
        name -> Map.get(named(), name, whole)
      end
    end)
  end

  defp cp(n, _whole) when n in 0..0x10FFFF and n not in 0xD800..0xDFFF, do: <<n::utf8>>
  defp cp(_, whole), do: whole

  defp named do
    %{"amp" => "&", "lt" => "<", "gt" => ">", "quot" => "\"", "apos" => "'", "nbsp" => " ", "copy" => "©", "reg" => "®",
      "mdash" => "—", "ndash" => "–", "hellip" => "…", "laquo" => "«", "raquo" => "»", "eacute" => "é", "aacute" => "á",
      "atilde" => "ã", "ccedil" => "ç", "otilde" => "õ", "ecirc" => "ê", "ocirc" => "ô", "iacute" => "í", "uacute" => "ú"}
  end

  @blocks ~w(p div br li ul ol h1 h2 h3 h4 h5 h6 tr table section article header footer blockquote pre title dt dd hr figcaption)
  @drop ~w(script style noscript template svg head)

  @doc "Readable text of an HTML document: scripts/styles dropped, blocks on their own lines, the title first."
  def html_text(html) do
    toks = tokens(html)
    title = toks |> Enum.drop_while(&(not match?({:open, "title", _}, &1))) |> Enum.drop(1) |> Enum.take_while(&match?({:text, _}, &1))
    title = title |> Enum.map_join(fn {:text, t} -> t end) |> entities() |> String.trim()

    {out, _} =
      Enum.reduce(toks, {[], []}, fn
        {:open, n, _}, {out, stack} when n in @drop -> {out, [n | stack]}
        {:close, n}, {out, [n | rest]} when n in @drop -> {out, rest}
        _, {out, [_ | _] = stack} -> {out, stack}
        {kind, n, _}, {out, []} when kind in [:open, :empty] and n in @blocks -> {["\n" | out], []}
        {:close, n}, {out, []} when n in @blocks -> {["\n" | out], []}
        {:text, t}, {out, []} -> {[entities(t) | out], []}
        {:text, t, :raw}, {out, []} -> {[t | out], []}
        _, acc -> acc
      end)

    body = out |> Enum.reverse() |> IO.iodata_to_binary() |> tidy()
    if title != "" and not String.starts_with?(body, title), do: title <> "\n" <> body, else: body
  end

  @doc "The character data of an XML document (elements separated by spaces)."
  def xml_text(xml) do
    xml |> tokens() |> Enum.flat_map(fn {:text, t} -> [entities(t)]; {:text, t, :raw} -> [t]; _ -> [" "] end) |> IO.iodata_to_binary() |> tidy()
  end

  @doc "Collapse runs of spaces, keep line structure, trim."
  def tidy(t) do
    t
    |> String.replace(" ", " ")
    |> String.split("\n")
    |> Enum.map(&(&1 |> String.replace(~r/[ \t\r]+/, " ") |> String.trim()))
    |> Enum.reject(&(&1 == ""))
    |> Enum.join("\n")
  end
end
