defmodule Vapor.Almizan.Format do
  @moduledoc """
  The canonical form of an Almizan file, **comments kept**: `vapor wzn
  fmt`, the language server's formatting and its Arabic/Latin lens.

  The canonical form is the printer's (`Vapor.Almizan.print/2`): one
  declaration per paragraph, Lisp indentation, closing parentheses on
  the last line, fractions in lowest terms, one script. The program does
  not change, and neither does its identity: `Almizan.hash/1` is computed
  on the tree, so formatting and changing script keep it. The tests check
  this on every shipped file.

  What the printer alone did not keep is the comments: the reader drops
  them. Here they are **glosses**, attached to the declaration they precede
  and printed above it. A comment inside a declaration moves above that
  declaration, in order (a tree has no place for it between two tokens).
  Comments after the last declaration stay at the end. A comment keeps its
  run of semicolons (`;`, `;;`, `;;;`) and its text, trimmed.

  Not done, on purpose: reordering declarations. The proposal behind this
  module wanted declarations sorted topologically, then alphabetically, so
  that two authors' files collapse to one hash. The order is the author's
  argument, the definitions before what uses them. Identity already does
  not depend on layout or script, and making it independent of order is a
  choice about the hash, not a rewrite of anyone's file.
  """
  alias Vapor.Almizan
  alias Vapor.Almizan.Syntax

  @doc """
  The canonical text: `{:ok, text}` (in `proj`, `:latin` or `:arabic`;
  default the file's own script) or `{:error, why}` when the text does not
  parse.
  """
  def format(text, proj \\ nil) do
    with {:ok, m} <- Almizan.parse(text),
         {:ok, glosses, trailer} <- glosses(text, length(m["decls"])) do
      proj = proj || Syntax.projection(strip_comments(text))

      body =
        m["decls"]
        |> Enum.zip(glosses)
        |> Enum.map_join("\n\n", fn {d, g} -> Enum.map_join(g, "", &(&1 <> "\n")) <> String.trim_trailing(Almizan.print(%{m | "decls" => [d]}, proj), "\n") end)

      tail = if trailer == [], do: "", else: "\n\n" <> Enum.join(trailer, "\n")
      {:ok, body <> tail <> "\n"}
    end
  end

  @doc "Whether `text` is already in canonical form (in its own script)."
  def canonical?(text), do: format(text) == {:ok, text}

  # comments by the top-level form they precede, sit in, or close the line of; and those after the last form
  defp glosses(text, n) do
    {forms, trailer, _depth, nil, _closed} = (text <> "\n") |> String.graphemes() |> Enum.reduce({[], [], 0, nil, false}, &step/2)
    forms = Enum.reverse(forms)

    if length(forms) == n,
      do: {:ok, forms, trailer},
      else: {:error, "#{length(forms)} top-level forms for #{n} declarations: not formatting"}
  end

  # {glosses per form (the current one first), comments waiting for the next form, depth,
  #  the comment being read with its owner, whether a top-level form closed on this line}
  defp step("\n", {f, p, d, {owner, c}, _}), do: place(f, p, d, owner, c)
  defp step("\n", {f, p, d, nil, _}), do: {f, p, d, nil, false}
  defp step(g, {f, p, d, {owner, c}, cl}), do: {f, p, d, {owner, c <> g}, cl}
  defp step(";", {[_ | _] = f, p, 0, nil, true}), do: {f, p, 0, {:form, ";"}, true}
  defp step(";", {f, p, 0, nil, cl}), do: {f, p, 0, {:next, ";"}, cl}
  defp step(";", {f, p, d, nil, cl}), do: {f, p, d, {:form, ";"}, cl}
  defp step("(", {f, p, 0, nil, cl}), do: {[p | f], [], 1, nil, cl}
  defp step("(", {f, p, d, nil, cl}), do: {f, p, d + 1, nil, cl}
  defp step(")", {f, p, 1, nil, _}), do: {f, p, 0, nil, true}
  defp step(")", {f, p, d, nil, cl}), do: {f, p, max(d - 1, 0), nil, cl}
  defp step(_, acc), do: acc

  # a finished comment: above the form it belongs to, or waiting for the next one
  defp place([cur | rest], p, d, :form, c), do: {[cur ++ [normal(c)] | rest], p, d, nil, false}
  defp place(f, p, d, _, c), do: {f, p ++ [normal(c)], d, nil, false}

  defp normal(c) do
    [_, semis, rest] = Regex.run(~r/^(;+)(.*)$/su, c)
    rest = String.trim(rest)
    if rest == "", do: semis, else: semis <> " " <> rest
  end

  defp strip_comments(text), do: text |> String.split("\n") |> Enum.map_join("\n", &(&1 |> String.split(";", parts: 2) |> hd()))
end
