defmodule Vapor.Vision.Bidi do
  @moduledoc """
  **From what the eye scans to what the text says**, for right-to-left
  lines.

  A reader of frames (CTC over columns) reads a line left to right: for
  Arabic or Hebrew that is the *visual* order, the reverse of the order in
  which the text is stored — except where left-to-right runs are embedded
  (numbers, Latin words), which the Unicode bidirectional algorithm lays
  out left to right inside the right-to-left line, and except for paired
  brackets, which it mirrors. `logical/1` undoes exactly that for a
  right-to-left paragraph: reverse the line, then restore every
  left-to-right run to its own order and swap the mirrored brackets back.

  This is the inverse of the display of a single-level RTL paragraph with
  embedded LTR runs (levels 1 and 2) — the case of printed Arabic. It is
  checked against `python-bidi`'s forward algorithm on every rendered test
  line (`test/vapor/scripts_test.exs`). Deeper embeddings (LTR inside RTL
  inside LTR, explicit controls) cannot be recovered from pixels at all:
  the controls are invisible.
  """

  @mirror %{"(" => ")", ")" => "(", "[" => "]", "]" => "[", "{" => "}", "}" => "{", "<" => ">", ">" => "<", "«" => "»", "»" => "«"}

  @doc "The logical (stored) order of a right-to-left line read in visual order."
  def logical(visual) when is_binary(visual) do
    visual
    |> String.graphemes()
    |> Enum.reverse()
    |> runs()
    |> Enum.flat_map(fn
      {:ltr, cs} -> Enum.reverse(cs)
      {:rtl, cs} -> Enum.map(cs, &Map.get(@mirror, &1, &1))
    end)
    |> Enum.join()
  end

  @doc "The visual order of a logical right-to-left line (the same rules forward; used to make labels)."
  def visual(logical) when is_binary(logical), do: logical(logical)

  @doc "Strong direction of a character: `:rtl`, `:ltr`, `:number` or `:neutral`."
  def class(c) do
    <<cp::utf8, _::binary>> = c

    cond do
      cp in ?0..?9 or cp in 0x0660..0x0669 or cp in 0x06F0..0x06F9 -> :number
      cp in 0x0590..0x08FF or cp in 0xFB1D..0xFDFF or cp in 0xFE70..0xFEFF -> :rtl
      cp in ?A..?Z or cp in ?a..?z or cp in 0x00C0..0x024F -> :ltr
      true -> :neutral
    end
  end

  # left-to-right runs: letters and numbers of the LTR classes, with the
  # separators that join digits (2020/12/28, 3.14, 1,000, 12:30) and the
  # spaces between Latin words of one run
  defp runs(cs) do
    t = List.to_tuple(cs)
    n = tuple_size(t)
    cls = fn i -> if i < 0 or i >= n, do: :none, else: class(elem(t, i)) end

    ltr? = fn i ->
      case cls.(i) do
        c when c in [:ltr, :number] -> true
        :neutral ->
          c = elem(t, i)
          (c in ["/", ".", ",", ":", "-", "+", "%"] and cls.(i - 1) == :number and cls.(i + 1) == :number) or
            (c == " " and cls.(i - 1) == :ltr and cls.(i + 1) == :ltr)
        _ -> false
      end
    end

    0..(n - 1)//1
    |> Enum.map(&{if(ltr?.(&1), do: :ltr, else: :rtl), elem(t, &1)})
    |> Enum.chunk_by(&elem(&1, 0))
    |> Enum.map(fn run -> {elem(hd(run), 0), Enum.map(run, &elem(&1, 1))} end)
  end
end
