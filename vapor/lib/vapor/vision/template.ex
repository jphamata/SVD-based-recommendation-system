defmodule Vapor.Vision.Template do
  @moduledoc """
  **A table column declares its own grammar.** The cells of a column of
  codes, dates or amounts share a shape — `AA-99999`, `99/99/9999`,
  `R$ 99.999,99` — and a reader that saw one cell's `5` as `õ`, or put a
  space inside `WF-62266`, broke the shape every other cell of the column
  keeps. So the shape is learned from the column itself (a consensus of its
  cells' readings) and each cell is decoded again **inside** it: a CTC
  Viterbi search over the shape's automaton, the same move as a grammar
  restricting a language model's tokens (`Vapor.Grammar`), applied to the
  frames of the OCR reader.

  A shape is a sequence of tokens: `{:class, :digit | :upper | :lower}`
  (one or more characters of the class) or `{:lit, char}` (that character).
  `"WF-62266"` → `[upper+, "-", digit+]`. A class every cell of the column
  spells the same way is a constant: in a column of amounts the shape is
  `["R", "$", " ", digit+, ".", digit+, ",", digit+]`.

  It never forces: a constrained reading is kept only when it costs the
  frames at most `gate` nats more than their free best path — a cell that
  really differs (an "N/A" in a column of amounts) keeps its own reading.
  Only shapes that hold digits are learned (structured data; prose columns
  are left to the language model).
  """

  @neg_inf -1.0e30

  @doc "The shape of a reading."
  def of(text) do
    text
    |> String.graphemes()
    |> Enum.map(&token/1)
    |> Enum.chunk_by(& &1)
    |> Enum.flat_map(fn [{:class, _} = t | _] -> [t]; lits -> lits end)
  end

  defp token(g) do
    cond do
      g =~ ~r/^[0-9]$/ -> {:class, :digit}
      String.upcase(g) == g and String.downcase(g) != g -> {:class, :upper}
      String.downcase(g) == g and String.upcase(g) != g -> {:class, :lower}
      true -> {:lit, g}
    end
  end

  @doc """
  The column's shape: the one at least `share` (0.6) of `texts` have, if
  there are at least `min` (3) of them and it holds a digit; else `:none`.
  """
  def consensus(texts, opts \\ []) do
    texts = Enum.reject(texts, &(&1 == ""))
    {share, min} = {Keyword.get(opts, :share, 0.6), Keyword.get(opts, :min, 3)}

    case texts |> Enum.map(&of/1) |> Enum.frequencies() |> Enum.max_by(&elem(&1, 1), fn -> nil end) do
      {shape, n} when n >= min and n >= share * length(texts) ->
        if Enum.member?(shape, {:class, :digit}), do: {:ok, fix_constants(shape, Enum.filter(texts, &(of(&1) == shape)))}, else: :none

      _ ->
        :none
    end
  end

  # a class token every cell of the consensus spells the same way (the "R"
  # of "R$", the "INV" of "INV-0042") is a constant of the column, not a class
  defp fix_constants(shape, texts) do
    parts = Enum.map(texts, &split/1)

    shape
    |> Enum.with_index()
    |> Enum.flat_map(fn
      {{:class, _} = t, i} ->
        case parts |> Enum.map(&Enum.at(&1, i)) |> Enum.uniq() do
          [same] -> same |> String.graphemes() |> Enum.map(&{:lit, &1})
          _ -> [t]
        end

      {t, _} ->
        [t]
    end)
  end

  # the substrings of a text, one per token of its shape
  defp split(text) do
    text
    |> String.graphemes()
    |> Enum.chunk_by(&token/1)
    |> Enum.flat_map(fn [g | _] = run -> if match?({:class, _}, token(g)), do: [Enum.join(run)], else: run end)
  end

  @doc "Score of the frames' free best path (the sum of every frame's best log-probability)."
  def free_score(lps), do: Enum.reduce(lps, 0.0, fn lp, acc -> acc + Enum.max(Tuple.to_list(lp)) end)

  @doc """
  The frames' best path with only `allowed` characters (and the blank):
  `{text, score}` — greedy over the restricted labels, which is the best
  path of the restricted label set.
  """
  def restricted(lps, labels, allowed) do
    keep = for k <- 1..(tuple_size(labels) - 1), MapSet.member?(allowed, elem(labels, k)), do: k
    keep = [0 | keep]

    {ks, score} =
      Enum.map_reduce(lps, 0.0, fn lp, acc ->
        k = Enum.max_by(keep, &elem(lp, &1))
        {k, acc + elem(lp, k)}
      end)

    text = ks |> Enum.chunk_by(& &1) |> Enum.map(&hd/1) |> Enum.reject(&(&1 == 0)) |> Enum.map_join(&elem(labels, &1)) |> String.trim()
    {text, score}
  end

  @doc """
  The best reading of the frames `lps` (per-frame log-probability tuples,
  blank = 0) inside `shape`: `{:ok, label_indices, score}` or `:none` when
  the shape cannot be spelled in the frames available. Option `letters`:
  the characters a letter class may take (default: every label of the class).
  """
  def decode(lps, labels, shape, opts \\ []) do
    toks = List.to_tuple(shape)
    n = tuple_size(toks)
    letters = Keyword.get(opts, :letters)
    allowed = Enum.map(shape, &labels_of(&1, labels, letters)) |> List.to_tuple()

    if n == 0 or Enum.any?(Tuple.to_list(allowed), &(&1 == [])) do
      :none
    else
      repeat? = fn i -> match?({:class, _}, elem(toks, i)) end
      # states: {:b, i} — after token i (i = −1: nothing yet); {:c, i, k} — in token i, last label k
      init = %{{:b, -1} => {0.0, []}}

      final =
        Enum.reduce(lps, init, fn lp, states ->
          Enum.reduce(states, %{}, fn {st, {sc, path}}, acc ->
            moves =
              case st do
                {:b, i} ->
                  [{{:b, i}, 0, nil}] ++
                    if(i >= 0 and repeat?.(i), do: for(k <- elem(allowed, i), do: {{:c, i, k}, k, k}), else: []) ++
                    if(i + 1 < n, do: for(k <- elem(allowed, i + 1), do: {{:c, i + 1, k}, k, k}), else: [])

                {:c, i, k} ->
                  [{{:c, i, k}, k, nil}, {{:b, i}, 0, nil}] ++
                    if(repeat?.(i), do: for(k2 <- elem(allowed, i), k2 != k, do: {{:c, i, k2}, k2, k2}), else: []) ++
                    if(i + 1 < n, do: for(k2 <- elem(allowed, i + 1), k2 != k, do: {{:c, i + 1, k2}, k2, k2}), else: [])
              end

            Enum.reduce(moves, acc, fn {to, emit, out}, acc ->
              s2 = sc + elem(lp, emit)
              p2 = if out, do: [out | path], else: path

              case acc do
                %{^to => {best, _}} when best >= s2 -> acc
                _ -> Map.put(acc, to, {s2, p2})
              end
            end)
          end)
        end)

      ends = for {{:b, i}, v} <- final, i == n - 1, do: v
      ends = ends ++ for {{:c, i, _}, v} <- final, i == n - 1, do: v

      case Enum.max_by(ends, &elem(&1, 0), fn -> nil end) do
        {score, path} when score > @neg_inf / 2 -> {:ok, Enum.reverse(path), score}
        _ -> :none
      end
    end
  end

  defp labels_of({:lit, g}, labels, _letters), do: for(k <- 1..(tuple_size(labels) - 1), elem(labels, k) == g, do: k)

  defp labels_of({:class, c}, labels, letters) do
    for k <- 1..(tuple_size(labels) - 1), token(elem(labels, k)) == {:class, c},
        c == :digit or letters == nil or MapSet.member?(letters, elem(labels, k)), do: k
  end

  @doc "Whether a reading has the shape."
  def fits?(text, shape), do: of(text) == shape
end
