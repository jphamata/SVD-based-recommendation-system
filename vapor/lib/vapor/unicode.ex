defmodule Vapor.Unicode do
  @moduledoc """
  Unicode normalization forms NFC and NFKC (UAX #15), exactly.

  OTP's `:unicode.characters_to_nfc_binary/1` composes across a blocking
  starter: for `и` U+0438, `๎` U+0E4E (a non-spacing mark of combining
  class 0), `◌̈` U+0308 it returns `ӥ` U+04E5, `๎` — but a class-0
  character blocks, and the normal form keeps all three. Tokenizers apply
  NFC before looking up tokens, so such a difference changes token ids.

  This module implements the algorithm from the standard's definition:
  full decomposition (canonical, or compatibility for NFKC; Hangul
  arithmetically), canonical ordering of combining marks, then canonical
  composition in which a mark composes with the last starter only if no
  character in between has a combining class of 0 or ≥ its own.

  The character data are OTP's (`:unicode_util.lookup/1`: decompositions
  and combining classes, Unicode #{:unicode_util.spec_version() |> Tuple.to_list() |> Enum.join(".")}).
  The primary-composite table is derived at compile time: a character with
  a canonical decomposition `d₁…dₙ` is a primary composite exactly when
  NFC keeps it in isolation (where OTP's composition is correct — no
  blocking arises), and its pair is `(NFC(d₁…dₙ₋₁), dₙ)`.
  """
  # Hangul syllable arithmetic (Unicode §3.12)
  @s_base 0xAC00
  @l_base 0x1100
  @v_base 0x1161
  @t_base 0x11A7
  @l_count 19
  @v_count 21
  @t_count 28
  @n_count @v_count * @t_count
  @s_count @l_count * @n_count

  @compose (for cp <- Enum.concat([0..0xD7FF, 0xE000..0x10FFFF]),
                not (cp >= 0xAC00 and cp < 0xAC00 + 11_172),
                %{canon: canon} = :unicode_util.lookup(cp),
                length(canon) >= 2,
                :unicode.characters_to_nfc_list([cp]) == [cp],
                {init, [{_, last}]} = Enum.split(canon, -1),
                [first] = :unicode.characters_to_nfc_list(Enum.map(init, &elem(&1, 1))),
                into: %{},
                do: {{first, last}, cp})

  @version :unicode_util.spec_version() |> Tuple.to_list() |> Enum.join(".")

  @doc """
  The Unicode version of the character data compiled in (OTP's). Normal
  forms — and so token ids — of characters added in later versions depend
  on it: it belongs to the identity of anything that tokenizes text
  reproducibly (`Vapor.Agent.Spec` records it).
  """
  def version, do: @version

  @doc "Number of primary composites in the table (a sanity figure for tests)."
  def composites, do: map_size(@compose)

  @doc "NFC of a UTF-8 binary."
  def nfc(bin), do: normalize(bin, :canon)

  @doc "NFKC of a UTF-8 binary."
  def nfkc(bin), do: normalize(bin, :compat)

  # below U+0300 every character is its own NFC and NFKC-stable except the
  # compatibility characters in Latin-1 (NBSP, ª, ², ¼ …): a fast path for
  # text that is mostly ASCII
  defp normalize(bin, form) do
    if quick?(bin, form) do
      bin
    else
      bin
      |> String.to_charlist()
      |> Enum.flat_map(&decompose(&1, form))
      |> reorder()
      |> compose()
      |> List.to_string()
    end
  end

  defp quick?(<<c, rest::binary>>, form) when c < 0x80, do: quick?(rest, form)
  defp quick?(<<>>, _), do: true
  defp quick?(_, _), do: false

  defp decompose(cp, _form) when cp >= @s_base and cp < @s_base + @s_count do
    s = cp - @s_base
    l = @l_base + div(s, @n_count)
    v = @v_base + div(rem(s, @n_count), @t_count)
    t = @t_base + rem(s, @t_count)
    if t == @t_base, do: [{l, 0}, {v, 0}], else: [{l, 0}, {v, 0}, {t, 0}]
  end

  defp decompose(cp, form) do
    %{canon: canon, compat: compat, ccc: ccc} = :unicode_util.lookup(cp)

    case {form, canon, compat} do
      # compatibility mappings carry their tag: {:compat | :noBreak | :font | …, list}
      {:compat, _, {_tag, [_ | _] = m}} -> Enum.flat_map(m, fn {_, c} -> decompose(c, :compat) end)
      {_, [_ | _], _} -> Enum.map(canon, fn {k, c} -> {c, k} end)
      _ -> [{cp, ccc}]
    end
  end

  # stable sort of each run of non-starters by combining class
  defp reorder(cs), do: reorder(cs, [], [])
  defp reorder([], run, acc), do: Enum.reverse(flush(run, acc))
  defp reorder([{_, 0} = c | rest], run, acc), do: reorder(rest, [], [c | flush(run, acc)])
  defp reorder([c | rest], run, acc), do: reorder(rest, [c | run], acc)

  defp flush([], acc), do: acc
  defp flush(run, acc), do: Enum.reverse(Enum.sort_by(Enum.reverse(run), &elem(&1, 1)), acc)

  # canonical composition. State: the characters before the last starter
  # (`done`, reversed), the last starter, the marks after it (`tail`,
  # reversed; all of non-zero class) and the class of the last mark. A
  # character composes with the starter unless blocked: some character
  # between them has class 0 or a class ≥ its own — with the marks in
  # canonical order, only the last mark needs checking.
  defp compose(cs) do
    {done, starter, tail, _} =
      Enum.reduce(cs, {[], nil, [], nil}, fn {c, k}, {done, s, tail, last} ->
        free = tail == [] or (k != 0 and last < k)

        cond do
          s != nil and free and composite(s, c) != nil -> {done, composite(s, c), tail, last}
          k == 0 -> {tail ++ emit(s, done), c, [], nil}
          true -> {done, s, [c | tail], k}
        end
      end)

    Enum.reverse(tail ++ emit(starter, done))
  end

  defp emit(nil, done), do: done
  defp emit(s, done), do: [s | done]

  defp composite(l, v) when l >= @l_base and l < @l_base + @l_count and v >= @v_base and v < @v_base + @v_count,
    do: @s_base + ((l - @l_base) * @v_count + (v - @v_base)) * @t_count

  defp composite(lv, t)
       when lv >= @s_base and lv < @s_base + @s_count and rem(lv - @s_base, @t_count) == 0 and
              t > @t_base and t < @t_base + @t_count,
       do: lv + (t - @t_base)

  defp composite(a, b), do: Map.get(@compose, {a, b})

end
