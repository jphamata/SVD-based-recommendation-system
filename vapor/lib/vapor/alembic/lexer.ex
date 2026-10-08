defmodule Vapor.Alembic.Lexer do
  @moduledoc """
  Alembic's lexer (docs/ALEMBIC.md §1). Tokens are `{kind, value, line, col}`
  with a closed set of kinds; identifiers stay binaries — nothing a user
  types ever becomes an atom.

  Newlines matter only at bracket depth 0 and only where a definition can
  end: a line ending in an operator, a comma or an opening keyword, or a
  line starting with `|>`, `then`, `else`, `and`, `or`, `in`, continues
  the previous one.
  """

  @keywords ~w(if then else let in for and or not true false nil xor)
  @max_tokens 400_000

  @doc "Tokens, or `{:error, message, line, col}`."
  def lex(text) when is_binary(text) do
    if byte_size(text) > 2_000_000 do
      {:error, "source larger than 2 MB", 1, 1}
    else
      try do
        toks = scan(text, 1, 1, 0, [])
        if length(toks) > @max_tokens, do: throw({:lex, "program too long (over #{@max_tokens} tokens)", 1, 1})
        {:ok, finish(toks)}
      catch
        {:lex, msg, l, c} -> {:error, msg, l, c}
      end
    end
  end

  # the scan accumulates in reverse; `finish` restores order, applies the
  # continuation rules and collapses runs of newlines
  defp finish(rev) do
    toks = Enum.reverse(rev)
    toks |> continuation([]) |> Enum.reverse()
  end

  @cont_after [:op, :comma, :lparen, :lbracket, :lbrace, :colon, :semicolon]
  @cont_kw ~w(if then else let in and or not xor for)
  @cont_before ~w(then else and or in xor)

  defp continuation([], acc), do: acc

  defp continuation([{:nl, _, _, _} = nl | rest], acc) do
    prev = List.first(acc)
    nxt = Enum.find(rest, fn {k, _, _, _} -> k != :nl end)

    cond do
      prev == nil -> continuation(rest, acc)
      match?({:nl, _, _, _}, prev) -> continuation(rest, acc)
      soft_after?(prev) -> continuation(rest, acc)
      nxt != nil and soft_before?(nxt) -> continuation(rest, acc)
      nxt == nil -> continuation(rest, acc)
      true -> continuation(rest, [nl | acc])
    end
  end

  defp continuation([t | rest], acc), do: continuation(rest, [t | acc])

  defp soft_after?({k, v, _, _}), do: k in @cont_after or (k == :kw and v in @cont_kw) or (k == :op and v in ["=", "=>"])
  defp soft_before?({:op, v, _, _}), do: v in ["|>", ".."] or false
  defp soft_before?({:kw, v, _, _}), do: v in @cont_before
  defp soft_before?(_), do: false

  # ------------------------------------------------------------------ scan

  defp scan(<<>>, l, c, _d, acc), do: [{:eof, nil, l, c} | acc]

  defp scan(<<"#", rest::binary>>, l, c, d, acc) do
    {rest, n} = skip_line(rest, 0)
    scan(rest, l, c + n + 1, d, acc)
  end

  defp scan(<<"\r\n", rest::binary>>, l, _c, d, acc), do: scan(rest, l + 1, 1, d, nl(acc, d, l))
  defp scan(<<"\n", rest::binary>>, l, _c, d, acc), do: scan(rest, l + 1, 1, d, nl(acc, d, l))
  defp scan(<<ch, rest::binary>>, l, c, d, acc) when ch in [?\s, ?\t, ?\r], do: scan(rest, l, c + 1, d, acc)

  defp scan(<<ch, _::binary>> = s, l, c, d, acc) when ch in ?0..?9 do
    {tok, rest, n} = number(s)
    scan(rest, l, c + n, d, push({elem(tok, 0), elem(tok, 1), l, c}, acc))
  end

  defp scan(<<"\"", rest::binary>>, l, c, d, acc) do
    {str, rest, n, lines} = string(rest, [], 1, 0, l, c)
    c2 = if lines > 0, do: n, else: c + n
    scan(rest, l + lines, c2, d, push({:str, str, l, c}, acc))
  end

  defp scan(s, l, c, d, acc) do
    case ident(s) do
      {"", _} -> symbol(s, l, c, d, acc)
      {name, rest} ->
        n = String.length(name)
        tok = if name in @keywords, do: {:kw, name, l, c}, else: {:id, name, l, c}
        scan(rest, l, c + n, d, push(tok, acc))
    end
  end

  @symbols [
    {"|>", :op}, {"=>", :op}, {"->", :op}, {"==", :op}, {"!=", :op}, {"<=", :op}, {">=", :op}, {"<<", :op}, {">>", :op},
    {"//", :op}, {"**", :op}, {"++", :op}, {"..", :op}, {"≤", :op}, {"≥", :op}, {"≠", :op},
    {"+", :op}, {"-", :op}, {"*", :op}, {"/", :op}, {"%", :op}, {"^", :op}, {"<", :op}, {">", :op}, {"=", :op},
    {"&", :op}, {"|", :op}, {"~", :op}, {".", :dot}, {"(", :lparen}, {")", :rparen}, {"[", :lbracket}, {"]", :rbracket},
    {"{", :lbrace}, {"}", :rbrace}, {",", :comma}, {":", :colon}, {";", :semicolon}, {"−", :op}, {"×", :op}, {"·", :op}
  ]
  @canon %{"≤" => "<=", "≥" => ">=", "≠" => "!=", "−" => "-", "×" => "*", "·" => "*"}

  defp symbol(s, l, c, d, acc) do
    case Enum.find(@symbols, fn {p, _} -> String.starts_with?(s, p) end) do
      {p, kind} ->
        rest = binary_part(s, byte_size(p), byte_size(s) - byte_size(p))
        d2 = case kind do k when k in [:lparen, :lbracket, :lbrace] -> d + 1; k when k in [:rparen, :rbracket, :rbrace] -> max(d - 1, 0); _ -> d end
        scan(rest, l, c + String.length(p), d2, push({kind, Map.get(@canon, p, p), l, c}, acc))

      nil ->
        g = case String.next_grapheme(s) do {g, _} -> g; nil -> "?" end
        throw({:lex, "unexpected character #{inspect(g)}", l, c})
    end
  end

  defp push(tok, acc), do: [tok | acc]

  defp nl(acc, 0, l), do: [{:nl, nil, l, 0} | acc]
  defp nl(acc, _d, _l), do: acc

  defp skip_line(<<"\n", _::binary>> = s, n), do: {s, n}
  defp skip_line(<<>>, n), do: {<<>>, n}
  defp skip_line(<<_, rest::binary>>, n), do: skip_line(rest, n + 1)

  defp ident(s), do: ident(s, [])
  defp ident(<<ch::utf8, rest::binary>>, acc) when ch in ?a..?z or ch in ?A..?Z or ch == ?_ or (ch >= 0xC0 and ch < 0x2000 and ch not in [0xD7, 0xF7]) do
    ident(rest, [<<ch::utf8>> | acc])
  end
  defp ident(<<ch::utf8, rest::binary>>, [_ | _] = acc) when ch in ?0..?9 or ch == ?', do: ident(rest, [<<ch::utf8>> | acc])
  defp ident(rest, acc), do: {acc |> Enum.reverse() |> IO.iodata_to_binary(), rest}

  defp number(s) do
    {int, rest} = digits(s, [])
    cond do
      # `1..5` is a range, not a float
      match?(<<".", d, _::binary>> when d in ?0..?9, rest) ->
        <<".", r2::binary>> = rest
        {frac, r3} = digits(r2, [])
        {exp, r4} = exponent(r3)
        txt = int <> "." <> frac <> exp
        {{:float, parse_float(txt)}, r4, String.length(txt)}

      match?(<<e, _::binary>> when e in [?e, ?E], rest) and exponent(rest) != {"", rest} ->
        {exp, r4} = exponent(rest)
        txt = int <> ".0" <> exp
        {{:float, parse_float(txt)}, r4, String.length(int <> exp)}

      true ->
        clean = String.replace(int, "_", "")
        if byte_size(clean) > 2000, do: throw({:lex, "integer literal longer than 2000 digits", 0, 0})
        {{:int, String.to_integer(clean)}, rest, String.length(int)}
    end
  end

  defp parse_float(txt) do
    case Float.parse(String.replace(txt, "_", "")) do
      {f, ""} -> f
      _ -> throw({:lex, "bad number #{txt}", 0, 0})
    end
  rescue
    _ -> throw({:lex, "number out of range #{txt}", 0, 0})
  end

  defp digits(<<d, rest::binary>>, acc) when d in ?0..?9 or d == ?_, do: digits(rest, [d | acc])
  defp digits(rest, acc), do: {acc |> Enum.reverse() |> List.to_string(), rest}

  defp exponent(<<e, sign, d, rest::binary>>) when e in [?e, ?E] and sign in [?+, ?-] and d in ?0..?9 do
    {ds, r} = digits(<<d, rest::binary>>, [])
    {<<e, sign>> <> ds, r}
  end

  defp exponent(<<e, d, rest::binary>>) when e in [?e, ?E] and d in ?0..?9 do
    {ds, r} = digits(<<d, rest::binary>>, [])
    {<<e>> <> ds, r}
  end

  defp exponent(rest), do: {"", rest}

  defp string(<<"\"", rest::binary>>, acc, n, lines, _l, _c), do: {acc |> Enum.reverse() |> IO.iodata_to_binary(), rest, n + 1, lines}
  defp string(<<"\\n", rest::binary>>, acc, n, lines, l, c), do: string(rest, ["\n" | acc], n + 2, lines, l, c)
  defp string(<<"\\t", rest::binary>>, acc, n, lines, l, c), do: string(rest, ["\t" | acc], n + 2, lines, l, c)
  defp string(<<"\\\"", rest::binary>>, acc, n, lines, l, c), do: string(rest, ["\"" | acc], n + 2, lines, l, c)
  defp string(<<"\\\\", rest::binary>>, acc, n, lines, l, c), do: string(rest, ["\\" | acc], n + 2, lines, l, c)
  defp string(<<"\n", rest::binary>>, acc, _n, lines, l, c), do: string(rest, ["\n" | acc], 1, lines + 1, l, c)
  defp string(<<ch::utf8, rest::binary>>, acc, n, lines, l, c), do: string(rest, [<<ch::utf8>> | acc], n + 1, lines, l, c)
  defp string(<<>>, _acc, _n, _lines, l, c), do: throw({:lex, "string not closed", l, c})
  defp string(<<_, _::binary>>, _acc, _n, _lines, l, c), do: throw({:lex, "invalid UTF-8 in string", l, c})
end
