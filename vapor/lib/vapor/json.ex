defmodule Vapor.JSON do
  @moduledoc """
  RFC 8259 JSON, in-tree (`deps: []`): the formats vapor ingests —
  `config.json`, `tokenizer.json`, safetensors headers — and the OpenAI wire
  format it serves are all JSON.

  Decoding maps objects to maps with string keys, arrays to lists, numbers to
  integers or floats, `true`/`false`/`null` to `true`/`false`/`nil`.
  Duplicate keys are an error (a safetensors header must not be able to say
  two things about one tensor). With `ordered: true` objects decode to
  `{:dict, [{key, value}]}` in document order instead — what a chat
  template needs to reproduce Python's insertion-ordered dicts — and
  `encode/1` writes such a value back in that order. Nesting is bounded (`@max_depth`), so a
  hostile document cannot exhaust the stack.
  """

  @max_depth 512

  @spec decode(binary, keyword) :: {:ok, term} | {:error, {non_neg_integer, String.t()}}
  def decode(bin, opts \\ []) when is_binary(bin) do
    o = {Keyword.get(opts, :ordered, false), Keyword.get(opts, :nonfinite, false)}
    {v, rest} = value(skip(bin), 0, o)

    case skip(rest) do
      <<>> -> {:ok, v}
      more -> throw({:error, byte_size(bin) - byte_size(more), "trailing data"})
    end
  catch
    {:error, pos, why} -> {:error, {pos, why}}
    {:error_at, rest, why} -> {:error, {byte_size(bin) - byte_size(rest), why}}
  end

  def decode!(bin) do
    case decode(bin) do
      {:ok, v} -> v
      {:error, {pos, why}} -> raise ArgumentError, "invalid JSON at byte #{pos}: #{why}"
    end
  end

  # ------------------------------------------------------------- decoding --

  defp skip(<<c, rest::binary>>) when c in [?\s, ?\t, ?\n, ?\r], do: skip(rest)
  defp skip(bin), do: bin

  defp value(_bin, depth, _o) when depth > @max_depth, do: throw({:error, 0, "nesting deeper than #{@max_depth}"})
  defp value(<<?{, rest::binary>>, d, o), do: object(skip(rest), %{}, d + 1, o)
  defp value(<<?[, rest::binary>>, d, o), do: array(skip(rest), [], d + 1, o)
  defp value(<<?", rest::binary>>, _d, _o), do: string(rest, [])
  defp value(<<"true", rest::binary>>, _d, _o), do: {true, rest}
  defp value(<<"false", rest::binary>>, _d, _o), do: {false, rest}
  defp value(<<"null", rest::binary>>, _d, _o), do: {nil, rest}
  defp value(<<"Infinity", rest::binary>>, _d, {_, true}), do: {:infinity, rest}
  defp value(<<"-Infinity", rest::binary>>, _d, {_, true}), do: {:neg_infinity, rest}
  defp value(<<"NaN", rest::binary>>, _d, {_, true}), do: {:nan, rest}
  defp value(<<c, _::binary>> = bin, _d, _o) when c == ?- or c in ?0..?9, do: number(bin)
  defp value(bin, _d, _o), do: throw({:error_at, bin, "unexpected token"})

  defp object(<<?}, rest::binary>>, acc, _d, o), do: {finish(acc, o), rest}
  defp object(bin, acc, d, o), do: members(bin, acc, d, o)

  # ordered: the map also records the key order under a reserved key
  defp finish(acc, {false, _}), do: acc
  defp finish(acc, {true, _}), do: {:dict, acc |> Map.get(:"$order", []) |> Enum.reverse() |> Enum.map(&{&1, Map.fetch!(acc, &1)})}

  # after '{' or ',' a member must follow (no trailing commas)
  defp members(bin, acc, d, o) do
    {k, rest} =
      case bin do
        <<?", r::binary>> -> string(r, [])
        _ -> throw({:error_at, bin, "expected a string key"})
      end

    if Map.has_key?(acc, k), do: throw({:error_at, rest, "duplicate key #{inspect(k)}"})

    rest =
      case skip(rest) do
        <<?:, r::binary>> -> skip(r)
        other -> throw({:error_at, other, "expected ':'"})
      end

    {v, rest} = value(rest, d, o)
    acc = Map.put(acc, k, v)
    acc = if elem(o, 0), do: Map.update(acc, :"$order", [k], &[k | &1]), else: acc

    case skip(rest) do
      <<?,, r::binary>> -> members(skip(r), acc, d, o)
      <<?}, r::binary>> -> {finish(acc, o), r}
      other -> throw({:error_at, other, "expected ',' or '}'"})
    end
  end

  defp array(<<?], rest::binary>>, acc, _d, _o), do: {Enum.reverse(acc), rest}
  defp array(bin, acc, d, o), do: elements(bin, acc, d, o)

  defp elements(bin, acc, d, o) do
    {v, rest} = value(bin, d, o)

    case skip(rest) do
      <<?,, r::binary>> -> elements(skip(r), [v | acc], d, o)
      <<?], r::binary>> -> {Enum.reverse([v | acc]), r}
      other -> throw({:error_at, other, "expected ',' or ']'"})
    end
  end

  # strings: runs of plain bytes are copied as sub-binaries
  defp string(bin, acc) do
    n = plain(bin, 0)
    <<run::binary-size(n), rest::binary>> = bin

    case rest do
      <<?", r::binary>> -> {IO.iodata_to_binary([acc, run]), r}
      <<?\\, r::binary>> -> escape(r, [acc, run])
      <<c, _::binary>> when c < 0x20 -> throw({:error_at, rest, "control character in string"})
      <<>> -> throw({:error_at, rest, "unterminated string"})
    end
  end

  defp plain(<<c, rest::binary>>, n) when c != ?" and c != ?\\ and c >= 0x20, do: plain(rest, n + 1)
  defp plain(_, n), do: n

  defp escape(<<?", r::binary>>, acc), do: string(r, [acc, ?"])
  defp escape(<<?\\, r::binary>>, acc), do: string(r, [acc, ?\\])
  defp escape(<<?/, r::binary>>, acc), do: string(r, [acc, ?/])
  defp escape(<<?b, r::binary>>, acc), do: string(r, [acc, ?\b])
  defp escape(<<?f, r::binary>>, acc), do: string(r, [acc, ?\f])
  defp escape(<<?n, r::binary>>, acc), do: string(r, [acc, ?\n])
  defp escape(<<?r, r::binary>>, acc), do: string(r, [acc, ?\r])
  defp escape(<<?t, r::binary>>, acc), do: string(r, [acc, ?\t])

  defp escape(<<?u, h::binary-4, r::binary>> = bin, acc) do
    cp = hex4(h, bin)

    cond do
      cp in 0xD800..0xDBFF ->
        case r do
          <<?\\, ?u, l::binary-4, r2::binary>> ->
            lo = hex4(l, r)
            unless lo in 0xDC00..0xDFFF, do: throw({:error_at, r, "unpaired high surrogate"})
            string(r2, [acc, <<0x10000 + (cp - 0xD800) * 0x400 + (lo - 0xDC00)::utf8>>])

          _ ->
            throw({:error_at, r, "unpaired high surrogate"})
        end

      cp in 0xDC00..0xDFFF ->
        throw({:error_at, bin, "unpaired low surrogate"})

      true ->
        string(r, [acc, <<cp::utf8>>])
    end
  end

  defp escape(bin, _acc), do: throw({:error_at, bin, "invalid escape"})

  defp hex4(h, at) do
    case Integer.parse(h, 16) do
      {v, ""} -> v
      _ -> throw({:error_at, at, "invalid \\u escape"})
    end
  end

  # plain integers (the bulk of a vocabulary) without the general scanner
  defp number(<<c, _::binary>> = bin) when c in ?1..?9 do
    case int_run(bin, 0) do
      {n, <<d, _::binary>>} when d in [?., ?e, ?E] -> _ = n; general(bin)
      {n, rest} -> {n, rest}
    end
  end

  defp number(bin), do: general(bin)

  defp int_run(<<c, rest::binary>>, acc) when c in ?0..?9, do: int_run(rest, acc * 10 + (c - ?0))
  defp int_run(rest, acc), do: {acc, rest}

  defp general(bin) do
    {int_part, rest} = take(bin, &(&1 == ?- or &1 in ?0..?9))
    {frac, rest} = if match?(<<?., _::binary>>, rest), do: take_frac(rest), else: {"", rest}
    {exp, rest} = if match?(<<e, _::binary>> when e in [?e, ?E], rest), do: take_exp(rest), else: {"", rest}

    valid_int = Regex.match?(~r/\A-?(0|[1-9][0-9]*)\z/, int_part)
    unless valid_int and (frac == "" or byte_size(frac) > 1), do: throw({:error_at, bin, "invalid number"})

    if frac == "" and exp == "" do
      {String.to_integer(int_part), rest}
    else
      mant = if frac == "", do: int_part <> ".0", else: int_part <> frac

      try do
        {String.to_float(mant <> exp), rest}
      rescue
        ArgumentError -> throw({:error_at, bin, "number out of binary64 range"})
      end
    end
  end

  defp take(bin, pred), do: take(bin, pred, 0)
  defp take(bin, pred, n) do
    case bin do
      <<_::binary-size(n), c, _::binary>> -> if pred.(c), do: take(bin, pred, n + 1), else: split(bin, n)
      _ -> split(bin, n)
    end
  end

  defp split(bin, n), do: {binary_part(bin, 0, n), binary_part(bin, n, byte_size(bin) - n)}

  defp take_frac(<<?., rest::binary>>) do
    {digits, rest} = take(rest, &(&1 in ?0..?9))
    {"." <> digits, rest}
  end

  defp take_exp(<<e, rest::binary>>) when e in [?e, ?E] do
    {sign, rest} = case rest do
      <<s, r::binary>> when s in [?+, ?-] -> {<<s>>, r}
      _ -> {"", rest}
    end

    {digits, rest} = take(rest, &(&1 in ?0..?9))
    if digits == "", do: throw({:error_at, rest, "invalid exponent"})
    {"e" <> sign <> digits, rest}
  end

  # ------------------------------------------------------------- encoding --

  @doc "Encode to a binary. Map keys may be strings or atoms; keys are emitted sorted (deterministic); `{:dict, pairs}` keeps its order."
  @spec encode(term) :: binary
  def encode(term), do: term |> enc() |> IO.iodata_to_binary()

  defp enc(nil), do: "null"
  defp enc(true), do: "true"
  defp enc(false), do: "false"
  defp enc(n) when is_integer(n), do: Integer.to_string(n)
  defp enc(f) when is_float(f), do: :erlang.float_to_binary(f, [:short])
  defp enc(a) when is_atom(a), do: enc_string(Atom.to_string(a))
  defp enc(s) when is_binary(s), do: enc_string(s)
  defp enc(l) when is_list(l), do: [?[, Enum.map_intersperse(l, ?,, &enc/1), ?]]
  defp enc({:dict, pairs}) when is_list(pairs), do: [?{, Enum.map_intersperse(pairs, ?,, fn {k, v} -> [enc_string(to_string(k)), ?:, enc(v)] end), ?}]

  defp enc(%{} = m) do
    pairs = m |> Enum.map(fn {k, v} -> {to_string(k), v} end) |> Enum.sort()
    [?{, Enum.map_intersperse(pairs, ?,, fn {k, v} -> [enc_string(k), ?:, enc(v)] end), ?}]
  end

  defp enc_string(s), do: [?", escape_out(s, []), ?"]

  defp escape_out(<<>>, acc), do: Enum.reverse(acc)
  defp escape_out(<<?", r::binary>>, acc), do: escape_out(r, ["\\\"" | acc])
  defp escape_out(<<?\\, r::binary>>, acc), do: escape_out(r, ["\\\\" | acc])
  defp escape_out(<<?\n, r::binary>>, acc), do: escape_out(r, ["\\n" | acc])
  defp escape_out(<<?\r, r::binary>>, acc), do: escape_out(r, ["\\r" | acc])
  defp escape_out(<<?\t, r::binary>>, acc), do: escape_out(r, ["\\t" | acc])

  defp escape_out(<<c, r::binary>>, acc) when c < 0x20,
    do: escape_out(r, ["\\u" <> String.pad_leading(Integer.to_string(c, 16), 4, "0") | acc])

  defp escape_out(<<c, r::binary>>, acc), do: escape_out(r, [c | acc])
end
