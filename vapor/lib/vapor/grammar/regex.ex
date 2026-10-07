defmodule Vapor.Grammar.Regex do
  @moduledoc """
  JSON Schema `pattern` (an ECMA-262 regular expression) and `format`,
  compiled to the byte grammar of `Vapor.Grammar` — so a constrained
  decoder emits *only* strings that match, by construction, instead of
  checking after the fact and retrying.

  Three layers, each exact:

    1. **Parse** the ECMA-262 subset that schemas use: literals and escapes
       (`\\d \\w \\s` and their negations, `\\xHH`, `\\uHHHH`, `\\t \\n …`),
       classes with ranges and negation, `.`, groups (capturing,
       `(?:…)`, named), `|`, the quantifiers `* + ? {n} {n,} {n,m}` (lazy
       forms denote the same language), and `^`/`$` at the ends of the
       top-level alternatives. A `pattern` is **not anchored** (it matches
       when found anywhere): an end without its anchor gets `[\\s\\S]*`.
       Look-around, back-references, `\\b`, `\\p{…}` and the like are a
       rejection naming the construct.
    2. **Characters to bytes**: a set of code points becomes the UTF-8 byte
       sequences that encode it (the range-splitting construction of RE2 /
       `utf8-ranges`), surrogates excluded — every accepted string is valid
       UTF-8.
    3. **Values to JSON**: inside a JSON string a character that must be
       escaped (`"`, `\\`, controls) is admitted only as its escape (`\\"`,
       `\\\\`, `\\n`, `\\u001f` …); every other character only raw. The
       decoded value then matches the pattern — checked against Python's
       `re` on sampled members and random strings
       (`test/vapor/regex_test.exs`).

  `format` is compiled to a pattern: `date`, `time`, `date-time` (RFC 3339,
  with the real calendar — no 30 February, 29 February only in leap years),
  `uuid`, `ipv4` (no leading zeros), and — as *subsets* that never emit an
  invalid value — `email` (dot-atom local part, DNS domain) and `hostname`
  (at most four labels of 62 characters). Other formats are a rejection.
  """
  import Bitwise
  alias Vapor.Rejection

  @max 0x10FFFF
  @any [{0, 0xD7FF}, {0xE000, @max}]
  # ECMA-262 WhiteSpace and LineTerminator
  @space [{0x09, 0x0D}, {0x20, 0x20}, {0xA0, 0xA0}, {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029},
          {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000}, {0xFEFF, 0xFEFF}]
  @digit [{?0, ?9}]
  @word [{?0, ?9}, {?A, ?Z}, {?_, ?_}, {?a, ?z}]
  @dot_excl [{0x0A, 0x0A}, {0x0D, 0x0D}, {0x2028, 0x2029}]

  @doc """
  The grammar of a JSON string (quotes included) whose value matches
  `pattern`: `{:ok, ir}` or `{:error, %Rejection{}}`.
  """
  def json_string(pattern) when is_binary(pattern) do
    with {:ok, ast} <- parse(pattern) do
      {:ok, {:seq, [{:lit, "\""}, to_ir(ast), {:lit, "\""}]}}
    end
  end

  @doc "The same for a JSON Schema `format`."
  def json_format(format) do
    case formats()[format] do
      nil -> {:error, Rejection.new({:json_schema, "format"}, "a format the grammar can enforce (#{format} is not; known: #{Enum.join(Map.keys(formats()) |> Enum.sort(), ", ")})", "drop it or pass lenient: true")}
      p -> json_string(p)
    end
  end

  @doc "The patterns behind the supported formats."
  def formats do
    month31 = "(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])"
    month30 = "(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)"
    feb = "02-(?:0[1-9]|1[0-9]|2[0-8])"
    # 29 February: years divisible by 4, except centuries not divisible by 400
    leap = "(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)-02-29"
    date = "(?:[0-9]{4}-(?:#{month31}|#{month30}|#{feb})|#{leap})"
    time = "(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
    atext = "[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]"
    label = "[A-Za-z0-9](?:[A-Za-z0-9-]{0,60}[A-Za-z0-9])?"
    octet = "(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"

    %{
      "date" => "^#{date}$",
      "time" => "^#{time}$",
      "date-time" => "^#{date}[Tt]#{time}$",
      "uuid" => "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
      "ipv4" => "^(?:#{octet}\\.){3}#{octet}$",
      "email" => "^#{atext}{1,32}(?:\\.#{atext}{1,32}){0,3}@#{label}(?:\\.#{label}){1,3}$",
      "hostname" => "^#{label}(?:\\.#{label}){0,3}$"
    }
  end

  # ---------------------------------------------------------------- parse --

  @doc """
  Parse a pattern into `{:ok, ast}` — `{:set, ranges}`, `{:cat, [ast]}`,
  `{:alt, [ast]}`, `{:rep, ast, min, max | :inf}` — with the unanchored
  ends already widened; or a rejection.
  """
  def parse(pattern) do
    cps = String.to_charlist(pattern)

    try do
      {alts, rest} = alternation(cps, 0)
      if rest != [], do: throw({:bad, "an unbalanced \")\""})
      alts = Enum.map(alts, &anchor/1)
      {:ok, simplify({:alt, alts})}
    catch
      {:bad, what} -> {:error, Rejection.new({:json_schema, "pattern"}, "a regular expression the grammar supports (#{what})", "simplify the pattern or pass lenient: true")}
    end
  rescue
    _ in [ArgumentError, UnicodeConversionError] -> {:error, Rejection.new({:json_schema, "pattern"}, "a valid UTF-8 pattern", "check the pattern")}
  end

  # a top-level alternative: its anchors, or [\s\S]* where one is missing
  defp anchor(items) do
    {start?, items} = case items do [:bol | r] -> {true, r}; r -> {false, r} end
    {end?, items} = case Enum.reverse(items) do [:eol | r] -> {true, Enum.reverse(r)}; _ -> {false, items} end
    if Enum.any?(items, &(&1 in [:bol, :eol])), do: throw({:bad, "^ or $ inside the pattern (only at its ends)"})
    anything = {:rep, {:set, @any}, 0, :inf}
    {:cat, (if start?, do: [], else: [anything]) ++ items ++ (if end?, do: [], else: [anything])}
  end

  # alternatives of a sequence of items, up to ")" or the end
  defp alternation(cps, depth) do
    {items, rest} = sequence(cps, [], depth)

    case rest do
      [?| | more] ->
        {alts, rest} = alternation(more, depth)
        {[items | alts], rest}

      _ ->
        {[items], rest}
    end
  end

  defp sequence([], acc, _d), do: {Enum.reverse(acc), []}
  defp sequence([c | _] = cps, acc, _d) when c in [?|, ?)], do: {Enum.reverse(acc), cps}

  defp sequence(cps, acc, depth) do
    {atom, rest} = atom(cps, depth)

    case atom do
      a when a in [:bol, :eol] -> sequence(rest, [a | acc], depth)
      a ->
        {a, rest} = quantified(a, rest)
        sequence(rest, [a | acc], depth)
    end
  end

  defp atom([?^ | rest], _d), do: {:bol, rest}
  defp atom([?$ | rest], _d), do: {:eol, rest}
  defp atom([?. | rest], _d), do: {{:set, minus(@any, @dot_excl)}, rest}
  defp atom([?[ | rest], _d), do: class(rest)
  defp atom([?\\ | rest], _d), do: escape(rest, :atom)

  defp atom([?( | rest], depth) do
    rest =
      case rest do
        [??, ?: | r] -> r
        [??, ?<, c | r] when c not in [?=, ?!] -> skip_name([c | r])
        [?? | _] -> throw({:bad, "look-around or a group modifier"})
        r -> r
      end

    case alternation(rest, depth + 1) do
      {alts, [?) | r]} -> {{:alt, Enum.map(alts, &{:cat, &1})}, r}
      _ -> throw({:bad, "an unclosed \"(\""})
    end
  end

  defp atom([c | _], _d) when c in [?*, ?+, ??], do: throw({:bad, "a quantifier with nothing to repeat"})
  defp atom([?{ | _] = cps, _d), do: if(quantifier(cps), do: throw({:bad, "a quantifier with nothing to repeat"}), else: {{:set, [{?{, ?{}]}, tl(cps)})
  defp atom([c | rest], _d), do: {{:set, [{c, c}]}, rest}

  defp skip_name(cps) do
    case Enum.split_while(cps, &(&1 != ?>)) do
      {_, [?> | r]} -> r
      _ -> throw({:bad, "an unclosed group name"})
    end
  end

  defp quantified(a, rest) do
    {q, rest} =
      case rest do
        [?* | r] -> {{0, :inf}, r}
        [?+ | r] -> {{1, :inf}, r}
        [?? | r] -> {{0, 1}, r}
        [?{ | _] = r -> (case quantifier(r) do nil -> {nil, r}; {mn, mx, r2} -> {{mn, mx}, r2} end)
        r -> {nil, r}
      end

    case q do
      nil -> {a, rest}
      {mn, mx} ->
        if mx != :inf and mx < mn, do: throw({:bad, "a quantifier {n,m} with m < n"})
        if mx != :inf and mx > 10_000, do: throw({:bad, "a quantifier above 10000"})
        if mx == :inf and nullable?(a), do: throw({:bad, "an unbounded repetition of something that can be empty"})
        # a lazy quantifier denotes the same language
        rest = case rest do [?? | r] -> r; r -> r end
        if match?([c | _] when c in [?*, ?+, ??], rest), do: throw({:bad, "a quantifier after a quantifier"})
        {{:rep, a, mn, mx}, rest}
    end
  end

  # {n}, {n,}, {n,m}: {min, max, rest} or nil (then "{" is a literal, as ECMA's Annex B)
  defp quantifier([?{ | r]) do
    {a, r} = Enum.split_while(r, &(&1 in ?0..?9))

    cond do
      a == [] -> nil
      match?([?} | _], r) -> n = List.to_integer(a); {n, n, tl(r)}
      match?([?,, ?} | _], r) -> {List.to_integer(a), :inf, Enum.drop(r, 2)}
      match?([?, | _], r) ->
        {b, r2} = Enum.split_while(tl(r), &(&1 in ?0..?9))
        if b != [] and match?([?} | _], r2), do: {List.to_integer(a), List.to_integer(b), tl(r2)}, else: nil
      true -> nil
    end
  end

  defp class(cps) do
    {neg, cps} = case cps do [?^ | r] -> {true, r}; r -> {false, r} end
    {ranges, rest} = class_items(cps, [])
    set = norm(ranges)
    {{:set, if(neg, do: minus(@any, set), else: set)}, rest}
  end

  defp class_items([?] | rest], acc), do: {acc, rest}
  defp class_items([], _acc), do: throw({:bad, "an unclosed \"[\""})

  defp class_items(cps, acc) do
    {lo, rest} = class_atom(cps)

    case {lo, rest} do
      {{:one, a}, [?-, c | r2]} when c != ?] ->
        case class_atom([c | r2]) do
          {{:one, b}, r3} ->
            if b < a, do: throw({:bad, "a class range out of order"})
            class_items(r3, [{a, b} | acc])

          {{:set, _}, _} ->
            throw({:bad, "a class range with a class escape at an end"})
        end

      {{:one, a}, r} -> class_items(r, [{a, a} | acc])
      {{:set, s}, r} -> class_items(r, s ++ acc)
    end
  end

  defp class_atom([?\\ | rest]) do
    case escape(rest, :class) do
      {{:set, [{a, a}]}, r} -> {{:one, a}, r}
      {{:set, s}, r} -> {{:set, s}, r}
    end
  end

  defp class_atom([c | rest]), do: {{:one, c}, rest}

  defp escape([], _), do: throw({:bad, "a pattern ending in \\"})

  defp escape([c | rest], ctx) do
    case c do
      ?d -> {{:set, @digit}, rest}
      ?D -> {{:set, minus(@any, @digit)}, rest}
      ?w -> {{:set, @word}, rest}
      ?W -> {{:set, minus(@any, @word)}, rest}
      ?s -> {{:set, @space}, rest}
      ?S -> {{:set, minus(@any, @space)}, rest}
      ?t -> one(9, rest)
      ?n -> one(10, rest)
      ?v -> one(11, rest)
      ?f -> one(12, rest)
      ?r -> one(13, rest)
      ?0 -> if match?([d | _] when d in ?0..?9, rest), do: throw({:bad, "an octal escape"}), else: one(0, rest)
      ?b when ctx == :class -> one(8, rest)
      ?b -> throw({:bad, "\\b (a word boundary)"})
      ?B -> throw({:bad, "\\B (a word boundary)"})
      ?x -> hexes(rest, 2)
      ?u -> if match?([?{ | _], rest), do: throw({:bad, "\\u{…} (the u flag)"}), else: hexes(rest, 4)
      ?c ->
        case rest do
          [l | r] when l in ?a..?z or l in ?A..?Z -> one(rem(l, 32), r)
          _ -> throw({:bad, "a \\c escape without a letter"})
        end
      d when d in ?1..?9 -> throw({:bad, "a back-reference"})
      p when p in [?p, ?P] -> throw({:bad, "\\p{…} (Unicode properties)"})
      ?k -> throw({:bad, "a named back-reference"})
      other -> one(other, rest)
    end
  end

  defp one(c, rest), do: {{:set, [{c, c}]}, rest}

  defp hexes(cps, n) do
    {h, rest} = Enum.split(cps, n)

    if length(h) == n and Enum.all?(h, &(&1 in ?0..?9 or &1 in ?a..?f or &1 in ?A..?F)) do
      v = List.to_integer(h, 16)
      if v in 0xD800..0xDFFF, do: throw({:bad, "an escaped surrogate"})
      one(v, rest)
    else
      throw({:bad, "a malformed \\x or \\u escape"})
    end
  end

  defp nullable?({:set, _}), do: false
  defp nullable?({:cat, xs}), do: Enum.all?(xs, &nullable?/1)
  defp nullable?({:alt, xs}), do: Enum.any?(xs, &nullable?/1)
  defp nullable?({:rep, a, mn, _}), do: mn == 0 or nullable?(a)

  defp simplify({:alt, [x]}), do: simplify(x)
  defp simplify({:alt, xs}), do: {:alt, Enum.map(xs, &simplify/1)}
  defp simplify({:cat, [x]}), do: simplify(x)
  defp simplify({:cat, xs}), do: {:cat, Enum.map(xs, &simplify/1)}
  defp simplify({:rep, a, mn, mx}), do: {:rep, simplify(a), mn, mx}
  defp simplify(x), do: x

  # ------------------------------------------------------------ code sets --

  @doc false
  def norm(ranges) do
    ranges
    |> Enum.sort()
    |> Enum.reduce([], fn
      {a, b}, [{c, d} | acc] when a <= d + 1 -> [{c, max(b, d)} | acc]
      r, acc -> [r | acc]
    end)
    |> Enum.reverse()
  end

  # the complement of `set` within `universe`
  defp minus(universe, set) do
    set = norm(set)

    Enum.flat_map(universe, fn {lo, hi} ->
      {out, cur} =
        Enum.reduce(set, {[], lo}, fn {a, b}, {out, cur} ->
          cond do
            b < cur or a > hi -> {out, cur}
            a > cur -> {[{cur, min(a - 1, hi)} | out], b + 1}
            true -> {out, max(cur, b + 1)}
          end
        end)

      out = if cur <= hi, do: [{cur, hi} | out], else: out
      Enum.reverse(out)
    end)
  end

  # ------------------------------------------------------- to the grammar --

  @doc false
  def to_ir({:set, ranges}), do: set_ir(ranges)
  def to_ir({:cat, xs}), do: {:seq, Enum.map(xs, &to_ir/1)}
  def to_ir({:alt, xs}), do: {:alt, Enum.map(xs, &to_ir/1)}
  def to_ir({:rep, a, mn, mx}), do: {:rep, to_ir(a), mn, mx}

  @json_escapes %{0x22 => "\\\"", 0x5C => "\\\\", 0x08 => "\\b", 0x0C => "\\f", 0x0A => "\\n", 0x0D => "\\r", 0x09 => "\\t"}

  # characters JSON must escape → their one escape; the rest → UTF-8 byte classes
  defp set_ir(ranges) do
    ranges = ranges |> norm() |> intersect(@any)
    escaped = for {a, b} <- intersect(ranges, [{0, 0x1F}, {0x22, 0x22}, {0x5C, 0x5C}]), c <- a..b, do: {:lit, json_escape(c)}
    raw = minus(ranges, [{0, 0x1F}, {0x22, 0x22}, {0x5C, 0x5C}])
    seqs = Enum.flat_map(raw, fn {a, b} -> utf8_split(a, b) end)
    seq_irs = Enum.map(seqs, fn bytes -> {:seq, Enum.map(bytes, fn {x, y} -> {:class, [x..y]} end)} end)

    case escaped ++ seq_irs do
      [] -> {:alt, []}
      [one] -> one
      many -> {:alt, many}
    end
  end

  defp json_escape(c), do: Map.get(@json_escapes, c) || "\\u00" <> String.downcase(Base.encode16(<<c>>))

  defp intersect(a, b), do: minus(a, minus(@any ++ [{0xD800, 0xDFFF}], b))

  @doc """
  The UTF-8 encodings of the code points `lo..hi` as a list of byte-range
  sequences (`[[{lo, hi}, …]]`): each sequence is a product of ranges, and
  together they encode exactly the given code points.
  """
  def utf8_split(lo, hi) do
    cond do
      lo > hi -> []
      true ->
        case Enum.find([0x7F, 0x7FF, 0xFFFF], &(lo <= &1 and hi > &1)) do
          nil -> split_same(lo, hi)
          b -> utf8_split(lo, b) ++ utf8_split(b + 1, hi)
        end
    end
  end

  defp split_same(lo, hi) do
    n = byte_size(<<lo::utf8>>)

    found =
      Enum.find_value(1..(n - 1)//1, fn i ->
        m = (1 <<< (6 * i)) - 1

        cond do
          band(lo, bnot(m)) == band(hi, bnot(m)) -> nil
          band(lo, m) != 0 -> split_same(lo, bor(lo, m)) ++ utf8_split(bor(lo, m) + 1, hi)
          band(hi, m) != m -> utf8_split(lo, band(hi, bnot(m)) - 1) ++ split_same(band(hi, bnot(m)), hi)
          true -> nil
        end
      end)

    found || [Enum.zip(:binary.bin_to_list(<<lo::utf8>>), :binary.bin_to_list(<<hi::utf8>>))]
  end
end
