defmodule Vapor.RegexTest do
  @moduledoc """
  JSON Schema `pattern` and `format` as byte grammars
  (`Vapor.Grammar.Regex`), checked for **equality of languages** against
  Python: strings drawn from each pattern (so the members are not only the
  easy ones), their mutations and random strings are judged by the grammar
  (over their JSON spelling) and by `re.search` / the standard library's
  parsers; the verdicts must agree on every string.
  """
  use ExUnit.Case, async: true
  alias Vapor.Grammar
  alias Vapor.Grammar.{JSONSchema, Regex}
  import Vapor.TestHelpers

  @patterns [
    "^[a-z]+$", "^\\d{5}(-\\d{4})?$", "^[A-Z]{2}-\\d{3,5}$", "abc", "^(\\+55 )?\\(\\d{2}\\) \\d{4,5}-\\d{4}$",
    "^[^@\\s]+@[^@\\s]+\\.[a-z]{2,}$", "^\\w+(\\.\\w+)*$", "colou?r", "^(?:true|false|maybe)$", "^[\\u00C0-\\u017F]+$",
    "^\"[a-z]+\\\\[a-z]+\"$", "^.{3}$", "^[\\s\\S]{0,4}$", "^\\t\\n?x$", "^[^a-z]*$", "^(ab|cd){2,3}$", "^x{2,}y$",
    "^\\x41\\u00e9$", "[0-9]$", "^[.$^*+?(){}|\\[\\]\\\\/-]+$", "^😀+ok$", "^a{0}b$", "^(a|b)*c$", "^[\\d\\-]{4,}$",
    "^(?<year>\\d{4})-(?<n>\\d+)$", "^\\S+\\s\\S+$", "^\\D\\0?$", "^[^\\u0000-\\u007f]{2}$", "^(foo|foobar)(bar)?$"
  ]

  # ---------------------------------------------------------- generation --

  # a random member of a parsed pattern (ASCII preferred, everything possible)
  defp gen({:set, ranges}) do
    ascii = Regex.norm(for {a, b} <- ranges, a <= 0x7E, do: {a, min(b, 0x7E)})
    pool = if ascii != [] and :rand.uniform() < 0.75, do: ascii, else: ranges

    case pool do
      [] -> throw(:empty)
      _ ->
        {a, b} = Enum.random(pool)
        [a + :rand.uniform(b - a + 1) - 1]
    end
  end

  defp gen({:cat, xs}), do: Enum.flat_map(xs, &gen/1)
  defp gen({:alt, xs}), do: xs |> Enum.shuffle() |> Enum.find_value(fn x -> try do gen(x) catch :empty -> nil end end) || throw(:empty)
  defp gen({:rep, a, mn, mx}), do: Enum.flat_map(1..(mn + :rand.uniform((if mx == :inf, do: mn + 3, else: mx) - mn + 1) - 1)//1, fn _ -> gen(a) end)

  defp mutate(cps) do
    pick = fn -> Enum.random([?a, ?Z, ?0, ?9, ?-, ?., ?@, ?\s, ?", ?\\, ?\n, 0xE9, 0x1F600, ?x, ?/]) end
    i = :rand.uniform(length(cps) + 1) - 1

    case :rand.uniform(3) do
      1 -> List.insert_at(cps, i, pick.())
      2 when cps != [] -> List.delete_at(cps, min(i, length(cps) - 1))
      _ when cps != [] -> List.replace_at(cps, min(i, length(cps) - 1), pick.())
      _ -> [pick.()]
    end
  end

  defp str(cps), do: List.to_string(cps)

  # the one JSON spelling the grammar admits: escapes for " \\ and controls, everything else raw
  defp json(s) do
    body =
      for <<c::utf8 <- s>>, into: "" do
        case c do
          ?" -> "\\\""
          ?\\ -> "\\\\"
          ?\n -> "\\n"
          ?\r -> "\\r"
          ?\t -> "\\t"
          8 -> "\\b"
          12 -> "\\f"
          c when c < 0x20 -> "\\u00" <> String.downcase(Base.encode16(<<c>>))
          c -> <<c::utf8>>
        end
      end

    "\"" <> body <> "\""
  end

  defp accepts?(ir, s) do
    case Grammar.advance(Grammar.new(ir), json(s)) do
      {:ok, g} -> Grammar.complete?(g)
      :reject -> false
    end
  end

  defp strings(ast, n) do
    members = for _ <- 1..n, m = (try do gen(ast) catch :empty -> nil end), m != nil, do: m
    near = for m <- members, k <- 1..2, do: Enum.reduce(1..k, m, fn _, acc -> mutate(acc) end)
    random = for _ <- 1..div(n, 2), do: mutate(mutate([]))
    {Enum.map(members, &str/1), Enum.map(near ++ random, &str/1)}
  end

  test "drawn members are accepted, and their JSON spelling is valid JSON with the same value" do
    :rand.seed(:exsss, {1, 2, 3})

    for p <- @patterns do
      {:ok, ast} = Regex.parse(p)
      {:ok, ir} = Regex.json_string(p)
      {members, _} = strings(ast, 30)

      for m <- members do
        assert accepts?(ir, m), "#{p}: #{inspect(m)}"
        assert {:ok, ^m} = Vapor.JSON.decode(json(m))
      end
    end
  end

  test "ECMA corners Python spells differently: [] matches nothing, [^] anything, {,3} and \\cJ" do
    {:ok, empty} = Regex.json_string("^[]?a$")
    assert accepts?(empty, "a") and not accepts?(empty, "]a")
    {:ok, any} = Regex.json_string("^[^]b$")
    assert accepts?(any, "\nb") and accepts?(any, "éb") and not accepts?(any, "b")
    {:ok, lit} = Regex.json_string("^x{,3}$")
    assert accepts?(lit, "x{,3}") and not accepts?(lit, "xx")
    {:ok, c} = Regex.json_string("^\\cJ[\\W\\d]$")
    assert accepts?(c, "\n7") and accepts?(c, "\n-") and not accepts?(c, "\na")
  end

  test "refused with the construct named: look-around, back-references, \\b, \\p, inner anchors, empty loops" do
    for {p, what} <- [{"(?=a)b", "look-around"}, {"(a)\\1", "back-reference"}, {"\\bword", "word boundary"},
                      {"\\p{L}+", "Unicode properties"}, {"a^b", "inside the pattern"}, {"(a*)*", "can be empty"},
                      {"a{3,2}", "m < n"}, {"[z-a]", "out of order"}, {"(ab", "unclosed"}, {"ab)", "unbalanced"},
                      {"*a", "nothing to repeat"}, {"\\u{1F600}", "u flag"}] do
      assert {:error, %Vapor.Rejection{bound: b}} = Regex.parse(p), p
      assert b =~ what, "#{p}: #{b}"
    end
  end

  test "UTF-8 splitting: every code point's encoding is covered, by exactly one sequence" do
    for {lo, hi} <- [{0, 0x7F}, {0x80, 0x10FFFF}, {0x7F0, 0x810}, {0xFFF0, 0x10010}, {0xE000, 0xE0FF}, {0x1F600, 0x1F64F}] do
      seqs = Regex.utf8_split(lo, hi)
      covers = fn cp -> Enum.count(seqs, fn s -> b = :binary.bin_to_list(<<cp::utf8>>); length(b) == length(s) and Enum.all?(Enum.zip(b, s), fn {x, {a, z}} -> x in a..z end) end) end

      for cp <- Enum.take_random(lo..hi, 300) ++ [lo, hi], cp not in 0xD800..0xDFFF, do: assert(covers.(cp) == 1, "#{inspect({lo, hi})}: #{cp}")
    end
  end

  test "in a JSON Schema: pattern and format constrain the string; conflicts are refused, lenient keeps the pattern" do
    {:ok, g} = JSONSchema.compile(%{"type" => "object", "properties" => %{"cep" => %{"type" => "string", "pattern" => "^\\d{5}-\\d{3}$"},
                                    "quando" => %{"format" => "date"}}, "required" => ["cep", "quando"]})
    ok = fn bytes -> match?({:ok, g2} when is_struct(g2), Grammar.advance(g, bytes)) and Grammar.complete?(elem(Grammar.advance(g, bytes), 1)) end
    assert ok.(~s({"cep": "01310-100", "quando": "2024-02-29"}))
    refute ok.(~s({"cep": "01310100", "quando": "2024-02-29"}))
    refute ok.(~s({"cep": "01310-100", "quando": "2023-02-29"}))

    assert {:error, %Vapor.Rejection{node: {:json_schema, "format"}}} = JSONSchema.compile(%{"type" => "string", "pattern" => "a", "format" => "date"})
    assert {:error, %Vapor.Rejection{node: {:json_schema, "pattern"}}} = JSONSchema.compile(%{"type" => "string", "pattern" => "a", "maxLength" => 3})
    assert {:error, %Vapor.Rejection{node: {:json_schema, "format"}}} = JSONSchema.compile(%{"type" => "string", "format" => "uri"})
    assert {:error, %Vapor.Rejection{node: {:json_schema, "format"}}} = JSONSchema.compile(%{"type" => "integer", "format" => "int64"})
    assert {:ok, _} = JSONSchema.compile(%{"type" => "string", "pattern" => "^a+$", "maxLength" => 3}, lenient: true)
    assert {:ok, _} = JSONSchema.compile(%{"type" => "string", "format" => "uri"}, lenient: true)
  end

  @tag :python
  @tag timeout: 600_000
  test "the same language as Python's re (ECMA semantics) and as the standard library's format parsers" do
    if python?(["json"]) do
      :rand.seed(:exsss, {4, 5, 6})

      pcases =
        for p <- @patterns do
          {:ok, ast} = Regex.parse(p)
          {members, others} = strings(ast, 40)
          %{pattern: p, strings: members ++ others}
        end

      fcases =
        for f <- Map.keys(Regex.formats()) do
          {:ok, ast} = Regex.parse(Regex.formats()[f])
          {members, others} = strings(ast, 120)
          extra = if f == "date", do: for(y <- [1900, 2000, 2023, 2024, 2100], m <- 1..12, d <- 28..31, do: :io_lib.format("~4..0B-~2..0B-~2..0B", [y, m, d]) |> to_string()), else: []
          %{format: f, strings: members ++ others ++ extra}
        end

      script = File.read!(Path.expand("../python/regex_oracle.py", __DIR__))
      {:ok, out} = Vapor.JSON.decode(py!(script, [], Vapor.JSON.encode(%{patterns: pcases, formats: fcases})))

      total =
        for {c, verdicts} <- Enum.zip(pcases, out["patterns"]), reduce: 0 do
          n ->
            {:ok, ir} = Regex.json_string(c.pattern)

            for {s, want} <- Enum.zip(c.strings, verdicts) do
              assert accepts?(ir, s) == want, "pattern #{inspect(c.pattern)} on #{inspect(s)}: Python says #{want}"
            end

            n + length(c.strings)
        end

      ftotal =
        for {c, verdicts} <- Enum.zip(fcases, out["formats"]), reduce: 0 do
          n ->
            {:ok, ir} = Regex.json_format(c.format)

            for {s, want} <- Enum.zip(c.strings, verdicts) do
              got = accepts?(ir, s)
              # email and hostname are subsets: everything emitted is valid; not everything valid is emitted
              if c.format in ["email", "hostname"],
                do: assert(not got or want, "format #{c.format} emitted the invalid #{inspect(s)}"),
                else: assert(got == want, "format #{c.format} on #{inspect(s)}: Python says #{want}")
            end

            n + length(c.strings)
        end

      IO.puts("\n  regex: #{total} strings over #{length(pcases)} patterns, #{ftotal} over #{length(fcases)} formats — same verdicts as Python")
    end
  end
end
