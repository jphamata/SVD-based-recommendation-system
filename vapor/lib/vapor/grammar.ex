defmodule Vapor.Grammar do
  @moduledoc """
  Byte-level grammars for constrained decoding — the output of a model made
  to *be* a member of a language (a JSON Schema, a tool call), not checked
  after the fact.

  A grammar is a term of a small IR over bytes:

      {:lit, bytes}                 the exact bytes
      {:seq, [g]}  {:alt, [g]}      sequence, alternation
      {:rep, g, min, max}           g repeated min..max times (max :inf); g must not match ""
      {:class, [lo..hi]}            one byte in the ranges
      {:str, %{min:, max:}}         a JSON string literal, quotes included (valid UTF-8,
                                    escapes checked, length counted in characters)
      {:num, :number | :integer}    a JSON number
      {:ref, name}                  a named rule (recursion; never left-recursive)
      {:substr, key}                any non-empty substring of the text whose suffix
                                    automaton (`suffix_automaton/1`) is `defs[key]`

  plus the rule table `defs`. Matching keeps a *set of configurations*;
  a configuration is a stack of pending items (a Thompson-style simulation
  of the pushdown automaton), so alternatives and optional members cost no
  backtracking: `advance/2` consumes one byte in every configuration at once,
  and the set is empty exactly when the bytes so far are no prefix of any
  member of the language. A configuration with an empty stack is a complete
  member (`complete?/1`); `dead?/1` says no byte can follow.

  Token-level masks are computed by walking the vocabulary trie
  (`Vapor.Grammar.Vocab`) with this matcher.
  """

  defstruct root: nil, defs: %{}, configs: nil

  @type t :: %__MODULE__{}

  @ws {:rep, {:class, [?\s..?\s, ?\t..?\t, ?\n..?\n, ?\r..?\r]}, 0, 16}

  @doc "A matcher at the start of the grammar `root` with rule table `defs`."
  def new(root, defs \\ %{}) do
    g = %__MODULE__{root: root, defs: Map.merge(builtin_defs(), defs)}
    %{g | configs: g.defs |> expand_all([[root]]) |> MapSet.new()}
  end

  @doc "JSON whitespace (bounded, so whitespace cannot run forever)."
  def ws, do: @ws

  @doc "Advance over bytes; `{:ok, g}` or `:reject`."
  def advance(%__MODULE__{} = g, bytes) when is_binary(bytes) do
    case step_bytes(g.configs, bytes, g.defs) do
      [] -> :reject
      cs -> {:ok, %{g | configs: MapSet.new(cs)}}
    end
  end

  @doc false
  # one byte on a list of configurations → deduplicated configurations
  def step_configs(configs, byte, defs), do: configs |> Enum.flat_map(&step(&1, byte, defs)) |> Enum.uniq()

  defp step_bytes(configs, <<>>, _defs), do: Enum.to_list(configs)

  defp step_bytes(configs, <<b, rest::binary>>, defs) do
    case step_configs(configs, b, defs) do
      [] -> []
      cs -> step_bytes(cs, rest, defs)
    end
  end

  @doc "Whether the bytes so far are a complete member."
  def complete?(%__MODULE__{configs: cs}), do: MapSet.member?(cs, [])

  @doc "Whether no byte can follow (complete with nothing pending)."
  def dead?(%__MODULE__{configs: cs}), do: Enum.all?(cs, &(&1 == []))

  # --------------------------------------------------------- expansion --

  @doc false
  def expand_all(defs, stacks), do: stacks |> Enum.flat_map(&expand(&1, defs)) |> Enum.uniq()

  # ε-closure: stacks whose top consumes a byte, or the empty stack
  defp expand([], _defs), do: [[]]
  defp expand([{:seq, gs} | rest], defs), do: expand(gs ++ rest, defs)
  defp expand([{:alt, gs} | rest], defs), do: Enum.flat_map(gs, &expand([&1 | rest], defs))
  defp expand([{:rep, g, min, max} | rest], defs), do: expand([{:rep_at, g, 0, min, max} | rest], defs)

  defp expand([{:rep_at, g, n, min, max} | rest], defs) do
    stop = if n >= min, do: expand(rest, defs), else: []
    more = if max == :inf or n < max, do: expand([g, {:rep_at, g, n + 1, min, max} | rest], defs), else: []
    stop ++ more
  end

  defp expand([{:ref, name} | rest], defs), do: expand([Map.fetch!(defs, name) | rest], defs)
  defp expand([{:lit, ""} | rest], defs), do: expand(rest, defs)
  defp expand([{:lit, b} | rest], _defs) when is_binary(b), do: [[{:lit_at, b, 0} | rest]]
  defp expand([{:str, o} | rest], _defs), do: [[{:lit_at, "\"", 0}, {:str_in, :body, 0, 0, o} | rest]]
  defp expand([{:num, k} | rest], _defs), do: [[{:num_in, :start, k} | rest]]
  defp expand([{:substr, key} | rest], _defs), do: [[{:sub_at, key, 0} | rest]]
  defp expand([{:sub_at, _, _} | _] = s, _defs), do: [s]
  defp expand([{:class, _} | _] = s, _defs), do: [s]
  defp expand([{:lit_at, _, _} | _] = s, _defs), do: [s]
  defp expand([{:str_in, _, _, _, _} | _] = s, _defs), do: [s]
  defp expand([{:num_in, _, _} | _] = s, _defs), do: [s]

  # ------------------------------------------------------------- steps --

  defp step([], _b, _defs), do: []

  defp step([{:lit_at, lit, i} | rest], b, defs) do
    if :binary.at(lit, i) == b do
      if i + 1 == byte_size(lit), do: expand(rest, defs), else: [[{:lit_at, lit, i + 1} | rest]]
    else
      []
    end
  end

  defp step([{:class, ranges} | rest], b, defs),
    do: if(Enum.any?(ranges, &(b in &1)), do: expand(rest, defs), else: [])

  # JSON string body: n characters so far, `pend` UTF-8 continuation bytes owed
  defp step([{:str_in, :body, n, 0, o} | rest], b, defs) do
    cond do
      b == ?" -> if n >= o.min, do: expand(rest, defs), else: []
      # an escape is one more character: it may start only if one more fits,
      # or the configuration would be alive now and dead four bytes later
      b == ?\\ -> if o.max == nil or n + 1 <= o.max, do: [[{:str_in, :esc, n, 0, o} | rest]], else: []
      b < 0x20 -> []
      b < 0x80 -> char(n, 0, o, rest)
      b in 0xC2..0xDF -> char(n, 1, o, rest)
      # well-formed UTF-8 (RFC 3629 table 3-7): the second byte of E0, ED,
      # F0 and F4 is narrowed — no overlong forms, no encoded surrogates
      # (ED A0–BF), nothing past U+10FFFF — so every accepted string is
      # valid UTF-8 for any strict parser
      b == 0xE0 -> char(n, {2, 0xA0, 0xBF}, o, rest)
      b == 0xED -> char(n, {2, 0x80, 0x9F}, o, rest)
      b in 0xE1..0xEF -> char(n, 2, o, rest)
      b == 0xF0 -> char(n, {3, 0x90, 0xBF}, o, rest)
      b == 0xF4 -> char(n, {3, 0x80, 0x8F}, o, rest)
      b in 0xF1..0xF3 -> char(n, 3, o, rest)
      true -> []
    end
  end

  defp step([{:str_in, :body, n, {k, lo, hi}, o} | rest], b, _defs),
    do: if(b in lo..hi, do: [[{:str_in, :body, n, k - 1, o} | rest]], else: [])

  defp step([{:str_in, :body, n, pend, o} | rest], b, _defs) when pend > 0,
    do: if(b in 0x80..0xBF, do: [[{:str_in, :body, n, pend - 1, o} | rest]], else: [])

  defp step([{:str_in, :esc, n, 0, o} | rest], b, _defs) do
    cond do
      b in ~c"\"\\/bfnrt" -> char(n, 0, o, rest)
      b == ?u -> [[{:str_in, {:u, 0, 0}, n, 0, o} | rest]]
      true -> []
    end
  end

  # \uXXXX: a UTF-16 code unit. A high surrogate (D800–DBFF) must be followed
  # by an escaped low one (DC00–DFFF) — the pair is one character; a lone
  # surrogate is not a character, so it is refused as soon as it shows (the
  # second hex digit), never accepted and stranded
  defp step([{:str_in, {:u, k, acc}, n, 0, o} | rest], b, _defs) do
    case hex(b) do
      nil -> []
      d ->
        acc = acc * 16 + d

        cond do
          k == 1 and acc in 0xDC..0xDF -> []
          k == 3 and acc in 0xD800..0xDBFF -> [[{:str_in, {:lo, :bs}, n, 0, o} | rest]]
          k == 3 -> char(n, 0, o, rest)
          true -> [[{:str_in, {:u, k + 1, acc}, n, 0, o} | rest]]
        end
    end
  end

  defp step([{:str_in, {:lo, :bs}, n, 0, o} | rest], ?\\, _defs), do: [[{:str_in, {:lo, :u}, n, 0, o} | rest]]
  defp step([{:str_in, {:lo, :u}, n, 0, o} | rest], ?u, _defs), do: [[{:str_in, {:lo, 0, 0}, n, 0, o} | rest]]
  defp step([{:str_in, {:lo, _}, _, _, _} | _], _b, _defs), do: []

  defp step([{:str_in, {:lo, k, acc}, n, 0, o} | rest], b, _defs) do
    case hex(b) do
      nil -> []
      d ->
        acc = acc * 16 + d

        cond do
          k == 0 and acc != 0xD -> []
          k == 1 and acc not in 0xDC..0xDF -> []
          k == 3 -> char(n, 0, o, rest)
          true -> [[{:str_in, {:lo, k + 1, acc}, n, 0, o} | rest]]
        end
    end
  end

  defp step([{:num_in, st, k} | rest], b, defs) do
    case num(st, b, k) do
      nil -> []
      st2 -> [[{:num_in, st2, k} | rest]] ++ if(num_final?(st2), do: expand(rest, defs), else: [])
    end
  end

  # inside a substring: follow the automaton; it may end after any byte
  defp step([{:sub_at, key, st} | rest], b, defs) do
    case Map.fetch!(defs, key) |> elem(st) |> Map.get(b) do
      nil -> []
      st2 -> [[{:sub_at, key, st2} | rest] | expand(rest, defs)]
    end
  end

  defp hex(b) when b in ?0..?9, do: b - ?0
  defp hex(b) when b in ?a..?f, do: b - ?a + 10
  defp hex(b) when b in ?A..?F, do: b - ?A + 10
  defp hex(_), do: nil

  # a character starts (counted once, at its lead byte); without a maximum
  # the count is not kept, so configurations inside long strings coincide
  defp char(n, pend, %{max: nil} = o, rest), do: [[{:str_in, :body, min(n + 1, o.min), pend, o} | rest]]
  defp char(n, pend, o, rest), do: if(n + 1 <= o.max, do: [[{:str_in, :body, n + 1, pend, o} | rest]], else: [])

  # JSON number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
  defp num(:start, ?-, _), do: :sign
  defp num(s, ?0, _) when s in [:start, :sign], do: :zero
  defp num(s, d, _) when s in [:start, :sign] and d in ?1..?9, do: :int
  defp num(:int, d, _) when d in ?0..?9, do: :int
  defp num(s, ?., :number) when s in [:zero, :int], do: :dot
  defp num(s, d, _) when s in [:dot, :frac] and d in ?0..?9, do: :frac
  defp num(s, e, :number) when s in [:zero, :int, :frac] and e in ~c"eE", do: :exp
  defp num(:exp, sg, _) when sg in ~c"+-", do: :exp_sign
  defp num(s, d, _) when s in [:exp, :exp_sign, :exp_digits] and d in ?0..?9, do: :exp_digits
  defp num(_, _, _), do: nil

  defp num_final?(s), do: s in [:zero, :int, :frac, :exp_digits]

  # ---------------------------------------------------- suffix automaton --

  @doc """
  The suffix automaton of a binary (Blumer et al.): the smallest automaton
  accepting exactly its substrings, built in linear time. As a tuple of
  transition maps (state 0 the start), for `{:substr, key}`.
  """
  def suffix_automaton(text) when is_binary(text) do
    init = %{0 => %{len: 0, link: -1, next: %{}}}

    {states, _last} =
      for <<b <- text>>, reduce: {init, 0} do
        {st, last} -> extend(st, last, b)
      end

    0..(map_size(states) - 1) |> Enum.map(&states[&1].next) |> List.to_tuple()
  end

  defp extend(st, last, b) do
    cur = map_size(st)
    st = Map.put(st, cur, %{len: st[last].len + 1, link: -1, next: %{}})
    {st, p} = add_edges(st, last, b, cur)

    st =
      if p == -1 do
        put_in(st[cur].link, 0)
      else
        q = st[p].next[b]

        if st[p].len + 1 == st[q].len do
          put_in(st[cur].link, q)
        else
          clone = map_size(st)
          st = Map.put(st, clone, %{st[q] | len: st[p].len + 1})
          st = redirect(st, p, b, q, clone)
          st = put_in(st[q].link, clone)
          put_in(st[cur].link, clone)
        end
      end

    {st, cur}
  end

  defp add_edges(st, -1, _b, _cur), do: {st, -1}

  defp add_edges(st, p, b, cur) do
    if Map.has_key?(st[p].next, b),
      do: {st, p},
      else: add_edges(put_in(st[p].next[b], cur), st[p].link, b, cur)
  end

  defp redirect(st, -1, _b, _q, _clone), do: st

  defp redirect(st, p, b, q, clone) do
    if st[p].next[b] == q, do: redirect(put_in(st[p].next[b], clone), st[p].link, b, q, clone), else: st
  end

  # ------------------------------------------------------- builtin rules --

  @doc "Rules every grammar has: `:value` (any JSON value), `:object`, `:array`."
  def builtin_defs do
    %{
      value: {:alt, [{:ref, :object}, {:ref, :array}, {:str, %{min: 0, max: nil}}, {:num, :number},
                     {:lit, "true"}, {:lit, "false"}, {:lit, "null"}]},
      object: {:seq, [{:lit, "{"}, @ws,
                      {:alt, [{:lit, "}"},
                              {:seq, [kv_any(), {:rep, {:seq, [@ws, {:lit, ","}, @ws, kv_any()]}, 0, :inf}, @ws, {:lit, "}"}]}]}]},
      array: {:seq, [{:lit, "["}, @ws,
                     {:alt, [{:lit, "]"},
                             {:seq, [{:ref, :value}, {:rep, {:seq, [@ws, {:lit, ","}, @ws, {:ref, :value}]}, 0, :inf}, @ws, {:lit, "]"}]}]}]}
    }
  end

  defp kv_any, do: {:seq, [{:str, %{min: 0, max: nil}}, @ws, {:lit, ":"}, @ws, {:ref, :value}]}
end
