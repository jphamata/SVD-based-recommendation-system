defmodule Vapor.Template do
  @moduledoc """
  A hermetic evaluator for the Jinja dialect of Hugging Face chat templates.

  A model's `chat_template` (in `tokenizer_config.json` or a `.jinja` file)
  is the only authoritative description of its prompt format — and of how
  it expects tools, tool calls and tool results to be written. Hard-coding
  formats (`Vapor.Chat`) breaks on every new family; executing Jinja through
  a dependency breaks `deps: []`. This module interprets the subset real
  templates use, with the environment `transformers` renders them in
  (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`,
  loop controls, `tojson` as `json.dumps(ensure_ascii=False, …)`,
  `raise_exception`, `strftime_now`), and is tested byte for byte against
  `jinja2` on templates taken from real models (`template_test`).

  Hermetic means: no I/O, no host state. A template that reads the clock
  (`strftime_now`, used by Llama 3.x templates) gets the time *passed in*
  (`now:` option) — the clock is an input, so a rendered prompt is a pure
  function of its arguments, which is what an immutable agent's journal
  needs (`Vapor.Agent`). Without `now:`, `strftime_now` is an error.

  Supported: `{{ }}`, `{% %}`, `{# #}` with `-` whitespace control;
  `if/elif/else`, `for … in … [if …]` with `else`, `loop.*` (`index`,
  `index0`, `revindex`, `revindex0`, `first`, `last`, `length`,
  `previtem`, `nextitem`, `cycle`), `break`/`continue`, `set` (names, tuple
  unpacking, `namespace` attributes, block form), `macro` with defaults,
  `filter` blocks, `generation` blocks; literals, slices, attribute/item
  access, calls with keyword arguments, Python's operators (chained
  comparisons, `in`, `~`, `//`, `**`, conditional expressions); the common
  filters, tests, and `str`/`dict` methods. Values follow Python: truth,
  `str()`/`repr()` of lists, dicts, floats and `None`, insertion-ordered
  dicts (`{:dict, pairs}`, as `Vapor.JSON.decode(…, ordered: true)` gives).
  Anything else is an error naming the construct, never a silent difference.
  """

  import Bitwise

  defstruct [:ast, :digest]

  @type t :: %__MODULE__{}

  @doc "Parse a template."
  @spec compile(binary) :: {:ok, t} | {:error, term}
  def compile(src) when is_binary(src) do
    ast = src |> lex() |> whitespace() |> parse_template()
    {:ok, %__MODULE__{ast: ast, digest: :crypto.hash(:sha256, src)}}
  catch
    {:template_error, why} -> {:error, {:syntax, why}}
  end

  @doc """
  Render with variables `vars` (a map of names to values; dicts as
  `{:dict, pairs}` or maps). Options: `now:` (a `DateTime` or
  `NaiveDateTime`, for `strftime_now`). `{:ok, text}` or `{:error, why}`
  (`raise_exception(msg)` gives `{:error, {:raised, msg}}`).
  """
  def render(tpl, vars, opts \\ [])

  def render(src, vars, opts) when is_binary(src) do
    with {:ok, t} <- compile(src), do: render(t, vars, opts)
  end

  def render(%__MODULE__{ast: ast}, vars, opts) do
    scope = Map.new(vars, fn {k, v} -> {to_string(k), normalize(v)} end)
    st = %{scopes: [scope], ns: %{}, next: 0, now: Keyword.get(opts, :now), macros: %{}}
    {out, _st} = exec(ast, st)
    {:ok, IO.iodata_to_binary(out)}
  catch
    {:raised, msg} -> {:error, {:raised, msg}}
    {:template_error, why} -> {:error, why}
  end

  # maps from callers become ordered dicts (sorted keys: a map has no order)
  defp normalize(%{} = m) when not is_struct(m), do: {:dict, m |> Enum.map(fn {k, v} -> {to_string(k), normalize(v)} end) |> Enum.sort()}
  defp normalize({:dict, pairs}), do: {:dict, Enum.map(pairs, fn {k, v} -> {k, normalize(v)} end)}
  defp normalize(l) when is_list(l), do: Enum.map(l, &normalize/1)
  defp normalize(a) when is_atom(a) and a not in [nil, true, false], do: Atom.to_string(a)
  defp normalize(v), do: v

  defp fail(why), do: throw({:template_error, why})

  # =================================================================== lexing

  # [{:text, s} | {kind, inner, left_ws, right_ws}], kind in :var | :block | :comment
  defp lex(src), do: lex(src, [])

  defp lex("", acc), do: Enum.reverse(acc)

  defp lex(src, acc) do
    case :binary.match(src, ["{{", "{%", "{#"]) do
      :nomatch ->
        Enum.reverse([{:text, src} | acc])

      {at, 2} ->
        text = binary_part(src, 0, at)
        open = binary_part(src, at, 2)
        rest = binary_part(src, at + 2, byte_size(src) - at - 2)
        {kind, close} = %{"{{" => {:var, "}}"}, "{%" => {:block, "%}"}, "{#" => {:comment, "#}"}}[open]
        {inner, rest} = until_close(rest, close, kind)
        {left, inner} = marker(inner, :left)
        {right, inner} = marker(inner, :right)
        acc = if text == "", do: acc, else: [{:text, text} | acc]
        lex(rest, [{kind, String.trim(inner), left, right} | acc])
    end
  end

  # the tag body up to its closing delimiter (string literals may contain it)
  defp until_close(src, close, :comment) do
    case :binary.split(src, close) do
      [inner, rest] -> {inner, rest}
      _ -> fail("unterminated comment")
    end
  end

  defp until_close(src, close, _kind), do: scan_close(src, close, 0, nil)

  defp scan_close(src, close, i, quote) when i < byte_size(src) do
    c = :binary.at(src, i)

    cond do
      quote != nil and c == ?\\ -> scan_close(src, close, i + 2, quote)
      quote != nil and c == quote -> scan_close(src, close, i + 1, nil)
      quote != nil -> scan_close(src, close, i + 1, quote)
      c in [?', ?"] -> scan_close(src, close, i + 1, c)
      binary_part(src, i, min(2, byte_size(src) - i)) == close ->
        {binary_part(src, 0, i), binary_part(src, i + 2, byte_size(src) - i - 2)}
      true -> scan_close(src, close, i + 1, nil)
    end
  end

  defp scan_close(_src, close, _i, _q), do: fail("missing #{close}")

  defp marker(<<"-", rest::binary>>, :left), do: {:minus, rest}
  defp marker(<<"+", rest::binary>>, :left), do: {:plus, rest}
  defp marker(s, :left), do: {nil, s}

  defp marker(s, :right) do
    cond do
      String.ends_with?(s, "-") -> {:minus, binary_part(s, 0, byte_size(s) - 1)}
      String.ends_with?(s, "+") -> {:plus, binary_part(s, 0, byte_size(s) - 1)}
      true -> {nil, s}
    end
  end

  # Whitespace control, as jinja2's lexer does it with trim_blocks and
  # lstrip_blocks: `-` strips all whitespace on its side; otherwise a block
  # or comment tag eats one newline after it (trim_blocks) and the spaces and
  # tabs between the start of its line and itself (lstrip_blocks) — "start
  # of its line" meaning a newline, or the start of the template, before them.
  defp whitespace(tokens) do
    {acc, _pending, _line_start} =
      Enum.reduce(tokens, {[], nil, true}, fn
        {:text, t}, {acc, pending, line_start} ->
          {t, line_start} = apply_right(t, pending, line_start)
          {[{:text, t, line_start} | acc], nil, false}

        {kind, _inner, left, right} = tag, {acc, _pending, line_start} ->
          {[tag | apply_left(acc, kind, left)], {kind, right}, line_start}
      end)

    acc
    |> Enum.reverse()
    |> Enum.flat_map(fn
      {:text, "", _} -> []
      {:text, t, _} -> [{:text, t}]
      tag -> [tag]
    end)
  end

  # the text after a tag: {text, whether it now starts a line}
  defp apply_right(t, nil, line_start), do: {t, line_start}

  defp apply_right(t, {_kind, :minus}, _ls) do
    stripped = String.trim_leading(t)
    removed = binary_part(t, 0, byte_size(t) - byte_size(stripped))
    {stripped, String.ends_with?(removed, "\n")}
  end

  defp apply_right(t, {_kind, :plus}, _ls), do: {t, false}
  defp apply_right(t, {:var, nil}, _ls), do: {t, false}

  defp apply_right(t, {_block_or_comment, nil}, _ls) do
    case t do
      <<"\r\n", r::binary>> -> {r, true}
      <<"\n", r::binary>> -> {r, true}
      _ -> {t, false}
    end
  end

  defp apply_left([{:text, t, ls} | rest], _kind, :minus), do: [{:text, String.trim_trailing(t), ls} | rest]
  defp apply_left(acc, _kind, :minus), do: acc
  defp apply_left(acc, _kind, :plus), do: acc
  defp apply_left(acc, :var, nil), do: acc

  defp apply_left([{:text, t, ls} | rest] = acc, _block, nil) do
    l_pos = case :binary.matches(t, "\n") do
      [] -> 0
      ms -> {at, _} = List.last(ms); at + 1
    end

    tail = binary_part(t, l_pos, byte_size(t) - l_pos)

    if (l_pos > 0 or ls) and tail != "" and String.trim(tail) == "",
      do: [{:text, binary_part(t, 0, l_pos), ls} | rest],
      else: acc
  end

  defp apply_left(acc, _kind, nil), do: acc

  # ================================================================== parsing

  defp parse_template(tokens) do
    case body(tokens, []) do
      {nodes, [], nil} -> nodes
      {_, _, tag} -> fail("unexpected {% #{tag} %}")
    end
  end

  # nodes until one of `ends` (a block keyword): {nodes, rest, end_keyword_and_args}
  defp body([], ends) do
    if ends == [], do: {[], [], nil}, else: fail("missing {% #{Enum.join(ends, " / ")} %}")
  end

  defp body([{:text, t} | rest], ends) do
    {nodes, rest, e} = body(rest, ends)
    {[{:text, t} | nodes], rest, e}
  end

  defp body([{:comment, _, _, _} | rest], ends), do: body(rest, ends)

  defp body([{:var, src, _, _} | rest], ends) do
    expr = parse_expr_full(src)
    {nodes, rest, e} = body(rest, ends)
    {[{:out, expr} | nodes], rest, e}
  end

  defp body([{:block, src, _, _} | rest], ends) do
    toks = tokenize(src)

    case toks do
      [{:name, kw} | args] ->
        if kw in ends do
          {[], rest, {kw, args}}
        else
          {node, rest} = statement(kw, args, rest)
          {nodes, rest, e} = body(rest, ends)
          {[node | nodes], rest, e}
        end

      _ ->
        fail("empty or malformed tag {% #{src} %}")
    end
  end

  defp statement("if", args, rest), do: if_chain(args, rest)

  defp statement("for", args, rest) do
    {targets, args} = targets(args)
    args = expect(args, {:name, "in"})
    # the iterable is read without a conditional expression: a trailing
    # `if` is the loop filter (jinja2's parse_tuple(with_condexpr=False))
    {iter, args} = or_(args)

    {cond_, args} =
      case args do
        [{:name, "if"} | a] -> expr(a)
        a -> {nil, a}
      end

    args = case args do
      [{:name, "recursive"}] -> []
      a -> a
    end

    if args != [], do: fail("unexpected tokens in for: #{inspect(args)}")
    {loop_body, rest, {e, _}} = body(rest, ["else", "endfor"])

    {else_body, rest} =
      if e == "else" do
        {b, rest, _} = body(rest, ["endfor"])
        {b, rest}
      else
        {[], rest}
      end

    {{:for, targets, iter, cond_, loop_body, else_body}, rest}
  end

  defp statement("set", args, rest) do
    {target, args} = set_target(args)

    case args do
      [{:op, "="} | e] ->
        {value, []} = expr_all(e)
        {{:set, target, value}, rest}

      filters ->
        {b, rest, _} = body(rest, ["endset"])
        {{:set_block, target, filters_of(filters), b}, rest}
    end
  end

  defp statement("macro", [{:name, name}, {:op, "("} | args], rest) do
    {params, args} = params(args, [])
    if args != [], do: fail("unexpected tokens after macro parameters")
    {b, rest, _} = body(rest, ["endmacro"])
    {{:macro, name, params, b}, rest}
  end

  defp statement("filter", args, rest) do
    {b, rest, _} = body(rest, ["endfilter"])
    {{:filter_block, filters_of([{:op, "|"} | args]), b}, rest}
  end

  defp statement("generation", [], rest) do
    {b, rest, _} = body(rest, ["endgeneration"])
    {{:block, b}, rest}
  end

  defp statement("break", [], rest), do: {:break, rest}
  defp statement("continue", [], rest), do: {:continue, rest}
  defp statement(kw, _args, _rest), do: fail("unsupported tag {% #{kw} %}")

  defp if_chain(args, rest) do
    {cond_, []} = expr_all(args)
    {b, rest, {e, eargs}} = body(rest, ["elif", "else", "endif"])

    case e do
      "endif" -> {{:if, [{cond_, b}], []}, rest}
      "else" ->
        {eb, rest, _} = body(rest, ["endif"])
        {{:if, [{cond_, b}], eb}, rest}

      "elif" ->
        {{:if, branches, eb}, rest} = if_chain(eargs, rest)
        {{:if, [{cond_, b} | branches], eb}, rest}
    end
  end

  defp targets(args) do
    {names, rest} = Enum.split_while(args, &(&1 != {:name, "in"}))
    names = Enum.reject(names, &(&1 in [{:op, ","}, {:op, "("}, {:op, ")"}]))
    if not Enum.all?(names, &match?({:name, _}, &1)), do: fail("for targets must be names")
    {Enum.map(names, &elem(&1, 1)), rest}
  end

  defp set_target([{:name, ns}, {:op, "."}, {:name, attr} | rest]), do: {{:attr, ns, attr}, rest}

  defp set_target(args) do
    {names, rest} = Enum.split_while(args, &(match?({:name, _}, &1) or &1 == {:op, ","}))
    names = for {:name, n} <- names, do: n

    case names do
      [n] -> {{:name, n}, rest}
      ns when ns != [] -> {{:names, ns}, rest}
      _ -> fail("set needs a target")
    end
  end

  defp params([{:op, ")"} | rest], acc), do: {Enum.reverse(acc), rest}
  defp params([{:op, ","} | rest], acc), do: params(rest, acc)

  defp params([{:name, n}, {:op, "="} | rest], acc) do
    {d, rest} = expr(rest)
    params(rest, [{n, d} | acc])
  end

  defp params([{:name, n} | rest], acc), do: params(rest, [{n, :required} | acc])
  defp params(other, _acc), do: fail("bad macro parameters #{inspect(other)}")

  # `| f(args) | g` after a set target or in a filter block
  defp filters_of([]), do: []

  defp filters_of([{:op, "|"} | rest]) do
    {{:filter, _, name, pos, kw}, rest} = filter({:lit, nil}, rest)
    [{name, pos, kw} | filters_of(rest)]
  end

  defp filters_of(other), do: fail("unexpected #{inspect(Enum.take(other, 3))} in a filter list")

  defp expect([t | rest], t), do: rest
  defp expect(other, t), do: fail("expected #{inspect(t)}, got #{inspect(Enum.take(other, 3))}")

  # =========================================================== expression lexer

  @ops ["**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "(", ")", "[", "]", "{", "}", ",", ":", ".", "|", "="]

  defp tokenize(src), do: tok(src, [])

  defp tok(<<>>, acc), do: Enum.reverse(acc)
  defp tok(<<c, r::binary>>, acc) when c in [?\s, ?\t, ?\n, ?\r], do: tok(r, acc)

  defp tok(<<q, r::binary>>, acc) when q in [?', ?"] do
    {s, r} = str_lit(r, q, [])
    tok(r, [{:str, s} | acc])
  end

  defp tok(<<c, _::binary>> = s, acc) when c in ?0..?9 do
    [num | _] = Regex.run(~r/\A\d+(\.\d+)?([eE][+-]?\d+)?/, s)
    r = binary_part(s, byte_size(num), byte_size(s) - byte_size(num))
    v = if String.contains?(num, [".", "e", "E"]), do: elem(Float.parse(num), 0), else: String.to_integer(num)
    tok(r, [{:num, v} | acc])
  end

  defp tok(<<c, _::binary>> = s, acc) when c in ?a..?z or c in ?A..?Z or c == ?_ do
    [name] = Regex.run(~r/\A[A-Za-z_][A-Za-z0-9_]*/, s)
    tok(binary_part(s, byte_size(name), byte_size(s) - byte_size(name)), [{:name, name} | acc])
  end

  defp tok(s, acc) do
    case Enum.find(@ops, &String.starts_with?(s, &1)) do
      nil -> fail("unexpected character in expression: #{inspect(String.slice(s, 0, 10))}")
      op -> tok(binary_part(s, byte_size(op), byte_size(s) - byte_size(op)), [{:op, op} | acc])
    end
  end

  defp str_lit(<<q, r::binary>>, q, acc), do: {IO.iodata_to_binary(Enum.reverse(acc)), r}

  defp str_lit(<<?\\, c, r::binary>>, q, acc) do
    case c do
      ?n -> str_lit(r, q, ["\n" | acc])
      ?t -> str_lit(r, q, ["\t" | acc])
      ?r -> str_lit(r, q, ["\r" | acc])
      ?\\ -> str_lit(r, q, ["\\" | acc])
      ?' -> str_lit(r, q, ["'" | acc])
      ?" -> str_lit(r, q, ["\"" | acc])
      ?0 -> str_lit(r, q, [<<0>> | acc])
      ?u -> <<h::binary-4, r2::binary>> = r; str_lit(r2, q, [<<String.to_integer(h, 16)::utf8>> | acc])
      ?x -> <<h::binary-2, r2::binary>> = r; str_lit(r2, q, [<<String.to_integer(h, 16)::utf8>> | acc])
      ?\n -> str_lit(r, q, acc)
      other -> str_lit(r, q, [<<?\\, other>> | acc])
    end
  end

  defp str_lit(<<c, r::binary>>, q, acc), do: str_lit(r, q, [<<c>> | acc])
  defp str_lit(<<>>, _q, _acc), do: fail("unterminated string literal")

  # ========================================================== expression parser

  defp parse_expr_full(src) do
    case expr(tokenize(src)) do
      {e, []} -> e
      {_, rest} -> fail("unexpected tokens #{inspect(Enum.take(rest, 4))} in {{ #{src} }}")
    end
  end

  defp expr_all(toks) do
    case expr(toks) do
      {e, []} -> {e, []}
      {e, [{:op, ","} | _] = rest} -> tuple_rest([e], rest)
      {_, rest} -> fail("unexpected tokens #{inspect(Enum.take(rest, 4))}")
    end
  end

  defp tuple_rest(acc, [{:op, ","} | rest]) do
    {e, rest} = expr(rest)
    tuple_rest([e | acc], rest)
  end

  defp tuple_rest(acc, []), do: {{:list, Enum.reverse(acc)}, []}

  # conditional expression
  defp expr(toks) do
    {a, rest} = or_(toks)

    case rest do
      [{:name, "if"} | r] ->
        {c, r} = or_(r)

        case r do
          [{:name, "else"} | r2] ->
            {b, r2} = expr(r2)
            {{:cond, a, c, b}, r2}

          _ ->
            {{:cond, a, c, nil}, r}
        end

      _ ->
        {a, rest}
    end
  end

  defp or_(t) do
    {a, r} = and_(t)
    or_more(a, r)
  end

  defp or_more(a, [{:name, "or"} | r]) do
    {b, r} = and_(r)
    or_more({:or, a, b}, r)
  end

  defp or_more(a, r), do: {a, r}

  defp and_(t) do
    {a, r} = not_(t)
    and_more(a, r)
  end

  defp and_more(a, [{:name, "and"} | r]) do
    {b, r} = not_(r)
    and_more({:and, a, b}, r)
  end

  defp and_more(a, r), do: {a, r}

  defp not_([{:name, "not"} | r]) do
    {a, r} = not_(r)
    {{:not, a}, r}
  end

  defp not_(t), do: compare(t)

  @cmp ~w(== != < > <= >=)

  defp compare(t) do
    {a, r} = math1(t)
    compare_more(a, [], r)
  end

  defp compare_more(a, ops, [{:op, op} | r]) when op in @cmp do
    {b, r} = math1(r)
    compare_more(a, [{op, b} | ops], r)
  end

  defp compare_more(a, ops, [{:name, "in"} | r]) do
    {b, r} = math1(r)
    compare_more(a, [{"in", b} | ops], r)
  end

  defp compare_more(a, ops, [{:name, "not"}, {:name, "in"} | r]) do
    {b, r} = math1(r)
    compare_more(a, [{"notin", b} | ops], r)
  end

  defp compare_more(a, [], r), do: {a, r}
  defp compare_more(a, ops, r), do: {{:compare, a, Enum.reverse(ops)}, r}

  defp math1(t) do
    {a, r} = concat(t)
    math1_more(a, r)
  end

  defp math1_more(a, [{:op, op} | r]) when op in ["+", "-"] do
    {b, r} = concat(r)
    math1_more({:bin, op, a, b}, r)
  end

  defp math1_more(a, r), do: {a, r}

  defp concat(t) do
    {a, r} = math2(t)
    concat_more(a, r)
  end

  defp concat_more(a, [{:op, "~"} | r]) do
    {b, r} = math2(r)
    concat_more({:bin, "~", a, b}, r)
  end

  defp concat_more(a, r), do: {a, r}

  defp math2(t) do
    {a, r} = pow(t)
    math2_more(a, r)
  end

  defp math2_more(a, [{:op, op} | r]) when op in ["*", "/", "//", "%"] do
    {b, r} = pow(r)
    math2_more({:bin, op, a, b}, r)
  end

  defp math2_more(a, r), do: {a, r}

  defp pow(t) do
    {a, r} = unary(t, true)

    case r do
      [{:op, "**"} | r2] ->
        {b, r2} = unary(r2, true)
        {{:bin, "**", a, b}, r2}

      _ ->
        {a, r}
    end
  end

  defp unary([{:op, "-"} | r], with_filter) do
    {a, r} = unary(r, false)
    filter_expr({:neg, a}, r, with_filter)
  end

  defp unary([{:op, "+"} | r], with_filter) do
    {a, r} = unary(r, false)
    filter_expr(a, r, with_filter)
  end

  defp unary(t, with_filter) do
    {a, r} = primary(t)
    {a, r} = postfix(a, r)
    filter_expr(a, r, with_filter)
  end

  defp filter_expr(a, r, false), do: {a, r}

  defp filter_expr(a, [{:op, "|"} | r], true) do
    {f, r} = filter(a, r)
    filter_expr(f, r, true)
  end

  defp filter_expr(a, [{:name, "is"} | r], true) do
    {neg, r} = case r do
      [{:name, "not"} | r2] -> {true, r2}
      _ -> {false, r}
    end

    {name, r} = case r do
      [{:name, n} | r2] -> {n, r2}
      [{:op, op} | r2] when op in ["==", "!=", "<", ">", "<=", ">="] -> {op, r2}
      _ -> fail("expected a test name after is")
    end

    {args, r} =
      case r do
        [{:op, "("} | r2] -> call_args(r2, [], [])
        [{kind, _} = t | _] when kind in [:str, :num] or (kind == :name and elem(t, 1) not in ["and", "or", "else", "if", "in", "not", "is"]) ->
          {a1, r2} = primary(r)
          {a1, r2} = postfix(a1, r2)
          {{[a1], []}, r2}
        _ -> {{[], []}, r}
      end

    {pos, kw} = args
    filter_expr({:test, a, name, pos, kw, neg}, r, true)
  end

  defp filter_expr(a, r, true), do: {a, r}

  defp filter(a, [{:name, name} | r]) do
    {name, r} = dotted(name, r)

    case r do
      [{:op, "("} | r2] ->
        {{pos, kw}, r2} = call_args(r2, [], [])
        {{:filter, a, name, pos, kw}, r2}

      _ ->
        {{:filter, a, name, [], []}, r}
    end
  end

  defp filter(_a, r), do: fail("expected a filter name, got #{inspect(Enum.take(r, 2))}")

  defp dotted(name, [{:op, "."}, {:name, n} | r]), do: dotted(name <> "." <> n, r)
  defp dotted(name, r), do: {name, r}

  defp primary([{:str, s} | r]), do: str_concat(s, r)
  defp primary([{:num, n} | r]), do: {{:lit, n}, r}
  defp primary([{:name, n} | r]) when n in ["true", "True"], do: {{:lit, true}, r}
  defp primary([{:name, n} | r]) when n in ["false", "False"], do: {{:lit, false}, r}
  defp primary([{:name, n} | r]) when n in ["none", "None"], do: {{:lit, nil}, r}
  defp primary([{:name, n} | r]), do: {{:var, n}, r}

  defp primary([{:op, "("} | r]) do
    case r do
      [{:op, ")"} | r2] -> {{:list, []}, r2}
      _ ->
        {e, r} = expr(r)

        case r do
          [{:op, ")"} | r2] -> {e, r2}
          [{:op, ","} | _] ->
            {items, r2} = seq(r, [e], ")")
            {{:list, items}, r2}
          _ -> fail("expected )")
        end
    end
  end

  defp primary([{:op, "["} | r]) do
    {items, r} = seq(r, [], "]")
    {{:list, items}, r}
  end

  defp primary([{:op, "{"} | r]), do: dict_lit(r, [])
  defp primary(t), do: fail("unexpected #{inspect(Enum.take(t, 3))}")

  # adjacent string literals concatenate ('a' 'b')
  defp str_concat(s, [{:str, s2} | r]), do: str_concat(s <> s2, r)
  defp str_concat(s, r), do: {{:lit, s}, r}

  # comma-separated expressions until `close`
  defp seq([{:op, close} | r], acc, close), do: {Enum.reverse(acc), r}
  defp seq([{:op, ","} | r], acc, close), do: seq(r, acc, close)

  defp seq(t, acc, close) do
    {e, r} = expr(t)
    seq(r, [e | acc], close)
  end

  defp dict_lit([{:op, "}"} | r], acc), do: {{:dict, Enum.reverse(acc)}, r}
  defp dict_lit([{:op, ","} | r], acc), do: dict_lit(r, acc)

  defp dict_lit(t, acc) do
    {k, r} = expr(t)
    r = expect(r, {:op, ":"})
    {v, r} = expr(r)
    dict_lit(r, [{k, v} | acc])
  end

  defp postfix(a, [{:op, "."}, {:name, n} | r]), do: postfix({:attr, a, n}, r)
  defp postfix(a, [{:op, "."}, {:num, n} | r]) when is_integer(n), do: postfix({:item, a, {:lit, n}}, r)

  defp postfix(a, [{:op, "["} | r]) do
    {sub, r} = subscript(r)
    postfix(sub.(a), r)
  end

  defp postfix(a, [{:op, "("} | r]) do
    {{pos, kw}, r} = call_args(r, [], [])
    postfix({:call, a, pos, kw}, r)
  end

  defp postfix(a, r), do: {a, r}

  # x[i] or x[a:b:c]
  defp subscript(r) do
    {start, r} = slice_part(r)

    case r do
      [{:op, "]"} | r2] when start != nil -> {&{:item, &1, start}, r2}
      [{:op, ":"} | r2] ->
        {stop, r2} = slice_part(r2)

        {step, r2} =
          case r2 do
            [{:op, ":"} | r3] -> slice_part(r3)
            _ -> {nil, r2}
          end

        [{:op, "]"} | r3] = r2
        {&{:slice, &1, start, stop, step}, r3}
    end
  end

  defp slice_part([{:op, op} | _] = r) when op in [":", "]"], do: {nil, r}
  defp slice_part(r), do: expr(r)

  defp call_args([{:op, ")"} | r], pos, kw), do: {{Enum.reverse(pos), Enum.reverse(kw)}, r}
  defp call_args([{:op, ","} | r], pos, kw), do: call_args(r, pos, kw)

  defp call_args([{:name, k}, {:op, "="} | r], pos, kw) do
    {v, r} = expr(r)
    call_args(r, pos, [{k, v} | kw])
  end

  defp call_args(t, pos, kw) do
    {v, r} = expr(t)
    call_args(r, [v | pos], kw)
  end

  # ================================================================ execution

  # loop controls (break/continue) carry the output produced before them
  defp exec(nodes, st) do
    Enum.reduce(nodes, {[], st}, fn node, {out, st} ->
      try do
        {o, st} = run(node, st)
        {[out | o], st}
      catch
        {:loop_ctl, kind, o, st2} -> throw({:loop_ctl, kind, [out | o], st2})
      end
    end)
  end

  defp run({:text, t}, st), do: {t, st}

  defp run({:out, e}, st) do
    {v, st} = ev(e, st)
    {to_s(v, st), st}
  end

  defp run({:if, branches, else_body}, st) do
    Enum.reduce_while(branches, :none, fn {c, b}, :none ->
      {v, _} = ev(c, st)
      if truthy?(v, st), do: {:halt, {:take, b}}, else: {:cont, :none}
    end)
    |> case do
      {:take, b} -> exec(b, st)
      :none -> exec(else_body, st)
    end
  end

  defp run({:for, targets, iter, cond_, b, else_body}, st) do
    {coll, st} = ev(iter, st)
    items = iterate(coll, st)

    # the loop filter applies before loop.* is computed (jinja2 semantics)
    items =
      if cond_,
        do: Enum.filter(items, fn it -> {v, _} = ev(cond_, push(st, bind_targets(targets, it))); truthy?(v, st) end),
        else: items

    n = length(items)

    if n == 0 do
      exec(else_body, st)
    else
      tup = List.to_tuple(items)
      outer = st.scopes

      # every iteration starts from a fresh scope: a `set` in the body is
      # seen neither by later iterations nor after the loop (namespaces are)
      {out, st} =
        Enum.reduce_while(0..(n - 1), {[], st}, fn i, {out, st} ->
          loop = %{"index0" => i, "index" => i + 1, "revindex" => n - i, "revindex0" => n - i - 1,
                   "first" => i == 0, "last" => i == n - 1, "length" => n,
                   "previtem" => if(i > 0, do: elem(tup, i - 1), else: {:undef, "previtem"}),
                   "nextitem" => if(i < n - 1, do: elem(tup, i + 1), else: {:undef, "nextitem"})}

          scope = targets |> bind_targets(elem(tup, i)) |> Map.put("loop", {:loop, loop})
          st1 = %{st | scopes: [scope | outer]}

          try do
            {o, st2} = exec(b, st1)
            {:cont, {[out | o], %{st2 | scopes: outer}}}
          catch
            {:loop_ctl, :break, o, st2} -> {:halt, {[out | o], %{st2 | scopes: outer}}}
            {:loop_ctl, :continue, o, st2} -> {:cont, {[out | o], %{st2 | scopes: outer}}}
          end
        end)

      {out, st}
    end
  end

  defp run(:break, st), do: throw({:loop_ctl, :break, [], st})
  defp run(:continue, st), do: throw({:loop_ctl, :continue, [], st})

  defp run({:set, {:name, n}, e}, st) do
    {v, st} = ev(e, st)
    {[], assign(st, n, v)}
  end

  defp run({:set, {:names, ns}, e}, st) do
    {v, st} = ev(e, st)
    vals = iterate(v, st)
    if length(vals) != length(ns), do: fail("cannot unpack #{length(vals)} values into #{length(ns)} names")
    {[], Enum.zip(ns, vals) |> Enum.reduce(st, fn {n, x}, st -> assign(st, n, x) end)}
  end

  defp run({:set, {:attr, n, attr}, e}, st) do
    {v, st} = ev(e, st)

    case lookup(st, n) do
      {:ns, ref} -> {[], %{st | ns: Map.update!(st.ns, ref, &dict_put(&1, attr, v))}}
      _ -> fail("cannot assign attribute #{attr} of #{n} (not a namespace)")
    end
  end

  defp run({:set_block, target, filters, b}, st) do
    {out, st} = exec(b, st)
    v = apply_filters(IO.iodata_to_binary(out), filters, st)
    run({:set, target, {:lit, v}}, st)
  end

  defp run({:filter_block, filters, b}, st) do
    {out, st} = exec(b, st)
    {to_s(apply_filters(IO.iodata_to_binary(out), filters, st), st), st}
  end

  defp run({:block, b}, st), do: exec(b, st)

  defp run({:macro, name, params, b}, st), do: {[], assign(st, name, {:macro, name, params, b})}

  defp apply_filters(v, filters, st) do
    Enum.reduce(filters, v, fn {name, args, kw}, acc ->
      {pos, st} = ev_list(args, st)
      {kws, _} = ev_kw(kw, st)
      filt(name, acc, pos, kws, st)
    end)
  end

  defp push(st, scope), do: %{st | scopes: [scope | st.scopes]}

  defp bind_targets([t], item), do: %{t => item}

  defp bind_targets(ts, item) do
    vals = if is_list(item), do: item, else: fail("cannot unpack #{inspect(item)}")
    if length(vals) != length(ts), do: fail("cannot unpack #{length(vals)} values into #{length(ts)} names")
    Map.new(Enum.zip(ts, vals))
  end

  defp assign(%{scopes: [s | rest]} = st, n, v), do: %{st | scopes: [Map.put(s, n, v) | rest]}

  defp lookup(st, n) do
    Enum.find_value(st.scopes, {:undef, n}, fn s -> case s do
      %{^n => v} -> {:found, v}
      _ -> nil
    end end)
    |> case do
      {:found, v} -> v
      other -> other
    end
  end

  # ============================================================== evaluation

  defp ev({:lit, v}, st), do: {v, st}
  defp ev({:var, n}, st), do: {lookup_global(st, n), st}

  defp ev({:list, es}, st) do
    {vs, st} = ev_list(es, st)
    {vs, st}
  end

  defp ev({:dict, kvs}, st) do
    {pairs, st} =
      Enum.map_reduce(kvs, st, fn {k, v}, st ->
        {kv, st} = ev(k, st)
        {vv, st} = ev(v, st)
        {{kv, vv}, st}
      end)

    {Enum.reduce(pairs, {:dict, []}, fn {k, v}, d -> dict_put(d, k, v) end), st}
  end

  defp ev({:attr, e, name}, st) do
    {v, st} = ev(e, st)
    {getattr(v, name, st), st}
  end

  defp ev({:item, e, k}, st) do
    {v, st} = ev(e, st)
    {kv, st} = ev(k, st)
    {getitem(v, kv, st), st}
  end

  defp ev({:slice, e, a, b, c}, st) do
    {v, st} = ev(e, st)
    [a, b, c] = Enum.map([a, b, c], fn x -> if x, do: elem(ev(x, st), 0), else: nil end)
    {slice(v, a, b, c), st}
  end

  defp ev({:call, f, args, kw}, st) do
    {fv, st} = ev(f, st)
    {pos, st} = ev_list(args, st)
    {kws, st} = ev_kw(kw, st)
    call(fv, pos, kws, st)
  end

  defp ev({:filter, e, name, args, kw}, st) do
    {v, st} = ev(e, st)
    {pos, st} = ev_list(args, st)
    {kws, st} = ev_kw(kw, st)
    {filt(name, v, pos, kws, st), st}
  end

  defp ev({:test, e, name, args, kw, neg}, st) do
    {v, st} = ev(e, st)
    {pos, st} = ev_list(args, st)
    {_kws, st} = ev_kw(kw, st)
    r = test(name, v, pos, st)
    {if(neg, do: not r, else: r), st}
  end

  defp ev({:not, e}, st) do
    {v, st} = ev(e, st)
    {not truthy?(v, st), st}
  end

  defp ev({:and, a, b}, st) do
    {va, st} = ev(a, st)
    if truthy?(va, st), do: ev(b, st), else: {va, st}
  end

  defp ev({:or, a, b}, st) do
    {va, st} = ev(a, st)
    if truthy?(va, st), do: {va, st}, else: ev(b, st)
  end

  defp ev({:neg, e}, st) do
    {v, st} = ev(e, st)
    if is_number(v), do: {-v, st}, else: fail("bad operand for unary -: #{repr(v, st)}")
  end

  defp ev({:cond, a, c, b}, st) do
    {vc, st} = ev(c, st)

    cond do
      truthy?(vc, st) -> ev(a, st)
      b -> ev(b, st)
      true -> {{:undef, "conditional"}, st}
    end
  end

  defp ev({:bin, op, a, b}, st) do
    {va, st} = ev(a, st)
    {vb, st} = ev(b, st)
    {binop(op, va, vb, st), st}
  end

  defp ev({:compare, a, ops}, st) do
    {va, st} = ev(a, st)

    {r, _, st} =
      Enum.reduce_while(ops, {true, va, st}, fn {op, e}, {true, left, st} ->
        {vb, st} = ev(e, st)
        if cmp(op, left, vb, st), do: {:cont, {true, vb, st}}, else: {:halt, {false, vb, st}}
      end)

    {r, st}
  end

  defp ev_list(es, st), do: Enum.map_reduce(es, st, &ev/2)

  defp ev_kw(kw, st) do
    {pairs, st} = Enum.map_reduce(kw, st, fn {k, e}, st -> {v, st} = ev(e, st); {{k, v}, st} end)
    {Map.new(pairs), st}
  end

  @globals ~w(namespace raise_exception strftime_now range dict)

  defp lookup_global(st, n) do
    case lookup(st, n) do
      {:undef, _} when n in @globals -> {:builtin, n}
      v -> v
    end
  end

  # ------------------------------------------------------------------- calls

  defp call({:builtin, "namespace"}, pos, kws, st) do
    ref = st.next
    init = case pos do
      [{:dict, pairs}] -> pairs
      [] -> []
      _ -> fail("namespace() takes keyword arguments")
    end

    pairs = Enum.reduce(Enum.sort(kws), {:dict, init}, fn {k, v}, d -> dict_put(d, k, v) end)
    # keyword arguments keep the call's order in Python; ours are sorted, which only
    # matters if a namespace itself is printed
    {{:ns, ref}, %{st | ns: Map.put(st.ns, ref, pairs), next: ref + 1}}
  end

  defp call({:builtin, "raise_exception"}, [msg | _], _kws, st), do: throw({:raised, to_s(msg, st)})

  defp call({:builtin, "strftime_now"}, [fmt], _kws, st) do
    case st.now do
      nil -> fail("strftime_now needs an explicit clock: render(…, now: datetime)")
      now -> {strftime(now, fmt), st}
    end
  end

  defp call({:builtin, "range"}, args, _kws, st) do
    list =
      case args do
        [n] -> range_list(0, n, 1)
        [a, b] -> range_list(a, b, 1)
        [a, b, s] -> range_list(a, b, s)
      end

    {list, st}
  end

  defp call({:builtin, "dict"}, [], kws, st), do: {{:dict, Enum.sort(kws)}, st}

  defp call({:macro, _name, params, b}, pos, kws, st) do
    scope =
      params
      |> Enum.with_index()
      |> Enum.reduce(%{}, fn {{p, default}, i}, acc ->
        v =
          cond do
            i < length(pos) -> Enum.at(pos, i)
            Map.has_key?(kws, p) -> kws[p]
            default == :required -> {:undef, p}
            true -> elem(ev(default, st), 0)
          end

        Map.put(acc, p, v)
      end)

    # a macro sees its arguments and the template's top-level names
    top = List.last(st.scopes)
    {out, st2} = exec(b, %{st | scopes: [scope, top]})
    {IO.iodata_to_binary(out), %{st | ns: st2.ns, next: st2.next}}
  end

  defp call({:method, v, name}, pos, kws, st), do: {method(v, name, pos, kws, st), st}

  defp call({:loop_cycle, i}, pos, _kws, st), do: {Enum.at(pos, rem(i, length(pos))), st}
  defp call({:undef, n}, _pos, _kws, _st), do: fail("#{n} is undefined (called)")
  defp call(other, _pos, _kws, st), do: fail("#{repr(other, st)} is not callable")

  defp range_list(a, b, s) when s > 0, do: if(a >= b, do: [], else: Enum.to_list(a..(b - 1)//s))
  defp range_list(a, b, s) when s < 0, do: if(a <= b, do: [], else: Enum.to_list(a..(b + 1)//s))

  # ----------------------------------------------------------------- access

  @dict_methods ~w(items keys values get)
  @str_methods ~w(startswith endswith split rsplit strip lstrip rstrip upper lower title capitalize replace join find count splitlines isdigit isalpha isspace)

  defp getattr({:undef, n}, name, _st), do: fail("#{n} is undefined (attribute #{name})")
  defp getattr({:ns, ref}, name, st), do: dict_get(Map.fetch!(st.ns, ref), name)

  defp getattr({:loop, m}, name, _st) do
    case name do
      "cycle" -> {:loop_cycle, m["index0"]}
      _ -> Map.get(m, name, {:undef, "loop." <> name})
    end
  end

  defp getattr({:dict, _} = d, name, _st) do
    if name in @dict_methods, do: {:method, d, name}, else: dict_get(d, name)
  end

  defp getattr(s, name, _st) when is_binary(s) and name in @str_methods, do: {:method, s, name}
  defp getattr(_v, name, _st), do: {:undef, name}

  defp getitem({:undef, n}, k, st), do: fail("#{n} is undefined (item #{repr(k, st)})")
  defp getitem({:dict, _} = d, k, _st), do: dict_get(d, k)
  defp getitem({:ns, ref}, k, st), do: dict_get(Map.fetch!(st.ns, ref), k)
  defp getitem({:loop, m}, k, _st), do: Map.get(m, k, {:undef, k})

  defp getitem(l, i, _st) when is_list(l) and is_integer(i) do
    n = length(l)
    j = if i < 0, do: n + i, else: i
    if j >= 0 and j < n, do: Enum.at(l, j), else: {:undef, "index #{i}"}
  end

  defp getitem(s, i, _st) when is_binary(s) and is_integer(i) do
    cs = String.codepoints(s)
    n = length(cs)
    j = if i < 0, do: n + i, else: i
    if j >= 0 and j < n, do: Enum.at(cs, j), else: {:undef, "index #{i}"}
  end

  defp getitem(v, k, st) when is_binary(k), do: getattr(v, k, st)
  defp getitem(_v, k, st), do: {:undef, repr(k, st)}

  defp dict_get({:dict, pairs}, k) do
    case List.keyfind(pairs, k, 0) do
      {_, v} -> v
      nil -> {:undef, if(is_binary(k), do: k, else: inspect(k))}
    end
  end

  defp dict_put({:dict, pairs}, k, v) do
    if List.keymember?(pairs, k, 0), do: {:dict, List.keyreplace(pairs, k, 0, {k, v})}, else: {:dict, pairs ++ [{k, v}]}
  end

  defp slice(v, a, b, c) do
    {items, wrap} = if is_binary(v), do: {String.codepoints(v), &Enum.join/1}, else: {iterate(v, nil), & &1}
    n = length(items)
    step = c || 1
    if step == 0, do: fail("slice step cannot be zero")
    norm = fn x, d -> cond do
      x == nil -> d
      x < 0 -> max(n + x, if(step > 0, do: 0, else: -1))
      true -> min(x, if(step > 0, do: n, else: n - 1))
    end end

    {start, stop} = if step > 0, do: {norm.(a, 0), norm.(b, n)}, else: {norm.(a, n - 1), norm.(b, -1)}
    tup = List.to_tuple(items)
    idx = if step > 0, do: Stream.iterate(start, &(&1 + step)) |> Enum.take_while(&(&1 < stop)), else: Stream.iterate(start, &(&1 + step)) |> Enum.take_while(&(&1 > stop))
    wrap.(Enum.map(idx, &elem(tup, &1)))
  end

  # ---------------------------------------------------------------- methods

  defp method({:dict, pairs}, "items", [], _kws, _st), do: Enum.map(pairs, fn {k, v} -> [k, v] end)
  defp method({:dict, pairs}, "keys", [], _kws, _st), do: Enum.map(pairs, &elem(&1, 0))
  defp method({:dict, pairs}, "values", [], _kws, _st), do: Enum.map(pairs, &elem(&1, 1))

  defp method({:dict, _} = d, "get", [k | rest], _kws, _st) do
    case dict_get(d, k) do
      {:undef, _} -> List.first(rest)
      v -> v
    end
  end

  defp method(s, "startswith", [p | _], _kws, _st), do: Enum.any?(List.wrap(p), &String.starts_with?(s, &1))
  defp method(s, "endswith", [p | _], _kws, _st), do: Enum.any?(List.wrap(p), &String.ends_with?(s, &1))
  defp method(s, "split", args, kws, _st), do: py_split(s, Enum.at(args, 0, kws["sep"]), Enum.at(args, 1, kws["maxsplit"] || -1))
  defp method(s, "rsplit", args, _kws, _st), do: py_rsplit(s, Enum.at(args, 0), Enum.at(args, 1, -1))
  defp method(s, "strip", args, _kws, _st), do: py_strip(s, Enum.at(args, 0), :both)
  defp method(s, "lstrip", args, _kws, _st), do: py_strip(s, Enum.at(args, 0), :leading)
  defp method(s, "rstrip", args, _kws, _st), do: py_strip(s, Enum.at(args, 0), :trailing)
  defp method(s, "upper", [], _kws, _st), do: String.upcase(s)
  defp method(s, "lower", [], _kws, _st), do: String.downcase(s)
  defp method(s, "title", [], _kws, _st), do: s |> String.split(~r/(?<=\s)|(?=\s)/u) |> Enum.map_join(&String.capitalize/1)
  defp method(s, "capitalize", [], _kws, _st), do: String.capitalize(s)
  defp method(s, "replace", [a, b | rest], _kws, _st), do: py_replace(s, a, b, Enum.at(rest, 0, -1))
  defp method(s, "join", [items], _kws, st), do: items |> iterate(st) |> Enum.map_join(s, &to_s(&1, st))
  defp method(s, "find", [sub | _], _kws, _st), do: (case :binary.match(s, sub) do {at, _} -> String.length(binary_part(s, 0, at)); :nomatch -> -1 end)
  defp method(s, "count", [sub | _], _kws, _st), do: length(:binary.matches(s, sub))
  defp method(s, "splitlines", [], _kws, _st), do: String.split(s, ~r/\r\n|\n|\r/) |> drop_last_empty()
  defp method(s, "isdigit", [], _kws, _st), do: s != "" and String.match?(s, ~r/\A\d+\z/u)
  defp method(s, "isalpha", [], _kws, _st), do: s != "" and String.match?(s, ~r/\A\p{L}+\z/u)
  defp method(s, "isspace", [], _kws, _st), do: s != "" and String.match?(s, ~r/\A\s+\z/u)
  defp method(v, name, _args, _kws, st), do: fail("unsupported method #{name} on #{repr(v, st)}")

  defp drop_last_empty(l), do: if(List.last(l) == "", do: Enum.drop(l, -1), else: l)

  defp py_split(s, nil, max) do
    parts = String.split(s, ~r/\s+/u, trim: true)
    if max < 0 or length(parts) <= max + 1, do: parts, else: Enum.take(parts, max) ++ [s |> String.trim_leading() |> nth_rest(max)]
  end

  defp py_split(s, sep, max) do
    if sep == "", do: fail("empty separator")
    if max < 0, do: :binary.split(s, sep, [:global]), else: split_n(s, sep, max)
  end

  defp nth_rest(s, n), do: s |> String.split(~r/\s+/u, parts: n + 1) |> List.last()

  defp split_n(s, _sep, 0), do: [s]

  defp split_n(s, sep, n) do
    case :binary.split(s, sep) do
      [a, b] -> [a | split_n(b, sep, n - 1)]
      [a] -> [a]
    end
  end

  defp py_rsplit(s, sep, -1), do: py_split(s, sep, -1)
  defp py_rsplit(s, sep, n), do: s |> String.reverse() |> py_split(String.reverse(sep), n) |> Enum.map(&String.reverse/1) |> Enum.reverse()

  defp py_strip(s, nil, :both), do: String.trim(s)
  defp py_strip(s, nil, :leading), do: String.trim_leading(s)
  defp py_strip(s, nil, :trailing), do: String.trim_trailing(s)

  defp py_strip(s, chars, side) do
    set = String.codepoints(chars)
    cs = String.codepoints(s)
    cs = if side in [:both, :leading], do: Enum.drop_while(cs, &(&1 in set)), else: cs
    cs = if side in [:both, :trailing], do: cs |> Enum.reverse() |> Enum.drop_while(&(&1 in set)) |> Enum.reverse(), else: cs
    Enum.join(cs)
  end

  defp py_replace(s, a, b, -1), do: String.replace(s, a, b)
  defp py_replace(s, a, b, n), do: Enum.reduce(1..max(n, 0)//1, s, fn _, acc -> String.replace(acc, a, b, global: false) end)

  # ---------------------------------------------------------------- filters

  defp filt(name, v, pos, kws, st) do
    case name do
      "tojson" -> tojson(v, Enum.at(pos, 0, kws["indent"]), kws["separators"], kws["sort_keys"] == true, kws["ensure_ascii"] == true, st)
      n when n in ["length", "count"] -> len(v, st)
      "trim" -> py_strip(to_s(v, st), Enum.at(pos, 0, kws["chars"]), :both)
      "items" -> (case v do {:undef, _} -> []; _ -> method(v, "items", [], %{}, st) end)
      "join" -> join(v, Enum.at(pos, 0, kws["d"] || ""), kws["attribute"], st)
      "string" -> to_s(v, st)
      "indent" -> indent(to_s(v, st), Enum.at(pos, 0, kws["width"] || 4), Enum.at(pos, 1, kws["first"] || false), Enum.at(pos, 2, kws["blank"] || false))
      "list" -> iterate(v, st)
      "selectattr" -> select_by(v, pos, st, true, true)
      "rejectattr" -> select_by(v, pos, st, true, false)
      "select" -> select_by(v, pos, st, false, true)
      "reject" -> select_by(v, pos, st, false, false)
      "replace" -> py_replace(to_s(v, st), Enum.at(pos, 0), Enum.at(pos, 1), Enum.at(pos, 2, -1))
      "unique" -> Enum.uniq_by(iterate(v, st), &key_of(&1))
      "map" -> map_filter(v, pos, kws, st)
      "lower" -> String.downcase(to_s(v, st))
      "upper" -> String.upcase(to_s(v, st))
      "capitalize" -> String.capitalize(to_s(v, st))
      "title" -> method(to_s(v, st), "title", [], %{}, st)
      n when n in ["default", "d"] ->
        dflt = Enum.at(pos, 0, kws["default_value"] || "")
        bool = Enum.at(pos, 1, kws["boolean"] || false)
        cond do
          match?({:undef, _}, v) -> dflt
          bool and not truthy?(v, st) -> dflt
          true -> v
        end
      "first" -> v |> iterate(st) |> List.first({:undef, "first"})
      "last" -> v |> iterate(st) |> List.last() || {:undef, "last"}
      "safe" -> v
      "int" -> to_int(v)
      "float" -> to_float(v)
      "abs" -> abs(v)
      "sort" -> sort_filter(v, pos, kws, st)
      "dictsort" -> v |> method("items", [], %{}, st) |> Enum.sort_by(&key_of(hd(&1)))
      "wordcount" -> v |> to_s(st) |> String.split(~r/\s+/u, trim: true) |> length()
      "round" -> Float.round(v * 1.0, Enum.at(pos, 0, 0))
      "batch" -> v |> iterate(st) |> Enum.chunk_every(Enum.at(pos, 0))
      "center" -> to_s(v, st)
      other -> fail("unsupported filter #{other}")
    end
  end

  defp len({:undef, _}, _st), do: 0
  defp len(s, _st) when is_binary(s), do: String.length(s)
  defp len({:dict, p}, _st), do: length(p)
  defp len({:ns, ref}, st), do: len(st.ns[ref], st)
  defp len(l, _st) when is_list(l), do: length(l)
  defp len(v, st), do: fail("object of type #{type_name(v)} has no len(): #{repr(v, st)}")

  defp join(v, sep, nil, st), do: v |> iterate(st) |> Enum.map_join(sep, &to_s(&1, st))
  defp join(v, sep, attr, st), do: v |> iterate(st) |> Enum.map_join(sep, &to_s(getattr(&1, attr, st), st))

  defp indent(s, width, first, blank) do
    pad = if is_binary(width), do: width, else: String.duplicate(" ", width)
    lines = String.split(s, "\n")

    {head, tail} = {hd(lines), tl(lines)}
    tail = Enum.map(tail, fn l -> if l == "" and not blank, do: l, else: pad <> l end)
    head = if first and (head != "" or blank), do: pad <> head, else: head
    Enum.join([head | tail], "\n")
  end

  defp select_by(v, pos, st, attr?, keep) do
    items = iterate(v, st)

    {get, test_name, args} =
      if attr? do
        [a | rest] = pos
        {&getattr(&1, a, st), List.first(rest), Enum.drop(rest, 1)}
      else
        {& &1, List.first(pos), Enum.drop(pos, 1)}
      end

    Enum.filter(items, fn it ->
      x = get.(it)
      r = if test_name, do: test(test_name, x, args, st), else: truthy?(x, st)
      r == keep
    end)
  end

  defp map_filter(v, pos, kws, st) do
    items = iterate(v, st)

    case {pos, kws} do
      {_, %{"attribute" => a}} -> Enum.map(items, fn it -> case getattr(it, a, st) do
          {:undef, _} -> Map.get(kws, "default", {:undef, a})
          x -> x
        end end)
      {[f | args], _} -> Enum.map(items, &filt(f, &1, args, %{}, st))
    end
  end

  defp sort_filter(v, pos, kws, st) do
    rev = Enum.at(pos, 0, kws["reverse"] || false)
    attr = kws["attribute"]
    key = if attr, do: &key_of(getattr(&1, attr, st)), else: &key_of/1
    sorted = Enum.sort_by(iterate(v, st), key)
    if rev, do: Enum.reverse(sorted), else: sorted
  end

  defp key_of(s) when is_binary(s), do: {1, s}
  defp key_of(n) when is_number(n), do: {0, n}
  defp key_of(v), do: {2, inspect(v)}

  defp to_int(v) when is_integer(v), do: v
  defp to_int(v) when is_float(v), do: trunc(v)
  defp to_int(v) when is_binary(v), do: (case Integer.parse(String.trim(v)) do {n, ""} -> n; _ -> 0 end)
  defp to_int(_), do: 0

  defp to_float(v) when is_number(v), do: v * 1.0
  defp to_float(v) when is_binary(v), do: (case Float.parse(String.trim(v)) do {f, ""} -> f; _ -> 0.0 end)
  defp to_float(_), do: 0.0

  # ------------------------------------------------------------------ tests

  defp test(name, v, args, st) do
    case name do
      "defined" -> not match?({:undef, _}, v)
      "undefined" -> match?({:undef, _}, v)
      "none" -> v == nil
      "string" -> is_binary(v)
      "mapping" -> match?({:dict, _}, v) or match?({:ns, _}, v)
      "iterable" -> is_binary(v) or is_list(v) or match?({:dict, _}, v)
      "sequence" -> is_binary(v) or is_list(v) or match?({:dict, _}, v)
      "number" -> is_number(v)
      "integer" -> is_integer(v)
      "float" -> is_float(v)
      "boolean" -> is_boolean(v)
      "true" -> v === true
      "false" -> v === false
      "callable" -> match?({:macro, _, _, _}, v) or match?({:builtin, _}, v) or match?({:method, _, _}, v)
      n when n in ["equalto", "eq", "=="] -> py_eq(v, hd(args))
      n when n in ["ne", "!="] -> not py_eq(v, hd(args))
      "sameas" -> v === hd(args)
      "in" -> cmp("in", v, hd(args), st)
      n when n in ["gt", ">"] -> cmp(">", v, hd(args), st)
      n when n in ["ge", ">="] -> cmp(">=", v, hd(args), st)
      n when n in ["lt", "<"] -> cmp("<", v, hd(args), st)
      n when n in ["le", "<="] -> cmp("<=", v, hd(args), st)
      "odd" -> is_integer(v) and rem(v, 2) != 0
      "even" -> is_integer(v) and rem(v, 2) == 0
      "divisibleby" -> rem(v, hd(args)) == 0
      "lower" -> is_binary(v) and String.downcase(v) == v
      "upper" -> is_binary(v) and String.upcase(v) == v
      other -> fail("unsupported test #{other}")
    end
  end

  # -------------------------------------------------------------- operators

  defp binop("+", a, b, _st) when is_number(a) and is_number(b), do: a + b
  defp binop("+", a, b, _st) when is_binary(a) and is_binary(b), do: a <> b
  defp binop("+", a, b, _st) when is_list(a) and is_list(b), do: a ++ b
  defp binop("-", a, b, _st) when is_number(a) and is_number(b), do: a - b
  defp binop("*", a, b, _st) when is_number(a) and is_number(b), do: a * b
  defp binop("*", a, n, _st) when is_binary(a) and is_integer(n), do: String.duplicate(a, max(n, 0))
  defp binop("*", a, n, _st) when is_list(a) and is_integer(n), do: List.duplicate(a, max(n, 0)) |> Enum.concat()
  defp binop("/", a, b, _st) when is_number(a) and is_number(b), do: a / b
  defp binop("//", a, b, _st) when is_integer(a) and is_integer(b), do: Integer.floor_div(a, b)
  defp binop("//", a, b, _st) when is_number(a) and is_number(b), do: Float.floor(a / b)
  defp binop("%", a, b, _st) when is_integer(a) and is_integer(b), do: Integer.mod(a, b)
  defp binop("%", a, b, st) when is_binary(a), do: percent_format(a, List.wrap(b), st)
  defp binop("**", a, b, _st) when is_integer(a) and is_integer(b) and b >= 0, do: Integer.pow(a, b)
  defp binop("**", a, b, _st) when is_number(a) and is_number(b), do: :math.pow(a, b)
  defp binop("~", a, b, st), do: to_s(a, st) <> to_s(b, st)
  defp binop(op, a, b, st), do: fail("unsupported operand types for #{op}: #{repr(a, st)} and #{repr(b, st)}")

  defp percent_format(fmt, args, st) do
    {out, _} =
      Regex.split(~r/%[sdr%]/, fmt, include_captures: true)
      |> Enum.map_reduce(args, fn
        "%%", rest -> {"%", rest}
        "%s", [x | rest] -> {to_s(x, st), rest}
        "%d", [x | rest] -> {Integer.to_string(to_int(x)), rest}
        "%r", [x | rest] -> {repr(x, st), rest}
        piece, rest -> {piece, rest}
      end)

    Enum.join(out)
  end

  defp cmp("==", a, b, _st), do: py_eq(a, b)
  defp cmp("!=", a, b, _st), do: not py_eq(a, b)
  defp cmp("in", a, b, st), do: member?(a, b, st)
  defp cmp("notin", a, b, st), do: not member?(a, b, st)
  defp cmp(op, a, b, _st) when (is_number(a) and is_number(b)) or (is_binary(a) and is_binary(b)), do: order(op, a, b)
  defp cmp(op, a, b, st), do: fail("cannot compare #{repr(a, st)} #{op} #{repr(b, st)}")

  defp order("<", a, b), do: a < b
  defp order(">", a, b), do: a > b
  defp order("<=", a, b), do: a <= b
  defp order(">=", a, b), do: a >= b

  defp member?(a, b, _st) when is_binary(a) and is_binary(b), do: String.contains?(b, a)
  defp member?(a, b, _st) when is_list(b), do: Enum.any?(b, &py_eq(a, &1))
  defp member?(a, {:dict, pairs}, _st), do: Enum.any?(pairs, fn {k, _} -> py_eq(a, k) end)
  defp member?(a, {:ns, ref}, st), do: member?(a, st.ns[ref], st)
  defp member?(_a, {:undef, _}, _st), do: false
  defp member?(a, b, st), do: fail("argument of type #{type_name(b)} is not iterable: #{repr(a, st)} in #{repr(b, st)}")

  defp py_eq(a, b) when is_number(a) and is_number(b), do: a == b
  defp py_eq({:undef, _}, {:undef, _}), do: true
  defp py_eq({:dict, a}, {:dict, b}), do: length(a) == length(b) and Enum.all?(a, fn {k, v} -> case List.keyfind(b, k, 0) do {_, w} -> py_eq(v, w); nil -> false end end)
  defp py_eq(a, b) when is_list(a) and is_list(b), do: length(a) == length(b) and Enum.all?(Enum.zip(a, b), fn {x, y} -> py_eq(x, y) end)
  defp py_eq(a, b), do: a === b

  # ----------------------------------------------------------- python values

  defp truthy?({:undef, _}, _st), do: false
  defp truthy?(nil, _st), do: false
  defp truthy?(false, _st), do: false
  defp truthy?(0, _st), do: false
  defp truthy?(x, _st) when is_float(x), do: x != 0.0
  defp truthy?("", _st), do: false
  defp truthy?([], _st), do: false
  defp truthy?({:dict, []}, _st), do: false
  defp truthy?(_, _st), do: true

  defp iterate({:undef, _}, _st), do: []
  defp iterate(l, _st) when is_list(l), do: l
  defp iterate({:dict, pairs}, _st), do: Enum.map(pairs, &elem(&1, 0))
  defp iterate(s, _st) when is_binary(s), do: String.codepoints(s)
  defp iterate(nil, _st), do: fail("'NoneType' object is not iterable")
  defp iterate(v, st), do: fail("#{repr(v, st)} is not iterable")

  defp type_name(v) when is_binary(v), do: "str"
  defp type_name(v) when is_integer(v), do: "int"
  defp type_name(v) when is_float(v), do: "float"
  defp type_name(v) when is_list(v), do: "list"
  defp type_name(nil), do: "NoneType"
  defp type_name(v) when is_boolean(v), do: "bool"
  defp type_name({:dict, _}), do: "dict"
  defp type_name(_), do: "object"

  @doc false
  # Python's str()
  def to_s(v, st \\ nil)
  def to_s({:undef, _}, _st), do: ""
  def to_s(s, _st) when is_binary(s), do: s
  def to_s(v, st), do: repr(v, st)

  @doc false
  # Python's repr()
  def repr(v, st \\ nil)
  def repr(nil, _), do: "None"
  def repr(true, _), do: "True"
  def repr(false, _), do: "False"
  def repr(n, _) when is_integer(n), do: Integer.to_string(n)
  def repr(f, _) when is_float(f), do: py_float(f)
  def repr(s, _) when is_binary(s), do: py_str_repr(s)
  def repr(l, st) when is_list(l), do: "[" <> Enum.map_join(l, ", ", &repr(&1, st)) <> "]"
  def repr({:dict, pairs}, st), do: "{" <> Enum.map_join(pairs, ", ", fn {k, v} -> repr(k, st) <> ": " <> repr(v, st) end) <> "}"
  def repr({:ns, ref}, st) when is_map(st), do: "<Namespace " <> repr(st.ns[ref], st) <> ">"
  def repr({:undef, _}, _), do: ""
  def repr({:macro, n, _, _}, _), do: "<Macro '#{n}'>"
  def repr(other, _), do: inspect(other)

  @doc false
  # repr(float): the shortest digits that round-trip, Python's layout
  def py_float(f) do
    s = :erlang.float_to_binary(f, [:short])
    {sign, s} = if String.starts_with?(s, "-"), do: {"-", binary_part(s, 1, byte_size(s) - 1)}, else: {"", s}
    {mant, exp} = case String.split(s, "e") do
      [m, e] -> {m, String.to_integer(e)}
      [m] -> {m, 0}
    end

    [ip, fp] = String.split(mant, ".")
    digits = String.trim_leading(ip <> fp, "0")
    lead_zeros = byte_size(ip <> fp) - byte_size(String.trim_leading(ip <> fp, "0"))
    digits = String.trim_trailing(digits, "0")
    digits = if digits == "", do: "0", else: digits
    # value = 0.d1d2… × 10^(point)
    point = byte_size(ip) - lead_zeros + exp

    body =
      cond do
        digits == "0" -> "0.0"
        point - 1 < -4 or point - 1 >= 16 ->
          e = point - 1
          m = if byte_size(digits) == 1, do: digits, else: binary_part(digits, 0, 1) <> "." <> binary_part(digits, 1, byte_size(digits) - 1)
          m <> "e" <> (if e < 0, do: "-", else: "+") <> String.pad_leading(Integer.to_string(abs(e)), 2, "0")
        point <= 0 -> "0." <> String.duplicate("0", -point) <> digits
        point >= byte_size(digits) -> digits <> String.duplicate("0", point - byte_size(digits)) <> ".0"
        true -> binary_part(digits, 0, point) <> "." <> binary_part(digits, point, byte_size(digits) - point)
      end

    sign <> body
  end

  defp py_str_repr(s) do
    q = if String.contains?(s, "'") and not String.contains?(s, "\""), do: "\"", else: "'"

    body =
      s
      |> String.codepoints()
      |> Enum.map_join(fn
        "\\" -> "\\\\"
        "\n" -> "\\n"
        "\r" -> "\\r"
        "\t" -> "\\t"
        ^q -> "\\" <> q
        <<c::utf8>> when c < 0x20 or c in 0x7F..0xA0 -> "\\x" <> String.pad_leading(String.downcase(Integer.to_string(c, 16)), 2, "0")
        ch -> ch
      end)

    q <> body <> q
  end

  # json.dumps(x, ensure_ascii=False, indent=…, separators=…, sort_keys=…)
  defp tojson(v, indent, separators, sort_keys, ensure_ascii, st) do
    {item_sep, key_sep} =
      case separators do
        [a, b] -> {a, b}
        nil -> if indent == nil, do: {", ", ": "}, else: {",", ": "}
      end

    indent = if is_integer(indent), do: String.duplicate(" ", indent), else: indent
    IO.iodata_to_binary(json(v, indent, 0, item_sep, key_sep, sort_keys, ensure_ascii, st))
  end

  defp json(nil, _, _, _, _, _, _, _), do: "null"
  defp json(true, _, _, _, _, _, _, _), do: "true"
  defp json(false, _, _, _, _, _, _, _), do: "false"
  defp json(n, _, _, _, _, _, _, _) when is_integer(n), do: Integer.to_string(n)
  defp json(f, _, _, _, _, _, _, _) when is_float(f), do: py_float(f)
  defp json(s, _, _, _, _, _, ea, _) when is_binary(s), do: json_str(s, ea)
  defp json({:ns, ref}, i, d, is, ks, sk, ea, st), do: json(st.ns[ref], i, d, is, ks, sk, ea, st)
  defp json({:undef, n}, _, _, _, _, _, _, _), do: fail("Object of type Undefined is not JSON serializable (#{n})")

  defp json(l, i, d, is, ks, sk, ea, st) when is_list(l) do
    if l == [], do: "[]", else: container("[", "]", Enum.map(l, &json(&1, i, d + 1, is, ks, sk, ea, st)), i, d, is)
  end

  defp json({:dict, pairs}, i, d, is, ks, sk, ea, st) do
    pairs = if sk, do: Enum.sort_by(pairs, &to_s(elem(&1, 0), st)), else: pairs

    if pairs == [] do
      "{}"
    else
      items = Enum.map(pairs, fn {k, v} -> [json_str(json_key(k, st), ea), ks, json(v, i, d + 1, is, ks, sk, ea, st)] end)
      container("{", "}", items, i, d, is)
    end
  end

  defp json(other, _, _, _, _, _, _, st), do: fail("Object is not JSON serializable: #{repr(other, st)}")

  defp json_key(k, _st) when is_binary(k), do: k
  defp json_key(k, st), do: if(is_float(k), do: py_float(k), else: (case k do nil -> "null"; true -> "true"; false -> "false"; _ -> to_s(k, st) end))

  defp container(open, close, items, nil, _d, is), do: [open, Enum.intersperse(items, is), close]

  defp container(open, close, items, ind, d, is) do
    nl = "\n" <> String.duplicate(ind, d + 1)
    [open, nl, Enum.intersperse(items, [is, nl]), "\n", String.duplicate(ind, d), close]
  end

  defp json_str(s, ensure_ascii) do
    body =
      for <<c::utf8 <- s>>, into: "" do
        case c do
          ?" -> "\\\""
          ?\\ -> "\\\\"
          ?\n -> "\\n"
          ?\r -> "\\r"
          ?\t -> "\\t"
          ?\b -> "\\b"
          ?\f -> "\\f"
          c when c < 0x20 -> "\\u" <> String.pad_leading(String.downcase(Integer.to_string(c, 16)), 4, "0")
          c when ensure_ascii and c > 0x7E and c < 0x10000 -> "\\u" <> String.pad_leading(String.downcase(Integer.to_string(c, 16)), 4, "0")
          c when ensure_ascii and c >= 0x10000 ->
            v = c - 0x10000
            hi = 0xD800 + (v >>> 10)
            lo = 0xDC00 + (v &&& 0x3FF)
            "\\u" <> String.downcase(Integer.to_string(hi, 16)) <> "\\u" <> String.downcase(Integer.to_string(lo, 16))
          c -> <<c::utf8>>
        end
      end

    "\"" <> body <> "\""
  end

  # ------------------------------------------------------------- strftime

  @months ~w(January February March April May June July August September October November December)
  @days ~w(Monday Tuesday Wednesday Thursday Friday Saturday Sunday)

  defp strftime(now, fmt) do
    d = if is_struct(now, DateTime), do: DateTime.to_naive(now), else: now
    dow = Date.day_of_week(NaiveDateTime.to_date(d))
    p2 = &String.pad_leading(Integer.to_string(&1), 2, "0")

    Regex.replace(~r/%[a-zA-Z%]/, fmt, fn
      "%d", _ -> p2.(d.day)
      "%m", _ -> p2.(d.month)
      "%Y", _ -> Integer.to_string(d.year)
      "%y", _ -> p2.(rem(d.year, 100))
      "%B", _ -> Enum.at(@months, d.month - 1)
      "%b", _ -> String.slice(Enum.at(@months, d.month - 1), 0, 3)
      "%A", _ -> Enum.at(@days, dow - 1)
      "%a", _ -> String.slice(Enum.at(@days, dow - 1), 0, 3)
      "%H", _ -> p2.(d.hour)
      "%M", _ -> p2.(d.minute)
      "%S", _ -> p2.(d.second)
      "%j", _ -> String.pad_leading(Integer.to_string(Date.day_of_year(NaiveDateTime.to_date(d))), 3, "0")
      "%%", _ -> "%"
      other, _ -> fail("unsupported strftime directive #{other}")
    end)
  end
end
