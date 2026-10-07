defmodule Vapor.Alembic.Parser do
  @moduledoc """
  Alembic's parser: tokens → a tree of tuples with fixed atoms
  (docs/ALEMBIC.md §2). A program is a sequence of definitions,

      name = expression
      name(a, b) = expression

  separated by newlines or `;`. Precedence, loosest first: lambda
  (`x => e`, `(a, b) => e`), `if/then/else` and `let … in`, `|>`, `or`,
  `and`, `not`, comparisons (chainable: `0 <= i < n`; also `in`,
  `not in`), `|`, `xor`, `&`, shifts, `..` ranges, `+ - ++`, `* / // %`,
  unary `-` and `~`, `^` (right-associative, so `-x^2 = -(x^2)`), then
  calls, indexing `xs[i]`, slicing `xs[a:b]` and fields `s.key`.
  """
  alias Vapor.Alembic.Lexer

  @max_depth 400

  @doc "`{:ok, [def]}` or `{:error, %{message, line, col}}`."
  def program(text) do
    with {:ok, toks} <- lex(text) do
      try do
        Process.put(:alembic_depth, 0)
        {defs, _} = defs(toks, [])
        {:ok, defs}
      catch
        {:parse, msg, l, c} -> {:error, %{message: msg, line: l, col: c}}
      end
    end
  end

  @doc "A single expression: `{:ok, ast}` or `{:error, …}`."
  def expression(text) do
    with {:ok, toks} <- lex(text) do
      try do
        Process.put(:alembic_depth, 0)
        toks = Enum.reject(toks, &match?({:nl, _, _, _}, &1))
        case expr(toks) do
          {e, [{:eof, _, _, _}]} -> {:ok, e}
          {_, [t | _]} -> throw(unexpected(t))
        end
      catch
        {:parse, msg, l, c} -> {:error, %{message: msg, line: l, col: c}}
      end
    end
  end

  defp lex(text) do
    case Lexer.lex(text) do
      {:ok, toks} -> {:ok, toks}
      {:error, m, l, c} -> {:error, %{message: m, line: l, col: c}}
    end
  end

  # ------------------------------------------------------------ definitions

  defp defs([{:eof, _, _, _}], acc), do: {Enum.reverse(acc), []}
  defp defs([{k, _, _, _} | rest], acc) when k in [:nl, :semicolon], do: defs(rest, acc)

  defp defs([{:id, name, l, c}, {:op, "=", _, _} | rest], acc) do
    {e, rest} = expr(rest)
    defs(end_def(rest), [{:def, name, nil, e, {l, c}} | acc])
  end

  defp defs([{:id, name, l, c}, {:lparen, _, _, _} | rest] = toks, acc) do
    case params(rest, []) do
      {:ok, ps, [{:op, "=", _, _} | rest2]} ->
        {e, rest3} = expr(rest2)
        defs(end_def(rest3), [{:def, name, ps, e, {l, c}} | acc])

      _ ->
        throw(perr("a definition is `name = …` or `name(a, b) = …`", toks))
    end
  end

  defp defs(toks, _acc), do: throw(perr("expected a definition (`name = …` or `name(a, b) = …`)", toks))

  defp end_def([{k, _, _, _} | rest]) when k in [:nl, :semicolon], do: rest
  defp end_def([{:eof, _, _, _}] = t), do: t
  defp end_def([t | _]), do: throw(unexpected(t, "end of the definition"))

  defp params([{:rparen, _, _, _} | rest], acc), do: {:ok, Enum.reverse(acc), rest}
  defp params([{:id, n, _, _}, {:comma, _, _, _} | rest], acc), do: params(rest, [n | acc])
  defp params([{:id, n, _, _}, {:rparen, _, _, _} | rest], acc), do: {:ok, Enum.reverse([n | acc]), rest}
  defp params(_, _), do: :error

  # ------------------------------------------------------------ expressions

  def expr(toks) do
    d = Process.get(:alembic_depth, 0)
    if d > @max_depth, do: throw(perr("expression nested deeper than #{@max_depth}", toks))
    Process.put(:alembic_depth, d + 1)
    r = lambda_or(toks)
    Process.put(:alembic_depth, d)
    r
  end

  defp lambda_or([{:id, n, l, c}, {:op, "=>", _, _} | rest]) do
    {body, rest} = expr(rest)
    {{:lambda, [{:pvar, n}], body, {l, c}}, rest}
  end

  defp lambda_or([{:lparen, _, l, c} | _] = toks) do
    case lambda_params(toks) do
      {:ok, ps, rest} ->
        {body, rest} = expr(rest)
        {{:lambda, ps, body, {l, c}}, rest}

      :no ->
        control(toks)
    end
  end

  defp lambda_or(toks), do: control(toks)

  # `( pattern, … ) =>` — patterns are names, `_`, tuples and lists of patterns
  defp lambda_params([{:lparen, _, _, _}, {:rparen, _, _, _}, {:op, "=>", _, _} | rest]), do: {:ok, [], rest}

  defp lambda_params([{:lparen, _, _, _} | rest] = toks) do
    case close_index(rest, 0, 0) do
      nil -> :no
      i ->
        case Enum.drop(rest, i + 1) do
          [{:op, "=>", _, _} | after_arrow] ->
            inner = Enum.take(rest, i)
            {ps, []} = pattern_list(inner ++ [{:eof, nil, 0, 0}], [])
            _ = toks
            {:ok, ps, after_arrow}

          _ -> :no
        end
    end
  end

  defp close_index([], _d, _i), do: nil
  defp close_index([{:rparen, _, _, _} | _], 0, i), do: i
  defp close_index([{k, _, _, _} | rest], d, i) when k in [:lparen, :lbracket, :lbrace], do: close_index(rest, d + 1, i + 1)
  defp close_index([{k, _, _, _} | rest], d, i) when k in [:rparen, :rbracket, :rbrace], do: close_index(rest, d - 1, i + 1)
  defp close_index([{:eof, _, _, _} | _], _d, _i), do: nil
  defp close_index([_ | rest], d, i), do: close_index(rest, d, i + 1)

  defp pattern_list([{:eof, _, _, _}], acc), do: {Enum.reverse(acc), []}

  defp pattern_list(toks, acc) do
    {p, rest} = pattern(toks)
    case rest do
      [{:comma, _, _, _} | r] -> pattern_list(r, [p | acc])
      [{:eof, _, _, _}] -> {Enum.reverse([p | acc]), []}
      [t | _] -> throw(unexpected(t, "a pattern"))
    end
  end

  @doc false
  def pattern([{:id, "_", _, _} | rest]), do: {:pany, rest}
  def pattern([{:id, n, _, _} | rest]), do: {{:pvar, n}, rest}

  def pattern([{:lparen, _, _, _} | rest]) do
    {ps, rest} = pattern_seq(rest, :rparen, [])
    {{:ptuple, ps}, rest}
  end

  def pattern([{:lbracket, _, _, _} | rest]) do
    {ps, rest} = pattern_seq(rest, :rbracket, [])
    {{:plist, ps}, rest}
  end

  def pattern([t | _]), do: throw(unexpected(t, "a name or a (tuple, of, names)"))

  defp pattern_seq([{close, _, _, _} | rest], close, acc), do: {Enum.reverse(acc), rest}

  defp pattern_seq(toks, close, acc) do
    {p, rest} = pattern(toks)
    case rest do
      [{:comma, _, _, _} | r] -> pattern_seq(r, close, [p | acc])
      [{^close, _, _, _} | r] -> {Enum.reverse([p | acc]), r}
      [t | _] -> throw(unexpected(t))
    end
  end

  defp control([{:kw, "if", l, c} | rest]) do
    {cnd, rest} = expr(rest)
    rest = expect(rest, :kw, "then")
    {a, rest} = expr(rest)
    rest = expect(rest, :kw, "else")
    {b, rest} = expr(rest)
    {{:if, cnd, a, b, {l, c}}, rest}
  end

  defp control([{:kw, "let", l, c} | rest]) do
    {binds, rest} = let_binds(rest, [])
    {body, rest} = expr(rest)
    {{:let, binds, body, {l, c}}, rest}
  end

  defp control(toks), do: pipe(toks)

  defp let_binds(toks, acc) do
    {p, rest} = pattern(toks)
    rest = expect(rest, :op, "=")
    {e, rest} = expr_no_in(rest)
    case rest do
      [{:comma, _, _, _} | r] -> let_binds(r, [{p, e} | acc])
      [{:semicolon, _, _, _} | r] -> let_binds(r, [{p, e} | acc])
      [{:kw, "in", _, _} | r] -> {Enum.reverse([{p, e} | acc]), r}
      [t | _] -> throw(unexpected(t, "`in` or `,` after a let binding"))
    end
  end

  # inside `let`, `in` closes the binding rather than testing membership
  defp expr_no_in(toks) do
    prev = Process.get(:alembic_no_in, false)
    Process.put(:alembic_no_in, true)
    r = expr(toks)
    Process.put(:alembic_no_in, prev)
    r
  end

  defp pipe(toks) do
    {a, rest} = disj(toks)
    pipe_rest(a, rest)
  end

  defp pipe_rest(a, [{:op, "|>", l, c} | rest]) do
    {f, rest} = disj(rest)
    call =
      case f do
        {:call, g, args, p} -> {:call, g, [a | args], p}
        g -> {:call, g, [a], {l, c}}
      end
    pipe_rest(call, rest)
  end

  defp pipe_rest(a, rest), do: {a, rest}

  defp disj(toks) do
    {a, rest} = conj(toks)
    disj_rest(a, rest)
  end

  defp disj_rest(a, [{:kw, "or", _, _} | rest]) do
    {b, rest} = conj(rest)
    disj_rest({:or, a, b}, rest)
  end

  defp disj_rest(a, rest), do: {a, rest}

  defp conj(toks) do
    {a, rest} = negation(toks)
    conj_rest(a, rest)
  end

  defp conj_rest(a, [{:kw, "and", _, _} | rest]) do
    {b, rest} = negation(rest)
    conj_rest({:and, a, b}, rest)
  end

  defp conj_rest(a, rest), do: {a, rest}

  defp negation([{:kw, "not", _, _} | rest]) do
    {a, rest} = negation(rest)
    {{:not, a}, rest}
  end

  defp negation(toks), do: comparison(toks)

  @cmp ["==", "!=", "<", "<=", ">", ">="]

  defp comparison(toks) do
    {a, rest} = bor(toks)
    {chain, rest} = cmp_chain(rest, [])
    if chain == [], do: {a, rest}, else: {{:cmp, a, chain, pos(toks)}, rest}
  end

  defp cmp_chain([{:op, op, _, _} | rest], acc) when op in @cmp do
    {b, rest} = bor(rest)
    cmp_chain(rest, [{op, b} | acc])
  end

  defp cmp_chain([{:kw, "in", _, _} | rest] = toks, acc) do
    if Process.get(:alembic_no_in, false) and acc == [] do
      {[], toks}
    else
      {b, rest} = bor(rest)
      cmp_chain(rest, [{"in", b} | acc])
    end
  end

  defp cmp_chain([{:kw, "not", _, _}, {:kw, "in", _, _} | rest], acc) do
    {b, rest} = bor(rest)
    cmp_chain(rest, [{"not in", b} | acc])
  end

  defp cmp_chain(rest, acc), do: {Enum.reverse(acc), rest}

  defp bor(toks), do: left(toks, &bxor/1, fn {:op, "|", _, _} -> "|"; _ -> nil end)
  defp bxor(toks), do: left(toks, &band/1, fn {:kw, "xor", _, _} -> "xor"; _ -> nil end)
  defp band(toks), do: left(toks, &shift/1, fn {:op, "&", _, _} -> "&"; _ -> nil end)
  defp shift(toks), do: left(toks, &range/1, fn {:op, op, _, _} when op in ["<<", ">>"] -> op; _ -> nil end)

  defp range(toks) do
    {a, rest} = additive(toks)
    case rest do
      [{:op, "..", l, c} | r] ->
        {b, r} = additive(r)
        {{:range, a, b, {l, c}}, r}
      _ -> {a, rest}
    end
  end

  defp additive(toks), do: left(toks, &multiplicative/1, fn {:op, op, _, _} when op in ["+", "-", "++"] -> op; _ -> nil end)
  defp multiplicative(toks), do: left(toks, &unary/1, fn {:op, op, _, _} when op in ["*", "/", "//", "%"] -> op; _ -> nil end)

  defp left(toks, next, opf) do
    {a, rest} = next.(toks)
    left_rest(a, rest, next, opf)
  end

  defp left_rest(a, [t | rest] = toks, next, opf) do
    case opf.(t) do
      nil -> {a, toks}
      op ->
        {b, rest} = next.(rest)
        left_rest({:bin, op, a, b, {elem(t, 2), elem(t, 3)}}, rest, next, opf)
    end
  end

  defp unary([{:op, "-", l, c} | rest]) do
    {a, rest} = unary(rest)
    case a do
      {:lit, n} when is_number(n) -> {{:lit, -n}, rest}
      _ -> {{:neg, a, {l, c}}, rest}
    end
  end

  defp unary([{:op, "+", _, _} | rest]), do: unary(rest)

  defp unary([{:op, "~", l, c} | rest]) do
    {a, rest} = unary(rest)
    {{:bnot, a, {l, c}}, rest}
  end

  defp unary(toks), do: power(toks)

  defp power(toks) do
    {a, rest} = postfix(toks)
    case rest do
      [{:op, op, l, c} | r] when op in ["^", "**"] ->
        {b, r} = unary(r)
        {{:bin, "^", a, b, {l, c}}, r}
      _ -> {a, rest}
    end
  end

  defp postfix(toks) do
    {a, rest} = atom(toks)
    postfix_rest(a, rest)
  end

  defp postfix_rest(a, [{:lparen, _, l, c} | rest]) do
    {args, rest} = seq(rest, :rparen, [])
    postfix_rest({:call, a, args, {l, c}}, rest)
  end

  defp postfix_rest(a, [{:lbracket, _, l, c} | rest]) do
    case rest do
      [{:colon, _, _, _} | r] ->
        {hi, r} = slice_end(r)
        postfix_rest({:slice, a, nil, hi, {l, c}}, r)

      _ ->
        {i, r} = expr(rest)
        case r do
          [{:rbracket, _, _, _} | r2] -> postfix_rest({:index, a, i, {l, c}}, r2)
          [{:colon, _, _, _} | r2] ->
            {hi, r3} = slice_end(r2)
            postfix_rest({:slice, a, i, hi, {l, c}}, r3)
          [t | _] -> throw(unexpected(t, "`]`"))
        end
    end
  end

  defp postfix_rest(a, [{:dot, _, l, c}, {:id, f, _, _} | rest]), do: postfix_rest({:field, a, f, {l, c}}, rest)
  defp postfix_rest(a, rest), do: {a, rest}

  defp slice_end([{:rbracket, _, _, _} | r]), do: {nil, r}

  defp slice_end(toks) do
    {hi, r} = expr(toks)
    {hi, expect(r, :rbracket, "]")}
  end

  defp atom([{:int, n, _, _} | rest]), do: {{:lit, n}, rest}
  defp atom([{:float, x, _, _} | rest]), do: {{:lit, x}, rest}
  defp atom([{:str, s, _, _} | rest]), do: {{:lit, s}, rest}
  defp atom([{:kw, "true", _, _} | rest]), do: {{:lit, true}, rest}
  defp atom([{:kw, "false", _, _} | rest]), do: {{:lit, false}, rest}
  defp atom([{:kw, "nil", _, _} | rest]), do: {{:lit, nil}, rest}
  defp atom([{:id, n, l, c} | rest]), do: {{:var, n, {l, c}}, rest}

  defp atom([{:lparen, _, l, c} | rest]) do
    case rest do
      [{:rparen, _, _, _} | r] -> {{:tuple, [], {l, c}}, r}
      _ ->
        {e, r} = expr(rest)
        case r do
          [{:rparen, _, _, _} | r2] -> {e, r2}
          [{:comma, _, _, _} | r2] ->
            {more, r3} = seq(r2, :rparen, [])
            {{:tuple, [e | more], {l, c}}, r3}
          [t | _] -> throw(unexpected(t, "`)`"))
        end
    end
  end

  defp atom([{:lbracket, _, l, c} | rest]) do
    case rest do
      [{:rbracket, _, _, _} | r] -> {{:list, [], {l, c}}, r}
      _ ->
        {e, r} = expr(rest)
        case r do
          [{:kw, "for", _, _} | _] ->
            {clauses, r2} = comp_clauses(r, [])
            {{:comp, e, clauses, {l, c}}, r2}
          [{:rbracket, _, _, _} | r2] -> {{:list, [e], {l, c}}, r2}
          [{:comma, _, _, _} | r2] ->
            {more, r3} = seq(r2, :rbracket, [])
            {{:list, [e | more], {l, c}}, r3}
          [t | _] -> throw(unexpected(t, "`]`, `,` or `for`"))
        end
    end
  end

  defp atom([{:lbrace, _, l, c} | rest]) do
    {pairs, rest} = map_pairs(rest, [])
    {{:map, pairs, {l, c}}, rest}
  end

  defp atom([{:kw, "if", _, _} | _] = toks), do: control(toks)
  defp atom([{:kw, "let", _, _} | _] = toks), do: control(toks)
  defp atom([{:kw, "not", _, _} | _] = toks), do: negation(toks)
  defp atom([t | _]), do: throw(unexpected(t))

  defp comp_clauses([{:kw, "for", _, _} | rest], acc) do
    {p, rest} = pattern(rest)
    rest = expect(rest, :kw, "in")
    {src, rest} = bor_no_in(rest)
    comp_clauses(rest, [{:for, p, src} | acc])
  end

  defp comp_clauses([{:kw, "if", _, _} | rest], acc) do
    {cnd, rest} = disj(rest)
    comp_clauses(rest, [{:filter, cnd} | acc])
  end

  defp comp_clauses([{:rbracket, _, _, _} | rest], acc), do: {Enum.reverse(acc), rest}
  defp comp_clauses([t | _], _), do: throw(unexpected(t, "`for`, `if` or `]`"))

  defp bor_no_in(toks), do: range_or_pipe(toks)
  defp range_or_pipe(toks), do: bor(toks)

  defp map_pairs([{:rbrace, _, _, _} | rest], acc), do: {Enum.reverse(acc), rest}

  defp map_pairs(toks, acc) do
    {k, rest} =
      case toks do
        [{:id, n, _, _}, {:colon, _, _, _} | r] -> {{:lit, n}, [{:colon, nil, 0, 0} | r]}
        _ -> expr(toks)
      end
    rest = expect(rest, :colon, ":")
    {v, rest} = expr(rest)
    case rest do
      [{:comma, _, _, _} | r] -> map_pairs(r, [{k, v} | acc])
      [{:rbrace, _, _, _} | r] -> {Enum.reverse([{k, v} | acc]), r}
      [t | _] -> throw(unexpected(t, "`,` or `}`"))
    end
  end

  defp seq([{close, _, _, _} | rest], close, acc), do: {Enum.reverse(acc), rest}

  defp seq(toks, close, acc) do
    {e, rest} = expr(toks)
    case rest do
      [{:comma, _, _, _} | r] -> seq(r, close, [e | acc])
      [{^close, _, _, _} | r] -> {Enum.reverse([e | acc]), r}
      [t | _] -> throw(unexpected(t, "`,` or a closing bracket"))
    end
  end

  defp expect([{k, v, _, _} | rest], k, v), do: rest
  defp expect([{k, _, _, _} | rest], k, _v) when k in [:rbracket, :rparen, :rbrace, :colon], do: rest
  defp expect([t | _], _k, v), do: throw(unexpected(t, "`#{v}`"))

  defp pos([{_, _, l, c} | _]), do: {l, c}

  defp unexpected(t, wanted \\ nil)
  defp unexpected({:eof, _, l, c}, wanted), do: {:parse, "unexpected end of input" <> want(wanted), l, c}
  defp unexpected({:nl, _, l, _}, wanted), do: {:parse, "unexpected end of line" <> want(wanted), l, 0}
  defp unexpected({_, v, l, c}, wanted), do: {:parse, "unexpected #{inspect(v)}" <> want(wanted), l, c}
  defp want(nil), do: ""
  defp want(w), do: ", expected " <> w

  defp perr(msg, [{_, _, l, c} | _]), do: {:parse, msg, l, c}
  defp perr(msg, _), do: {:parse, msg, 0, 0}
end
