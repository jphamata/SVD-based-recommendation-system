defmodule Vapor.Expr do
  @moduledoc """
  The expression language every typed-in problem shares
  (docs/WORKBENCH.md §1): ordinary infix arithmetic with functions,
  comparisons and `if`, numbers carrying **units in brackets**
  (`9.81[m/s^2]`, `3[kN]`), symbolic differentiation and simplification,
  and compilation to native BEAM code.

  An expression is a small tree:

      {:n, float} | {:v, name} | {:q, si_value, dim, unit_text}
      {op, a, b}  (op ∈ + − * / ^ < > <= >= ==)  | {:neg, a} | {:f, name, [args]}

  Nothing in the language can reach the host: the parser accepts only the
  operators and the functions listed in `functions/0`, and `compile/2`
  turns the tree into a module of arithmetic on a tuple of floats — the
  only code it can ever generate. Compiled modules are cached by the
  expression's hash, so a solver calling a right-hand side a million
  times pays for compilation once.

  BEAM floats have no ∞ or NaN: a division by zero or `log(−1)` raises,
  and the solvers turn that into a rejected step or a refusal that says
  where — never a silent NaN propagating through a result.
  """
  alias Vapor.Units

  @functions %{
    "sin" => 1, "cos" => 1, "tan" => 1, "asin" => 1, "acos" => 1, "atan" => 1, "atan2" => 2, "sinh" => 1, "cosh" => 1,
    "tanh" => 1, "asinh" => 1, "acosh" => 1, "atanh" => 1, "exp" => 1, "log" => 1, "ln" => 1, "log10" => 1, "log2" => 1,
    "sqrt" => 1, "cbrt" => 1, "abs" => 1, "sign" => 1, "floor" => 1, "ceil" => 1, "min" => :n, "max" => :n, "hypot" => 2,
    "pow" => 2, "if" => 3, "step" => 1, "erf" => 1, "erfc" => 1, "sinc" => 1, "sq" => 1, "clamp" => 3, "mod" => 2
  }

  @doc "The functions the language knows, with their arity (`:n` = any)."
  def functions, do: @functions

  @constants %{"pi" => :math.pi(), "π" => :math.pi(), "e" => :math.exp(1.0)}

  # ================================================================ parsing

  @doc "Parse text into a tree: `{:ok, tree}` or `{:error, message}` (with the column)."
  def parse(text) when is_binary(text) do
    toks = lex(text)

    case expr(toks) do
      {t, [{:eof, _}]} -> {:ok, t}
      {_, [{tok, col} | _]} -> {:error, "unexpected #{show(tok)} at column #{col}"}
    end
  catch
    {:parse_error, msg} -> {:error, msg}
  end

  @doc "Parse, raising on error."
  def parse!(text) do
    case parse(text) do
      {:ok, t} -> t
      {:error, m} -> raise ArgumentError, "expression #{inspect(text)}: #{m}"
    end
  end

  defp show({:num, n}), do: to_string(n)
  defp show({:id, n}), do: inspect(n)
  defp show({:unit, u}), do: "[#{u}]"
  defp show(:eof), do: "end of input"
  defp show(op), do: inspect(to_string(op))

  defp lex(text), do: lex(String.to_charlist(text), 1, [])

  defp lex([], col, acc), do: Enum.reverse([{:eof, col} | acc])
  defp lex([c | r], col, acc) when c in [?\s, ?\t, ?\n, ?\r], do: lex(r, col + 1, acc)
  defp lex([?*, ?* | r], col, acc), do: lex(r, col + 2, [{:^, col} | acc])
  defp lex([?<, ?= | r], col, acc), do: lex(r, col + 2, [{:<=, col} | acc])
  defp lex([?>, ?= | r], col, acc), do: lex(r, col + 2, [{:>=, col} | acc])
  defp lex([?=, ?= | r], col, acc), do: lex(r, col + 2, [{:==, col} | acc])
  defp lex([c | r], col, acc) when c in [?+, ?-, ?*, ?/, ?^, ?(, ?), ?,, ?<, ?>], do: lex(r, col + 1, [{List.to_atom([c]), col} | acc])
  defp lex([?· | r], col, acc), do: lex(r, col + 1, [{:*, col} | acc])
  defp lex([?× | r], col, acc), do: lex(r, col + 1, [{:*, col} | acc])
  defp lex([?− | r], col, acc), do: lex(r, col + 1, [{:-, col} | acc])

  defp lex([?[ | r], col, acc) do
    case Enum.split_while(r, &(&1 != ?])) do
      {u, [?] | rest]} -> lex(rest, col + length(u) + 2, [{{:unit, to_string(u)}, col} | acc])
      _ -> throw({:parse_error, "unclosed [ at column #{col}"})
    end
  end

  defp lex([c | _] = l, col, acc) when c in ?0..?9 or c == ?. do
    {num, rest} = number(l)
    s = to_string(num)
    v = case Float.parse(if String.starts_with?(s, "."), do: "0" <> s, else: s) do
      {f, ""} -> f
      _ -> throw({:parse_error, "bad number #{s} at column #{col}"})
    end
    lex(rest, col + length(num), [{{:num, v}, col} | acc])
  end

  defp lex([c | _] = l, col, acc) do
    if ident_start?(c) do
      {id, rest} = Enum.split_while(l, &ident?/1)
      lex(rest, col + length(id), [{{:id, to_string(id)}, col} | acc])
    else
      throw({:parse_error, "unexpected character #{inspect(to_string([c]))} at column #{col}"})
    end
  end

  defp number(l) do
    {int, r} = Enum.split_while(l, &(&1 in ?0..?9 or &1 == ?.))
    case r do
      [e | r2] when e in [?e, ?E] ->
        case r2 do
          [s | r3] when s in [?+, ?-] -> (case Enum.split_while(r3, &(&1 in ?0..?9)) do {[], _} -> {int, r}; {d, r4} -> {int ++ [e, s | d], r4} end)
          _ -> (case Enum.split_while(r2, &(&1 in ?0..?9)) do {[], _} -> {int, r}; {d, r4} -> {int ++ [e | d], r4} end)
        end
      _ -> {int, r}
    end
  end

  defp ident_start?(c), do: c in ?a..?z or c in ?A..?Z or c == ?_ or (c > 127 and c not in [?·, ?×, ?−])
  defp ident?(c), do: ident_start?(c) or c in ?0..?9

  # precedence climbing: comparison < additive < multiplicative < unary < power < postfix
  defp expr(t), do: comparison(t)

  defp comparison(t) do
    {a, r} = additive(t)
    case r do
      [{op, _} | r2] when op in [:<, :>, :<=, :>=, :==] -> (({b, r3} = additive(r2)); {{op, a, b}, r3})
      _ -> {a, r}
    end
  end

  defp additive(t) do
    {a, r} = multiplicative(t)
    additive_more(a, r)
  end

  defp additive_more(a, [{op, _} | r]) when op in [:+, :-], do: ((fn {b, r2} -> additive_more({op, a, b}, r2) end).(multiplicative(r)))
  defp additive_more(a, r), do: {a, r}

  defp multiplicative(t) do
    {a, r} = unary(t)
    multiplicative_more(a, r)
  end

  defp multiplicative_more(a, [{op, _} | r]) when op in [:*, :/], do: ((fn {b, r2} -> multiplicative_more({op, a, b}, r2) end).(unary(r)))
  # implicit product: 2x, 3(x+1), (a)(b)
  defp multiplicative_more(a, [{tok, _} | _] = r) when tok == :"(" or (is_tuple(tok) and elem(tok, 0) == :id) do
    {b, r2} = unary(r)
    multiplicative_more({:*, a, b}, r2)
  end
  defp multiplicative_more(a, r), do: {a, r}

  defp unary([{:-, _} | r]), do: ((fn {a, r2} -> {neg(a), r2} end).(unary(r)))
  defp unary([{:+, _} | r]), do: unary(r)
  defp unary(t), do: power(t)

  defp neg({:n, x}), do: {:n, -x}
  defp neg(a), do: {:neg, a}

  defp power(t) do
    {a, r} = postfix(t)
    case r do
      [{:^, _} | r2] -> (({b, r3} = unary(r2)); {{:^, a, b}, r3})
      _ -> {a, r}
    end
  end

  # a number followed by [unit] is a quantity; anything followed by [unit] is scaled by it
  defp postfix(t) do
    {a, r} = primary(t)
    case r do
      [{{:unit, u}, col} | r2] ->
        case {Units.parse(u), Units.affine(u), a} do
          # a reading on an affine scale (°C, °F): only right after a number — 25[°C] is 298.15 K
          {{:ok, {f, dim}}, {:ok, {_, off}}, {:n, x}} -> {{:q, x * f + off, dim, u}, r2}
          {{:ok, _}, {:ok, _}, _} -> throw({:parse_error, "#{u} is a reading, not a unit: it follows a number (25[#{u}]); to scale an expression write [K] or [degC] (column #{col})"})
          {{:ok, {f, dim}}, :none, _} ->
            q = {:q, f, dim, u}
            {(case a do {:n, x} -> {:q, x * f, dim, u}; _ -> {:*, a, q} end), r2}

          {{:error, why}, _, _} -> throw({:parse_error, "#{why} (column #{col})"})
        end
      _ -> {a, r}
    end
  end

  defp primary([{{:num, v}, _} | r]), do: {{:n, v}, r}

  defp primary([{{:id, name}, col}, {:"(", _} | r]) do
    unless Map.has_key?(@functions, name), do: throw({:parse_error, "unknown function #{inspect(name)} at column #{col} (known: #{@functions |> Map.keys() |> Enum.sort() |> Enum.join(", ")})"})
    {args, r2} = args(r, [])
    want = @functions[name]
    if want != :n and want != length(args), do: throw({:parse_error, "#{name} takes #{want} argument(s), got #{length(args)} at column #{col}"})
    if want == :n and args == [], do: throw({:parse_error, "#{name} needs arguments at column #{col}"})
    {{:f, name, args}, r2}
  end

  defp primary([{{:id, name}, _} | r]) do
    case Map.fetch(@constants, name) do
      {:ok, v} -> {{:n, v}, r}
      :error -> {{:v, name}, r}
    end
  end

  defp primary([{:"(", col} | r]) do
    case expr(r) do
      {a, [{:")", _} | r2]} -> {a, r2}
      _ -> throw({:parse_error, "missing ) for ( at column #{col}"})
    end
  end

  defp primary([{{:unit, u}, col} | r]) do
    if Units.affine(u) != :none, do: throw({:parse_error, "#{u} is a reading: write a number before it (column #{col})"})
    case Units.parse(u) do
      {:ok, {f, dim}} -> {{:q, f, dim, u}, r}
      {:error, why} -> throw({:parse_error, why})
    end
  end

  defp primary([{tok, col} | _]), do: throw({:parse_error, "unexpected #{show(tok)} at column #{col}"})

  defp args([{:")", _} | r], acc), do: {Enum.reverse(acc), r}

  defp args(t, acc) do
    {a, r} = expr(t)
    case r do
      [{:",", _} | r2] -> args(r2, [a | acc])
      [{:")", _} | r2] -> {Enum.reverse([a | acc]), r2}
      [{tok, col} | _] -> throw({:parse_error, "expected , or ) but found #{show(tok)} at column #{col}"})
    end
  end

  # ============================================================== analysis

  @doc "The free variables of a tree, sorted."
  def vars(t), do: t |> do_vars(MapSet.new()) |> Enum.sort()

  defp do_vars({:v, n}, s), do: MapSet.put(s, n)
  defp do_vars({:f, _, as}, s), do: Enum.reduce(as, s, &do_vars/2)
  defp do_vars({:neg, a}, s), do: do_vars(a, s)
  defp do_vars({op, a, b}, s) when is_atom(op), do: do_vars(b, do_vars(a, s))
  defp do_vars(_, s), do: s

  @doc "Substitute variables by trees (`%{name => tree}`)."
  def subst({:v, n} = v, m), do: Map.get(m, n, v)
  def subst({:f, f, as}, m), do: {:f, f, Enum.map(as, &subst(&1, m))}
  def subst({:neg, a}, m), do: {:neg, subst(a, m)}
  def subst({op, a, b}, m) when is_atom(op), do: {op, subst(a, m), subst(b, m)}
  def subst(t, _), do: t

  # ============================================================ evaluation

  @doc "Evaluate with `env` (`%{name => number}`): a float (SI for quantities). Raises on an unknown variable or a domain error."
  def eval(t, env \\ %{})
  def eval({:n, x}, _), do: x
  def eval({:q, x, _, _}, _), do: x
  def eval({:v, n}, env), do: (case Map.fetch(env, n) do {:ok, v} -> v * 1.0; :error -> raise ArgumentError, "unknown variable #{inspect(n)}" end)
  def eval({:neg, a}, env), do: -eval(a, env)
  def eval({:+, a, b}, env), do: eval(a, env) + eval(b, env)
  def eval({:-, a, b}, env), do: eval(a, env) - eval(b, env)
  def eval({:*, a, b}, env), do: eval(a, env) * eval(b, env)
  def eval({:/, a, b}, env), do: eval(a, env) / eval(b, env)
  def eval({:^, a, b}, env), do: rpow(eval(a, env), eval(b, env))
  def eval({:<, a, b}, env), do: bool(eval(a, env) < eval(b, env))
  def eval({:>, a, b}, env), do: bool(eval(a, env) > eval(b, env))
  def eval({:<=, a, b}, env), do: bool(eval(a, env) <= eval(b, env))
  def eval({:>=, a, b}, env), do: bool(eval(a, env) >= eval(b, env))
  def eval({:==, a, b}, env), do: bool(eval(a, env) == eval(b, env))
  def eval({:f, "if", [c, a, b]}, env), do: if(eval(c, env) != 0.0, do: eval(a, env), else: eval(b, env))
  def eval({:f, f, as}, env), do: fun(f, Enum.map(as, &eval(&1, env)))

  defp bool(true), do: 1.0
  defp bool(false), do: 0.0

  @doc false
  def rpow(a, b) do
    cond do
      b == 2.0 -> a * a
      b == 1.0 -> a
      b == 0.0 -> 1.0
      b == 0.5 -> :math.sqrt(a)
      true -> :math.pow(a, b)
    end
  end

  @doc false
  def fun("sin", [x]), do: :math.sin(x)
  def fun("cos", [x]), do: :math.cos(x)
  def fun("tan", [x]), do: :math.tan(x)
  def fun("asin", [x]), do: :math.asin(x)
  def fun("acos", [x]), do: :math.acos(x)
  def fun("atan", [x]), do: :math.atan(x)
  def fun("atan2", [y, x]), do: :math.atan2(y, x)
  def fun("sinh", [x]), do: :math.sinh(x)
  def fun("cosh", [x]), do: :math.cosh(x)
  def fun("tanh", [x]), do: :math.tanh(x)
  def fun("asinh", [x]), do: :math.asinh(x)
  def fun("acosh", [x]), do: :math.acosh(x)
  def fun("atanh", [x]), do: :math.atanh(x)
  def fun("exp", [x]), do: :math.exp(x)
  def fun(l, [x]) when l in ["log", "ln"], do: :math.log(x)
  def fun("log10", [x]), do: :math.log10(x)
  def fun("log2", [x]), do: :math.log2(x)
  def fun("sqrt", [x]), do: :math.sqrt(x)
  def fun("cbrt", [x]), do: if(x < 0, do: -:math.pow(-x, 1 / 3), else: :math.pow(x, 1 / 3))
  def fun("abs", [x]), do: abs(x)
  def fun("sign", [x]), do: (cond do x > 0 -> 1.0; x < 0 -> -1.0; true -> 0.0 end)
  def fun("floor", [x]), do: Float.floor(x * 1.0)
  def fun("ceil", [x]), do: Float.ceil(x * 1.0)
  def fun("min", xs), do: Enum.min(xs)
  def fun("max", xs), do: Enum.max(xs)
  def fun("hypot", [x, y]), do: :math.sqrt(x * x + y * y)
  def fun("pow", [x, y]), do: rpow(x, y)
  def fun("step", [x]), do: if(x >= 0, do: 1.0, else: 0.0)
  def fun("erf", [x]), do: :math.erf(x)
  def fun("erfc", [x]), do: :math.erfc(x)
  def fun("sinc", [x]), do: if(x == 0, do: 1.0, else: :math.sin(x) / x)
  def fun("sq", [x]), do: x * x
  def fun("clamp", [x, lo, hi]), do: x |> max(lo) |> min(hi)
  def fun("mod", [x, y]), do: x - y * Float.floor(x / y)

  # ======================================================= differentiation

  @doc "∂t/∂x, simplified."
  def diff(t, x), do: t |> d(x) |> simplify()

  defp d({:n, _}, _), do: {:n, 0.0}
  defp d({:q, _, _, _}, _), do: {:n, 0.0}
  defp d({:v, x}, x), do: {:n, 1.0}
  defp d({:v, _}, _), do: {:n, 0.0}
  defp d({:neg, a}, x), do: {:neg, d(a, x)}
  defp d({:+, a, b}, x), do: {:+, d(a, x), d(b, x)}
  defp d({:-, a, b}, x), do: {:-, d(a, x), d(b, x)}
  defp d({:*, a, b}, x), do: {:+, {:*, d(a, x), b}, {:*, a, d(b, x)}}
  defp d({:/, a, b}, x), do: {:/, {:-, {:*, d(a, x), b}, {:*, a, d(b, x)}}, {:^, b, {:n, 2.0}}}

  defp d({:^, a, b}, x) do
    if x in vars(b) do
      # a^b = exp(b ln a): (a^b)(b' ln a + b a'/a)
      {:*, {:^, a, b}, {:+, {:*, d(b, x), {:f, "ln", [a]}}, {:/, {:*, b, d(a, x)}, a}}}
    else
      {:*, {:*, b, {:^, a, {:-, b, {:n, 1.0}}}}, d(a, x)}
    end
  end

  defp d({op, _, _}, _) when op in [:<, :>, :<=, :>=, :==], do: {:n, 0.0}
  defp d({:f, "if", [c, a, b]}, x), do: {:f, "if", [c, d(a, x), d(b, x)]}
  defp d({:f, f, [a]}, x), do: {:*, d1(f, a), d(a, x)}
  defp d({:f, "atan2", [y, z]}, x), do: {:/, {:-, {:*, z, d(y, x)}, {:*, y, d(z, x)}}, {:+, {:^, z, {:n, 2.0}}, {:^, y, {:n, 2.0}}}}
  defp d({:f, "hypot", [a, b]}, x), do: {:/, {:+, {:*, a, d(a, x)}, {:*, b, d(b, x)}}, {:f, "hypot", [a, b]}}
  defp d({:f, "pow", [a, b]}, x), do: d({:^, a, b}, x)
  defp d({:f, "mod", [a, _b]}, x), do: d(a, x)
  defp d({:f, "clamp", [a, lo, hi]}, x), do: {:f, "if", [{:<, a, lo}, d(lo, x), {:f, "if", [{:>, a, hi}, d(hi, x), d(a, x)]}]}
  defp d({:f, mm, as}, x) when mm in ["min", "max"] do
    # the derivative of the active argument (ties: the first)
    [h | t] = as
    Enum.reduce(t, {h, d(h, x)}, fn b, {cur, dc} ->
      cmp = if mm == "min", do: {:<=, cur, b}, else: {:>=, cur, b}
      {{:f, mm, [cur, b]}, {:f, "if", [cmp, dc, d(b, x)]}}
    end) |> elem(1)
  end

  defp d1("sin", a), do: {:f, "cos", [a]}
  defp d1("cos", a), do: {:neg, {:f, "sin", [a]}}
  defp d1("tan", a), do: {:+, {:n, 1.0}, {:^, {:f, "tan", [a]}, {:n, 2.0}}}
  defp d1("asin", a), do: {:/, {:n, 1.0}, {:f, "sqrt", [{:-, {:n, 1.0}, {:^, a, {:n, 2.0}}}]}}
  defp d1("acos", a), do: {:neg, {:/, {:n, 1.0}, {:f, "sqrt", [{:-, {:n, 1.0}, {:^, a, {:n, 2.0}}}]}}}
  defp d1("atan", a), do: {:/, {:n, 1.0}, {:+, {:n, 1.0}, {:^, a, {:n, 2.0}}}}
  defp d1("sinh", a), do: {:f, "cosh", [a]}
  defp d1("cosh", a), do: {:f, "sinh", [a]}
  defp d1("tanh", a), do: {:-, {:n, 1.0}, {:^, {:f, "tanh", [a]}, {:n, 2.0}}}
  defp d1("asinh", a), do: {:/, {:n, 1.0}, {:f, "sqrt", [{:+, {:^, a, {:n, 2.0}}, {:n, 1.0}}]}}
  defp d1("acosh", a), do: {:/, {:n, 1.0}, {:f, "sqrt", [{:-, {:^, a, {:n, 2.0}}, {:n, 1.0}}]}}
  defp d1("atanh", a), do: {:/, {:n, 1.0}, {:-, {:n, 1.0}, {:^, a, {:n, 2.0}}}}
  defp d1("exp", a), do: {:f, "exp", [a]}
  defp d1(l, a) when l in ["log", "ln"], do: {:/, {:n, 1.0}, a}
  defp d1("log10", a), do: {:/, {:n, 1.0}, {:*, a, {:n, :math.log(10.0)}}}
  defp d1("log2", a), do: {:/, {:n, 1.0}, {:*, a, {:n, :math.log(2.0)}}}
  defp d1("sqrt", a), do: {:/, {:n, 0.5}, {:f, "sqrt", [a]}}
  defp d1("cbrt", a), do: {:/, {:n, 1.0}, {:*, {:n, 3.0}, {:^, {:f, "cbrt", [a]}, {:n, 2.0}}}}
  defp d1("abs", a), do: {:f, "sign", [a]}
  defp d1(z, _) when z in ["sign", "floor", "ceil", "step"], do: {:n, 0.0}
  defp d1("erf", a), do: {:*, {:n, 2 / :math.sqrt(:math.pi())}, {:f, "exp", [{:neg, {:^, a, {:n, 2.0}}}]}}
  defp d1("erfc", a), do: {:*, {:n, -2 / :math.sqrt(:math.pi())}, {:f, "exp", [{:neg, {:^, a, {:n, 2.0}}}]}}
  defp d1("sq", a), do: {:*, {:n, 2.0}, a}
  defp d1("sinc", a), do: {:f, "if", [{:==, a, {:n, 0.0}}, {:n, 0.0}, {:/, {:-, {:*, a, {:f, "cos", [a]}}, {:f, "sin", [a]}}, {:^, a, {:n, 2.0}}}]}

  # =========================================================== simplifying

  @doc "Constant folding and the identities 0+x, 1·x, 0·x, x^1, x^0, −(−x), x−x, x/x (syntactic)."
  def simplify(t) do
    s = simp(t)
    if s == t, do: s, else: simplify(s)
  end

  defp simp({:neg, a}) do
    case simp(a) do
      {:n, x} -> {:n, -x}
      {:neg, b} -> b
      b -> {:neg, b}
    end
  end

  defp simp({:f, f, as}) do
    as = Enum.map(as, &simp/1)
    if f != "if" and Enum.all?(as, &match?({:n, _}, &1)) do
      try do
        {:n, fun(f, Enum.map(as, &elem(&1, 1)))}
      rescue
        _ -> {:f, f, as}
      end
    else
      case {f, as} do
        {"if", [{:n, c}, a, b]} -> if c != 0.0, do: a, else: b
        {"if", [_, a, a]} -> a
        _ -> {:f, f, as}
      end
    end
  end

  defp simp({op, a, b}) when op in [:+, :-, :*, :/, :^, :<, :>, :<=, :>=, :==] do
    a = simp(a)
    b = simp(b)

    case {op, a, b} do
      {_, {:n, x}, {:n, y}} when op in [:+, :-, :*] -> {:n, apply(Kernel, op, [x, y])}
      {:/, {:n, x}, {:n, y}} when y != 0 -> {:n, x / y}
      {:^, {:n, x}, {:n, y}} -> (try do {:n, rpow(x, y)} rescue _ -> {:^, a, b} end)
      {:+, {:n, z}, x} when z == 0 -> x
      {:+, x, {:n, z}} when z == 0 -> x
      {:+, x, {:neg, y}} -> {:-, x, y}
      {:+, x, {:*, {:n, c}, y}} when c < 0 -> {:-, x, {:*, {:n, -c}, y}}
      {:-, x, {:*, {:n, c}, y}} when c < 0 -> {:+, x, {:*, {:n, -c}, y}}
      {:+, x, {:n, c}} when c < 0 -> {:-, x, {:n, -c}}
      {:-, x, {:n, z}} when z == 0 -> x
      {:-, {:n, z}, x} when z == 0 -> {:neg, x}
      {:-, x, x} -> {:n, 0.0}
      {:-, x, {:neg, y}} -> {:+, x, y}
      {:*, {:n, z}, _} when z == 0 -> {:n, 0.0}
      {:*, _, {:n, z}} when z == 0 -> {:n, 0.0}
      {:*, {:n, o}, x} when o == 1 -> x
      {:*, x, {:n, o}} when o == 1 -> x
      {:*, {:n, m}, x} when m == -1 -> {:neg, x}
      {:*, x, {:n, m}} when m == -1 -> {:neg, x}
      {:*, x, {:n, _} = c} -> {:*, c, x}
      {:*, {:n, p}, {:*, {:n, q}, x}} -> {:*, {:n, p * q}, x}
      {:*, {:neg, x}, y} -> {:neg, {:*, x, y}}
      {:*, x, {:neg, y}} -> {:neg, {:*, x, y}}
      {:/, {:n, z}, _} when z == 0 -> {:n, 0.0}
      {:/, x, {:n, o}} when o == 1 -> x
      {:/, x, x} -> {:n, 1.0}
      {:^, x, {:n, o}} when o == 1 -> x
      {:^, _, {:n, z}} when z == 0 -> {:n, 1.0}
      {:^, {:^, x, {:n, p}}, {:n, q}} -> {:^, x, {:n, p * q}}
      {cmp, {:n, x}, {:n, y}} when cmp in [:<, :>, :<=, :>=, :==] -> {:n, bool(apply(Kernel, cmp, [x, y]))}
      _ -> {op, a, b}
    end
  end

  defp simp(t), do: t

  # ============================================================== printing

  @doc "A tree as text, with the fewest parentheses."
  def to_text(t), do: pr(t, 0)

  defp prec(op) when op in [:<, :>, :<=, :>=, :==], do: 1
  defp prec(op) when op in [:+, :-], do: 2
  defp prec(op) when op in [:*, :/], do: 3
  defp prec(:neg), do: 4
  defp prec(:^), do: 5

  defp pr({:n, x}, ctx) do
    s = num(x)
    if x < 0 and ctx > 2, do: "(" <> s <> ")", else: s
  end

  defp pr({:q, x, _dim, u}, _) do
    {:ok, {f, _}} = Units.parse(u)
    off = case Units.affine(u) do {:ok, {_, o}} -> o; :none -> 0.0 end
    num((x - off) / f) <> "[" <> u <> "]"
  end
  defp pr({:v, n}, _), do: n
  defp pr({:f, f, as}, _), do: f <> "(" <> Enum.map_join(as, ", ", &pr(&1, 0)) <> ")"
  defp pr({:neg, a}, ctx), do: wrap("−" <> pr(a, 4), ctx > 4)

  defp pr({op, a, b}, ctx) do
    p = prec(op)
    {l, r} = if op == :^, do: {p + 1, p}, else: {p, p + 1}
    sym = %{+: " + ", -: " − ", *: "·", /: "/", ^: "^", <: " < ", >: " > ", <=: " ≤ ", >=: " ≥ ", ==: " = "}[op]
    wrap(pr(a, l) <> sym <> pr(b, r), p < ctx)
  end

  defp wrap(s, true), do: "(" <> s <> ")"
  defp wrap(s, false), do: s

  @doc "A float shown with up to 10 significant digits, without trailing zeros."
  def num(x) when is_integer(x), do: Integer.to_string(x)

  def num(x) do
    cond do
      x == trunc(x) and abs(x) < 1.0e15 -> Integer.to_string(trunc(x))
      abs(x) >= 1.0e-4 and abs(x) < 1.0e9 -> :erlang.float_to_binary(x, [:compact, decimals: 10]) |> trim_float()
      true -> :io_lib.format("~.10g", [x]) |> to_string() |> String.replace(~r/\.?0+e/, "e")
    end
  end

  defp trim_float(s), do: if(String.contains?(s, "."), do: s |> String.trim_trailing("0") |> String.trim_trailing("."), else: s)

  @doc "A tree as LaTeX."
  def to_latex({:n, x}), do: num(x)
  def to_latex({:q, x, _, u}), do: num(x) <> "\\,\\mathrm{" <> u <> "}"
  def to_latex({:v, n}), do: if(String.length(n) > 1, do: "\\mathrm{#{n}}", else: n)
  def to_latex({:neg, a}), do: "-" <> lparen(a, 4)
  def to_latex({:+, a, b}), do: to_latex(a) <> " + " <> to_latex(b)
  def to_latex({:-, a, b}), do: to_latex(a) <> " - " <> lparen(b, 3)
  def to_latex({:*, a, b}), do: lparen(a, 3) <> " \\cdot " <> lparen(b, 3)
  def to_latex({:/, a, b}), do: "\\frac{" <> to_latex(a) <> "}{" <> to_latex(b) <> "}"
  def to_latex({:^, a, b}), do: lparen(a, 5) <> "^{" <> to_latex(b) <> "}"
  def to_latex({:f, "sqrt", [a]}), do: "\\sqrt{" <> to_latex(a) <> "}"
  def to_latex({:f, f, as}), do: "\\operatorname{#{f}}\\left(" <> Enum.map_join(as, ", ", &to_latex/1) <> "\\right)"
  def to_latex({op, a, b}), do: to_latex(a) <> %{<: " < ", >: " > ", <=: " \\le ", >=: " \\ge ", ==: " = "}[op] <> to_latex(b)

  defp lparen({op, _, _} = t, ctx) when is_atom(op), do: if(prec(op) < ctx, do: "\\left(" <> to_latex(t) <> "\\right)", else: to_latex(t))
  defp lparen(t, _), do: to_latex(t)

  # ============================================================ dimensions

  @doc """
  The dimension of a tree given the variables' dimensions (`%{name =>
  dim}`; unknown names are dimensionless): `{:ok, dim}` or `{:error,
  message}`. Addition and comparison need equal dimensions; transcendental
  functions need pure numbers; powers of a dimensioned base need a
  constant exponent.
  """
  def dim(t, dims \\ %{}) do
    {:ok, dm(t, dims)}
  catch
    {:dim_error, msg} -> {:error, msg}
  end

  defp dm({:n, _}, _), do: Units.none()
  defp dm({:q, _, d, _}, _), do: d
  defp dm({:v, n}, ds), do: Map.get(ds, n, Units.none())
  defp dm({:neg, a}, ds), do: dm(a, ds)
  defp dm({op, a, b} = t, ds) when op in [:+, :-, :<, :>, :<=, :>=, :==] do
    {x, y} = {dm(a, ds), dm(b, ds)}
    unless Units.same?(x, y), do: throw({:dim_error, "#{if op in [:+, :-], do: "adding", else: "comparing"} #{Units.label(x)} and #{Units.label(y)} in #{to_text(t)}"})
    if op in [:+, :-], do: x, else: Units.none()
  end
  defp dm({:*, a, b}, ds), do: Units.add(dm(a, ds), dm(b, ds))
  defp dm({:/, a, b}, ds), do: Units.sub(dm(a, ds), dm(b, ds))

  defp dm({:^, a, b} = t, ds) do
    da = dm(a, ds)
    db = dm(b, ds)
    unless Units.none?(db), do: throw({:dim_error, "an exponent must be a pure number in #{to_text(t)}"})
    cond do
      Units.none?(da) -> Units.none()
      vars(b) == [] -> Units.scale(da, eval(b))
      true -> throw({:dim_error, "a dimensioned base (#{Units.format(da)}) needs a constant exponent in #{to_text(t)}"})
    end
  end

  defp dm({:f, f, [a]}, ds) when f in ["abs", "floor", "ceil", "sq", "sqrt", "cbrt", "sign", "step"] do
    da = dm(a, ds)
    case f do
      "sq" -> Units.scale(da, 2)
      "sqrt" -> Units.scale(da, 0.5)
      "cbrt" -> Units.scale(da, 1 / 3)
      x when x in ["sign", "step"] -> Units.none()
      _ -> da
    end
  end

  defp dm({:f, f, as} = t, ds) when f in ["min", "max", "hypot", "clamp", "mod"] do
    [h | rest] = Enum.map(as, &dm(&1, ds))
    unless Enum.all?(rest, &Units.same?(&1, h)), do: throw({:dim_error, "#{f} of different dimensions in #{to_text(t)}"})
    h
  end

  defp dm({:f, "if", [c, a, b]} = t, ds) do
    _ = dm(c, ds)
    {x, y} = {dm(a, ds), dm(b, ds)}
    unless Units.same?(x, y), do: throw({:dim_error, "the branches of if differ (#{Units.format(x)} vs #{Units.format(y)}) in #{to_text(t)}"})
    x
  end

  defp dm({:f, "atan2", [y, x]} = t, ds) do
    unless Units.same?(dm(y, ds), dm(x, ds)), do: throw({:dim_error, "atan2 of different dimensions in #{to_text(t)}"})
    Units.none()
  end

  defp dm({:f, "pow", [a, b]}, ds), do: dm({:^, a, b}, ds)

  defp dm({:f, f, as} = t, ds) do
    Enum.each(as, fn a ->
      da = dm(a, ds)
      unless Units.none?(da), do: throw({:dim_error, "#{f} needs a pure number, got #{Units.label(da)} in #{to_text(t)}"})
    end)
    Units.none()
  end

  # ============================================================ compilation

  @doc """
  Compile trees over the variables `names` (in order) into a function
  `fn tuple_of_floats -> [float] end` (or a single float with one tree),
  backed by a BEAM module cached by content. The generated code is only
  arithmetic on `elem(vars, i)`: the tree's grammar is the bound.
  """
  def compile(trees, names) when is_list(trees) do
    idx = names |> Enum.with_index() |> Map.new()
    missing = trees |> Enum.flat_map(&vars/1) |> Enum.uniq() |> Enum.reject(&Map.has_key?(idx, &1))
    if missing != [], do: raise(ArgumentError, "unbound variable(s): #{Enum.join(missing, ", ")}")
    key = :crypto.hash(:sha256, :erlang.term_to_binary({trees, names})) |> binary_part(0, 10) |> Base.encode16()
    name = "Elixir.Vapor.Expr.JIT.M" <> key <> "_" <> Integer.to_string(length(names))
    known = try do String.to_existing_atom(name) rescue ArgumentError -> nil end

    cond do
      known != nil and Code.ensure_loaded?(known) -> &known.run/1
      # open input must not grow the atom table and the code server without bound: past the
      # cap, a new expression is interpreted (same values, slower) and no atom is minted
      jit_count() >= jit_cap() -> interpreted(trees, names)
      true -> jit(String.to_atom(name), trees, idx)
    end
  end

  def compile(tree, names), do: (f = compile([tree], names); fn v -> hd(f.(v)) end)

  @doc "How many expressions have been compiled to modules in this VM, and the cap past which they are interpreted."
  def jit_count, do: :persistent_term.get({__MODULE__, :jit}, nil) |> then(&if(&1, do: :counters.get(&1, 1), else: 0))
  def jit_cap, do: Application.get_env(:vapor, :expr_jit_cap, 4096)

  defp interpreted(trees, names) do
    fn v -> env = names |> Enum.with_index() |> Map.new(fn {n, i} -> {n, elem(v, i)} end); Enum.map(trees, &eval(&1, env)) end
  end

  defp jit(mod, trees, idx) do
    c = :persistent_term.get({__MODULE__, :jit}, nil) || (c = :counters.new(1, [:atomics]); :persistent_term.put({__MODULE__, :jit}, c); c)
    :counters.add(c, 1, 1)
    do_jit(mod, trees, idx)
  end

  defp do_jit(mod, trees, idx) do
    unless Code.ensure_loaded?(mod) do
      body = Enum.map(trees, &q(&1, idx))
      ast = quote do
        @moduledoc false
        def run(var!(v)) do
          _ = var!(v)
          unquote(body)
        end
      end

      :global.trans({{__MODULE__, mod}, self()}, fn ->
        unless Code.ensure_loaded?(mod), do: Module.create(mod, ast, Macro.Env.location(__ENV__))
      end)
    end

    &mod.run/1
  end

  defp q({:n, x}, _), do: x
  defp q({:q, x, _, _}, _), do: x
  defp q({:v, n}, idx), do: quote(do: elem(var!(v), unquote(idx[n])))
  defp q({:neg, a}, idx), do: quote(do: -unquote(q(a, idx)))
  defp q({:+, a, b}, idx), do: quote(do: unquote(q(a, idx)) + unquote(q(b, idx)))
  defp q({:-, a, b}, idx), do: quote(do: unquote(q(a, idx)) - unquote(q(b, idx)))
  defp q({:*, a, b}, idx), do: quote(do: unquote(q(a, idx)) * unquote(q(b, idx)))
  defp q({:/, a, b}, idx), do: quote(do: unquote(q(a, idx)) / unquote(q(b, idx)))
  defp q({:^, a, {:n, 2.0}}, idx), do: quote(do: (case unquote(q(a, idx)) do x -> x * x end))
  defp q({:^, a, {:n, 3.0}}, idx), do: quote(do: (case unquote(q(a, idx)) do x -> x * x * x end))
  defp q({:^, a, b}, idx), do: quote(do: Vapor.Expr.rpow(unquote(q(a, idx)), unquote(q(b, idx))))
  defp q({op, a, b}, idx) when op in [:<, :>, :<=, :>=, :==], do: quote(do: if(unquote({op, [], [q(a, idx), q(b, idx)]}), do: 1.0, else: 0.0))
  defp q({:f, "if", [c, a, b]}, idx), do: quote(do: if(unquote(q(c, idx)) != 0.0, do: unquote(q(a, idx)), else: unquote(q(b, idx))))

  defp q({:f, f, [a]}, idx) when f in ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log10", "log2", "sqrt", "erf", "erfc"],
    do: quote(do: :math.unquote(String.to_atom(f))(unquote(q(a, idx))))

  defp q({:f, l, [a]}, idx) when l in ["log", "ln"], do: quote(do: :math.log(unquote(q(a, idx))))
  defp q({:f, "abs", [a]}, idx), do: quote(do: abs(unquote(q(a, idx))))
  defp q({:f, "sq", [a]}, idx), do: quote(do: (case unquote(q(a, idx)) do x -> x * x end))
  defp q({:f, f, as}, idx), do: quote(do: Vapor.Expr.fun(unquote(f), unquote(Enum.map(as, &q(&1, idx)))))

  # ============================================================== worksheet

  @doc """
  An engineering worksheet: lines `name = expression` (optionally `in
  [unit]` to choose how the result is shown) or bare expressions,
  evaluated in order with units; `#` starts a comment. Returns one entry
  per line: `%{line, name, value (SI), dim, unit, shown, text, error}` —
  an error stops nothing after it but names the line and the cause.
  """
  def worksheet(text) when is_binary(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.reduce({[], %{}, %{}}, fn {raw, n}, {acc, env, dims} ->
      line = raw |> String.split("#", parts: 2) |> hd() |> String.trim()

      if line == "" do
        {acc, env, dims}
      else
        case sheet_line(line, env, dims) do
          {:ok, name, v, d, shown, unit, tree} ->
            entry = %{line: n, name: name, value: v, dim: Tuple.to_list(d), unit: unit, shown: shown, text: to_text(tree), dimension: Units.describe(d), error: nil}
            env = if name, do: Map.put(env, name, v), else: env
            dims = if name, do: Map.put(dims, name, d), else: dims
            {[entry | acc], env, dims}

          {:error, why} ->
            {[%{line: n, name: nil, value: nil, error: why, source: line} | acc], env, dims}
        end
      end
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  defp sheet_line(line, env, dims) do
    {body, target} =
      case Regex.run(~r/^(.*)\s+(?:in|em|->|→)\s*\[([^\]]+)\]\s*$/u, line) do
        [_, b, u] -> {b, u}
        nil -> {line, nil}
      end

    {name, rhs} =
      case Regex.run(~r/^\s*([A-Za-z_\x{00C0}-\x{1FFF}][\w\x{00C0}-\x{1FFF}]*)\s*=(?!=)\s*(.+)$/u, body) do
        [_, n, r] -> {n, r}
        nil -> {nil, body}
      end

    with {:ok, tree} <- parse(rhs),
         [] <- Enum.reject(vars(tree), &Map.has_key?(env, &1)) |> (fn m -> if m == [], do: [], else: {:error, "unknown name(s): #{Enum.join(m, ", ")}"} end).(),
         {:ok, d} <- dim(tree, dims),
         {:ok, v} <- safe_eval(tree, env),
         {:ok, shown, unit} <- show_in(v, d, target) do
      {:ok, name, v, d, shown, unit, tree}
    end
  end

  defp safe_eval(tree, env) do
    {:ok, eval(tree, env)}
  rescue
    ArithmeticError -> {:error, "arithmetic error (division by zero, or a function outside its domain) in #{to_text(tree)}"}
    e in ArgumentError -> {:error, Exception.message(e)}
  end

  defp show_in(v, d, nil), do: {:ok, v, Units.format(d)}

  defp show_in(v, d, target) do
    case Units.convert(v, d, target) do
      {:ok, x} -> {:ok, x, target}
      {:error, why} -> {:error, why}
    end
  end
end
