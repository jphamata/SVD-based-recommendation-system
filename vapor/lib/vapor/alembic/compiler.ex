defmodule Vapor.Alembic.Compiler do
  @moduledoc """
  Alembic's evaluator: the tree is compiled once into nested closures
  (`fn env -> value end`), so a verifier called a hundred thousand times
  in a search pays for the walk over the tree once (docs/ALEMBIC.md §3).

  Every way a program can run long — a call, an element of a loop — pays
  **fuel**; running out throws, never hangs. Recursion depth is bounded.
  Integers are arbitrary-precision but capped in size, lists in length.
  The heap of the process that runs it is capped by `Vapor.Alembic.Sandbox`.
  Nothing here can reach the host: the only operations are the ones below
  and in `Vapor.Alembic.Builtins`.
  """
  alias Vapor.Alembic.Builtins

  @max_depth 5_000
  @max_bits 65_536
  @max_list 2_000_000

  def max_bits, do: @max_bits
  def max_list, do: @max_list

  # ------------------------------------------------------------- run state

  @doc "Charge `n` units of fuel."
  def tick(n \\ 1) do
    f = Process.get(:alembic_fuel, 0) - n
    Process.put(:alembic_fuel, f)
    if f < 0, do: throw({:alembic, "out of fuel (the program ran longer than its budget — make it cheaper or raise the fuel)"})
    :ok
  end

  def fail(msg), do: throw({:alembic, msg})

  # ------------------------------------------------------------- compile

  @doc "Compile an expression tree with the given local names in scope."
  def compile(ast, locals \\ MapSet.new())

  def compile({:lit, v}, _), do: fn _ -> v end

  def compile({:var, n, _pos}, locals) do
    if MapSet.member?(locals, n) do
      fn env -> Map.fetch!(env, n) end
    else
      fn _ -> global(n) end
    end
  end

  def compile({:list, items, _}, locals) do
    cs = Enum.map(items, &compile(&1, locals))
    fn env -> Enum.map(cs, & &1.(env)) end
  end

  def compile({:tuple, items, _}, locals) do
    cs = Enum.map(items, &compile(&1, locals))
    fn env -> cs |> Enum.map(& &1.(env)) |> List.to_tuple() end
  end

  def compile({:map, pairs, _}, locals) do
    cs = Enum.map(pairs, fn {k, v} -> {compile(k, locals), compile(v, locals)} end)
    fn env -> Map.new(cs, fn {k, v} -> {k.(env), v.(env)} end) end
  end

  def compile({:and, a, b}, locals) do
    {ca, cb} = {compile(a, locals), compile(b, locals)}
    fn env -> truthy(ca.(env)) and truthy(cb.(env)) end
  end

  def compile({:or, a, b}, locals) do
    {ca, cb} = {compile(a, locals), compile(b, locals)}
    fn env -> truthy(ca.(env)) or truthy(cb.(env)) end
  end

  def compile({:not, a}, locals) do
    ca = compile(a, locals)
    fn env -> not truthy(ca.(env)) end
  end

  def compile({:neg, a, _}, locals) do
    ca = compile(a, locals)
    fn env -> neg(ca.(env)) end
  end

  def compile({:bnot, a, _}, locals) do
    ca = compile(a, locals)
    fn env -> (v = ca.(env); if is_integer(v), do: Bitwise.bnot(v), else: fail("~ needs an integer, got #{Builtins.type(v)}")) end
  end

  def compile({:bin, op, a, b, _}, locals) do
    {ca, cb} = {compile(a, locals), compile(b, locals)}
    f = Builtins.binop(op)
    fn env -> f.(ca.(env), cb.(env)) end
  end

  def compile({:range, a, b, _}, locals) do
    {ca, cb} = {compile(a, locals), compile(b, locals)}
    fn env -> Builtins.range(ca.(env), cb.(env), true) end
  end

  def compile({:cmp, a, chain, _}, locals) do
    ca = compile(a, locals)
    cs = Enum.map(chain, fn {op, b} -> {Builtins.cmpop(op), compile(b, locals)} end)
    fn env -> cmp_chain(ca.(env), cs, env) end
  end

  def compile({:if, c, a, b, _}, locals) do
    {cc, ca, cb} = {compile(c, locals), compile(a, locals), compile(b, locals)}
    fn env -> if truthy(cc.(env)), do: ca.(env), else: cb.(env) end
  end

  def compile({:let, binds, body, _}, locals) do
    {cbinds, locals2} =
      Enum.map_reduce(binds, locals, fn {p, e}, ls ->
        ce = compile(e, ls)
        {{p, ce}, MapSet.union(ls, pattern_names(p))}
      end)

    cbody = compile(body, locals2)

    fn env ->
      env2 = Enum.reduce(cbinds, env, fn {p, ce}, en -> bind(p, ce.(en), en) end)
      cbody.(env2)
    end
  end

  def compile({:lambda, ps, body, _}, locals) do
    names = Enum.reduce(ps, MapSet.new(), &MapSet.union(pattern_names(&1), &2))
    cbody = compile(body, MapSet.union(locals, names))
    arity = length(ps)

    fn env ->
      {:fn, "λ", arity, fn args ->
         env2 = ps |> Enum.zip(args) |> Enum.reduce(env, fn {p, v}, en -> bind(p, v, en) end)
         cbody.(env2)
       end}
    end
  end

  def compile({:call, f, args, {l, c}}, locals) do
    cf = compile(f, locals)
    cargs = Enum.map(args, &compile(&1, locals))
    where = case f do {:var, n, _} -> n; _ -> "function" end

    fn env ->
      fv = cf.(env)
      apply_fn(fv, Enum.map(cargs, & &1.(env)), where, {l, c})
    end
  end

  def compile({:index, a, i, _}, locals) do
    {ca, ci} = {compile(a, locals), compile(i, locals)}
    fn env -> Builtins.index(ca.(env), ci.(env)) end
  end

  def compile({:slice, a, i, j, _}, locals) do
    ca = compile(a, locals)
    ci = if i, do: compile(i, locals), else: fn _ -> nil end
    cj = if j, do: compile(j, locals), else: fn _ -> nil end
    fn env -> Builtins.slice(ca.(env), ci.(env), cj.(env)) end
  end

  def compile({:field, a, f, _}, locals) do
    ca = compile(a, locals)
    fn env -> Builtins.field(ca.(env), f) end
  end

  def compile({:comp, e, clauses, _}, locals) do
    {steps, locals2} =
      Enum.map_reduce(clauses, locals, fn
        {:for, p, src}, ls -> {{:for, p, compile(src, ls)}, MapSet.union(ls, pattern_names(p))}
        {:filter, c}, ls -> {{:filter, compile(c, ls)}, ls}
      end)

    ce = compile(e, locals2)

    fn env ->
      out = comp(steps, env, ce, [])
      if length(out) > @max_list, do: fail("a list longer than #{@max_list} elements")
      Enum.reverse(out)
    end
  end

  # ------------------------------------------------------------ helpers

  defp comp([], env, ce, acc), do: [ce.(env) | acc]

  defp comp([{:for, p, src} | rest], env, ce, acc) do
    items = Builtins.iterable(src.(env))
    Enum.reduce(items, acc, fn v, a ->
      tick()
      comp(rest, bind(p, v, env), ce, a)
    end)
  end

  defp comp([{:filter, c} | rest], env, ce, acc) do
    if truthy(c.(env)), do: comp(rest, env, ce, acc), else: acc
  end

  defp cmp_chain(_a, [], _env), do: true

  defp cmp_chain(a, [{f, cb} | rest], env) do
    b = cb.(env)
    if f.(a, b), do: cmp_chain(b, rest, env), else: false
  end

  @doc "Truthiness: `false`, `nil`, `0`, `0.0`, `[]` and `\"\"` are false."
  def truthy(v) when v in [false, nil, 0, [], ""], do: false
  def truthy(v) when is_float(v) and v == 0.0, do: false
  def truthy(_), do: true

  def neg(v) when is_number(v), do: -v
  def neg(v), do: fail("cannot negate #{Builtins.type(v)}")

  @doc "The names a pattern binds."
  def pattern_names({:pvar, n}), do: MapSet.new([n])
  def pattern_names(:pany), do: MapSet.new()
  def pattern_names({k, ps}) when k in [:ptuple, :plist], do: Enum.reduce(ps, MapSet.new(), &MapSet.union(pattern_names(&1), &2))

  @doc "Bind a value to a pattern in an environment."
  def bind({:pvar, n}, v, env), do: Map.put(env, n, v)
  def bind(:pany, _v, env), do: env

  def bind({:ptuple, ps}, v, env) when is_tuple(v) and tuple_size(v) == length(ps),
    do: ps |> Enum.zip(Tuple.to_list(v)) |> Enum.reduce(env, fn {p, x}, e -> bind(p, x, e) end)

  def bind({:ptuple, ps}, v, env) when is_list(v) and length(v) == length(ps), do: bind({:plist, ps}, v, env)

  def bind({:plist, ps}, v, env) when is_list(v) and length(v) == length(ps),
    do: ps |> Enum.zip(v) |> Enum.reduce(env, fn {p, x}, e -> bind(p, x, e) end)

  def bind({:plist, ps}, v, env) when is_tuple(v), do: bind({:ptuple, ps}, v, env)
  def bind(p, v, _env), do: fail("cannot unpack #{Builtins.show_short(v)} into #{length(elem(p, 1))} names")

  @doc "Call a callable value with a list of arguments."
  def apply_fn({:fn, name, arity, f}, args, _where, _pos) do
    n = length(args)

    ok =
      case arity do
        :any -> true
        {lo, hi} -> n >= lo and n <= hi
        k -> n == k
      end

    unless ok, do: fail("#{name} takes #{arity_text(arity)}, got #{n}")
    tick()
    d = Process.get(:alembic_rdepth, 0)
    if d >= @max_depth, do: fail("recursion deeper than #{@max_depth} calls")
    Process.put(:alembic_rdepth, d + 1)
    r = f.(args)
    Process.put(:alembic_rdepth, d)
    r
  end

  def apply_fn(v, _args, where, {l, c}), do: fail("#{where} is #{Builtins.type(v)}, not a function (line #{l}, column #{c})")

  defp arity_text({lo, hi}), do: "#{lo} to #{hi} arguments"
  defp arity_text(1), do: "1 argument"
  defp arity_text(k), do: "#{k} arguments"

  # ------------------------------------------------------------ globals

  @doc "Look a global name up: the program's definitions, then the builtins."
  def global(n) do
    g = Process.get(:alembic_globals, %{})

    case Map.fetch(g, n) do
      {:ok, {:value, v}} -> v
      {:ok, {:thunk, c}} -> force(n, c)
      {:ok, :evaluating} -> fail("#{n} is defined in terms of itself")
      :error ->
        case Builtins.get(n) do
          nil -> fail("unknown name #{n}" <> suggest(n, Map.keys(g)))
          v -> v
        end
    end
  end

  defp force(n, c) do
    Process.put(:alembic_globals, Map.put(Process.get(:alembic_globals), n, :evaluating))
    v = c.(%{})
    Process.put(:alembic_globals, Map.put(Process.get(:alembic_globals), n, {:value, v}))
    v
  end

  defp suggest(n, names) do
    all = names ++ Builtins.names()
    case Enum.max_by(all, &String.jaro_distance(&1, n), fn -> nil end) do
      nil -> ""
      best -> if String.jaro_distance(best, n) > 0.8, do: " (did you mean #{best}?)", else: ""
    end
  end
end
