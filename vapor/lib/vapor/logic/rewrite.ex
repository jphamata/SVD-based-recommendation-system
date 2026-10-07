defmodule Vapor.Logic.Rewrite do
  @moduledoc """
  Equational reasoning by Knuth–Bendix completion (docs/LOGICA.md §4):
  from the axioms of an algebraic theory (groups, rings, monoids, any
  equations typed in), orient each equation by the lexicographic path
  order, compute critical pairs, and repeat until every pair joins — a
  **convergent rewriting system**. Then the word problem is decided: two
  terms are equal in every model of the axioms exactly when their normal
  forms coincide. Each `decide/3` answer comes with both rewrite
  derivations, which anyone can replay rule by rule.

  Syntax: infix `*` and `+` (left-associative), prefix functions `f(x, y)`,
  constants. Variables are identifiers from `u` to `z` (optionally with
  digits or primes) unless declared with `vars a b c`. A precedence for
  the path order is given with `precedence i > * > e` (default: the
  symbols by decreasing arity, then in order of appearance).

      vars x y z
      precedence i > * > e
      e * x = x
      i(x) * x = e
      (x * y) * z = x * (y * z)

  completes to the ten rules of the free group's word problem (Knuth &
  Bendix 1970).
  """

  @max_rules 200
  @max_steps 4000

  # ================================================================ terms

  @doc "Parse a term with the given variable set."
  def parse_term(s, vars) do
    case tterm(lex(s), vars) do
      {t, []} -> {:ok, t}
      {_, r} -> {:error, "unexpected #{inspect(hd(r))} in #{s}"}
    end
  catch
    {:tparse, w} -> {:error, w}
  end

  defp lex(s) do
    Regex.scan(~r/[\p{L}_][\p{L}\p{N}_']*|\d+|[*+()·,]/u, s) |> Enum.map(&hd/1) |> Enum.map(fn "·" -> "*"; x -> x end)
  end

  defp tterm(t, v), do: (({a, r} = tprod(t, v)); tsum(a, r, v))
  defp tsum(a, ["+" | r], v), do: (({b, r2} = tprod(r, v)); tsum({:fn, "+", [a, b]}, r2, v))
  defp tsum(a, r, _), do: {a, r}
  defp tprod(t, v), do: (({a, r} = tatom(t, v)); tprod_more(a, r, v))
  defp tprod_more(a, ["*" | r], v), do: (({b, r2} = tatom(r, v)); tprod_more({:fn, "*", [a, b]}, r2, v))
  defp tprod_more(a, r, _), do: {a, r}
  defp tatom(["(" | r], v), do: (case tterm(r, v) do {a, [")" | r2]} -> {a, r2}; _ -> throw({:tparse, "missing )"}) end)
  defp tatom([f, "(" | r], v), do: (({args, r2} = targs(r, v, [])); {{:fn, f, args}, r2})
  defp tatom([x | r], v), do: {if(MapSet.member?(v, x), do: {:var, x}, else: {:fn, x, []}), r}
  defp tatom([], _), do: throw({:tparse, "unexpected end"})
  defp targs(t, v, acc) do
    {a, r} = tterm(t, v)
    case r do
      ["," | r2] -> targs(r2, v, [a | acc])
      [")" | r2] -> {Enum.reverse([a | acc]), r2}
      _ -> throw({:tparse, "expected , or )"})
    end
  end

  @doc "A term as text."
  def show({:var, x}), do: x
  def show({:fn, op, [a, b]}) when op in ["*", "+"], do: "#{paren(a, op)} #{op} #{paren(b, op, :right)}"
  def show({:fn, f, []}), do: f
  def show({:fn, f, args}), do: "#{f}(#{Enum.map_join(args, ", ", &show/1)})"
  defp paren(t, op, side \\ :left)
  defp paren({:fn, o, [_, _]} = t, op, side) when o in ["*", "+"], do: if(o != op or side == :right, do: "(#{show(t)})", else: show(t))
  defp paren(t, _, _), do: show(t)

  defp tvars({:var, x}), do: [x]
  defp tvars({:fn, _, as}), do: Enum.flat_map(as, &tvars/1)
  defp occurs?(x, t), do: x in tvars(t)
  defp size({:var, _}), do: 1
  defp size({:fn, _, as}), do: 1 + Enum.sum(Enum.map(as, &size/1))

  # ------------------------------------------------------ substitution & co

  defp apply_s({:var, x} = v, s), do: Map.get(s, x, v)
  defp apply_s({:fn, f, as}, s), do: {:fn, f, Enum.map(as, &apply_s(&1, s))}

  defp unify(a, b), do: unify([{a, b}], %{}, true)
  defp unify([], s, _), do: {:ok, s}
  defp unify([{a, b} | rest], s, _) do
    a = walk(a, s)
    b = walk(b, s)
    cond do
      a == b -> unify(rest, s, true)
      match?({:var, _}, a) -> bind(elem(a, 1), b, rest, s)
      match?({:var, _}, b) -> bind(elem(b, 1), a, rest, s)
      true ->
        {:fn, f, xs} = a
        {:fn, g, ys} = b
        if f == g and length(xs) == length(ys), do: unify(Enum.zip(xs, ys) ++ rest, s, true), else: :fail
    end
  end
  defp bind(x, t, rest, s) do
    t2 = resolve(t, s)
    if occurs?(x, t2), do: :fail, else: unify(rest, Map.put(s, x, t2), true)
  end
  defp walk({:var, x} = v, s), do: (case Map.fetch(s, x) do {:ok, t} -> walk(t, s); :error -> v end)
  defp walk(t, _), do: t
  defp resolve(t, s), do: (case walk(t, s) do {:fn, f, as} -> {:fn, f, Enum.map(as, &resolve(&1, s))}; v -> v end)

  # one-way matching of a pattern onto a term
  defp match(p, t), do: mt([{p, t}], %{})
  defp mt([], s), do: {:ok, s}
  defp mt([{{:var, x}, t} | r], s) do
    case Map.fetch(s, x) do
      {:ok, ^t} -> mt(r, s)
      {:ok, _} -> :fail
      :error -> mt(r, Map.put(s, x, t))
    end
  end
  defp mt([{{:fn, f, xs}, {:fn, f, ys}} | r], s) when length(xs) == length(ys), do: mt(Enum.zip(xs, ys) ++ r, s)
  defp mt(_, _), do: :fail

  defp rename({l, r}, tag) do
    vs = Enum.uniq(tvars(l) ++ tvars(r))
    s = Map.new(vs, &{&1, {:var, "#{&1}#{tag}"}})
    {apply_s(l, s), apply_s(r, s)}
  end

  # ------------------------------------------------------------ rewriting

  @doc "The normal form of a term and its derivation (`[{rule_index, term_after}]`)."
  def normalize(t, rules), do: normalize(t, rules, [], 0)

  defp normalize(t, _rules, steps, k) when k > 10_000, do: {t, Enum.reverse(steps)}
  defp normalize(t, rules, steps, k) do
    case rewrite_once(t, rules) do
      nil -> {t, Enum.reverse(steps)}
      {t2, i} -> normalize(t2, rules, [{i, t2} | steps], k + 1)
    end
  end

  # innermost-first: arguments, then the root
  defp rewrite_once({:var, _}, _), do: nil
  defp rewrite_once({:fn, f, as} = t, rules) do
    case Enum.find_value(Enum.with_index(as), fn {a, i} -> (case rewrite_once(a, rules) do nil -> nil; {a2, k} -> {List.replace_at(as, i, a2), k} end) end) do
      {as2, k} -> {{:fn, f, as2}, k}
      nil ->
        Enum.find_value(Enum.with_index(rules), fn {{l, r}, i} ->
          case match(l, t) do
            {:ok, s} -> {apply_s(r, s), i}
            :fail -> nil
          end
        end)
    end
  end

  # ---------------------------------------------------- the path order

  @doc false
  # lexicographic path order with precedence `prec` (symbol → rank, higher is bigger)
  def lpo_gt(s, t, prec) do
    case {s, t} do
      {{:var, _}, _} -> false
      {{:fn, _, _}, {:var, x}} -> occurs?(x, s)
      {{:fn, f, ss}, {:fn, g, ts}} ->
        Enum.any?(ss, fn si -> si == t or lpo_gt(si, t, prec) end) or
          (rank(prec, f) > rank(prec, g) and Enum.all?(ts, &lpo_gt(s, &1, prec))) or
          (f == g and lex_gt(ss, ts, prec) and Enum.all?(ts, &lpo_gt(s, &1, prec)))
    end
  end

  defp rank(prec, f), do: Map.get(prec, f, 0)
  defp lex_gt([a | as], [b | bs], prec), do: if(a == b, do: lex_gt(as, bs, prec), else: lpo_gt(a, b, prec))
  defp lex_gt(_, _, _), do: false

  # ============================================================ completion

  @doc """
  Complete a theory given as text (see the moduledoc):
  `{:ok, %{rules, steps, critical_pairs, precedence}}` or `{:error, why}`
  (an equation neither side of which is bigger: the order cannot orient
  it; or the budget ran out — completion need not terminate).
  """
  def complete(text) do
    with {:ok, eqs, vars, prec} <- parse(text) do
      case huet(eqs, [], prec, 0, 0) do
        {:ok, rules, steps, cps} ->
          {:ok, %{rules: rules, rules_text: Enum.map(rules, fn {l, r} -> "#{show(l)} → #{show(r)}" end), steps: steps, critical_pairs: cps,
                  precedence: prec |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.map(&elem(&1, 0)), vars: MapSet.to_list(vars)}}
        e -> e
      end
    end
  end

  @doc "Parse a theory: `{:ok, equations, variables, precedence}`."
  def parse(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#", parts: 2) |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    {decl, rest} = Enum.split_with(lines, &(&1 =~ ~r/^(vars|precedence|variáveis|precedência)\b/u))
    vars_decl = Enum.find_value(decl, fn l -> if l =~ ~r/^(vars|variáveis)\b/u, do: l |> String.split() |> tl() end)
    prec_decl = Enum.find_value(decl, fn l -> if l =~ ~r/^(precedence|precedência)\b/u, do: l |> String.replace(~r/^\S+\s*/u, "") |> String.split(">") |> Enum.map(&String.trim/1) end)
    idents = rest |> Enum.flat_map(&lex/1) |> Enum.filter(&(&1 =~ ~r/^[\p{L}_]/u)) |> Enum.uniq()
    vars = MapSet.new(vars_decl || Enum.filter(idents, &(&1 =~ ~r/^[u-z]\d*'*$/)))

    eqs =
      Enum.reduce_while(rest, {:ok, []}, fn l, {:ok, acc} ->
        case String.split(l, "=", parts: 2) do
          [a, b] -> (with {:ok, x} <- parse_term(a, vars), {:ok, y} <- parse_term(b, vars) do {:cont, {:ok, acc ++ [{x, y}]}} else e -> {:halt, e} end)
          _ -> {:halt, {:error, "an equation s = t expected: #{l}"}}
        end
      end)

    with {:ok, eqs} <- eqs do
      syms = eqs |> Enum.flat_map(fn {a, b} -> symbols(a) ++ symbols(b) end) |> Enum.uniq()
      order = prec_decl || (syms |> Enum.with_index() |> Enum.sort_by(fn {{_, ar}, i} -> {-ar, i} end) |> Enum.map(fn {{f, _}, _} -> f end))
      missing = Enum.map(syms, &elem(&1, 0)) -- order
      order = order ++ missing
      n = length(order)
      {:ok, eqs, vars, order |> Enum.with_index() |> Map.new(fn {f, i} -> {f, n - i} end)}
    end
  end

  defp symbols({:var, _}), do: []
  defp symbols({:fn, f, as}), do: [{f, length(as)} | Enum.flat_map(as, &symbols/1)]

  # Huet's procedure: equations E, rules R
  defp huet([], rules, _prec, steps, cps), do: {:ok, rules, steps, cps}

  defp huet(eqs, rules, prec, steps, cps) do
    cond do
      steps > @max_steps -> {:error, "completion did not finish in #{@max_steps} steps (it need not terminate for every theory)"}
      length(rules) > @max_rules -> {:error, "more than #{@max_rules} rules: completion is diverging"}
      true ->
        # the smallest equation first
        [{a, b} | rest] = Enum.sort_by(eqs, fn {a, b} -> size(a) + size(b) end)
        {na, _} = normalize(a, rules)
        {nb, _} = normalize(b, rules)
        cond do
          na == nb -> huet(rest, rules, prec, steps + 1, cps)
          lpo_gt(na, nb, prec) -> add_rule({na, nb}, rest, rules, prec, steps, cps)
          lpo_gt(nb, na, prec) -> add_rule({nb, na}, rest, rules, prec, steps, cps)
          true -> {:error, "cannot orient #{show(na)} = #{show(nb)} with this precedence (try another, or the theory needs ordered completion)"}
        end
    end
  end

  # variables renamed x, y, z, u, v, w… in order of appearance (the rules read as written by hand)
  defp canon({l, r}) do
    names = ~w(x y z u v w) ++ Enum.map(1..50, &"x#{&1}")
    vs = Enum.uniq(tvars(l) ++ tvars(r))
    s = vs |> Enum.zip(names) |> Map.new(fn {v, n} -> {v, {:var, n}} end)
    {apply_s(l, s), apply_s(r, s)}
  end

  defp add_rule(rule, eqs, rules, prec, steps, cps) do
    {l, r} = new = canon(rule)
    # inter-reduce: rules whose left side the new rule rewrites go back to the equations; right sides normalised
    {keep, back} = Enum.split_with(rules, fn {l2, _} -> rewrite_once(l2, [new]) == nil end)
    keep = Enum.map(keep, fn {l2, r2} -> {l2, elem(normalize(r2, [new | keep] |> Enum.uniq()), 0)} end)
    rules2 = keep ++ [{l, r}]
    pairs = critical_pairs(new, rules2)
    huet(eqs ++ back ++ pairs, rules2, prec, steps + 1, cps + length(pairs))
  end

  # every overlap of `rule` with each rule (both ways), at non-variable positions
  defp critical_pairs(rule, rules) do
    Enum.flat_map(rules, fn other -> overlaps(rename(rule, "'"), rename(other, "\"")) ++ overlaps(rename(other, "\""), rename(rule, "'")) end)
    |> Enum.uniq()
  end

  defp overlaps({l1, r1}, {l2, r2}) do
    for {sub, pos} <- subterms(l1), not match?({:var, _}, sub), {:ok, s} <- [unify(sub, l2)],
        a = resolve(replace(l1, pos, r2), s), b = resolve(r1, s), a != b, do: {a, b}
  end

  defp subterms(t, pos \\ [])
  defp subterms({:var, _} = t, pos), do: [{t, Enum.reverse(pos)}]
  defp subterms({:fn, _, as} = t, pos), do: [{t, Enum.reverse(pos)} | Enum.flat_map(Enum.with_index(as), fn {a, i} -> subterms(a, [i | pos]) end)]

  defp replace(_t, [], u), do: u
  defp replace({:fn, f, as}, [i | p], u), do: {:fn, f, List.update_at(as, i, &replace(&1, p, u))}

  # ============================================================== deciding

  @doc """
  Decide `s = t` modulo a completed system: `%{equal, normal_forms,
  derivations}` — both terms rewritten to normal form with every step.
  """
  def decide(rules, s_text, t_text, vars \\ ~w(u v w x y z)) do
    vs = MapSet.new(vars)
    with {:ok, s} <- parse_term(s_text, vs), {:ok, t} <- parse_term(t_text, vs) do
      {ns, ds} = normalize(s, rules)
      {nt, dt} = normalize(t, rules)
      show_d = fn d -> Enum.map(d, fn {i, u} -> %{rule: i, term: show(u)} end) end
      {:ok, %{equal: ns == nt, normal_forms: [show(ns), show(nt)], derivations: [show_d.(ds), show_d.(dt)]}}
    end
  end
end
