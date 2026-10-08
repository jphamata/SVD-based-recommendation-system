defmodule Vapor.Logic.Formula do
  @moduledoc """
  Propositional formulas typed in as text (docs/LOGIC.md §3): `!a`,
  `a & b`, `a | b`, `a -> b`, `a <-> b`, `a ^ b` (xor), `true`, `false`
  (and ¬ ∧ ∨ → ↔ ⊕). A formula becomes CNF by Tseitin's transformation
  (linear size, equisatisfiable), so any claim of propositional logic is
  settled by the SAT solver:

    * `valid?/1` — a tautology iff its negation is unsatisfiable: the
      answer is **proved** with a DRUP certificate, or **refuted** with a
      falsifying assignment (checked by evaluating the formula);
    * `equivalent?/2` — two formulas, the same truth function;
    * `satisfy/1` — a model, checked by evaluation.
  """
  alias Vapor.Logic.{DRUP, SAT}

  @doc "Parse: `{:ok, tree}` with `{:var, name} | {:not, a} | {:and | :or | :imp | :iff | :xor, a, b} | true | false`."
  def parse(text) do
    toks = lex(String.to_charlist(text), [])
    case iff(toks) do
      {t, []} -> {:ok, t}
      {_, [tok | _]} -> {:error, "unexpected #{inspect(tok)}"}
    end
  catch
    {:fparse, w} -> {:error, w}
  end

  defp lex([], acc), do: Enum.reverse(acc)
  defp lex([c | r], acc) when c in [?\s, ?\t, ?\n], do: lex(r, acc)
  defp lex([?<, ?-, ?> | r], acc), do: lex(r, [:iff | acc])
  defp lex([?<, ?=, ?> | r], acc), do: lex(r, [:iff | acc])
  defp lex([?-, ?> | r], acc), do: lex(r, [:imp | acc])
  defp lex([?=, ?> | r], acc), do: lex(r, [:imp | acc])
  defp lex([?& , ?& | r], acc), do: lex(r, [:and | acc])
  defp lex([?|, ?| | r], acc), do: lex(r, [:or | acc])
  defp lex([c | r], acc) when c in [?!, ?~, ?¬], do: lex(r, [:not | acc])
  defp lex([c | r], acc) when c in [?&, ?∧, ?*], do: lex(r, [:and | acc])
  defp lex([c | r], acc) when c in [?|, ?∨, ?+], do: lex(r, [:or | acc])
  defp lex([c | r], acc) when c in [?→], do: lex(r, [:imp | acc])
  defp lex([c | r], acc) when c in [?↔], do: lex(r, [:iff | acc])
  defp lex([c | r], acc) when c in [?^, ?⊕], do: lex(r, [:xor | acc])
  defp lex([?( | r], acc), do: lex(r, [:lp | acc])
  defp lex([?) | r], acc), do: lex(r, [:rp | acc])
  defp lex([c | _] = l, acc) when c in ?a..?z or c in ?A..?Z or c == ?_ or c in ?0..?9 do
    {w, r} = Enum.split_while(l, &(&1 in ?a..?z or &1 in ?A..?Z or &1 == ?_ or &1 in ?0..?9))
    tok = case to_string(w) do
      x when x in ["true", "T", "1", "verdadeiro"] -> true
      x when x in ["false", "F", "0", "falso"] -> false
      x when x in ["and", "e"] -> :and
      x when x in ["or", "ou"] -> :or
      x when x in ["not", "nao", "não"] -> :not
      x when x in ["implies"] -> :imp
      x when x in ["iff"] -> :iff
      x when x in ["xor"] -> :xor
      x -> {:var, x}
    end
    lex(r, [tok | acc])
  end
  defp lex([c | _], _), do: throw({:fparse, "unexpected character #{inspect(to_string([c]))}"})

  defp iff(t), do: (({a, r} = imp(t)); case r do [:iff | r2] -> (({b, r3} = iff(r2)); {{:iff, a, b}, r3}); _ -> {a, r} end)
  defp imp(t), do: (({a, r} = disj(t)); case r do [:imp | r2] -> (({b, r3} = imp(r2)); {{:imp, a, b}, r3}); _ -> {a, r} end)
  defp disj(t), do: (({a, r} = xor(t)); more(a, r, :or, &xor/1))
  defp xor(t), do: (({a, r} = conj(t)); more(a, r, :xor, &conj/1))
  defp conj(t), do: (({a, r} = neg(t)); more(a, r, :and, &neg/1))
  defp more(a, [op | r], op, next), do: (({b, r2} = next.(r)); more({op, a, b}, r2, op, next))
  defp more(a, r, _, _), do: {a, r}
  defp neg([:not | r]), do: (({a, r2} = neg(r)); {{:not, a}, r2})
  defp neg([:lp | r]), do: (case iff(r) do {a, [:rp | r2]} -> {a, r2}; _ -> throw({:fparse, "missing )"}) end)
  defp neg([{:var, _} = v | r]), do: {v, r}
  defp neg([b | r]) when is_boolean(b), do: {b, r}
  defp neg([t | _]), do: throw({:fparse, "unexpected #{inspect(t)}"})
  defp neg([]), do: throw({:fparse, "unexpected end"})

  @doc "The variables of a formula, sorted."
  def vars(f), do: f |> do_vars(MapSet.new()) |> Enum.sort()
  defp do_vars({:var, n}, s), do: MapSet.put(s, n)
  defp do_vars({:not, a}, s), do: do_vars(a, s)
  defp do_vars({_, a, b}, s), do: do_vars(b, do_vars(a, s))
  defp do_vars(_, s), do: s

  @doc "Evaluate under `%{name => boolean}`."
  def eval(true, _), do: true
  def eval(false, _), do: false
  def eval({:var, n}, m), do: Map.fetch!(m, n)
  def eval({:not, a}, m), do: not eval(a, m)
  def eval({:and, a, b}, m), do: eval(a, m) and eval(b, m)
  def eval({:or, a, b}, m), do: eval(a, m) or eval(b, m)
  def eval({:imp, a, b}, m), do: not eval(a, m) or eval(b, m)
  def eval({:iff, a, b}, m), do: eval(a, m) == eval(b, m)
  def eval({:xor, a, b}, m), do: eval(a, m) != eval(b, m)

  @doc "Tseitin CNF of `f` asserted true: `%{vars, clauses, names}` (names: variable → index)."
  def tseitin(f) do
    names = vars(f) |> Enum.with_index(1) |> Map.new()
    {root, {next, cls}} = ts(f, names, {map_size(names) + 1, []})
    %{vars: next - 1, clauses: [[root] | cls] |> Enum.reverse(), names: names}
  end

  defp fresh({n, c}), do: {n, {n + 1, c}}
  defp add({n, c}, cl), do: {n, [cl | c]}

  defp ts({:var, x}, names, st), do: {names[x], st}
  defp ts(true, _names, st), do: (({v, st} = fresh(st)); {v, add(st, [v])})
  defp ts(false, _names, st), do: (({v, st} = fresh(st)); {v, add(st, [-v])})
  defp ts({:not, a}, names, st), do: (({x, st} = ts(a, names, st)); {-x, st})

  defp ts({op, a, b}, names, st) do
    {x, st} = ts(a, names, st)
    {y, st} = ts(b, names, st)
    {v, st} = fresh(st)
    cls = case op do
      :and -> [[-v, x], [-v, y], [v, -x, -y]]
      :or -> [[-v, x, y], [v, -x], [v, -y]]
      :imp -> [[-v, -x, y], [v, x], [v, -y]]
      :iff -> [[-v, -x, y], [-v, x, -y], [v, x, y], [v, -x, -y]]
      :xor -> [[-v, x, y], [-v, -x, -y], [v, -x, y], [v, x, -y]]
    end
    {v, Enum.reduce(cls, st, &add(&2, &1))}
  end

  @doc "Satisfy: `{:sat, assignment}` (checked) or `{:unsat, certificate}`."
  def satisfy(f) do
    cnf = tseitin(f)
    case SAT.solve(cnf) do
      {:sat, m, st} ->
        a = Map.new(cnf.names, fn {n, i} -> {n, m[i]} end)
        {:sat, %{assignment: a, checked: eval(f, a), stats: st}}
      {:unsat, proof, st} -> {:unsat, %{proof_lemmas: length(proof), drup: DRUP.check(cnf, proof), stats: st}}
      {:unknown, _, st} -> {:unknown, %{stats: st}}
    end
  end

  @doc "Is `f` a tautology? `{:proved, cert}` (¬f refuted, DRUP-checked) or `{:refuted, counterexample}`."
  def valid?(f) do
    case satisfy({:not, f}) do
      {:unsat, cert} -> if match?({:ok, _}, cert.drup), do: {:proved, cert}, else: {:checker_rejected, cert}
      {:sat, %{assignment: a}} -> {:refuted, %{counterexample: a, value: eval(f, a)}}
      other -> other
    end
  end

  @doc "Do two formulas define the same truth function?"
  def equivalent?(f, g), do: valid?({:iff, f, g})

  @doc "A formula as text."
  def to_text(true), do: "⊤"
  def to_text(false), do: "⊥"
  def to_text({:var, x}), do: x
  def to_text({:not, a}), do: "¬" <> wrap(a)
  def to_text({op, a, b}), do: wrap(a) <> %{and: " ∧ ", or: " ∨ ", imp: " → ", iff: " ↔ ", xor: " ⊕ "}[op] <> wrap(b)
  defp wrap({op, _, _} = t) when op in [:and, :or, :imp, :iff, :xor], do: "(" <> to_text(t) <> ")"
  defp wrap(t), do: to_text(t)
end
