defmodule Vapor.Logic do
  @moduledoc """
  The logic desk (docs/LOGIC.md): claims of any of four logics typed in
  as text, each settled by a procedure that proposes and a checker that
  decides — and the same desk open to any outside proposer (a person, a
  search, a language model through the MCP server's `logic_check` tool),
  because acceptance never depends on who proposed.

  | text | logic | verdict and certificate |
  |---|---|---|
  | `p cnf …` (DIMACS) | propositional, CNF | model (checked by evaluation) or DRUP refutation (checked by `Vapor.Logic.DRUP`) |
  | `valid: (p -> q) -> (!q -> !p)` / `sat: …` / `equiv: a ; b` | propositional formulas | the same, via Tseitin |
  | `schur 3`, `vdw 3 2`, `ramsey 3 3`, … (with or without n) | finite combinatorics | the number: a model below it, a refutation at it |
  | equations + `complete` / `decide s = t` | equational theories | a convergent rewriting system; normal forms with derivations |
  | `vars …` `hyp …` `claim …` | polynomial (real/complex algebraic geometry) | Gröbner certificate {1}, or the remainder |
  | `maximize …` / `minimize …` + linear constraints | linear arithmetic over ℚ | exact simplex: optimal (primal + dual, zero gap), infeasible (Farkas), unbounded (ray) — `Vapor.Logic.LP` |
  | the same + `int x, y` / `bin z` | mixed-integer linear arithmetic | branch and bound: the tree, with a dual or Farkas certificate at every leaf — `Vapor.Logic.MIP` |
  | `causal` + edges `a -> b`, `a <-> b` + `identify y \| do(x)`, `backdoor x -> y \| z`, `dsep a; b \| c` | causal diagrams (do-calculus) | the estimand by the ID algorithm or a checked hedge; back-door validity or the open path; d-separation or the connecting path — `Vapor.Logic.Causal` |
  """
  alias Vapor.Logic.{Causal, DRUP, Formula, Groebner, LP, MIP, Problems, Rewrite, SAT}

  @integer_decl ~S"^\s*(int|integer|inteiro|inteiros|bin|binary|binario|binário)\s+"

  @doc "Settle a claim given as text. `{:ok, %{kind, verdict, …}}` or `{:error, why}`."
  def run(text) when is_binary(text) do
    t = String.trim(text)
    first = t |> String.split("\n") |> hd() |> String.trim() |> String.downcase()

    cond do
      first in ["causal", "causal:"] -> Causal.run(t)
      t =~ ~r/^\s*p\s+cnf\b/m or first =~ ~r/^c\s/ -> dimacs(t)
      first =~ ~r/^(max|maximi[sz]e|maximizar|min|minimi[sz]e|minimizar)\b/ and integer?(t) -> integer_linear(t)
      first =~ ~r/^(max|maximi[sz]e|maximizar|min|minimi[sz]e|minimizar)\b/ -> linear(t)
      first =~ ~r/^(valid|válida|tautology|tautologia)\s*:/ -> formula(:valid, rest(t))
      first =~ ~r/^(sat|satisfy|satisfaz)\s*:/ -> formula(:sat, rest(t))
      first =~ ~r/^(equiv|equivalent|equivalentes)\s*:/ -> formula(:equiv, rest(t))
      first =~ ~r/^(schur|vdw|waerden|ramsey|pigeonhole|pombos|queens|rainhas)\b/ -> combinatorics(first)
      t =~ ~r/^\s*(hyp|hipótese|claim|tese|member)\b/m -> Groebner.run(t) |> tag("polynomial")
      t =~ ~r/=/ -> equational(t)
      true -> {:error, "not recognised: start with 'p cnf', 'valid:', 'sat:', 'equiv:', 'schur', 'vdw', 'ramsey', 'pigeonhole', 'queens', 'causal', 'maximize'/'minimize', polynomial lines (vars/hyp/claim) or equations"}
    end
  end

  @doc """
  The checker side of the desk, for an outside proposer (a person, a
  search, a language model through the MCP tool `logic_check`): the claim
  as text and a **proposal** — the checker decides, never the proposer.

  | claim | proposal | accepted when |
  |---|---|---|
  | DIMACS | `%{"model" => [literals or true variables]}` | every clause has a true literal |
  | DIMACS | `%{"drup" => [[lits]…]}` | `Vapor.Logic.DRUP` verifies the refutation (empty clause derived) |
  | `schur k` | `%{"witness" => [colours of 1..n]}` | colours in 1..k, no x + y = z monochromatic → S(k) ≥ n |
  | `vdw k r` | `%{"witness" => [colours of 1..n]}` | colours in 1..r, no monochromatic k-term progression → W(k; r) > n |
  | `ramsey s t` | `%{"n" => n, "red" => [[a, b]…]}` | no red K_s, no blue K_t → R(s, t) > n |
  | `sat: φ` / `valid: φ` | `%{"assignment" => %{var => bool}}` | φ true (a model) / φ false (a counterexample: validity refuted) |
  | `maximize`/`minimize` LP | `%{"x" => %{var => "p/q"}, "y" => ["p/q"…]}` | x feasible, y dual-feasible, zero duality gap — exactly (optimality) |
  | `maximize`/`minimize` LP | `%{"farkas" => ["p/q"…]}` | Aᵀy ≥ 0 with the row signs, bᵀy < 0 (infeasibility) |
  | `maximize`/`minimize` LP | `%{"x" => …, "ray" => %{var => "p/q"}}` | x feasible, d an improving recession direction (unboundedness) |
  | the same with `int`/`bin` (MIP) | `%{"incumbent" => %{var => "p/q"} or nil, "objective" => "p/q", "tree" => node}`, a node `%{"split", "at", "le", "ge"}` or `%{"leaf" => "infeasible" \| "bound", "y" => […]}` | `Vapor.Logic.MIP.check/2`: the splits cover every integer point, every leaf's y holds, the incumbent is feasible, integral and as good as claimed |

  `{:ok, %{accepted, claim, reason}}` or `{:error, why}` when no checker fits.
  """
  def check(text, proposal) when is_binary(text) and is_map(proposal) do
    t = String.trim(text)
    first = t |> String.split("\n") |> hd() |> String.trim() |> String.downcase()
    ints = fn l -> is_list(l) and Enum.all?(l, &is_integer/1) end

    cond do
      t =~ ~r/^\s*p\s+cnf\b/m or first =~ ~r/^c\s/ ->
        with {:ok, cnf} <- SAT.dimacs(t) do
          cond do
            ints.(proposal["model"]) ->
              trues = proposal["model"] |> Enum.filter(&(&1 > 0)) |> MapSet.new()
              bad = Enum.find(cnf.clauses, fn c -> not Enum.any?(c, fn l -> MapSet.member?(trues, abs(l)) == (l > 0) end) end)
              {:ok, %{accepted: bad == nil, claim: "satisfiable", reason: if(bad, do: "clause #{inspect(bad)} is false under the proposed model", else: "every clause has a true literal")}}

            is_list(proposal["drup"]) and Enum.all?(proposal["drup"], ints) ->
              case DRUP.check(cnf, proposal["drup"]) do
                {:ok, m} -> {:ok, %{accepted: true, claim: "unsatisfiable", reason: "DRUP refutation verified: #{m.checked} lemmas checked by unit propagation"}}
                {:error, e} -> {:ok, %{accepted: false, claim: "unsatisfiable", reason: "refutation rejected: #{inspect(e)}"}}
              end

            true -> {:error, "a CNF takes model: [literals] or drup: [[literals]…]"}
          end
        end

      first =~ ~r/^schur\s+(\d+)/ and ints.(proposal["witness"]) ->
        [_, k] = Regex.run(~r/^schur\s+(\d+)/, first); k = String.to_integer(k); w = proposal["witness"]
        ok = w != [] and Enum.all?(w, &(&1 in 1..k)) and Problems.schur_ok?(w)
        {:ok, %{accepted: ok, claim: "S(#{k}) ≥ #{length(w)}", reason: if(ok, do: "no monochromatic x + y = z among 1..#{length(w)}", else: "a monochromatic solution of x + y = z, or a colour outside 1..#{k}")}}

      first =~ ~r/^(vdw|waerden)\s+(\d+)\s+(\d+)/ and ints.(proposal["witness"]) ->
        [_, _, k, r] = Regex.run(~r/^(vdw|waerden)\s+(\d+)\s+(\d+)/, first); {k, r} = {String.to_integer(k), String.to_integer(r)}; w = proposal["witness"]
        ok = w != [] and Enum.all?(w, &(&1 in 1..r)) and Problems.vdw_ok?(w, k)
        {:ok, %{accepted: ok, claim: "W(#{k}; #{r}) > #{length(w)}", reason: if(ok, do: "no monochromatic #{k}-term progression in 1..#{length(w)}", else: "a monochromatic #{k}-term progression, or a colour outside 1..#{r}")}}

      first =~ ~r/^ramsey\s+(\d+)\s+(\d+)/ and is_integer(proposal["n"]) and is_list(proposal["red"]) ->
        [_, a, b] = Regex.run(~r/^ramsey\s+(\d+)\s+(\d+)/, first); {a, b} = {String.to_integer(a), String.to_integer(b)}; n = proposal["n"]
        ok = n >= 1 and n <= 40 and Enum.all?(proposal["red"], &(ints.(&1) and length(&1) == 2 and Enum.all?(&1, fn v -> v in 1..n end))) and Problems.ramsey_ok?(proposal["red"], a, b, n)
        {:ok, %{accepted: ok, claim: "R(#{a}, #{b}) > #{n}", reason: if(ok, do: "no red K#{a} and no blue K#{b} on #{n} vertices", else: "a monochromatic clique, or a malformed colouring")}}

      first =~ ~r/^(sat|satisfy|satisfaz|valid|válida|tautology|tautologia)\s*:/ and is_map(proposal["assignment"]) ->
        with {:ok, f} <- Formula.parse(rest(t)) do
          a = Map.new(proposal["assignment"], fn {k, v} -> {to_string(k), v == true} end)
          missing = Formula.vars(f) -- Map.keys(a)
          if missing != [] do
            {:error, "assignment misses #{Enum.join(missing, ", ")}"}
          else
            v = Formula.eval(f, a)
            if first =~ ~r/^(sat|satisfy|satisfaz)/,
              do: {:ok, %{accepted: v, claim: "satisfiable", reason: if(v, do: "the formula is true under the assignment", else: "the formula is false under it")}},
              else: {:ok, %{accepted: not v, claim: "not valid", reason: if(v, do: "the formula is true under it: not a counterexample", else: "a counterexample: the formula is false under it")}}
          end
        end

      first =~ ~r/^(max|maximi[sz]e|maximizar|min|minimi[sz]e|minimizar)\b/ and integer?(t) ->
        with {:ok, p} <- MIP.parse(t), {:ok, cert} <- mip_proposal(proposal) do
          r = MIP.check(p, cert)
          {:ok, %{accepted: r.accepted, claim: if(cert.incumbent, do: "optimal", else: "infeasible"), reason: r.reason}}
        end

      first =~ ~r/^(max|maximi[sz]e|maximizar|min|minimi[sz]e|minimizar)\b/ ->
        with {:ok, p} <- LP.parse(t) do
          rat = fn v -> try do LP.rat(if is_number(v), do: v, else: to_string(v)) rescue _ -> nil end end
          xmap = fn m -> if is_map(m), do: Map.new(m, fn {k, v} -> {to_string(k), rat.(v)} end), else: nil end
          ylist = fn l -> if is_list(l), do: Enum.map(l, rat), else: nil end
          cert =
            cond do
              is_list(proposal["farkas"]) -> %{status: :infeasible, y: ylist.(proposal["farkas"])}
              is_map(proposal["ray"]) -> %{status: :unbounded, x: xmap.(proposal["x"]) || %{}, ray: xmap.(proposal["ray"])}
              is_map(proposal["x"]) and is_list(proposal["y"]) -> %{status: :optimal, x: xmap.(proposal["x"]), y: ylist.(proposal["y"])}
              true -> nil
            end
          bad = cert == nil or Enum.any?(List.wrap(cert[:y]) ++ Map.values(cert[:x] || %{}) ++ Map.values(cert[:ray] || %{}), &is_nil/1)
          if bad do
            {:error, "an LP takes x + y (optimality), farkas (infeasibility) or x + ray (unboundedness), numbers as \"p/q\" or decimals"}
          else
            r = LP.check(p, cert)
            {:ok, %{accepted: r.accepted, claim: Atom.to_string(cert.status), reason: r.reason}}
          end
        end

      true -> {:error, "no checker for this claim and proposal (see Vapor.Logic.check/2)"}
    end
  end

  defp integer?(t), do: t |> String.split("\n") |> Enum.any?(&(&1 |> String.split("#") |> hd() |> String.match?(Regex.compile!(@integer_decl, "i"))))

  defp integer_linear(t) do
    case MIP.solve(t) do
      {:ok, _} = r ->
        v = MIP.present(r)
        {:ok, Map.merge(v, %{kind: "integer linear", verdict: Atom.to_string(v.status), certified: match?(%{accepted: true}, v[:check])})}

      e ->
        e
    end
  end

  # a MIP certificate from JSON: rationals as "p/q" or decimals, the tree as nested maps
  defp mip_proposal(prop) do
    rat = fn v -> try do LP.rat(if is_number(v), do: v, else: to_string(v)) rescue _ -> throw(:bad) end end

    tree = fn
      %{"split" => v, "at" => k, "le" => lo, "ge" => hi}, f when is_binary(v) and is_integer(k) -> {:branch, v, k, f.(lo, f), f.(hi, f)}
      %{"leaf" => kind, "y" => y}, _f when kind in ["infeasible", "bound"] and is_list(y) -> {:leaf, String.to_existing_atom(kind), Enum.map(y, rat)}
      _, _ -> throw(:bad)
    end

    inc =
      case prop["incumbent"] do
        nil -> nil
        m when is_map(m) -> %{x: Map.new(m, fn {k, v} -> {to_string(k), rat.(v)} end), objective: rat.(prop["objective"])}
        _ -> throw(:bad)
      end

    {:ok, %{incumbent: inc, tree: tree.(prop["tree"], tree)}}
  catch
    :bad -> {:error, "a MIP proposal is incumbent (a map, or null for infeasible) + objective + tree (split/at/le/ge or leaf/y), numbers as \"p/q\" or decimals"}
  end

  defp linear(t) do
    case LP.solve(t) do
      {:ok, _} = r ->
        v = LP.present(r)
        {:ok, Map.merge(v, %{kind: "linear", verdict: Atom.to_string(v.status), certified: v.check.accepted})}
      e -> e
    end
  end

  defp rest(t), do: t |> String.split(":", parts: 2) |> List.last() |> String.trim()
  defp tag({:ok, r}, k), do: {:ok, Map.put(r, :kind, k)}
  defp tag(e, _), do: e

  # ---------------------------------------------------------------- SAT

  defp dimacs(t) do
    with {:ok, cnf} <- SAT.dimacs(t) do
      if cnf.vars > 5000 or length(cnf.clauses) > 50_000, do: {:error, "at most 5 000 variables and 50 000 clauses here"}, else: settle(cnf, nil, "cnf")
    end
  end

  @doc false
  def settle(cnf, check_model, kind, opts \\ []) do
    {us, r} = :timer.tc(fn -> SAT.solve(cnf, conflicts: Keyword.get(opts, :conflicts, 500_000)) end)
    case r do
      {:sat, m, st} ->
        ok = Enum.all?(cnf.clauses, fn c -> Enum.any?(c, fn l -> m[abs(l)] == (l > 0) end) end)
        extra = if check_model, do: check_model.(m), else: nil
        {:ok, %{kind: kind, verdict: "satisfiable", model: m |> Enum.filter(fn {_, v} -> v end) |> Enum.map(&elem(&1, 0)) |> Enum.sort(),
                certificate: %{model_checked: ok, extra: extra}, stats: st, ms: div(us, 1000)}}

      {:unsat, proof, st} ->
        {cus, check} = if Keyword.get(opts, :check, true), do: :timer.tc(fn -> DRUP.check(cnf, proof) end), else: {0, :skipped}
        {:ok, %{kind: kind, verdict: "unsatisfiable", certificate: %{drup: check_json(check), lemmas: length(proof), check_ms: div(cus, 1000)}, stats: st, ms: div(us, 1000),
                proof_head: proof |> Enum.take(12)}}

      {:unknown, _, st} -> {:ok, %{kind: kind, verdict: "unknown (budget exhausted)", stats: st, ms: div(us, 1000)}}
    end
  end

  defp check_json({:ok, m}), do: Map.put(m, :valid, true)
  defp check_json({:error, e}), do: %{valid: false, error: inspect(e)}
  defp check_json(:skipped), do: %{valid: nil, skipped: true}

  # -------------------------------------------------------------- formulas

  defp formula(:equiv, s) do
    case String.split(s, ";", parts: 2) do
      [a, b] -> (with {:ok, f} <- Formula.parse(a), {:ok, g} <- Formula.parse(b), do: verdict(Formula.equivalent?(f, g), "equivalence", "#{Formula.to_text(f)} ≡ #{Formula.to_text(g)}"))
      _ -> {:error, "equiv: a ; b"}
    end
  end

  defp formula(:valid, s), do: with({:ok, f} <- Formula.parse(s), do: verdict(Formula.valid?(f), "validity", Formula.to_text(f)))

  defp formula(:sat, s) do
    with {:ok, f} <- Formula.parse(s) do
      case Formula.satisfy(f) do
        {:sat, r} -> {:ok, %{kind: "formula", claim: Formula.to_text(f), verdict: "satisfiable", model: r.assignment, certificate: %{evaluates_to: r.checked}}}
        {:unsat, c} -> {:ok, %{kind: "formula", claim: Formula.to_text(f), verdict: "unsatisfiable", certificate: %{drup: check_json(c.drup), lemmas: c.proof_lemmas}}}
        {:unknown, _} -> {:ok, %{kind: "formula", verdict: "unknown"}}
      end
    end
  end

  defp verdict({:proved, c}, what, claim), do: {:ok, %{kind: "formula", question: what, claim: claim, verdict: "proved", certificate: %{drup: check_json(c.drup), lemmas: c.proof_lemmas}}}
  defp verdict({:refuted, c}, what, claim), do: {:ok, %{kind: "formula", question: what, claim: claim, verdict: "refuted", counterexample: c.counterexample}}
  defp verdict(other, what, claim), do: {:ok, %{kind: "formula", question: what, claim: claim, verdict: inspect(elem(other, 0))}}

  # --------------------------------------------------------- combinatorics

  @doc """
  The number behind a Ramsey-type problem: `schur k`, `vdw k r`, `ramsey
  s t` (searches n upward until the encoding is refuted, returning the
  last model and the refutation), or the same with an explicit `n` to
  settle one instance. Bounded so a request cannot run for hours.
  """
  def combinatorics(line) do
    nums = Regex.scan(~r/\d+/, line) |> Enum.map(fn [d] -> String.to_integer(d) end)
    kind = line |> String.split() |> hd()
    case {kind, nums} do
      {"schur", [k]} when k in 1..3 -> threshold(fn n -> Problems.schur(n, k) end, fn p, m -> Problems.schur_ok?(p.decode.(m)) end, 1, 60, "Schur number S(#{k})", -1)
      {"schur", [k, n]} when k in 1..4 and n in 1..60 -> one(Problems.schur(n, k), fn p, m -> Problems.schur_ok?(p.decode.(m)) end)
      {w, [k, r]} when w in ["vdw", "waerden"] and k in 2..4 and r in 1..3 -> threshold(fn n -> Problems.vdw(k, r, n) end, fn p, m -> Problems.vdw_ok?(p.decode.(m), k) end, k, 40, "van der Waerden W(#{k}; #{r})")
      {w, [k, r, n]} when w in ["vdw", "waerden"] and n in 1..60 -> one(Problems.vdw(k, r, n), fn p, m -> Problems.vdw_ok?(p.decode.(m), k) end)
      {"ramsey", [s, t]} when s in 2..3 and t in 2..4 -> threshold(fn n -> Problems.ramsey(s, t, n) end, fn p, m -> Problems.ramsey_ok?(p.decode.(m), s, t, p.vars |> then(fn v -> round((1 + :math.sqrt(1 + 8 * v)) / 2) end)) end, max(s, t), 12, "Ramsey R(#{s}, #{t})")
      {"ramsey", [s, t, n]} when n in 2..12 -> one(Problems.ramsey(s, t, n), fn p, m -> Problems.ramsey_ok?(p.decode.(m), s, t, n) end)
      {w, [p, h]} when w in ["pigeonhole", "pombos"] and p <= 9 and h <= 8 -> one(Problems.pigeonhole(p, h), nil)
      {w, [n]} when w in ["queens", "rainhas"] and n in 1..30 -> one(Problems.queens(n), nil)
      _ -> {:error, "schur k [n] (k ≤ 3 to search), vdw k r [n], ramsey s t [n] (s ≤ 3, t ≤ 4), pigeonhole p h, queens n — within the bounds this desk runs"}
    end
  end

  defp one(p, check) do
    with {:ok, r} <- settle(p, check && fn m -> check.(p, m) end, "combinatorics") do
      r = if r.verdict == "satisfiable", do: Map.put(r, :witness, p.decode.(Map.new(r.model, &{&1, true}) |> fill(p.vars))), else: r
      {:ok, Map.put(r, :problem, p.name)}
    end
  end

  defp fill(m, n), do: Map.merge(Map.new(1..max(n, 1), &{&1, false}), m)

  # the number is the first refuted n (van der Waerden, Ramsey) or the last satisfiable one (Schur: offset −1)
  defp threshold(gen, check, from, upto, name, offset \\ 0) do
    Enum.reduce_while(from..upto, nil, fn n, last ->
      p = gen.(n)
      {:ok, r} = settle(p, fn m -> check.(p, m) end, "combinatorics", conflicts: 200_000)
      case r.verdict do
        "satisfiable" -> {:cont, %{n: n, witness: p.decode.(Map.new(r.model, &{&1, true}) |> fill(p.vars)), checked: r.certificate.extra}}
        "unsatisfiable" -> {:halt, {:ok, %{kind: "combinatorics", problem: name, verdict: "#{name} = #{n + offset}", value: n + offset, refuted_at: n, below: last, refutation: r.certificate, stats: r.stats}}}
        _ -> {:halt, {:ok, %{kind: "combinatorics", problem: name, verdict: "undecided at n = #{n} within the budget", below: last}}}
      end
    end)
    |> case do
      {:ok, _} = ok -> ok
      _ -> {:ok, %{kind: "combinatorics", problem: name, verdict: "no refutation up to n = #{upto}"}}
    end
  end

  # ------------------------------------------------------------ equations

  defp equational(t) do
    lines = String.split(t, "\n")
    {decide, theory} = Enum.split_with(lines, &(&1 =~ ~r/^\s*(decide|decida)\b/i))
    theory = theory |> Enum.reject(&(String.trim(&1) =~ ~r/^(complete|complete:|completar)$/i)) |> Enum.join("\n")
    with {:ok, r} <- Rewrite.complete(theory) do
      {:ok, _, vars, _} = Rewrite.parse(theory)
      answers =
        for d <- decide do
          case String.split(Regex.replace(~r/^\s*(decide|decida)\s*/i, d, ""), "=", parts: 2) do
            [a, b] -> (case Rewrite.decide(r.rules, String.trim(a), String.trim(b), MapSet.to_list(vars)) do {:ok, x} -> Map.put(x, :question, String.trim(d)); {:error, w} -> %{question: d, error: w} end)
            _ -> %{question: d, error: "decide s = t"}
          end
        end
      {:ok, %{kind: "equational", verdict: "completed: #{length(r.rules)} rules", rules: r.rules_text, steps: r.steps, critical_pairs: r.critical_pairs, precedence: r.precedence, decisions: answers}}
    end
  end
end
