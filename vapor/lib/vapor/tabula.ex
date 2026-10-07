defmodule Vapor.Tabula do
  @moduledoc """
  The **tabula** — the emerald tablet, the alchemists' text of law — for
  contracts and regulations written as norms over facts (docs/TABULA.md).

  The pain: a contract whose clauses, under some combination of events,
  oblige a party to do what another clause forbids — or to do two things
  that cannot both be done — is found by a court, years later. Whether such
  a combination *exists* is a question of propositional logic, and it can
  be settled before signing.

      parties buyer seller
      facts delivered late defective force_majeure
      exclusive pay withhold
      assume not (late and not delivered)
      C1: if delivered and not defective then buyer must pay seller
      C2: if late then buyer may withhold
      C3: if defective then buyer must not pay
      C4: if force_majeure then seller is exempt from deliver
      C5: seller must deliver buyer
      C4 overrides C5

  Norms (deontic modalities; Hohfeld's positions in parentheses): `must`
  (duty — the counterparty's claim), `must not` (prohibition), `may`
  (privilege), `is exempt from` / `need not` (no duty). Portuguese works
  too: `deve`, `não deve`, `pode`, `está isento de`; `se … então`.

  For every pair of clauses on the same party and action whose modalities
  clash — duty/prohibition, prohibition/privilege, duty/exemption, or two
  duties on actions declared `exclusive` — the question "can both
  conditions hold, given the `assume`d background facts?" goes to the SAT
  solver (`Vapor.Logic.Formula`, Tseitin): **satisfiable** is an antinomy
  with the scenario that triggers it (checked by evaluating the clauses);
  **unsatisfiable** is a proof, checked by `Vapor.Logic.DRUP`, that the
  clash can never arise. Also reported: **silences** (a scenario in which
  no clause says anything about an action that clauses do govern
  elsewhere) — a gap, not an error. A clash between clauses one of which
  `overrides` the other (lex specialis, lex posterior: the person says which)
  is **resolved**, reported with its scenario, not counted as an antinomy;
  a cycle of overrides is refused.

  What this is **not**: an interpreter of natural language or of law. The
  person (or a model, as a draft the person reviews) writes the clauses in
  this form; the tablet decides only what follows from them.
  """
  alias Vapor.Logic.Formula

  defmodule Clause do
    @moduledoc "One norm: `id`, `cond` (a formula over facts, or `true`), `party`, `modality` (`:must | :must_not | :may | :exempt`), `action`, `counterparty`, `text`."
    defstruct [:id, :cond, :party, :modality, :action, :counterparty, :text]
  end

  defstruct parties: [], facts: [], exclusive: [], assume: [], clauses: [], overrides: []

  @max_clauses 400
  @max_facts 200

  # ============================================================ parsing

  @doc "Parse a tablet: `{:ok, tabula}` or `{:error, why}` (with the line)."
  def parse(text) when is_binary(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.map(fn {l, i} -> {l |> String.split("#", parts: 2) |> hd() |> String.trim(), i} end)
    |> Enum.reject(&(elem(&1, 0) == ""))
    |> Enum.reduce_while({:ok, %__MODULE__{}}, fn {line, ln}, {:ok, t} ->
      case line(line, t) do
        {:ok, t} -> {:cont, {:ok, t}}
        {:error, why} -> {:halt, {:error, "line #{ln}: #{why}"}}
      end
    end)
    |> case do
      {:ok, t} -> finish(t)
      e -> e
    end
  end

  defp line(line, t) do
    cond do
      m = Regex.run(~r/^(parties|partes|facts|fatos|exclusive|exclusivos|assume|premissa)\s*:?\s+(.+)$/iu, line) ->
        [_, kw, rest] = m
        header(String.downcase(kw), rest, t)

      m = Regex.run(~r/^([\p{L}\p{N}_.\-]{1,32})\s+(?:overrides|prevalece sobre)\s+([\p{L}\p{N}_.\-]{1,32})$/iu, line) ->
        [_, a, b] = m
        {:ok, %{t | overrides: t.overrides ++ [{a, b}]}}

      true ->
        clause(line, t)
    end
  end

  defp header(kw, rest, t) when kw in ["parties", "partes"], do: names(rest, &%{t | parties: t.parties ++ &1})
  defp header(kw, rest, t) when kw in ["facts", "fatos"], do: names(rest, &%{t | facts: t.facts ++ &1})
  defp header(kw, rest, t) when kw in ["exclusive", "exclusivos"], do: names(rest, &%{t | exclusive: t.exclusive ++ [&1]})

  defp header(_assume, rest, t) do
    with {:ok, f} <- formula(rest), do: {:ok, %{t | assume: t.assume ++ [{rest, f}]}}
  end

  defp names(rest, put) do
    ns = rest |> String.split(~r/[\s,]+/u, trim: true)
    case Enum.find(ns, &(not name?(&1))) do
      nil -> {:ok, put.(ns)}
      bad -> {:error, "#{inspect(bad)} is not a name (letters, digits, _)"}
    end
  end

  defp name?(n), do: Regex.match?(~r/^[\p{L}_][\p{L}\p{N}_]*$/u, n) and String.length(n) <= 64 and String.downcase(n) not in ~w(and or not e ou não nao true false)

  @modal [{~r/^(.+?)\s+(?:must not|shall not|não deve|nao deve)\s+(.+)$/iu, :must_not},
          {~r/^(.+?)\s+(?:is exempt from|need not|está isento de|esta isento de|não precisa|nao precisa)\s+(.+)$/iu, :exempt},
          {~r/^(.+?)\s+(?:must|shall|deve)\s+(.+)$/iu, :must},
          {~r/^(.+?)\s+(?:may|pode)\s+(.+)$/iu, :may}]

  defp clause(line, t) do
    with [_, id, body] <- Regex.run(~r/^([\p{L}\p{N}_.\-]{1,32})\s*:\s*(.+)$/u, line) || {:error, "not a clause (ID: [if … then] party must|must not|may|is exempt from action [counterparty])"},
         {:ok, cond_text, norm} <- split_condition(body),
         {:ok, cond} <- (if cond_text, do: formula(cond_text), else: {:ok, true}),
         {:ok, party, modality, action, counter} <- norm(norm) do
      {:ok, %{t | clauses: t.clauses ++ [%Clause{id: id, cond: cond, party: party, modality: modality, action: action, counterparty: counter, text: line}]}}
    end
  end

  defp split_condition(body) do
    case Regex.run(~r/^(?:if|se)\s+(.+?)\s*,?\s+(?:then|então|entao)\s+(.+)$/iu, body) do
      [_, c, n] -> {:ok, c, n}
      nil -> if Regex.match?(~r/^(if|se)\s/iu, body), do: {:error, "a condition needs 'then' (então)"}, else: {:ok, nil, body}
    end
  end

  defp norm(text) do
    Enum.find_value(@modal, {:error, "no modality: use must, must not, may or is exempt from"}, fn {re, m} ->
      case Regex.run(re, text) do
        [_, party, rest] ->
          case String.split(rest, ~r/\s+/u, trim: true) do
            [a] -> {:ok, String.trim(party), m, a, nil}
            [a, cp] -> {:ok, String.trim(party), m, a, cp}
            _ -> {:error, "the action is one name, optionally followed by the counterparty"}
          end

        nil ->
          nil
      end
    end)
  end

  defp formula(text) do
    case Formula.parse(text) do
      {:ok, f} -> {:ok, f}
      {:error, why} -> {:error, "condition #{inspect(String.slice(text, 0, 40))}: #{why}"}
    end
  end

  defp finish(t) do
    ids = Enum.map(t.clauses, & &1.id)
    used = (Enum.flat_map(t.clauses, &Formula.vars(&1.cond)) ++ Enum.flat_map(t.assume, &Formula.vars(elem(&1, 1)))) |> Enum.uniq()
    parties = Enum.flat_map(t.clauses, &[&1.party | List.wrap(&1.counterparty)]) |> Enum.uniq()
    bad = Enum.find(used, &(&1 not in t.facts))
    badp = if t.parties != [], do: Enum.find(parties, &(&1 not in t.parties))

    cond do
      t.clauses == [] -> {:error, "no clauses"}
      length(t.clauses) > @max_clauses -> {:error, "more than #{@max_clauses} clauses"}
      length(t.facts) > @max_facts -> {:error, "more than #{@max_facts} facts"}
      length(Enum.uniq(ids)) != length(ids) -> {:error, "clause #{hd(ids -- Enum.uniq(ids))} is defined twice"}
      bad != nil -> {:error, "#{bad} is used in a condition but not declared in facts"}
      badp != nil -> {:error, "#{badp} is not a declared party"}
      (o = Enum.find(t.overrides, fn {a, b} -> a not in ids or b not in ids end)) != nil -> {:error, "#{elem(o, 0)} overrides #{elem(o, 1)}: no such clause"}
      cyclic?(t.overrides) -> {:error, "the overrides form a cycle: no clause can prevail"}
      true -> {:ok, t}
    end
  end

  defp cyclic?(edges) do
    nodes = edges |> Enum.flat_map(&Tuple.to_list/1) |> Enum.uniq()
    succ = Enum.group_by(edges, &elem(&1, 0), &elem(&1, 1))
    reach = fn reach, from, seen -> Enum.reduce(Map.get(succ, from, []), seen, fn n, s -> if MapSet.member?(s, n), do: s, else: reach.(reach, n, MapSet.put(s, n)) end) end
    Enum.any?(nodes, fn n -> MapSet.member?(reach.(reach, n, MapSet.new()), n) end)
  end

  # ============================================================ analysis

  @clash %{{:must, :must_not} => {:antinomy, "a duty and a prohibition of the same act"},
           {:must_not, :may} => {:conflict, "a prohibition and a privilege of the same act"},
           {:must, :exempt} => {:conflict, "a duty and an exemption from it"}}

  @doc """
  Every clash, decided. `%{verdict: :consistent | :antinomies, findings,
  silences, checked_pairs}`. Each finding is `%{kind, why, clauses, party,
  actions, scenario}` (the scenario re-evaluated against both clauses);
  `checked_pairs` lists the pairs proved never to clash, each with its DRUP
  check.
  """
  def analyze(%__MODULE__{} = t) do
    pairs =
      for {a, i} <- Enum.with_index(t.clauses), {b, j} <- Enum.with_index(t.clauses), i < j, a.party == b.party,
          kind = clash(a, b, t), kind != nil, do: {a, b, kind}

    background = Enum.reduce(t.assume, true, fn {_, f}, acc -> conj(acc, f) end)

    {findings, proofs} =
      Enum.reduce(pairs, {[], []}, fn {a, b, {kind, why}}, {fs, ps} ->
        f = conj(background, conj(a.cond, b.cond))

        case Formula.satisfy(f) do
          {:sat, %{assignment: asg}} ->
            sc = scenario(t, asg)
            true = holds?(a.cond, sc) and holds?(b.cond, sc)
            kind = if prevails(t, a.id, b.id), do: :resolved, else: kind
            finding = %{kind: kind, why: why, clauses: [a.id, b.id], party: a.party, actions: Enum.uniq([a.action, b.action]), scenario: sc}
            {[(if kind == :resolved, do: Map.put(finding, :prevails, prevails(t, a.id, b.id)), else: finding) | fs], ps}

          {:unsat, cert} ->
            {fs, [%{clauses: [a.id, b.id], drup: match?({:ok, _}, cert.drup), proof_lemmas: cert.proof_lemmas} | ps]}
        end
      end)

    {resolved, findings} = findings |> Enum.reverse() |> Enum.split_with(&(&1.kind == :resolved))

    %{verdict: if(findings == [], do: :consistent, else: :antinomies), findings: findings, resolved: resolved, checked_pairs: Enum.reverse(proofs),
      silences: silences(t, background), clauses: length(t.clauses)}
  end

  defp clash(a, b, t) do
    cond do
      a.action == b.action -> Map.get(@clash, {a.modality, b.modality}) || Map.get(@clash, {b.modality, a.modality})
      a.modality == :must and b.modality == :must and exclusive?(t, a.action, b.action) -> {:antinomy, "two duties that cannot both be done"}
      true -> nil
    end
  end

  # the clause that prevails between two, if an override (direct or by a chain) says so
  defp prevails(t, a, b) do
    cond do
      above?(t.overrides, a, b) -> a
      above?(t.overrides, b, a) -> b
      true -> nil
    end
  end

  defp above?(edges, a, b), do: above?(edges, a, b, MapSet.new())

  defp above?(edges, a, b, seen) do
    Enum.any?(edges, fn {x, y} -> x == a and (y == b or (not MapSet.member?(seen, y) and above?(edges, y, b, MapSet.put(seen, y)))) end)
  end

  defp exclusive?(t, x, y), do: Enum.any?(t.exclusive, &(x in &1 and y in &1))

  defp conj(true, f), do: f
  defp conj(f, true), do: f
  defp conj(f, g), do: {:and, f, g}

  defp holds?(true, _), do: true
  defp holds?(f, sc), do: Formula.eval(f, sc)

  defp scenario(t, asg), do: Map.new(t.facts, &{&1, Map.get(asg, &1, false)})

  # for each (party, action) some clause governs: a scenario no clause covers
  defp silences(t, background) do
    t.clauses
    |> Enum.group_by(&{&1.party, &1.action})
    |> Enum.flat_map(fn {{party, action}, cs} ->
      if Enum.any?(cs, &(&1.cond == true)) do
        []
      else
        covered = cs |> Enum.map(& &1.cond) |> Enum.reduce(fn f, acc -> {:or, acc, f} end)

        case Formula.satisfy(conj(background, {:not, covered})) do
          {:sat, %{assignment: asg}} -> [%{party: party, action: action, scenario: scenario(t, asg), clauses: Enum.map(cs, & &1.id)}]
          _ -> []
        end
      end
    end)
  end

  # ============================================================ execution

  @doc """
  The normative positions in force under a concrete set of facts (a map
  `fact => boolean`; facts not given are false): every active clause, the
  Hohfeld correlative of each duty (the counterparty's claim), and the
  clashes among them. Refuses a scenario that violates an `assume`.
  """
  def positions(%__MODULE__{} = t, facts) do
    sc = Map.merge(Map.new(t.facts, &{&1, false}), Map.new(facts, fn {k, v} -> {to_string(k), v == true} end))
    unknown = Map.keys(sc) -- t.facts

    broken = Enum.find(t.assume, fn {_, f} -> unknown == [] and not Formula.eval(f, sc) end)

    cond do
      unknown != [] -> {:error, "unknown facts: #{Enum.join(unknown, ", ")}"}
      broken != nil -> {:error, "the facts violate the assumption #{elem(broken, 0)}"}
      true ->
        active = Enum.filter(t.clauses, &holds?(&1.cond, sc))
        claims = for c <- active, c.modality == :must, c.counterparty != nil, do: %{holder: c.counterparty, against: c.party, action: c.action, from: c.id}

        clashes =
          for {a, i} <- Enum.with_index(active), {b, j} <- Enum.with_index(active), i < j, a.party == b.party,
              {kind, why} = clash(a, b, t) || {nil, nil}, kind != nil do
            case prevails(t, a.id, b.id) do
              nil -> %{kind: kind, why: why, clauses: [a.id, b.id]}
              w -> %{kind: :resolved, why: why, clauses: [a.id, b.id], prevails: w}
            end
          end

        # a clause overridden by an active clause it clashes with is not in force
        silenced = for %{kind: :resolved, clauses: cs, prevails: w} <- clashes, c <- cs, c != w, do: c

        {:ok, %{active: active |> Enum.reject(&(&1.id in silenced)) |> Enum.map(&%{id: &1.id, party: &1.party, modality: &1.modality, action: &1.action, counterparty: &1.counterparty}),
                overridden: Enum.uniq(silenced), claims: Enum.reject(claims, &(&1.from in silenced)), clashes: Enum.reject(clashes, &(&1.kind == :resolved))}}
    end
  end

  @doc "A modality as words."
  def modality_text(:must), do: "must"
  def modality_text(:must_not), do: "must not"
  def modality_text(:may), do: "may"
  def modality_text(:exempt), do: "is exempt from"
end
