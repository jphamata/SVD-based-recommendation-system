defmodule Vapor.TabulaTest do
  use ExUnit.Case, async: true
  alias Vapor.Tabula
  alias Vapor.Logic.Formula

  @sale """
  parties buyer seller
  facts delivered late defective force_majeure
  exclusive pay withhold
  assume not (late and not delivered)
  C1: if delivered and not defective then buyer must pay seller
  C2: if late then buyer may withhold
  C3: if defective then buyer must not pay
  C4: if force_majeure then seller is exempt from deliver
  C5: seller must deliver buyer
  C6: if late and delivered then buyer must withhold
  """

  test "an antinomy is found with the scenario that triggers it; a pair that never clashes is proved so (DRUP)" do
    {:ok, t} = Tabula.parse(@sale)
    r = Tabula.analyze(t)
    assert r.verdict == :antinomies
    anti = Enum.find(r.findings, &(&1.clauses == ["C1", "C6"]))
    assert anti.kind == :antinomy and anti.scenario["late"] and anti.scenario["delivered"] and not anti.scenario["defective"]
    # the scenario is real: both conditions hold there, and the background assumption too
    for id <- ["C1", "C6"], do: assert(Formula.eval(Enum.find(t.clauses, &(&1.id == id)).cond, anti.scenario))
    assert [%{clauses: ["C1", "C3"], drup: true}] = r.checked_pairs
  end

  test "an override (lex specialis) turns a clash into a resolution; the positions in force follow it" do
    {:ok, t} = Tabula.parse(@sale <> "C4 overrides C5\nC6 overrides C1\n")
    r = Tabula.analyze(t)
    assert r.verdict == :consistent
    assert Enum.map(r.resolved, &{&1.clauses, &1.prevails}) |> Enum.sort() == [{["C1", "C6"], "C6"}, {["C4", "C5"], "C4"}]

    {:ok, pos} = Tabula.positions(t, %{"force_majeure" => true, "delivered" => true, "late" => true})
    ids = Enum.map(pos.active, & &1.id)
    assert "C4" in ids and "C6" in ids
    refute "C5" in ids or "C1" in ids
    assert pos.clashes == []
    assert Enum.sort(pos.overridden) == ["C1", "C5"]
  end

  test "silences: a scenario in which nothing is said about an act other clauses govern" do
    {:ok, t} = Tabula.parse(@sale)
    %{silences: s} = Tabula.analyze(t)
    pay = Enum.find(s, &(&1.action == "pay"))
    refute pay.scenario["delivered"] or pay.scenario["defective"]
  end

  test "Hohfeld: every duty owed to someone is that someone's claim" do
    {:ok, t} = Tabula.parse(@sale)
    {:ok, pos} = Tabula.positions(t, %{"delivered" => true})
    assert %{holder: "seller", against: "buyer", action: "pay", from: "C1"} in pos.claims
    assert %{holder: "buyer", against: "seller", action: "deliver", from: "C5"} in pos.claims
  end

  test "in Portuguese, with the same decisions" do
    texto = """
    partes locador locatario
    fatos atraso reforma
    L1: se atraso então locatario deve pagar_multa locador
    L2: se reforma então locatario está isento de pagar_multa
    L3: locador pode cobrar
    L2 prevalece sobre L1
    """

    {:ok, t} = Tabula.parse(texto)
    r = Tabula.analyze(t)
    assert r.verdict == :consistent
    assert [%{clauses: ["L1", "L2"], prevails: "L2"}] = r.resolved
  end

  test "the background assumptions rule scenarios out: a clash that needs an impossible combination is proved absent" do
    text = """
    facts a b
    assume not (a and b)
    X1: if a then p must go
    X2: if b then p must not go
    """

    {:ok, t} = Tabula.parse(text)
    assert %{verdict: :consistent, checked_pairs: [%{drup: true}]} = Tabula.analyze(t)
    {:ok, t2} = Tabula.parse(String.replace(text, "assume not (a and b)\n", ""))
    assert %{verdict: :antinomies, findings: [%{scenario: %{"a" => true, "b" => true}}]} = Tabula.analyze(t2)
  end

  test "refusals: undeclared facts and parties, duplicate ids, cycles of overrides, no modality, bad names" do
    assert {:error, m1} = Tabula.parse("facts a\nX: if b then p must go\n")
    assert m1 =~ "not declared"
    assert {:error, m2} = Tabula.parse("parties p\nX: q must go\n")
    assert m2 =~ "not a declared party"
    assert {:error, _} = Tabula.parse("X: p must go\nX: p may go\n")
    assert {:error, m3} = Tabula.parse("X: p must go\nY: p may stay\nX overrides Y\nY overrides X\n")
    assert m3 =~ "cycle"
    assert {:error, _} = Tabula.parse("X: p wants go\n")
    assert {:error, _} = Tabula.parse("facts a$b\nX: p must go\n")
    assert {:error, _} = Tabula.parse("X: if a p must go\n")
    {:ok, t} = Tabula.parse("facts a\nassume a\nX: p must go\n")
    assert {:error, msg} = Tabula.positions(t, %{"a" => false})
    assert msg =~ "assumption"
    assert {:error, _} = Tabula.positions(t, %{"zz" => true})
  end

  @tag timeout: 120_000
  test "stress: 120 clauses over 30 facts — every finding re-evaluated, every proof checked" do
    :rand.seed(:exsss, {2, 7, 1})
    facts = for i <- 1..30, do: "f#{i}"
    mods = ["must", "must not", "may", "is exempt from"]

    clauses =
      for i <- 1..120 do
        lits = for _ <- 1..Enum.random(1..3), do: (if :rand.uniform(2) == 1, do: "", else: "not ") <> Enum.random(facts)
        "K#{i}: if #{Enum.join(lits, " and ")} then p#{Enum.random(1..3)} #{Enum.random(mods)} act#{Enum.random(1..4)}"
      end

    {:ok, t} = Tabula.parse("facts #{Enum.join(facts, " ")}\n" <> Enum.join(clauses, "\n"))
    r = Tabula.analyze(t)
    assert r.findings != []

    for f <- r.findings, id <- f.clauses do
      assert Formula.eval(Enum.find(t.clauses, &(&1.id == id)).cond, f.scenario)
    end

    assert Enum.all?(r.checked_pairs, & &1.drup)
  end
end
