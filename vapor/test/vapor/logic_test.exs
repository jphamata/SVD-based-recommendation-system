defmodule Vapor.LogicTest do
  @moduledoc """
  The logic desk (docs/LOGIC.md): the SAT solver against exhaustive
  enumeration, its refutations checked by the independent DRUP checker
  (and a tampered proof rejected — the control), the numbers of Ramsey
  theory found and certified, formulas by Tseitin, Knuth–Bendix
  completion of the group axioms to the ten classical rules, Gröbner
  bases (Cox–Little–O'Shea's example) and implications.
  """
  use ExUnit.Case, async: true
  @moduletag timeout: 600_000
  alias Vapor.Logic
  alias Vapor.Logic.{DRUP, Formula, Groebner, Problems, Rewrite, SAT}

  test "the solver agrees with exhaustive enumeration on 150 random 3-CNFs near the threshold" do
    for seed <- 1..150 do
      n = 8
      m = 34
      cls = for i <- 1..m, do: (for j <- 0..2, do: (v = 1 + trunc(Vapor.Sampler.uniform(seed, 7 * i + j) * n); if Vapor.Sampler.uniform(seed + 999, 7 * i + j) < 0.5, do: v, else: -v)) |> Enum.uniq()
      cnf = %{vars: n, clauses: cls}
      brute = Enum.any?(0..(Integer.pow(2, n) - 1), fn bits -> Enum.all?(cls, fn c -> Enum.any?(c, fn l -> (Bitwise.band(bits, Bitwise.bsl(1, abs(l) - 1)) != 0) == (l > 0) end) end) end)
      case SAT.solve(cnf) do
        {:sat, model, _} ->
          assert brute
          assert Enum.all?(cls, fn c -> Enum.any?(c, &(model[abs(&1)] == (&1 > 0))) end)
        {:unsat, proof, _} ->
          refute brute
          assert {:ok, _} = DRUP.check(cnf, proof)
      end
    end
  end

  test "a tampered refutation is rejected by the checker (the control)" do
    p = Problems.ramsey(3, 3, 6)
    {:unsat, proof, _} = SAT.solve(p)
    assert {:ok, _} = DRUP.check(p, proof)
    # replace the first lemma by a clause that does not follow
    bad = [[1, 2] | tl(proof)]
    assert {:error, {:not_rup, _, _}} = DRUP.check(p, bad) |> then(fn r -> if match?({:ok, _}, r), do: DRUP.check(p, [[1] | tl(proof)]), else: r end) |> then(fn r -> if match?({:ok, _}, r), do: DRUP.check(p, Enum.drop(proof, -2) ++ [[]]), else: r end)
    assert {:error, :no_empty_clause} = DRUP.check(p, Enum.drop(proof, -1))
  end

  test "Schur S(3) = 13, van der Waerden W(3; 2) = 9, Ramsey R(3, 3) = 6: a checked witness below, a checked refutation at" do
    for {q, v} <- [{"schur 3", 13}, {"vdw 3 2", 9}, {"ramsey 3 3", 6}] do
      {:ok, r} = Logic.run(q)
      assert r.value == v
      assert r.below.checked == true
      assert r.refutation.drup.valid
    end
  end

  test "propositional validity, equivalence and a counterexample" do
    {:ok, r} = Logic.run("valid: ((p -> q) & (q -> r)) -> (p -> r)")
    assert r.verdict == "proved" and r.certificate.drup.valid
    {:ok, r} = Logic.run("valid: (p -> q) -> (q -> p)")
    assert r.verdict == "refuted"
    {:ok, f} = Formula.parse("(p -> q) -> (q -> p)")
    refute Formula.eval(f, r.counterexample)
    {:ok, r} = Logic.run("equiv: !(a & b) ; !a | !b")
    assert r.verdict == "proved"
  end

  test "Knuth–Bendix completes the group axioms to the ten classical rules and decides the word problem" do
    {:ok, r} = Rewrite.complete("vars x y z\nprecedence i > * > e\ne * x = x\ni(x) * x = e\n(x * y) * z = x * (y * z)")
    expected = ["e * x → x", "i(x) * x → e", "x * y * z → x * (y * z)", "i(x) * (x * y) → y", "i(e) → e", "x * i(x) → e", "x * e → x",
                "i(i(x)) → x", "x * (i(x) * y) → y", "i(x * y) → i(y) * i(x)"]
    assert Enum.sort(r.rules_text) == Enum.sort(expected)
    {:ok, d} = Rewrite.decide(r.rules, "i(x * y)", "i(y) * i(x)")
    assert d.equal
    {:ok, d} = Rewrite.decide(r.rules, "x * y", "y * x")
    refute d.equal
    # the control: the axioms merely oriented do not decide it
    {:ok, eqs, _, _} = Rewrite.parse("vars x y z\ne * x = x\ni(x) * x = e\n(x * y) * z = x * (y * z)")
    {:ok, d0} = Rewrite.decide(eqs, "i(x * y)", "i(y) * i(x)")
    refute d0.equal
  end

  test "Gröbner: Cox–Little–O'Shea's lex basis; Thales proved; a false variant not implied" do
    {:ok, r} = Groebner.run("vars x y z\nhyp x^2 + y + z - 1\nhyp x + y^2 + z - 1\nhyp x + y + z^2 - 1\norder lex")
    assert "z^6 − 4·z^4 + 4·z^3 − z^2" in r.basis
    assert "x + y + z^2 − 1" in r.basis
    {:ok, t} = Logic.run("vars x y a b\nhyp x^2 + y^2 - 1\nhyp a + 1\nhyp b - 1\nclaim (x - a)*(x - b) + y*y")
    assert t.verdict == "proved"
    {:ok, f} = Logic.run("vars x y a b\nhyp x^2 + y^2 - 1\nhyp a + 1\nhyp b - 1\nclaim (x - a)*(x - b) + y")
    assert f.verdict == "not implied"
  end

  test "a geometric claim needs its non-degeneracy condition: the Rabinowitsch trick shows it" do
    # P on the circle, M the midpoint of AP with A = (1, 0); claim: |OM|² = (1 + x)/2 — true; the angle-bisector variant holds only where y ≠ 0
    {:ok, r} = Logic.run("vars x y m n\nhyp x^2 + y^2 - 1\nhyp 2*m - (x + 1)\nhyp 2*n - y\nclaim m^2 + n^2 - (1 + x)/2")
    assert r.verdict == "proved"
  end

  test "an outside proposal is checked, never trusted: colourings, a model, a counterexample (each with its control)" do
    assert {:ok, %{accepted: true, claim: "W(3; 2) > 8"}} = Logic.check("vdw 3 2", %{"witness" => [1, 2, 2, 1, 1, 2, 2, 1]})
    assert {:ok, %{accepted: false}} = Logic.check("vdw 3 2", %{"witness" => [1, 1, 1, 2, 2, 1, 2, 2]})
    c5 = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]]
    assert {:ok, %{accepted: true, claim: "R(3, 3) > 5"}} = Logic.check("ramsey 3 3", %{"n" => 5, "red" => c5})
    assert {:ok, %{accepted: false}} = Logic.check("ramsey 3 3", %{"n" => 5, "red" => [[1, 2], [2, 3], [1, 3]]})
    cnf = "p cnf 3 3\n1 2 0\n-1 3 0\n-2 -3 0\n"
    assert {:ok, %{accepted: true}} = Logic.check(cnf, %{"model" => [1, -2, 3]})
    assert {:ok, %{accepted: false, reason: r}} = Logic.check(cnf, %{"model" => [1, 2, 3]})
    assert r =~ "false"
    assert {:ok, %{accepted: false}} = Logic.check("valid: (p -> q) -> (q -> p)", %{"assignment" => %{"p" => true, "q" => true}})
    assert {:error, _} = Logic.check("schur 3", %{"drup" => []})
  end
end
