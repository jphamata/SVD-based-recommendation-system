defmodule Vapor.AludelTest do
  use ExUnit.Case, async: true
  alias Vapor.Aludel, as: A
  alias Vapor.Logic.LP

  defp p!(text, vars), do: (fn {:ok, p} -> p end).(A.parse(text, vars))
  defp box!(pairs), do: (fn {:ok, b} -> b end).(A.box(pairs))
  defp neg?(q), do: LP.qsign(q) < 0

  describe "the text form" do
    test "products, powers, fractions, decimals, implicit multiplication and unary minus" do
      p = p!("2x(y - 1/2)^2 - 0.25*x + -(x)", ["x", "y"])
      # at (1, 1): 2·1·(1/4) − 1/4 − 1 = −3/4
      assert A.eval(p, [{1, 1}, {1, 1}]) == {-3, 4}
      assert A.degrees(p) == [1, 2]
    end

    test "unknown names, bad exponents, division by a variable and stray characters are refused" do
      assert {:error, m} = A.parse("x + z", ["x"])
      assert m =~ "unknown variable z"
      assert {:error, _} = A.parse("x^-1", ["x"])
      assert {:error, _} = A.parse("x^1.5", ["x"])
      assert {:error, _} = A.parse("1/x", ["x"])
      assert {:error, _} = A.parse("x $ 1", ["x"])
      assert {:error, _} = A.parse("(x + 1", ["x"])
      assert {:error, m2} = A.parse("x^30", ["x"])
      assert m2 =~ "24"
    end
  end

  describe "the decision" do
    test "a perfect square touching zero at a midpoint is certified; the witness replays" do
      p = p!("x^2 - x + 1/4", ["x"])
      b = box!([{0, 1}])
      assert {:certified, %{witness: w}} = A.decide(p, b)
      assert :ok = A.check(p, b, w)
    end

    test "a refutation is an exact point with its exact value" do
      p = p!("x^2*y - 3/2 + y^2", ["x", "y"])
      b = box!([{-1, 1}, {-1, 1}])
      assert {:refuted, %{point: pt, value: v}} = A.decide(p, b)
      assert A.eval(p, pt) == v and neg?(v)
    end

    test "strict and non-strict are different claims" do
      p = p!("x^2", ["x"])
      b = box!([{0, 1}])
      assert {:certified, _} = A.decide(p, b)
      assert {:refuted, %{value: {0, 1}}} = A.decide(p, b, sense: :pos)
    end

    test "Motzkin's polynomial (non-negative, not a sum of squares): ≥ 0 is 'exhausted' — said, not faked — and > −1/1000 is certified" do
      m = p!("x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1", ["x", "y"])
      b = box!([{-2, 2}, {-2, 2}])
      assert {:exhausted, %{cell: cell}} = A.decide(m, b, depth: 16)
      assert length(cell) == 2
      eps = A.add(m, A.const(2, "1/1000"))
      assert {:certified, %{witness: w}} = A.decide(eps, b, sense: :pos)
      assert :ok = A.check(eps, b, w, sense: :pos)
    end

    test "a witness does not transfer: tampered bits, another polynomial, truncation — all refused by check" do
      p = p!("x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1001/1000", ["x", "y"])
      b = box!([{-2, 2}, {-2, 2}])
      {:certified, %{witness: w}} = A.decide(p, b, sense: :pos)
      assert {:error, _} = A.check(p, b, %{w | bits: 1, tree: "00"}, sense: :pos)
      assert {:error, _} = A.check(A.sub(p, A.const(2, "2/1000")), b, w, sense: :pos)
      assert {:error, _} = A.check(p, b, %{w | bits: w.bits - 1}, sense: :pos)
      assert {:error, _} = A.check(p, b, %{w | tree: "zz"}, sense: :pos)
    end

    test "soundness on 80 random polynomials: every 'certified' holds at 60 random exact points, every 'refuted' point is negative" do
      :rand.seed(:exsss, {3, 1, 4})

      for t <- 1..80 do
        n = Enum.random(1..3)
        vars = Enum.take(["x", "y", "z"], n)
        terms = for _ <- 1..Enum.random(2..6), do: "#{Enum.random(-9..9)}/#{Enum.random(1..4)}*" <> Enum.map_join(vars, "*", &"#{&1}^#{Enum.random(0..3)}")
        p = p!(Enum.join(terms, " + ") <> " + #{Enum.random(0..12)}", vars)
        b = box!(for _ <- vars, do: (lo = Enum.random(-3..1); {lo, lo + Enum.random(1..3)}))

        case A.decide(p, b, depth: 14, cells: 4_000) do
          {:certified, %{witness: w}} ->
            assert :ok = A.check(p, b, w)

            for _ <- 1..60 do
              pt = for {lo, hi} <- b, do: LP.qadd(lo, LP.qmul(LP.qsub(hi, lo), LP.q(:rand.uniform(1001) - 1, 1000)))
              refute neg?(A.eval(p, pt)), "case #{t}"
            end

          {:refuted, %{point: pt, value: v}} ->
            assert A.eval(p, pt) == v and neg?(v)

          {:exhausted, _} ->
            :ok
        end
      end
    end

    test "three variables, and the enclosure contains the exact values" do
      p = p!("x*y*z - x^2 + y*z^2 + 1", ["x", "y", "z"])
      b = box!([{-1, 1}, {0, 2}, {"-1/2", "1/2"}])
      {lo, hi} = A.enclose(p, b, 2)
      :rand.seed(:exsss, {5, 5, 5})

      for _ <- 1..100 do
        pt = for {l, h} <- b, do: LP.qadd(l, LP.qmul(LP.qsub(h, l), LP.q(:rand.uniform(97), 97)))
        v = A.eval(p, pt)
        assert LP.qcmp(lo, v) <= 0 and LP.qcmp(v, hi) <= 0
      end
    end
  end

  describe "barrier certificates" do
    setup do
      vars = ["x", "y"]
      pp = &p!(&1, vars)
      stable = %{vars: vars, field: [pp.("y"), pp.("-x - y")], domain: [{-2, 2}, {-2, 2}], init: [{"-1/2", "1/2"}, {"-1/2", "1/2"}], unsafe: [{"3/2", 2}, {"3/2", 2}]}
      {:ok, pp: pp, stable: stable}
    end

    test "a damped oscillator: x² + y² − 1 separates the initial from the unsafe set — three certified conditions", %{pp: pp, stable: sys} do
      r = A.barrier(sys, pp.("x^2 + y^2 - 1"))
      assert r.verdict == :proved
      assert Enum.map(r.conditions, & &1.name) == ["initial", "unsafe", "flow"]
    end

    test "a wrong candidate and an unstable system are refuted at exact points", %{pp: pp, stable: sys} do
      assert %{verdict: :refuted, conditions: cs} = A.barrier(sys, pp.("x^2 + y^2 - 5"))
      assert {:refuted, _} = Enum.find(cs, &(&1.name == "unsafe")).result
      unstable = %{sys | field: [pp.("x"), pp.("y")]}
      assert %{verdict: :refuted, conditions: cs2} = A.barrier(unstable, pp.("x^2 + y^2 - 1"))
      assert {:refuted, _} = Enum.find(cs2, &(&1.name == "flow")).result
    end

    test "synthesis: an exact LP proposes, the decision accepts; a nonlinear field too", %{pp: pp, stable: sys} do
      assert {:ok, b, rep} = A.synthesize(sys, degree: 2)
      assert rep.verdict == :proved and rep.lp_rows < rep.lp_rows_total
      assert A.barrier(sys, b).verdict == :proved

      # ẋ = −x + y², ẏ = −y (stable at the origin)
      nl = %{sys | field: [pp.("-x + y^2"), pp.("-y")], domain: [{-1, 1}, {-1, 1}], init: [{"-1/4", "1/4"}, {"-1/4", "1/4"}], unsafe: [{"3/4", 1}, {"3/4", 1}]}
      assert {:ok, _b, %{verdict: :proved}} = A.synthesize(nl, degree: 2, split: 1)
    end

    test "when the unsafe set meets the initial set no barrier exists, and the LP says so with a certificate", %{stable: sys} do
      bad = %{sys | unsafe: [{"-1/4", "1/4"}, {"-1/4", "1/4"}]}
      assert {:error, msg} = A.synthesize(bad, degree: 2)
      assert msg =~ "infeasible"
    end
  end
end
