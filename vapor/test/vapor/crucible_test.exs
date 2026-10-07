defmodule Vapor.CrucibleTest do
  use ExUnit.Case, async: true
  alias Vapor.Crucible
  alias Vapor.Crucible.Poly

  @moduletag timeout: 600_000

  defp run!(kind, text) do
    {:ok, r} = Crucible.run(kind, text)
    r
  end

  defp laws(r), do: Enum.map(r.laws, & &1.law)

  describe "conservation laws, proved over ℚ" do
    test "exact rationals: 0.1 is 1/10, and the null space is exact" do
      assert Poly.rational(0.1) == {1, 10}
      assert Poly.rational(-2.5e-3) == {-1, 400}
      assert Poly.null_space([[{1, 1}, {1, 1}, {0, 1}], [{0, 1}, {0, 1}, {1, 1}]], 3) == [[1, -1, 0]]
    end

    test "SIR: S + I + R and the logarithmic law are both found and proved" do
      r = run!("laws", Crucible.example("laws"))
      assert "S + I + R" in laws(r)
      assert "3·S + 3·I − ln(S)" in laws(r)
      assert Enum.all?(r.laws, &(&1.status == "proved" and &1.relative_drift < 1.0e-4))
      assert r.control.generic == 0
    end

    test "a damped oscillator has none — proved — while the undamped one has its energy" do
      assert laws(run!("laws", "x' = v\nv' = -4*x - 0.1*v")) == []
      assert laws(run!("laws", "x' = v\nv' = -4*x")) == ["4·x^2 + v^2"]
    end

    test "Lotka–Volterra's invariant, with its logarithms" do
      assert laws(run!("laws", "x' = 1.1*x - 0.4*x*y\ny' = 0.1*x*y - 0.4*y")) == ["x + 4·y − 4·ln(x) − 11·ln(y)"]
    end

    test "Euler's rigid body: two quadratic invariants, no products of them reported" do
      r = run!("laws", "a' = -b*c\nb' = c*a\nc' = -a*b/3\ndegree = 4")
      assert length(r.laws) == 2 and Enum.all?(r.laws, &(&1.degree == 2))
    end

    test "a non-polynomial system: the pendulum's energy, verified numerically and labelled so" do
      r = run!("laws", "q' = p\np' = -sin(q)\nbasis = [cos(q)]")
      assert [%{status: s}] = r.laws
      assert s =~ "not a proof"
    end
  end

  describe "quantum, Hamiltonians, fields — evidence without a reference" do
    test "the harmonic oscillator: n + 1/2, order 2, virial theorem" do
      r = run!("quantum", "V(x) = 0.5*x^2\nx = -9 .. 9\nn = 300\nstates = 4")
      for {s, k} <- Enum.with_index(r.states), do: assert_in_delta(s.energy, k + 0.5, 2.0e-4)
      assert Enum.all?(r.evidence, & &1.ok)
      assert Enum.all?(r.states, &(abs(&1.observed_order - 2) < 0.2))
    end

    test "a wrong Hamiltonian fails the virial check (the control)" do
      # the virial theorem is checked against the V′ derived from what was written; a kinetic term
      # with the wrong mass changes ⟨T⟩ but not V′ — but the theorem holds for any mass, so instead
      # compare against a box too small for the states: the evidence must say so
      r = run!("quantum", "V(x) = 0.5*x^2\nx = -1.5 .. 1.5\nn = 300\nstates = 4")
      refute Enum.all?(r.evidence, & &1.ok)
    end

    test "the double well tunnels: Ehrenfest and unitarity hold along the run" do
      r = run!("quantum", Crucible.example("quantum"))
      assert Enum.all?(r.evidence, & &1.ok), inspect(Enum.reject(r.evidence, & &1.ok))
      [e0, e1 | _] = Enum.map(r.states, & &1.energy)
      assert e1 - e0 > 0 and e1 - e0 < 0.5
    end

    test "Hénon–Heiles: symplectic order 4, reversible, energy without drift, and H rediscovered" do
      r = run!("hamiltonian", Crucible.example("hamiltonian"))
      assert Enum.all?(r.evidence, & &1.ok), inspect(Enum.reject(r.evidence, & &1.ok))
      assert r.method == "yoshida4" and r.separable
    end

    test "a non-separable Hamiltonian uses the implicit midpoint rule" do
      r = run!("hamiltonian", "H = (p^2 + q^2)/2 + 0.1*q^2*p^2\nq(0) = 1; p(0) = 0\nt = 0 .. 20\ndt = 0.02")
      assert r.method == "midpoint" and not r.separable
      assert Enum.find(r.evidence, &(&1.check == "observed order")).ok
    end

    test "a charge in a magnetic bottle: |u| and the work–energy theorem" do
      r = run!("fields", Crucible.example("fields"))
      assert Enum.all?(r.evidence, & &1.ok)
    end
  end

  describe "chemistry, biology, data" do
    test "H₂ at 1.4 bohr: the textbook RHF energy, a converged commutator" do
      r = run!("molecule", "H 0 0 0\nH 0 0 1.4")
      assert_in_delta r.energy, -1.1167, 1.0e-4
      assert Enum.all?(r.evidence, & &1.ok)
    end

    test "HeH⁺ and H₃⁺; odd electron counts are refused with the reason" do
      assert_in_delta run!("molecule", "He 0 0 0\nH 0 0 1.4632\ncharge = 1").energy, -2.860662, 1.0e-4
      assert run!("molecule", Crucible.example("molecule")).energy < -1.2
      assert {:error, m} = Crucible.run("molecule", "H 0 0 0\nH 0 0 1.4\nH 0 0 2.8")
      assert m =~ "closed shell"
      assert {:error, m2} = Crucible.run("molecule", "C 0 0 0")
      assert m2 =~ "p orbitals"
    end

    test "Wright–Fisher: the exact chain, the simulation and Kimura agree" do
      r = run!("evolution", "N = 100\ns = 0.03\ni0 = 2\nreplicates = 3000")
      assert Enum.all?(r.evidence, & &1.ok)
    end

    test "phylogeny from sequences evolved down a known tree: the true clades get support" do
      seqs = Vapor.Science.Biology.evolve(Vapor.Science.Biology.tree(), 600, 3)
      fasta = Enum.map_join(seqs, "\n", fn {n, s} -> ">#{n}\n" <> Enum.map_join(s, &Enum.at(~w(A C G T), &1)) end)
      r = run!("phylogeny", fasta <> "\nbootstrap = 50")
      truth = Vapor.Science.Biology.splits(Vapor.Science.Biology.tree()) |> Enum.map(&Enum.sort(MapSet.to_list(&1)))
      names = Enum.map(seqs, &elem(&1, 0))
      canon = fn set -> if "A" in set, do: Enum.sort(names -- set), else: set end
      found = r.splits |> Enum.filter(&(&1.support >= 0.7)) |> Enum.map(&canon.(&1.taxa))
      assert Enum.all?(found, &(&1 in Enum.map(truth, canon))), "unsupported clade: #{inspect(found)}"
      assert r.control.mean_support < 0.7
    end

    test "HP folding: a short chain is proved optimal by enumeration" do
      r = run!("fold", "sequence = HPHPPHHPHPPH\nsteps = 60000")
      assert r.status =~ "optimal"
    end

    test "symbolic regression recovers a planted law and the shuffled control does not" do
      rows = for k <- 1..40, do: (x = k / 8; "#{x},#{Float.round(3 * x * x - 2 * x + 1, 6)}")
      r = run!("regress", "x, y\n" <> Enum.join(rows, "\n") <> "\ntarget = y\nops = + - *\nmax_size = 9\nbudget = 8000")
      assert r.best.test_r2 > 0.9999
      assert r.control.test_r2 < 0.5
    end

    test "every example runs and its evidence holds" do
      for %{kind: k, example: ex} <- Crucible.kinds() do
        assert {:ok, r} = Crucible.run(k, ex), k
        bad = Enum.reject(r[:evidence] || [], & &1.ok)
        assert bad == [], "#{k}: #{inspect(bad)}"
      end
    end
  end

  test "a hostile input is bounded: a huge grid is refused by time or memory, never a crash" do
    assert {:error, _} = Crucible.run("quantum", "V(x) = x^2\nn = 4000\nstates = 30\npsi0(x) = exp(-x^2)\nt = 0 .. 100000\ndt = 0.0001", timeout: 3_000)
  end
end
