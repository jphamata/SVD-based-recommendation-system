defmodule Vapor.Crucible do
  @moduledoc """
  **Crucible** — open science (docs/CRUCIBLE.md). Where the 0.11 science
  laboratory ran eleven experiments chosen in advance, the Crucible takes
  the user's own system in each domain and answers with **evidence that
  does not need a reference solution**: observed orders of convergence,
  conservation laws proved over ℚ, theorems that must hold for any input
  (virial, Ehrenfest, work–energy), two independent methods that must
  agree, and controls that a wrong method would fail.

  The examples are starting points — every one is text to be edited.
  `run(kind, text)` runs one; `kinds/0` lists them with an example each.
  Every run is bounded (a sandboxed process with a heap cap and a time
  limit) because the input is open.
  """
  alias Vapor.Crucible.{Evolution, Fields, Fold, Hamiltonian, Laws, Molecule, Phylo, Quantum, Reactions, Regress}

  @kinds [
    {"quantum", Quantum, "bound states and dynamics of any 1-D potential",
     """
     # a double well: tunnelling splits the lowest pair of levels
     V(x) = 0.25*(x^2 - 4)^2
     x = -6 .. 6
     n = 400
     states = 4
     psi0(x) = exp(-(x - 2)^2)       # start in the right well
     t = 0 .. 20
     """},
    {"hamiltonian", Hamiltonian, "any Hamiltonian, integrated symplectically, with its invariants",
     """
     # Hénon–Heiles: a classic system that is chaotic at higher energies
     H = (p1^2 + p2^2)/2 + (q1^2 + q2^2)/2 + q1^2*q2 - q2^3/3
     q1(0) = 0.1; q2(0) = 0.0; p1(0) = 0.3; p2(0) = 0.25
     t = 0 .. 200
     dt = 0.02
     """},
    {"laws", Laws, "conservation laws of any ODE system — proved over ℚ when polynomial",
     """
     # an SIR epidemic: which quantities never change?
     S' = -0.3*S*I
     I' = 0.3*S*I - 0.1*I
     R' = 0.1*I
     S(0) = 0.99; I(0) = 0.01; R(0) = 0
     t = 0 .. 100
     degree = 2
     """},
    {"reactions", Reactions, "a reaction network, deterministic and stochastic",
     """
     A + B <-> C ; kf = 2, kb = 0.5
     C -> D ; k = 0.3
     A0 = 1; B0 = 0.8
     t = 0 .. 20
     volume = 300
     runs = 40
     """},
    {"evolution", Evolution, "the fate of a mutant: exact chain, simulation, diffusion",
     """
     N = 200
     s = 0.02
     i0 = 1
     replicates = 4000
     """},
    {"phylogeny", Phylo, "a tree from your aligned sequences, with bootstrap and a control",
     """
     >human
     ACGTACGTTAGCATCGATCGATGCTAGCTAGGCTAACGATCGTAGCTAGCTAGCATGCATCGA
     >chimp
     ACGTACGTTAGCATCGATCGATGCTAGCTAGGCTAACGATCGTAGCTAGCTAGCATGCATCGG
     >gorilla
     ACGTACGTTAGCATCGTTCGATGCTAGCTAGGCTAACGATCGTAGCAAGCTAGCATGCATCGG
     >orangutan
     ACGTACCTTAGCATCGTTCGATGCTTGCTAGGCTAACGATCGTAGCAAGCTAGCTTGCATCGG
     >gibbon
     ACGAACCTTAGCAACGTTCGATGCTTGCTAGGATAACGATCGTAGCAAGCTTGCTTGCATAGG
     bootstrap = 100
     """},
    {"molecule", Molecule, "Hartree–Fock for any arrangement of H and He",
     """
     # the trihydrogen cation, an equilateral triangle
     H 0 0 0
     H 1.65 0 0
     H 0.825 1.42894 0
     charge = 1
     units = bohr
     """},
    {"fold", Fold, "your HP sequence folded, with an optimality gap",
     """
     sequence = HPHPPHHPHPPHPHHPPHPH
     steps = 200000
     """},
    {"fields", Fields, "a charged particle in any E and B, relativistic",
     """
     # a magnetic bottle: B grows away from the middle, the particle bounces
     Bz = 1 + 0.05*z^2
     Bx = -0.05*x*z
     By = -0.05*y*z
     x(0) = 0.5; y(0) = 0; z(0) = 0
     ux(0) = 0; uy(0) = 0.4; uz(0) = 0.3
     t = 0 .. 200
     dt = 0.02
     """},
    {"regress", Regress, "a law from data: symbolic regression with held-out rows",
     """
     # drag force against speed (made with F = 0.3 v^2 + 0.05 v, plus noise)
     v, F
     0.5, 0.101
     1.0, 0.348
     1.5, 0.752
     2.0, 1.302
     2.5, 2.003
     3.0, 2.851
     3.5, 3.853
     4.0, 5.003
     4.5, 6.302
     5.0, 7.748
     5.5, 9.351
     6.0, 11.102
     6.5, 13.003
     7.0, 15.051
     target = F
     ops = + - * /
     max_size = 11
     budget = 15000
     """}
  ]

  @doc "The domains: `[%{kind, about, example}]`."
  def kinds, do: Enum.map(@kinds, fn {k, _, about, ex} -> %{kind: k, about: about, example: ex} end)

  @doc "The example text of a domain."
  def example(kind), do: Enum.find_value(@kinds, fn {k, _, _, ex} -> if k == kind, do: ex end)

  @doc "Run a domain on the user's text, sandboxed: `{:ok, result}` or `{:error, why}`."
  def run(kind, text, opts \\ []) do
    case List.keyfind(@kinds, kind, 0) do
      nil -> {:error, "unknown domain #{kind}: #{Enum.map_join(@kinds, ", ", &elem(&1, 0))}"}
      {_, mod, _, _} ->
        if byte_size(text) > 2_000_000 do
          {:error, "input larger than 2 MB"}
        else
          t0 = System.monotonic_time(:millisecond)
          case Vapor.Alembic.sandbox(fn -> safe(fn -> mod.run(text) end) end, heap_mb: Keyword.get(opts, :heap_mb, 1024), timeout: Keyword.get(opts, :timeout, 240_000)) do
            {:ok, {:ok, r}} -> {:ok, Map.put(r, :ms, System.monotonic_time(:millisecond) - t0)}
            {:ok, {:error, e}} -> {:error, to_string(e)}
            {:error, :memory} -> {:error, "the computation needed more memory than allowed — make the problem smaller"}
            {:error, :timeout} -> {:error, "the computation took longer than its limit — make the problem smaller"}
            {:error, {:crash, why}} -> {:error, "failed: #{inspect(why) |> String.slice(0, 200)}"}
          end
        end
    end
  end

  defp safe(f) do
    f.()
  rescue
    e -> {:error, Exception.message(e)}
  catch
    {:alembic, m} -> {:error, m}
    kind, why -> {:error, "#{kind}: #{inspect(why) |> String.slice(0, 200)}"}
  end
end
