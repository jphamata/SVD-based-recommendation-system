defmodule Vapor.Crucible.Fold do
  @moduledoc """
  Folding the user's own HP sequence on the square lattice (docs/CRUCIBLE.md
  §9; the HP model of Dill 1985), with an **optimality gap** instead of a
  bare number:

    * the search: Monte Carlo with pivot, corner and end moves and
      annealing (`Vapor.Science.Biology.fold/2`), seeded;
    * an **upper bound on contacts** that holds for every fold: the square
      lattice is bipartite, so H–H contacts join residues of opposite
      parity, and a residue has at most 2 free neighbours (3 at the chain's
      ends) — the bound is min over the two parity classes of their summed
      capacities. E ≥ −bound;
    * for short chains (≤ 14), **exhaustive enumeration**: the true optimum.

  The verdict says which of the three it is: *optimal (proved by the
  bound)*, *optimal (proved by enumeration)*, or *best found, within k of
  the bound*.
  """
  alias Vapor.Science.Biology

  def run(text) do
    seq = case Regex.run(~r/\b([HPhp]{4,200})\b/, text) do [_, s] -> String.upcase(s); nil -> nil end
    steps = case Regex.run(~r/steps\s*=\s*(\d+)/, text) do [_, n] -> min(String.to_integer(n), 2_000_000); nil -> 200_000 end

    cond do
      seq == nil -> {:error, "write the sequence of H (hydrophobic) and P (polar) residues, e.g. HPHPPHHPHPPHPHHPPHPH"}
      true ->
        found = Biology.fold(seq, steps: steps)
        bound = bound(seq)
        exact = if String.length(seq) <= 14, do: Biology.ground_state(seq)
        random = Biology.random_energy(seq)
        status =
          cond do
            exact && found.energy == exact.energy -> "optimal (equal to exhaustive enumeration)"
            found.energy == -bound -> "optimal (proved: it reaches the contact bound)"
            exact -> "not optimal: enumeration finds #{exact.energy}"
            true -> "best found; the bound allows at most #{bound - -found.energy} more contact(s)"
          end

        {:ok, %{kind: "fold", sequence: seq, energy: found.energy, coords: Enum.map(found.coords, &Tuple.to_list/1), bound: -bound, exact: exact && exact.energy,
                random_mean: random, status: status,
                evidence: [
                  %{check: "gap", ok: exact == nil or found.energy == exact.energy, detail: "found #{found.energy}; the parity bound is #{-bound}#{if exact, do: "; enumeration: #{exact.energy}", else: ""}"},
                  %{check: "control", ok: found.energy < random, detail: "random self-avoiding conformations average #{Float.round(random, 2)}"}
                ],
                says: "E = #{found.energy} — #{status}"}}
    end
  end

  @doc "Upper bound on H–H topological contacts on the square lattice."
  def bound(seq) do
    n = String.length(seq)
    hs = seq |> String.graphemes() |> Enum.with_index() |> Enum.filter(fn {c, _} -> c == "H" end) |> Enum.map(&elem(&1, 1))
    cap = fn i -> if i == 0 or i == n - 1, do: 3, else: 2 end
    even = hs |> Enum.filter(&(rem(&1, 2) == 0)) |> Enum.map(cap) |> Enum.sum()
    odd = hs |> Enum.filter(&(rem(&1, 2) == 1)) |> Enum.map(cap) |> Enum.sum()
    min(even, odd)
  end
end
