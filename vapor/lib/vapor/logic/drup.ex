defmodule Vapor.Logic.DRUP do
  @moduledoc """
  An independent checker of DRUP proofs (docs/LOGICA.md §1): a refutation
  of a CNF is a sequence of clauses, each of which must follow from the
  formula and the clauses before it by **reverse unit propagation** —
  assume the clause false, propagate units, reach a conflict — ending in
  the empty clause. The SAT competitions accept UNSAT answers only with
  such proofs (DRAT-trim checks them); this is the same rule, written
  without any code of `Vapor.Logic.SAT`: occurrence lists instead of
  watched literals, maps instead of atomics.

  Checking runs **backwards from the empty clause**, as DRAT-trim does:
  the conflict that refutes ¬∅ marks the clauses it used, and only marked
  lemmas are checked in turn (each against the formula and the lemmas
  before it). The lemmas never marked are not needed by the proof and
  are reported as such — `core` counts the ones that are.
  """

  @doc """
  Check a proof: `{:ok, %{lemmas, checked, core_lemmas, core_clauses}}`,
  `{:error, {:not_rup, index, lemma}}`, or `{:error, :no_empty_clause}`.
  """
  def check(%{clauses: clauses}, proof) do
    if List.last(proof) != [] do
      {:error, :no_empty_clause}
    else
      m = length(clauses)
      all = clauses ++ proof
      db = all |> Enum.with_index() |> Map.new(fn {c, i} -> {i, c} end)
      occ = all |> Enum.with_index() |> Enum.reduce(%{}, fn {c, i}, occ -> Enum.reduce(c, occ, fn l, o -> Map.update(o, l, [i], &[i | &1]) end) end)
      occ = Map.new(occ, fn {l, ids} -> {l, Enum.sort(ids)} end)
      last = m + length(proof) - 1
      unit_ids = for({c, i} <- Enum.with_index(all), match?([_], c), do: {hd(c), i})

      # backwards: the empty clause first, then every marked lemma
      Enum.reduce_while(last..m//-1, {MapSet.new([last]), 0}, fn id, {marked, n} ->
        if MapSet.member?(marked, id) do
          case rup(db, occ, unit_ids, db[id], id) do
            {:ok, used} -> {:cont, {MapSet.union(marked, used), n + 1}}
            :fail -> {:halt, {:error, {:not_rup, id - m, db[id]}}}
          end
        else
          {:cont, {marked, n}}
        end
      end)
      |> case do
        {:error, _} = e -> e
        {marked, n} ->
          {:ok, %{lemmas: length(proof), checked: n, core_lemmas: Enum.count(marked, &(&1 >= m)), core_clauses: Enum.count(marked, &(&1 < m))}}
      end
    end
  end

  # RUP of `lemma` against clauses with id < limit; on success the ids of the clauses the conflict used
  defp rup(db, occ, unit_ids, lemma, limit) do
    a0 = Map.new(lemma, fn l -> {abs(l), {l < 0, :assumed}} end)
    if Enum.any?(lemma, &(-&1 in lemma)) do
      {:ok, MapSet.new()}
    else
      units = Enum.take_while(unit_ids, fn {_, id} -> id < limit end)
      {a, q, conflict} =
        Enum.reduce_while(units, {a0, Enum.map(lemma, &(-&1)), nil}, fn {u, id}, {a, q, _} ->
          case Map.get(a, abs(u)) do
            nil -> {:cont, {Map.put(a, abs(u), {u > 0, id}), [u | q], nil}}
            {v, _} -> if v == (u > 0), do: {:cont, {a, q, nil}}, else: {:halt, {a, q, id}}
          end
        end)
      case conflict || prop(db, occ, limit, a, Enum.reverse(q)) do
        {:conflict, cid, a} -> {:ok, trace(db, a, cid)}
        :fixpoint -> :fail
        cid when is_integer(cid) -> {:ok, trace(db, a, cid)}
      end
    end
  end

  defp value(a, l), do: (case Map.get(a, abs(l)) do nil -> nil; {v, _} -> v == (l > 0) end)

  defp prop(_db, _occ, _lim, _a, []), do: :fixpoint

  defp prop(db, occ, lim, a, [l | q]) do
    res =
      Enum.reduce_while(Map.get(occ, -l, []), {a, q}, fn cid, {a, q} ->
        if cid >= lim do
          {:halt, {a, q}}
        else
          {sat, free} = Enum.reduce(db[cid], {false, []}, fn x, {s, f} -> (case value(a, x) do true -> {true, f}; nil -> {s, [x | f]}; false -> {s, f} end) end)
          cond do
            sat -> {:cont, {a, q}}
            free == [] -> {:halt, {:conflict, cid, a}}
            match?([_], free) -> (u = hd(free); {:cont, {Map.put(a, abs(u), {u > 0, cid}), [u | q]}})
            true -> {:cont, {a, q}}
          end
        end
      end)

    case res do
      {:conflict, _, _} = c -> c
      {a, q} -> prop(db, occ, lim, a, q)
    end
  end

  # the clauses behind a conflict: the conflicting clause and, recursively, the reasons of its literals
  defp trace(db, a, cid), do: trace(db, a, [cid], MapSet.new())
  defp trace(_db, _a, [], seen), do: seen
  defp trace(db, a, [c | rest], seen) do
    if MapSet.member?(seen, c) do
      trace(db, a, rest, seen)
    else
      reasons = for l <- db[c], {_, r} <- [Map.get(a, abs(l))], is_integer(r), r != c, do: r
      trace(db, a, reasons ++ rest, MapSet.put(seen, c))
    end
  end
end
