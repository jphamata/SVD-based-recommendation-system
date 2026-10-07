defmodule Vapor.Logic.SAT do
  @moduledoc """
  A conflict-driven clause-learning SAT solver that **writes its proof**
  (docs/LOGICA.md §1), and the problems it settles.

  The solver: two watched literals per clause, first-UIP learning with
  clause minimisation by self-subsumption, activity-ordered decisions
  with phase saving, Luby restarts. On SAT it returns the model (which
  anyone checks in linear time); on UNSAT it returns every learned
  clause in order — a **DRUP proof** — which `Vapor.Logic.DRUP` checks
  by reverse unit propagation, with code that shares nothing with the
  solver. The solver proposes; the checker decides.

  Input is DIMACS CNF (the format of every SAT competition) or a
  propositional formula (`Vapor.Logic.Formula`), and the classic
  generators of finite combinatorics live in `Vapor.Logic.Problems`
  (Schur numbers, van der Waerden numbers, Ramsey numbers, pigeonhole,
  N-queens, graph colouring, Sudoku).
  """

  @doc "Parse DIMACS CNF: `{:ok, %{vars, clauses}}` or `{:error, why}`."
  def dimacs(text) do
    toks = text |> String.split("\n") |> Enum.reject(&(String.starts_with?(String.trim(&1), "c") or String.starts_with?(String.trim(&1), "%")))
    {header, body} = Enum.split_with(toks, &String.starts_with?(String.trim(&1), "p"))
    lits = body |> Enum.join(" ") |> String.split() |> Enum.map(&Integer.parse/1)

    cond do
      Enum.any?(lits, &(&1 == :error or elem(&1, 1) != "")) -> {:error, "DIMACS: a token is not an integer"}
      true ->
        clauses = lits |> Enum.map(&elem(&1, 0)) |> Enum.chunk_by(&(&1 == 0)) |> Enum.reject(&(&1 == [0] or hd(&1) == 0)) |> Enum.map(&Enum.uniq/1)
        vars = case header do
          [h | _] -> (case String.split(h) do ["p", "cnf", v, _c] -> String.to_integer(v); _ -> 0 end)
          [] -> 0
        end
        vars = max(vars, clauses |> List.flatten() |> Enum.map(&abs/1) |> Enum.max(fn -> 0 end))
        {:ok, %{vars: vars, clauses: clauses}}
    end
  end

  @doc "A CNF as DIMACS text."
  def to_dimacs(%{vars: v, clauses: cs}), do: "p cnf #{v} #{length(cs)}\n" <> Enum.map_join(cs, "", &(Enum.join(&1, " ") <> " 0\n"))

  @doc """
  Solve: `{:sat, model}` (model: `%{var => boolean}`), `{:unsat, proof}`
  (proof: learned clauses in order, ending with `[]`) or `{:unknown,
  :budget}` when `conflicts:` (default 2 000 000) runs out.
  `stats` are in the third element: `{tag, value, stats}`.
  """
  def solve(%{vars: n, clauses: clauses}, opts \\ []) do
    budget = Keyword.get(opts, :conflicts, 2_000_000)
    assign = :atomics.new(n + 1, signed: true)
    level = :atomics.new(n + 1, signed: true)
    reason = :atomics.new(n + 1, signed: true)
    phase = :atomics.new(n + 1, signed: true)
    db = :ets.new(:sat_clauses, [:set, :private])

    try do
      st = %{n: n, assign: assign, level: level, reason: reason, phase: phase, db: db, watches: %{}, next_id: 1, trail: [], lim: [], qhead: :queue.new(),
             act: Map.new(1..max(n, 1), &{&1, 0.0}), inc: 1.0, proof: [], conflicts: 0, decisions: 0, props: 0, budget: budget, learnts: 0}

      # trivial clauses: empty → UNSAT; tautologies dropped; units enqueued at level 0
      case Enum.reduce_while(clauses, {:ok, st}, fn c, {:ok, st} ->
             cond do
               c == [] -> {:halt, {:unsat_now, st}}
               Enum.any?(c, &(-&1 in c)) -> {:cont, {:ok, st}}
               true -> add_clause(st, c, false)
             end
           end) do
        {:unsat_now, st} -> {:unsat, [[]], stats(st)}
        {:ok, st} ->
          case search(st, 0, luby(1) * 100, 1) do
            {:sat, st} -> {:sat, model(st), stats(st)}
            {:unsat, st} -> {:unsat, Enum.reverse([[] | st.proof]), stats(st)}
            {:unknown, st} -> {:unknown, :budget, stats(st)}
          end
        other -> other
      end
    after
      :ets.delete(db)
    end
  end

  defp stats(st), do: %{conflicts: st.conflicts, decisions: st.decisions, propagations: st.props, learned: st.learnts, vars: st.n}

  defp model(st), do: for(v <- 1..max(st.n, 1)//1, st.n > 0, into: %{}, do: {v, :atomics.get(st.assign, v) == 1})

  # value of a literal: 1 true, -1 false, 0 unassigned
  defp val(st, l) do
    a = :atomics.get(st.assign, abs(l))
    cond do
      a == 0 -> 0
      (a == 1) == (l > 0) -> 1
      true -> -1
    end
  end

  defp enqueue(st, l, reason_id) do
    v = abs(l)
    :atomics.put(st.assign, v, if(l > 0, do: 1, else: 2))
    :atomics.put(st.level, v, length(st.lim))
    :atomics.put(st.reason, v, reason_id)
    :atomics.put(st.phase, v, if(l > 0, do: 1, else: 2))
    %{st | trail: [l | st.trail], qhead: :queue.in(l, st.qhead)}
  end

  # add an original (learnt = false) or learned clause; returns {:ok, st} or a conflict marker at level 0
  defp add_clause(st, [l], _learnt) do
    case val(st, l) do
      1 -> {:cont, {:ok, st}}
      0 -> {:cont, {:ok, enqueue(st, l, 0)}}
      -1 -> {:halt, {:unsat_now, st}}
    end
  end

  defp add_clause(st, lits, _learnt) do
    id = st.next_id
    :ets.insert(st.db, {id, List.to_tuple(lits)})
    [a, b | _] = lits
    st = %{st | next_id: id + 1, watches: st.watches |> Map.update(-a, [id], &[id | &1]) |> Map.update(-b, [id], &[id | &1])}
    {:cont, {:ok, st}}
  end

  # ------------------------------------------------------------- propagate

  # returns {:ok, st} or {:conflict, clause_id, st}
  defp propagate(st) do
    case :queue.out(st.qhead) do
      {:empty, _} -> {:ok, st}
      {{:value, l}, q} -> propagate_lit(%{st | qhead: q, props: st.props + 1}, l)
    end
  end

  defp propagate_lit(st, l) do
    # clauses watching ¬l (i.e. stored under key l, since we index by the literal whose truth falsifies the watch)
    ws = Map.get(st.watches, l, [])
    st = %{st | watches: Map.put(st.watches, l, [])}
    visit(st, l, ws, [])
  end

  defp visit(st, l, [], keep), do: propagate(%{st | watches: Map.update(st.watches, l, keep, &(keep ++ &1))})

  defp visit(st, l, [cid | rest], keep) do
    [{_, c}] = :ets.lookup(st.db, cid)
    falsel = -l
    # make sure the false literal is at position 1
    c = if elem(c, 0) == falsel, do: c |> put_elem(0, elem(c, 1)) |> put_elem(1, falsel), else: c
    first = elem(c, 0)

    if val(st, first) == 1 do
      :ets.insert(st.db, {cid, c})
      visit(st, l, rest, [cid | keep])
    else
      case find_watch(st, c, 2, tuple_size(c)) do
        nil ->
          :ets.insert(st.db, {cid, c})
          case val(st, first) do
            -1 ->
              # conflict: restore the remaining watches
              {:conflict, cid, %{st | watches: Map.update(st.watches, l, keep ++ [cid | rest], &(keep ++ [cid | rest] ++ &1)), qhead: :queue.new()}}
            0 -> visit(enqueue(st, first, cid), l, rest, [cid | keep])
            1 -> visit(st, l, rest, [cid | keep])
          end

        k ->
          nl = elem(c, k)
          c = c |> put_elem(1, nl) |> put_elem(k, falsel)
          :ets.insert(st.db, {cid, c})
          visit(%{st | watches: Map.update(st.watches, -nl, [cid], &[cid | &1])}, l, rest, keep)
      end
    end
  end

  defp find_watch(_st, _c, k, n) when k >= n, do: nil
  defp find_watch(st, c, k, n), do: if(val(st, elem(c, k)) != -1, do: k, else: find_watch(st, c, k + 1, n))

  # ---------------------------------------------------------------- search

  defp search(st, nconf, limit, restarts) do
    case propagate(st) do
      {:conflict, cid, st} ->
        st = %{st | conflicts: st.conflicts + 1}
        cond do
          st.lim == [] -> {:unsat, st}
          st.conflicts >= st.budget -> {:unknown, st}
          true ->
            {learnt, back} = analyze(st, cid)
            st = %{st | act: Enum.reduce(learnt, st.act, fn l, a -> Map.update!(a, abs(l), &(&1 + st.inc)) end)}
            st = backjump(st, back)
            st = %{st | proof: [learnt | st.proof], learnts: st.learnts + 1, inc: st.inc * 1.05}
            st =
              case learnt do
                [u] -> enqueue(st, u, 0)
                [u | _] ->
                  {:cont, {:ok, st}} = add_clause(st, learnt, true)
                  enqueue(st, u, st.next_id - 1)
              end
            st = if st.inc > 1.0e100, do: %{st | inc: st.inc * 1.0e-100, act: Map.new(st.act, fn {k, v} -> {k, v * 1.0e-100} end)}, else: st
            search(st, nconf + 1, limit, restarts)
        end

      {:ok, st} ->
        cond do
          nconf >= limit ->
            st = backjump(st, 0)
            search(st, 0, luby(restarts + 1) * 100, restarts + 1)
          true ->
            case pick(st) do
              nil -> {:sat, st}
              v ->
                lit = if :atomics.get(st.phase, v) == 1, do: v, else: -v
                st = %{st | lim: [st.trail | st.lim], decisions: st.decisions + 1}
                search(enqueue(st, lit, 0), nconf, limit, restarts)
            end
        end
    end
  end

  defp pick(st) do
    Enum.reduce(st.act, {nil, -1.0}, fn {v, a}, {best, ba} ->
      if a > ba and :atomics.get(st.assign, v) == 0, do: {v, a}, else: {best, ba}
    end)
    |> elem(0)
  end

  defp backjump(st, lvl) do
    depth = length(st.lim)
    if depth <= lvl do
      st
    else
      # lim is a stack of trails at each decision; keep the one at level `lvl`
      target = Enum.at(st.lim, depth - lvl - 1)
      undo = Enum.take(st.trail, length(st.trail) - length(target))
      Enum.each(undo, fn l -> v = abs(l); :atomics.put(st.assign, v, 0); :atomics.put(st.reason, v, 0) end)
      %{st | trail: target, lim: Enum.drop(st.lim, depth - lvl), qhead: :queue.new()}
    end
  end

  # first-UIP conflict analysis; returns {learnt clause (asserting literal first), backjump level}
  defp analyze(st, cid) do
    cur = length(st.lim)
    [{_, c}] = :ets.lookup(st.db, cid)
    {learnt, _seen} = walk(st, Tuple.to_list(c), st.trail, MapSet.new(), [], 0, cur, nil)
    learnt = minimise(st, learnt)
    [uip | rest] = learnt
    rest = Enum.sort_by(rest, &(-:atomics.get(st.level, abs(&1))))
    back = case rest do [] -> 0; [l | _] -> :atomics.get(st.level, abs(l)) end
    {[uip | rest], back}
  end

  defp walk(st, lits, trail, seen, out, count, cur, pivot) do
    {seen, out, count} =
      Enum.reduce(lits, {seen, out, count}, fn l, {seen, out, count} ->
        v = abs(l)
        if v == pivot or MapSet.member?(seen, v) or :atomics.get(st.level, v) == 0 do
          {seen, out, count}
        else
          seen = MapSet.put(seen, v)
          if :atomics.get(st.level, v) == cur, do: {seen, out, count + 1}, else: {seen, [l | out], count}
        end
      end)

    # the next literal of the current level on the trail that is marked
    [p | trail] = Enum.drop_while(trail, &(not MapSet.member?(seen, abs(&1))))
    count = count - 1
    if count == 0 do
      {[-p | out], seen}
    else
      r = :atomics.get(st.reason, abs(p))
      [{_, c}] = :ets.lookup(st.db, r)
      walk(st, Tuple.to_list(c), trail, seen, out, count, cur, abs(p))
    end
  end

  # drop a literal whose reason's other literals are all in the clause (self-subsumption)
  defp minimise(st, [uip | rest]) do
    inset = MapSet.new(Enum.map([uip | rest], &abs/1))
    kept = Enum.reject(rest, fn l ->
      r = :atomics.get(st.reason, abs(l))
      r != 0 and (
        [{_, c}] = :ets.lookup(st.db, r)
        c |> Tuple.to_list() |> Enum.all?(fn x -> abs(x) == abs(l) or MapSet.member?(inset, abs(x)) or :atomics.get(st.level, abs(x)) == 0 end))
    end)
    [uip | kept]
  end

  @doc false
  def luby(i) do
    # the Luby sequence 1 1 2 1 1 2 4 …
    k = Enum.find(1..64, fn k -> i <= Integer.pow(2, k) - 1 end)
    if i == Integer.pow(2, k) - 1, do: Integer.pow(2, k - 1), else: luby(i - Integer.pow(2, k - 1) + 1)
  end
end
