defmodule Vapor.Logic.MIP do
  @moduledoc """
  Mixed-integer linear programming in exact rational arithmetic, with an
  optimality certificate that any reader can check by multiplication
  (docs/LOGIC.md §6). The text form is `Vapor.Logic.LP`'s, plus the integer
  declarations:

      maximize 5a + 4b + 3c
      2a + 3b + c <= 5
      4a + b + 2c <= 11
      3a + 4b + 2c <= 8
      int a, b, c          # integer variables (≥ 0 unless declared free)
      bin d                # 0/1 variables

  **The solver** is branch and bound over `LP`'s exact simplex: each node is
  the relaxation with extra bounds; a fractional integer variable `v = 7/2`
  splits the node into `v ≤ 3` and `v ≥ 4`, which together keep every
  integer point of the parent. A node is closed when its relaxation is
  infeasible, or when its bound cannot beat the incumbent.

  **The certificate** is the search tree, the incumbent, and at each leaf
  the object that closes it:

  | leaf | object | the check |
  |---|---|---|
  | infeasible | Farkas `y` for the node's rows | `Aᵀy ≥ 0` with the row signs, `bᵀy < 0` |
  | bounded | a dual-feasible `y` for the node's rows | `Aᵀy ≥ c` (free variables `=`), `bᵀy + c₀ ≤ z*` (max form) |

  `check/2` shares nothing with the solver: it re-derives every node's rows
  from the branching decisions, confirms that each split is `v ≤ k` /
  `v ≥ k + 1` on an integer variable (so the leaves cover every integer
  point), checks every leaf's `y` by weak duality, and checks the
  incumbent's feasibility, integrality and value. A wrong optimum cannot
  pass: some leaf would have to hold the better point, and its `y` bounds
  that leaf by `z*`. This is the shape of certificate that VIPR (Cheung,
  Gleixner & Steffy, 2017) standardised for exact MIP solvers; branching
  only, no cutting planes.

  Bounds, stated: at most 200 variables and 200 rows (the LP's limits, with
  the branching rows counted), `max_nodes` (default 20 000) nodes. Running
  out gives `:exhausted` with the incumbent and the best open bound: a gap,
  not a guess. A relaxation that is unbounded at the root is refused. With
  rational data an integer program whose relaxation is unbounded is either
  infeasible or unbounded (Meyer, 1974), and telling which needs an
  integer ray this solver does not search for; bounding the variables
  closes the question.
  """
  alias Vapor.Logic.LP
  import LP, only: [qadd: 2, qmul: 2, qsign: 1, qcmp: 2, show: 1]

  @max_nodes 20_000

  # ---------------------------------------------------------------- parsing

  @doc """
  Parse the text form: `{:ok, %{lp, int}}` where `lp` is `LP.parse/1`'s
  problem (with `bin` variables bounded by 1) and `int` the sorted integer
  variables.
  """
  def parse(text) when is_binary(text) do
    lines = String.split(text, "\n")
    decl? = &(&1 =~ ~r/^\s*(int|integer|inteiro|inteiros|bin|binary|binario|binário)\s+/i)
    {decls, rest} = Enum.split_with(lines, fn l -> l |> String.split("#") |> hd() |> decl?.() end)

    {ints, bins} =
      Enum.reduce(decls, {[], []}, fn l, {is, bs} ->
        [kw | names] = l |> String.split("#") |> hd() |> String.split(~r/[\s,]+/, trim: true)
        if String.downcase(kw) in ["bin", "binary", "binario", "binário"], do: {is, bs ++ names}, else: {is ++ names, bs}
      end)

    with {:ok, lp} <- LP.parse(Enum.join(rest, "\n")) do
      bad = Enum.reject(ints ++ bins, &(&1 =~ ~r/^[A-Za-z_][A-Za-z0-9_]*$/))
      unknown = Enum.reject(ints ++ bins, &(&1 in lp.vars))

      cond do
        bad != [] -> {:error, "not a variable name: #{Enum.join(bad, ", ")}"}
        unknown != [] -> {:error, "declared integer but unused in the objective and the rows: #{Enum.join(unknown, ", ")}"}
        ints ++ bins == [] -> {:error, "no integer variables: declare them with `int x, y` or `bin z` (or solve it as an LP)"}
        true ->
          rows = lp.rows ++ for(b <- Enum.uniq(bins), do: {%{b => {1, 1}}, :le, {1, 1}})
          {:ok, %{lp: %{lp | rows: rows, free: lp.free -- bins}, int: Enum.sort(Enum.uniq(ints ++ bins))}}
      end
    end
  end

  # ------------------------------------------------------------------ solve

  @doc """
  Solve a MIP (its text or `parse/1`'s map). `{:ok, result}`:

    * `status: :optimal` with `x`, `objective` and `certificate` (checked
      before returning: `check` holds `check/2`'s verdict);
    * `status: :infeasible` with a certificate whose every leaf is a
      Farkas certificate;
    * `status: :exhausted` with the incumbent (if any) and `bound`, the best
      open relaxation value: the optimum lies between them.

  Options: `max_nodes` (default #{@max_nodes}).
  """
  def solve(text_or_problem, opts \\ [])
  def solve(text, opts) when is_binary(text), do: with({:ok, p} <- parse(text), do: solve(p, opts))

  def solve(%{lp: lp, int: ints} = p, opts) do
    budget = Keyword.get(opts, :max_nodes, @max_nodes)

    case relax(lp, []) do
      {:unbounded, _} ->
        {:error, "the relaxation is unbounded: bound the integer variables (an unbounded relaxation leaves the MIP infeasible or unbounded, undecided here)"}

      _ ->
        st = %{incumbent: nil, nodes: 0, budget: budget, open_bound: nil}
        {tree, st} = branch(lp, ints, [], st)

        case {st.open_bound, st.incumbent} do
          {nil, nil} ->
            cert = %{incumbent: nil, tree: tree}
            {:ok, %{status: :infeasible, sense: lp.sense, nodes: st.nodes, certificate: cert, check: check(p, cert)}}

          {nil, {x, z}} ->
            cert = %{incumbent: %{x: x, objective: z}, tree: tree}
            {:ok, %{status: :optimal, sense: lp.sense, x: x, objective: z, nodes: st.nodes, certificate: cert, check: check(p, cert)}}

          {bound, inc} ->
            {:ok, %{status: :exhausted, sense: lp.sense, nodes: st.nodes, bound: bound,
                    x: inc && elem(inc, 0), objective: inc && elem(inc, 1)}}
        end
    end
  end

  # one node: its relaxation, then close it or split it (depth first, the "≤" side first);
  # out of budget, a node is still closed when it can be, and only left open when it would split
  defp branch(lp, ints, bounds, st) do
    st = %{st | nodes: st.nodes + 1}

    case relax(lp, bounds) do
      {:infeasible, y} ->
        {{:leaf, :infeasible, y}, st}

      {:optimal, x, z, y} ->
        v = fractional(x, ints)

        cond do
          st.incumbent != nil and not improves?(lp.sense, z, elem(st.incumbent, 1)) ->
            {{:leaf, :bound, y}, st}

          v == nil ->
            {{:leaf, :bound, y}, %{st | incumbent: {x, z}}}

          st.nodes >= st.budget ->
            {{:open}, %{st | open_bound: better_bound(lp.sense, st.open_bound, z)}}

          true ->
            k = floor_q(x[v])
            {lo, st} = branch(lp, ints, bounds ++ [{v, :le, k}], st)
            {hi, st} = branch(lp, ints, bounds ++ [{v, :ge, k + 1}], st)
            # bounded leaves stay valid as the incumbent improves inside the subtree: z* only gets better
            {{:branch, v, k, lo, hi}, st}
        end
    end
  end

  defp better_bound(_sense, nil, b), do: b
  defp better_bound(_sense, a, nil), do: a
  defp better_bound(:max, a, b), do: if(qcmp(a, b) >= 0, do: a, else: b)
  defp better_bound(:min, a, b), do: if(qcmp(a, b) <= 0, do: a, else: b)

  defp improves?(:max, z, best), do: qcmp(z, best) > 0
  defp improves?(:min, z, best), do: qcmp(z, best) < 0

  defp relax(lp, bounds) do
    {:ok, r} = LP.solve(%{lp | rows: lp.rows ++ rows_of(bounds)})

    case r.status do
      :optimal -> {:optimal, r.x, r.objective, r.certificate.y}
      :infeasible -> {:infeasible, r.certificate.y}
      :unbounded -> {:unbounded, r.x}
    end
  end

  defp rows_of(bounds), do: for({v, op, k} <- bounds, do: {%{v => {1, 1}}, op, {k, 1}})

  defp fractional(x, ints), do: Enum.find(ints, fn v -> elem(x[v], 1) != 1 end)

  defp floor_q({n, d}), do: Integer.floor_div(n, d)

  # ------------------------------------------------------------------ check

  @doc """
  Check a certificate `%{incumbent: nil | %{x, objective}, tree}` against a
  problem using exact multiplication and comparison only. `tree` is
  `{:branch, var, k, le_subtree, ge_subtree}` or `{:leaf, :infeasible |
  :bound, y}`. Returns `%{accepted, reason}`. Anyone may propose a
  certificate (a person, a search, a model through MCP): acceptance never
  depends on who proposed it.
  """
  def check(%{lp: lp, int: ints}, %{tree: tree} = cert) do
    inc = cert[:incumbent]

    with :ok <- check_incumbent(lp, ints, inc),
         {:ok, leaves} <- check_tree(lp, ints, tree, [], inc, 0) do
      case inc do
        nil -> %{accepted: true, reason: "#{leaves} leaves, each infeasible by Farkas: no integer point exists"}
        %{objective: z} -> %{accepted: true, reason: "incumbent feasible and integral; #{leaves} leaves cover every integer point and none can beat #{show(z)}: optimal"}
      end
    else
      {:error, why} -> %{accepted: false, reason: why}
    end
  end

  def check(_, _), do: %{accepted: false, reason: "malformed certificate"}

  defp check_incumbent(_lp, _ints, nil), do: :ok

  defp check_incumbent(lp, ints, %{x: x, objective: z}) do
    value = Enum.reduce(lp.vars, lp.c0, fn v, s -> qadd(s, qmul(Map.get(lp.c, v, {0, 1}), Map.get(x, v, {0, 1}))) end)

    cond do
      Enum.any?(lp.vars, &(not Map.has_key?(x, &1))) -> {:error, "the incumbent misses a variable"}
      Enum.any?(ints, &(elem(x[&1], 1) != 1)) -> {:error, "the incumbent is not integral"}
      Enum.any?(lp.vars, &(&1 not in lp.free and qsign(x[&1]) < 0)) -> {:error, "the incumbent has a negative variable declared ≥ 0"}
      (i = violated(lp.rows, x)) != nil -> {:error, "the incumbent violates constraint #{i + 1}"}
      qcmp(value, z) != 0 -> {:error, "the incumbent's objective is #{show(value)}, not #{show(z)}"}
      true -> :ok
    end
  end

  defp check_incumbent(_, _, _), do: {:error, "malformed incumbent"}

  defp violated(rows, x) do
    Enum.find_value(Enum.with_index(rows), fn {{co, op, rhs}, i} ->
      s = qcmp(Enum.reduce(co, {0, 1}, fn {v, a}, acc -> qadd(acc, qmul(a, Map.get(x, v, {0, 1}))) end), rhs)
      if (op == :le and s > 0) or (op == :ge and s < 0) or (op == :eq and s != 0), do: i
    end)
  end

  @max_depth 400

  defp check_tree(_lp, _ints, _t, _bounds, _inc, depth) when depth > @max_depth, do: {:error, "the tree is deeper than #{@max_depth}"}

  defp check_tree(lp, ints, {:branch, v, k, lo, hi}, bounds, inc, depth) when is_binary(v) and is_integer(k) do
    if v in ints do
      with {:ok, a} <- check_tree(lp, ints, lo, bounds ++ [{v, :le, k}], inc, depth + 1),
           {:ok, b} <- check_tree(lp, ints, hi, bounds ++ [{v, :ge, k + 1}], inc, depth + 1),
           do: {:ok, a + b}
    else
      {:error, "a branch on #{v}, which is not an integer variable: the split would lose points"}
    end
  end

  defp check_tree(lp, _ints, {:leaf, :infeasible, y}, bounds, _inc, _depth) do
    node = %{lp | rows: lp.rows ++ rows_of(bounds)}

    case LP.check(node, %{status: :infeasible, y: y}) do
      %{accepted: true} -> {:ok, 1}
      _ -> {:error, "a leaf marked infeasible has no valid Farkas certificate (bounds #{describe(bounds)})"}
    end
  end

  defp check_tree(_lp, _ints, {:leaf, :bound, _y}, _bounds, nil, _depth),
    do: {:error, "a leaf closed by bound needs an incumbent to compare with"}

  defp check_tree(lp, _ints, {:leaf, :bound, y}, bounds, %{objective: z}, _depth) do
    node = %{lp | rows: lp.rows ++ rows_of(bounds)}

    case dual_bound(node, y) do
      {:ok, b} ->
        # max form: bᵀy + c₀ ≥ every feasible value at the leaf; it must not exceed z*
        ok = if lp.sense == :max, do: qcmp(b, z) <= 0, else: qcmp(b, z) >= 0
        if ok, do: {:ok, 1}, else: {:error, "a leaf's bound #{show(b)} could beat the incumbent #{show(z)} (bounds #{describe(bounds)})"}

      {:error, why} ->
        {:error, "#{why} (bounds #{describe(bounds)})"}
    end
  end

  defp check_tree(_, _, _, _, _, _), do: {:error, "malformed tree node"}

  # weak duality: y with the row signs and Aᵀy ≥ sgn·c (= on free variables) bounds every
  # feasible x by sgn·cᵀx ≤ bᵀy; returned in the problem's own sense, constant included
  defp dual_bound(p, y) when is_list(y) and length(y) == length(p.rows) do
    sgn = if p.sense == :max, do: 1, else: -1
    sign_ok = Enum.zip(p.rows, y) |> Enum.all?(fn {{_, op, _}, yi} -> case op do :le -> qsign(yi) >= 0; :ge -> qsign(yi) <= 0; :eq -> true end end)
    aty = fn v -> Enum.zip(p.rows, y) |> Enum.reduce({0, 1}, fn {{co, _, _}, yi}, s -> qadd(s, qmul(Map.get(co, v, {0, 1}), yi)) end) end

    feasible =
      Enum.all?(p.vars, fn v ->
        s = qcmp(aty.(v), qmul({sgn, 1}, Map.get(p.c, v, {0, 1})))
        if v in p.free, do: s == 0, else: s >= 0
      end)

    by = Enum.zip(p.rows, y) |> Enum.reduce({0, 1}, fn {{_, _, r}, yi}, s -> qadd(s, qmul(r, yi)) end)

    if sign_ok and feasible, do: {:ok, qadd(qmul({sgn, 1}, by), p.c0)}, else: {:error, "a leaf's y is not dual-feasible"}
  end

  defp dual_bound(_, _), do: {:error, "a leaf's y has the wrong length"}

  defp describe([]), do: "none"
  defp describe(bounds), do: Enum.map_join(bounds, ", ", fn {v, op, k} -> "#{v} #{if op == :le, do: "≤", else: "≥"} #{k}" end)

  # ------------------------------------------------------------ presentation

  @doc "A JSON-ready view of a result (rationals as exact strings and floats; the tree as nested maps)."
  def present({:ok, r}) do
    num = fn x -> %{exact: show(x), value: LP.to_float(x)} end

    base = %{status: r.status, sense: r.sense, nodes: r.nodes}
    base = if r[:x], do: Map.put(base, :x, Map.new(r.x, fn {k, v} -> {k, num.(v)} end)), else: base
    base = if r[:objective], do: Map.put(base, :objective, num.(r.objective)), else: base
    base = if r[:bound], do: Map.put(base, :bound, num.(r.bound)), else: base
    base = if r[:check], do: Map.put(base, :check, r.check), else: base
    if r[:certificate], do: Map.put(base, :tree, tree_view(r.certificate.tree)), else: base
  end

  def present(e), do: e

  defp tree_view({:branch, v, k, lo, hi}), do: %{split: v, at: k, le: tree_view(lo), ge: tree_view(hi)}
  defp tree_view({:leaf, kind, y}), do: %{leaf: kind, y: Enum.map(y, &show/1)}
  defp tree_view({:open}), do: %{open: true}
end
