defmodule Vapor.Logic.LP do
  @moduledoc """
  Linear programming in **exact rational arithmetic**, with certificates
  any reader can check by multiplication (docs/LOGICA.md §5,
  docs/FINANCAS.md §8).

      maximize 3x + 2y
      subject to
      x + y <= 4
      x + 3y <= 6
      x <= 3
      free z            # variables are ≥ 0 unless declared free

  The simplex method (two phases, Bland's rule — it cannot cycle) runs on
  rationals, so "optimal", "infeasible" and "unbounded" are decided, not
  estimated. Each verdict comes with the object that proves it, found by
  solving the **alternative system** and checked by `check/2`, which only
  multiplies and compares:

  | verdict | certificate | the check |
  |---|---|---|
  | optimal | the primal x and a dual y | x feasible; y dual-feasible (Aᵀy ≥ c with the row signs); cᵀx = bᵀy |
  | infeasible | Farkas y | Aᵀy ≥ 0, y ≥ 0 on ≤ rows, ≤ 0 on ≥ rows, bᵀy < 0 |
  | unbounded | a feasible x and a ray d ≥ 0 | A d has the row's sign (≤ 0, ≥ 0, = 0) and cᵀd > 0 |

  The same object is what a proposer — a person, a search, a language
  model through `logic_check` — must hand in: acceptance never depends on
  who proposed.
  """

  # ------------------------------------------------------------ rationals

  @doc false
  def q(n, d \\ 1)
  def q(_, 0), do: raise(ArgumentError, "zero denominator")
  def q(n, d) when d < 0, do: q(-n, -d)
  def q(n, d), do: (g = Integer.gcd(n, d); {div(n, g), div(d, g)})

  def qadd({a, b}, {c, d}), do: q(a * d + c * b, b * d)
  def qsub({a, b}, {c, d}), do: q(a * d - c * b, b * d)
  def qmul({a, b}, {c, d}), do: q(a * c, b * d)
  def qdiv({a, b}, {c, d}) when c != 0, do: q(a * d, b * c)
  def qneg({a, b}), do: {-a, b}
  def qsign({a, _}), do: (cond do a > 0 -> 1; a < 0 -> -1; true -> 0 end)
  def qcmp(x, y), do: qsign(qsub(x, y))
  def qzero?({a, _}), do: a == 0
  def qdot(xs, ys), do: Enum.zip(xs, ys) |> Enum.reduce({0, 1}, fn {a, b}, s -> qadd(s, qmul(a, b)) end)

  @doc "A rational from an integer, a decimal string (`\"0.25\"`), a fraction (`\"3/4\"`) or a float (by its shortest decimal)."
  def rat(x) when is_integer(x), do: {x, 1}
  def rat({n, d}) when is_integer(n) and is_integer(d), do: q(n, d)
  def rat(x) when is_float(x), do: rat(:erlang.float_to_binary(x, [:short]))
  def rat(s) when is_binary(s) do
    s = String.trim(s)
    case String.split(s, "/") do
      [a, b] -> qdiv(rat(a), rat(b))
      [a] -> %{c: c, e: e} = Vapor.Finance.Money.parse!(a); q(c, Integer.pow(10, e))
    end
  end

  def to_float({n, d}), do: n / d
  def show({n, 1}), do: Integer.to_string(n)
  def show({n, d}), do: "#{n}/#{d}"

  # ---------------------------------------------------------------- parsing

  @doc """
  Parse the text form into `%{sense, vars, c, rows: [{coeffs, op, rhs}]}`
  (coefficients as a map var → rational; `op` ∈ `:le`, `:ge`, `:eq`), the
  free variables in `free`.
  """
  def parse(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#") |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    {obj, rest} = Enum.split_with(lines, &(&1 =~ ~r/^(max|maximi[sz]e|maximizar|min|minimi[sz]e|minimizar)\b/i))
    case obj do
      [o] ->
        [_, w, e] = Regex.run(~r/^(\w+)\s+(.*)$/, o)
        sense = if String.downcase(w) =~ ~r/^max/, do: :max, else: :min
        with {:ok, {c, k}} <- linear(e) do
          rest = Enum.reject(rest, &(&1 =~ ~r/^(subject to|s\.t\.|st|sujeito a|tal que)\s*:?$/i))
          {frees, cons} = Enum.split_with(rest, &(&1 =~ ~r/^(free|livre)\s+/i))
          free = frees |> Enum.flat_map(&(&1 |> String.replace(~r/^(free|livre)\s+/i, "") |> String.split(~r/[\s,]+/, trim: true)))
          rows =
            Enum.reduce_while(cons, {:ok, []}, fn l, {:ok, acc} ->
              case Regex.run(~r/^(.*?)(<=|>=|==|=|≤|≥)(.*)$/u, l) do
                [_, lhs, op, rhs] ->
                  with {:ok, {a, ka}} <- linear(lhs), {:ok, {b, kb}} <- linear(rhs) do
                    coeffs = Map.merge(a, Map.new(b, fn {v, x} -> {v, qneg(x)} end), fn _, x, y -> qadd(x, y) end) |> Map.reject(fn {_, x} -> qzero?(x) end)
                    opn = case op do o when o in ["<=", "≤"] -> :le; o when o in [">=", "≥"] -> :ge; _ -> :eq end
                    {:cont, {:ok, acc ++ [{coeffs, opn, qsub(kb, ka)}]}}
                  else
                    {:error, w} -> {:halt, {:error, "#{w} in #{inspect(l)}"}}
                  end
                _ -> {:halt, {:error, "a constraint needs <=, >= or =: #{inspect(l)}"}}
              end
            end)
          with {:ok, rows} <- rows do
            vars = (Map.keys(c) ++ Enum.flat_map(rows, fn {a, _, _} -> Map.keys(a) end) ++ free) |> Enum.uniq() |> Enum.sort()
            cond do
              rows == [] -> {:error, "no constraints"}
              length(vars) > 200 or length(rows) > 200 -> {:error, "at most 200 variables and 200 constraints here"}
              true -> {:ok, %{sense: sense, vars: vars, c: c, c0: k, rows: rows, free: Enum.uniq(free)}}
            end
          end
        end
      [] -> {:error, "the first line is the objective: maximize … or minimize …"}
      _ -> {:error, "one objective only"}
    end
  end

  # a linear expression: Σ ±(coef)(var) and constants → {%{var => coef}, constant}
  defp linear(e) do
    e = e |> String.replace("−", "-") |> String.replace(~r/\s+/, "")
    e = if e == "", do: "0", else: e
    terms = Regex.scan(~r/([+-]?)((?:\d+(?:\.\d+)?(?:\/\d+)?)?)\*?([A-Za-z_][A-Za-z0-9_]*)?/u, e) |> Enum.reject(fn [whole | _] -> whole == "" end)
    consumed = terms |> Enum.map(&hd/1) |> Enum.join()
    if consumed != e do
      {:error, "cannot read #{inspect(e)} as a linear expression"}
    else
      Enum.reduce(terms, {:ok, {%{}, {0, 1}}}, fn parts, {:ok, {m, k}} ->
        [_, sign, coef | v] = parts ++ [nil]
        v = List.first(v)
        c = if coef == "", do: {1, 1}, else: rat(coef)
        c = if sign == "-", do: qneg(c), else: c
        cond do
          v == nil and coef == "" -> {:ok, {m, k}}
          v == nil -> {:ok, {m, qadd(k, c)}}
          true -> {:ok, {Map.update(m, v, c, &qadd(&1, c)), k}}
        end
      end)
    end
  end

  # ------------------------------------------------------------ the solver

  @doc """
  Solve a parsed problem (or its text). `{:ok, %{status, x, objective,
  certificate, …}}` with status `:optimal`, `:infeasible` or `:unbounded`.
  """
  def solve(text) when is_binary(text), do: with({:ok, p} <- parse(text), do: solve(p))

  def solve(%{} = p) do
    {cols, a, b, ops, c} = standard(p)
    case simplex_general(a, ops, b, c) do
      {:optimal, xs, _} ->
        x = unsplit(p, cols, xs)
        obj = qadd(qdot(Enum.map(p.vars, &Map.get(p.c, &1, {0, 1})), Enum.map(p.vars, &x[&1])), p.c0)
        # the dual: minimise bᵀy s.t. Aᵀy ≥ c (columns), y ≥ 0 on ≤, ≤ 0 on ≥, free on = (for the max form)
        y = dual(a, ops, b, c)
        cert = %{x: x, y: y}
        {:ok, %{status: :optimal, sense: p.sense, x: x, objective: obj, certificate: cert, check: check(p, %{status: :optimal, x: x, y: y})}}
      :infeasible ->
        y = farkas(a, ops, b)
        {:ok, %{status: :infeasible, sense: p.sense, certificate: %{y: y}, check: check(p, %{status: :infeasible, y: y})}}
      {:unbounded, xs} ->
        x = unsplit(p, cols, xs)
        d = ray(a, ops, c) |> then(&unsplit(p, cols, &1))
        {:ok, %{status: :unbounded, sense: p.sense, x: x, certificate: %{x: x, ray: d}, check: check(p, %{status: :unbounded, x: x, ray: d})}}
    end
  end

  # columns: each variable once (≥ 0), or twice (+/−) when free; the objective as a max
  defp standard(p) do
    cols = Enum.flat_map(p.vars, fn v -> if v in p.free, do: [{v, 1}, {v, -1}], else: [{v, 1}] end)
    a = for {coeffs, _, _} <- p.rows, do: (for {v, s} <- cols, do: (x = Map.get(coeffs, v, {0, 1}); if(s == 1, do: x, else: qneg(x))))
    b = for {_, _, r} <- p.rows, do: r
    ops = for {_, op, _} <- p.rows, do: op
    sgn = if p.sense == :max, do: 1, else: -1
    c = for {v, s} <- cols, do: (x = Map.get(p.c, v, {0, 1}); qmul({sgn * s, 1}, x))
    {cols, a, b, ops, c}
  end

  defp unsplit(p, cols, xs) do
    Enum.zip(cols, xs) |> Enum.reduce(Map.new(p.vars, &{&1, {0, 1}}), fn {{v, s}, x}, m -> Map.update!(m, v, &qadd(&1, qmul({s, 1}, x))) end)
  end

  @doc false
  # maximise cᵀx, rows a_i x (op) b_i, x ≥ 0 — two-phase simplex, Bland's rule, rationals
  def simplex_general(a, ops, b, c) do
    m = length(a); n = length(c)
    # slack (+1 on ≤), surplus (−1 on ≥); then flip rows with b < 0
    nslack = Enum.count(ops, &(&1 != :eq))
    {rows, _} =
      Enum.zip([a, ops, b]) |> Enum.map_reduce(0, fn {row, op, bi}, k ->
        slack = for j <- 0..(nslack - 1)//1, do: (cond do j == k and op == :le -> {1, 1}; j == k and op == :ge -> {-1, 1}; true -> {0, 1} end)
        {{row ++ slack, bi}, if(op == :eq, do: k, else: k + 1)}
      end)
    rows = Enum.map(rows, fn {r, bi} -> if qsign(bi) < 0, do: {Enum.map(r, &qneg/1), qneg(bi)}, else: {r, bi} end)
    nt = n + nslack
    # artificials on every row: basis = artificials; phase 1 maximises −Σ art
    tab = rows |> Enum.with_index() |> Enum.map(fn {{r, bi}, i} -> List.to_tuple(r ++ (for j <- 0..(m - 1), do: if(i == j, do: {1, 1}, else: {0, 1})) ++ [bi]) end)
    basis = Enum.map(0..(m - 1), &(nt + &1))
    c1 = List.duplicate({0, 1}, nt) ++ List.duplicate({-1, 1}, m)
    case run_simplex(tab, basis, c1, nt + m) do
      {:optimal, tab, basis} ->
        val = Enum.zip(basis, tab) |> Enum.reduce({0, 1}, fn {j, row}, s -> qadd(s, qmul(Enum.at(c1, j), elem(row, tuple_size(row) - 1))) end)
        if qsign(val) < 0 do
          :infeasible
        else
          # drive artificials out of the basis where possible, then drop their columns
          {tab, basis} = drive_out(tab, basis, nt)
          keep = Enum.zip(tab, basis) |> Enum.reject(fn {_, j} -> j >= nt end)
          tab2 = Enum.map(keep, fn {row, _} -> List.to_tuple(Enum.take(Tuple.to_list(row), nt) ++ [elem(row, tuple_size(row) - 1)]) end)
          basis2 = Enum.map(keep, &elem(&1, 1))
          c2 = c ++ List.duplicate({0, 1}, nslack)
          case run_simplex(tab2, basis2, c2, nt) do
            {:optimal, t, bs} -> {:optimal, primal(t, bs, n), bs}
            {:unbounded, t, bs, _} -> {:unbounded, primal(t, bs, n)}
          end
        end
      {:unbounded, _, _, _} -> :infeasible  # cannot happen: phase 1 is bounded by 0
    end
  end

  defp primal(tab, basis, n) do
    vals = Map.new(Enum.zip(basis, tab), fn {j, row} -> {j, elem(row, tuple_size(row) - 1)} end)
    for j <- 0..(n - 1), do: Map.get(vals, j, {0, 1})
  end

  defp drive_out(tab, basis, nt) do
    Enum.reduce(Enum.with_index(basis), {tab, basis}, fn {j, i}, {tab, basis} ->
      if j >= nt do
        row = Enum.at(tab, i)
        case Enum.find(0..(nt - 1), &(not qzero?(elem(row, &1)))) do
          nil -> {tab, basis}            # a redundant row: the artificial stays at 0 and the row is dropped
          col -> {pivot(tab, i, col), List.replace_at(basis, i, col)}
        end
      else
        {tab, basis}
      end
    end)
  end

  # tableau rows are tuples [coeffs…, rhs]; maximise cᵀx; Bland: lowest-index entering column with positive reduced cost
  defp run_simplex(tab, basis, c, ncols, it \\ 0) do
    cb = Enum.map(basis, &Enum.at(c, &1))
    reduced = fn j -> qsub(Enum.at(c, j), Enum.zip(cb, tab) |> Enum.reduce({0, 1}, fn {cbi, row}, s -> qadd(s, qmul(cbi, elem(row, j))) end)) end
    entering = Enum.find(0..(ncols - 1), fn j -> j not in basis and qsign(reduced.(j)) > 0 end)
    cond do
      entering == nil -> {:optimal, tab, basis}
      it > 50_000 -> raise "simplex: iteration limit"
      true ->
        cands = for {row, i} <- Enum.with_index(tab), qsign(elem(row, entering)) > 0, do: {qdiv(elem(row, tuple_size(row) - 1), elem(row, entering)), Enum.at(basis, i), i}
        case cands do
          [] -> {:unbounded, tab, basis, entering}
          _ ->
            {_, _, i} = Enum.min(cands, fn {r1, b1, _}, {r2, b2, _} -> s = qcmp(r1, r2); s < 0 or (s == 0 and b1 <= b2) end)
            run_simplex(pivot(tab, i, entering), List.replace_at(basis, i, entering), c, ncols, it + 1)
        end
    end
  end

  defp pivot(tab, i, j) do
    prow = Enum.at(tab, i)
    pv = elem(prow, j)
    prow = prow |> Tuple.to_list() |> Enum.map(&qdiv(&1, pv)) |> List.to_tuple()
    tab
    |> Enum.with_index()
    |> Enum.map(fn
      {_, ^i} -> prow
      {row, _} ->
        f = elem(row, j)
        if qzero?(f), do: row, else: row |> Tuple.to_list() |> Enum.zip(Tuple.to_list(prow)) |> Enum.map(fn {x, p} -> qsub(x, qmul(f, p)) end) |> List.to_tuple()
    end)
  end

  # the dual of max{cᵀx : a_i x op b_i, x ≥ 0}: min bᵀy s.t. Aᵀy ≥ c; y ≥ 0 (≤), y ≤ 0 (≥), free (=)
  defp dual(a, ops, b, c) do
    m = length(a)
    at = Enum.zip_with(a, & &1)
    # variables y split by sign: ≤ rows y = u, ≥ rows y = −u, = rows y = u − v
    ycols = Enum.flat_map(Enum.with_index(ops), fn {op, i} -> case op do :le -> [{i, 1}]; :ge -> [{i, -1}]; :eq -> [{i, 1}, {i, -1}] end end)
    rows = for col <- at, do: (for {i, s} <- ycols, do: qmul({s, 1}, Enum.at(col, i)))
    obj = for {i, s} <- ycols, do: qneg(qmul({s, 1}, Enum.at(b, i)))       # max −bᵀy
    case simplex_general(rows, List.duplicate(:ge, length(rows)), c, obj) do
      {:optimal, us, _} -> Enum.zip(ycols, us) |> Enum.reduce(List.duplicate({0, 1}, m), fn {{i, s}, u}, y -> List.update_at(y, i, &qadd(&1, qmul({s, 1}, u))) end)
      _ -> nil
    end
  end

  # Farkas: y with Aᵀy ≥ 0, sign by row, bᵀy = −1
  defp farkas(a, ops, b) do
    m = length(a)
    at = Enum.zip_with(a, & &1)
    ycols = Enum.flat_map(Enum.with_index(ops), fn {op, i} -> case op do :le -> [{i, 1}]; :ge -> [{i, -1}]; :eq -> [{i, 1}, {i, -1}] end end)
    rows = for col <- at, do: (for {i, s} <- ycols, do: qmul({s, 1}, Enum.at(col, i)))
    norm = for {i, s} <- ycols, do: qmul({s, 1}, Enum.at(b, i))
    case simplex_general(rows ++ [norm], List.duplicate(:ge, length(rows)) ++ [:eq], List.duplicate({0, 1}, length(rows)) ++ [{-1, 1}], List.duplicate({0, 1}, length(ycols))) do
      {:optimal, us, _} -> Enum.zip(ycols, us) |> Enum.reduce(List.duplicate({0, 1}, m), fn {{i, s}, u}, y -> List.update_at(y, i, &qadd(&1, qmul({s, 1}, u))) end)
      _ -> nil
    end
  end

  # an unbounded ray: d ≥ 0, a_i d (≤ 0 | ≥ 0 | = 0) by row, cᵀd = 1
  defp ray(a, ops, c) do
    rows = a ++ [c]
    rhs = List.duplicate({0, 1}, length(a)) ++ [{1, 1}]
    case simplex_general(rows, ops ++ [:eq], rhs, List.duplicate({0, 1}, length(c))) do
      {:optimal, d, _} -> d
      _ -> List.duplicate({0, 1}, length(c))
    end
  end

  # -------------------------------------------------------------- the check

  @doc """
  Check a certificate against a problem using only exact multiplication
  and comparison. `cert` is `%{status: :optimal, x, y}`,
  `%{status: :infeasible, y}` or `%{status: :unbounded, x, ray}`; x and the
  ray map variable names to rationals, y is a list (one per constraint).
  `%{accepted, reason}`.
  """
  def check(p, cert) do
    rows = p.rows
    row_val = fn coeffs, x -> Enum.reduce(coeffs, {0, 1}, fn {v, a}, s -> qadd(s, qmul(a, Map.get(x, v, {0, 1}))) end) end
    feasible = fn x ->
      cond do
        Enum.any?(p.vars, &(not Map.has_key?(x, &1))) -> {false, "x misses a variable"}
        Enum.any?(p.vars, &(&1 not in p.free and qsign(x[&1]) < 0)) -> {false, "a variable declared ≥ 0 is negative"}
        true ->
          case Enum.find(Enum.with_index(rows), fn {{co, op, rhs}, _} -> s = qcmp(row_val.(co, x), rhs); (op == :le and s > 0) or (op == :ge and s < 0) or (op == :eq and s != 0) end) do
            nil -> {true, "every constraint holds"}
            {_, i} -> {false, "constraint #{i + 1} is violated"}
          end
      end
    end
    sign_ok = fn y -> Enum.zip(rows, y) |> Enum.all?(fn {{_, op, _}, yi} -> case op do :le -> qsign(yi) >= 0; :ge -> qsign(yi) <= 0; :eq -> true end end) end
    aty = fn y, v -> Enum.zip(rows, y) |> Enum.reduce({0, 1}, fn {{co, _, _}, yi}, s -> qadd(s, qmul(Map.get(co, v, {0, 1}), yi)) end) end
    by = fn y -> Enum.zip(rows, y) |> Enum.reduce({0, 1}, fn {{_, _, r}, yi}, s -> qadd(s, qmul(r, yi)) end) end
    sgn = if p.sense == :max, do: 1, else: -1
    cx = fn x -> Enum.reduce(p.vars, {0, 1}, fn v, s -> qadd(s, qmul(Map.get(p.c, v, {0, 1}), Map.get(x, v, {0, 1}))) end) end

    case cert do
      %{status: :optimal, x: x, y: y} when is_list(y) and length(y) == length(rows) ->
        {ok, why} = feasible.(x)
        # dual feasibility in the max form: (Aᵀy)_v ≥ sgn·c_v for v ≥ 0, = for free v
        dual_ok = sign_ok.(y) and Enum.all?(p.vars, fn v ->
          s = qcmp(aty.(y, v), qmul({sgn, 1}, Map.get(p.c, v, {0, 1})))
          if v in p.free, do: s == 0, else: s >= 0
        end)
        gap = qsub(qmul({sgn, 1}, cx.(x)), by.(y))
        cond do
          not ok -> %{accepted: false, reason: "x is not feasible: " <> why}
          not dual_ok -> %{accepted: false, reason: "y is not dual-feasible"}
          not qzero?(gap) -> %{accepted: false, reason: "duality gap #{show(gap)} ≠ 0: x is not optimal (or y is not)"}
          true -> %{accepted: true, reason: "x feasible, y dual-feasible, cᵀx = bᵀy = #{show(qmul({sgn, 1}, by.(y)))}: optimal"}
        end
      %{status: :infeasible, y: y} when is_list(y) and length(y) == length(rows) ->
        ok = sign_ok.(y) and Enum.all?(p.vars, fn v -> s = qsign(aty.(y, v)); if v in p.free, do: s == 0, else: s >= 0 end) and qsign(by.(y)) < 0
        %{accepted: ok, reason: if(ok, do: "Farkas: Aᵀy ≥ 0 with the row signs and bᵀy = #{show(by.(y))} < 0 — no x exists", else: "not a Farkas certificate")}
      %{status: :unbounded, x: x, ray: d} ->
        {ok, why} = feasible.(x)
        ray_ok = Enum.all?(p.vars, &(&1 in p.free or qsign(Map.get(d, &1, {0, 1})) >= 0)) and
          Enum.all?(rows, fn {co, op, _} -> s = qsign(row_val.(co, d)); case op do :le -> s <= 0; :ge -> s >= 0; :eq -> s == 0 end end) and
          qsign(qmul({sgn, 1}, cx.(d))) > 0
        cond do
          not ok -> %{accepted: false, reason: "x is not feasible: " <> why}
          not ray_ok -> %{accepted: false, reason: "d is not an improving ray of the feasible set"}
          true -> %{accepted: true, reason: "x feasible and x + t·d stays feasible for every t ≥ 0 while the objective grows without bound"}
        end
      _ -> %{accepted: false, reason: "malformed certificate"}
    end
  end

  @doc "A JSON-ready view of a result (rationals as strings and floats)."
  def present({:ok, r}) do
    show_map = fn m -> Map.new(m, fn {k, v} -> {k, %{exact: show(v), value: to_float(v)}} end) end
    base = %{status: r.status, sense: r.sense, check: r.check}
    base = if r[:x], do: Map.put(base, :x, show_map.(r.x)), else: base
    base = if r[:objective], do: Map.put(base, :objective, %{exact: show(r.objective), value: to_float(r.objective)}), else: base
    cert = r.certificate
    base = if cert[:y], do: Map.put(base, :y, Enum.map(cert.y, &%{exact: show(&1), value: to_float(&1)})), else: base
    if cert[:ray], do: Map.put(base, :ray, show_map.(cert.ray)), else: base
  end
  def present(e), do: e
end
