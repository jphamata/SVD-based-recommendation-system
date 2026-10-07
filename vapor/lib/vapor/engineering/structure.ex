defmodule Vapor.Engineering.Structure do
  @moduledoc """
  Plane frames and trusses by the direct stiffness method, with natural
  frequencies (docs/ENGENHARIA.md §3) — the analysis behind every beam,
  portal frame, bridge truss and building storey a structural engineer
  checks.

      node 1 0 0
      node 2 6[m] 0
      node 3 6[m] 4[m]
      support 1 fixed            # fixed | pinned | roller-x | roller-y | x y rz
      support 3 pinned
      beam 1 2 E=200[GPa] A=5000[mm^2] I=8e7[mm^4] rho=7850[kg/m^3] n=8
      truss 2 3 E=200[GPa] A=1000[mm^2]
      load 2 fx=10[kN] fy=-20[kN] mz=0
      udl 1 2 w=-5[kN/m]         # along the member, global y, per metre
      modes 4

  Euler–Bernoulli elements with consistent loads are exact at the nodes
  for point and uniform loads; `n=` subdivides a member (for diagrams
  and for the frequencies, which converge with it). The answer:
  displacements, reactions, each member's axial force, shear and bending
  moment along it, the frequencies and mode shapes — and the
  **certificate**: global equilibrium (ΣFx, ΣFy, ΣM of loads and
  reactions) and the residual of K u = f.
  """
  alias Vapor.{Dense, Expr}

  # ================================================================ parsing

  @doc "Parse and analyse. `{:ok, result}` or `{:error, why}`."
  def run(text) do
    with {:ok, m} <- parse(text), do: analyse(m)
  end

  def parse(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.reduce_while({:ok, %{nodes: %{}, order: [], supports: %{}, members: [], loads: [], udls: [], modes: 0}}, fn {raw, ln}, {:ok, acc} ->
      l = raw |> String.split("#", parts: 2) |> hd() |> String.trim()
      case statement(String.split(l), acc) do
        {:ok, acc} -> {:cont, {:ok, acc}}
        {:error, w} -> {:halt, {:error, "line #{ln}: #{w}"}}
      end
    end)
    |> case do
      {:ok, %{members: []}} -> {:error, "no members (beam a b … or truss a b …)"}
      {:ok, %{supports: s}} when map_size(s) == 0 -> {:error, "no supports: the structure would fly away"}
      other -> other
    end
  end

  defp statement([], acc), do: {:ok, acc}

  defp statement(["node", id, x, y], acc) do
    with {:ok, xv} <- q(x), {:ok, yv} <- q(y), do: {:ok, %{acc | nodes: Map.put(acc.nodes, id, {xv, yv}), order: acc.order ++ [id]}}
  end

  defp statement(["support", id | kinds], acc) do
    dofs =
      Enum.flat_map(kinds, fn
        "fixed" -> [0, 1, 2]
        "engaste" -> [0, 1, 2]
        "pinned" -> [0, 1]
        "rotula" -> [0, 1]
        "roller-x" -> [1]
        "roller-y" -> [0]
        "x" -> [0]
        "y" -> [1]
        "rz" -> [2]
        other -> throw({:bad, "support kind #{other} (fixed, pinned, roller-x, roller-y, x, y, rz)"})
      end)
    need(acc, id, fn -> {:ok, %{acc | supports: Map.update(acc.supports, id, Enum.uniq(dofs), &Enum.uniq(&1 ++ dofs))}} end)
  catch
    {:bad, w} -> {:error, w}
  end

  defp statement([kind, a, b | kvs], acc) when kind in ["beam", "truss", "viga", "barra"] do
    with {:ok, o} <- props(kvs) do
      cond do
        not Map.has_key?(acc.nodes, a) or not Map.has_key?(acc.nodes, b) -> {:error, "node #{a} or #{b} is not defined"}
        a == b -> {:error, "a member joins two different nodes"}
        not Map.has_key?(o, "E") or not Map.has_key?(o, "A") -> {:error, "#{kind} needs E= and A="}
        kind in ["beam", "viga"] and not Map.has_key?(o, "I") -> {:error, "beam needs I="}
        true ->
          {:ok, %{acc | members: acc.members ++ [%{a: a, b: b, truss: kind in ["truss", "barra"], e: o["E"], area: o["A"], i: o["I"] || 0.0, rho: o["rho"] || 0.0, n: trunc(o["n"] || 1) |> max(1) |> min(200)}]}}
      end
    end
  end

  defp statement(["load", id | kvs], acc) do
    with {:ok, o} <- props(kvs), do: need(acc, id, fn -> {:ok, %{acc | loads: acc.loads ++ [{id, o["fx"] || 0.0, o["fy"] || 0.0, o["mz"] || 0.0}]}} end)
  end

  defp statement(["udl", a, b | kvs], acc) do
    with {:ok, o} <- props(kvs) do
      if Enum.any?(acc.members, &({&1.a, &1.b} in [{a, b}, {b, a}])), do: {:ok, %{acc | udls: acc.udls ++ [{a, b, o["w"] || 0.0}]}}, else: {:error, "no member between #{a} and #{b} (declare it before its udl)"}
    end
  end

  defp statement(["modes", k], acc), do: (case Integer.parse(k) do {v, ""} -> {:ok, %{acc | modes: min(max(v, 0), 20)}}; _ -> {:error, "modes n"} end)
  defp statement([w | _], _), do: {:error, "not understood: #{w} (node, support, beam, truss, load, udl, modes)"}

  defp need(acc, id, f), do: if(Map.has_key?(acc.nodes, id), do: f.(), else: {:error, "node #{id} is not defined"})

  defp q(s) do
    with {:ok, t} <- Expr.parse(s) do
      if Expr.vars(t) == [], do: {:ok, Expr.eval(t)}, else: {:error, "a number (with optional [unit]) expected: #{s}"}
    end
  end

  defp props(kvs) do
    Enum.reduce_while(kvs, {:ok, %{}}, fn kv, {:ok, m} ->
      case String.split(kv, "=", parts: 2) do
        [k, v] -> (case q(v) do {:ok, x} -> {:cont, {:ok, Map.put(m, k, x)}}; e -> {:halt, e} end)
        _ -> {:halt, {:error, "expected key=value: #{kv}"}}
      end
    end)
  end

  # ================================================================= model

  # members subdivided into elements; new interior nodes named a~b~k
  defp elements(m) do
    Enum.reduce(m.members, {m.nodes, m.order, []}, fn mem, {nodes, order, els} ->
      {xa, ya} = nodes[mem.a]
      {xb, yb} = nodes[mem.b]
      ids = [mem.a] ++ (for k <- 1..(mem.n - 1)//1, do: "#{mem.a}~#{mem.b}~#{k}") ++ [mem.b]
      nodes = Enum.reduce(Enum.with_index(ids), nodes, fn {id, k}, ns -> Map.put_new(ns, id, {xa + (xb - xa) * k / mem.n, ya + (yb - ya) * k / mem.n}) end)
      order = order ++ Enum.reject(ids, &(&1 in order))
      w = m.udls |> Enum.filter(fn {a, b, _} -> {a, b} in [{mem.a, mem.b}, {mem.b, mem.a}] end) |> Enum.map(&elem(&1, 2)) |> Enum.sum()
      new = for [p, r] <- Enum.chunk_every(ids, 2, 1, :discard), do: Map.merge(mem, %{a: p, b: r, w: w, member: "#{mem.a}-#{mem.b}"})
      {nodes, order, els ++ new}
    end)
  end

  defp geometry(nodes, e) do
    {xa, ya} = nodes[e.a]
    {xb, yb} = nodes[e.b]
    l = :math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
    {l, (xb - xa) / l, (yb - ya) / l}
  end

  defp k_local(e, l) do
    ea = e.e * e.area / l
    ei = if e.truss, do: 0.0, else: e.e * e.i
    [k1, k2, k3, k4] = [12 * ei / l ** 3, 6 * ei / l ** 2, 4 * ei / l, 2 * ei / l]
    [[ea, 0, 0, -ea, 0, 0], [0, k1, k2, 0, -k1, k2], [0, k2, k3, 0, -k2, k4],
     [-ea, 0, 0, ea, 0, 0], [0, -k1, -k2, 0, k1, -k2], [0, k2, k4, 0, -k2, k3]]
  end

  defp m_local(e, l) do
    m = e.rho * e.area * l
    if e.truss do
      # a bar's consistent mass moves in both directions
      f = m / 6
      [[2 * f, 0, 0, f, 0, 0], [0, 2 * f, 0, 0, f, 0], [0, 0, 0, 0, 0, 0], [f, 0, 0, 2 * f, 0, 0], [0, f, 0, 0, 2 * f, 0], [0, 0, 0, 0, 0, 0]]
    else
      c = m / 420
      [[140 * c, 0, 0, 70 * c, 0, 0], [0, 156 * c, 22 * l * c, 0, 54 * c, -13 * l * c], [0, 22 * l * c, 4 * l * l * c, 0, 13 * l * c, -3 * l * l * c],
       [70 * c, 0, 0, 140 * c, 0, 0], [0, 54 * c, 13 * l * c, 0, 156 * c, -22 * l * c], [0, -13 * l * c, -3 * l * l * c, 0, -22 * l * c, 4 * l * l * c]]
    end
  end

  defp t_mat(c, s) do
    r = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    z = [0, 0, 0]
    Enum.map(r, &(&1 ++ z)) ++ Enum.map(r, &(z ++ &1))
  end

  # equivalent nodal loads (local) of a uniform global-y load w per unit length
  defp p_eq(e, l, c, s) do
    {qa, qt} = {e.w * s, e.w * c}
    if e.truss, do: [qa * l / 2, qt * l / 2, 0.0, qa * l / 2, qt * l / 2, 0.0],
               else: [qa * l / 2, qt * l / 2, qt * l * l / 12, qa * l / 2, qt * l / 2, -qt * l * l / 12]
  end

  # ============================================================== analysis

  defp analyse(m) do
    {nodes, order, els} = elements(m)
    nidx = order |> Enum.with_index() |> Map.new()
    ndof = 3 * length(order)
    dof = fn id, k -> 3 * nidx[id] + k end

    {kg, mg, f} =
      Enum.reduce(els, {%{}, %{}, %{}}, fn e, {kg, mg, f} ->
        {l, c, s} = geometry(nodes, e)
        t = t_mat(c, s)
        tt = Dense.transpose(t)
        ke = tt |> Dense.matmul(k_local(e, l)) |> Dense.matmul(t)
        me = tt |> Dense.matmul(m_local(e, l)) |> Dense.matmul(t)
        pe = Dense.matvec(tt, p_eq(e, l, c, s))
        map = for nd <- [e.a, e.b], k <- 0..2, do: dof.(nd, k)
        kg = scatter(kg, map, ke)
        mg = scatter(mg, map, me)
        f = Enum.zip(map, pe) |> Enum.reduce(f, fn {d, v}, f -> Map.update(f, d, v, &(&1 + v)) end)
        {kg, mg, f}
      end)

    f = Enum.reduce(m.loads, f, fn {id, fx, fy, mz}, f ->
      [{dof.(id, 0), fx}, {dof.(id, 1), fy}, {dof.(id, 2), mz}] |> Enum.reduce(f, fn {d, v}, f -> Map.update(f, d, v, &(&1 + v)) end)
    end)

    fixed = for {id, ds} <- m.supports, d <- ds, into: MapSet.new(), do: dof.(id, d)
    # a rotation nothing resists (a node of trusses only) is restrained, and said so
    free_rot = for id <- order, d = dof.(id, 2), not MapSet.member?(fixed, d), abs(Map.get(kg, {d, d}, 0.0)) < 1.0e-9, do: d
    fixed = Enum.reduce(free_rot, fixed, &MapSet.put(&2, &1))
    free = Enum.reject(0..(ndof - 1), &MapSet.member?(fixed, &1))
    fidx = free |> Enum.with_index() |> Map.new()
    kff = for {{i, j}, v} <- kg, Map.has_key?(fidx, i), Map.has_key?(fidx, j), into: %{}, do: {{fidx[i], fidx[j]}, v}
    ff = Enum.map(free, &Map.get(f, &1, 0.0))

    case Dense.sparse_spd_solve(kff, ff) do
      {:ok, uf, info} ->
        u = Enum.reduce(Enum.zip(free, uf), List.duplicate(0.0, ndof), fn {d, v}, u -> List.replace_at(u, d, v) end)
        ut = List.to_tuple(u)
        ku = for i <- 0..(ndof - 1), do: Enum.reduce(0..(ndof - 1), 0.0, fn j, acc -> (v = Map.get(kg, {i, j}); if v, do: acc + v * elem(ut, j), else: acc) end)
        reactions = for d <- Enum.sort(MapSet.to_list(fixed)), into: %{}, do: {d, Enum.at(ku, d) - Map.get(f, d, 0.0)}
        resid = Enum.map(free, fn d -> Enum.at(ku, d) - Map.get(f, d, 0.0) end) |> Dense.norm_inf()
        fscale = f |> Map.values() |> Enum.map(&abs/1) |> Enum.max(fn -> 1.0 end) |> max(1.0e-30)
        members = member_forces(els, nodes, ut, dof)
        eq = equilibrium(m, nodes, order, f, reactions, dof, els)

        result = %{
          nodes: Map.new(order, fn id -> {id, %{x: elem(nodes[id], 0), y: elem(nodes[id], 1), ux: elem(ut, dof.(id, 0)), uy: elem(ut, dof.(id, 1)), rz: elem(ut, dof.(id, 2))}} end),
          order: order, elements: Enum.map(els, &%{a: &1.a, b: &1.b, member: &1.member, truss: &1.truss}),
          reactions: for(id <- order, Enum.any?(0..2, &MapSet.member?(fixed, dof.(id, &1))), (r = for(k <- 0..2, do: Map.get(reactions, dof.(id, k)))), Enum.any?(Enum.take(r, 3), &(&1 != nil)), into: %{},
                         do: {id, %{fx: Enum.at(r, 0), fy: Enum.at(r, 1), mz: Enum.at(r, 2)}}) |> Map.reject(fn {id, _} -> String.contains?(id, "~") and not Map.has_key?(m.supports, id) end),
          members: members, dofs: length(free), bandwidth: info.bandwidth, auto_restrained_rotations: length(free_rot),
          max_displacement: order |> Enum.map(&:math.sqrt(elem(ut, dof.(&1, 0)) ** 2 + elem(ut, dof.(&1, 1)) ** 2)) |> Enum.max(),
          certificate: Map.merge(eq, %{residual: resid / fscale})
        }

        {:ok, if(m.modes > 0, do: Map.put(result, :modes, modes(kg, mg, free, fidx, m.modes, order, dof)), else: result)}

      {:error, {:not_positive_definite, _}} ->
        {:error, "the structure is a mechanism (unstable): add supports or members — a free node, a hinge chain or an unrestrained direction"}
    end
  end

  defp scatter(g, map, ke) do
    for {i, r} <- Enum.with_index(map), {j, c} <- Enum.with_index(map), reduce: g do
      g -> (v = Enum.at(Enum.at(ke, r), c) * 1.0; if v == 0.0, do: g, else: Map.update(g, {i, j}, v, &(&1 + v)))
    end
  end

  # end forces per element (local N, V, M) and diagrams along each element
  defp member_forces(els, nodes, ut, dof) do
    for e <- els do
      {l, c, s} = geometry(nodes, e)
      ue = for nd <- [e.a, e.b], k <- 0..2, do: elem(ut, dof.(nd, k))
      fl = Dense.sub(Dense.matvec(k_local(e, l), Dense.matvec(t_mat(c, s), ue)), p_eq(e, l, c, s))
      [n1, v1, m1 | _] = fl
      {qa, qt} = {e.w * s, e.w * c}
      xs = for k <- 0..8, do: l * k / 8
      %{a: e.a, b: e.b, member: e.member, length: l, truss: e.truss,
        axial: Enum.map(xs, &(-n1 - qa * &1)), shear: Enum.map(xs, &(v1 + qt * &1)),
        moment: Enum.map(xs, &(-m1 + v1 * &1 + qt * &1 * &1 / 2)), x: xs, end_forces: fl}
    end
  end

  # ΣFx, ΣFy and ΣM about the origin of the applied loads (nodal + distributed) and the reactions
  defp equilibrium(m, nodes, _order, _f, reactions, dof, els) do
    rx = for {id, _} <- nodes, d = dof.(id, 0), Map.has_key?(reactions, d), reduce: 0.0, do: (acc -> acc + reactions[d])
    ry = for {id, _} <- nodes, d = dof.(id, 1), Map.has_key?(reactions, d), reduce: 0.0, do: (acc -> acc + reactions[d])
    rm = for {id, {x, y}} <- nodes, reduce: 0.0 do
      acc -> acc + x * Map.get(reactions, dof.(id, 1), 0.0) - y * Map.get(reactions, dof.(id, 0), 0.0) + Map.get(reactions, dof.(id, 2), 0.0)
    end
    {lx, ly, lm} = Enum.reduce(m.loads, {0.0, 0.0, 0.0}, fn {id, fx, fy, mz}, {a, b, c} -> {x, y} = nodes[id]; {a + fx, b + fy, c + x * fy - y * fx + mz} end)
    {dy, dm} = Enum.reduce(els, {0.0, 0.0}, fn e, {a, b} ->
      {l, _, _} = geometry(nodes, e)
      {xa, _} = nodes[e.a]; {xb, _} = nodes[e.b]
      {a + e.w * l, b + e.w * l * (xa + xb) / 2}
    end)
    scale = Enum.max([abs(lx), abs(ly + dy), 1.0e-30, rx |> abs(), ry |> abs()])
    %{sum_fx: rx + lx, sum_fy: ry + ly + dy, sum_m: rm + lm + dm, relative: Enum.max([abs(rx + lx), abs(ry + ly + dy)]) / scale}
  end

  defp modes(kg, mg, free, fidx, k, order, dof) do
    n = length(free)
    km = for i <- free, do: for(j <- free, do: Map.get(kg, {i, j}, 0.0))
    mm = for i <- free, do: for(j <- free, do: Map.get(mg, {i, j}, 0.0))
    # rotational DOFs of trusses carry no mass: give them a vanishing one (they condense out at ω → ∞)
    mm = for {r, i} <- Enum.with_index(mm), do: List.update_at(r, i, &(if &1 == 0.0, do: 1.0e-12 * (Enum.at(Enum.at(km, i), i) + 1.0), else: &1))

    if n > 400 do
      %{error: "#{n} degrees of freedom: modal analysis is limited to 400 here"}
    else
      case Dense.geigh(km, mm) do
        {:ok, {vals, vecs}} ->
          for {lam, v} <- Enum.zip(vals, vecs) |> Enum.take(k) do
            shape = Map.new(order, fn id -> {id, Enum.map(0..2, fn kk -> (d = dof.(id, kk); if i = fidx[d], do: Enum.at(v, i), else: 0.0) end)} end)
            %{omega: :math.sqrt(max(lam, 0.0)), hz: :math.sqrt(max(lam, 0.0)) / (2 * :math.pi()), shape: shape}
          end
        {:error, _} -> %{error: "the mass matrix is not positive definite (give rho= to the members)"}
      end
    end
  end
end
