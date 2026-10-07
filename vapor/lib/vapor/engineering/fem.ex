defmodule Vapor.Engineering.FEM do
  @moduledoc """
  Two-dimensional linear elasticity by finite elements
  (docs/ENGENHARIA.md §4): plane stress or plane strain, bilinear
  isoparametric quadrilaterals (Q4) with 2×2 Gauss quadrature — the
  element of every introductory FEM course and of many production
  codes' first pass on a plate, a bracket, a wall.

      plate x=0..10 y=0..1 nx=40 ny=4          # a structured mesh
      material E=200[GPa] nu=0.3 t=10[mm]       # plane stress (strain: add 'strain')
      fix x=0                                    # an edge, both directions (or: fix x=0 ux)
      traction x=10 ty=-1[MPa]                   # a traction on an edge (force per area)
      node 101 2.5 0.5 / quad 1 2 3 4            # or an explicit mesh

  Two elements: `element qm6` (the default) adds Wilson's incompatible
  bending modes in Taylor's form, condensed out per element — it does not
  lock in bending, so a cantilever two elements deep is within 1 % of the
  beam's deflection where plain `element q4` is 29 % stiff (the
  shear-locking every engineer meets in Q4 meshes, shown, not hidden).

  Both are checked by the **patch test** (Irons): on a patch of
  distorted elements, a linear displacement imposed on the boundary must
  be reproduced exactly inside, with exactly constant stress — the
  condition for convergence. `patch_test/0` runs it.

  Outputs: displacements, the deformed mesh, stresses (σx, σy, τxy and
  von Mises) averaged at the nodes, reactions, and the residual and
  equilibrium of the solve.
  """
  alias Vapor.{Dense, Expr}

  @g 1 / :math.sqrt(3)
  @gauss [{-@g, -@g}, {@g, -@g}, {@g, @g}, {-@g, @g}]

  @doc "Parse and solve. `{:ok, result}` or `{:error, why}`."
  def run(text) do
    with {:ok, m} <- parse(text), do: solve(m)
  end

  # ================================================================ parsing

  def parse(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.reduce_while({:ok, %{nodes: %{}, quads: [], e: nil, nu: 0.3, t: 1.0, strain: false, fix: [], tractions: [], loads: [], element: :qm6}}, fn {raw, n}, {:ok, acc} ->
      l = raw |> String.split("#", parts: 2) |> hd() |> String.trim()
      case stmt(String.split(l), acc) do
        {:ok, acc} -> {:cont, {:ok, acc}}
        {:error, w} -> {:halt, {:error, "line #{n}: #{w}"}}
      end
    end)
    |> case do
      {:ok, %{e: nil}} -> {:error, "material E=… nu=… is missing"}
      {:ok, %{quads: []}} -> {:error, "no elements (plate … or quad …)"}
      {:ok, %{fix: []}} -> {:error, "no supports (fix x=… or fix y=… or fix node id)"}
      other -> other
    end
  end

  defp stmt([], acc), do: {:ok, acc}

  defp stmt(["plate" | kvs], acc) do
    with {:ok, o} <- kv(kvs),
         {x0, x1} <- o["x"] || {:error, "plate needs x=a..b"},
         {y0, y1} <- o["y"] || {:error, "plate needs y=c..d"} do
      nx = trunc(o["nx"] || 10) |> max(1) |> min(200)
      ny = trunc(o["ny"] || 2) |> max(1) |> min(100)
      if (nx + 1) * (ny + 1) > 8000, do: {:error, "at most 8 000 nodes"}, else: (
        id = fn i, j -> "#{i}:#{j}" end
        nodes = for i <- 0..nx, j <- 0..ny, into: acc.nodes, do: {id.(i, j), {x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny}}
        quads = for i <- 0..(nx - 1), j <- 0..(ny - 1), do: [id.(i, j), id.(i + 1, j), id.(i + 1, j + 1), id.(i, j + 1)]
        {:ok, %{acc | nodes: nodes, quads: acc.quads ++ quads}})
    end
  end

  defp stmt(["node", id, x, y], acc), do: with({:ok, a} <- q(x), {:ok, b} <- q(y), do: {:ok, %{acc | nodes: Map.put(acc.nodes, id, {a, b})}})
  defp stmt(["quad", a, b, c, d], acc), do: if(Enum.all?([a, b, c, d], &Map.has_key?(acc.nodes, &1)), do: {:ok, %{acc | quads: acc.quads ++ [[a, b, c, d]]}}, else: {:error, "quad: undefined node"})

  defp stmt(["material" | kvs], acc) do
    with {:ok, o} <- kv(Enum.reject(kvs, &(&1 == "strain"))) do
      {:ok, %{acc | e: o["E"], nu: o["nu"] || 0.3, t: o["t"] || 1.0, strain: "strain" in kvs}}
    end
  end

  defp stmt(["element", kind], acc) when kind in ["q4", "qm6"], do: {:ok, %{acc | element: String.to_atom(kind)}}
  defp stmt(["fix", "node", id | dirs], acc), do: {:ok, %{acc | fix: acc.fix ++ [{:node, id, dirs(dirs)}]}}
  defp stmt(["fix", where | dirs], acc), do: with({:ok, sel} <- edge(where), do: {:ok, %{acc | fix: acc.fix ++ [{:edge, sel, dirs(dirs)}]}})
  defp stmt(["traction", where | kvs], acc), do: with({:ok, sel} <- edge(where), {:ok, o} <- kv(kvs), do: {:ok, %{acc | tractions: acc.tractions ++ [{sel, o["tx"] || 0.0, o["ty"] || 0.0}]}})
  defp stmt(["load", id | kvs], acc), do: with({:ok, o} <- kv(kvs), do: {:ok, %{acc | loads: acc.loads ++ [{id, o["fx"] || 0.0, o["fy"] || 0.0}]}})
  defp stmt([w | _], _), do: {:error, "not understood: #{w} (plate, node, quad, material, fix, traction, load)"}

  defp dirs([]), do: [0, 1]
  defp dirs(ds), do: Enum.flat_map(ds, fn "ux" -> [0]; "uy" -> [1]; "x" -> [0]; "y" -> [1]; _ -> [] end)

  defp edge(w) do
    case String.split(w, "=", parts: 2) do
      [axis, v] when axis in ["x", "y"] -> with({:ok, val} <- q(v), do: {:ok, {axis, val}})
      _ -> {:error, "an edge is x=value or y=value"}
    end
  end

  defp q(s) do
    with {:ok, t} <- Expr.parse(s), do: (if Expr.vars(t) == [], do: {:ok, Expr.eval(t)}, else: {:error, "a number expected: #{s}"})
  end

  defp kv(list) do
    Enum.reduce_while(list, {:ok, %{}}, fn item, {:ok, m} ->
      case String.split(item, "=", parts: 2) do
        [k, v] ->
          case String.split(v, "..", parts: 2) do
            [a, b] -> (with {:ok, x} <- q(a), {:ok, y} <- q(b) do {:cont, {:ok, Map.put(m, k, {x, y})}} else e -> {:halt, e} end)
            [_] -> (case q(v) do {:ok, x} -> {:cont, {:ok, Map.put(m, k, x)}}; e -> {:halt, e} end)
          end
        _ -> {:halt, {:error, "expected key=value: #{item}"}}
      end
    end)
  end

  # =============================================================== element

  defp d_matrix(m) do
    {e, nu} = {m.e, m.nu}
    if m.strain do
      c = e / ((1 + nu) * (1 - 2 * nu))
      [[c * (1 - nu), c * nu, 0.0], [c * nu, c * (1 - nu), 0.0], [0.0, 0.0, c * (1 - 2 * nu) / 2]]
    else
      c = e / (1 - nu * nu)
      [[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1 - nu) / 2]]
    end
  end

  # strain–displacement matrix B (3×8) and det J at (ξ, η)
  defp b_matrix(xy, xi, eta) do
    dn_dxi = [-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4]
    dn_deta = [-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4]
    xs = Enum.map(xy, &elem(&1, 0))
    ys = Enum.map(xy, &elem(&1, 1))
    j11 = Dense.dot(dn_dxi, xs); j12 = Dense.dot(dn_dxi, ys)
    j21 = Dense.dot(dn_deta, xs); j22 = Dense.dot(dn_deta, ys)
    det = j11 * j22 - j12 * j21
    if det <= 0, do: throw({:fem, "an element is inverted or degenerate (det J ≤ 0): list its nodes counter-clockwise"})
    dx = Enum.zip_with(dn_dxi, dn_deta, fn a, b -> (j22 * a - j12 * b) / det end)
    dy = Enum.zip_with(dn_dxi, dn_deta, fn a, b -> (-j21 * a + j11 * b) / det end)
    b = [Enum.flat_map(dx, &[&1, 0.0]), Enum.flat_map(dy, &[0.0, &1]), Enum.flat_map(Enum.zip(dx, dy), fn {a, c} -> [c, a] end)]
    {b, det}
  end

  defp k_elem(xy, d, t, :q4) do
    Enum.reduce(@gauss, List.duplicate(List.duplicate(0.0, 8), 8), fn {xi, eta}, k ->
      {b, det} = b_matrix(xy, xi, eta)
      kb = Dense.transpose(b) |> Dense.matmul(Dense.matmul(d, b))
      Enum.zip_with(k, kb, fn r1, r2 -> Enum.zip_with(r1, r2, &(&1 + &2 * det * t)) end)
    end)
  end

  # QM6 (Taylor, Beresford & Wilson 1976): Q4 plus the incompatible modes 1−ξ², 1−η²,
  # their strains scaled by det J₀/det J so that the patch test still passes; condensed out
  defp k_elem(xy, d, t, :qm6) do
    {kuu, kua, kaa} = qm6_blocks(xy, d, t)
    {:ok, kaa_inv} = Dense.inverse(kaa)
    corr = kua |> Dense.matmul(kaa_inv) |> Dense.matmul(Dense.transpose(kua))
    Enum.zip_with(kuu, corr, fn r1, r2 -> Enum.zip_with(r1, r2, &(&1 - &2)) end)
  end

  defp qm6_blocks(xy, d, t) do
    {j0inv, det0} = jac0(xy)
    z = fn r, c -> List.duplicate(List.duplicate(0.0, c), r) end
    Enum.reduce(@gauss, {z.(8, 8), z.(8, 4), z.(4, 4)}, fn {xi, eta}, {kuu, kua, kaa} ->
      {b, det} = b_matrix(xy, xi, eta)
      g = g_matrix(j0inv, det0 / det, xi, eta)
      add = fn m, a, bb -> Enum.zip_with(m, Dense.transpose(a) |> Dense.matmul(Dense.matmul(d, bb)), fn r1, r2 -> Enum.zip_with(r1, r2, &(&1 + &2 * det * t)) end) end
      {add.(kuu, b, b), add.(kua, b, g), add.(kaa, g, g)}
    end)
  end

  defp jac0(xy) do
    xs = Enum.map(xy, &elem(&1, 0)); ys = Enum.map(xy, &elem(&1, 1))
    dxi = [-0.25, 0.25, 0.25, -0.25]; deta = [-0.25, -0.25, 0.25, 0.25]
    {j11, j12, j21, j22} = {Dense.dot(dxi, xs), Dense.dot(dxi, ys), Dense.dot(deta, xs), Dense.dot(deta, ys)}
    det = j11 * j22 - j12 * j21
    {{j22 / det, -j12 / det, -j21 / det, j11 / det}, det}
  end

  # strains of the incompatible modes at (ξ, η)
  defp g_matrix({a11, a12, a21, a22}, scale, xi, eta) do
    # [d/dx; d/dy] = J₀⁻¹ [d/dξ; d/dη]; J⁻¹ maps (dξ, dη) derivatives with rows (a11 a12; a21 a22)
    d1 = {(a11 * (-2 * xi)) * scale, (a21 * (-2 * xi)) * scale}
    d2 = {(a12 * (-2 * eta)) * scale, (a22 * (-2 * eta)) * scale}
    [{px1, py1}, {px2, py2}] = [d1, d2]
    [[px1, 0.0, px2, 0.0], [0.0, py1, 0.0, py2], [py1, px1, py2, px2]]
  end

  # ================================================================= solve

  defp solve(m) do
    ids = m.nodes |> Map.keys() |> Enum.sort()
    idx = ids |> Enum.with_index() |> Map.new()
    d = d_matrix(m)
    xyof = fn q -> Enum.map(q, &m.nodes[&1]) end

    {kg, _} =
      Enum.reduce(m.quads, {%{}, 0}, fn q, {kg, k} ->
        ke = k_elem(xyof.(q), d, m.t, m.element)
        map = for n <- q, c <- 0..1, do: 2 * idx[n] + c
        kg = for {i, r} <- Enum.with_index(map), {j, c} <- Enum.with_index(map), reduce: kg do
          kg -> Map.update(kg, {i, j}, Enum.at(Enum.at(ke, r), c), &(&1 + Enum.at(Enum.at(ke, r), c)))
        end
        {kg, k + 1}
      end)

    f = tractions(m, idx)
    f = Enum.reduce(m.loads, f, fn {id, fx, fy}, f -> f |> Map.update(2 * idx[id], fx, &(&1 + fx)) |> Map.update(2 * idx[id] + 1, fy, &(&1 + fy)) end)
    fixed = fixed_dofs(m, idx)
    solve_system(m, ids, idx, kg, f, fixed, %{}, d)
  catch
    {:fem, w} -> {:error, w}
  end

  @doc false
  def solve_system(m, ids, idx, kg, f, fixed, prescribed, d) do
    ndof = 2 * length(ids)
    fixed = Enum.reduce(Map.keys(prescribed), fixed, &MapSet.put(&2, &1))
    free = Enum.reject(0..(ndof - 1), &MapSet.member?(fixed, &1))
    fidx = free |> Enum.with_index() |> Map.new()
    # move the prescribed displacements to the right side
    rhs = Enum.map(free, fn i -> Map.get(f, i, 0.0) - Enum.reduce(prescribed, 0.0, fn {j, v}, acc -> acc + Map.get(kg, {i, j}, 0.0) * v end) end)
    kff = for {{i, j}, v} <- kg, Map.has_key?(fidx, i), Map.has_key?(fidx, j), into: %{}, do: {{fidx[i], fidx[j]}, v}

    case Dense.sparse_spd_solve(kff, rhs) do
      {:ok, uf, info} ->
        u = Enum.reduce(prescribed, Map.new(Enum.zip(free, uf)), fn {j, v}, acc -> Map.put(acc, j, v) end)
        u = for i <- 0..(ndof - 1), do: Map.get(u, i, 0.0)
        ut = List.to_tuple(u)
        ku = Enum.reduce(kg, %{}, fn {{i, j}, v}, acc -> Map.update(acc, i, v * elem(ut, j), &(&1 + v * elem(ut, j))) end)
        resid = free |> Enum.map(&(Map.get(ku, &1, 0.0) - Map.get(f, &1, 0.0))) |> Dense.norm_inf()
        reac = for i <- MapSet.to_list(fixed), into: %{}, do: {i, Map.get(ku, i, 0.0) - Map.get(f, i, 0.0)}
        fsum = {Enum.reduce(f, 0.0, fn {i, v}, a -> if rem(i, 2) == 0, do: a + v, else: a end), Enum.reduce(f, 0.0, fn {i, v}, a -> if rem(i, 2) == 1, do: a + v, else: a end)}
        rsum = {Enum.reduce(reac, 0.0, fn {i, v}, a -> if rem(i, 2) == 0, do: a + v, else: a end), Enum.reduce(reac, 0.0, fn {i, v}, a -> if rem(i, 2) == 1, do: a + v, else: a end)}
        fscale = f |> Map.values() |> Enum.map(&abs/1) |> Enum.max(fn -> 1.0 end) |> max(1.0e-30)
        stress = nodal_stress(m, ids, idx, ut, d)

        {:ok, %{nodes: Map.new(ids, fn id -> {x, y} = m.nodes[id]; {id, %{x: x, y: y, ux: elem(ut, 2 * idx[id]), uy: elem(ut, 2 * idx[id] + 1)}} end),
                quads: m.quads, stress: stress, dofs: length(free), bandwidth: info.bandwidth,
                max_von_mises: stress |> Map.values() |> Enum.map(& &1.von_mises) |> Enum.max(fn -> 0.0 end),
                max_displacement: ids |> Enum.map(&:math.sqrt(elem(ut, 2 * idx[&1]) ** 2 + elem(ut, 2 * idx[&1] + 1) ** 2)) |> Enum.max(),
                certificate: %{residual: resid / fscale, sum_fx: elem(fsum, 0) + elem(rsum, 0), sum_fy: elem(fsum, 1) + elem(rsum, 1)}}}

      {:error, _} -> {:error, "the model is a mechanism: the supports do not prevent rigid motion"}
    end
  end

  defp fixed_dofs(m, idx) do
    Enum.reduce(m.fix, MapSet.new(), fn
      {:node, id, ds}, s -> Enum.reduce(ds, s, &MapSet.put(&2, 2 * idx[id] + &1))
      {:edge, {axis, v}, ds}, s ->
        k = if axis == "x", do: 0, else: 1
        on = for {id, xy} <- m.nodes, abs(elem(xy, k) - v) < 1.0e-9 * (1 + abs(v)), do: id
        if on == [], do: throw({:fem, "no node lies on #{axis}=#{v}"})
        for id <- on, dd <- ds, reduce: s, do: (s -> MapSet.put(s, 2 * idx[id] + dd))
    end)
  end

  # consistent nodal forces of a uniform traction on the element edges lying on a line
  defp tractions(m, idx) do
    Enum.reduce(m.tractions, %{}, fn {{axis, v}, tx, ty}, f ->
      k = if axis == "x", do: 0, else: 1
      on = fn id -> abs(elem(m.nodes[id], k) - v) < 1.0e-9 * (1 + abs(v)) end
      edges = for q <- m.quads, [a, b] <- Enum.chunk_every(q ++ [hd(q)], 2, 1, :discard), on.(a) and on.(b), do: {a, b}
      if edges == [], do: throw({:fem, "no element edge lies on #{axis}=#{v}"})
      Enum.reduce(edges, f, fn {a, b}, f ->
        {xa, ya} = m.nodes[a]; {xb, yb} = m.nodes[b]
        len = :math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
        [{a, 0.5}, {b, 0.5}] |> Enum.reduce(f, fn {n, w}, f ->
          f |> Map.update(2 * idx[n], tx * len * m.t * w, &(&1 + tx * len * m.t * w)) |> Map.update(2 * idx[n] + 1, ty * len * m.t * w, &(&1 + ty * len * m.t * w))
        end)
      end)
    end)
  end

  defp nodal_stress(m, _ids, idx, ut, d) do
    {sums, counts} =
      Enum.reduce(m.quads, {%{}, %{}}, fn q, {s, c} ->
        xy = Enum.map(q, &m.nodes[&1])
        ue = for n <- q, k <- 0..1, do: elem(ut, 2 * idx[n] + k)
        alpha = if Map.get(m, :element, :q4) == :qm6, do: qm6_alpha(xy, d, m.t, ue), else: nil
        # stresses at the element's corners, from the B matrix there (ξ, η = ±1)
        Enum.zip(q, [{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}])
        |> Enum.reduce({s, c}, fn {n, {xi, eta}}, {s, c} ->
          {b, det} = b_matrix(xy, xi, eta)
          eps = Dense.matvec(b, ue)
          eps = if alpha, do: (({j0inv, det0} = jac0(xy)); Enum.zip_with(eps, Dense.matvec(g_matrix(j0inv, det0 / det, xi, eta), alpha), &(&1 + &2))), else: eps
          sig = Dense.matvec(d, eps)
          {Map.update(s, n, sig, &Enum.zip_with(&1, sig, fn a, bb -> a + bb end)), Map.update(c, n, 1, &(&1 + 1))}
        end)
      end)

    Map.new(sums, fn {n, [sx, sy, txy]} ->
      k = counts[n]
      {sx, sy, txy} = {sx / k, sy / k, txy / k}
      {n, %{sx: sx, sy: sy, txy: txy, von_mises: :math.sqrt(sx * sx - sx * sy + sy * sy + 3 * txy * txy)}}
    end)
  end

  defp qm6_alpha(xy, d, t, ue) do
    {_, kua, kaa} = qm6_blocks(xy, d, t)
    {:ok, a} = Dense.solve(kaa, Dense.matvec(Dense.transpose(kua), ue))
    Enum.map(a, &(-&1))
  end

  # ============================================================ patch test

  @doc """
  Irons' patch test: five distorted quadrilaterals filling the unit
  square, the linear field u = 1e-3·(1 + 2x + 3y), v = 1e-3·(−2 + x − y)
  imposed on the four corners of the square, the inner four nodes free.
  Returns the largest displacement error at the inner nodes and the
  largest deviation from the exact constant stress.
  """
  def patch_test(kind \\ :q4) do
    nodes = %{"1" => {0.0, 0.0}, "2" => {1.0, 0.0}, "3" => {1.0, 1.0}, "4" => {0.0, 1.0},
              "5" => {0.2, 0.15}, "6" => {0.75, 0.25}, "7" => {0.8, 0.7}, "8" => {0.3, 0.8}}
    quads = [["1", "2", "6", "5"], ["2", "3", "7", "6"], ["3", "4", "8", "7"], ["4", "1", "5", "8"], ["5", "6", "7", "8"]]
    m = %{nodes: nodes, quads: quads, e: 1.0e3, nu: 0.25, t: 1.0, strain: false, fix: [], tractions: [], loads: [], element: kind}
    ue = fn {x, y} -> {1.0e-3 * (1 + 2 * x + 3 * y), 1.0e-3 * (-2 + x - y)} end
    ids = nodes |> Map.keys() |> Enum.sort()
    idx = ids |> Enum.with_index() |> Map.new()
    d = d_matrix(m)

    kg = Enum.reduce(quads, %{}, fn q, kg ->
      ke = k_elem(Enum.map(q, &nodes[&1]), d, 1.0, kind)
      map = for n <- q, c <- 0..1, do: 2 * idx[n] + c
      for {i, r} <- Enum.with_index(map), {j, c} <- Enum.with_index(map), reduce: kg do
        kg -> Map.update(kg, {i, j}, Enum.at(Enum.at(ke, r), c), &(&1 + Enum.at(Enum.at(ke, r), c)))
      end
    end)

    prescribed = for id <- ["1", "2", "3", "4"], {u, v} = ue.(nodes[id]), k <- [{0, u}, {1, v}], into: %{}, do: {2 * idx[id] + elem(k, 0), elem(k, 1)}
    {:ok, r} = solve_system(m, ids, idx, kg, %{}, MapSet.new(), prescribed, d)
    err = for id <- ["5", "6", "7", "8"], do: (({u, v} = ue.(nodes[id])); max(abs(r.nodes[id].ux - u), abs(r.nodes[id].uy - v)))
    # exact strain: εx = 2e-3, εy = −1e-3, γ = 3e-3 + 1e-3
    exact = Dense.matvec(d, [2.0e-3, -1.0e-3, 4.0e-3])
    serr = for {_, s} <- r.stress, do: Enum.zip([s.sx, s.sy, s.txy], exact) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()
    %{displacement_error: Enum.max(err), stress_error: Enum.max(serr), stress: exact}
  end
end
