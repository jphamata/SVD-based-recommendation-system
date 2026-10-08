defmodule Vapor.Sketch do
  @moduledoc """
  From a sketch to a drawing an engineer or an architect can use
  (docs/SCENE.md §5):

    * **vectorise** (`vectorize/2`) — the strokes are thinned to a skeleton
      graph; each chain is fitted by a line (total least squares), a circle
      or arc (Kåsa's algebraic fit) or split into lines (Ramer–Douglas–
      Peucker); then **beautified by constraints**: endpoints that nearly
      meet are welded, near-horizontal/vertical/45° lines are snapped,
      near-parallel ones made parallel, and every welded corner re-solved
      as the intersection of its lines. The constraints found are listed —
      what the drawing *meant*, stated. Exports: SVG and DXF (R12 ASCII,
      which every CAD program opens);
    * **floor plan → 3D** (`plan/2`) — the snapped lines are walls; a gap
      between collinear walls is a door; the rooms are the faces of the
      planar wall graph (doors closed), with their areas at the given
      scale; the walls are extruded into a mesh (`Vapor.Geom`), exported
      as glTF (GLB) for any 3D viewer, and walked through in the console.

  What it is not: a photorealistic render of the sketch. That needs a
  generative model's weights (`Vapor.Diffusion` runs Stable Diffusion's
  img2img with a checkpoint the user loads; none ships, so none is
  measured here).
  """
  alias Vapor.Geom.Mesh
  alias Vapor.Modal.Image
  alias Vapor.Scene

  # ============================================================ vectorise

  @doc """
  Vectorise a sketch: `%{w, h, lines: [%{x0, y0, x1, y1}], circles: [%{cx,
  cy, r}], arcs: [%{cx, cy, r, a0, a1}], constraints: [...]}`. Options:
  `snap: false` (the control: fitted primitives, no beautification),
  `max` (working size, 640).
  """
  def vectorize(%Image{} = img, opts \\ []) do
    full = Scene.fit(img, Keyword.get(opts, :max, 640))
    {w, h} = {full.w, full.h}
    g = Vapor.Vision.Segment.gray(full)
    ink = Vapor.Vision.Segment.ink(g)
    mask = List.to_tuple(for y <- 0..(h - 1), x <- 0..(w - 1), do: elem(elem(ink, y), x) == 1)
    thick = Scene.dilate_mask(w, h, mask, 1)
    {nodes, chains} = Scene.skeleton(w, h, thick)
    # the stroke's width: ink pixels per skeleton pixel (thinning shortens each free end by about half of it)
    skel_len = chains |> Enum.map(&length(&1.px)) |> Enum.sum() |> max(1)
    stroke = Enum.count(Tuple.to_list(mask), & &1) / skel_len
    size = max(w, h)
    tol = Keyword.get(opts, :tolerance, max(size * 0.006, 2.0))

    prims =
      chains
      |> Enum.map(fn c -> pts(c, nodes) end)
      |> Enum.reject(fn ps -> length(ps) < 4 and path_len(ps) < size * 0.02 end)
      |> Enum.flat_map(&fit(&1, tol))

    lines = for {:line, l} <- prims, do: l
    circles = for {:circle, c} <- prims, do: c
    arcs = for {:arc, a} <- prims, do: a

    if Keyword.get(opts, :snap, true) do
      {lines, cons} = beautify(lines, size)
      %{w: w, h: h, lines: lines, circles: circles, arcs: arcs, constraints: cons, stroke: stroke}
    else
      %{w: w, h: h, lines: lines, circles: circles, arcs: arcs, constraints: [], stroke: stroke}
    end
  end

  defp pts(%{from: f, to: t, px: px}, nodes) do
    xy = fn k -> {x, y, _} = nodes[k]; {x * 1.0, y * 1.0} end
    mid = Enum.map(px, fn {x, y} -> {x * 1.0, y * 1.0} end)
    [xy.(f) | mid] ++ if(t, do: [xy.(t)], else: [])
  end

  defp path_len(ps), do: ps |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [{a, b}, {c, d}] -> :math.sqrt((a - c) ** 2 + (b - d) ** 2) end) |> Enum.sum()

  # a chain → one line, one circle/arc, or several lines
  defp fit(ps, tol) do
    {a, b} = {hd(ps), List.last(ps)}
    closed = dist(a, b) < tol * 2 and length(ps) > 12

    case line_fit(ps) do
      {:ok, dev} when dev <= tol and not closed ->
        [{:line, endpoints(ps)}]

      _ ->
        case circle_fit(ps) do
          {cx, cy, r, res} when res <= tol * 0.8 and r > tol * 3 ->
            sweep = sweep(ps, cx, cy)
            cond do
              closed or sweep > 2 * :math.pi() * 0.92 -> [{:circle, %{cx: cx, cy: cy, r: r}}]
              sweep > :math.pi() / 5 -> [{:arc, %{cx: cx, cy: cy, r: r, a0: angle(a, cx, cy), a1: angle(b, cx, cy), sweep: sweep}}]
              true -> split(ps, tol)
            end

          _ ->
            split(ps, tol)
        end
    end
  end

  defp split(ps, tol) do
    ps |> Scene.rdp(tol * 1.5) |> Enum.chunk_every(2, 1, :discard) |> Enum.reject(fn [p, q] -> dist(p, q) < tol end)
    |> Enum.map(fn [{x0, y0}, {x1, y1}] -> {:line, %{x0: x0, y0: y0, x1: x1, y1: y1}} end)
  end

  defp dist({a, b}, {c, d}), do: :math.sqrt((a - c) ** 2 + (b - d) ** 2)
  defp angle({x, y}, cx, cy), do: :math.atan2(y - cy, x - cx)

  # total angle swept along the points (unwrapped)
  defp sweep(ps, cx, cy) do
    ps |> Enum.map(&angle(&1, cx, cy)) |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [a, b] -> d = b - a; d - 2 * :math.pi() * Float.round(d / (2 * :math.pi())) end) |> Enum.sum() |> abs()
  end

  # total least squares: the principal axis; returns the max distance to it
  defp line_fit(ps) do
    n = length(ps)
    {mx, my} = {Enum.sum(Enum.map(ps, &elem(&1, 0))) / n, Enum.sum(Enum.map(ps, &elem(&1, 1))) / n}
    {sxx, sxy, syy} = Enum.reduce(ps, {0.0, 0.0, 0.0}, fn {x, y}, {a, b, c} -> {a + (x - mx) ** 2, b + (x - mx) * (y - my), c + (y - my) ** 2} end)
    th = 0.5 * :math.atan2(2 * sxy, sxx - syy)
    {dx, dy} = {:math.cos(th), :math.sin(th)}
    {:ok, ps |> Enum.map(fn {x, y} -> abs(-(x - mx) * dy + (y - my) * dx) end) |> Enum.max()}
  end

  # endpoints projected on the fitted axis
  defp endpoints(ps) do
    n = length(ps)
    {mx, my} = {Enum.sum(Enum.map(ps, &elem(&1, 0))) / n, Enum.sum(Enum.map(ps, &elem(&1, 1))) / n}
    {sxx, sxy, syy} = Enum.reduce(ps, {0.0, 0.0, 0.0}, fn {x, y}, {a, b, c} -> {a + (x - mx) ** 2, b + (x - mx) * (y - my), c + (y - my) ** 2} end)
    th = 0.5 * :math.atan2(2 * sxy, sxx - syy)
    {dx, dy} = {:math.cos(th), :math.sin(th)}
    proj = fn {x, y} -> t = (x - mx) * dx + (y - my) * dy; {mx + t * dx, my + t * dy} end
    {{x0, y0}, {x1, y1}} = {proj.(hd(ps)), proj.(List.last(ps))}
    %{x0: x0, y0: y0, x1: x1, y1: y1}
  end

  # Kåsa: x² + y² + D x + E y + F = 0 by least squares
  defp circle_fit(ps) when length(ps) < 6, do: nil

  defp circle_fit(ps) do
    rows = Enum.map(ps, fn {x, y} -> {[x, y, 1.0], -(x * x + y * y)} end)
    ata = for i <- 0..2, do: (for j <- 0..2, do: Enum.reduce(rows, 0.0, fn {r, _}, s -> s + Enum.at(r, i) * Enum.at(r, j) end))
    atb = for i <- 0..2, do: Enum.reduce(rows, 0.0, fn {r, b}, s -> s + Enum.at(r, i) * b end)

    case solve3(ata, atb) do
      [d, e, f] ->
        {cx, cy} = {-d / 2, -e / 2}
        r2 = cx * cx + cy * cy - f
        if r2 <= 0, do: nil, else: (r = :math.sqrt(r2); {cx, cy, r, ps |> Enum.map(fn p -> abs(dist(p, {cx, cy}) - r) end) |> Enum.max()})

      _ ->
        nil
    end
  end

  defp solve3(a, b) do
    det = fn [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]] -> a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1) end
    d = det.(a)
    if abs(d) < 1.0e-9, do: nil, else: (for k <- 0..2, do: det.(Enum.map(Enum.zip(a, b), fn {row, bv} -> List.replace_at(row, k, bv) end)) / d)
  end

  # ---------------------------------------------------------- beautify

  @snap_deg 4.0

  defp beautify(lines, size) do
    weld = max(size * 0.02, 6.0)
    # 1. orientation snapping (0, 45, 90, 135°) and parallel groups
    {lines, cons} =
      Enum.map_reduce(Enum.with_index(lines), [], fn {l, i}, cons ->
        a = deg(l)
        target = Enum.find([0.0, 45.0, 90.0, 135.0, 180.0], fn t -> abs(a - t) < @snap_deg end)
        if target, do: {rotate_to(l, rem(trunc(target), 180) * 1.0), [%{kind: kind_of(target), line: i} | cons]}, else: {l, cons}
      end)

    free = for {l, i} <- Enum.with_index(lines), Enum.find(cons, &(&1.line == i)) == nil, do: {l, i}
    {lines, cons} =
      Enum.reduce(free, {lines, cons}, fn {l, i}, {ls, cs} ->
        case Enum.find(Enum.with_index(ls), fn {m, j} -> j < i and abs(ang_diff(deg(l), deg(m))) < 3.0 end) do
          {m, j} -> {List.replace_at(ls, i, rotate_to(l, deg(m))), [%{kind: "parallel", lines: [j, i]} | cs]}
          nil -> {ls, cs}
        end
      end)

    # 1b. collinear alignment: horizontal (vertical) lines within a few pixels of one another share one y (x)
    lines = align(lines, :h, weld / 2) |> align(:v, weld / 2)

    # 2. weld endpoints: cluster ends within `weld`; each cluster's point = least-squares intersection of its lines
    ends = for {l, i} <- Enum.with_index(lines), {e, p} <- [{0, {l.x0, l.y0}}, {1, {l.x1, l.y1}}], do: {i, e, p}
    clusters = cluster(ends, weld)

    lines =
      Enum.reduce(clusters, lines, fn members, ls ->
        if length(members) < 2, do: ls, else: (
          p = intersect_ls(Enum.map(members, fn {i, _, _} -> Enum.at(ls, i) end), members)
          Enum.reduce(members, ls, fn {i, e, _}, ls -> List.update_at(ls, i, fn l -> if e == 0, do: %{l | x0: elem(p, 0), y0: elem(p, 1)}, else: %{l | x1: elem(p, 0), y1: elem(p, 1)} end) end))
      end)

    lines = Enum.reject(lines, fn l -> :math.sqrt((l.x1 - l.x0) ** 2 + (l.y1 - l.y0) ** 2) < weld end)
    corners = Enum.count(clusters, &(length(&1) >= 2))
    perp = for {a, i} <- Enum.with_index(lines), {b, j} <- Enum.with_index(lines), i < j, shares_end?(a, b), abs(abs(ang_diff(deg(a), deg(b))) - 90) < 0.5, do: %{kind: "perpendicular", lines: [i, j]}
    {lines, Enum.reverse(cons) ++ perp ++ [%{kind: "welded_corners", count: corners}]}
  end

  defp align(lines, dir, tol) do
    hv? = fn l -> if dir == :h, do: abs(l.y1 - l.y0) < 1.0e-6, else: abs(l.x1 - l.x0) < 1.0e-6 end
    off = fn l -> if dir == :h, do: l.y0, else: l.x0 end
    idx = for {l, i} <- Enum.with_index(lines), hv?.(l), do: {off.(l), i, :math.sqrt((l.x1 - l.x0) ** 2 + (l.y1 - l.y0) ** 2) + 1.0e-6}

    groups =
      idx |> Enum.sort() |> Enum.chunk_while([], fn {o, _, _} = e, acc ->
        case acc do
          [] -> {:cont, [e]}
          [{o2, _, _} | _] when o - o2 <= tol -> {:cont, [e | acc]}
          _ -> {:cont, Enum.reverse(acc), [e]}
        end
      end, fn acc -> {:cont, Enum.reverse(acc), []} end)

    Enum.reduce(Enum.reject(groups, &(&1 == [])), lines, fn g, ls ->
      mean = Enum.sum(for {o, _, w} <- g, do: o * w) / Enum.sum(for {_, _, w} <- g, do: w)
      Enum.reduce(g, ls, fn {_, i, _}, ls -> List.update_at(ls, i, fn l -> if dir == :h, do: %{l | y0: mean, y1: mean}, else: %{l | x0: mean, x1: mean} end) end)
    end)
  end

  defp kind_of(t) when t in [0.0, 180.0], do: "horizontal"
  defp kind_of(90.0), do: "vertical"
  defp kind_of(_), do: "diagonal_45"

  defp deg(l), do: (a = :math.atan2(l.y1 - l.y0, l.x1 - l.x0) * 180 / :math.pi(); if(a < 0, do: a + 180, else: a))
  defp ang_diff(a, b), do: (d = a - b; d - 180 * Float.round(d / 180))

  defp rotate_to(l, target) do
    {mx, my} = {(l.x0 + l.x1) / 2, (l.y0 + l.y1) / 2}
    half = :math.sqrt((l.x1 - l.x0) ** 2 + (l.y1 - l.y0) ** 2) / 2
    t = target * :math.pi() / 180
    {dx, dy} = {:math.cos(t) * half, :math.sin(t) * half}
    # keep the original direction of travel
    sign = if (l.x1 - l.x0) * dx + (l.y1 - l.y0) * dy >= 0, do: 1, else: -1
    %{l | x0: mx - sign * dx, y0: my - sign * dy, x1: mx + sign * dx, y1: my + sign * dy}
  end

  defp shares_end?(a, b), do: Enum.any?(for p <- [{a.x0, a.y0}, {a.x1, a.y1}], q <- [{b.x0, b.y0}, {b.x1, b.y1}], do: dist(p, q) < 1.0e-6)

  defp cluster(ends, r) do
    Enum.reduce(ends, [], fn {_, _, p} = e, cl ->
      case Enum.find_index(cl, fn members -> Enum.any?(members, fn {_, _, q} -> dist(p, q) < r end) end) do
        nil -> cl ++ [[e]]
        k -> List.update_at(cl, k, &(&1 ++ [e]))
      end
    end)
  end

  # the point nearest (least squares) to all the lines through a corner; parallel lines: the mean of the ends
  defp intersect_ls(ls, members) do
    rows = for l <- ls, dx = l.x1 - l.x0, dy = l.y1 - l.y0, n = :math.sqrt(dx * dx + dy * dy), n > 1.0e-9, do: {-dy / n, dx / n, (-dy * l.x0 + dx * l.y0) / n}
    {a11, a12, a22, b1, b2} = Enum.reduce(rows, {0.0, 0.0, 0.0, 0.0, 0.0}, fn {a, b, c}, {s11, s12, s22, t1, t2} -> {s11 + a * a, s12 + a * b, s22 + b * b, t1 + a * c, t2 + b * c} end)
    d = a11 * a22 - a12 * a12
    if abs(d) < 1.0e-6 do
      {Enum.sum(for {_, _, {x, _}} <- members, do: x) / length(members), Enum.sum(for {_, _, {_, y}} <- members, do: y) / length(members)}
    else
      {(b1 * a22 - b2 * a12) / d, (a11 * b2 - a12 * b1) / d}
    end
  end

  # ============================================================== exports

  @doc "SVG of a vectorised drawing (strokes in currentColor)."
  def svg(%{w: w, h: h} = v) do
    f = &:erlang.float_to_binary(&1 * 1.0, decimals: 2)
    lines = Enum.map_join(v.lines, "\n", fn l -> ~s(  <line x1="#{f.(l.x0)}" y1="#{f.(l.y0)}" x2="#{f.(l.x1)}" y2="#{f.(l.y1)}"/>) end)
    circles = Enum.map_join(v.circles, "\n", fn c -> ~s(  <circle cx="#{f.(c.cx)}" cy="#{f.(c.cy)}" r="#{f.(c.r)}"/>) end)
    arcs = Enum.map_join(v.arcs, "\n", fn a ->
      {x0, y0, x1, y1} = {a.cx + a.r * :math.cos(a.a0), a.cy + a.r * :math.sin(a.a0), a.cx + a.r * :math.cos(a.a1), a.cy + a.r * :math.sin(a.a1)}
      large = if a.sweep > :math.pi(), do: 1, else: 0
      sw = if ccw?(a), do: 1, else: 0
      ~s(  <path d="M #{f.(x0)} #{f.(y0)} A #{f.(a.r)} #{f.(a.r)} 0 #{large} #{sw} #{f.(x1)} #{f.(y1)}"/>)
    end)

    """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 #{w} #{h}" width="#{w}" height="#{h}">
    <g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    #{lines}
    #{circles}
    #{arcs}
    </g></svg>
    """
  end

  defp ccw?(a), do: (d = a.a1 - a.a0; d = d - 2 * :math.pi() * Float.floor(d / (2 * :math.pi())); d <= :math.pi() == (a.sweep <= :math.pi()))

  @doc "DXF (R12 ASCII) of a vectorised drawing; y flipped (CAD's y points up); `scale` units per pixel."
  def dxf(%{h: h} = v, scale \\ 1.0) do
    n = fn x -> :erlang.float_to_binary(x * scale * 1.0, decimals: 4) end
    yy = fn y -> n.(h - y) end
    deg = fn a -> :erlang.float_to_binary(-a * 180 / :math.pi() * 1.0, decimals: 4) end

    ents =
      Enum.map(v.lines, fn l -> "0\nLINE\n8\n0\n10\n#{n.(l.x0)}\n20\n#{yy.(l.y0)}\n30\n0.0\n11\n#{n.(l.x1)}\n21\n#{yy.(l.y1)}\n31\n0.0\n" end) ++
        Enum.map(v.circles, fn c -> "0\nCIRCLE\n8\n0\n10\n#{n.(c.cx)}\n20\n#{yy.(c.cy)}\n30\n0.0\n40\n#{n.(c.r)}\n" end) ++
        Enum.map(v.arcs, fn a -> "0\nARC\n8\n0\n10\n#{n.(a.cx)}\n20\n#{yy.(a.cy)}\n30\n0.0\n40\n#{n.(a.r)}\n50\n#{deg.(a.a1)}\n51\n#{deg.(a.a0)}\n" end)

    "0\nSECTION\n2\nENTITIES\n" <> Enum.join(ents) <> "0\nENDSEC\n0\nEOF\n"
  end

  # ============================================================ floor plan

  @doc """
  A floor plan from a sketch: `%{walls, doors, rooms: [%{polygon, area}],
  mesh, glb (base64), scale}`. `scale:` metres per pixel (default: the
  longest wall is taken as `longest:` metres, 8 by default); `height:`
  wall height (2.7 m); `door:` the gap range that counts as a door,
  in metres (0.6–1.4).
  """
  def plan(%Image{} = img, opts \\ []) do
    v = vectorize(img, Keyword.put_new(opts, :max, 640))
    walls = Enum.filter(v.lines, fn l -> len(l) > 8 end)
    longest = runs(walls) |> Enum.max(fn -> 1.0 end)
    scale = Keyword.get_lazy(opts, :scale, fn -> Keyword.get(opts, :longest, 8.0) / longest end)
    {dmin, dmax} = Keyword.get(opts, :door, {0.6, 1.4})
    # a gap between skeleton ends is the drawn gap plus one stroke width (each end loses half)
    doors = walls |> doors(dmin / scale + v.stroke, dmax / scale + v.stroke) |> Enum.map(&%{&1 | width: &1.width - v.stroke})
    closed = walls ++ Enum.map(doors, &Map.take(&1, [:x0, :y0, :x1, :y1]))
    rooms = faces(closed) |> Enum.map(fn poly -> %{polygon: poly, area: abs(shoelace(poly)) * scale * scale} end) |> Enum.filter(&(&1.area > 1.0))
    mesh = extrude(walls, doors, rooms, scale, Keyword.get(opts, :height, 2.7))
    %{walls: walls, doors: doors, rooms: rooms, scale: scale, mesh: mesh, glb: Base.encode64(Vapor.Geom.glb(mesh)), w: v.w, h: v.h}
  end

  # lengths of straight runs: collinear axis-aligned walls that touch, joined (a T-junction splits a wall)
  defp runs(walls) do
    {aligned, other} = Enum.split_with(walls, &(axis(&1) != nil))

    joined =
      aligned
      |> Enum.group_by(fn l -> {axis(l), Float.round(offset(l), 0)} end)
      |> Enum.flat_map(fn {{ax, _}, ls} ->
        ivs = ls |> Enum.map(fn l -> if ax == :h, do: Enum.sort([l.x0, l.x1]), else: Enum.sort([l.y0, l.y1]) end) |> Enum.sort()
        ivs |> Enum.reduce([], fn [a, b], acc ->
          case acc do
            [[c, d] | rest] when a <= d + 1.0 -> [[c, max(b, d)] | rest]
            _ -> [[a, b] | acc]
          end
        end)
        |> Enum.map(fn [a, b] -> b - a end)
      end)

    joined ++ Enum.map(other, &len/1)
  end

  defp len(l), do: :math.sqrt((l.x1 - l.x0) ** 2 + (l.y1 - l.y0) ** 2)

  # a door: two collinear walls (same snapped axis and offset) whose facing ends are a door-width apart
  defp doors(walls, lo, hi) do
    for {a, i} <- Enum.with_index(walls), {b, j} <- Enum.with_index(walls), i < j, axis(a) != nil, axis(a) == axis(b),
        abs(offset(a) - offset(b)) < 3, {p, q} = facing(a, b), g = dist(p, q), g >= lo and g <= hi,
        do: %{x0: elem(p, 0), y0: elem(p, 1), x1: elem(q, 0), y1: elem(q, 1), width: g}
  end

  defp axis(l) do
    cond do
      abs(l.y1 - l.y0) < 1.0e-6 -> :h
      abs(l.x1 - l.x0) < 1.0e-6 -> :v
      true -> nil
    end
  end

  defp offset(l), do: if(axis(l) == :h, do: l.y0, else: l.x0)

  defp facing(a, b) do
    pa = [{a.x0, a.y0}, {a.x1, a.y1}]
    pb = [{b.x0, b.y0}, {b.x1, b.y1}]
    (for p <- pa, q <- pb, do: {p, q}) |> Enum.min_by(fn {p, q} -> dist(p, q) end)
  end

  # the bounded faces of the planar graph of segments (split at every intersection), by the half-edge walk
  defp faces(segs) do
    pts = split_all(segs)
    key = fn {x, y} -> {Float.round(x * 1.0, 1), Float.round(y * 1.0, 1)} end
    edges = pts |> Enum.map(fn {p, q} -> {key.(p), key.(q)} end) |> Enum.reject(fn {p, q} -> p == q end) |> Enum.uniq()
    adj = Enum.reduce(edges, %{}, fn {p, q}, acc -> acc |> Map.update(p, [q], &[q | &1]) |> Map.update(q, [p], &[p | &1]) end)
    adj = Map.new(adj, fn {p, ns} -> {p, ns |> Enum.uniq() |> Enum.sort_by(fn q -> ang(p, q) end)} end)
    halfs = for {p, ns} <- adj, q <- ns, do: {p, q}

    {faces, _} =
      Enum.reduce(halfs, {[], MapSet.new()}, fn h, {faces, used} ->
        if MapSet.member?(used, h), do: {faces, used}, else: (
          {face, used} = walk_face(h, h, adj, [], used, 0)
          {[face | faces], used})
      end)

    # bounded faces wind one way (positive area in image coordinates here); drop the outer face
    faces |> Enum.filter(&(length(&1) >= 3 and shoelace(&1) > 0))
  end

  defp walk_face({p, q} = h, start, adj, acc, used, n) do
    used = MapSet.put(used, h)
    acc = [p | acc]
    # next half-edge: at q, turn to the neighbour just before p in angular order
    ns = adj[q]
    i = Enum.find_index(ns, &(&1 == p))
    r = Enum.at(ns, rem(i - 1 + length(ns), length(ns)))
    nxt = {q, r}
    if nxt == start or n > 500, do: {Enum.reverse(acc), used}, else: walk_face(nxt, start, adj, acc, used, n + 1)
  end

  defp ang({x0, y0}, {x1, y1}), do: :math.atan2(y1 - y0, x1 - x0)

  defp shoelace(poly) do
    poly |> Enum.zip(tl(poly) ++ [hd(poly)]) |> Enum.reduce(0.0, fn {{x0, y0}, {x1, y1}}, s -> s + x0 * y1 - x1 * y0 end) |> Kernel./(2)
  end

  # every segment split at its intersections with the others (and at others' endpoints on it)
  defp split_all(segs) do
    Enum.flat_map(segs, fn s ->
      cuts = for t <- segs, t != s, p = cross(s, t), p != nil, do: p
      ts = [0.0, 1.0] ++ Enum.map(cuts, &param(s, &1))
      ts = ts |> Enum.filter(&(&1 >= -1.0e-9 and &1 <= 1 + 1.0e-9)) |> Enum.sort() |> Enum.dedup_by(&Float.round(&1, 4))
      ts |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> {at(s, a), at(s, b)} end)
    end)
  end

  defp at(s, t), do: {s.x0 + t * (s.x1 - s.x0), s.y0 + t * (s.y1 - s.y0)}
  defp param(s, {x, y}), do: (dx = s.x1 - s.x0; dy = s.y1 - s.y0; ((x - s.x0) * dx + (y - s.y0) * dy) / (dx * dx + dy * dy))

  defp cross(s, t) do
    {x1, y1, x2, y2} = {s.x0, s.y0, s.x1, s.y1}
    {x3, y3, x4, y4} = {t.x0, t.y0, t.x1, t.y1}
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1.0e-9 do
      nil
    else
      u = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
      v = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / d
      if u >= -1.0e-6 and u <= 1 + 1.0e-6 and v >= -1.0e-6 and v <= 1 + 1.0e-6, do: {x1 + u * (x2 - x1), y1 + u * (y2 - y1)}, else: nil
    end
  end

  # walls as boxes (thickness 0.15 m), lintels over doors (from 2.1 m), and a floor slab per room
  defp extrude(walls, doors, rooms, scale, height) do
    t = 0.075
    boxes =
      Enum.map(walls, fn l -> box(l, scale, t, 0.0, height) end) ++
        Enum.map(doors, fn d -> box(d, scale, t, min(2.1, height - 0.1), height) end)

    floors = Enum.map(rooms, fn r -> slab(r.polygon, scale) end)
    merge(boxes ++ floors)
  end

  defp box(l, scale, t, z0, z1) do
    {x0, y0, x1, y1} = {l.x0 * scale, l.y0 * scale, l.x1 * scale, l.y1 * scale}
    {dx, dy} = {x1 - x0, y1 - y0}
    n = :math.sqrt(dx * dx + dy * dy)
    {nx, ny} = {-dy / n * t, dx / n * t}
    # extend along the wall by t so corners close
    {ex, ey} = {dx / n * t, dy / n * t}
    base = [{x0 - ex + nx, y0 - ey + ny}, {x1 + ex + nx, y1 + ey + ny}, {x1 + ex - nx, y1 + ey - ny}, {x0 - ex - nx, y0 - ey - ny}]
    # y-up world: (x, z, y) with the plan's y as depth
    vs = for z <- [z0, z1], {x, y} <- base, do: [x, z, y]
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    {vs, faces, {0.86, 0.84, 0.80}}
  end

  defp slab(poly, scale) do
    vs = Enum.map(poly, fn {x, y} -> [x * scale, 0.0, y * scale] end)
    # a fan (the rooms of a plan are simple polygons; convex in the usual case)
    faces = for i <- 1..(length(poly) - 2), do: [0, i + 1, i]
    {vs, faces, {0.55, 0.42, 0.30}}
  end

  defp merge(parts) do
    {vs, fs, cs, _} =
      Enum.reduce(parts, {[], [], [], 0}, fn {v, f, {r, g, b}}, {vs, fs, cs, off} ->
        {vs ++ List.flatten(v), fs ++ (f |> List.flatten() |> Enum.map(&(&1 + off))), cs ++ List.flatten(List.duplicate([r, g, b], length(v))), off + length(v)}
      end)

    %Mesh{vertices: Enum.map(vs, &(&1 * 1.0)), faces: fs, colors: cs}
  end
end
