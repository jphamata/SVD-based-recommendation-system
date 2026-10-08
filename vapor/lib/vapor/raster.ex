defmodule Vapor.Raster do
  @moduledoc """
  Raster operations shared by the image tools (`Vapor.Sketch`): area-average
  resizing, dilation of a mask, Zhang–Suen thinning and the skeleton graph
  of a mask (Zhang & Suen 1984: thinning, then tracing chains between
  endpoint and junction clusters), and Ramer–Douglas–Peucker simplification.
  Masks are boolean tuples of `w·h` pixels, row-major.
  """
  alias Vapor.Modal.Image

  @doc "An image resized (area average) so that its longer side is at most `max`."
  def fit(%Image{w: w, h: h} = img, max) do
    s = max(w, h) / max
    if s <= 1.0, do: rgb(img), else: resize(rgb(img), max(round(w / s), 1), max(round(h / s), 1))
  end

  defp rgb(%Image{c: 3} = img), do: img
  defp rgb(%Image{c: 1, w: w, h: h, px: px}), do: Image.new(w, h, 3, px |> Tuple.to_list() |> Enum.flat_map(&[&1, &1, &1]))

  @doc "Area-average resize to nw × nh (RGB)."
  def resize(%Image{w: w, h: h, px: px}, nw, nh) do
    sx = w / nw
    sy = h / nh

    vals =
      for y <- 0..(nh - 1), x <- 0..(nw - 1) do
        {x0, x1} = {trunc(x * sx), max(trunc((x + 1) * sx) - 1, trunc(x * sx))}
        {y0, y1} = {trunc(y * sy), max(trunc((y + 1) * sy) - 1, trunc(y * sy))}
        n = (x1 - x0 + 1) * (y1 - y0 + 1)

        {r, g, b} =
          for yy <- y0..min(y1, h - 1), xx <- x0..min(x1, w - 1), reduce: {0.0, 0.0, 0.0} do
            {r, g, b} -> i = (yy * w + xx) * 3; {r + elem(px, i), g + elem(px, i + 1), b + elem(px, i + 2)}
          end

        [r / n, g / n, b / n]
      end

    Image.new(nw, nh, 3, List.flatten(vals))
  end

  # a boolean tuple: pixels within r of a true pixel of `f` (two passes of a box max)
  defp dilate(w, h, f, r) do
    base = for i <- 0..(w * h - 1), do: f.(i)
    rowp = base |> Enum.chunk_every(w) |> Enum.map(&box_any(&1, r))
    cols = rowp |> Enum.zip_with(& &1) |> Enum.map(&box_any(&1, r))
    cols |> Enum.zip_with(& &1) |> List.flatten() |> List.to_tuple()
  end

  # sliding-window "any" over a list
  defp box_any(l, r) do
    t = List.to_tuple(l)
    n = tuple_size(t)
    pref = Enum.scan(l, 0, fn v, acc -> acc + if(v, do: 1, else: 0) end) |> List.to_tuple()
    count = fn a, b -> elem(pref, b) - if(a > 0, do: elem(pref, a - 1), else: 0) end
    for i <- 0..(n - 1), do: count.(max(i - r, 0), min(i + r, n - 1)) > 0
  end

  @doc "Zhang–Suen thinning of a boolean tuple (w×h): a one-pixel skeleton (boolean tuple)."
  def thin(w, h, mask) do
    grid = :atomics.new(w * h, signed: false)
    for i <- 0..(w * h - 1), elem(mask, i), do: :atomics.put(grid, i + 1, 1)
    at = fn x, y -> if x < 0 or y < 0 or x >= w or y >= h, do: 0, else: :atomics.get(grid, y * w + x + 1) end
    thin_loop(w, h, grid, at)
    List.to_tuple(for i <- 1..(w * h), do: :atomics.get(grid, i) == 1)
  end

  defp thin_loop(w, h, grid, at) do
    changed =
      Enum.map([0, 1], fn step ->
        del =
          for y <- 0..(h - 1), x <- 0..(w - 1), at.(x, y) == 1 do
            p = [at.(x, y - 1), at.(x + 1, y - 1), at.(x + 1, y), at.(x + 1, y + 1), at.(x, y + 1), at.(x - 1, y + 1), at.(x - 1, y), at.(x - 1, y - 1)]
            [p2, _p3, p4, _p5, p6, _p7, p8, _p9] = p
            b = Enum.sum(p)
            a = Enum.zip(p, tl(p) ++ [hd(p)]) |> Enum.count(fn {u, v} -> u == 0 and v == 1 end)
            cond1 = if step == 0, do: p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0, else: p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
            if b >= 2 and b <= 6 and a == 1 and cond1, do: y * w + x + 1, else: nil
          end
          |> Enum.reject(&is_nil/1)

        Enum.each(del, &:atomics.put(grid, &1, 0))
        length(del)
      end)

    if Enum.sum(changed) > 0, do: thin_loop(w, h, grid, at), else: :ok
  end

  @doc """
  The skeleton graph of a boolean mask (tuple, w×h): `{nodes, chains}` —
  nodes `%{id => {x, y, degree}}`, chains `[%{from, to, px}]` (`to` nil
  for a dead end). Thinning, then tracing.
  """
  def skeleton(w, h, mask), do: skeleton_graph(w, h, thin(w, h, mask))

  @doc "A boolean mask dilated by `r` pixels (a square window)."
  def dilate_mask(w, h, mask, r), do: dilate(w, h, fn i -> elem(mask, i) end, r)

  # the skeleton as a graph: nodes are clusters of endpoint/junction pixels
  # (by the crossing number, so a staircase is not a junction); edges are
  # the pixel chains between them, each traced once
  defp skeleton_graph(w, h, skel) do
    on = fn {x, y} -> x >= 0 and y >= 0 and x < w and y < h and elem(skel, y * w + x) end
    ring = [{0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}]
    nb = fn {x, y} -> for {dx, dy} <- ring, on.({x + dx, y + dy}), do: {x + dx, y + dy} end
    cross = fn {x, y} -> vals = Enum.map(ring, fn {dx, dy} -> on.({x + dx, y + dy}) end); Enum.zip(vals, tl(vals) ++ [hd(vals)]) |> Enum.count(fn {u, v} -> not u and v end) end
    pts = for y <- 0..(h - 1), x <- 0..(w - 1), on.({x, y}), do: {x, y}
    node_px = for p <- pts, (c = cross.(p); c != 2 or length(nb.(p)) <= 1), into: MapSet.new(), do: p

    # cluster adjacent node pixels
    {cluster, _} =
      Enum.reduce(node_px, {%{}, 0}, fn p, {cl, k} ->
        if Map.has_key?(cl, p), do: {cl, k}, else: {flood(p, node_px, nb, k, cl), k + 1}
      end)

    centers =
      cluster |> Enum.group_by(&elem(&1, 1), &elem(&1, 0))
      |> Map.new(fn {k, ps} -> {k, {round(Enum.sum(Enum.map(ps, &elem(&1, 0))) / length(ps)), round(Enum.sum(Enum.map(ps, &elem(&1, 1))) / length(ps)), length(Enum.uniq(Enum.flat_map(ps, fn p -> Enum.reject(nb.(p), &Map.has_key?(cluster, &1)) end)))}} end)

    {chains, _} =
      Enum.reduce(Map.keys(cluster), {[], MapSet.new()}, fn p, {chains, seen} ->
        Enum.reduce(Enum.reject(nb.(p), &Map.has_key?(cluster, &1)), {chains, seen}, fn q, {chains, seen} ->
          if MapSet.member?(seen, q), do: {chains, seen}, else: (
            {path, stop, seen} = walk(q, [q], MapSet.put(seen, q), nb, cluster)
            {from, to} = {cluster[p], stop && cluster[stop]}
            {[%{from: from, to: to, px: path} | chains], seen})
        end)
      end)

    # closed loops with no endpoint or junction (a circle, a rectangle): cut each at one pixel
    {_cluster, centers, chains} = loops(pts, cluster, centers, chains, nb)
    {centers, chains}
  end

  defp loops(pts, cluster, centers, chains, nb) do
    seen = chains |> Enum.flat_map(& &1.px) |> MapSet.new()

    Enum.reduce(pts, {cluster, centers, chains, seen}, fn p, {cl, ce, ch, seen} ->
      if Map.has_key?(cl, p) or MapSet.member?(seen, p) do
        {cl, ce, ch, seen}
      else
        k = map_size(ce) + 1_000_000
        cl = Map.put(cl, p, k)
        {x, y} = p
        ce = Map.put(ce, k, {x, y, 2})

        case Enum.reject(nb.(p), &(MapSet.member?(seen, &1) or Map.has_key?(cl, &1))) do
          [] -> {cl, ce, ch, MapSet.put(seen, p)}
          [q | _] ->
            {path, _stop, seen} = walk(q, [q], MapSet.put(MapSet.put(seen, q), p), nb, cl)
            {cl, ce, [%{from: k, to: k, px: path} | ch], seen}
        end
      end
    end)
    |> then(fn {cl, ce, ch, _} -> {cl, ce, ch} end)
  end

  defp flood(p, set, nb, k, cl) do
    Enum.reduce(nb.(p), Map.put(cl, p, k), fn q, cl -> if MapSet.member?(set, q) and not Map.has_key?(cl, q), do: flood(q, set, nb, k, cl), else: cl end)
  end

  # follow a chain until a node pixel (returned) or a dead end (nil)
  defp walk({cx, cy} = cur, path, seen, nb, cluster) do
    ns = nb.(cur)
    case Enum.find(ns, &Map.has_key?(cluster, &1)) do
      stop when stop != nil and length(path) > 1 -> {Enum.reverse(path), stop, seen}
      _ ->
        case ns |> Enum.reject(&(MapSet.member?(seen, &1) or Map.has_key?(cluster, &1))) |> Enum.sort_by(fn {x, y} -> abs(x - cx) + abs(y - cy) end) do
          [] -> {Enum.reverse(path), Enum.find(ns, &Map.has_key?(cluster, &1)), seen}
          [nxt | _] -> walk(nxt, [nxt | path], MapSet.put(seen, nxt), nb, cluster)
        end
    end
  end

  @doc "Ramer–Douglas–Peucker simplification of a point list."
  def rdp(pts, _tol) when length(pts) < 3, do: pts

  def rdp(pts, tol) do
    {a, b} = {hd(pts), List.last(pts)}
    {i, d} = pts |> Enum.with_index() |> Enum.map(fn {p, i} -> {i, seg_dist(p, a, b)} end) |> Enum.max_by(&elem(&1, 1))
    if d > tol, do: (left = rdp(Enum.take(pts, i + 1), tol); right = rdp(Enum.drop(pts, i), tol); left ++ tl(right)), else: [a, b]
  end

  defp seg_dist({px, py}, {ax, ay}, {bx, by}) do
    {dx, dy} = {bx - ax, by - ay}
    l2 = dx * dx + dy * dy
    t = if l2 == 0, do: 0.0, else: max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    :math.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)
  end
end
