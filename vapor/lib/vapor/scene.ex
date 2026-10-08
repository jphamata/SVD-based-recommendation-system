defmodule Vapor.Scene do
  @moduledoc """
  Giving a picture life (docs/SCENE.md): any image — a photograph, a
  painting, an AI render, a drawing — becomes a **living scene** the
  console plays forever: layers at depths with the hidden background
  reconstructed, so the camera can move and the picture opens up; a
  ground the inhabitants walk on, by path search; light, weather, wind,
  particles; and direction by prompt. Everything the engine does is a
  function of the scene and a seed — a loop with controlled entropy,
  replayable, savable (`Vapor.Archive`) and exportable as one HTML file
  that plays offline.

  The analysis, from first principles and without a trained model:

    * **regions** — SLIC superpixels (Achanta et al. 2012) in CIELAB,
      merged over the region adjacency graph by colour;
    * **depth** — the ground-plane prior: an object stands where its lowest
      pixels are, and on a ground plane seen from eye height the depth of a
      point below the horizon is ∝ 1/(y − y_horizon); the sky (a smooth,
      bright region at the top) is at infinity. A *heuristic*, said as
      such, and editable layer by layer; a monocular depth network, when
      one is loaded, is the drop-in replacement;
    * **layers with what is behind them** — each depth layer keeps its own
      pixels and fills the band hidden behind nearer layers by push-pull
      interpolation (Gortler et al. 1996), so a moved camera reveals a
      plausible continuation instead of a hole;
    * **the walkable ground** — cells of the ground regions below the
      horizon, for A* in the engine;
    * **light** — the brightest pixels' centroid and colour.

  For **drawings** (`rig/2`): the strokes are thinned (Zhang & Suen 1984)
  to a skeleton graph, its chains become bones, a mesh covers the ink and
  each vertex is skinned to its nearest bones — the engine animates the
  drawing (wave, walk, dance, breathe) by forward kinematics and linear
  blend skinning. The motion comes from the skeleton's topology, not
  from knowing what the drawing depicts.

  **Direction** (`direct/1`): a prompt in Portuguese or English becomes
  operations (weather, time of day, wind, lights, inhabitants, camera,
  entropy, animation, destinations) by a vocabulary — every word it does
  not understand is reported, never silently guessed.
  """
  alias Vapor.Modal.Image

  # ================================================================ images

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

  defp lab({r, g, b}) do
    lin = fn c -> if c <= 0.04045, do: c / 12.92, else: :math.pow((c + 0.055) / 1.055, 2.4) end
    {r, g, b} = {lin.(r), lin.(g), lin.(b)}
    {x, y, z} = {(0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, 0.2126 * r + 0.7152 * g + 0.0722 * b, (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883}
    f = fn t -> if t > 0.008856, do: :math.pow(t, 1 / 3), else: 7.787 * t + 16 / 116 end
    {116 * f.(y) - 16, 500 * (f.(x) - f.(y)), 200 * (f.(y) - f.(z))}
  end

  defp pixel(%Image{w: w, px: px}, x, y), do: (i = (y * w + x) * 3; {elem(px, i), elem(px, i + 1), elem(px, i + 2)})

  # ================================================================== SLIC

  @doc "SLIC superpixels: a label per pixel (tuple of rows) and the label count."
  def slic(%Image{w: w, h: h} = img, k \\ 160, iters \\ 5, m \\ 12.0) do
    labs = for y <- 0..(h - 1), x <- 0..(w - 1), do: lab(pixel(img, x, y))
    labs = List.to_tuple(labs)
    s = max(trunc(:math.sqrt(w * h / k)), 4)
    centers = for cy <- div(s, 2)..(h - 1)//s, cx <- div(s, 2)..(w - 1)//s, do: {elem(labs, cy * w + cx), cx * 1.0, cy * 1.0}
    centers = List.to_tuple(centers)
    labels = :atomics.new(w * h, signed: true)

    centers =
      Enum.reduce(1..iters, centers, fn _, centers ->
        dist = :atomics.new(w * h, signed: false)
        for i <- 1..(w * h), do: :atomics.put(dist, i, 0xFFFFFFFFFFFF)

        for ci <- 0..(tuple_size(centers) - 1) do
          {{l, a, b}, cx, cy} = elem(centers, ci)
          for y <- max(trunc(cy) - s, 0)..min(trunc(cy) + s, h - 1), x <- max(trunc(cx) - s, 0)..min(trunc(cx) + s, w - 1) do
            {l2, a2, b2} = elem(labs, y * w + x)
            dc = (l - l2) ** 2 + (a - a2) ** 2 + (b - b2) ** 2
            ds = ((x - cx) ** 2 + (y - cy) ** 2) / (s * s) * m * m
            d = trunc((dc + ds) * 1000)
            idx = y * w + x + 1
            if d < :atomics.get(dist, idx), do: (:atomics.put(dist, idx, d); :atomics.put(labels, idx, ci))
          end
        end

        sums =
          Enum.reduce(0..(w * h - 1), %{}, fn i, acc ->
            c = :atomics.get(labels, i + 1)
            {l, a, b} = elem(labs, i)
            {x, y} = {rem(i, w), div(i, w)}
            Map.update(acc, c, {l, a, b, x, y, 1}, fn {sl, sa, sb, sx, sy, n} -> {sl + l, sa + a, sb + b, sx + x, sy + y, n + 1} end)
          end)

        for ci <- 0..(tuple_size(centers) - 1) do
          case sums[ci] do
            {sl, sa, sb, sx, sy, n} -> {{sl / n, sa / n, sb / n}, sx / n, sy / n}
            nil -> elem(centers, ci)
          end
        end
        |> List.to_tuple()
      end)

    rows = for y <- 0..(h - 1), do: (for x <- 0..(w - 1), do: :atomics.get(labels, y * w + x + 1)) |> List.to_tuple()
    {List.to_tuple(rows), tuple_size(centers), labs}
  end

  # =============================================================== regions

  # merge superpixels over the adjacency graph by colour, smallest distance first
  defp merge(rows, labs, w, h, target, max_dist) do
    at = fn x, y -> elem(elem(rows, y), x) end

    stats =
      Enum.reduce(0..(h - 1), %{}, fn y, acc ->
        Enum.reduce(0..(w - 1), acc, fn x, acc ->
          {l, a, b} = elem(labs, y * w + x)
          Map.update(acc, at.(x, y), {l, a, b, 1}, fn {sl, sa, sb, n} -> {sl + l, sa + a, sb + b, n + 1} end)
        end)
      end)

    edges =
      for y <- 0..(h - 1), x <- 0..(w - 1), {dx, dy} <- [{1, 0}, {0, 1}], x + dx < w, y + dy < h,
          p = at.(x, y), q = at.(x + dx, y + dy), p != q, into: MapSet.new(), do: {min(p, q), max(p, q)}

    parent = Map.new(Map.keys(stats), &{&1, &1})
    do_merge(parent, stats, MapSet.to_list(edges), map_size(stats), target, max_dist)
  end

  defp find(parent, x), do: if(parent[x] == x, do: x, else: find(parent, parent[x]))

  defp do_merge(parent, stats, edges, count, target, max_dist) do
    mean = fn {l, a, b, n} -> {l / n, a / n, b / n} end
    d = fn p, q -> {l1, a1, b1} = mean.(stats[p]); {l2, a2, b2} = mean.(stats[q]); :math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2) end
    live = edges |> Enum.map(fn {p, q} -> {find(parent, p), find(parent, q)} end) |> Enum.reject(fn {p, q} -> p == q end) |> Enum.uniq()

    # small regions merge first whatever their colour (noise), then by colour
    best =
      Enum.min_by(live, fn {p, q} -> {sm, sn} = {elem(stats[p], 3), elem(stats[q], 3)}; d.(p, q) * if(min(sm, sn) < 40, do: 0.1, else: 1.0) end, fn -> nil end)

    case best do
      nil -> parent
      {p, q} ->
        small = min(elem(stats[p], 3), elem(stats[q], 3)) < 40
        if count <= target and not small and d.(p, q) > 0, do: parent, else: (if not small and d.(p, q) > max_dist and count <= target * 2, do: parent, else: (
          {l1, a1, b1, n1} = stats[p]
          {l2, a2, b2, n2} = stats[q]
          stats = stats |> Map.put(p, {l1 + l2, a1 + a2, b1 + b2, n1 + n2}) |> Map.delete(q)
          do_merge(Map.put(parent, q, p), stats, live, count - 1, target, max_dist)))
    end
  end

  # ============================================================== analysis

  @doc """
  Analyse a picture into a scene: `%{w, h, horizon, layers: [%{id, depth,
  kind, png (base64 RGBA), area}], walk: %{cols, rows, cells}, light,
  palette, image}`. Options: `max` (display size, 768), `regions` (12).
  """
  def analyze(%Image{} = img, opts \\ []) do
    full = fit(img, Keyword.get(opts, :max, 768))
    small = fit(full, Keyword.get(opts, :analysis, 256))
    {sw, sh} = {small.w, small.h}
    {rows, _k, labs} = slic(small, Keyword.get(opts, :superpixels, 140))
    parent = merge(rows, labs, sw, sh, Keyword.get(opts, :regions, 12), 18.0)
    region = fn x, y -> find(parent, elem(elem(rows, y), x)) end
    reg = for y <- 0..(sh - 1), do: List.to_tuple(for x <- 0..(sw - 1), do: region.(x, y))
    reg = List.to_tuple(reg)

    info = region_info(reg, small, sw, sh)
    sky = grow_sky(for({r, i} <- info, sky?(i, sw, sh, info), do: r), info, sh)
    horizon = horizon(info, sky, sh)
    depth = Map.new(info, fn {r, i} -> {r, if(r in sky, do: 1000.0, else: ground_depth(i.bottom, horizon, sh))} end)
    groups = layer_groups(depth, Keyword.get(opts, :layers, 6))

    # the label map at display resolution (nearest), then masks per layer
    {fw, fh} = {full.w, full.h}
    {kx, ky} = {sw / fw, sh / fh}
    lmap = groups |> Enum.flat_map(fn {lid, rs, _} -> Enum.map(rs, &{&1, lid}) end) |> Map.new()
    labels_full = for y <- 0..(fh - 1), x <- 0..(fw - 1), do: lmap[elem(elem(reg, min(trunc(y * ky), sh - 1)), min(trunc(x * kx), sw - 1))]
    labels_full = List.to_tuple(labels_full)

    layers =
      for {lid, rs, d} <- Enum.sort_by(groups, fn {_, _, d} -> -d end) do
        kind = cond do
          Enum.all?(rs, &(&1 in sky)) -> "sky"
          d == Enum.max(Map.values(depth)) -> "backdrop"
          Enum.any?(rs, fn r -> info[r].bottom >= sh - 2 end) -> "ground"
          true -> "object"
        end

        nearer = for {l2, _, d2} <- groups, d2 < d, do: l2
        backmost = d == groups |> Enum.map(&elem(&1, 2)) |> Enum.max()
        png = layer_png(full, labels_full, lid, nearer, backmost)
        %{id: lid, depth: d, kind: kind, png: png, area: Enum.sum(for r <- rs, do: info[r].area) / (sw * sh), regions: length(rs)}
      end

    %{w: fw, h: fh, horizon: horizon / sh, layers: layers, walk: walk_grid(reg, info, sky, horizon, sw, sh), light: light(small), palette: palette(info, small),
      image: "data:image/png;base64," <> Base.encode64(Image.png(full)), regions: map_size(info)}
  end

  defp region_info(reg, img, w, h) do
    Enum.reduce(0..(h - 1), %{}, fn y, acc ->
      Enum.reduce(0..(w - 1), acc, fn x, acc ->
        r = elem(elem(reg, y), x)
        {cr, cg, cb} = pixel(img, x, y)
        lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
        grad = if x + 1 < w, do: (({r2, g2, b2} = pixel(img, x + 1, y)); abs(0.299 * r2 + 0.587 * g2 + 0.114 * b2 - lum)), else: 0.0
        Map.update(acc, r, %{area: 1, top: y, bottom: y, left: x, right: x, sum: {cr, cg, cb}, lum: lum, grad: grad, ys: [y]},
          fn i -> %{i | area: i.area + 1, top: min(i.top, y), bottom: max(i.bottom, y), left: min(i.left, x), right: max(i.right, x),
                        sum: {elem(i.sum, 0) + cr, elem(i.sum, 1) + cg, elem(i.sum, 2) + cb}, lum: i.lum + lum, grad: i.grad + grad,
                        ys: if(length(i.ys) < 400, do: [y | i.ys], else: i.ys)} end)
      end)
    end)
    |> Map.new(fn {r, i} -> {r, %{i | lum: i.lum / i.area, grad: i.grad / i.area, sum: {elem(i.sum, 0) / i.area, elem(i.sum, 1) / i.area, elem(i.sum, 2) / i.area}}} end)
  end

  # the sky: touches the top, ends in the upper part, smooth and bright (or bluish)
  defp sky?(i, _w, h, info) do
    grads = info |> Map.values() |> Enum.map(& &1.grad) |> Enum.sort()
    med = Enum.at(grads, div(length(grads), 2))
    {r, _g, b} = i.sum
    i.top == 0 and i.bottom < 0.7 * h and i.grad <= med and (i.lum > 0.45 or b > r + 0.05)
  end

  # the sky continues downwards: a smooth region starting where the sky ends, of a sky-like colour
  defp grow_sky([], _info, _h), do: []

  defp grow_sky(sky, info, h) do
    grads = info |> Map.values() |> Enum.map(& &1.grad) |> Enum.sort()
    med = Enum.at(grads, div(length(grads), 2))
    low = Enum.max(for r <- sky, do: info[r].bottom)

    # a smooth region starting just under a sky region, of a colour close to that one's
    more =
      for {r, i} <- info, r not in sky, i.top <= low + 2, i.top > 0, i.bottom < 0.7 * h, i.grad <= max(2 * med, 0.006),
          above = Enum.min_by(sky, fn q -> abs(info[q].bottom - i.top) end),
          ({cr, cg, cb} = i.sum; {sr, sg, sb} = info[above].sum; abs(cr - sr) + abs(cg - sg) + abs(cb - sb) < 0.3), do: r

    if more == [], do: sky, else: grow_sky(sky ++ more, info, h)
  end

  defp horizon(info, [], h), do: (_ = info; 0.4 * h)
  defp horizon(info, sky, h), do: min(Enum.max(for r <- sky, do: info[r].bottom) + 1, 0.75 * h) |> max(0.15 * h)

  @doc "Ground-plane depth of a contact point at row y (rows below the horizon): 1 at the bottom row, growing towards the horizon."
  def ground_depth(y, horizon, h) do
    if y <= horizon + 1, do: 60.0, else: min((h - horizon) / (y - horizon), 60.0)
  end

  # cluster regions into at most n depth layers (log-depth, nearest neighbours)
  defp layer_groups(depth, n) do
    sorted = depth |> Enum.sort_by(&elem(&1, 1))
    {groups, _} =
      Enum.reduce(sorted, {[], nil}, fn {r, d}, {gs, last} ->
        if last && :math.log(d) - :math.log(last) < 0.18, do: {List.update_at(gs, -1, fn {rs, ds} -> {[r | rs], [d | ds]} end), d}, else: {gs ++ [{[r], [d]}], d}
      end)

    groups = shrink(groups, n)
    for {{rs, ds}, i} <- Enum.with_index(groups), do: {i, rs, Enum.sum(ds) / length(ds)}
  end

  defp shrink(groups, n) when length(groups) <= n, do: groups

  defp shrink(groups, n) do
    means = Enum.map(groups, fn {_, ds} -> :math.log(Enum.sum(ds) / length(ds)) end)
    {_, i} = means |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> b - a end) |> Enum.with_index() |> Enum.min()
    {a, b} = {Enum.at(groups, i), Enum.at(groups, i + 1)}
    merged = {elem(a, 0) ++ elem(b, 0), elem(a, 1) ++ elem(b, 1)}
    shrink(List.replace_at(groups, i, merged) |> List.delete_at(i + 1), n)
  end

  # RGBA of one layer: its pixels, and a band behind nearer layers filled by push-pull
  defp layer_png(%Image{w: w, h: h, px: px}, labels, lid, nearer, backmost) do
    own = fn i -> elem(labels, i) == lid end
    near = MapSet.new(nearer)
    hidden = fn i -> MapSet.member?(near, elem(labels, i)) end
    # alpha: own pixels; and (dilated) hidden pixels near own ones — everything hidden for the back layer
    band = if backmost, do: nil, else: dilate(w, h, own, 14)

    known = for i <- 0..(w * h - 1), do: own.(i)
    want = for i <- 0..(w * h - 1), do: own.(i) or (hidden.(i) and (backmost or elem(band, i)))
    colors = for i <- 0..(w * h - 1), do: {elem(px, 3 * i), elem(px, 3 * i + 1), elem(px, 3 * i + 2)}
    filled = push_pull(w, h, colors, known)

    wt = List.to_tuple(want)
    raw =
      for y <- 0..(h - 1), into: <<>> do
        <<0>> <> for(x <- 0..(w - 1), into: <<>>, do: (i = y * w + x; {r, g, b} = elem(filled, i); <<c8(r), c8(g), c8(b), if(elem(wt, i), do: 255, else: 0)>>))
      end

    "data:image/png;base64," <> Base.encode64(png_rgba(w, h, raw))
  end

  defp c8(v), do: round(min(1.0, max(0.0, v)) * 255)

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

  @doc """
  Push-pull fill (Gortler et al. 1996): known pixels keep their colour,
  unknown ones take a smooth interpolation from a pyramid of weighted
  averages. Returns a tuple of `{r, g, b}`.
  """
  def push_pull(w, h, colors, known) do
    level0 = for {c, k} <- Enum.zip(colors, known), do: if(k, do: {c, 1.0}, else: {{0.0, 0.0, 0.0}, 0.0})
    pp(w, h, List.to_tuple(level0)) |> Tuple.to_list() |> Enum.map(&elem(&1, 0)) |> List.to_tuple()
  end

  defp pp(w, h, lvl) when w <= 1 and h <= 1, do: lvl

  defp pp(w, h, lvl) do
    {cw, ch} = {div(w + 1, 2), div(h + 1, 2)}
    coarse =
      for y <- 0..(ch - 1), x <- 0..(cw - 1) do
        Enum.reduce(for(dy <- 0..1, dx <- 0..1, xx = 2 * x + dx, yy = 2 * y + dy, xx < w, yy < h, do: elem(lvl, yy * w + xx)), {{0.0, 0.0, 0.0}, 0.0}, fn {{r, g, b}, wt}, {{sr, sg, sb}, sw} ->
          {{sr + r * wt, sg + g * wt, sb + b * wt}, sw + wt}
        end)
        |> then(fn {{r, g, b}, wt} -> if wt > 0, do: {{r / wt, g / wt, b / wt}, min(wt, 1.0)}, else: {{0.0, 0.0, 0.0}, 0.0} end)
      end
      |> List.to_tuple()

    up = pp(cw, ch, coarse)

    # pull: bilinear from the coarse level (cell centres at 2i + 0.5)
    cat = fn i, j -> elem(elem(up, min(max(j, 0), ch - 1) * cw + min(max(i, 0), cw - 1)), 0) end

    for y <- 0..(h - 1), x <- 0..(w - 1) do
      {c, wt} = elem(lvl, y * w + x)
      {fx, fy} = {(x - 0.5) / 2, (y - 0.5) / 2}
      {i0, j0} = {floor(fx), floor(fy)}
      {tx, ty} = {fx - i0, fy - j0}
      top = mix(cat.(i0, j0), cat.(i0 + 1, j0), tx)
      bot = mix(cat.(i0, j0 + 1), cat.(i0 + 1, j0 + 1), tx)
      {mix(mix(top, bot, ty), c, wt), 1.0}
    end
    |> List.to_tuple()
  end

  defp mix({r1, g1, b1}, {r2, g2, b2}, t), do: {r1 * (1 - t) + r2 * t, g1 * (1 - t) + g2 * t, b1 * (1 - t) + b2 * t}

  @doc "A PNG of 8-bit RGBA rows already prefixed with filter bytes."
  def png_rgba(w, h, raw) do
    chunk = fn type, data -> <<byte_size(data)::32>> <> type <> data <> <<:erlang.crc32(type <> data)::32>> end
    <<137, 80, 78, 71, 13, 10, 26, 10>> <> chunk.("IHDR", <<w::32, h::32, 8, 6, 0, 0, 0>>) <> chunk.("IDAT", :zlib.compress(raw)) <> chunk.("IEND", <<>>)
  end

  # walkable cells: ground regions (touching the bottom edge), below the horizon
  defp walk_grid(reg, info, sky, horizon, w, h) do
    cell = max(div(max(w, h), 40), 2)
    {cols, rows} = {div(w, cell), div(h, cell)}
    ground = for {r, i} <- info, r not in sky, i.bottom >= h - 2, into: MapSet.new(), do: r

    cells =
      for cy <- 0..(rows - 1), cx <- 0..(cols - 1) do
        ys = (cy * cell)..min((cy + 1) * cell - 1, h - 1)
        xs = (cx * cell)..min((cx + 1) * cell - 1, w - 1)
        n = Enum.count(for(y <- ys, x <- xs, do: MapSet.member?(ground, elem(elem(reg, y), x))), & &1)
        if cy * cell > horizon + cell and n >= 0.8 * Enum.count(ys) * Enum.count(xs), do: 1, else: 0
      end

    %{cols: cols, rows: rows, cells: cells}
  end

  defp light(%Image{w: w, h: h} = img) do
    lums = for y <- 0..(h - 1), x <- 0..(w - 1), do: ({r, g, b} = pixel(img, x, y); {0.299 * r + 0.587 * g + 0.114 * b, x, y, {r, g, b}})
    top = lums |> Enum.sort_by(&elem(&1, 0), :desc) |> Enum.take(max(div(w * h, 100), 1))
    n = length(top)
    {sx, sy} = Enum.reduce(top, {0.0, 0.0}, fn {_, x, y, _}, {a, b} -> {a + x, b + y} end)
    {cr, cg, cb} = Enum.reduce(top, {0.0, 0.0, 0.0}, fn {_, _, _, {r, g, b}}, {a, c, d} -> {a + r, c + g, d + b} end)
    %{x: sx / n / w, y: sy / n / h, color: hex({cr / n, cg / n, cb / n})}
  end

  defp palette(info, _img) do
    info |> Map.values() |> Enum.sort_by(& &1.area, :desc) |> Enum.take(6) |> Enum.map(&hex(&1.sum))
  end

  defp hex({r, g, b}), do: "#" <> Enum.map_join([r, g, b], fn v -> v |> c8() |> Integer.to_string(16) |> String.pad_leading(2, "0") end)

  # ============================================================ drawings

  @doc """
  A rig for a drawing: `%{w, h, png, bones: [%{id, parent, x0, y0, x1,
  y1}], mesh: %{vertices, triangles, weights}, endpoints, junctions}`.
  Coordinates in pixels of the fitted drawing.
  """
  def rig(%Image{} = img, opts \\ []) do
    full = fit(img, Keyword.get(opts, :max, 512))
    {w, h} = {full.w, full.h}
    g = Vapor.Vision.Segment.gray(full)
    ink = Vapor.Vision.Segment.ink(g)
    mask = for y <- 0..(h - 1), x <- 0..(w - 1), do: elem(elem(ink, y), x) == 1
    # thicken thin strokes so a double line is one skeleton
    mt = List.to_tuple(mask)
    thick = dilate(w, h, fn i -> elem(mt, i) end, 2)
    skel = thin(w, h, thick)
    {nodes, chains} = skeleton_graph(w, h, skel)
    size = max(w, h)
    # spurs: short chains that end in an endpoint (thinning's whiskers)
    endpoint? = fn k -> k == nil or elem(nodes[k], 2) <= 1 end
    chains = Enum.reject(chains, fn c -> length(c.px) < size * 0.04 and (endpoint?.(c.to) or endpoint?.(c.from)) end)
    {bones, root} = bones(chains, nodes, w, h, mask)
    mesh = mesh(w, h, List.to_tuple(mask), bones, max(div(size, 32), 6))
    alpha = for y <- 0..(h - 1), into: <<>>, do: <<0>> <> for(x <- 0..(w - 1), into: <<>>, do: (v = :binary.at(g.gray, y * w + x); <<v, v, v, if(elem(thick, y * w + x), do: min(255, (255 - v) * 2), else: 0)>>))

    %{w: w, h: h, png: "data:image/png;base64," <> Base.encode64(png_rgba(w, h, alpha)), bones: bones, root: root, mesh: mesh,
      endpoints: Enum.count(degrees(chains), fn {_, d} -> d == 1 end), junctions: Enum.count(degrees(chains), fn {_, d} -> d >= 3 end)}
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

  @doc false
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

  # node degrees in the (spur-free) chain graph; a dead-end chain adds an endpoint
  defp degrees(chains) do
    Enum.reduce(chains, %{}, fn c, acc ->
      acc = Map.update(acc, c.from, 1, &(&1 + 1))
      if c.to, do: Map.update(acc, c.to, 1, &(&1 + 1)), else: Map.update(acc, {:end, c.from, length(c.px)}, 1, &(&1 + 1))
    end)
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

  # bones: each chain, node centre to node centre, simplified (RDP) into segments;
  # parents follow the chains outwards from the root node (the junction nearest the ink's centroid)
  defp bones(chains, centers, w, h, mask) do
    {sx, sy, n} = Enum.reduce(Enum.with_index(mask), {0, 0, 0}, fn {m, i}, {sx, sy, n} -> if m, do: {sx + rem(i, w), sy + div(i, w), n + 1}, else: {sx, sy, n} end)
    {cx, cy} = if n > 0, do: {sx / n, sy / n}, else: {w / 2, h / 2}
    junctions = for {k, {_, _, deg}} <- centers, deg >= 3, do: k
    pool = if junctions == [], do: Map.keys(centers), else: junctions
    root = Enum.min_by(pool, fn k -> {x, y, _} = centers[k]; (x - cx) ** 2 + (y - cy) ** 2 end, fn -> nil end)
    tol = max(w, h) * 0.025

    # BFS over nodes from the root; each chain oriented away from it
    adj = Enum.reduce(chains, %{}, fn c, acc -> if c.to, do: acc |> Map.update(c.from, [c.to], &[c.to | &1]) |> Map.update(c.to, [c.from], &[c.from | &1]), else: acc end)
    dist = if root, do: bfs(adj, [root], %{root => 0}), else: %{}
    xy = fn k -> {x, y, _} = centers[k]; {x, y} end

    {bones, _} =
      chains
      |> Enum.sort_by(fn c -> Map.get(dist, c.from, 99) end)
      |> Enum.reduce({[], %{}}, fn c, {bones, ends} ->
        forward = c.to == nil or Map.get(dist, c.from, 99) <= Map.get(dist, c.to, 99)
        {a, b} = if forward, do: {c.from, c.to}, else: {c.to, c.from}
        pts = if forward, do: c.px, else: Enum.reverse(c.px)
        start = xy.(a)
        stop = if b, do: xy.(b), else: List.last(pts)
        poly = rdp([start | pts] ++ [stop], tol) |> Enum.dedup()
        parent0 = Map.get(ends, a, -1)

        {new, last} =
          poly |> Enum.chunk_every(2, 1, :discard) |> Enum.reduce({[], parent0}, fn [{x0, y0}, {x1, y1}], {acc, parent} ->
            id = length(bones) + length(acc)
            {acc ++ [%{id: id, parent: parent, x0: x0, y0: y0, x1: x1, y1: y1}], id}
          end)

        ends = if b != nil and new != [] and not Map.has_key?(ends, b), do: Map.put(ends, b, last), else: ends
        {bones ++ new, ends}
      end)

    {rx, ry} = if root, do: xy.(root), else: {round(cx), round(cy)}
    {bones, %{x: rx, y: ry}}
  end

  defp bfs(_adj, [], dist), do: dist
  defp bfs(adj, [p | rest], dist) do
    nexts = for q <- Map.get(adj, p, []), not Map.has_key?(dist, q), do: q
    dist = Enum.reduce(nexts, dist, &Map.put(&2, &1, dist[p] + 1))
    bfs(adj, rest ++ nexts, dist)
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

  # a grid mesh over the ink (dilated), each vertex skinned to its two nearest bones
  defp mesh(w, h, mask, bones, step) do
    near = dilate(w, h, fn i -> elem(mask, i) end, step)
    gx = div(w, step)
    gy = div(h, step)
    inside = fn i, j -> x = min(i * step, w - 1); y = min(j * step, h - 1); elem(near, y * w + x) end
    verts = for j <- 0..gy, i <- 0..gx, do: {i, j}
    idx = verts |> Enum.with_index() |> Map.new()

    tris =
      for j <- 0..(gy - 1), i <- 0..(gx - 1), inside.(i, j) or inside.(i + 1, j) or inside.(i, j + 1) or inside.(i + 1, j + 1),
          t <- [[{i, j}, {i + 1, j}, {i + 1, j + 1}], [{i, j}, {i + 1, j + 1}, {i, j + 1}]], do: Enum.map(t, &idx[&1])

    used = tris |> List.flatten() |> Enum.uniq() |> Enum.sort()
    remap = used |> Enum.with_index() |> Map.new()
    vlist = Enum.map(used, fn k -> {i, j} = Enum.at(verts, k); [min(i * step, w - 1), min(j * step, h - 1)] end)

    weights =
      Enum.map(vlist, fn [x, y] ->
        ds = bones |> Enum.map(fn b -> {b.id, seg_dist({x, y}, {b.x0, b.y0}, {b.x1, b.y1})} end) |> Enum.sort_by(&elem(&1, 1)) |> Enum.take(2)
        case ds do
          [] -> []
          [{b, _}] -> [b, 1.0]
          [{b1, d1}, {b2, d2}] ->
            {w1, w2} = {1 / (d1 + 1) ** 4, 1 / (d2 + 1) ** 4}
            [b1, w1 / (w1 + w2), b2, w2 / (w1 + w2)]
        end
      end)

    %{vertices: vlist, triangles: Enum.map(tris, fn t -> Enum.map(t, &remap[&1]) end), weights: weights}
  end

  # ============================================================ direction

  @nums %{"um" => 1, "uma" => 1, "one" => 1, "dois" => 2, "duas" => 2, "two" => 2, "três" => 3, "tres" => 3, "three" => 3,
          "quatro" => 4, "four" => 4, "cinco" => 5, "five" => 5, "seis" => 6, "six" => 6, "alguns" => 4, "algumas" => 4, "some" => 4, "few" => 3,
          "muitos" => 12, "muitas" => 12, "many" => 12, "vários" => 8, "várias" => 8, "several" => 8}

  @vocab [
    {~w(chuva chovendo chover rain raining rainy), {:weather, "rain"}},
    {~w(tempestade tempestuosa tempestuoso storm stormy thunderstorm trovão trovoada thunder relâmpago relâmpagos lightning), {:weather, "storm"}},
    {~w(neve nevando nevar snow snowing snowy), {:weather, "snow"}},
    {~w(neblina névoa nevoeiro bruma fog foggy mist misty), {:weather, "fog"}},
    {~w(limpo ensolarado sol claro clear sunny), {:weather, "clear"}},
    {~w(noite noturno noturna night midnight lua moon), {:time, "night"}},
    {~w(amanhecer aurora madrugada alvorada dawn sunrise), {:time, "dawn"}},
    {~w(dia meio-dia tarde day noon daytime afternoon), {:time, "day"}},
    {~w(entardecer anoitecer crepúsculo pôr-do-sol poente dusk sunset twilight), {:time, "dusk"}},
    {~w(vento ventania brisa wind windy breeze gust), {:wind, 0.6}},
    {~w(velas vela candle candles), {:lights, "candles"}},
    {~w(tochas tocha torch torches fogueira fogo lareira fire bonfire fireplace), {:lights, "torches"}},
    {~w(lanternas lanterna luzes luz lamparinas lamps lanterns lights lamp), {:lights, "lanterns"}},
    {~w(pessoas pessoa gente aldeões aldeão personagens personagem npc npcs people person villagers villager characters character guards guard guarda guardas cavaleiros cavaleiro knights knight moradores morador), {:spawn, "people"}},
    {~w(pássaros pássaro passaros aves ave pombos corvos birds bird crows pigeons), {:spawn, "birds"}},
    {~w(vagalumes vaga-lumes vagalume fireflies firefly), {:spawn, "fireflies"}},
    {~w(brasas faíscas centelhas embers sparks), {:spawn, "embers"}},
    {~w(fumaça fumo smoke), {:spawn, "smoke"}},
    {~w(borboletas borboleta butterflies butterfly), {:spawn, "butterflies"}},
    {~w(folhas leaves), {:spawn, "leaves"}},
    {~w(orbite orbitar órbita gire girar orbit rotate), {:camera, "orbit"}},
    {~w(aproxime aproximar zoom-in perto approach closer push-in), {:camera, "push_in"}},
    {~w(afaste afastar longe zoom-out away pull-out), {:camera, "pull_out"}},
    {~w(parada parado fixa fixo still static), {:camera, "still"}},
    {~w(calmo calma tranquilo tranquila sereno calm quiet gentle peaceful), {:entropy, 0.15}},
    {~w(caótico caótica agitado agitada frenético chaotic busy wild lively), {:entropy, 0.9}},
    {~w(acene acenar aceno wave waving), {:animate, "wave"}},
    {~w(ande andar caminhe caminhar walk walking), {:animate, "walk"}},
    {~w(dance dançar dança dancing), {:animate, "dance"}},
    {~w(respire respirar breathe breathing), {:animate, "breathe"}},
    {~w(balance balançar sway swaying), {:animate, "sway"}},
    {~w(ciclo cycle), {:cycle, true}},
    {~w(rápido rápida depressa fast faster quick), {:speed, 1.8}},
    {~w(devagar lento lenta slow slower slowly), {:speed, 0.5}}
  ]

  @places %{"esquerda" => "left", "left" => "left", "direita" => "right", "right" => "right", "centro" => "center", "meio" => "center", "center" => "center",
            "middle" => "center", "fundo" => "back", "back" => "back", "frente" => "front", "front" => "front"}

  @stop ~w(e a o as os de da do das dos em no na nos nas com para por que um uma uns umas the and of in on with to at into a an some please por-favor favor bem muito mais menos pouco mais ponha coloque adicione faça deixe tornar torne quero want make add put let it is are be has have mostrar mostre tenha haja comece start agora now cena scene imagem image está esta este isso this that there aí lá ao à ir vá go goes indo andando caminhando passeando walking strolling figura desenho personagem-desenho depois então then after também also)

  @doc """
  A prompt as operations: `%{ops: [map], unknown: [word]}`. Negations
  ("sem chuva", "no rain", "pare", "stop") turn weather back to clear or
  remove what follows; numbers and quantity words set counts; "forte",
  "heavy", "leve", "light" set intensities; "até a esquerda", "to the
  door"… set a destination for the inhabitants.

  Since 0.12 the direction reaches **one inhabitant at a time and a
  moment in time**: a prompt is split into clauses (`,` `;` `then`
  `depois` `e então`); a clause that names an inhabitant — one created
  in the prompt ("a guard named Ana", "uma pessoa chamada Ana") or one
  already in the scene (`known`) — becomes an operation on it: `Ana walks
  to the door`, `Ana says "hello"`, `Ana waves / dances / sits / jumps /
  runs / stops`, `Ana follows Bento`, `Ana patrols`, `Ana flees`; and a
  clause that begins with a time ("after 5 seconds", "aos 10 s", "at
  12s", "em 3 segundos") becomes a keyframe (`at`) the engine plays then.
  """
  def direct(prompt, known \\ []) do
    clauses = String.split(prompt, ~r/\s*(?:[,;]|\bthen\b|\bdepois\b(?!\s+de\s+\d)|\be então\b|\band then\b)\s*/iu, trim: true)
    # a clause that is only a time ("after 2 seconds, Ana dances") times the clause after it
    {ops, unknown, _names, _last, _pending} =
      Enum.reduce(clauses, {[], [], MapSet.new(Enum.map(known, &String.downcase/1)), nil, nil}, fn clause, {ops, unk, names, last, pending} ->
        {at, body} = timing(clause)
        if String.trim(body) == "" and at != nil do
          {ops, unk, names, last, at}
        else
          at = at || pending
          {cops, cunk, names, last} = clause_ops(body, names, last)
          cops = if at, do: Enum.map(cops, &Map.put(&1, "at", at)), else: cops
          {ops ++ cops, unk ++ cunk, names, last, nil}
        end
      end)
    # one setting per scalar key among the untimed, untargeted operations (the last said wins)
    {plain, special} = Enum.split_with(ops, &(not Map.has_key?(&1, "at") and not Map.has_key?(&1, "npc") and not Map.has_key?(&1, "name")))
    %{ops: finish(plain, []).ops ++ special, unknown: Enum.uniq(unknown)}
  end


  defp timing(clause) do
    case Regex.run(~r/^\s*(?:after|depois\s+de|em|at|aos?|no\s+segundo|in)\s+(\d+(?:[.,]\d+)?)\s*(?:s|seg|segs|segundos?|seconds?|sec)?\b[:,]?\s*/iu, clause) do
      [whole, n] -> {elem(Float.parse(String.replace(n, ",", ".")), 0), String.replace_prefix(clause, whole, "")}
      nil -> {nil, clause}
    end
  end

  @actions [{~w(acena acenar aceno waves wave waving), "wave"}, {~w(dança dançar dances dance dancing), "dance"}, {~w(senta sentar sits sit sitting), "sit"},
            {~w(pula pular salta saltar jumps jump jumping), "jump"}, {~w(corre correr runs run running), "run"}, {~w(levanta levantar stands stand), "stand"}]
  @behaviours [{~w(para parar pare fica ficar stops stop stays stay waits wait espera esperar), "idle"}, {~w(patrulha patrulhar patrols patrol), "patrol"},
               {~w(passeia passear vagueia vaguear wanders wander roams roam), "wander"}, {~w(foge fugir flees flee), "flee"}]

  # pronouns refer to the inhabitant named last; a role noun with a verb ("a guard patrols") names one by its role
  @pronouns ~w(he she they him her ele ela eles elas)
  @roles ~w(guard knight person man woman child kid villager farmer soldier traveller traveler merchant guarda cavaleiro pessoa homem mulher criança menino menina morador moradora camponês camponesa soldado viajante mercador)
  @moves ~w(walks walk goes go heads head moves move strolls stroll vai ir anda andar caminha caminhar dirige-se)

  defp acting?(body, lw) do
    Regex.match?(~r/["“«]/u, body) or
      Enum.any?(lw, fn w -> w in @moves or w in ~w(says say diz dizer fala falar follows follow segue seguir) or
                            Enum.any?(@actions, fn {ws, _} -> w in ws end) or Enum.any?(@behaviours, fn {ws, _} -> w in ws end) end)
  end

  defp clause_ops(body, names, last) do
    # a new named inhabitant: "… named Ana", "… chamada Ana", "… called Ana"
    {created, body2, names} =
      case Regex.run(~r/\b(?:named|called|chamad[oa]|de\s+nome)\s+([\p{Lu}][\p{L}\-]*)/u, body) do
        [whole, name] -> {[%{"spawn" => "people", "name" => name, "count" => 1}], String.replace(body, whole, ""), MapSet.put(names, String.downcase(name))}
        nil -> {[], body, names}
      end

    words = body2 |> String.replace(~r/["“”«»].*?["“”«»]/u, " ") |> String.replace(~r/[.,;:!?()]/u, " ") |> String.split()
    lw = Enum.map(words, &String.downcase/1)
    created_name = case created do [%{"name" => n}] -> n; _ -> nil end
    target = Enum.find(words, &MapSet.member?(names, String.downcase(&1))) || (if last && Enum.any?(lw, &(&1 in @pronouns)), do: last)
    role = if created == [] and target == nil, do: Enum.find(lw, &(&1 in @roles))
    acts = acting?(body2, lw)

    cond do
      created_name && acts ->
        # "a knight named Arthur walks to the door": created, then directed
        {created ++ npc_ops(created_name, body2, words), [], names, created_name}

      created != [] ->
        # "add a guard named Ana" — the name was the point; any other words go through the vocabulary (minus the people word)
        rest = direct_words(Enum.reject(words, &(&1 in ~w(add a an um uma ponha coloque adicione))))
        {created ++ Enum.reject(rest.ops, &(&1["spawn"] == "people")), rest.unknown, names, created_name}

      target ->
        {npc_ops(target, body2, words), [], names, target}

      role && acts ->
        name = String.capitalize(role)
        {npc_ops(name, body2, words), [], MapSet.put(names, String.downcase(name)), name}

      true ->
        r = direct_words(words)
        {created ++ r.ops, r.unknown, names, last}
    end
  end

  defp npc_ops(name, body, words) do
    lw = Enum.map(words, &String.downcase/1)
    said = case Regex.run(~r/["“«](.+?)["”»]/u, body) do [_, q] -> q; nil -> nil end
    action = Enum.find_value(@actions, fn {ws, a} -> if Enum.any?(lw, &(&1 in ws)), do: a end)
    behaviour = Enum.find_value(@behaviours, fn {ws, b} -> if Enum.any?(lw, &(&1 in ws)), do: b end)
    place = Enum.find_value(lw, fn w -> Map.get(@places, w) || (if w in ~w(porta door entrada entrance), do: "door") || (if w in ~w(luz light fogo fire), do: "light") end)
    follow = case Regex.run(~r/\b(?:segue|seguir|follows|follow)\s+(?:a\s+|o\s+)?([\p{L}][\p{L}\-]*)/iu, body) do [_, who] -> who; nil -> nil end
    ops =
      [said && %{"npc" => name, "say" => said},
       follow && %{"npc" => name, "follow" => follow},
       place && follow == nil && %{"npc" => name, "goto" => place},
       action && %{"npc" => name, "action" => action},
       behaviour && follow == nil && place == nil && %{"npc" => name, "behavior" => behaviour}]
      |> Enum.filter(& &1)
    if ops == [], do: [%{"npc" => name, "behavior" => "wander"}], else: ops
  end

  defp direct_words(words) do
    lookup = for {ws, op} <- @vocab, w <- ws, into: %{}, do: {w, op}
    {ops, unknown, _} =
      Enum.reduce(Enum.map(words, &String.downcase/1), {[], [], %{count: nil, neg: false, intensity: nil, goto: false, just: false}}, fn w, {ops, unk, st} ->
        {ops2, unk2, st2} = step_word(w, ops, unk, st, lookup)
        {ops2, unk2, %{st2 | just: length(ops2) > length(ops) and st2.just != :keep}}
      end)
    %{ops: ops, unknown: unknown}
  end

  defp intensity(ops, unk, %{just: true} = st, v), do: {set_intensity(ops, v), unk, %{st | just: :keep}}
  defp intensity(ops, unk, st, v), do: {ops, unk, %{st | intensity: v}}

  defp step_word(w, ops, unk, st, lookup) do
        cond do
          w in ~w(sem no without pare parar stop remova remover remove tire tirar acabe) -> {ops, unk, %{st | neg: true}}
          # an intensity right after a weather or wind applies to it; before one, to the next
          w in ~w(forte fortes pesada pesado intensa intenso heavy strong intense) -> intensity(ops, unk, st, 1.0)
          w in ~w(leve leves fraca fraco suave light soft) -> intensity(ops, unk, st, 0.3)
          Map.has_key?(@nums, w) -> {ops, unk, %{st | count: @nums[w]}}
          Regex.match?(~r/^\d+$/, w) -> {ops, unk, %{st | count: min(String.to_integer(w), 60)}}
          w in ~w(até para to towards toward) -> {ops, unk, %{st | goto: true}}
          Map.has_key?(@places, w) and st.goto -> {ops ++ [%{"goto" => @places[w]}], unk, %{st | goto: false}}
          w in ~w(porta door entrada entrance) and st.goto -> {ops ++ [%{"goto" => "door"}], unk, %{st | goto: false}}
          w in ~w(luz light fogo fire) and st.goto -> {ops ++ [%{"goto" => "light"}], unk, %{st | goto: false}}
          Map.has_key?(lookup, w) -> {ops ++ [op(lookup[w], st)], unk, %{st | count: nil, neg: false, intensity: nil}}
          w in @stop or String.length(w) <= 2 -> {ops, unk, st}
          true -> {ops, unk ++ [w], st}
        end
  end

  defp finish(ops, unknown) do
    # one setting per scalar key (the last said wins); additions all kept
    scalar = ~w(weather time wind camera entropy animate speed cycle)
    {kept, _} =
      ops |> Enum.reverse() |> Enum.reduce({[], MapSet.new()}, fn op, {acc, seen} ->
        case Enum.find(scalar, &Map.has_key?(op, &1)) do
          nil -> {[op | acc], seen}
          k -> if MapSet.member?(seen, k), do: {acc, seen}, else: {[op | acc], MapSet.put(seen, k)}
        end
      end)

    %{ops: Enum.uniq(kept), unknown: Enum.uniq(unknown)}
  end

  defp op({:weather, kind}, %{neg: true}), do: %{"weather" => "clear", "was" => kind}
  defp op({:weather, kind}, st), do: %{"weather" => kind, "intensity" => st.intensity || 0.7}
  defp op({:spawn, kind}, %{neg: true}), do: %{"remove" => kind}
  defp op({:spawn, kind}, st), do: %{"spawn" => kind, "count" => st.count || default_count(kind)}
  defp op({:lights, kind}, %{neg: true}), do: %{"lights" => "off", "kind" => kind}
  defp op({:lights, kind}, _st), do: %{"lights" => "on", "kind" => kind}
  defp op({:wind, v}, %{neg: true}), do: (_ = v; %{"wind" => 0.0})
  defp op({:wind, v}, st), do: %{"wind" => st.intensity || v}
  defp op({:time, t}, _st), do: %{"time" => t}
  defp op({:camera, c}, _st), do: %{"camera" => c}
  defp op({:entropy, e}, _st), do: %{"entropy" => e}
  defp op({:animate, a}, %{neg: true}), do: %{"animate" => "none", "was" => a}
  defp op({:animate, a}, _st), do: %{"animate" => a}
  defp op({:cycle, v}, %{neg: true}), do: (_ = v; %{"cycle" => false})
  defp op({:cycle, v}, _st), do: %{"cycle" => v}
  defp op({:speed, s}, _st), do: %{"speed" => s}

  defp default_count("people"), do: 3
  defp default_count("birds"), do: 7
  defp default_count("fireflies"), do: 24
  defp default_count("butterflies"), do: 6
  defp default_count(_), do: 1

  # an intensity word after (or before) a weather/wind op applies to the last one
  defp set_intensity([], _), do: []

  defp set_intensity(ops, v) do
    {last, rest} = List.pop_at(ops, -1)
    cond do
      Map.has_key?(last, "intensity") -> rest ++ [Map.put(last, "intensity", v)]
      Map.has_key?(last, "wind") -> rest ++ [Map.put(last, "wind", v)]
      true -> ops
    end
  end

  # ============================================================== export

  @doc """
  A standalone HTML page that plays a scene offline: the engine (the
  console's, sliced between its markers) and the scene inlined. `title`
  names it.
  """
  def standalone(scene_json, title \\ "vapor — cena viva") do
    html = File.read!(Path.join([to_string(:code.priv_dir(:vapor)), "console", "index.html"]))
    [_, engine] = String.split(html, "/* scene-engine:start */", parts: 2)
    [engine, _] = String.split(engine, "/* scene-engine:end */", parts: 2)
    safe = String.replace(scene_json, "</", "<\\/")

    """
    <!doctype html>
    <html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>#{title |> String.replace("<", "&lt;")}</title>
    <style>html,body{margin:0;height:100%;background:#0b1516;color:#d8e6e3;font:14px system-ui,sans-serif}
    #stage{width:100vw;height:100vh;display:block;touch-action:none}
    .bar{position:fixed;left:16px;bottom:16px;display:flex;gap:8px;align-items:center;background:rgba(11,21,22,.72);padding:8px 12px;border-radius:10px;backdrop-filter:blur(6px)}
    .bar button{background:#17393a;color:#d8e6e3;border:0;border-radius:7px;padding:6px 10px;cursor:pointer}</style></head>
    <body><canvas id="stage"></canvas>
    <div class="bar"><button id="pp">⏸</button><span id="info">vapor · drag to look around, scroll to zoom</span></div>
    <script>
    #{engine}
    const SCENE = #{safe};
    const eng = SceneEngine.create(document.getElementById("stage"), SCENE, { fill: true });
    eng.play();
    document.getElementById("pp").onclick = (e) => { if (eng.playing) { eng.pause(); e.target.textContent = "▶"; } else { eng.play(); e.target.textContent = "⏸"; } };
    </script></body></html>
    """
  end

  @doc false
  def replay("scene", %{"prompt" => p}) when is_binary(p) and byte_size(p) <= 4000, do: {:ok, direct(p)}
  def replay(_, _), do: {:error, :bad_recipe}

end
