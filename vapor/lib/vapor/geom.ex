defmodule Vapor.Geom.Mesh do
  @moduledoc "A triangle mesh: `vertices` as a flat list `[x, y, z, …]` (binary64), `faces` as a flat list of vertex indices `[a, b, c, …]` (counter-clockwise outside), optional per-vertex `colors` `[r, g, b, …]` in [0, 1]."
  @enforce_keys [:vertices, :faces]
  defstruct [:vertices, :faces, colors: nil]
end

defmodule Vapor.Geom do
  @moduledoc """
  3-D from first principles: meshes from height maps and from signed
  distance fields, three file formats, and a deterministic renderer.

    * `heightmap/2` — an image becomes a relief (a grid of quads, two
      triangles each, closed underneath into a solid when `solid: true`).
    * `sdf/3` — an implicit surface `f(x, y, z) = 0` on a grid, by
      **marching tetrahedra** (each cube cut into six tetrahedra sharing the
      main diagonal): no 256-case table, no ambiguous cases, a watertight
      surface for a closed level set; vertices on shared edges are welded.
    * `glb/1`, `obj/1`, `ply/1` — glTF 2.0 binary (what three.js, Blender
      and every viewer open), Wavefront OBJ and binary PLY.
    * `render/2` — a z-buffered rasterizer in binary64 (perspective camera,
      Lambert shading plus ambient, the top-left fill rule), so the picture
      of a mesh is the same bits on every machine; `turntable/2` renders a
      camera orbit as a video.

  Measured in `test/vapor/geom_test.exs`: trimesh reads all three files
  back with the same vertices and faces; the marching-tetrahedra sphere is
  watertight and its volume and area converge to 4/3·πr³ and 4πr²; the
  rendered silhouette of a sphere covers the analytic disc.
  """
  import Bitwise
  alias Vapor.Geom.Mesh
  alias Vapor.Modal.Image
  alias Vapor.Studio.Video

  # ------------------------------------------------------------- builders --

  @doc "A relief from an image's luma: `width` × `depth` (units), heights × `height`. Options `solid` (true), `height` (0.2)."
  def heightmap(%Image{w: w, h: h} = img, opts \\ []) do
    zs = Keyword.get(opts, :height, 0.2)
    lum = fn x, y -> v = Image.at(img, x, y, 0); if img.c == 3, do: 0.2126 * v + 0.7152 * Image.at(img, x, y, 1) + 0.0722 * Image.at(img, x, y, 2), else: v end
    sx = 1.0 / max(w - 1, 1)
    sy = 1.0 / max(h - 1, 1) * (h - 1) / max(w - 1, 1)
    top = for y <- 0..(h - 1), x <- 0..(w - 1), do: [x * sx - 0.5, lum.(x, y) * zs, y * sy - 0.5 * (h - 1) / max(w - 1, 1)]
    id = fn x, y -> y * w + x end
    quads = for y <- 0..(h - 2), x <- 0..(w - 2), do: [id.(x, y), id.(x, y + 1), id.(x + 1, y), id.(x + 1, y), id.(x, y + 1), id.(x + 1, y + 1)]
    colors = for y <- 0..(h - 1), x <- 0..(w - 1), k <- 0..2, do: Image.at(img, x, y, if(img.c == 3, do: k, else: 0))

    if Keyword.get(opts, :solid, true) do
      n = w * h
      bottom = for [x, _y, z] <- top, do: [x, -0.02, z]
      b = fn i -> i + n end
      under = for y <- 0..(h - 2), x <- 0..(w - 2), do: [b.(id.(x, y)), b.(id.(x + 1, y)), b.(id.(x, y + 1)), b.(id.(x + 1, y)), b.(id.(x + 1, y + 1)), b.(id.(x, y + 1))]
      edge = fn pts -> pts |> Enum.chunk_every(2, 1, :discard) |> Enum.flat_map(fn [p, q] -> [p, q, b.(p), q, b.(q), b.(p)] end) end
      sides =
        edge.(for x <- 0..(w - 1), do: id.(x, 0)) ++ edge.(for y <- 0..(h - 1), do: id.(w - 1, y)) ++
          edge.(for x <- (w - 1)..0//-1, do: id.(x, h - 1)) ++ edge.(for y <- (h - 1)..0//-1, do: id.(0, y))

      %Mesh{vertices: List.flatten(top ++ bottom), faces: List.flatten(quads ++ under ++ sides), colors: colors ++ colors}
    else
      %Mesh{vertices: List.flatten(top), faces: List.flatten(quads), colors: colors}
    end
  end

  # the six tetrahedra of a cube, around the diagonal 0–6 (corners in bit order x, y, z)
  @tets [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]]

  @doc """
  The surface `f = 0` of `f` (a function of `{x, y, z}`, negative inside)
  over the box `[lo, hi]³` with `n` cells per side, by marching tetrahedra.
  """
  def sdf(f, n, {lo, hi} \\ {-1.0, 1.0}) do
    step = (hi - lo) / n
    pos = fn i, j, k -> {lo + i * step, lo + j * step, lo + k * step} end
    vals = for i <- 0..n, j <- 0..n, k <- 0..n, into: %{}, do: {{i, j, k}, f.(pos.(i, j, k))}
    corner = fn {i, j, k}, c -> {i + (c &&& 1), j + (c >>> 1 &&& 1), k + (c >>> 2 &&& 1)} end
    # corner order of the classic cube numbering: 0 (000) 1 (100) 2 (110) 3 (010) 4 (001) 5 (101) 6 (111) 7 (011)
    cube = [0, 1, 3, 2, 4, 5, 7, 6]

    tris =
      for i <- 0..(n - 1), j <- 0..(n - 1), k <- 0..(n - 1), tet <- @tets, tri <- tetra(Enum.map(tet, &corner.({i, j, k}, Enum.at(cube, &1))), vals), do: tri

    weld(tris, fn {a, b} -> interp(pos, vals, a, b) end)
  end

  # triangles of one tetrahedron as lists of edges {inside_corner, outside_corner};
  # the winding is fixed afterwards (weld/2), from the edge directions
  defp tetra(ps, vals) do
    case Enum.split_with(ps, &(vals[&1] < 0)) do
      {[], _} -> []
      {_, []} -> []
      {[a], [b, c, d]} -> [[{a, b}, {a, c}, {a, d}]]
      {[b, c, d], [a]} -> [[{b, a}, {c, a}, {d, a}]]
      # the four crossing edges, in cyclic order around the quad
      {[a, b], [c, d]} -> [[{a, c}, {a, d}, {b, d}], [{a, c}, {b, d}, {b, c}]]
    end
  end

  defp interp(pos, vals, a, b) do
    {va, vb} = {vals[a], vals[b]}
    t = va / (va - vb)
    {xa, ya, za} = apply_pos(pos, a)
    {xb, yb, zb} = apply_pos(pos, b)
    {xa + (xb - xa) * t, ya + (yb - ya) * t, za + (zb - za) * t}
  end

  defp apply_pos(pos, {i, j, k}), do: pos.(i, j, k)

  # one vertex per grid edge; each triangle wound so its normal points from inside to outside
  defp weld(tris, at) do
    {faces, _index, verts} =
      Enum.reduce(tris, {[], %{}, []}, fn tri, {faces, index, verts} ->
        {ids, index, verts} =
          Enum.reduce(tri, {[], index, verts}, fn {a, b}, {ids, index, verts} ->
            key = {a, b}
            case index do
              %{^key => i} -> {[i | ids], index, verts}
              _ -> (fn i -> {[i | ids], Map.put(index, key, i), [{at.({a, b}), a, b} | verts]} end).(map_size(index))
            end
          end)

        {[Enum.reverse(ids) | faces], index, verts}
      end)

    vt = verts |> Enum.reverse() |> List.to_tuple()
    pts = for {p, _, _} <- Tuple.to_list(vt), do: p

    faces =
      faces
      |> Enum.reverse()
      |> Enum.map(fn [i, j, k] = f ->
        {p, a, b} = elem(vt, i)
        {q, _, _} = elem(vt, j)
        {r, _, _} = elem(vt, k)
        nrm = cross(sub(q, p), sub(r, p))
        # the edge's outside corner minus its inside corner points outward
        out = sub(grid(b), grid(a))
        if dot(nrm, out) >= 0, do: f, else: [i, k, j]
      end)

    %Mesh{vertices: pts |> Enum.flat_map(&Tuple.to_list/1), faces: List.flatten(faces)}
  end

  defp grid({i, j, k}), do: {i * 1.0, j * 1.0, k * 1.0}

  # ----------------------------------------------------------- measures --

  @doc "Enclosed volume (divergence theorem; positive for outward winding)."
  def volume(%Mesh{} = m), do: m |> triangles() |> Enum.reduce(0.0, fn {a, b, c}, s -> s + dot(a, cross(b, c)) / 6.0 end)

  @doc "Surface area."
  def area(%Mesh{} = m), do: m |> triangles() |> Enum.reduce(0.0, fn {a, b, c}, s -> s + norm(cross(sub(b, a), sub(c, a))) / 2.0 end)

  @doc "Whether every edge is shared by exactly two faces, in opposite directions (a closed, consistently oriented surface)."
  def watertight?(%Mesh{faces: f}) do
    edges = f |> Enum.chunk_every(3) |> Enum.flat_map(fn [a, b, c] -> [{a, b}, {b, c}, {c, a}] end)
    counts = Enum.frequencies(edges)
    Enum.all?(counts, fn {{a, b}, n} -> n == 1 and Map.get(counts, {b, a}) == 1 end)
  end

  defp triangles(%Mesh{vertices: v, faces: f}) do
    vt = v |> Enum.chunk_every(3) |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    f |> Enum.chunk_every(3) |> Enum.map(fn [a, b, c] -> {elem(vt, a), elem(vt, b), elem(vt, c)} end)
  end

  defp sub({a, b, c}, {d, e, f}), do: {a - d, b - e, c - f}
  defp cross({a, b, c}, {d, e, f}), do: {b * f - c * e, c * d - a * f, a * e - b * d}
  defp dot({a, b, c}, {d, e, f}), do: a * d + b * e + c * f
  defp norm(v), do: :math.sqrt(dot(v, v))

  # -------------------------------------------------------------- formats --

  @doc "Wavefront OBJ (vertices and faces, 1-based)."
  def obj(%Mesh{vertices: v, faces: f}) do
    vs = v |> Enum.chunk_every(3) |> Enum.map(fn [x, y, z] -> "v #{fmt(x)} #{fmt(y)} #{fmt(z)}\n" end)
    fs = f |> Enum.chunk_every(3) |> Enum.map(fn [a, b, c] -> "f #{a + 1} #{b + 1} #{c + 1}\n" end)
    IO.iodata_to_binary(["# vapor\n", vs, fs])
  end

  defp fmt(x), do: :erlang.float_to_binary(x * 1.0, [:short])

  @doc "Binary little-endian PLY (float32 positions, optional uchar colours, int32 faces)."
  def ply(%Mesh{vertices: v, faces: f, colors: c}) do
    nv = div(length(v), 3)
    nf = div(length(f), 3)
    color_props = if c, do: "property uchar red\nproperty uchar green\nproperty uchar blue\n", else: ""
    header = "ply\nformat binary_little_endian 1.0\nelement vertex #{nv}\nproperty float x\nproperty float y\nproperty float z\n#{color_props}element face #{nf}\nproperty list uchar int vertex_indices\nend_header\n"
    cols = if c, do: c |> Enum.map(&round(min(1.0, max(0.0, &1)) * 255)) |> Enum.chunk_every(3), else: List.duplicate(nil, nv)

    verts =
      v
      |> Enum.chunk_every(3)
      |> Enum.zip(cols)
      |> Enum.map(fn {[x, y, z], col} -> <<x::float-32-little, y::float-32-little, z::float-32-little>> <> if(col, do: :binary.list_to_bin(col), else: <<>>) end)

    faces = f |> Enum.chunk_every(3) |> Enum.map(fn [a, b, cc] -> <<3, a::32-little-signed, b::32-little-signed, cc::32-little-signed>> end)
    IO.iodata_to_binary([header, verts, faces])
  end

  @doc "glTF 2.0 binary: one mesh, float32 positions (with bounds), uint32 indices, optional vertex colours."
  def glb(%Mesh{vertices: v, faces: f, colors: c}) do
    nv = div(length(v), 3)
    pos = for x <- v, into: <<>>, do: <<x::float-32-little>>
    idx = for i <- f, into: <<>>, do: <<i::32-little>>
    col = if c, do: for(x <- c, into: <<>>, do: <<min(1.0, max(0.0, x))::float-32-little>>), else: <<>>
    pad4 = fn b -> b <> :binary.copy(<<0>>, rem(4 - rem(byte_size(b), 4), 4)) end
    {pos, idx, col} = {pad4.(pos), pad4.(idx), pad4.(col)}
    xyz = Enum.chunk_every(v, 3)
    mins = for k <- 0..2, do: xyz |> Enum.map(&Enum.at(&1, k)) |> Enum.min() |> f32
    maxs = for k <- 0..2, do: xyz |> Enum.map(&Enum.at(&1, k)) |> Enum.max() |> f32
    views = [%{"buffer" => 0, "byteOffset" => 0, "byteLength" => nv * 12, "target" => 34962},
             %{"buffer" => 0, "byteOffset" => byte_size(pos), "byteLength" => length(f) * 4, "target" => 34963}] ++
            if(c, do: [%{"buffer" => 0, "byteOffset" => byte_size(pos) + byte_size(idx), "byteLength" => nv * 12, "target" => 34962}], else: [])
    accessors = [%{"bufferView" => 0, "componentType" => 5126, "count" => nv, "type" => "VEC3", "min" => mins, "max" => maxs},
                 %{"bufferView" => 1, "componentType" => 5125, "count" => length(f), "type" => "SCALAR"}] ++
                if(c, do: [%{"bufferView" => 2, "componentType" => 5126, "count" => nv, "type" => "VEC3"}], else: [])
    attrs = if c, do: %{"POSITION" => 0, "COLOR_0" => 2}, else: %{"POSITION" => 0}
    bin = pos <> idx <> col

    json =
      Vapor.JSON.encode(%{"asset" => %{"version" => "2.0", "generator" => "vapor"}, "scene" => 0, "scenes" => [%{"nodes" => [0]}], "nodes" => [%{"mesh" => 0}],
                          "meshes" => [%{"primitives" => [%{"attributes" => attrs, "indices" => 1, "mode" => 4}]}], "buffers" => [%{"byteLength" => byte_size(bin)}],
                          "bufferViews" => views, "accessors" => accessors})

    json = json <> :binary.copy(" ", rem(4 - rem(byte_size(json), 4), 4))
    total = 12 + 8 + byte_size(json) + 8 + byte_size(bin)
    <<"glTF", 2::32-little, total::32-little, byte_size(json)::32-little, "JSON", json::binary, byte_size(bin)::32-little, "BIN\0", bin::binary>>
  end

  defp f32(x), do: Vapor.F32.to_float(Vapor.F32.from_float(x * 1.0))

  # ------------------------------------------------------------- renderer --

  @doc """
  Render a mesh to an image. Options: `width`, `height` (256), `azimuth`,
  `elevation` (degrees; 30, 20), `distance` (2.6), `fov` (40°), `light`
  (direction, default from the camera's upper left), `color` (when the mesh
  has none), `background`.
  """
  def render(%Mesh{} = m, opts \\ []) do
    {w, h} = {Keyword.get(opts, :width, 256), Keyword.get(opts, :height, 256)}
    az = Keyword.get(opts, :azimuth, 30.0) * :math.pi() / 180
    el = Keyword.get(opts, :elevation, 20.0) * :math.pi() / 180
    dist = Keyword.get(opts, :distance, 2.6)
    f = 1.0 / Vapor.CR.sin_f64(Keyword.get(opts, :fov, 40.0) * :math.pi() / 360) * Vapor.CR.cos_f64(Keyword.get(opts, :fov, 40.0) * :math.pi() / 360)
    bg = Keyword.get(opts, :background, {0.95, 0.96, 0.97})
    base = Keyword.get(opts, :color, {0.27, 0.55, 0.62})
    {ca, sa, ce, se} = {Vapor.CR.cos_f64(az), Vapor.CR.sin_f64(az), Vapor.CR.cos_f64(el), Vapor.CR.sin_f64(el)}
    # camera on a sphere looking at the origin; world → camera
    cam = fn {x, y, z} ->
      {x1, z1} = {ca * x - sa * z, sa * x + ca * z}
      {y2, z2} = {ce * y + se * z1, -se * y + ce * z1}
      {x1, y2, z2 + dist}
    end

    # towards the light, in camera space (camera looks down +z, y up)
    light = normalize(Keyword.get(opts, :light, {-0.4, 0.6, -0.7}))
    vt = m.vertices |> Enum.chunk_every(3) |> Enum.map(fn [x, y, z] -> cam.({x, y, z}) end) |> List.to_tuple()
    ct = if m.colors, do: m.colors |> Enum.chunk_every(3) |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
    proj = fn {x, y, z} -> {(x * f / z * 0.5 + 0.5) * w, (0.5 - y * f / z * 0.5) * h, z} end

    {zbuf, cbuf} =
      m.faces
      |> Enum.chunk_every(3)
      |> Enum.reduce({%{}, %{}}, fn [i, j, k], {zb, cb} ->
        {p0, p1, p2} = {elem(vt, i), elem(vt, j), elem(vt, k)}
        nrm = normalize(cross(sub(p1, p0), sub(p2, p0)))
        if elem(p0, 2) <= 0.05 or elem(p1, 2) <= 0.05 or elem(p2, 2) <= 0.05 do
          {zb, cb}
        else
          # two-sided: the normal that faces the camera (at the origin), then Lambert + ambient
          front = if dot(nrm, p0) > 0, do: scale_v(nrm, -1.0), else: nrm
          shade = 0.25 + 0.75 * max(0.0, dot(front, light))
          col = if ct, do: avg3(elem(ct, i), elem(ct, j), elem(ct, k)), else: base
          raster(proj.(p0), proj.(p1), proj.(p2), w, h, zb, cb, scale3(col, shade))
        end
      end)

    _ = zbuf
    vals = for y <- 0..(h - 1), x <- 0..(w - 1), c = Map.get(cbuf, y * w + x, bg), k <- 0..2, do: elem(c, k)
    %Image{w: w, h: h, c: 3, px: List.to_tuple(vals)}
  end

  defp scale_v({a, b, c}, k), do: {a * k, b * k, c * k}
  defp avg3({a, b, c}, {d, e, f}, {g, h, i}), do: {(a + d + g) / 3, (b + e + h) / 3, (c + f + i) / 3}
  defp scale3({a, b, c}, s), do: {min(1.0, a * s), min(1.0, b * s), min(1.0, c * s)}
  defp normalize(v), do: (fn n -> if n == 0.0, do: v, else: (fn {a, b, c} -> {a / n, b / n, c / n} end).(v) end).(norm(v))

  # edge functions over the bounding box; pixel centres; top-left rule; nearest z wins (ties: first drawn)
  defp raster({x0, y0, z0}, {x1, y1, z1}, {x2, y2, z2}, w, h, zb, cb, col) do
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    if area == 0.0 do
      {zb, cb}
    else
      {x1, y1, z1, x2, y2, z2, area} = if area < 0, do: {x2, y2, z2, x1, y1, z1, -area}, else: {x1, y1, z1, x2, y2, z2, area}
      xs = [x0, x1, x2]
      ys = [y0, y1, y2]
      {xa, xb} = {max(0, trunc(Float.floor(Enum.min(xs)))), min(w - 1, trunc(Float.ceil(Enum.max(xs))))}
      {ya, yb} = {max(0, trunc(Float.floor(Enum.min(ys)))), min(h - 1, trunc(Float.ceil(Enum.max(ys))))}
      edge = fn ax, ay, bx, by, px, py -> (bx - ax) * (py - ay) - (by - ay) * (px - ax) end
      tl = fn ax, ay, bx, by -> (ay == by and bx < ax) or by < ay end

      Enum.reduce(ya..yb//1, {zb, cb}, fn py, acc ->
        Enum.reduce(xa..xb//1, acc, fn px, {zb, cb} ->
          {cx, cy} = {px + 0.5, py + 0.5}
          w0 = edge.(x1, y1, x2, y2, cx, cy)
          w1 = edge.(x2, y2, x0, y0, cx, cy)
          w2 = edge.(x0, y0, x1, y1, cx, cy)
          inside = (w0 > 0 or (w0 == 0 and tl.(x1, y1, x2, y2))) and (w1 > 0 or (w1 == 0 and tl.(x2, y2, x0, y0))) and (w2 > 0 or (w2 == 0 and tl.(x0, y0, x1, y1)))

          if inside do
            z = (w0 * z0 + w1 * z1 + w2 * z2) / area
            key = py * w + px
            if z < Map.get(zb, key, :infinity), do: {Map.put(zb, key, z), Map.put(cb, key, col)}, else: {zb, cb}
          else
            {zb, cb}
          end
        end)
      end)
    end
  end

  @doc "A camera orbit as a video: `frames` views around the vertical axis. Options as `render/2`, plus `frames` (24), `fps` (12)."
  def turntable(%Mesh{} = m, opts \\ []) do
    n = Keyword.get(opts, :frames, 24)
    a0 = Keyword.get(opts, :azimuth, 30.0)
    frames = for i <- 0..(n - 1), do: render(m, Keyword.put(opts, :azimuth, a0 + 360.0 * i / n))
    %Video{fps: Keyword.get(opts, :fps, 12) * 1.0, frames: frames}
  end
end
