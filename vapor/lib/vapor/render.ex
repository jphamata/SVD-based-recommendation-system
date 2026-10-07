defmodule Vapor.Render do
  @moduledoc """
  Physically based rendering (docs/RENDER.md): a Monte Carlo path tracer
  — the light-transport equation solved by sampling, the way film
  renderers make photographs — that is also the **reference** for the
  console's progressive GPU tracer (the same scene format, the same
  materials, compared pixel statistics).

  Materials: Lambertian `diffuse` (cosine-weighted sampling), `metal`
  (mirror with roughness), `glass` (Fresnel–Schlick reflection and Snell
  refraction, total internal reflection), `emit` (area lights). Lights:
  emissive objects, a sky (uniform or a vertical gradient) and a sun
  (directional, sampled explicitly — next-event estimation). Russian
  roulette ends paths without bias. Rows are spread over every scheduler;
  each pixel's random stream is a function of (seed, pixel, sample), so
  the image is the same on any machine and any number of cores.

  Correctness is tested the way renderer authors test: the **white
  furnace** (an object of albedo a in a uniform environment of radiance L
  must show exactly a·L — energy conservation, no sampling bias; a
  deliberately wrong estimator, the control, fails it) and the **N^−½
  convergence** of the error with samples per pixel.

      camera pos=0,1.2,4.5 look=0,0.8,0 fov=45
      sky top=0.55,0.7,1.0 bottom=1,1,1
      sun dir=0.4,1,0.3 color=1,0.95,0.85 power=2.5
      plane y=0 mat=diffuse albedo=0.75,0.75,0.75 checker=0.5
      sphere c=0,0.8,0 r=0.8 mat=glass ior=1.5
      sphere c=-1.7,0.6,-0.5 r=0.6 mat=metal albedo=0.95,0.75,0.4 rough=0.08
      sphere c=1.6,0.5,0.3 r=0.5 mat=diffuse albedo=0.8,0.2,0.15
      box min=-0.4,0,-2 max=0.4,1.2,-1.4 mat=diffuse albedo=0.3,0.5,0.8
      sphere c=0,4,0 r=0.5 mat=emit color=1,0.9,0.8 power=12
  """
  import Bitwise

  # ================================================================ scene

  @doc "Parse a scene description: `{:ok, scene}` or `{:error, why}`."
  def parse(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.reduce_while({:ok, %{camera: %{pos: {0.0, 1.0, 4.0}, look: {0.0, 0.5, 0.0}, fov: 45.0, aperture: 0.0, focus: nil}, sky: %{top: {0.5, 0.7, 1.0}, bottom: {1.0, 1.0, 1.0}},
                                    sun: nil, objects: [], exposure: 1.0}}, fn {raw, n}, {:ok, acc} ->
      l = raw |> String.split("#", parts: 2) |> hd() |> String.trim()
      case String.split(l) do
        [] -> {:cont, {:ok, acc}}
        [kind | kvs] ->
          case kv(kvs) do
            {:ok, o} ->
              case stmt(kind, o, acc) do
                {:ok, acc} -> {:cont, {:ok, acc}}
                {:error, w} -> {:halt, {:error, "line #{n}: #{w}"}}
              end
            {:error, w} -> {:halt, {:error, "line #{n}: #{w}"}}
          end
      end
    end)
  end

  defp kv(list) do
    Enum.reduce_while(list, {:ok, %{}}, fn item, {:ok, m} ->
      case String.split(item, "=", parts: 2) do
        [k, v] ->
          nums = String.split(v, ",")
          parsed = Enum.map(nums, &Float.parse/1)
          cond do
            Enum.all?(parsed, &match?({_, ""}, &1)) ->
              vals = Enum.map(parsed, &elem(&1, 0))
              {:cont, {:ok, Map.put(m, k, case vals do [x] -> x; [x, y, z] -> {x, y, z}; _ -> vals end)}}
            k == "mat" -> {:cont, {:ok, Map.put(m, k, v)}}
            true -> {:halt, {:error, "#{k}=#{v}: a number or x,y,z expected"}}
          end
        _ -> {:halt, {:error, "expected key=value: #{item}"}}
      end
    end)
  end

  defp v3(x) when is_number(x), do: {x * 1.0, x * 1.0, x * 1.0}
  defp v3({_, _, _} = v), do: v
  defp v3(_), do: nil

  defp material(o) do
    kind = o["mat"] || "diffuse"
    if kind not in ["diffuse", "metal", "glass", "emit"], do: throw({:scene, "mat: diffuse, metal, glass or emit"})
    %{kind: kind, albedo: v3(o["albedo"]) || {0.8, 0.8, 0.8}, rough: o["rough"] || 0.0, ior: o["ior"] || 1.5,
      emit: if(kind == "emit", do: mul(v3(o["color"]) || {1.0, 1.0, 1.0}, o["power"] || 1.0), else: {0.0, 0.0, 0.0}), checker: o["checker"]}
  end

  defp stmt("camera", o, acc), do: {:ok, %{acc | camera: %{acc.camera | pos: v3(o["pos"]) || acc.camera.pos, look: v3(o["look"]) || acc.camera.look, fov: o["fov"] || acc.camera.fov, aperture: o["aperture"] || 0.0, focus: o["focus"]}}}
  defp stmt("sky", o, acc), do: {:ok, %{acc | sky: %{top: v3(o["top"] || o["color"]) || acc.sky.top, bottom: v3(o["bottom"] || o["color"]) || acc.sky.bottom}}}
  defp stmt("sun", o, acc), do: {:ok, %{acc | sun: %{dir: norm(v3(o["dir"]) || {0.3, 1.0, 0.2}), color: mul(v3(o["color"]) || {1.0, 1.0, 1.0}, o["power"] || 2.0), size: o["size"] || 0.03}}}
  defp stmt("exposure", o, acc), do: {:ok, %{acc | exposure: o["value"] || 1.0}}
  defp stmt("sphere", o, acc), do: add(acc, %{type: :sphere, c: v3(o["c"]) || {0.0, 0.0, 0.0}, r: o["r"] || 1.0, m: material(o)})
  defp stmt("plane", o, acc), do: add(acc, %{type: :plane, y: o["y"] || 0.0, m: material(o)})
  defp stmt("box", o, acc) do
    {a, b} = {v3(o["min"]) || {-0.5, -0.5, -0.5}, v3(o["max"]) || {0.5, 0.5, 0.5}}
    add(acc, %{type: :box, min: vmin(a, b), max: vmax(a, b), m: material(o)})
  end
  defp stmt(k, _o, _acc), do: {:error, "unknown statement #{k} (camera, sky, sun, exposure, sphere, plane, box)"}

  defp add(acc, obj), do: (if length(acc.objects) >= 200, do: {:error, "at most 200 objects"}, else: {:ok, %{acc | objects: acc.objects ++ [obj]}})

  # ============================================================== vectors

  defp add3({a, b, c}, {x, y, z}), do: {a + x, b + y, c + z}
  defp sub({a, b, c}, {x, y, z}), do: {a - x, b - y, c - z}
  defp mul({a, b, c}, s) when is_number(s), do: {a * s, b * s, c * s}
  defp mul({a, b, c}, {x, y, z}), do: {a * x, b * y, c * z}
  defp dot({a, b, c}, {x, y, z}), do: a * x + b * y + c * z
  defp cross({a, b, c}, {x, y, z}), do: {b * z - c * y, c * x - a * z, a * y - b * x}
  defp norm(v), do: mul(v, 1 / :math.sqrt(max(dot(v, v), 1.0e-300)))
  defp vmin({a, b, c}, {x, y, z}), do: {min(a, x), min(b, y), min(c, z)}
  defp vmax({a, b, c}, {x, y, z}), do: {max(a, x), max(b, y), max(c, z)}
  defp lum({r, g, b}), do: 0.2126 * r + 0.7152 * g + 0.0722 * b

  # ============================================================ intersection

  defp hit(objects, o, d), do: hit(objects, o, d, 1.0e30, nil)
  defp hit([], _o, _d, t, best), do: if(best, do: {t, best}, else: nil)
  defp hit([obj | rest], o, d, tmax, best) do
    case isect(obj, o, d) do
      {t, n} when t > 1.0e-4 and t < tmax -> hit(rest, o, d, t, {obj, n})
      _ -> hit(rest, o, d, tmax, best)
    end
  end

  defp isect(%{type: :sphere, c: c, r: r}, o, d) do
    oc = sub(o, c)
    b = dot(oc, d)
    q = dot(oc, oc) - r * r
    disc = b * b - q
    if disc < 0 do
      nil
    else
      s = :math.sqrt(disc)
      t = if -b - s > 1.0e-4, do: -b - s, else: -b + s
      if t > 1.0e-4, do: {t, mul(sub(add3(o, mul(d, t)), c), 1 / r)}, else: nil
    end
  end

  defp isect(%{type: :plane, y: y}, {_, oy, _}, {_, dy, _}) do
    if abs(dy) < 1.0e-12, do: nil, else: (t = (y - oy) / dy; if t > 1.0e-4, do: {t, {0.0, if(dy < 0, do: 1.0, else: -1.0), 0.0}}, else: nil)
  end

  defp isect(%{type: :box, min: {x0, y0, z0}, max: {x1, y1, z1}}, {ox, oy, oz}, {dx, dy, dz}) do
    {tx0, tx1} = slab(ox, dx, x0, x1)
    {ty0, ty1} = slab(oy, dy, y0, y1)
    {tz0, tz1} = slab(oz, dz, z0, z1)
    tn = Enum.max([tx0, ty0, tz0])
    tf = Enum.min([tx1, ty1, tz1])
    cond do
      tn > tf or tf < 1.0e-4 -> nil
      true ->
        t = if tn > 1.0e-4, do: tn, else: tf
        p = {ox + t * dx, oy + t * dy, oz + t * dz}
        {t, box_normal(p, {x0, y0, z0}, {x1, y1, z1})}
    end
  end

  defp slab(o, d, a, b) when abs(d) < 1.0e-12, do: if(o >= a and o <= b, do: {-1.0e30, 1.0e30}, else: {1.0e30, -1.0e30})
  defp slab(o, d, a, b), do: (t1 = (a - o) / d; t2 = (b - o) / d; {min(t1, t2), max(t1, t2)})

  defp box_normal({px, py, pz}, {x0, y0, z0}, {x1, y1, z1}) do
    cands = [{abs(px - x0), {-1.0, 0.0, 0.0}}, {abs(px - x1), {1.0, 0.0, 0.0}}, {abs(py - y0), {0.0, -1.0, 0.0}}, {abs(py - y1), {0.0, 1.0, 0.0}}, {abs(pz - z0), {0.0, 0.0, -1.0}}, {abs(pz - z1), {0.0, 0.0, 1.0}}]
    cands |> Enum.min_by(&elem(&1, 0)) |> elem(1)
  end

  # ================================================================ random

  # a stateless hash RNG (PCG-like): uniform in [0, 1) from (pixel stream, index)
  defp rnd(stream, k) do
    x = (stream * 0x9E3779B1 + k * 0x85EBCA77 + 0x632BE5AB) &&& 0xFFFFFFFF
    x = bxor(x, x >>> 16) * 0x7FEB352D &&& 0xFFFFFFFF
    x = bxor(x, x >>> 15) * 0x846CA68B &&& 0xFFFFFFFF
    x = bxor(x, x >>> 16)
    x / 4_294_967_296
  end

  # ================================================================ shading

  defp sky(scene, d) do
    {_, y, _} = d
    t = 0.5 * (y + 1)
    add3(mul(scene.sky.bottom, 1 - t), mul(scene.sky.top, t))
  end

  defp onb(n) do
    a = if abs(elem(n, 0)) > 0.9, do: {0.0, 1.0, 0.0}, else: {1.0, 0.0, 0.0}
    t = norm(cross(a, n))
    {t, cross(n, t)}
  end

  defp cosine_dir(n, u1, u2) do
    r = :math.sqrt(u1)
    phi = 2 * :math.pi() * u2
    {t, b} = onb(n)
    norm(add3(add3(mul(t, r * :math.cos(phi)), mul(b, r * :math.sin(phi))), mul(n, :math.sqrt(max(0.0, 1 - u1)))))
  end

  # the control of the furnace test: directions uniform on the hemisphere, weighted as if cosine-distributed (biased)
  defp uniform_dir(n, u1, u2) do
    z = u1
    r = :math.sqrt(max(0.0, 1 - z * z))
    phi = 2 * :math.pi() * u2
    {t, b} = onb(n)
    norm(add3(add3(mul(t, r * :math.cos(phi)), mul(b, r * :math.sin(phi))), mul(n, z)))
  end

  defp albedo(%{albedo: a, checker: nil}, _p), do: a
  defp albedo(%{albedo: a, checker: s}, {x, _, z}) do
    if rem(floor(x / s) + floor(z / s), 2) == 0, do: a, else: mul(a, 0.35)
  end

  defp reflect(d, n), do: sub(d, mul(n, 2 * dot(d, n)))

  @doc false
  # radiance along a ray; `opts.biased` swaps in the wrong estimator (the furnace control)
  def radiance(scene, o, d, stream, opts \\ %{}) do
    trace(scene, o, d, stream, 0, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, true, opts)
  end

  defp trace(scene, o, d, stream, depth, thr, acc, specular_prev, opts) do
    case hit(scene.objects, o, d) do
      nil ->
        env = sky(scene, d)
        # the sun seen directly (or by a specular path), not double-counted after a diffuse bounce
        sun = if scene.sun && specular_prev && dot(d, scene.sun.dir) > :math.cos(scene.sun.size), do: mul(scene.sun.color, 1 / (2 * :math.pi() * (1 - :math.cos(scene.sun.size)))), else: {0.0, 0.0, 0.0}
        add3(acc, mul(thr, add3(env, sun)))

      {t, {obj, n}} ->
        p = add3(o, mul(d, t))
        m = obj.m
        acc = add3(acc, mul(thr, m.emit))
        k = depth * 8
        cond do
          m.kind == "emit" -> acc
          depth >= Map.get(opts, :max_depth, 12) -> acc
          true ->
            {thr, acc, survive} = roulette(thr, acc, depth, stream, k)
            if not survive do
              acc
            else
              case m.kind do
                "diffuse" ->
                  a = albedo(m, p)
                  nn = if dot(n, d) > 0, do: mul(n, -1.0), else: n
                  acc = add3(acc, sun_light(scene, p, nn, mul(thr, a), stream, k))
                  nd = if opts[:biased], do: uniform_dir(nn, rnd(stream, k + 1), rnd(stream, k + 2)), else: cosine_dir(nn, rnd(stream, k + 1), rnd(stream, k + 2))
                  trace(scene, add3(p, mul(nn, 1.0e-4)), nd, stream, depth + 1, mul(thr, a), acc, false, opts)

                "metal" ->
                  nn = if dot(n, d) > 0, do: mul(n, -1.0), else: n
                  r = reflect(d, nn)
                  r = if m.rough > 0, do: norm(add3(r, mul(cosine_dir(nn, rnd(stream, k + 1), rnd(stream, k + 2)), m.rough))), else: r
                  if dot(r, nn) <= 0, do: acc, else: trace(scene, add3(p, mul(nn, 1.0e-4)), r, stream, depth + 1, mul(thr, m.albedo), acc, m.rough < 0.2, opts)

                "glass" ->
                  {nn, eta, cosi} = if dot(d, n) < 0, do: {n, 1 / m.ior, -dot(d, n)}, else: {mul(n, -1.0), m.ior, dot(d, n)}
                  k2 = 1 - eta * eta * (1 - cosi * cosi)
                  f0 = ((1 - m.ior) / (1 + m.ior)) ** 2
                  fres = f0 + (1 - f0) * :math.pow(1 - cosi, 5)
                  if k2 < 0 or rnd(stream, k + 3) < fres do
                    trace(scene, add3(p, mul(nn, 1.0e-4)), reflect(d, nn), stream, depth + 1, thr, acc, true, opts)
                  else
                    tdir = norm(add3(mul(d, eta), mul(nn, eta * cosi - :math.sqrt(k2))))
                    trace(scene, sub(p, mul(nn, 1.0e-4)), tdir, stream, depth + 1, mul(thr, m.albedo), acc, true, opts)
                  end
              end
            end
        end
    end
  end

  defp roulette(thr, acc, depth, stream, k) do
    if depth < 3 do
      {thr, acc, true}
    else
      q = min(0.95, max(lum(thr), 0.05))
      if rnd(stream, k + 4) < q, do: {mul(thr, 1 / q), acc, true}, else: {thr, acc, false}
    end
  end

  # next-event estimation toward the sun: its cone sampled, shadow ray, Lambertian weight
  defp sun_light(%{sun: nil}, _p, _n, _w, _s, _k), do: {0.0, 0.0, 0.0}
  defp sun_light(scene, p, n, w, stream, k) do
    sun = scene.sun
    l = norm(add3(sun.dir, mul(cosine_dir(sun.dir, rnd(stream, k + 5), rnd(stream, k + 6)), sun.size)))
    c = dot(n, l)
    if c <= 0 or hit(scene.objects, add3(p, mul(n, 1.0e-4)), l) != nil, do: {0.0, 0.0, 0.0}, else: mul(mul(w, sun.color), c / :math.pi())
  end

  # ================================================================ camera

  defp camera_ray(cam, w, h, px, py, stream, s) do
    fwd = norm(sub(cam.look, cam.pos))
    right = norm(cross(fwd, {0.0, 1.0, 0.0}))
    up = cross(right, fwd)
    th = :math.tan(cam.fov * :math.pi() / 360)
    aspect = w / h
    u = ((px + rnd(stream, s * 101 + 9)) / w * 2 - 1) * th * aspect
    v = (1 - (py + rnd(stream, s * 101 + 10)) / h * 2) * th
    d = norm(add3(add3(fwd, mul(right, u)), mul(up, v)))
    if cam.aperture > 0 do
      focus = cam.focus || :math.sqrt(dot(sub(cam.look, cam.pos), sub(cam.look, cam.pos)))
      target = add3(cam.pos, mul(d, focus))
      r = cam.aperture * :math.sqrt(rnd(stream, s * 101 + 11))
      a = 2 * :math.pi() * rnd(stream, s * 101 + 12)
      o = add3(add3(cam.pos, mul(right, r * :math.cos(a))), mul(up, r * :math.sin(a)))
      {o, norm(sub(target, o))}
    else
      {cam.pos, d}
    end
  end

  # ================================================================ render

  @doc """
  Render: `%{w, h, spp, linear (HDR rows of {r, g, b}), png, ms, rays}`.
  Options: `width` (160), `height` (100), `spp` (16), `seed` (1), `biased`
  (the furnace control).
  """
  def render(scene, opts \\ []) do
    w = Keyword.get(opts, :width, 160) |> max(1) |> min(1920)
    h = Keyword.get(opts, :height, 100) |> max(1) |> min(1080)
    spp = Keyword.get(opts, :spp, 16) |> max(1) |> min(4096)
    seed = Keyword.get(opts, :seed, 1)
    o = %{biased: Keyword.get(opts, :biased, false), max_depth: Keyword.get(opts, :max_depth, 12)}
    t0 = System.monotonic_time(:millisecond)

    rows =
      Vapor.Play.pmap(Enum.to_list(0..(h - 1)), fn y ->
        for x <- 0..(w - 1) do
          stream = seed * 1_000_003 + y * w + x
          sum = Enum.reduce(0..(spp - 1), {0.0, 0.0, 0.0}, fn s, acc ->
            {ro, rd} = camera_ray(scene.camera, w, h, x, y, stream, s)
            add3(acc, radiance(scene, ro, rd, stream * 4099 + s, o))
          end)
          mul(sum, 1 / spp)
        end
      end)

    %{w: w, h: h, spp: spp, linear: rows, png: png(rows, scene.exposure), ms: System.monotonic_time(:millisecond) - t0, rays: w * h * spp}
  end

  @doc "Tone map (ACES filmic approximation, Narkowicz) and sRGB-encode HDR rows to a PNG."
  def png(rows, exposure \\ 1.0) do
    h = length(rows)
    w = length(hd(rows))
    px = for row <- rows, {r, g, b} <- row, c <- [r, g, b], do: srgb(aces(c * exposure))
    Vapor.Modal.Image.png(Vapor.Modal.Image.new(w, h, 3, px))
  end

  defp aces(x), do: max(0.0, min(1.0, x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)))
  defp srgb(c), do: (v = if c <= 0.0031308, do: 12.92 * c, else: 1.055 * :math.pow(c, 1 / 2.4) - 0.055; round(min(max(v, 0.0), 1.0) * 255) / 255)

  # ================================================================ checks

  @doc """
  The white furnace: a diffuse sphere of albedo `a` inside a uniform
  environment of radiance 1 must show exactly `a` on every pixel it
  covers. Returns `%{mean, max_error, expected}` over the sphere's pixels.
  """
  def furnace(a \\ 0.8, opts \\ []) do
    {:ok, scene} = parse("camera pos=0,0,3 look=0,0,0 fov=40\nsky top=1,1,1 bottom=1,1,1\nsphere c=0,0,0 r=1 mat=diffuse albedo=#{a},#{a},#{a}")
    r = render(scene, Keyword.merge([width: 32, height: 32, spp: 64], opts))
    # the sphere's pixels: those whose centre ray hits it
    vals = for {row, y} <- Enum.with_index(r.linear), {{v, _, _}, x} <- Enum.with_index(row), on_sphere?(x, y, 32, 32), do: v
    %{mean: Enum.sum(vals) / length(vals), max_error: vals |> Enum.map(&abs(&1 - a)) |> Enum.max(), expected: a, pixels: length(vals)}
  end

  @doc """
  The furnace under a sky that brightens upward, L(ω) = (1 + ω_y)/2: a
  convex Lambertian surface of albedo a and normal n must show exactly
  a·(1/2 + n_y/3) (the cosine-weighted mean of L over its hemisphere). A
  biased estimator — uniform directions treated as cosine-distributed,
  the control — gives a·(1/2 + n_y/4) and is caught. Returns the mean
  error over pixels whose normal points up (n_y > 0.5).
  """
  def furnace_gradient(a \\ 0.8, opts \\ []) do
    {:ok, scene} = parse("camera pos=0,0,3 look=0,0,0 fov=40\nsky top=1,1,1 bottom=0,0,0\nsphere c=0,0,0 r=1 mat=diffuse albedo=#{a},#{a},#{a}")
    {w, h} = {32, 32}
    r = render(scene, Keyword.merge([width: w, height: h, spp: 256], opts))
    errs =
      for {row, y} <- Enum.with_index(r.linear), {{v, _, _}, x} <- Enum.with_index(row), on_sphere?(x, y, w, h), (ny = normal_y(x, y, w, h)) > 0.5,
          do: v - a * (0.5 + ny / 3)
    %{mean_error: Enum.sum(errs) / length(errs), pixels: length(errs), spp: r.spp}
  end

  defp normal_y(x, y, w, h) do
    th = :math.tan(40 * :math.pi() / 360)
    d = norm({((x + 0.5) / w * 2 - 1) * th, (1 - (y + 0.5) / h * 2) * th, -1.0})
    o = {0.0, 0.0, 3.0}
    b = dot(o, d)
    t = -b - :math.sqrt(b * b - 8.0)
    elem(add3(o, mul(d, t)), 1)
  end

  defp on_sphere?(x, y, w, h) do
    th = :math.tan(40 * :math.pi() / 360)
    u = ((x + 0.5) / w * 2 - 1) * th
    v = (1 - (y + 0.5) / h * 2) * th
    d = norm({u, v, -1.0})
    # ray from (0,0,3): hits the unit sphere with a margin (no edge pixels)
    b = dot({0.0, 0.0, 3.0}, d)
    b * b - 8.0 > 0.3
  end
end
