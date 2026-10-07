defmodule Vapor.Studio.Nodes.Geom do
  @moduledoc "3-D nodes: relief from an image, implicit shapes by marching tetrahedra, rendering and turntables (`Vapor.Geom`)."
  @behaviour Vapor.Studio.Node
  alias Vapor.Geom

  defp n(type, title, doc, inputs, outputs, params),
    do: {__MODULE__, %{type: type, version: 1, category: "3d", title: title, doc: doc, inputs: inputs, outputs: outputs, params: params}}

  @impl true
  def nodes do
    [n("geom.heightmap", "Relief from image", "The image's luma as height; a closed solid (for printing) unless `solid` is off.", [image: :image], [mesh: :mesh],
       [height: {:float, 0.0, 2.0, 0.2}, solid: {:bool, true}, max_side: {:int, 8, 512, 128}]),
     n("geom.shape", "Implicit shape", "An implicit surface by marching tetrahedra: sphere, torus, rounded box, gyroid shell, or two spheres smoothly united.", [], [mesh: :mesh],
       [shape: {:enum, ~w(sphere torus box gyroid blobs), "torus"}, size: {:float, 0.1, 1.0, 0.6}, thickness: {:float, 0.02, 0.5, 0.2},
        resolution: {:int, 8, 96, 32}]),
     n("geom.render", "Render", "A z-buffered, Lambert-shaded view (the same pixels on every machine).", [mesh: :mesh], [image: :image],
       [width: {:int, 16, 2048, 384}, height: {:int, 16, 2048, 384}, azimuth: {:float, -360.0, 360.0, 30.0}, elevation: {:float, -89.0, 89.0, 25.0},
        distance: {:float, 0.5, 20.0, 2.6}]),
     n("geom.turntable", "Turntable", "An orbit of the camera, as a video.", [mesh: :mesh], [video: :video],
       [frames: {:int, 2, 360, 24}, fps: {:float, 1.0, 60.0, 12.0}, size: {:int, 16, 1024, 192}, elevation: {:float, -89.0, 89.0, 25.0}, distance: {:float, 0.5, 20.0, 2.6}])]
  end

  @impl true
  def run("geom.heightmap", %{image: img}, p, ctx) do
    s = max(img.w, img.h)
    img =
      if s > p.max_side,
        do: Vapor.Studio.Resample.resize(img, max(2, round(img.w * p.max_side / s)), max(2, round(img.h * p.max_side / s)), "area", worker: ctx.worker),
        else: img

    {:ok, %{mesh: Geom.heightmap(img, height: p.height, solid: p.solid)}}
  end

  def run("geom.shape", _, p, _) do
    r = p.size
    t = p.thickness
    len = fn {x, y, z} -> :math.sqrt(x * x + y * y + z * z) end

    f =
      case p.shape do
        "sphere" -> fn v -> len.(v) - r end
        "torus" -> fn {x, y, z} -> d = :math.sqrt(x * x + z * z) - r; :math.sqrt(d * d + y * y) - t end
        "box" -> fn {x, y, z} -> q = {abs(x) - r + t, abs(y) - r + t, abs(z) - r + t}; len.(vmax(q)) + min(max3(q), 0.0) - t end
        "gyroid" ->
          fn {x, y, z} = v ->
            k = 2.0 * :math.pi() / r
            g = Vapor.CR.sin_f64(k * x) * Vapor.CR.cos_f64(k * y) + Vapor.CR.sin_f64(k * y) * Vapor.CR.cos_f64(k * z) + Vapor.CR.sin_f64(k * z) * Vapor.CR.cos_f64(k * x)
            max(abs(g) / k - t / 2, len.(v) - 0.9)
          end

        "blobs" ->
          fn {x, y, z} ->
            a = len.({x - r * 0.6, y, z}) - r * 0.7
            b = len.({x + r * 0.6, y, z}) - r * 0.7
            h = max(t - abs(a - b), 0.0) / t
            min(a, b) - h * h * t * 0.25
          end
      end

    {:ok, %{mesh: Geom.sdf(f, p.resolution)}}
  end

  def run("geom.render", %{mesh: m}, p, _),
    do: {:ok, %{image: Geom.render(m, width: p.width, height: p.height, azimuth: p.azimuth, elevation: p.elevation, distance: p.distance)}}

  def run("geom.turntable", %{mesh: m}, p, _),
    do: {:ok, %{video: Geom.turntable(m, frames: p.frames, fps: p.fps, width: p.size, height: p.size, elevation: p.elevation, distance: p.distance)}}

  defp vmax({a, b, c}), do: {max(a, 0.0), max(b, 0.0), max(c, 0.0)}
  defp max3({a, b, c}), do: max(a, max(b, c))
end
