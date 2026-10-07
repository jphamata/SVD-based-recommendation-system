defmodule Vapor.RenderTest do
  @moduledoc """
  The path tracer (docs/RENDER.md) tested the way renderer authors test
  theirs: the white furnace (energy conservation), the gradient furnace
  (the estimator's distribution; a biased estimator is the control and is
  caught), the N^−½ convergence of the error with samples per pixel, the
  determinism of the image, and the console's GPU tracer against this
  reference in headless Chromium.
  """
  use ExUnit.Case, async: true
  @moduletag timeout: 600_000
  alias Vapor.Render

  @scene """
  camera pos=0,1.2,4.5 look=0,0.8,0 fov=45
  sky top=0.55,0.7,1.0 bottom=1,1,1
  sun dir=0.4,1,0.3 color=1,0.95,0.85 power=2.5
  plane y=0 mat=diffuse albedo=0.75,0.75,0.75 checker=0.5
  sphere c=0,0.8,0 r=0.8 mat=glass ior=1.5
  sphere c=-1.7,0.6,-0.5 r=0.6 mat=metal albedo=0.95,0.75,0.4 rough=0.08
  sphere c=1.6,0.5,0.3 r=0.5 mat=diffuse albedo=0.8,0.2,0.15
  box min=-0.4,0,-2 max=0.4,1.2,-1.4 mat=diffuse albedo=0.3,0.5,0.8
  """

  test "the white furnace: albedo a under radiance 1 reads a on every pixel" do
    f = Render.furnace(0.8, spp: 8)
    assert f.pixels > 400
    assert f.max_error < 1.0e-9
  end

  test "the gradient furnace: a(½ + n_y/3) within noise; the biased estimator (the control) is caught" do
    good = Render.furnace_gradient(0.8)
    bad = Render.furnace_gradient(0.8, biased: true)
    assert abs(good.mean_error) < 0.005
    # cosine-weighted mean of (1 + ω_y)/2 is ½ + n_y/3; the biased one gives ½ + n_y/4: −a·n_y/12 ≈ −0.05 on n_y ∈ (0.5, 1]
    assert bad.mean_error < -0.03
  end

  # diffuse transport under a sky and a sun sampled explicitly: finite, moderate variance, so the RMS
  # error must fall as N^−½. (Small lights reached only by BSDF sampling — a sun through glass, a
  # small emitter — are heavy-tailed for any unidirectional path tracer without multiple importance
  # sampling: the honest limit noted in docs/RENDER.md, so they are not part of this check.)
  @diffuse """
  camera pos=0,1.2,4.5 look=0,0.6,0 fov=45
  sky top=0.55,0.7,1.0 bottom=1,1,1
  sun dir=0.4,1,0.3 color=1,0.95,0.85 power=2.5
  plane y=0 mat=diffuse albedo=0.75,0.75,0.75 checker=0.5
  sphere c=-0.9,0.6,0 r=0.6 mat=diffuse albedo=0.3,0.6,0.8
  sphere c=0.9,0.5,0.3 r=0.5 mat=diffuse albedo=0.8,0.2,0.15
  box min=-0.4,0,-2 max=0.4,1.2,-1.4 mat=diffuse albedo=0.9,0.9,0.9
  """

  test "the error falls as N^−½ with samples per pixel" do
    {:ok, s} = Render.parse(@diffuse)
    ref = Render.render(s, width: 16, height: 10, spp: 4096, seed: 99).linear |> List.flatten()
    err = fn spp ->
      img = Render.render(s, width: 16, height: 10, spp: spp, seed: 7).linear |> List.flatten()
      Enum.zip(img, ref) |> Enum.map(fn {{a, b, c}, {x, y, z}} -> (a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2 end) |> then(&:math.sqrt(Enum.sum(&1) / length(&1)))
    end
    ns = [8, 32, 128, 512]
    es = Enum.map(ns, err)
    # least-squares slope of log error against log N
    xs = Enum.map(ns, &:math.log/1); ys = Enum.map(es, &:math.log/1)
    mx = Enum.sum(xs) / 4; my = Enum.sum(ys) / 4
    slope = Enum.zip(xs, ys) |> Enum.map(fn {x, y} -> (x - mx) * (y - my) end) |> Enum.sum() |> Kernel./(xs |> Enum.map(&((&1 - mx) ** 2)) |> Enum.sum())
    assert hd(es) > List.last(es) * 4
    assert slope < -0.38 and slope > -0.62
  end

  test "the same seed is the same picture; another seed is another sample of it" do
    {:ok, s} = Render.parse(@scene)
    a = Render.render(s, width: 24, height: 16, spp: 4, seed: 3)
    b = Render.render(s, width: 24, height: 16, spp: 4, seed: 3)
    c = Render.render(s, width: 24, height: 16, spp: 4, seed: 4)
    assert a.png == b.png and a.linear == b.linear
    refute a.linear == c.linear
    assert <<137, "PNG", _::binary>> = a.png
  end

  test "a malformed scene is refused with its line" do
    assert {:error, m} = Render.parse("camera pos=0,1,4\nsphere c=0,0,0 r=oops")
    assert m =~ "line 2"
  end

  @tag :playwright
  test "the console's GPU tracer (WebGL2 on SwiftShader) passes the gradient furnace and agrees with this reference" do
    dir = Path.join(System.tmp_dir!(), "vapor-gpu-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    scene = Path.join(dir, "scene.txt")
    File.write!(scene, @scene)
    js = Path.expand("../js/gpu_tracer.mjs", __DIR__)
    tracer = Path.join([to_string(:code.priv_dir(:vapor)), "console", "gpu_tracer.js"])
    {out, 0} = System.cmd("node", [js, tracer, scene, "256"], cd: dir, stderr_to_stdout: true)
    {:ok, j} = out |> String.split("\n", trim: true) |> List.last() |> Vapor.JSON.decode()
    assert j["errors"] == []
    assert abs(j["furnace"]["mean_error"]) < 0.01
    {:ok, s} = Render.parse(@scene)
    r = Render.render(s, width: j["w"], height: j["h"], spp: 256, seed: 5)
    mean = r.linear |> List.flatten() |> Enum.map(fn {a, b, c} -> 0.2126 * a + 0.7152 * b + 0.0722 * c end) |> then(&(Enum.sum(&1) / length(&1)))
    # two independent Monte Carlo estimates of the same mean radiance (glass caustics make it the noisiest quantity)
    assert abs(j["scene_mean"] - mean) / mean < 0.03
  end
end
