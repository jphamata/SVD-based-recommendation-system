defmodule Vapor.GeomTest do
  @moduledoc """
  3-D (`Vapor.Geom`): marching tetrahedra gives a closed, consistently
  wound sphere whose volume and area converge to the analytic values (the
  control: half the resolution is further off); the relief of an image is a
  closed solid; GLB, OBJ and PLY are read back by trimesh with the same
  vertices, faces and volume; the rendered silhouette of a sphere covers the
  analytic disc, and rendering is deterministic.
  """
  use ExUnit.Case, async: true
  alias Vapor.Geom
  alias Vapor.Modal.Image
  import Vapor.TestHelpers

  defp sphere(n), do: Geom.sdf(fn {x, y, z} -> :math.sqrt(x * x + y * y + z * z) - 0.7 end, n)

  test "marching tetrahedra: watertight sphere; volume and area converge with the resolution" do
    {v_true, a_true} = {4 / 3 * :math.pi() * 0.7 ** 3, 4 * :math.pi() * 0.7 ** 2}
    coarse = sphere(12)
    fine = sphere(32)
    assert Geom.watertight?(fine) and Geom.watertight?(coarse)
    ev = fn m -> abs(Geom.volume(m) - v_true) / v_true end
    ea = fn m -> abs(Geom.area(m) - a_true) / a_true end
    assert ev.(fine) < 0.005 and ea.(fine) < 0.005
    assert ev.(fine) < ev.(coarse) and ea.(fine) < ea.(coarse)
  end

  test "a relief from an image is a closed, outward-wound solid whose volume grows with the height" do
    img = Image.scene(24, 18, seed: 4)
    m1 = Geom.heightmap(img, height: 0.1)
    m2 = Geom.heightmap(img, height: 0.3)
    assert Geom.watertight?(m1) and Geom.volume(m1) > 0 and Geom.volume(m2) > Geom.volume(m1)
  end

  test "the rendered sphere covers the analytic disc; the same mesh renders to the same bits" do
    m = sphere(24)
    img = Geom.render(m, width: 128, height: 128, distance: 2.6, fov: 40.0)
    assert img == Geom.render(m, width: 128, height: 128, distance: 2.6, fov: 40.0)
    bg = {0.95, 0.96, 0.97}
    covered = Enum.count(0..(128 * 128 - 1), fn p -> {elem(img.px, 3 * p), elem(img.px, 3 * p + 1), elem(img.px, 3 * p + 2)} != bg end)
    # a sphere of radius r at distance d subtends a disc of angular radius asin(r/d)
    f = 1 / :math.tan(20 * :math.pi() / 180)
    rad_px = :math.tan(:math.asin(0.7 / 2.6)) * f * 64
    disc = :math.pi() * rad_px * rad_px
    assert abs(covered - disc) / disc < 0.03
  end

  @tag :trimesh
  test "GLB, OBJ and PLY read back by trimesh: same vertices, faces, volume; watertight" do
    m = sphere(16)
    dir = Path.join(System.tmp_dir!(), "vapor-geom-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "m.glb"), Geom.glb(m))
    File.write!(Path.join(dir, "m.obj"), Geom.obj(m))
    File.write!(Path.join(dir, "m.ply"), Geom.ply(%{m | colors: List.duplicate(0.5, length(m.vertices))}))

    out = py!("""
    import trimesh, sys
    for f in ['m.glb', 'm.obj', 'm.ply']:
        x = trimesh.load(sys.argv[1] + '/' + f, force='mesh', process=False)
        print(len(x.vertices), len(x.faces), int(x.is_watertight), round(x.volume, 5))
    """, [dir])

    v = div(length(m.vertices), 3)
    fc = div(length(m.faces), 3)
    for line <- String.split(String.trim(out), "\n") do
      [nv, nf, wt, vol] = String.split(line)
      assert {String.to_integer(nv), String.to_integer(nf), wt} == {v, fc, "1"}
      assert_in_delta String.to_float(vol), Geom.volume(m), 1.0e-4
    end
  end
end
