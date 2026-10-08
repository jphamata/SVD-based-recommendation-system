defmodule Vapor.SketchTest do
  @moduledoc """
  Sketch → drawing (docs/SKETCH.md §1–2), on hand-wobbled sketches whose
  intent is known (`priv/quality/sketch`, drawn by a script): a rectangle
  drawn 2.2° askew comes back square, a wobbly circle as a circle, a 45°
  hypotenuse at 45°, a free 29.5° line left alone; the control is the same
  fit without the constraints. A floor plan becomes rooms with their
  areas, doors with their widths, and a 3D model that trimesh opens.
  """
  use ExUnit.Case, async: true
  alias Vapor.Sketch

  @moduletag timeout: 600_000
  @dir Path.expand("../../priv/quality/sketch", __DIR__)

  defp load(f), do: elem(Vapor.Docs.Pictures.read(:png, File.read!(Path.join(@dir, f))), 1).image
  defp truth, do: @dir |> Path.join("truth.json") |> File.read!() |> Vapor.JSON.decode() |> elem(1)
  defp deg(l), do: (a = :math.atan2(l.y1 - l.y0, l.x1 - l.x0) * 180 / :math.pi(); if(a < 0, do: a + 180, else: a))

  test "shapes: the askew rectangle comes back square and closed; the circle a circle; 45° snapped; a free angle kept" do
    v = Sketch.vectorize(load("shapes.png"))
    t = truth()
    [cx, cy, r] = t["circle"]
    assert [c] = v.circles
    assert abs(c.cx - cx) < 3 and abs(c.cy - cy) < 3 and abs(c.r - r) < 3

    rect = Enum.filter(v.lines, fn l -> max(l.y0, l.y1) < 260 and max(l.x0, l.x1) < 300 end)
    assert length(rect) == 4
    for l <- rect, do: assert(deg(l) in [0.0, 90.0] or abs(deg(l) - 180) < 1.0e-9, inspect(l))
    # closed: every corner shared by exactly two sides
    ends = Enum.flat_map(rect, fn l -> [{l.x0, l.y0}, {l.x1, l.y1}] end) |> Enum.frequencies()
    assert map_size(ends) == 4 and Enum.all?(ends, fn {_, n} -> n == 2 end)
    [rx, ry] = t["rect_center"]
    {mx, my} = {Enum.sum(Enum.map(Map.keys(ends), &elem(&1, 0))) / 4, Enum.sum(Enum.map(Map.keys(ends), &elem(&1, 1))) / 4}
    assert abs(mx - rx) < 4 and abs(my - ry) < 4

    assert Enum.any?(v.lines, fn l -> abs(deg(l) - 135) < 1.0e-9 or abs(deg(l) - 45) < 1.0e-9 end)
    assert Enum.any?(v.lines, fn l -> abs(deg(l) - (180 - t["free_line_deg"])) < 1.5 end), "the free line keeps its angle"
    assert Enum.any?(v.constraints, &(&1.kind == "perpendicular"))

    # the control: no constraints, the rectangle stays askew
    raw = Sketch.vectorize(load("shapes.png"), snap: false)
    askew = raw.lines |> Enum.filter(fn l -> max(l.y0, l.y1) < 260 and max(l.x0, l.x1) < 300 end) |> Enum.map(&deg/1)
    assert Enum.any?(askew, fn a -> a > 0.5 and a < 89.5 end)
  end

  test "exports: SVG and DXF carry every primitive" do
    v = Sketch.vectorize(load("shapes.png"))
    svg = Sketch.svg(v)
    assert length(Regex.scan(~r/<line /, svg)) == length(v.lines) and length(Regex.scan(~r/<circle /, svg)) == 1
    dxf = Sketch.dxf(v)
    assert String.starts_with?(dxf, "0\nSECTION\n2\nENTITIES\n") and String.ends_with?(dxf, "0\nEOF\n")
    assert length(Regex.scan(~r/\nLINE\n/, dxf)) == length(v.lines) and length(Regex.scan(~r/\nCIRCLE\n/, dxf)) == 1
  end

  test "floor plan: rooms of 12 and 20 m², doors of 0.9 and 1.0 m, at the scale of the longest wall" do
    p = Sketch.plan(load("plan.png"))
    areas = p.rooms |> Enum.map(& &1.area) |> Enum.sort()
    assert length(areas) == 2
    for {a, want} <- Enum.zip(areas, [12.0, 20.0]), do: assert(abs(a - want) / want < 0.03, "#{a} vs #{want}")
    widths = p.doors |> Enum.map(&(&1.width * p.scale)) |> Enum.sort()
    assert length(widths) == 2
    for {wd, want} <- Enum.zip(widths, [0.9, 1.0]), do: assert(abs(wd - want) < 0.08, "#{wd} vs #{want}")
    assert abs(p.scale - 1 / 40) / (1 / 40) < 0.03
  end

  @tag :trimesh
  test "the plan's 3D model opens in trimesh, 2.7 m tall, the footprint of the plan" do
    p = Sketch.plan(load("plan.png"))
    path = Path.join(System.tmp_dir!(), "vapor-plan-#{System.unique_integer([:positive])}.glb")
    File.write!(path, Base.decode64!(p.glb))
    {out, 0} = System.cmd("python3", ["-c", "import trimesh,sys; m=trimesh.load(sys.argv[1], force='mesh'); b=m.bounds; print(b[1][1]-b[0][1], b[1][0]-b[0][0], b[1][2]-b[0][2])", path])
    [h, dx, dz] = out |> String.split() |> Enum.map(&String.to_float/1)
    assert abs(h - 2.7) < 1.0e-3
    assert abs(dx - 8.15) < 0.2 and abs(dz - 4.15) < 0.2
  end
end
