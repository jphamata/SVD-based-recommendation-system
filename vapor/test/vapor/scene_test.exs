defmodule Vapor.SceneTest do
  @moduledoc """
  The living scene's analysis (docs/CENA.md), on pictures whose geometry
  is known (`priv/quality/scene`, drawn by a script): the sky found and
  pushed to infinity, objects ordered by where they stand, the ground
  walkable, the light where the fire is; a drawing's skeleton with its
  limbs where they were drawn; and prompts turned into operations, with
  every word not understood reported.
  """
  use ExUnit.Case, async: true
  alias Vapor.Scene

  @moduletag timeout: 600_000
  @dir Path.expand("../../priv/quality/scene", __DIR__)

  defp load(f), do: elem(Vapor.Docs.Pictures.read(:png, File.read!(Path.join(@dir, f))), 1).image
  defp truth, do: @dir |> Path.join("truth.json") |> File.read!() |> Vapor.JSON.decode() |> elem(1)

  test "outdoors: the sky at infinity, the horizon where the hills meet it, near things in front, the ground walkable" do
    s = Scene.analyze(load("outdoor.png"))
    t = truth()["outdoor"]
    assert abs(s.horizon - t["horizon"]) < 0.06, "horizon #{s.horizon}"
    [far | _] = s.layers
    assert far.kind == "sky" and far.depth >= 1000
    # layers come far to near, the nearest is the ground under the camera
    depths = Enum.map(s.layers, & &1.depth)
    assert depths == Enum.sort(depths, :desc)
    assert List.last(s.layers).kind == "ground"
    # walkable cells exist, and none above the horizon
    rows_walk = for {c, i} <- Enum.with_index(s.walk.cells), c == 1, do: div(i, s.walk.cols) / s.walk.rows
    assert rows_walk != [] and Enum.min(rows_walk) > s.horizon
    # every layer is a PNG with alpha
    assert Enum.all?(s.layers, &String.starts_with?(&1.png, "data:image/png;base64,"))
  end

  test "indoors: the hearth is the light, warm; the back wall far, the floor near and walkable" do
    s = Scene.analyze(load("guild.png"))
    assert s.light.x > 0.12 and s.light.x < 0.28 and s.light.y > 0.3 and s.light.y < 0.5
    "#" <> hex = s.light.color
    {r, b} = {String.to_integer(String.slice(hex, 0, 2), 16), String.to_integer(String.slice(hex, 4, 2), 16)}
    assert r > b + 60, "a warm light: #{s.light.color}"
    assert List.last(s.layers).kind == "ground" and Enum.sum(s.walk.cells) > 100
  end

  test "push-pull fills a hole smoothly from its border" do
    # a 9×9 field, left half black, right half white, the middle column unknown
    w = 9
    colors = for _y <- 0..8, x <- 0..8, do: if(x < 4, do: {0.0, 0.0, 0.0}, else: {1.0, 1.0, 1.0})
    known = for _y <- 0..8, x <- 0..8, do: x != 4
    f = Scene.push_pull(w, 9, colors, known)
    {r, _, _} = elem(f, 4 * w + 4)
    assert r > 0.2 and r < 0.8
    assert elem(f, 4 * w + 0) == {0.0, 0.0, 0.0} and elem(f, 4 * w + 8) == {1.0, 1.0, 1.0}
  end

  test "a drawing's skeleton: four limb ends where they were drawn, bones in one tree" do
    r = Scene.rig(load("figure.png"))
    assert r.endpoints == 4
    ids = MapSet.new(r.bones, & &1.id)
    assert Enum.all?(r.bones, &(&1.parent == -1 or MapSet.member?(ids, &1.parent)))
    tips = for b <- r.bones, do: {b.x1, b.y1}
    for [x, y] <- truth()["figure"]["ends"] do
      assert Enum.any?(tips, fn {tx, ty} -> abs(tx - x) + abs(ty - y) < 30 end), "no bone ends near #{x},#{y}"
    end
    # the mesh is skinned: every vertex has weights summing to 1
    assert Enum.all?(r.mesh.weights, fn ws -> ws == [] or abs(Enum.sum(Enum.take_every(tl(ws), 2)) - 1.0) < 1.0e-9 end)
  end

  test "direction: Portuguese and English prompts become operations; unknown words are reported, not guessed" do
    d = Scene.direct("uma noite de chuva forte com vento, três pessoas e vagalumes; orbite devagar")
    assert %{"weather" => "rain", "intensity" => 1.0} in d.ops
    assert %{"time" => "night"} in d.ops and %{"wind" => 0.6} in d.ops
    assert %{"spawn" => "people", "count" => 3} in d.ops and %{"camera" => "orbit"} in d.ops
    assert d.unknown == []

    e = Scene.direct("a heavy storm at dusk with many crows, chaotic, xyzzy")
    assert %{"weather" => "storm", "intensity" => 1.0} in e.ops and %{"spawn" => "birds", "count" => 12} in e.ops
    assert %{"entropy" => 0.9} in e.ops and e.unknown == ["xyzzy"]

    assert Scene.direct("sem chuva").ops == [%{"weather" => "clear", "was" => "rain"}]
    assert Scene.direct("forte chuva e vento leve").ops == [%{"weather" => "rain", "intensity" => 1.0}, %{"wind" => 0.3}]
    assert %{"goto" => "door"} in Scene.direct("os guardas vão até a porta").ops
  end

  test "export: a standalone page carries the engine and the scene, and nothing that closes its script early" do
    html = Scene.standalone(Vapor.JSON.encode(%{layers: [], note: "</script><script>alert(1)"}))
    assert html =~ "SceneEngine"
    refute html =~ "</script><script>alert"
  end

  describe "directing the inhabitants (0.12)" do
    test "a named inhabitant is created and directed in the same clause; a pronoun and a time reach it later" do
      r = Scene.direct(~s(a knight named Arthur walks to the door, then at 3s he says "hello" and waves; a guard patrols))
      assert %{"spawn" => "people", "name" => "Arthur", "count" => 1} in r.ops
      assert %{"npc" => "Arthur", "goto" => "door"} in r.ops
      assert %{"npc" => "Arthur", "say" => "hello", "at" => 3.0} in r.ops
      assert %{"npc" => "Arthur", "action" => "wave", "at" => 3.0} in r.ops
      # an unnamed role with a verb is named by its role
      assert %{"npc" => "Guard", "behavior" => "patrol"} in r.ops
      assert r.unknown == []
    end

    test "Portuguese: a time alone in its clause times the next one; an inhabitant already in the scene is found by name" do
      r = Scene.direct("depois de 2 segundos, Maria dança perto da luz", ["Maria"])
      assert %{"npc" => "Maria", "action" => "dance", "at" => 2.0} in r.ops
      assert %{"npc" => "Maria", "goto" => "light", "at" => 2.0} in r.ops
    end

    test "the control: without a name, a pronoun or a role, nothing is aimed at an inhabitant" do
      r = Scene.direct("heavy rain, then at 4s wind")
      refute Enum.any?(r.ops, &Map.has_key?(&1, "npc"))
      assert Enum.any?(r.ops, &(&1["at"] == 4.0))
    end
  end
end
