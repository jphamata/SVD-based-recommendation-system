defmodule Vapor.SceneOpsTest do
  use ExUnit.Case, async: true
  alias Vapor.Scene.Ops

  test "operations parse into what the engine applies; unknown lines are named, not guessed" do
    {:ok, ops, probs} = Ops.parse("""
    add circle sun { x: 0.8, y: 0.2 + 0.1*sin(t), r: 0.05, color: "#ffcc55" }
    add particles snow { count: 50, x: noise(i, 1), y: fract(noise(i, 2) + t/10), color: hsl(200, 50, 90) }
    set sun.r = 0.08
    set world.time = "night"
    at 5: remove sun
    paint it blue
    add dragon smaug { }
    """)
    assert [%{"entity" => %{"id" => "sun", "kind" => "circle", "props" => %{"y" => ["+", 0.2, _]}}}, %{"entity" => %{"props" => %{"color" => %{"hsl" => [200, 50, 90]}}}},
            %{"set" => %{"id" => "sun", "key" => "r", "value" => 0.08}}, %{"time" => "night"}, %{"remove_entity" => "sun", "at" => 5.0}] = ops
    assert [p1, p2] = probs
    assert p1 =~ "line 6" and p2 =~ "dragon"
  end

  test "expressions are data: a property that tries to call anything else is refused" do
    {:ok, [], [p]} = Ops.parse("add circle c { x: system(1) }")
    assert p =~ "not available"
  end

  test "the vocabulary still directs, inside the operation language" do
    {:ok, ops, []} = Ops.parse(~s(direct "night, light rain"))
    assert Enum.any?(ops, &(&1["time"] == "night")) and Enum.any?(ops, &(&1["weather"] == "rain"))
  end

  test "a blank scene, edited and exported, carries its operations" do
    {:ok, s, []} = Ops.apply_text(Ops.blank(w: 640, h: 400), "add text t { x: 0.5, y: 0.5, text: \"hi\" }")
    assert [%{"entity" => _}] = s["ops"]
    html = Vapor.Scene.standalone(Vapor.JSON.encode(s), "t")
    assert html =~ "SceneEngine" and html =~ "\"hi\""
    assert Ops.summary(s) =~ "t (text)"
  end
end
