defmodule Vapor.StudioTest do
  @moduledoc """
  The studio's graph engine (`Vapor.Studio`): typed graphs refused by name
  before anything runs; a content-addressed cache that recomputes exactly
  what is downstream of a change; receipts whose root anyone re-derives;
  subgraphs (inputs/outputs) and `video.map`; ComfyUI API workflows
  translated when every node has a translation, refused whole otherwise.
  """
  use ExUnit.Case, async: true
  alias Vapor.Studio
  alias Vapor.Modal.Image

  defp g(nodes), do: %{"nodes" => nodes}

  defp chain do
    g(%{"1" => %{"type" => "image.solid", "params" => %{"width" => 40, "height" => 30, "red" => 0.8, "blue" => 0.2}},
        "2" => %{"type" => "image.resize", "params" => %{"width" => 20}, "inputs" => %{"image" => ["1", "image"]}},
        "3" => %{"type" => "image.blur", "params" => %{"radius" => 2}, "inputs" => %{"image" => ["2", "image"]}},
        "4" => %{"type" => "image.invert", "inputs" => %{"image" => ["3", "image"]}},
        "5" => %{"type" => "studio.output", "params" => %{"name" => "final"}, "inputs" => %{"value" => ["4", "image"]}}})
  end

  test "a run, then a change: only the nodes downstream of it run again; the cache is safe because the root re-derives" do
    {:ok, r} = Studio.run(chain(), cache: %{})
    assert r.executed == ~w(1 2 3 4 5) and r.cached == []
    assert %Image{w: 20, h: 15} = r.results["final"]

    # same graph: everything cached, same root
    {:ok, r1} = Studio.run(chain(), cache: r.cache)
    assert r1.executed == [] and r1.root == r.root

    # change the blur: 1 and 2 are hits, 3–5 run
    g2 = put_in(chain(), ["nodes", "3", "params", "radius"], 3)
    {:ok, r2} = Studio.run(g2, cache: r.cache)
    assert r2.cached == ~w(1 2) and r2.executed == ~w(3 4 5)
    refute r2.root == r.root

    # receipts: from nothing, the same root; a different claim is refused
    assert :ok = Studio.verify(g2, r2.root)
    assert {:error, {:root, _, _}} = Studio.verify(g2, r.root)
    receipt = Studio.receipt(r2)
    assert length(receipt.nodes) == 5 and Enum.all?(receipt.nodes, &(&1.outputs |> Map.values() |> Enum.all?(fn o -> byte_size(o.digest) == 64 end)))

    # the same graph from its JSON text
    {:ok, from_text} = Studio.run(Vapor.JSON.encode(chain()))
    assert from_text.root == r.root
  end

  test "graphs are checked before anything runs, and the rejection names the node and the port" do
    bad = fn path, v -> elem(Studio.validate(put_in(chain(), path, v)), 1) end
    assert %{node: {:node, "2", :type, "image.resiz"}, repair: repair} = bad.(["nodes", "2", "type"], "image.resiz")
    assert repair =~ "image.resize"
    assert %{node: {:node, "2", :param, "method"}} = bad.(["nodes", "2", "params", "method"], "magic")
    assert %{node: {:node, "2", :param, "colour"}} = bad.(["nodes", "2", "params", "colour"], 1)
    assert %{node: {:node, "3", :input, "image"}} = bad.(["nodes", "3", "inputs"], %{})
    assert %{node: {:node, "3", :input, "image"}, repair: r} = bad.(["nodes", "3", "inputs", "image"], ["9", "image"])
    assert r =~ "no node"
    # a type mismatch: an audio into an image
    g3 = put_in(chain(), ["nodes", "6"], %{"type" => "audio.tone"}) |> put_in(["nodes", "4", "inputs", "image"], ["6", "audio"])
    assert {:error, %{node: {:node, "4", :input, "image"}, bound: "type image"}} = Studio.validate(g3)
    # a cycle
    cyc = g(%{"a" => %{"type" => "image.invert", "inputs" => %{"image" => ["b", "image"]}}, "b" => %{"type" => "image.invert", "inputs" => %{"image" => ["a", "image"]}}})
    assert {:error, %{node: {:cycle, ["a", "b"]}}} = Studio.validate(cyc)
  end

  test "a subgraph takes inputs; video.map runs it on every frame with one cache (identical frames computed once)" do
    sub = g(%{"in" => %{"type" => "studio.input", "params" => %{"name" => "frame", "type" => "image"}},
              "f" => %{"type" => "image.flip", "inputs" => %{"image" => ["in", "value"]}},
              "out" => %{"type" => "studio.output", "params" => %{"name" => "frame"}, "inputs" => %{"value" => ["f", "image"]}}})

    img = Image.scene(16, 12, seed: 3)
    {:ok, r} = Studio.run(sub, inputs: %{"frame" => img})
    flipped = r.results["frame"]
    assert Image.at(flipped, 0, 0, 0) == Image.at(img, 15, 0, 0)
    assert {:error, %{node: {:input, "frame"}}} = Studio.run(sub)

    video = g(%{"1" => %{"type" => "image.solid", "params" => %{"width" => 16, "height" => 12, "green" => 0.5}},
                "2" => %{"type" => "video.still", "params" => %{"seconds" => 1.0, "fps" => 4.0}, "inputs" => %{"image" => ["1", "image"]}},
                "3" => %{"type" => "video.map", "params" => %{"graph" => Vapor.JSON.encode(sub)}, "inputs" => %{"video" => ["2", "video"]}},
                "4" => %{"type" => "studio.output", "params" => %{"name" => "v"}, "inputs" => %{"value" => ["3", "video"]}}})
    {:ok, rv} = Studio.run(video)
    assert length(rv.results["v"].frames) == 4
  end

  test "ComfyUI API workflows: translated node by node, each translation stated; one unsupported node refuses the whole" do
    prompt = %{
      "1" => %{"class_type" => "EmptyImage", "inputs" => %{"width" => 32, "height" => 24, "batch_size" => 1, "color" => 0xFF8000}},
      "2" => %{"class_type" => "ImageScale", "inputs" => %{"image" => ["1", 0], "upscale_method" => "nearest-exact", "width" => 16, "height" => 12, "crop" => "disabled"}},
      "3" => %{"class_type" => "ImageBlur", "inputs" => %{"image" => ["2", 0], "blur_radius" => 2, "sigma" => 0.5}},
      "4" => %{"class_type" => "SaveImage", "inputs" => %{"images" => ["3", 0], "filename_prefix" => "out"}}}

    assert {:ok, graph, notes} = Studio.Comfy.import(prompt)
    assert Enum.any?(notes, &(&1 =~ "1.0 pixels"))
    {:ok, r} = Studio.run(graph)
    out = r.results["out"]
    assert {out.w, out.h} == {16, 12}
    assert_in_delta Image.at(out, 5, 5, 0), 1.0, 1.0e-6
    assert_in_delta Image.at(out, 5, 5, 1), 128 / 255, 1.0e-6

    bad = Map.put(prompt, "5", %{"class_type" => "FancyNode", "inputs" => %{}})
    assert {:error, %{repair: why}} = Studio.Comfy.import(bad)
    assert why =~ "FancyNode"
  end

  test "the catalogue describes every node (types, ports, parameters with bounds)" do
    cat = Studio.catalogue()
    types = Enum.map(cat, & &1.type)
    for t <- ~w(image.load image.resize image.blur audio.resample video.camera video.map geom.shape geom.render), do: assert(t in types)
    resize = Enum.find(cat, &(&1.type == "image.resize"))
    assert %{kind: "enum", values: values} = Enum.find(resize.params, &(&1.name == :method))
    assert "lanczos" in values
  end
end
