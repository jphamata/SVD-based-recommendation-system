defmodule Vapor.AssayLayersTest do
  @moduledoc """
  The Assay's `layers` tool (the Perception Encoder's question: which layer
  should a probe read?): a planted network whose classes are linearly
  readable in the middle and folded at the output; the protocol must find
  the middle, beat the output on test, and find nothing when the labels
  are shuffled.
  """
  use ExUnit.Case, async: true
  alias Vapor.Assay
  alias Vapor.Assay.Layers

  @moduletag timeout: 300_000

  test "the planted network is read best in the middle, not at its output" do
    {:ok, out} = Assay.run("layers", Assay.example("layers"))
    assert out.chosen_layer in [0, 1, 2]
    assert out.test.chosen > out.test.last + 0.15
    assert out.chosen_vs_last.p_mcnemar < 0.01
    assert Enum.all?(out.evidence, & &1.ok)
    # the test split is read once: what was chosen on validation is what is reported
    assert Enum.at(out.validation_accuracy, out.chosen_layer) == Enum.max(out.validation_accuracy)
  end

  test "labels that carry nothing: the signal check fails, as it must" do
    doc = Vapor.JSON.decode!(Assay.example("layers"))
    noise = Enum.map(Enum.with_index(doc["labels"]), fn {_, i} -> rem(:erlang.phash2({:noise, i}), 3) end)
    {:ok, out} = Layers.run(Vapor.JSON.encode(%{doc | "labels" => noise}))
    refute hd(out.evidence).ok
  end

  test "deterministic: the same input gives the same answer" do
    ex = Layers.example(120, 5)
    assert Layers.run(ex) == Layers.run(ex)
    assert {:error, _} = Layers.run(~s({"labels": [1, 2], "layers": [[[1], [2]]]}))
  end
end
