defmodule Vapor.Studio.Nodes.Core do
  @moduledoc "Graph plumbing: subgraph inputs and outputs, constants."
  @behaviour Vapor.Studio.Node
  alias Vapor.Rejection

  @types ~w(image mask audio video mesh text number tensor latent json)

  @impl true
  def nodes do
    [{__MODULE__, %{type: "studio.input", version: 1, category: "graph", title: "Input",
                    doc: "A value given to the graph when it runs (`run(graph, inputs: %{name => value})`) — what makes a graph a reusable subgraph.",
                    inputs: [], outputs: [value: :json], params: [name: {:string, "in"}, type: {:enum, @types, "image"}]}},
     {__MODULE__, %{type: "studio.output", version: 1, category: "graph", title: "Output",
                    doc: "Names a result of the graph (`run.results[name]`).", inputs: [value: :json], outputs: [value: :json],
                    params: [name: {:string, "out"}]}},
     {__MODULE__, %{type: "text.value", version: 1, category: "text", title: "Text", doc: "A text constant.", inputs: [],
                    outputs: [text: :text], params: [text: {:string, ""}]}},
     {__MODULE__, %{type: "number.value", version: 1, category: "text", title: "Number", doc: "A number constant.", inputs: [],
                    outputs: [number: :number], params: [value: {:float, -1.0e12, 1.0e12, 0.0}]}}]
  end

  @impl true
  def run("studio.input", _ins, %{name: name, type: t}, ctx) do
    case Map.fetch(ctx.inputs, name) do
      {:ok, v} ->
        if Vapor.Studio.Value.is?(String.to_existing_atom(t), v), do: {:ok, %{value: v}},
          else: {:error, Rejection.new({:input, name}, "a value of type #{t}", "give run/2 the right input")}

      :error -> {:error, Rejection.new({:input, name}, "a value for the input #{inspect(name)}", "pass it in run(graph, inputs: %{#{inspect(name)} => …})")}
    end
  end

  def run("studio.output", %{value: v}, _p, _ctx), do: {:ok, %{value: v}}
  def run("text.value", _, %{text: t}, _), do: {:ok, %{text: t}}
  def run("number.value", _, %{value: v}, _), do: {:ok, %{number: v}}
end
