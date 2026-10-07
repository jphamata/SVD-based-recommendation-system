defmodule Vapor.Studio.Node do
  @moduledoc """
  A node of the studio: a pure function from typed inputs and parameters to
  typed outputs.

  A node module groups related nodes: `nodes/0` returns `[{module, spec}]`
  and `run/4` receives the node type first. Each spec declares a node:

      %{type: "image.resize", version: 1, category: "image", title: "Resize",
        inputs: [image: :image, mask: {:mask, :optional}],
        outputs: [image: :image],
        params: [width: {:int, 1, 16384, 512}, method: {:enum, ~w(bilinear lanczos), "lanczos"}],
        doc: "…"}

  Parameter kinds: `{:int, min, max, default}`, `{:float, min, max,
  default}`, `{:enum, values, default}`, `{:bool, default}`, `{:string,
  default}`, `{:data, default}` (bytes, base64 in JSON). `run/4` receives
  the node type, the inputs and the normalized parameters (every one present, in range) and
  a context (`worker`, `dir`), and returns `{:ok, %{port => value}}` or
  `{:error, %Vapor.Rejection{}}`.

  **Determinism is the contract**: the same inputs and parameters give the
  same bits on every machine. The studio relies on it twice — to cache a
  node by the digest of what it depends on, and to let anyone re-derive a
  result from its receipt. A node that reads the clock or the network has no
  place here; `version` is bumped when a node's output bits change.
  """
  alias Vapor.Rejection

  @callback nodes() :: [{module, map}]
  @callback run(type :: String.t(), inputs :: map, params :: map, ctx :: map) :: {:ok, map} | {:error, Rejection.t()}

  @doc "Fill defaults, check kinds and ranges; `{:ok, params}` or a rejection naming the parameter."
  def params(spec, given, node_id \\ nil) do
    given = Map.new(given || %{}, fn {k, v} -> {to_string(k), v} end)
    known = Enum.map(spec.params, fn {k, _} -> to_string(k) end)

    case Enum.reject(Map.keys(given), &(&1 in known)) do
      [unknown | _] ->
        {:error, Rejection.new({:node, node_id, :param, unknown}, "one of #{inspect(known)}", "remove it or check the node type #{spec.type}")}

      [] ->
        Enum.reduce_while(spec.params, {:ok, %{}}, fn {k, kind}, {:ok, acc} ->
          case check(kind, Map.get(given, to_string(k), :default)) do
            {:ok, v} -> {:cont, {:ok, Map.put(acc, k, v)}}
            {:error, bound} -> {:halt, {:error, Rejection.new({:node, node_id, :param, to_string(k)}, bound, "give a value within the bound")}}
          end
        end)
    end
  end

  defp check({:int, _lo, _hi, d}, :default), do: {:ok, d}
  defp check({:int, lo, hi, _}, v) when is_integer(v) and v >= lo and v <= hi, do: {:ok, v}
  defp check({:int, lo, hi, _}, v) when is_float(v) and v == trunc(v) and v >= lo and v <= hi, do: {:ok, trunc(v)}
  defp check({:int, lo, hi, _}, _), do: {:error, "an integer in #{lo}..#{hi}"}
  defp check({:float, _lo, _hi, d}, :default), do: {:ok, d * 1.0}
  defp check({:float, lo, hi, _}, v) when is_number(v) and v >= lo and v <= hi, do: {:ok, v * 1.0}
  defp check({:float, lo, hi, _}, _), do: {:error, "a number in [#{lo}, #{hi}]"}
  defp check({:enum, _vs, d}, :default), do: {:ok, d}
  defp check({:enum, vs, _}, v) when is_binary(v), do: if(v in vs, do: {:ok, v}, else: {:error, "one of #{Enum.join(vs, ", ")}"})
  defp check({:enum, vs, _}, _), do: {:error, "one of #{Enum.join(vs, ", ")}"}
  defp check({:bool, d}, :default), do: {:ok, d}
  defp check({:bool, _}, v) when is_boolean(v), do: {:ok, v}
  defp check({:bool, _}, _), do: {:error, "true or false"}
  defp check({:string, d}, :default), do: {:ok, d}
  defp check({:string, _}, v) when is_binary(v), do: {:ok, v}
  defp check({:string, _}, _), do: {:error, "a string"}
  defp check({:data, d}, :default), do: {:ok, d}
  defp check({:data, _}, {:bytes, b}) when is_binary(b), do: {:ok, b}
  defp check({:data, _}, v) when is_binary(v) do
    case Base.decode64(v, ignore: :whitespace) do
      {:ok, b} -> {:ok, b}
      :error -> {:error, "base64 bytes"}
    end
  end
  defp check({:data, _}, _), do: {:error, "base64 bytes"}

  @doc "Inputs as `{port, type, optional?}`."
  def inputs(spec) do
    Enum.map(spec.inputs, fn
      {p, {t, :optional}} -> {p, t, true}
      {p, t} -> {p, t, false}
    end)
  end

  @doc "A JSON-friendly description of a spec (for the console and the agent tools)."
  def describe(spec) do
    %{type: spec.type, version: spec.version, category: spec[:category] || "misc", title: spec[:title] || spec.type, doc: spec[:doc] || "",
      inputs: Enum.map(inputs(spec), fn {p, t, opt} -> %{name: p, type: t, optional: opt} end),
      outputs: Enum.map(spec.outputs, fn {p, t} -> %{name: p, type: t} end),
      params: Enum.map(spec.params, fn {k, kind} -> Map.put(kind_json(kind), :name, k) end)}
  end

  defp kind_json({:int, lo, hi, d}), do: %{kind: "int", min: lo, max: hi, default: d}
  defp kind_json({:float, lo, hi, d}), do: %{kind: "float", min: lo, max: hi, default: d}
  defp kind_json({:enum, vs, d}), do: %{kind: "enum", values: vs, default: d}
  defp kind_json({:bool, d}), do: %{kind: "bool", default: d}
  defp kind_json({:string, d}), do: %{kind: "string", default: d}
  defp kind_json({:data, _}), do: %{kind: "data", default: nil}
end
