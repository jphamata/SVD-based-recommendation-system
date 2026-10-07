defmodule Vapor.Lock.Adapters.Linear do
  @moduledoc """
  Tier 3, the smallest topology: an affine map between row spaces,
  `out = rows·Wᵀ + b` — the **projector** of a multimodal model (LLaVA's
  vision-to-language projection, a probe, a bridge between two frozen
  models' spaces). `Vapor.Modal.Bridge` fits one in closed form.

  `config.json`: `{"model_type": "vapor_linear", "in_width": k,
  "out_width": m, "from": "image", "to": "text"}` (`from`/`to` are labels
  for the modal hub); weights `weight : f32[m, k]`, `bias : f32[m]`.
  `in_width` must be a multiple of 16 (pad the rows with zeros).

  Contract `:map`: `rows : f32[T, k]` → `out : f32[T, m]`.
  Build options: `rows` (static `T`, default 1).
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  defmodule Config do
    @moduledoc "An admitted affine map."
    defstruct [:in_width, :out_width, :from, :to, :raw]
  end

  @impl true
  def id, do: "linear"

  @impl true
  def claim(%{config: %{"model_type" => "vapor_linear"}}), do: {:claim, 100}
  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def admit(%{config: c}, ws, _opts) do
    {k, m} = {c["in_width"], c["out_width"]}

    cond do
      not (is_integer(k) and k > 0 and rem(k, 16) == 0) -> no("in_width", "a positive multiple of 16")
      not (is_integer(m) and m > 0) -> no("out_width", "a positive integer")
      not match?(%Tensor{shape: [^m, ^k]}, ws["weight"]) -> {:error, Rejection.new({:weight, "weight"}, "f32[#{m}, #{k}]", "check the checkpoint")}
      not match?(%Tensor{shape: [^m]}, ws["bias"]) -> {:error, Rejection.new({:weight, "bias"}, "f32[#{m}]", "check the checkpoint")}
      true -> {:ok, spec(%Config{in_width: k, out_width: m, from: c["from"], to: c["to"], raw: c}), ws}
    end
  end

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: "vapor_linear", lineage: ["vapor_linear"], interface: :map, config: c,
          width: c.out_width, in_width: c.in_width, features: [],
          modality: %{in: [label(c.from)], out: [label(c.to)]}, digest: Vapor.Canonical.hex_digest({:linear, c.raw})}
  end

  defp label(nil), do: :rows
  defp label(s) when is_binary(s), do: String.to_atom(s)

  @impl true
  def expected(%Spec{config: c}), do: [{"weight", [c.out_width, c.in_width], :matrix}, {"bias", [c.out_width], :bias}]

  @impl true
  def build(%Spec{config: c}, ws, opts) do
    t = Keyword.get(opts, :rows, 1)
    w = Tensor.widen(ws["weight"])
    b = Tensor.widen(ws["bias"]) |> then(&Tensor.new(:f32, [1, c.out_width], &1.data))
    x = T.input(:rows, :f32, [t, c.in_width])
    out = T.add(T.linear(x, T.ref(:weight, T.const(w))), T.ref(:bias, T.const(b)))
    {:ok, Program.new([out: out], lets: [weight: T.const(w), bias: T.const(b)])}
  end

  defp no(field, bound), do: {:error, Rejection.new({:config, field}, bound, "check the config")}
end
