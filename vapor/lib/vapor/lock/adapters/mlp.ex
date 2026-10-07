defmodule Vapor.Lock.Adapters.MLP do
  @moduledoc """
  Tier 3, a topology: the multilayer perceptron — `rows` through affine
  maps with an activation between them. It is the classifier of a probe,
  a diffusion model's denoiser over small images (`Vapor.Modal.Diffusion`),
  the head of a frozen encoder, a projector with a hidden layer (LLaVA-1.5's
  is two layers with GELU).

  `config.json`: `{"model_type": "vapor_mlp", "in_width": k,
  "hidden_sizes": [h₁, …], "out_width": m, "hidden_act": "gelu" | "silu" |
  "relu" | "tanh", "out_act": null | "sigmoid" | "tanh", "labels": […]?}`;
  weights `layers.{i}.weight : f32[out, in]`, `layers.{i}.bias : f32[out]`
  for i = 0 … len(hidden_sizes). `in_width` and every hidden size must be
  multiples of 16 (pad the rows with zeros).

  Contract `:map`: `rows : f32[T, k]` → `out : f32[T, m]`. Build options:
  `rows` (static `T`, default 1), `storage` (`:f32` | `:bf16` matrices).
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  defmodule Config do
    @moduledoc "An admitted perceptron."
    defstruct [:in_width, :hidden, :out_width, :act, :out_act, :from, :to, :raw]
  end

  @impl true
  def id, do: "mlp"

  @impl true
  def claim(%{config: %{"model_type" => "vapor_mlp"}}), do: {:claim, 100}
  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @acts %{"gelu" => :gelu, "silu" => :silu, "swish" => :silu, "relu" => :relu, "tanh" => :tanh}

  @impl true
  def admit(%{config: c}, ws, _opts) do
    {k, hs, m} = {c["in_width"], c["hidden_sizes"] || [], c["out_width"]}

    cond do
      not (is_integer(k) and k > 0 and rem(k, 16) == 0) -> no("in_width", "a positive multiple of 16")
      not (is_list(hs) and Enum.all?(hs, &(is_integer(&1) and &1 > 0 and rem(&1, 16) == 0))) -> no("hidden_sizes", "a list of positive multiples of 16")
      not (is_integer(m) and m > 0) -> no("out_width", "a positive integer")
      not Map.has_key?(@acts, c["hidden_act"] || "gelu") -> no("hidden_act", "one of #{Enum.join(Map.keys(@acts), ", ")}")
      c["out_act"] not in [nil, "sigmoid", "tanh"] -> no("out_act", "null, sigmoid or tanh")
      true ->
        cfg = %Config{in_width: k, hidden: hs, out_width: m, act: @acts[c["hidden_act"] || "gelu"], out_act: c["out_act"] && String.to_atom(c["out_act"]),
                      from: c["from"], to: c["to"], raw: Map.drop(c, ~w(torch_dtype dtype transformers_version))}

        case Enum.find(expected(spec(cfg)), fn {n, s, _} -> not match?(%Tensor{shape: ^s}, ws[n]) end) do
          nil -> {:ok, spec(cfg), ws}
          {n, s, _} -> {:error, Rejection.new({:weight, n}, "f32#{inspect(s, charlists: :as_lists)}", "check the checkpoint against config.json")}
        end
    end
  end

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: "vapor_mlp", lineage: ["vapor_mlp"], interface: :map, config: c,
          width: c.out_width, in_width: c.in_width, features: [],
          modality: %{in: [label(c.from)], out: [label(c.to)]}, digest: Vapor.Canonical.hex_digest({:mlp, c.raw})}
  end

  defp label(nil), do: :rows
  defp label(s) when is_binary(s), do: String.to_atom(s)

  defp widths(c), do: [c.in_width | c.hidden] ++ [c.out_width]

  @impl true
  def expected(%Spec{config: c}) do
    widths(c)
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.with_index()
    |> Enum.flat_map(fn {[i, o], l} -> [{"layers.#{l}.weight", [o, i], :matrix}, {"layers.#{l}.bias", [o], :bias}] end)
  end

  @impl true
  def build(%Spec{config: c}, ws, opts) do
    t = Keyword.get(opts, :rows, 1)
    store = Keyword.get(opts, :storage, :f32)
    n = length(c.hidden) + 1
    x0 = T.input(:rows, :f32, [t, c.in_width])

    {out, lets} =
      Enum.reduce(0..(n - 1), {x0, []}, fn l, {x, lets} ->
        w = ws["layers.#{l}.weight"] |> Tensor.widen() |> then(&if(store == :bf16, do: Tensor.to_bf16(&1), else: &1))
        b = ws["layers.#{l}.bias"] |> Tensor.widen() |> then(&Tensor.new(:f32, [1, hd(&1.shape)], &1.data))
        {wn, bn} = {:"layers.#{l}.weight", :"layers.#{l}.bias"}
        y = T.add(T.linear(x, T.ref(wn, T.const(w))), T.ref(bn, T.const(b)))
        y = if l < n - 1, do: act(y, c.act), else: out_act(y, c.out_act)
        name = :"layers.#{l}.out"
        {T.ref(name, y), [{name, y}, {bn, T.const(b)}, {wn, T.const(w)} | lets]}
      end)

    {:ok, Program.new([out: out], lets: Enum.reverse(lets))}
  end

  defp act(y, :gelu), do: T.gelu(y)
  defp act(y, :silu), do: T.silu(y)
  defp act(y, :relu), do: T.relu(y)
  defp act(y, :tanh), do: T.tanh(y)
  defp out_act(y, nil), do: y
  defp out_act(y, :sigmoid), do: T.sigmoid(y)
  defp out_act(y, :tanh), do: T.tanh(y)

  defp no(field, bound), do: {:error, Rejection.new({:config, field}, bound, "check the config")}
end
