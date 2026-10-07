defmodule Vapor.Lock.Adapters.Granite do
  @moduledoc """
  Tier 2 of the model airlock, a **blueprint**: IBM Granite 3.x is the
  Llama topology plus four multipliers, so the whole adapter is the map
  from its configuration onto the decoder's knobs.

  | `config.json` | Hugging Face (`modeling_granite.py`) | decoder knob |
  |---|---|---|
  | `embedding_multiplier` | `inputs_embeds * m` | `embed_scale` |
  | `attention_multiplier` | attention `scaling = m` (instead of 1/√dh) | `attn_scale` |
  | `residual_multiplier` | `residual + branch * m` (both branches) | `residual_scale` |
  | `logits_scaling` | `logits / s` | `logit_divisor` |

  Everything else (attention biases, RoPE scaling, tied embeddings, the
  checks on activations and MLP biases) is the Llama family's own
  admission, unchanged. The original map is kept, so `Config.to_map/1`
  writes the checkpoint back as Granite.
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.Lock.Adapters.Decoder
  alias Vapor.Lock.Spec
  alias Vapor.Model.Config
  alias Vapor.Rejection

  @impl true
  def id, do: "granite"

  @impl true
  def claim(%{config: %{"model_type" => "granite"}}), do: {:claim, 100}
  def claim(_), do: :no

  @impl true
  def admit(%{config: map}, ws, _opts) do
    nums = for k <- ~w(embedding_multiplier attention_multiplier residual_multiplier logits_scaling), do: {k, map[k]}

    with :ok <- numbers(nums),
         {:ok, c} <- Config.from_map(Map.put(map, "model_type", "llama")) do
      v = Map.new(nums, fn {k, x} -> {k, x && x * 1.0} end)

      c = %{c | arch: "granite", raw: Map.drop(map, ~w(torch_dtype dtype transformers_version)),
                embed_scale: v["embedding_multiplier"], attn_scale: v["attention_multiplier"],
                residual_scale: v["residual_multiplier"], logit_divisor: v["logits_scaling"]}

      {:ok, spec(c), ws}
    end
  end

  defp numbers(kvs) do
    case Enum.find(kvs, fn {_, x} -> x != nil and not (is_number(x) and x > 0) end) do
      nil -> :ok
      {k, x} -> {:error, Rejection.new({:config, k}, "a positive number or absent (got #{inspect(x)})", "check the config")}
    end
  end

  @impl true
  def build(spec, ws, opts), do: Decoder.build(spec, ws, opts)

  @impl true
  def owns?(%Config{arch: "granite"}), do: true
  def owns?(_), do: false

  @impl true
  def spec(%Config{} = c), do: %Spec{Decoder.spec(c) | adapter: __MODULE__, family: "granite", lineage: ["granite", "llama"]}

  @impl true
  def expected(spec), do: Decoder.expected(spec)

  @impl true
  def taps(spec), do: Decoder.taps(spec)
end
