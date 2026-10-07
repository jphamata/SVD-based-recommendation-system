defmodule Vapor.Lock.Adapters.Decoder do
  @moduledoc """
  The pre-norm decoder topology (`Vapor.Model.Llama` over
  `Vapor.Model.Config`): Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE, Mixtral,
  Gemma 3, DeepSeek-V3 — and, through aliases and blueprints, every family
  that is one of these under other names or knobs.

  Claims a manifest whose `model_type` is one of `Config.families/0` (score
  100) or a GGUF the format reader already admitted. A checkpoint whose
  tensors follow the decoder layout but whose `model_type` is unknown is a
  near miss, with the repair spelled out.

  Build options: those of `Vapor.Model.Llama.program/3`.
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.Lock.Spec
  alias Vapor.Model.{Config, Llama}

  @impl true
  def id, do: "decoder"

  @impl true
  def claim(%{admitted: %Config{}}), do: {:claim, 100}

  def claim(%{config: %{"model_type" => t}} = m) do
    cond do
      t in Config.families() -> {:claim, 100}
      layout?(m) -> {:near, "the tensors follow the pre-norm decoder layout, but model_type #{inspect(t)} is not one of " <>
                            "#{Enum.join(Config.families(), ", ")} — register an alias whose \"like\" is the closest family"}
      true -> :no
    end
  end

  def claim(_), do: :no

  defp layout?(%{tensors: ts}) when is_map(ts),
    do: Enum.any?(Map.keys(ts), &String.match?(&1, ~r/^model\.layers\.0\.(self_attn|mlp)\./))

  defp layout?(_), do: false

  @impl true
  def admit(%{admitted: %Config{} = c}, ws, _opts), do: {:ok, spec(c), ws}

  def admit(%{config: map}, ws, _opts) do
    with {:ok, c} <- Config.from_map(map), do: {:ok, spec(c), ws}
  end

  @impl true
  def build(%Spec{config: c}, ws, opts), do: Llama.program(c, ws, opts)

  @impl true
  def ring_window(%Spec{config: %Config{} = c}, s), do: Config.ring_window(c, s)

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: c.arch, lineage: [c.arch], interface: :causal_lm, config: c,
          vocab: c.vocab, width: c.hidden, max_pos: c.max_pos, bos: c.bos, eos: c.eos |> List.wrap() |> Enum.reject(&is_nil/1),
          features: [:paged, :sample, :bf16, :sb4, :inject, :hidden, :last],
          modality: %{in: [:text, :rows], out: [:text]},
          digest: Vapor.Canonical.hex_digest({:decoder, c.raw || Map.from_struct(%{c | raw: nil})})}
  end

  @impl true
  def expected(%Spec{config: c}), do: Llama.expected_weights(c)

  @doc """
  Measurable inputs of the decoder's matrices: each layer's normalised
  attention input (read by q, k, v — not for latent attention), its
  normalised MLP input (gate, up — dense layers only) and the final norm
  (the output head, unless tied to the embedding).
  """
  @impl true
  def taps(%Spec{config: %Config{} = c}) do
    layers =
      for l <- 0..(c.layers - 1) do
        p = "model.layers.#{l}."
        attn = if c.mla, do: [], else: [{:"layers.#{l}.attn_in", Enum.map(~w(q_proj k_proj v_proj), &(p <> "self_attn.#{&1}.weight"))}]
        mlp = if Config.moe_layer?(c, l), do: [], else: [{:"layers.#{l}.mlp_in", [p <> "mlp.gate_proj.weight", p <> "mlp.up_proj.weight"]}]
        attn ++ mlp
      end

    List.flatten(layers) ++ if(c.tie, do: [], else: [{:final_norm, ["lm_head.weight"]}])
  end
end
