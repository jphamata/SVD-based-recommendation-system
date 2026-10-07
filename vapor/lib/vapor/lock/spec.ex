defmodule Vapor.Lock.Spec do
  @moduledoc """
  What the core is allowed to know about a model: its **contract**, never
  its family.

  A spec is produced by an adapter of the model airlock (`Vapor.Lock`) and
  is the only model description that crosses into the core (engine,
  embedder, server, merger, quality gate). Everything family-specific —
  tensor names, normalisation variants, routing rules, head layouts —
  stays inside `config`, which is opaque to the core and read only by the
  adapter that produced it.

    * `adapter` — the module that builds programs for this spec;
    * `family` — a display name and lineage (`"phi3"` admitted through the
      `"llama"` blueprint has `family: "phi3"`, `lineage: ["phi3", "llama"]`);
    * `interface` — the program contract (`Vapor.Lock.Contract`):
      `:causal_lm` (tokens → next-token logits, caches as state),
      `:encoder` (rows → hidden rows), `:codec` (rows ↔ codes);
    * `modality` — `%{in: [...], out: [...]}` over `:text | :image | :audio | :rows | :codes`;
    * `in_width` — the row width a `:encoder`, `:codec` or `:map` program reads;
    * `vocab`, `width` (row width of the residual stream — what soft
      tokens, projectors and pooling need), `rows` (static row count of an
      encoder), `max_pos`, `bos`, `eos` (a list);
    * `features` — what `build/3` accepts beyond the contract
      (`:paged`, `:sample`, `:bf16`, `:sb4`, `:inject`, `:hidden`, `:last`);
    * `digest` — canonical digest of the admitted configuration (what a
      merge receipt or a certificate refers to).
  """
  @enforce_keys [:adapter, :family, :interface]
  defstruct [:adapter, :family, :interface, :config, :vocab, :width, :in_width, :rows, :max_pos, :bos, :digest,
             lineage: [], eos: [], features: [], parts: %{}, modality: %{in: [:text], out: [:text]}]

  @type t :: %__MODULE__{}

  @doc "Whether the spec's builder accepts `feature`."
  def supports?(%__MODULE__{features: fs}, feature), do: feature in fs
end
