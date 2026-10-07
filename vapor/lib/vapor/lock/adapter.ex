defmodule Vapor.Lock.Adapter do
  @moduledoc """
  The behaviour of a model-airlock adapter (`Vapor.Lock`).

  An adapter turns what arrives from the outside — a *manifest* (the
  checkpoint's configuration map, its tensor index, where it came from) and
  the weights — into a `Vapor.Lock.Spec` and, on request, a program of the
  certified algebra. It is the only place where a model family is known.

  A manifest is a map:

      %{source: :hf | :gguf | :memory, config: map | nil, tensors: %{name => shape},
        path: String.t() | nil, admitted: term | nil}

  (`admitted` carries a configuration a format reader already checked — a
  GGUF file arrives that way.)

  Adapters come in three tiers, by how much of a new model is new:

    1. **alias** — data only (`Vapor.Lock.Alias`): a family that *is* an
       existing topology under other names, fused tensors or extra config
       keys. Written as a map or a JSON file, registered at run time.
    2. **blueprint** — a few lines of code that map a configuration onto
       the knobs of an existing topology (`Vapor.Lock.Adapters.Granite`).
    3. **topology** — a new program shape built from the algebra
       (`Vapor.Lock.Adapters.Encoder`, `Vapor.Lock.Adapters.Codec`).

  An adapter is a module, or `{module, data}` when one module serves many
  descriptors (every alias is `{Vapor.Lock.Alias, descriptor}`); the data
  is then the first argument of every callback.
  """
  alias Vapor.{Program, Rejection}
  alias Vapor.Lock.Spec

  @type manifest :: %{required(:source) => atom, optional(atom) => term}

  @doc "A stable identifier (`\"decoder\"`, `\"phi3\"`, …)."
  @callback id() :: String.t()

  @doc """
  Whether the adapter recognises the manifest: `{:claim, score}` (higher
  wins; 100 = by declared type, 50 = by tensor signature), `{:near, why}`
  (not mine, but here is what would make it so — shown in diagnostics) or
  `:no`.
  """
  @callback claim(manifest) :: {:claim, pos_integer} | {:near, String.t()} | :no

  @doc "Check the manifest and the weights: a spec and the weights in the names `build/3` reads."
  @callback admit(manifest, weights :: map, keyword) :: {:ok, Spec.t(), map} | {:error, Rejection.t()}

  @doc "The program of a spec over admitted weights."
  @callback build(Spec.t(), weights :: map, keyword) :: {:ok, Program.t()} | {:error, Rejection.t()}

  @doc "Whether a configuration value (already admitted) belongs to this adapter."
  @callback owns?(config :: term) :: boolean

  @doc "The spec of an admitted configuration."
  @callback spec(config :: term) :: Spec.t()

  @doc "Every tensor `build/3` reads: `{name, shape, kind}`."
  @callback expected(Spec.t()) :: [{String.t(), [pos_integer], atom}]

  @doc """
  The let-bound activations of the program that feed matrices, and which:
  `[{let_name, [tensor names]}]` — what `Vapor.Merge.calibrate/3` measures.
  """
  @callback taps(Spec.t()) :: [{atom, [String.t()]}]

  @doc """
  The sliding window `w` when it binds on **every** attention layer within a
  cache of `s` positions, else `nil` — what lets `Vapor.Engine` keep a
  sequence's pages in a ring (no position older than `w` is ever read).
  """
  @callback ring_window(Spec.t(), pos_integer) :: pos_integer | nil

  @optional_callbacks expected: 1, taps: 1, ring_window: 2
end
