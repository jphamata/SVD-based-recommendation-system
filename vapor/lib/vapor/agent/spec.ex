defmodule Vapor.Agent.Spec do
  @moduledoc """
  An agent is a value. Everything that decides its behaviour — the model's
  identity, its instructions, the tools it may call (with their schemas,
  effect classes and versions), the sampling policy, the capabilities it is
  granted — is a field of this struct, and the agent's identity is the
  canonical digest of that struct (`digest/1`). Two operators with the same
  digest run the same agent; a different digest is a different agent.

  Nothing here is ever modified in place. Improving an agent (new
  instructions, a new tool, a fine-tuned model) is `evolve/2`, which returns
  a *new* spec whose `parent` is the old digest — so an agent has a
  lineage, every journal names the exact version that produced it, and
  "which agent did this?" always has a precise answer.

  Tool entries (`tool/1`): `name`, `description`, `parameters` (JSON
  Schema), `effect` and `version`. The effect class is what replay and
  capability checks rely on:

    * `"pure"` — a deterministic function of its arguments (arithmetic,
      lookup in a pinned corpus): re-executed on replay, and the result must
      match the journal;
    * `"observe"` — reads the world (a web page, a sensor, the clock): the
      result is recorded and replay reads the record, never the world;
    * `"act"` — changes the world (sends, writes, pays): runs only when the
      spec grants it, at most once per idempotency key, and is never
      re-executed by replay.

  Implementations (functions) are bound at run time by name; the `version`
  string is the operator's claim about which implementation that is, and
  replay of `pure` tools checks the claim.
  """
  alias Vapor.Canonical

  @enforce_keys [:name, :model]
  defstruct name: nil, instructions: "", model: nil, tools: [], grants: [], parent: nil,
            policy: %{"temperature" => 0.0, "max_tokens" => 256, "max_steps" => 8, "seed" => 0},
            env: %{}

  @type t :: %__MODULE__{}

  @effects ~w(pure observe act)

  @doc """
  A spec. `model` names the model: `%{"kind" => "local", "id" => digest}`
  (a vapor engine: decisions are re-derivable bit for bit), or a remote
  one (`%{"kind" => "openai", "base_url" => …, "model" => …}`,
  `%{"kind" => "anthropic", "model" => …}`: decisions are observations).
  """
  def new(fields) do
    fields = Map.new(fields)
    spec = struct!(__MODULE__, Map.update(fields, :policy, default_policy(), &Map.merge(default_policy(), stringify(&1))))
    %{spec | tools: Enum.map(spec.tools, &tool/1), model: stringify(spec.model),
             env: Map.merge(%{"unicode" => Vapor.Unicode.version(), "canonical" => "cbor-rfc8949-length-first"}, stringify(spec.env))}
  end

  defp default_policy, do: %{"temperature" => 0.0, "max_tokens" => 256, "max_steps" => 8, "seed" => 0}

  @doc "A tool entry, checked: name, description, parameters, effect ∈ pure | observe | act, version."
  def tool(t) do
    t = stringify(t)
    unless is_binary(t["name"]) and t["effect"] in @effects, do: raise(ArgumentError, "a tool needs a name and an effect in #{inspect(@effects)}: #{inspect(t)}")
    Map.merge(%{"description" => "", "parameters" => %{"type" => "object"}, "version" => "1"}, Map.take(t, ~w(name description parameters effect version)))
  end

  @doc "The agent's identity: the canonical digest (hex) of the whole spec."
  def digest(%__MODULE__{} = s), do: Canonical.hex_digest({:vapor_agent, 1, Map.from_struct(s)})

  @doc "A new version: `changes` applied, `parent` set to this spec's digest."
  def evolve(%__MODULE__{} = s, changes) do
    s |> Map.from_struct() |> Map.merge(Map.new(changes)) |> Map.put(:parent, digest(s)) |> new()
  end

  @doc "The tools in the OpenAI shape the backends and templates take."
  def openai_tools(%__MODULE__{tools: tools}) do
    for t <- tools, do: %{"type" => "function", "function" => %{"name" => t["name"], "description" => t["description"], "parameters" => t["parameters"]}}
  end

  @doc false
  def stringify(%{} = m) when not is_struct(m), do: Map.new(m, fn {k, v} -> {to_string(k), stringify(v)} end)
  def stringify(l) when is_list(l), do: Enum.map(l, &stringify/1)
  def stringify(v), do: v
end
