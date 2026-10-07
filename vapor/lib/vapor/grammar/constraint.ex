defmodule Vapor.Grammar.Constraint do
  @moduledoc """
  The constraint one generated sequence carries: which token ids may come
  next, and how each chosen token moves the grammar.

  Two modes:

    * `:strict` — the whole output is a member of the grammar (structured
      outputs, `tool_choice: "required"`). Once the bytes form a complete
      member the end-of-sequence ids become allowed; once nothing can
      follow, *only* they are.
    * `{:lazy, trigger}` — free text until the generated text contains
      `trigger` (e.g. Qwen's `<tool_call>`, Mistral's `[TOOL_CALLS]`); then
      the grammar governs from the byte after the trigger until it is
      complete, and the text is free again (several calls may follow). This
      is how a model may either answer or call a tool (`tool_choice: "auto"`).

  Everything is a function of the tokens chosen so far, so a constrained
  sequence stays reproducible: same logits, seed and step ⇒ same tokens.
  A guarantee worth stating: when the unconstrained choice is allowed, the
  constrained choice is the same token (greedy picks the first maximum of a
  subset that contains it).
  """
  alias Vapor.Grammar
  alias Vapor.Grammar.Vocab

  defstruct [:start, :grammar, :vocab, :mode, :eos, phase: :constrained, text: "", cache: %{}, calls: 0]

  @type t :: %__MODULE__{}

  @doc "A constraint from a grammar matcher, a prepared vocabulary and the end-of-sequence ids."
  def new(%Grammar{} = g, %Vocab{} = v, eos_ids, mode \\ :strict) do
    phase = if mode == :strict, do: :constrained, else: :free
    %__MODULE__{start: g, grammar: g, vocab: v, mode: mode, eos: List.wrap(eos_ids), phase: phase}
  end

  @doc "`:all`, or `{:only, MapSet}` of the ids allowed next (and the updated cache)."
  def allowed(%__MODULE__{phase: :free} = c), do: {:all, c}
  def allowed(%__MODULE__{phase: :done} = c), do: {{:only, MapSet.new(c.eos)}, c}

  def allowed(%__MODULE__{grammar: g} = c) do
    key = g.configs

    {ids, c} =
      case c.cache do
        %{^key => ids} -> {ids, c}
        cache -> ids = Vocab.allowed(c.vocab, g); {ids, %{c | cache: Map.put(cache, key, ids)}}
      end

    ids =
      cond do
        c.mode == :strict and Grammar.dead?(g) -> MapSet.new(c.eos)
        c.mode == :strict and Grammar.complete?(g) -> MapSet.union(ids, MapSet.new(c.eos))
        true -> ids
      end

    {{:only, ids}, c}
  end

  @doc "Move past token `id` with surface `bytes`: `{:ok, c}` or `{:error, :rejected}`."
  def advance(%__MODULE__{phase: :free, mode: {:lazy, trigger}} = c, _id, bytes) do
    text = c.text <> bytes

    case :binary.match(text, trigger) do
      {at, len} ->
        after_trigger = binary_part(text, at + len, byte_size(text) - at - len)
        c = %{c | phase: :constrained, grammar: c.start, text: ""}
        consume(c, after_trigger)

      :nomatch ->
        # keep only a tail that may still begin the trigger
        keep = min(byte_size(text), byte_size(trigger) - 1)
        {:ok, %{c | text: binary_part(text, byte_size(text) - keep, keep)}}
    end
  end

  def advance(%__MODULE__{phase: :done} = c, _id, _bytes), do: {:ok, c}

  def advance(%__MODULE__{} = c, id, bytes) do
    if id in c.eos, do: {:ok, %{c | phase: :done}}, else: consume(c, bytes)
  end

  defp consume(c, bytes) do
    case Grammar.advance(c.grammar, bytes) do
      {:ok, g} ->
        c = %{c | grammar: g}

        # a lazy constraint is done with a call once it is complete and
        # nothing structural is pending; the text is free again
        if match?({:lazy, _}, c.mode) and Grammar.complete?(g) and Enum.all?(g.configs, &trailing_ws?/1),
          do: {:ok, %{c | phase: :free, calls: c.calls + 1}},
          else: {:ok, c}

      :reject ->
        {:error, :rejected}
    end
  end

  defp trailing_ws?([]), do: true
  defp trailing_ws?([{:class, _} | rest]), do: trailing_ws?(rest)
  defp trailing_ws?([{:rep_at, {:class, _}, _, _, _} | rest]), do: trailing_ws?(rest)
  defp trailing_ws?(_), do: false

  @doc "Whether the constrained output is complete (strict: a full member; lazy: no call open)."
  def complete?(%__MODULE__{phase: :free}), do: true
  def complete?(%__MODULE__{phase: :done}), do: true
  def complete?(%__MODULE__{grammar: g}), do: Grammar.complete?(g)
end
