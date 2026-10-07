defmodule Vapor.Lock.Contract do
  @moduledoc """
  The program contracts of the model airlock: what a built program must
  look like for the core to drive it without knowing the model.

  The airlock checks every program an adapter builds against the contract
  its spec declares — a faulty adapter is stopped at the lock, not inside
  the engine. A program is *self-describing* (its inputs, outputs and
  state are typed terms), so the contract is a statement about sorts:

  | interface | inputs | outputs | state |
  |---|---|---|---|
  | `:causal_lm` | `tok, pos : s32[t]` (no `pos` for a `:recurrent` spec) (+ `last`, `sampling`, `table`, `slot`, `soft`, `soft_mask` by option) | `logits : f32[·, vocab]` and/or `hidden : f32[·, width]` | every `{x, x_next}` pair: same sort |
  | `:encoder` | `rows : f32[T, k]` (or, a text tower, `tok : s32[T]` and `pick : s32[1]`), `horizon : s32[T]` | `hidden : f32[T, width]` | none |
  | `:codec` (encode) | `rows : f32[T, k]` | `codes : s32[T]` | none |
  | `:codec` (decode) | `codes : s32[T]` | `rows : f32[T, k]` | none |
  | `:map` | `rows : f32[T, k]` | `out : f32[T, width]` | none |
  """
  alias Vapor.{Program, Rejection}
  alias Vapor.Algebra.Term
  alias Vapor.Lock.Spec

  @doc "`:ok` or a rejection naming the broken clause."
  def check(%Spec{interface: iface} = spec, %Program{} = p, opts \\ []) do
    ins = Map.new(Program.inputs(p), fn {:input, n, dt, s} -> {n, {dt, s}} end)
    # a model of several programs (Whisper: encoder + decoder) declares the
    # contract of each extra part; the default part keeps the spec's own
    iface = case Keyword.get(opts, :part) do
      nil -> iface
      part -> Map.get(spec.parts, part, {:undeclared, part})
    end

    with {:ok, outs} <- out_sorts(p),
         :ok <- state(p, ins, outs) do
      clauses(iface, spec, ins, outs, opts)
    end
  end

  defp out_sorts(p) do
    Enum.reduce_while(p.outputs, {:ok, %{}}, fn {n, t}, {:ok, acc} ->
      case Term.infer(t) do
        {:ok, sort} -> {:cont, {:ok, Map.put(acc, n, sort)}}
        err -> {:halt, err}
      end
    end)
  end

  defp state(p, ins, outs) do
    case Enum.find(p.state, fn {i, o} -> Map.get(ins, i) == nil or Map.get(outs, o) == nil or
                                       Term.shape_max(elem(ins[i], 1)) != Term.shape_max(elem(outs[o], 1)) or
                                       elem(ins[i], 0) != elem(outs[o], 0) end) do
      nil -> :ok
      {i, o} -> no({:state, i, o}, "state #{i} ← #{o} with equal sorts")
    end
  end

  defp clauses(:causal_lm, spec, ins, outs, _opts) do
    with :ok <- sort(ins, :tok, :s32, 1),
         # a recurrent model (an SSM) has no positions: its state is the history
         :ok <- (if :recurrent in spec.features and not Map.has_key?(ins, :pos), do: :ok, else: sort(ins, :pos, :s32, 1)),
         :ok <- need(Map.has_key?(outs, :logits) or Map.has_key?(outs, :hidden), :outputs, "logits or hidden"),
         :ok <- last_width(outs, :logits, spec.vocab),
         :ok <- last_width(outs, :hidden, spec.width) do
      :ok
    end
  end

  # an encoder reads rows (image patches, audio frames…) or token ids (a
  # text tower), plus the horizon of every row
  defp clauses(:encoder, spec, ins, outs, _opts) do
    with :ok <- (if Map.has_key?(ins, :tok), do: sort(ins, :tok, :s32, 1), else: sort(ins, :rows, :f32, 2)),
         :ok <- sort(ins, :horizon, :s32, 1),
         :ok <- need(Map.has_key?(outs, :hidden), :outputs, "hidden"),
         :ok <- last_width(outs, :hidden, spec.width) do
      :ok
    end
  end

  defp clauses(:codec, _spec, ins, outs, _opts) do
    cond do
      Map.has_key?(outs, :codes) -> with :ok <- sort(ins, :rows, :f32, 2), do: sort(outs, :codes, :s32, 1)
      Map.has_key?(outs, :rows) -> with :ok <- sort(ins, :codes, :s32, 1), do: sort(outs, :rows, :f32, 2)
      true -> no(:outputs, "codes (encode) or rows (decode)")
    end
  end

  defp clauses({:undeclared, part}, _spec, _ins, _outs, _opts), do: no({:part, part}, "a part the spec declares")

  defp clauses(:map, spec, ins, outs, _opts) do
    with :ok <- sort(ins, :rows, :f32, 2),
         :ok <- need(Map.has_key?(outs, :out), :outputs, "out"),
         :ok <- last_width(outs, :out, spec.width) do
      :ok
    end
  end

  defp clauses(other, _spec, _ins, _outs, _opts), do: no({:interface, other}, ":causal_lm, :encoder, :codec or :map")

  defp sort(map, name, dt, rank) do
    case Map.get(map, name) do
      {^dt, s} when length(s) == rank -> :ok
      nil -> no({:contract, name}, "#{name} : #{dt} of rank #{rank} present")
      got -> no({:contract, name}, "#{name} : #{dt} of rank #{rank} (got #{inspect(got)})")
    end
  end

  defp last_width(outs, name, w) do
    case Map.get(outs, name) do
      nil -> :ok
      {:f32, s} when w == nil or (is_integer(w) and length(s) == 2) ->
        if w == nil or List.last(s) == w, do: :ok, else: no({:contract, name}, "#{name} : f32[·, #{w}]")
      got -> no({:contract, name}, "#{name} : f32[·, #{inspect(w)}] (got #{inspect(got)})")
    end
  end

  defp need(true, _n, _b), do: :ok
  defp need(false, n, b), do: no({:contract, n}, b)

  defp no(node, bound), do: {:error, Rejection.new(node, bound, "fix the adapter: the program breaks its declared contract")}
end
