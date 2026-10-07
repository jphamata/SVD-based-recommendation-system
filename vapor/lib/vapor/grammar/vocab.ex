defmodule Vapor.Grammar.Vocab do
  @moduledoc """
  A tokenizer's vocabulary prepared for constrained decoding: the token
  bytes in a trie, and the class of *string-safe* tokens.

  Which tokens may come next is decided by walking the trie with the
  grammar's configuration set (`Vapor.Grammar.step_configs/3`): a subtree is
  abandoned the moment its prefix is rejected, and every token whose bytes
  survive is allowed. Inside a JSON string with no length limit most of the
  vocabulary is allowed and the walk would visit all of it, so tokens are
  split once, at build time: a token is string-safe when its bytes are
  printable, contain no `"` or `\\`, and form whole UTF-8 characters — such a
  token takes every unbounded string configuration to an equal one, so all
  of them are allowed together without a walk, and only the trie of the
  other tokens is walked (the precomputation in the spirit of XGrammar's
  context-independent tokens).

  Special tokens (control tokens such as `<|im_end|>`) and empty tokens are
  never produced by a grammar; the end of a constrained output is signalled
  by the end-of-sequence ids given to `Vapor.Grammar.Constraint`.
  """
  alias Vapor.Grammar

  defstruct size: 0, trie: nil, unsafe: nil, safe: MapSet.new(), excluded: MapSet.new(), by_len: %{}

  @type t :: %__MODULE__{}

  @doc "Build from a `Vapor.Tokenizer` (or from `{surfaces_tuple, special_ids}`)."
  def build(%Vapor.Tokenizer{} = tk) do
    special = MapSet.new(tk.special, &elem(&1, 1))
    build({tk.surface, special})
  end

  def build({surfaces, special}) when is_tuple(surfaces) do
    tokens =
      for id <- 0..(tuple_size(surfaces) - 1),
          s = elem(surfaces, id),
          s != "" and not MapSet.member?(special, id),
          do: {s, id}

    {safe, unsafe} = Enum.split_with(tokens, fn {s, _} -> safe?(s) end)

    # string-safe tokens by their length in characters — code points, as JSON
    # Schema and the grammar count them, not graphemes (for maxLength)
    by_len = safe |> Enum.group_by(fn {s, _} -> length(String.to_charlist(s)) end, &elem(&1, 1))

    %__MODULE__{size: tuple_size(surfaces), trie: trie(tokens), unsafe: trie(unsafe),
                safe: MapSet.new(safe, &elem(&1, 1)), by_len: by_len,
                excluded: MapSet.union(MapSet.new(special), MapSet.new(for id <- 0..(tuple_size(surfaces) - 1), elem(surfaces, id) == "", do: id))}
  end

  @doc "Token ids the grammar admits next (a MapSet)."
  def allowed(%__MODULE__{} = v, %Grammar{configs: cs, defs: defs}) do
    cs = Enum.to_list(cs)

    if cs != [] and Enum.all?(cs, &in_string?/1) do
      walk(v.unsafe, cs, defs, safe_room(v, cs))
    else
      walk(v.trie, cs, defs, MapSet.new())
    end
  end

  # every configuration inside a JSON string body, at a character boundary
  defp in_string?([{:str_in, :body, _, 0, _} | _]), do: true
  defp in_string?(_), do: false

  # the safe tokens that fit: all of them without a maximum, else those of
  # at most `max − n` characters (a safe token adds exactly its characters)
  defp safe_room(v, cs) do
    room =
      cs
      |> Enum.map(fn [{:str_in, :body, n, 0, o} | _] -> if o.max == nil, do: :inf, else: o.max - n end)
      |> Enum.reduce(0, fn r, acc -> if acc == :inf or r == :inf, do: :inf, else: max(r, acc) end)

    case room do
      :inf -> v.safe
      r -> for({len, ids} <- v.by_len, len <= r, id <- ids, into: MapSet.new(), do: id)
    end
  end


  defp walk({_ids, children}, cs, defs, acc) do
    Enum.reduce(children, acc, fn {byte, {ids, _} = child}, acc ->
      case Grammar.step_configs(cs, byte, defs) do
        [] -> acc
        next -> walk(child, next, defs, Enum.reduce(ids, acc, &MapSet.put(&2, &1)))
      end
    end)
  end

  # printable, no quote or backslash, whole UTF-8 characters
  defp safe?(s) do
    String.valid?(s) and not String.contains?(s, ["\"", "\\"]) and
      not Enum.any?(:binary.bin_to_list(s), &(&1 < 0x20 or &1 == 0x7F))
  end

  # {ids ending here, %{byte => child}}
  defp trie(tokens) do
    tokens
    |> Enum.group_by(fn {s, _} -> :binary.first(s) end)
    |> Map.new(fn {b, group} -> {b, node(group, 1)} end)
    |> then(&{[], &1})
  end

  defp node(group, depth) do
    {here, deeper} = Enum.split_with(group, fn {s, _} -> byte_size(s) == depth end)

    children =
      deeper
      |> Enum.group_by(fn {s, _} -> :binary.at(s, depth) end)
      |> Map.new(fn {b, g} -> {b, node(g, depth + 1)} end)

    {Enum.map(here, &elem(&1, 1)), children}
  end
end
