defmodule Vapor.Modal.Text do
  @moduledoc """
  Text through any `:causal_lm` spec — model-agnostic: the airlock builds
  the program, `Vapor.Modal.Runner` runs it (native worker or oracle), and
  nothing here knows the family.

    * `logits/4` — teacher-forced next-token logits for every prefix of a
      sequence (one call), the basis of `Vapor.Quality.Text.bits_per_token/3`;
    * `generate/5` — decoding one token per step with the caches carried as
      state (`Vapor.Lock.zero_state/1`), sampled by `Vapor.Sampler` (greedy
      by default), optionally with soft rows injected (`soft:`).
  """
  alias Vapor.{Lock, Sampler, Tensor}
  alias Vapor.Modal.Runner

  @doc "Logit rows (lists of floats), one per position of `ids`."
  def logits(spec, weights, ids, opts \\ []) do
    s = Keyword.get(opts, :max_seq, max(16, length(ids)))
    {:ok, p} = Lock.build(spec, weights, max_seq: s)
    n = length(ids)
    env = Map.merge(Lock.zero_state(p), %{tok: ids(ids), pos: ids(Enum.to_list(0..(n - 1)))})
    out = Runner.run(p, env, opts)
    out.logits |> Tensor.to_floats() |> Enum.chunk_every(spec.vocab)
  end

  @doc """
  Continue `prompt` (ids) by `n` tokens. Options: `temperature`, `seed`,
  `stop` (ids), `max_seq`, `worker`; `soft: {rows, mask}` for the prompt
  (lists, one row per prompt position; mask 1 replaces the token there).
  Returns the generated ids.
  """
  def generate(spec, weights, prompt, n, opts \\ []) do
    s = Keyword.get(opts, :max_seq, length(prompt) + n + 1)
    inject = Keyword.has_key?(opts, :soft)
    {:ok, pre} = Lock.build(spec, weights, max_seq: s, logits: :last, inject: inject)
    {:ok, dec} = Lock.build(spec, weights, max_seq: s, max_tokens: 1, logits: :last)
    params = Sampler.params(opts)
    stop = Keyword.get(opts, :stop, [])
    k = length(prompt)

    env = Map.merge(Lock.zero_state(pre), %{tok: ids(prompt), pos: ids(Enum.to_list(0..(k - 1))), last: ids([k - 1])})

    env =
      case Keyword.get(opts, :soft) do
        nil -> env
        {rows, mask} ->
          Map.merge(env, %{soft: Tensor.from_list(:f32, [k, spec.width], List.flatten(rows)),
                           soft_mask: Tensor.from_list(:f32, [k, 1], Enum.map(mask, &(&1 * 1.0)))})
      end

    out = Runner.run(pre, env, opts)
    first = Sampler.sample(out.logits.data, params, 0)

    Enum.reduce_while(1..n//1, {[first], carry(pre, out), k}, fn step, {acc, caches, pos} ->
      last = hd(acc)

      if last in stop or step == n do
        {:halt, {acc, caches, pos}}
      else
        e = Map.merge(caches, %{tok: ids([last]), pos: ids([pos]), last: ids([0])})
        o = Runner.run(dec, e, opts)
        {:cont, {[Sampler.sample(o.logits.data, params, step) | acc], carry(dec, o), pos + 1}}
      end
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  defp carry(p, out), do: Map.new(p.state, fn {i, o} -> {i, Map.fetch!(out, o)} end)

  defp ids(list), do: Tensor.from_list(:s32, [length(list)], list)
end
