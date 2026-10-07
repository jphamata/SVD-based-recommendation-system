defmodule Vapor.Speculative do
  @moduledoc """
  Speculative decoding (greedy): a cheap *draft* model proposes `k`
  tokens one at a time; the *target* model scores all of them in one step
  (`k + 1` rows); the longest prefix the target agrees with is kept, plus
  the target's own next token.

  The guarantee is not "close to" the target's greedy output — it is
  *equal* to it, token for token: the target's logits for a row do not
  depend on how many rows share the step (batch invariance of every
  kernel), so verifying `k + 1` positions at once gives the very bits that
  `k + 1` single decoding steps would. The draft affects only speed.

  Rejected positions leave stale rows in both KV caches; they lie beyond
  the accepted length and are overwritten before any row attends to them
  (row `p` reads rows `0 … p` only), so no cleanup is needed.

  Both models are resident sessions (`Vapor.Runtime.Session`) of programs
  with contiguous caches and `logits: :all`, built for at least `k + 1`
  rows per step.
  """
  alias Vapor.{Sampler, Tensor}
  alias Vapor.Runtime.Session

  @doc """
  Generate up to `n` tokens after `prompt` with target `t` and draft `d`
  (sessions; `vocab` the shared vocabulary size). Returns
  `{tokens, %{target_steps, draft_steps, proposed, accepted}}`.
  """
  def generate(%Session{} = t, %Session{} = d, prompt, n, k, vocab, opts \\ []) do
    eos = Keyword.get(opts, :eos, [])
    plen = length(prompt)
    tl = step!(t, prompt, 0, vocab)
    _ = step!(d, prompt, 0, vocab)
    first = Sampler.argmax(List.last(tl))
    st = %{target_steps: 1, draft_steps: 1, proposed: 0, accepted: 0}
    loop([first], first, plen, [first], n, k, t, d, vocab, eos, st)
  end

  # out: generated so far (reversed); last: last accepted token at `pos`;
  # pending: tokens the draft has not processed yet (ending with `last`)
  defp loop(out, _last, _pos, _pending, n, _k, _t, _d, _v, _eos, st) when length(out) >= n,
    do: {out |> Enum.reverse() |> Enum.take(n), st}

  defp loop([last | _] = out, last, pos, pending, n, k, t, d, v, eos, st) do
    if last in eos do
      {Enum.reverse(out), st}
    else
      k = min(k, n - length(out))
      # the draft catches up on `pending`, then proposes k tokens
      p0 = pos - length(pending) + 1
      dl = step!(d, pending, p0, v)
      d1 = Sampler.argmax(List.last(dl))

      {props, _} =
        Enum.map_reduce(1..(k - 1)//1, d1, fn i, prev ->
          [row] = step!(d, [prev], pos + i, v)
          {prev, Sampler.argmax(row)}
        end)
        |> then(fn {ps, lastp} -> {ps ++ [lastp], nil} end)

      # the target scores last, d1 … dk at pos … pos + k in one step
      rows = step!(t, [last | props], pos, v)
      greedy = Enum.map(rows, &Sampler.argmax/1)
      agree = Enum.zip(props, greedy) |> Enum.take_while(fn {a, b} -> a == b end) |> length()
      kept = Enum.take(props, agree) ++ [Enum.at(greedy, agree)]

      # the draft processed last, d1 … d(k-1): it still needs whatever it
      # has not seen of the kept tokens
      seen = min(agree, k - 1)
      pending = Enum.drop(kept, seen)
      st = %{st | target_steps: st.target_steps + 1, draft_steps: st.draft_steps + k, proposed: st.proposed + k, accepted: st.accepted + agree}
      out = Enum.reverse(kept) ++ out
      out = trim_eos(out, eos)
      loop(out, hd(out), pos + length(kept), pending, n, k, t, d, v, eos, st)
    end
  end

  # stop at the first EOS among the kept tokens
  defp trim_eos(out, []), do: out

  defp trim_eos(out, eos) do
    rev = Enum.reverse(out)

    case Enum.find_index(rev, &(&1 in eos)) do
      nil -> out
      i -> rev |> Enum.take(i + 1) |> Enum.reverse()
    end
  end

  defp step!(s, toks, p0, v) do
    n = length(toks)
    ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end
    {:ok, %{logits: l}, _} = Session.step(s, %{tok: ids.(toks), pos: ids.(Enum.to_list(p0..(p0 + n - 1)))}, [:logits])
    for i <- 0..(n - 1), do: binary_part(l.data, i * v * 4, v * 4)
  end
end
