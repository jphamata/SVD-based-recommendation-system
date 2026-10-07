defmodule Vapor.Streaming do
  @moduledoc """
  **Unbounded context in constant memory, every distance in range** — the
  attention-sink stream of StreamingLLM (Xiao et al. 2024) with RoPE
  applied in the cache's own frame.

  Two things break a RoPE transformer fed past its training length:

  1. *Out-of-distribution angles.* The score of a query at position m and
     a key at n is `qᵀ R_{n−m} k`: only the distance matters, but a
     distance never seen in training is a rotation the model never
     learned to read.
  2. *Memory.* A cache of every past key grows without bound — and a
     sliding window that simply drops the oldest keys loses the first
     tokens, on which trained models park a large share of their
     attention (the "sinks"); quality collapses when they leave.

  The cure here is first-principles RoPE algebra rather than a new
  kernel. Because `R_a R_b = R_{a+b}` and a score depends only on
  `n − m`, *positions may be re-based at every step*: the cache stores
  keys **unrotated**, `s = sinks + window` rows (the first `sinks` tokens
  pinned, the rest a ring overwritten in turn), and each step gathers the
  rows in logical order and rotates them at their cache positions
  `0 … s − 1`, the new query at `s − 1`. No distance ever exceeds `s` —
  chosen within the training length — and no table, no position counter,
  no memory grows: the stream is unbounded.

  The model is any Llama-family checkpoint `Vapor.Model.load/2` accepts
  (standard attention), compiled with `kv: {:stream, sinks, window}`; the
  orchestration (which ring row a token is written to, the gather order)
  is computed here, in integers. Until the cache is full the computation
  is exactly the ordinary causal model's (tested: the same bits).
  """
  alias Vapor.Runtime.{Session, Substrates}
  alias Vapor.Tensor

  defstruct [:session, :comp, :sinks, :window, :cap, :vocab, p: 0]

  @doc """
  Open a stream on `dir` (a checkpoint) or a loaded `program`. Options:
  `sinks` (4), `window` (the rest of the training length: `cap − sinks`),
  `cap` (`sinks + window`), `worker` (required: the session's worker).
  """
  def open(dir, opts) do
    sinks = Keyword.get(opts, :sinks, 4)
    cap = Keyword.get_lazy(opts, :cap, fn -> sinks + Keyword.fetch!(opts, :window) end)
    window = cap - sinks

    with {:ok, %{program: p}} <- Vapor.Model.load(dir, max_seq: cap, kv: {:stream, sinks, window}, max_tokens: cap),
         {:ok, comp} <- Vapor.Compile.Lower.lower(p),
         {:ok, s} <- Session.open(Keyword.fetch!(opts, :worker), comp, isa: Substrates.host_isa()) do
      {:ok, %__MODULE__{session: s, comp: comp, sinks: sinks, window: window, cap: cap}}
    end
  end

  @doc """
  Feed tokens: `{stream, logits}` with one row of logits per token (row t
  predicts token t + 1). While the cache is not full, tokens go in chunks;
  after, one per step.
  """
  def feed(%__MODULE__{} = st, tokens) do
    {st, rows} = chunks(st, tokens, [])
    {st, rows}
  end

  defp chunks(st, [], acc), do: {st, acc |> Enum.reverse() |> Enum.concat()}

  defp chunks(st, tokens, acc) do
    n = if st.p < st.cap, do: min(length(tokens), st.cap - st.p), else: 1
    {now, rest} = Enum.split(tokens, n)
    {st, rows} = step(st, now)
    chunks(st, rest, [rows | acc])
  end

  defp step(st, toks) do
    n = length(toks)
    positions = Enum.to_list(st.p..(st.p + n - 1))
    s32 = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end

    {wrow, pos, gidx} =
      if st.p + n <= st.cap do
        # filling: the cache frame is the sequence's own
        {positions, positions, Enum.to_list(0..(st.cap - 1))}
      else
        p = st.p
        ring = fn x -> st.sinks + rem(x - st.sinks, st.window) end
        window = for x <- (p - st.window + 1)..p, do: ring.(x)
        {[ring.(p)], [st.cap - 1], Enum.to_list(0..(st.sinks - 1)//1) ++ window}
      end

    {:ok, out, _} = Session.step(st.session, %{tok: s32.(toks), wrow: s32.(wrow), pos: s32.(pos), gidx: s32.(gidx)}, [:logits])
    [_, v] = out.logits.shape
    rows = out.logits |> Tensor.to_floats() |> Enum.chunk_every(v)
    {%{st | p: st.p + n}, rows}
  end

  @doc "Close the stream's session."
  def close(%__MODULE__{session: s}), do: Session.close(s)

  @doc """
  Bits per token of `tokens` under teacher forcing, from token `from` on
  (the first `from` predictions are not scored): the measure of how well
  a stream keeps reading.
  """
  def bits(rows, tokens, from \\ 0) do
    pairs = Enum.zip(Enum.drop(rows, -1), tl(tokens)) |> Enum.drop(from)

    total =
      Enum.reduce(pairs, 0.0, fn {row, y}, acc ->
        m = Enum.max(row)
        lse = m + :math.log(Enum.reduce(row, 0.0, fn v, a -> a + :math.exp(v - m) end))
        acc + (lse - Enum.at(row, y)) / :math.log(2)
      end)

    total / max(length(pairs), 1)
  end
end
