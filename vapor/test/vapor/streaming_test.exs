defmodule Vapor.StreamingTest do
  @moduledoc """
  `Vapor.Streaming` on the language model vapor trained itself
  (`priv/lm`, 64-byte training length): until its cache fills, the stream
  computes exactly the causal model's logits; past it, the stream keeps
  reading at the quality of its first window, 14× beyond the training
  length in a cache of 64 rows — while the same model fed the same text
  with growing positions (the control) loses ~3 bits per byte to angles it
  never learned.
  """
  use ExUnit.Case, async: false
  alias Vapor.Runtime.{Session, Substrates}
  alias Vapor.Streaming, as: S

  @moduletag :native
  @moduletag timeout: 900_000
  @dir Path.expand("../../priv/lm/model", __DIR__)

  defp ids(xs), do: Vapor.Tensor.from_list(:s32, [length(xs)], xs)

  defp causal(toks, max_seq, w) do
    {:ok, %{program: p}} = Vapor.Model.load(@dir, max_seq: max_seq)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {:ok, s} = Session.open(w, comp, isa: Substrates.host_isa())
    {:ok, o, _} = Session.step(s, %{tok: ids(toks), pos: ids(Enum.to_list(0..(length(toks) - 1)))}, [:logits])
    Session.close(s)
    o.logits |> Vapor.Tensor.to_floats() |> Enum.chunk_every(256)
  end

  setup_all do
    {:ok, w} = Vapor.Runtime.Worker.start_link(exec: Vapor.TestHelpers.worker_exec(:host))
    text = Vapor.Quality.Suite.corpus("pt_holdout.txt") |> binary_part(0, 900)
    {:ok, w: w, toks: :binary.bin_to_list(text)}
  end

  test "until the cache is full, the stream is the causal model, bit for bit", %{w: w, toks: toks} do
    ref = causal(Enum.take(toks, 64), 64, w)
    {:ok, st} = S.open(@dir, sinks: 4, window: 60, worker: w)
    {st, rows} = S.feed(st, Enum.take(toks, 40))
    {st, more} = S.feed(st, Enum.slice(toks, 40, 24))
    S.close(st)
    assert rows ++ more == ref
  end

  test "14× past the training length in 64 rows: the stream reads on; growing positions (the control) collapse", %{w: w, toks: toks} do
    {:ok, st} = S.open(@dir, sinks: 4, window: 60, worker: w)
    # the cache never grows: every state input is 64 rows
    assert for({:input, n, :f32, [64, _]} <- Vapor.Program.inputs(st.comp.program), String.starts_with?(to_string(n), ["k", "v"]), do: n) != []
    {st, rows} = S.feed(st, toks)
    S.close(st)
    first = S.bits(Enum.take(rows, 64), Enum.take(toks, 64))
    stream = S.bits(rows, toks, 64)
    dense = S.bits(causal(toks, 1024, w), toks, 64)
    assert stream < first + 0.75, "stream #{stream} vs first window #{first}"
    assert dense > stream + 1.5, "dense #{dense} vs stream #{stream}"
  end
end
