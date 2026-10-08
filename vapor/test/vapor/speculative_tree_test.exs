defmodule Vapor.SpeculativeTreeTest do
  @moduledoc """
  Tree speculative decoding (`Vapor.Speculative.Tree`): several drafts
  verified in one target step, each in its own slot over the accepted
  context's shared pages. The output must be the target's plain greedy
  decoding, token for token, for every shape of tree, every page size and
  any draft — the drafts may only change the number of target steps. The
  control: a draft that proposes garbage accepts nothing (and still gives
  the same tokens), while prompt lookup on text that repeats accepts most.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Sampler, Tensor}
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Runtime.{Session, Substrates, Worker}
  alias Vapor.Speculative.Tree
  import Vapor.TestHelpers

  @moduletag :native
  @moduletag timeout: 900_000

  setup_all do
    {:ok, c} = Config.from_map(tiny_config("llama", %{"vocab_size" => 128, "max_position_embeddings" => 128}))
    ws = tiny_weights(c, 7)
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, c: c, ws: ws, w: w}
  end

  # the plain definition: a contiguous cache, one token per step
  defp greedy(c, ws, w, prompt, n) do
    {:ok, p} = Decoder.program(c, ws, max_seq: 128, max_tokens: 64)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {:ok, s} = Session.open(w, comp, isa: Substrates.host_isa())
    v = c.vocab
    ids = &Tensor.from_list(:s32, [length(&1)], &1)
    row = fn l -> binary_part(l.data, byte_size(l.data) - 4 * v, 4 * v) end
    {:ok, %{logits: l}, _} = Session.step(s, %{tok: ids.(prompt), pos: ids.(Enum.to_list(0..(length(prompt) - 1)))}, [:logits])

    {toks, _} =
      Enum.map_reduce(1..n, {Sampler.argmax(row.(l)), length(prompt)}, fn _, {t, p} ->
        {:ok, %{logits: l}, _} = Session.step(s, %{tok: ids.([t]), pos: ids.([p])}, [:logits])
        {t, {Sampler.argmax(row.(l)), p + 1}}
      end)

    toks
  end

  # a prompt that repeats itself (what retrieval answers and code edits look like)
  @prompt Enum.map(1..10, &rem(&1 * 37, 120) + 3) |> List.duplicate(3) |> List.flatten()

  test "= the target's greedy decoding for every tree shape, page size and draft", %{c: c, ws: ws, w: w} do
    n = 40
    want = greedy(c, ws, w, @prompt, n)

    forks =
      for {page, branches, depth} <- [{8, 1, 4}, {8, 4, 4}, {4, 3, 2}, {16, 2, 6}, {8, 4, 0}] do
        {:ok, tr} = Tree.open(c, ws, w, max_seq: 128, page: page, branches: max(branches, 1), max_tokens: 64)
        {got, st} = Tree.generate(tr, @prompt, n, branches: branches, depth: depth)
        assert got == want, "page #{page}, #{branches} branches, depth #{depth}"
        assert st.target_steps <= n + 1
        st.fork_wins
      end

    # the path where a forked branch wins (its pages become the context's) ran
    assert Enum.sum(forks) > 0
  end

  test "drafts change only the step count: garbage accepts nothing, prompt lookup accepts most where the output follows the context",
       %{w: w} do
    # a planted model (`Vapor.Quality.Planted`): a bigram of a short text,
    # whose greedy output follows the text's own statistics
    text = String.to_charlist("o gato subiu no muro. o gato comeu o rato. o rato subiu. ")
    pl = Vapor.Quality.Planted.bigram(text, 128)
    {:ok, c} = Config.from_map(pl.config)
    ws = pl.weights
    n = 40
    prompt = text
    want = greedy(c, ws, w, prompt, n)
    {:ok, tr} = Tree.open(c, ws, w, max_seq: 128, page: 8, branches: 4, max_tokens: 64)

    # garbage: four branches of tokens the target never chooses next
    junk = fn ctx, b, k -> for i <- 1..b, do: List.duplicate(rem(List.last(ctx) + 50 + i, 128), k) end
    {got, bad} = Tree.generate(tr, prompt, n, branches: 4, depth: 4, draft: junk)
    assert got == want
    assert bad.accepted <= 2

    {got, good} = Tree.generate(tr, prompt, n, branches: 4, depth: 4)
    assert got == want
    # tokens per target step after the prompt: well above one
    assert n / (good.target_steps - 1) > 3.0
    assert good.target_steps < bad.target_steps
  end

  test "prompt lookup: continuations of the last n-gram's earlier occurrences, most recent first, one per first token" do
    ctx = [1, 2, 3, 9, 1, 2, 3, 7, 5, 1, 2, 3]
    assert Tree.lookup(ctx, 4, 2) == [[7, 5], [9, 1]]
    assert Tree.lookup([4, 5, 6], 4, 3) == []
    assert Tree.lookup(ctx, 1, 2) == [[7, 5]]
    # copied with overlap: a loop is proposed for the whole depth
    assert Tree.lookup([8, 1, 2, 1, 2], 1, 5) == [[1, 2, 1, 2, 1]]
  end
end
