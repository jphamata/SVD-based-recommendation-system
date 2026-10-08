defmodule Vapor.EngineTest do
  @moduledoc """
  Phase P5 — generation: deterministic sampling and the continuous-batching
  engine over a paged KV cache. The central property is batch invariance:
  a request's tokens do not depend on what else is running, how its prompt
  was chunked, where its pages are, or how many threads the worker has.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Engine, Sampler, Tensor}
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Runtime.{Session, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :native
  @moduletag timeout: 900_000

  defp model do
    {:ok, c} = Config.from_map(tiny_config("llama", %{"vocab_size" => 128}))
    {c, tiny_weights(c, 3)}
  end

  defp engine(c, ws, opts \\ []) do
    # (Keyword.merge: with `++` the defaults came first and won, so the
    # overrides below were silently ignored)
    {:ok, e} = Engine.start_link(Keyword.merge([config: c, weights: ws, max_seq: 64, page: 8, sequences: 4, step_tokens: 16], opts))
    e
  end

  # the same greedy/sampled continuation by the plain contiguous program,
  # one sequence at a time, token by token
  defp reference(c, ws, prompt, n, params) do
    {:ok, p} = Decoder.program(c, ws, max_seq: 64)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, s} = Session.open(w, comp, isa: Substrates.host_isa())
    ids = &Tensor.from_list(:s32, [length(&1)], &1)
    last = fn t -> t.data |> Vapor.F32.decode() |> Enum.map(&Vapor.F32.to_float/1) |> Enum.chunk_every(c.vocab) |> List.last() end
    k = length(prompt)
    {:ok, %{logits: l}, _} = Session.step(s, %{tok: ids.(prompt), pos: ids.(Enum.to_list(0..(k - 1)))}, [:logits])

    # plain temperature sampling happens on the substrate: a one-op program
    sampler = Vapor.Compile.Lower.lower(Vapor.Program.new(next: Vapor.Algebra.Term.sample(
      Vapor.Algebra.Term.input(:l, :f32, [1, c.vocab]), Vapor.Algebra.Term.input(:p, :f32, [1, 2])))) |> elem(1)

    choose = fn row, i ->
      if Sampler.native?(params) do
        {inv, u} = Sampler.native(params, i)
        {:ok, r} = Vapor.Runtime.Native.run(w, sampler, %{l: Tensor.from_list(:f32, [1, c.vocab], row), p: Tensor.from_list(:f32, [1, 2], [inv, u])},
                                            isa: Substrates.host_isa(), mode: :native)
        hd(Tensor.to_list(r.outputs.next))
      else
        Sampler.sample(row, params, i)
      end
    end

    {toks, _} =
      Enum.map_reduce(0..(n - 1), l, fn i, l ->
        t = choose.(last.(l), i)
        {:ok, %{logits: l2}, _} = Session.step(s, %{tok: ids.([t]), pos: ids.([k + i])}, [:logits])
        {t, l2}
      end)

    toks
  end

  test "sampler: greedy is the first maximum; sampling is a pure function of (logits, seed, step); exp within 2 ulp on [−700, 0]" do
    assert Sampler.argmax([1.0, 3.0, 3.0, -1.0]) == 1
    l = for i <- 0..199, do: :math.sin(i * 0.7) * 4
    p = Sampler.params(temperature: 0.9, top_k: 40, top_p: 0.95, seed: 7)
    a = for s <- 0..50, do: Sampler.sample(l, p, s)
    assert a == for(s <- 0..50, do: Sampler.sample(l, p, s))
    assert length(Enum.uniq(a)) > 3
    top40 = l |> Enum.with_index() |> Enum.sort_by(fn {x, _} -> -x end) |> Enum.take(40) |> Enum.map(&elem(&1, 1))
    assert Enum.all?(a, &(&1 in top40))
    # top_k: 1 is greedy whatever the temperature
    assert Enum.uniq(for s <- 0..20, do: Sampler.sample(l, Sampler.params(temperature: 2.0, top_k: 1, seed: s), s)) == [Sampler.argmax(l)]

    for x <- Enum.map(0..2800, &(-&1 / 4.0)) do
      assert abs(Sampler.exp(x) - :math.exp(x)) <= 4.5e-16 * :math.exp(x)
    end
  end

  test "engine = the contiguous program token by token (greedy and sampled)" do
    {c, ws} = model()
    e = engine(c, ws)
    prompt = [5, 99, 3, 17, 42, 8, 1]

    for params <- [[], [temperature: 1.0, top_p: 0.9, seed: 11], [temperature: 0.7, seed: 4]] do
      {:ok, got, :length, usage} = Engine.complete(e, prompt, [max_tokens: 12] ++ params)
      assert got == reference(c, ws, prompt, 12, Sampler.params(params))
      assert usage == %{prompt_tokens: 7, completion_tokens: 12}
    end
  end

  test "batch invariance: 9 concurrent requests (more than the 4 sequences) = each alone; any chunking, threads" do
    {c, ws} = model()
    reqs = for i <- 1..9, do: {Enum.map(1..(3 + rem(i * 7, 20)), &rem(&1 * i * 13, 128)), [max_tokens: 5 + rem(i, 6), temperature: 1.0, seed: i]}

    alone =
      for {prompt, opts} <- reqs do
        e = engine(c, ws, sequences: 1, step_tokens: 64)
        {:ok, ids, _, _} = Engine.complete(e, prompt, opts)
        ids
      end

    for opts <- [[], [step_tokens: 5], [threads: 3, step_tokens: 7]] do
      e = engine(c, ws, opts)
      tasks = for {prompt, o} <- reqs, do: Task.async(fn -> Engine.complete(e, prompt, o) end)
      got = Enum.map(tasks, &(Task.await(&1, 300_000) |> elem(1)))
      assert got == alone, inspect(opts)
      info = Engine.info(e)
      assert info.active == 0 and info.queued == 0
      # batching happened: more rows than steps
      assert info.stats.rows > info.stats.steps
    end
  end

  test "a worker lost under the engine ends the requests it held with :error; the engine reopens and keeps serving" do
    {c, ws} = model()
    e = engine(c, ws)
    {:ok, want, :length, _} = Engine.complete(e, [4, 5, 6], max_tokens: 6)
    # kill the worker's OS process while the engine is idle
    port = :sys.get_state(:sys.get_state(e).worker).port
    {:os_pid, pid} = Port.info(port, :os_pid)
    System.cmd("kill", ["-9", Integer.to_string(pid)])

    assert {:ok, [], :error, _} = Engine.complete(e, [4, 5, 6], max_tokens: 6)
    assert {:ok, ^want, :length, _} = Engine.complete(e, [4, 5, 6], max_tokens: 6)
    assert Engine.info(e).stats.failures == 1
  end

  test "replicas: a pool answers every request as one engine does; a dead replica ends only its own requests and is replaced" do
    {c, ws} = model()
    reqs = for i <- 1..6, do: {Enum.map(1..(3 + rem(i * 5, 9)), &rem(&1 * i * 7, 128)), [max_tokens: 8 + i, temperature: 0.9, seed: i]}
    one = engine(c, ws)
    want = for {p, o} <- reqs, do: Engine.complete(one, p, o) |> elem(1)

    pool = engine(c, ws, replicas: 2, threads: 2)
    tasks = for {p, o} <- reqs, do: Task.async(fn -> Engine.complete(pool, p, o) end)
    assert Enum.map(tasks, &(Task.await(&1, 300_000) |> elem(1))) == want
    info = Engine.info(pool)
    assert info.replicas == 2 and info.active == 0 and info.queued == 0
    assert info.stats.tokens == Enum.sum(for {_, o} <- reqs, do: o[:max_tokens])
    assert Map.values(info.load) == [0, 0]

    # one sequence per replica: requests 1 and 3 go to replica 0 (3 waits in its queue)
    small = engine(c, ws, replicas: 2, sequences: 1)
    long = [1, 2, 3]
    refs = for _ <- 1..4, do: elem(Engine.generate(small, long, max_tokens: 60), 1)
    Process.exit(:sys.get_state(small).replicas[0], :kill)
    [r1, r2, r3, r4] = Enum.map(refs, &Engine.collect(&1, [], 60_000))
    assert {:ok, _, :error, _} = r3
    assert elem(r1, 2) in [:error, :length]
    {:ok, full, :length, _} = Engine.complete(one, long, max_tokens: 60)
    assert {:ok, ^full, :length, _} = r2
    assert {:ok, ^full, :length, _} = r4
    assert {:ok, ^full, :length, _} = Engine.complete(small, long, max_tokens: 60)
    assert Engine.info(small).restarts == 1
  end

  test "a request lives as long as its receiver: a dead receiver or cancel/2 frees its slot; the others are unaffected" do
    {c, ws} = model()
    e = engine(c, ws)
    me = self()
    {:ok, want, :length, _} = Engine.complete(e, [9, 8, 7], max_tokens: 20, temperature: 0.7, seed: 4)

    # three receivers that vanish mid-generation (a client hanging up)
    doomed =
      for i <- 1..3 do
        spawn(fn ->
          {:ok, _} = Engine.generate(e, [i, i + 1, i + 2], max_tokens: 56)
          send(me, {:started, self()})
          Process.sleep(:infinity)
        end)
      end

    for _ <- doomed, do: assert_receive({:started, _}, 10_000)
    {:ok, ref} = Engine.generate(e, [9, 8, 7], max_tokens: 20, temperature: 0.7, seed: 4)
    Enum.each(doomed, &Process.exit(&1, :kill))
    assert {:ok, ^want, :length, _} = Engine.collect(ref, [], 60_000)

    info = Engine.info(e)
    assert info.stats.cancelled == 3 and info.active == 0 and info.queued == 0
    assert info.stats.tokens < 20 + 20 + 3 * 56

    # on purpose, running or still queued
    {:ok, r} = Engine.generate(e, [1, 2, 3], max_tokens: 56)
    Engine.cancel(e, r)
    assert_receive {:vapor, ^r, {:done, :cancelled, _}}, 10_000

    # through a pool: the pool watches the receiver and withdraws from its replica
    pool = engine(c, ws, replicas: 2)
    p = spawn(fn -> {:ok, _} = Engine.generate(pool, [1, 2, 3], max_tokens: 56); send(me, :started); Process.sleep(:infinity) end)
    assert_receive :started, 10_000
    Process.exit(p, :kill)
    Process.sleep(200)
    pi = Engine.info(pool)
    assert pi.stats.cancelled == 1 and pi.active == 0 and Map.values(pi.load) == [0, 0]
    assert {:ok, _, :length, _} = Engine.complete(pool, [1, 2, 3], max_tokens: 4)
  end

  test "storage: :bf16 — the engine's tokens are those of the f32 engine over the rounded weights" do
    {c, ws} = model()
    rounded = Map.new(ws, fn {k, t} -> {k, if(length(t.shape) == 2, do: Tensor.widen(Tensor.to_bf16(t)), else: t)} end)
    prompt = [5, 99, 3, 17, 42, 8, 1]

    for params <- [[], [temperature: 0.8, seed: 3]] do
      {:ok, a, _, _} = Engine.complete(engine(c, ws, storage: :bf16), prompt, [max_tokens: 10] ++ params)
      {:ok, b, _, _} = Engine.complete(engine(c, rounded), prompt, [max_tokens: 10] ++ params)
      assert a == b
    end
  end

  test "circular cache: a window binding on every layer holds only the pages it can read, with the bits of the uncapped cache" do
    {:ok, c} = Config.from_map(tiny_config("mistral", %{"vocab_size" => 128, "sliding_window" => 8, "max_position_embeddings" => 64}))
    ws = tiny_weights(c, 5)
    reqs = for i <- 1..3, do: {Enum.map(1..(17 + 3 * i), &rem(&1 * (i + 2) * 7, 128)), [max_tokens: 24, temperature: 1.0, seed: i]}

    # 4 pages in the pool: each request alone would need 6 or 7 uncapped
    # (44..50 positions / 8); the ring needs ⌈(8 + 9 − 1)/8⌉ = 2 (the bound
    # and its tightness: Vapor.SlidingWindowTest, on all-row logits)
    e = engine(c, ws, pages: 4, sequences: 2, step_tokens: 9)
    info = Engine.info(e)
    assert {info.window, info.ring_pages} == {8, 2}

    tasks = for {prompt, o} <- reqs, do: Task.async(fn -> Engine.complete(e, prompt, o) end)
    got = Enum.map(tasks, &(Task.await(&1, 300_000) |> elem(1)))

    # the reference: the contiguous program (every position kept), token by token
    for {{prompt, o}, ids} <- Enum.zip(reqs, got) do
      assert ids == reference(c, ws, prompt, 24, Sampler.params(o))
    end

    # the ring is really reusing pages: two sequences of 44+ positions ran at
    # once in 4 pages of 8 — uncapped, the pool cannot hold even one
    assert Engine.info(e).stats.rows > Engine.info(e).stats.steps
    {:ok, h} = Config.from_map(tiny_config("mistral", %{"vocab_size" => 128, "sliding_window" => 8, "max_position_embeddings" => 64,
                                                        "layer_types" => ["sliding_attention", "full_attention"]}))
    assert h.layer_types == [:sliding, :full]
    # layer_types is read or refused, never skipped
    assert {:error, %Vapor.Rejection{}} =
             Config.from_map(tiny_config("mistral", %{"sliding_window" => 8, "layer_types" => ["sliding_attention"]}))
    assert {:error, %Vapor.Rejection{}} =
             Config.from_map(tiny_config("mistral", %{"layer_types" => ["sliding_attention", "full_attention"]}))
    hybrid = engine(h, tiny_weights(h, 5), pages: 4, sequences: 2, step_tokens: 9)
    assert Engine.info(hybrid).window == nil
    assert {:error, {:kv_pages, 6, 4}} = Engine.generate(hybrid, elem(hd(reqs), 0), max_tokens: 24)

    # a hybrid (a global layer among sliding ones) keeps every position
    {:ok, g} = Config.from_map(tiny_config("mistral", %{"vocab_size" => 128, "sliding_window" => 8}))
    assert Config.ring_window(%{g | layer_types: [:sliding, :full]}, 64) == nil
    assert Config.ring_window(g, 64) == 8
    assert Config.ring_window(g, 8) == nil
  end

  test "stop conditions and refusals" do
    {c, ws} = model()
    e = engine(c, ws)
    assert {:error, :empty_prompt} = Engine.generate(e, [])
    assert {:error, :token_out_of_range} = Engine.generate(e, [128])
    assert {:error, {:context_length, 70, 64}} = Engine.generate(e, [1, 2], max_tokens: 68)
    # more pages than the pool has: refused, not queued forever
    small = engine(c, ws, pages: 4)
    assert {:error, {:kv_pages, 5, 4}} = Engine.generate(small, [1, 2], max_tokens: 38)

    {:ok, ids, :length, _} = Engine.complete(e, [1, 2, 3], max_tokens: 4)
    eos = hd(ids)
    {:ok, e2} = Engine.start_link(config: %{c | eos: eos}, weights: ws, max_seq: 64, page: 8, sequences: 2)
    assert {:ok, [], :eos, %{completion_tokens: 0}} = Engine.complete(e2, [1, 2, 3], max_tokens: 4)
  end
end
