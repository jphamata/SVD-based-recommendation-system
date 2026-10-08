defmodule Vapor.ThreadsTest do
  @moduledoc """
  HPC — intra-operation data parallelism. A worker with a pool of `n`
  threads splits each partitionable kernel call into contiguous row ranges;
  every row is computed by the same instructions in the same order, so the
  result must be bit-identical for every `n` (and to the oracle), including
  per-thread scratch (RoPE, attention) and output rows `ldy` apart (GEMV
  column ranges).
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Runtime.{Native, Session, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :native
  @moduletag timeout: 600_000

  defp lower!(p), do: elem(Lower.lower(p), 1)

  defp cases do
    {:ok, c} = Config.from_map(tiny_config("qwen2", %{"hidden_size" => 256, "intermediate_size" => 512, "vocab_size" => 128}))
    ws = tiny_weights(c)
    toks = [5, 3, 127, 0, 64, 9, 9, 1, 2, 3, 77]
    n = length(toks)
    menv = Map.merge(Decoder.empty_caches(c, 16), %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})
    {:ok, f32} = Decoder.program(c, ws, max_seq: 16)
    {:ok, q4} = Decoder.program(c, ws, max_seq: 16, quantize: :sb4)

    canon =
      for {name, prog, env} <- canon_programs(), do: {name, lower!(prog), env}

    canon ++
      [{"attention block", lower!(attention_block()), attention_env(13)},
       # one row: the partition is over kv heads
       {"attention, one decoding row", lower!(attention_block(t: 1)), attention_env(1)},
       {"paged attention, one decoding row", lower!(attention_block(t: 1, paged: {8, 4, 1})),
        Map.merge(attention_env(1), %{slot: Tensor.from_list(:s32, [1], [0]), table: Tensor.from_list(:s32, [1, 4], [3, 1, 0, 2]),
                                      k: Tensor.from_list(:f32, [32, 32], List.duplicate(0.0, 1024)),
                                      v: Tensor.from_list(:f32, [32, 32], List.duplicate(0.0, 1024))})},
       {"model f32", lower!(f32), menv},
       {"model sb4", lower!(q4), menv}]
  end

  test "1, 2, 3 and 4 threads give the same bits as the oracle" do
    workers = Map.new([1, 2, 3, 4], fn n -> {n, elem(Worker.start_link(exec: worker_exec(:host), threads: n), 1)} end)
    assert Worker.info(workers[3]).threads == 3

    for {name, c, env} <- cases() do
      {:ok, ref} = Native.run_oracle(c, env)

      for {n, w} <- workers do
        for isa <- Vapor.Runtime.Substrates.host_isas() do
          {:ok, got} = Native.run(w, c, env, isa: isa, mode: :native)
          assert got.outputs == ref.outputs, "#{name}, #{n} threads (#{isa})"
        end
      end
    end
  end

  test "sessions on a threaded worker: decode steps identical to a single-threaded run" do
    {:ok, c} = Config.from_map(tiny_config("llama"))
    {:ok, p} = Decoder.program(c, tiny_weights(c), max_seq: 16)
    comp = lower!(p)
    ids = &Tensor.from_list(:s32, [length(&1)], &1)

    logits = fn threads ->
      {:ok, w} = Worker.start_link(exec: worker_exec(:host), threads: threads)
      {:ok, s} = Session.open(w, comp, isa: Substrates.host_isa())
      {:ok, %{logits: a}, _} = Session.step(s, %{tok: ids.([1, 2, 3]), pos: ids.([0, 1, 2])}, [:logits])
      for(i <- 3..9, do: elem(Session.step(s, %{tok: ids.([i]), pos: ids.([i])}, [:logits]), 1).logits) ++ [a]
    end

    assert logits.(1) == logits.(2)
    assert logits.(1) == logits.(4)
  end

  test "a large GEMV scales across the pool (reported, not asserted: the machine may be shared)" do
    w = T.const(Tensor.random(:f32, [2048, 2048], 1))
    c = lower!(Program.new(y: T.linear(T.input(:x, :f32, [2048]), w)))
    env = %{x: Tensor.random(:f32, [2048], 2)}
    cores = System.schedulers_online()

    times =
      for n <- Enum.uniq([1, 2, cores]) do
        {:ok, wk} = Worker.start_link(exec: worker_exec(:host), threads: n)
        {:ok, first} = Native.run(wk, c, env, isa: Substrates.host_isa(), mode: :native)
        best = Enum.min(for _ <- 1..5, do: elem(Native.run(wk, c, env, isa: Substrates.host_isa(), mode: :native), 1).elapsed_ns)
        {n, best, first.outputs}
      end

    [{1, t1, o1} | rest] = times
    for {_, _, o} <- rest, do: assert(o == o1)
    IO.puts("\n  GEMV 2048² f32: " <> Enum.map_join(times, ", ", fn {n, t, _} -> "#{n} thr #{Float.round(t / 1.0e6, 2)} ms (×#{Float.round(t1 / t, 2)})" end))
  end
end
