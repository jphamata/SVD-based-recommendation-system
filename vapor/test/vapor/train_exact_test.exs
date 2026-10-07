defmodule Vapor.TrainExactTest do
  @moduledoc """
  `reduce: :exact` (0.15): the step's gradient is the correctly rounded
  mean of the exact sum of the micro-batch gradients (`Vapor.Amalgam`), so
  the bits depend on the set of micro-batches only — not on how many
  workers, which computes what, the arrival order, a crash, nor on the
  count being a power of two.
  """
  use ExUnit.Case, async: false
  alias Vapor.Tensor
  alias Vapor.Train.LM
  alias Vapor.Runtime.Worker
  import Vapor.TestHelpers

  @corpus "A amálgama não lembra a ordem: soma tudo, arredonda uma vez. " <>
            "The amalgam does not remember the order: it sums everything and rounds once. "

  setup_all do
    c = LM.new(d: 32, layers: 1, heads: 2, ff: 64, seq: 16, seqs: 2, vocab: 256)
    corpus = String.duplicate(@corpus, 8)
    {:ok, c: c, corpus: corpus}
  end

  defp workers(n), do: for(_ <- 1..n, do: elem(Worker.start_link(exec: worker_exec(:host)), 1))
  defp start(c, corpus, opts), do: LM.start(c, corpus, Keyword.merge([steps: 20, seed: 3, lr: 1.0e-2, warmup: 2], opts))

  @tag :native
  test "three micro-batches (not a power of two): 1, 2 or 3 workers, any assignment, a crash, the oracle — one digest", %{c: c, corpus: corpus} do
    run = start(c, corpus, micro: 3, reduce: :exact)
    one = LM.train(run, corpus, workers(1), 2)
    two = LM.train(run, corpus, workers(2), 2)
    three = LM.train(run, corpus, workers(3), 2, assign: fn k, i -> 7 * k + 5 * i + 1 end)
    assert LM.digest(one) == LM.digest(two)
    assert LM.digest(one) == LM.digest(three)
    assert one.losses == three.losses

    [w1, w2] = workers(2)
    r = LM.train(run, corpus, [w1, w2], 1)
    Process.unlink(w2)
    Process.exit(w2, :kill)
    r = LM.train(r, corpus, [w1, w2], 1)
    assert LM.digest(r) == LM.digest(one)

    assert LM.digest(LM.train(run, corpus, [], 1)) == LM.digest(LM.train(run, corpus, workers(1), 1))
  end

  @tag :native
  test "the exact mean is another definition than the tree (the test can fail), and both learn", %{c: c, corpus: corpus} do
    ws = workers(2)
    tree = start(c, corpus, micro: 4, chunk: 2) |> LM.train(corpus, ws, 3)
    exact = start(c, corpus, micro: 4, reduce: :exact) |> LM.train(corpus, ws, 3)
    refute LM.digest(tree) == LM.digest(exact)

    # same mathematics, different rounding: the parameters agree closely
    worst =
      for {n, t} <- tree.params, reduce: 0.0 do
        acc -> Enum.zip(Tensor.to_floats(t), Tensor.to_floats(exact.params[n])) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max() |> max(acc)
      end

    assert worst < 1.0e-3

    [{_, last} | _] = exact.losses
    {_, first} = List.last(exact.losses)
    assert last < first
  end

  @tag :native
  test "checkpoint and resume in exact mode continue with the bits of an uninterrupted run", %{c: c, corpus: corpus} do
    ws = workers(2)
    run = start(c, corpus, micro: 5, reduce: :exact)
    straight = LM.train(run, corpus, ws, 3)
    half = LM.train(run, corpus, ws, 1)
    path = Path.join(System.tmp_dir!(), "vapor-exact-#{System.unique_integer([:positive])}.safetensors")
    :ok = LM.checkpoint(half, path)
    resumed = run |> LM.resume(path) |> LM.train(corpus, ws, 2)
    assert LM.digest(resumed) == LM.digest(straight)
  end

  test "options are checked: the tree still needs a power of two, the exact mode does not", %{c: c, corpus: corpus} do
    assert_raise ArgumentError, ~r/power of two/, fn -> LM.start(c, corpus, micro: 3) end
    assert_raise ArgumentError, ~r/reduce/, fn -> LM.start(c, corpus, micro: 2, reduce: :ring) end
    assert_raise ArgumentError, ~r/positive/, fn -> LM.start(c, corpus, micro: 0, reduce: :exact) end
    assert %LM.Run{micro: 7, reduce: :exact} = LM.start(c, corpus, micro: 7, reduce: :exact)
  end
end
