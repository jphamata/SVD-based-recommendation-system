defmodule Vapor.ShardTest do
  @moduledoc """
  Tensor parallelism across worker processes (`Vapor.Shard`): column-parallel
  products and the all-gather MLP are bit-identical to one worker; the
  row-parallel (split-k) form is shown to move bits — the reason the exact
  form exists.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Shard, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Native, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :native

  setup_all do
    workers = for _ <- 1..3, do: elem(Worker.start_link(exec: worker_exec(:host)), 1)
    {:ok, workers: workers}
  end

  defp one(wk, x, w) do
    {:ok, c} = Lower.lower(Program.new(y: T.linear(T.input(:x, :f32, x.shape), T.const(w))))
    {:ok, r} = Native.run(wk, c, %{x: x}, isa: Substrates.host_isa(), mode: :native)
    r.outputs.y
  end

  test "column-parallel over 2 and 3 workers = one worker, bit for bit (uneven splits too)", %{workers: ws} do
    x = Tensor.random(:f32, [5, 256], 1)
    w = Tensor.random(:f32, [100, 256], 2, scale: 0.1)
    ref = one(hd(ws), x, w)
    for n <- [2, 3], do: assert(Shard.linear(Enum.take(ws, n), x, w) == {:ok, ref})
  end

  test "the exact sharded MLP (all-gather, then column-parallel) = the one-worker MLP", %{workers: ws} do
    x = Tensor.random(:f32, [3, 64], 3)
    {g, u, d} = {Tensor.random(:f32, [176, 64], 4, scale: 0.2), Tensor.random(:f32, [176, 64], 5, scale: 0.2), Tensor.random(:f32, [64, 176], 6, scale: 0.2)}
    p = Program.new(y: T.linear(T.mul(T.silu(T.linear(T.input(:x, :f32, [3, 64]), T.const(g))), T.linear(T.input(:x, :f32, [3, 64]), T.const(u))), T.const(d)))
    {:ok, c} = Lower.lower(p)
    {:ok, r} = Native.run(hd(ws), c, %{x: x}, isa: Substrates.host_isa(), mode: :native)
    assert Shard.mlp(ws, x, {g, u, d}) == {:ok, r.outputs.y}
  end

  test "row-parallel (split-k + sum of partials) moves bits — the measured reason for the all-gather", %{workers: ws} do
    x = Tensor.random(:f32, [8, 512], 7)
    w = Tensor.random(:f32, [64, 512], 8)
    {differ, total} = Shard.row_parallel_drift(Enum.take(ws, 2), x, w)
    assert total == 512 and differ > total / 4
  end
end
