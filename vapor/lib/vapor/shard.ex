defmodule Vapor.Shard do
  @moduledoc """
  Tensor parallelism that keeps the bits: a matrix too large for one worker
  is split across several (separate processes — on one machine or on
  several, a worker is a port), and the result is **bit-identical** to the
  single-worker run.

  The constraint comes from first principles. The canonical dot product
  accumulates 16 lanes sequentially and then sums them in a fixed tree;
  every output element is one such dot product. So:

    * **column-parallel** (split `W` by output rows, every shard reads the
      whole `x`) is exact: each output element is computed, whole, on one
      shard, by the same instructions;
    * **row-parallel** (split the contraction `k`, then sum the partial
      results — Megatron's second half, with an all-reduce) is *not*: lane
      `l`'s accumulation restarts at zero on each shard, and the partial
      sums meet in a different order. The bits change (measured by
      `row_parallel_drift/3`, and by the test suite).

  Hence the exact form of a sharded MLP replaces Megatron's
  row-parallel + all-reduce with an **all-gather** of the intermediate
  activation followed by column-parallel again (`mlp/4`): every layer
  stays exact; the price is the communication volume — `b·inter` floats
  gathered instead of `b·d` reduced (×(inter/d), about ×2.7–4 for SwiGLU
  MLPs). Attention is exact when sharded by heads (heads are independent).
  This is the trade a reproducible distributed runtime has to make, stated
  with its cost rather than hidden.
  """
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Native, Substrates}

  @doc """
  `x·Wᵀ` with `W : f32[n, k]` split by rows across `workers` (one shard per
  worker, in parallel). Returns `{:ok, y}` — the bits of the one-worker
  product — or the first error.
  """
  def linear(workers, %Tensor{shape: [b, k]} = x, %Tensor{shape: [_n, k]} = w, opts \\ []) when workers != [] do
    shards = split_rows(w, length(workers))

    results =
      Enum.zip(workers, shards)
      |> Enum.map(fn {wk, ws} -> Task.async(fn -> run_linear(wk, x, ws, opts) end) end)
      |> Enum.map(&Task.await(&1, :infinity))

    case Enum.find(results, &match?({:error, _}, &1)) do
      nil -> {:ok, concat_columns(Enum.map(results, &elem(&1, 1)), b)}
      err -> err
    end
  end

  @doc """
  A SwiGLU MLP `down(silu(x·Gᵀ) ⊙ x·Uᵀ)` sharded across `workers`, exactly:
  `G` and `U` column-parallel, the intermediate all-gathered, `down`
  column-parallel. Bit-identical to the one-worker MLP.
  """
  def mlp(workers, x, {g, u, down}, opts \\ []) do
    [b, _] = x.shape

    with {:ok, gate} <- linear(workers, x, g, opts),
         {:ok, up} <- linear(workers, x, u, opts) do
      # the all-gather: every shard receives the whole intermediate
      inter = elementwise(workers, gate, up, b, opts)
      linear(workers, inter, down, opts)
    end
  end

  @doc """
  The bits a row-parallel (split-`k`, sum of partials) product would give,
  against the exact column-parallel one: `{differing elements, total}`.
  The partial products are each exact canonical dot products; only their
  sum is reassociated — which is enough to move bits.
  """
  def row_parallel_drift(workers, %Tensor{shape: [b, k]} = x, %Tensor{shape: [n, k]} = w, opts \\ []) do
    parts = length(workers)
    kk = div(k, parts)
    true = rem(k, parts) == 0 and rem(kk, 16) == 0

    partials =
      for {wk, i} <- Enum.with_index(workers) do
        xs = cols(x, i * kk, kk)
        wsl = cols(w, i * kk, kk)
        {:ok, y} = run_linear(wk, xs, wsl, opts)
        Tensor.to_list(y)
      end

    summed = partials |> Enum.zip() |> Enum.map(fn t -> t |> Tuple.to_list() |> Enum.reduce(&Vapor.F32.add(&2, &1)) end)
    {:ok, exact} = linear(workers, x, w, opts)
    {Enum.count(Enum.zip(summed, Tensor.to_list(exact)), fn {a, e} -> a != e end), b * n}
  end

  # ------------------------------------------------------------ internals --

  defp run_linear(wk, %Tensor{shape: [b, k]} = x, %Tensor{} = w, opts) do
    p = Program.new(y: T.linear(T.input(:x, :f32, [b, k]), T.const(w)))

    with {:ok, c} <- Lower.lower(p),
         {:ok, r} <- Native.run(wk, c, %{x: x}, isa: Keyword.get(opts, :isa, Substrates.host_isa()), mode: :native) do
      {:ok, r.outputs.y}
    end
  end

  defp elementwise(_workers, gate, up, b, _opts) do
    # silu(gate) ⊙ up is elementwise: any partition is exact; evaluated by
    # the oracle here (it is the canonical microprogram, the same bits)
    [_, n] = gate.shape
    p = Program.new(y: T.mul(T.silu(T.input(:g, :f32, [b, n])), T.input(:u, :f32, [b, n])))
    Vapor.Runtime.Oracle.eval_program(p, %{g: gate, u: up}).y
  end

  defp split_rows(%Tensor{shape: [n, k]} = w, parts) do
    w = Tensor.widen(w)
    sizes = for i <- 0..(parts - 1), do: div(n, parts) + if(i < rem(n, parts), do: 1, else: 0)

    {shards, _} =
      Enum.map_reduce(Enum.reject(sizes, &(&1 == 0)), 0, fn s, at ->
        {Tensor.new(:f32, [s, k], binary_part(w.data, at * k * 4, s * k * 4)), at + s}
      end)

    shards
  end

  defp concat_columns(parts, b) do
    rows =
      for r <- 0..(b - 1), into: <<>> do
        for %Tensor{shape: [_, n]} = t <- parts, into: <<>>, do: binary_part(t.data, r * n * 4, n * 4)
      end

    n = parts |> Enum.map(fn %Tensor{shape: [_, n]} -> n end) |> Enum.sum()
    Tensor.new(:f32, [b, n], rows)
  end

  defp cols(%Tensor{shape: [r, k]} = t, from, len) do
    t = Tensor.widen(t)
    Tensor.new(:f32, [r, len], for(i <- 0..(r - 1), into: <<>>, do: binary_part(t.data, (i * k + from) * 4, len * 4)))
  end
end
