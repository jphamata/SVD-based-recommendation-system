defmodule Vapor.Algebra.Scan do
  @moduledoc """
  The `scan` generator over the affine SSM monoid, for exact (integer /
  fixed-point) states, and its segmented lifting for ragged batches.

  The combine functions are the Lean-extracted `affine_op/2` and
  `seg_affine/2`; their associativity (`Vapor.affineOp_assoc`,
  `Vapor.segAffine_monoid`) is what makes the tree-shaped `parallel/1`
  equal to the sequential `sequential/1` — asserted in the test suite over
  random ragged batches, i.e. the theorem exercised on the extracted code.

  Elements are `{{a, b}, flag}`: the step `h ↦ a·h + b`; `flag = true`
  starts a new sequence (no padding between sequences).
  """
  alias Vapor.Extracted

  @neutral {{1, 0}, false}
  def neutral, do: @neutral

  @doc "Inclusive prefix scan, left to right."
  def sequential(xs) do
    {out, _} = Enum.map_reduce(xs, @neutral, fn x, acc -> dup(Extracted.seg_affine(acc, x)) end)
    out
  end

  defp dup(y), do: {y, y}

  @doc """
  Inclusive prefix scan by recursive pairing (the Blelloch/Ladner–Fischer
  schedule: O(log n) depth). Scans the pairwise products, then recovers the
  even positions as `prefix ⊗ x[2i]`.
  """
  def parallel(xs) when length(xs) <= 1, do: sequential(xs)

  def parallel(xs) do
    pairs =
      xs
      |> Enum.chunk_every(2)
      |> Enum.map(fn
        [a, b] -> Extracted.seg_affine(a, b)
        [a] -> a
      end)

    sp = pairs |> parallel() |> List.to_tuple()
    xt = List.to_tuple(xs)

    for i <- 0..(length(xs) - 1) do
      cond do
        rem(i, 2) == 1 -> elem(sp, div(i, 2))
        i == 0 -> Extracted.seg_affine(@neutral, elem(xt, 0))
        true -> Extracted.seg_affine(elem(sp, div(i, 2) - 1), elem(xt, i))
      end
    end
  end

  @doc "Apply a scanned element to an initial state."
  def apply({{a, b}, _flag}, h0), do: a * h0 + b
end
