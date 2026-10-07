defmodule Vapor.ProvenRuntimeTest do
  @moduledoc "The extracted theorems, exercised where the runtime uses them."
  use ExUnit.Case, async: true
  alias Vapor.Algebra.Scan
  alias Vapor.Verify.BankPad

  test "segmented affine scan: parallel schedule = sequential (associativity, Theorem 9.1)" do
    :rand.seed(:exsss, {1, 2, 3})

    for len <- [1, 2, 3, 7, 16, 33, 100] do
      xs = for _ <- 1..len, do: {{Enum.random(-3..3), Enum.random(-50..50)}, :rand.uniform() < 0.2}
      assert Scan.parallel(xs) == Scan.sequential(xs)
    end
  end

  test "a set flag restarts the recurrence: ragged batches without padding" do
    # two sequences [(2,1),(3,0)] and [(5,5)] packed back to back
    xs = [{{2, 1}, true}, {{3, 0}, false}, {{5, 5}, true}]
    [a, b, c] = Scan.sequential(xs)
    assert Scan.apply(a, 10) == 21
    assert Scan.apply(b, 10) == 3 * (2 * 10 + 1)
    assert Scan.apply(c, 10) == 5 * 10 + 5, "the second sequence ignores the first"
  end

  test "coprime padding makes every warp access conflict-free (Theorem 8.1)" do
    for banks <- [8, 16, 32, 64], stride <- 1..130 do
      s = BankPad.padded_stride(stride, banks)
      assert s - stride in [0, 1]
      assert BankPad.banks_touched(s, banks) |> Enum.uniq() |> length() == banks
    end

    assert BankPad.banks_touched(32, 32) |> Enum.uniq() == [0], "unpadded: 32-way conflict"
  end
end
