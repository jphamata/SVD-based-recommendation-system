defmodule Vapor.BalanceTest do
  @moduledoc """
  Quantile Balancing (Kimi K3 report §2.3.3, Appendix C), checked in exact
  rational arithmetic against the problem it solves — the balanced
  assignment, whose optimum the rational simplex certifies.

  Measured here, on seeded random batches: the relaxation is integral, as
  the report says (a b-matching); one step of Eq. 14 lowers the worst load;
  but Algorithm 1 *as stated* — thresholds at the (k+1)-th and (q+1)-th
  largest entries themselves — stops at a fixed point whose top-k routing
  is mostly unbalanced, because those thresholds put margins exactly at
  zero: they create the ties the appendix says have measure zero. The same
  alternation with thresholds at the midpoints (an equally exact minimiser,
  by the appendix's own derivation) reaches the certified optimum every
  time. The training recipe is not affected: it applies a bias to the
  *next* batch, where ties are again measure-zero, and its histogram
  estimator interpolates.
  """
  use ExUnit.Case, async: true
  alias Vapor.Train.Balance, as: B

  @moduletag timeout: 600_000

  defp batches do
    :rand.seed(:exsss, {7, 8, 9})
    rat = fn -> {:rand.uniform(9999), 10_000} end
    for {m, n, k, count} <- [{8, 4, 1, 12}, {12, 6, 2, 6}], _ <- 1..count do
      {for(_ <- 1..m, do: for(_ <- 1..n, do: rat.())), k}
    end
  end

  test "the balanced assignment's relaxation is integral, with a checked dual certificate (Eq. 20)" do
    for {s, k} <- batches() do
      {:ok, opt} = B.assignment(s, k)
      assert opt.integral and opt.certified
      routes = for i <- 0..(length(s) - 1), do: for({^i, j} <- opt.x, do: j)
      assert B.loads(routes, length(hd(s))) |> Enum.uniq() == [div(length(s) * k, length(hd(s)))]
      assert B.score(s, routes) == opt.objective
    end
  end

  test "Algorithm 1 as stated: a fixed point, but its own thresholds create ties, and its routing is mostly unbalanced" do
    runs = for {s, k} <- batches(), do: {s, k, B.alternate(s, k, 60)}
    assert Enum.all?(runs, fn {_, _, r} -> is_integer(r.fixed) end)
    unbalanced = Enum.reject(runs, fn {_, _, r} -> r.balanced end)
    assert length(unbalanced) * 2 > length(runs)
    # every unbalanced fixed point has a token whose k-th and (k+1)-th choices tie exactly
    assert Enum.all?(unbalanced, fn {s, k, r} -> B.ties(s, r.bias, k) != [] end)
  end

  test "the same alternation with midpoint thresholds reaches the certified optimum" do
    for {s, k} <- batches() do
      {:ok, opt} = B.assignment(s, k)
      r = B.alternate(s, k, 60, threshold: :midpoint)
      assert r.balanced and r.rounds <= 10
      assert B.score(s, r.routes) == opt.objective
    end
  end

  test "one step of Eq. 14 lowers the worst load" do
    {worst0, worst1} =
      batches()
      |> Enum.map(fn {s, k} ->
        n = length(hd(s))
        zero = List.duplicate({0, 1}, n)
        worst = fn b -> Enum.max(B.loads(B.route(s, b, k), n)) end
        {worst.(zero), worst.(B.step(s, zero, k))}
      end)
      |> Enum.unzip()

    assert Enum.sum(worst1) < Enum.sum(worst0)
  end
end
