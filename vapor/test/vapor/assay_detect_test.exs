defmodule Vapor.AssayDetectTest do
  @moduledoc """
  The Assay's `detect` tool: SAM 3's cgF1, with the matching exact and
  certified (the rational simplex), the paper's two open choices made and
  reported, a greedy matching shown to lose what the optimal one keeps, and
  a control — predictions shuffled across images — that must fall to chance.
  """
  use ExUnit.Case, async: true
  alias Vapor.Assay
  alias Vapor.Assay.Detect
  alias Vapor.Logic.LP

  @moduletag timeout: 300_000

  defp r(list), do: Enum.map(list, &LP.rat/1)

  test "IoU is exact" do
    assert Detect.iou(r([0, 0, 10, 10]), r([5, 0, 15, 10])) == {1, 3}
    assert Detect.iou(r([0, 0, 10, 10]), r([10, 0, 20, 10])) == {0, 1}
    assert Detect.iou(r([0, 0, 10, 10]), r([0, 0, 10, 10])) == {1, 1}
  end

  test "the matching maximises total IoU (against every partial matching), with a checked certificate" do
    :rand.seed(:exsss, {3, 1, 4})
    box = fn -> x = :rand.uniform(20); y = :rand.uniform(20); [x, y, x + 3 + :rand.uniform(8), y + 3 + :rand.uniform(8)] end

    for _ <- 1..25 do
      preds = for _ <- 1..3, do: r(box.())
      truths = for _ <- 1..3, do: r(box.())
      {:ok, m, true} = Detect.match(preds, truths)
      total = Enum.reduce(m, {0, 1}, fn {_, _, v}, acc -> LP.qadd(acc, v) end)
      # every injective partial map from predictions to truths (or to nothing)
      best =
        for a <- [nil, 0, 1, 2], b <- [nil, 0, 1, 2], c <- [nil, 0, 1, 2], js = Enum.reject([a, b, c], &is_nil/1), js == Enum.uniq(js) do
          [a, b, c] |> Enum.with_index() |> Enum.reject(fn {j, _} -> is_nil(j) end)
          |> Enum.reduce({0, 1}, fn {j, i}, acc -> LP.qadd(acc, Detect.iou(Enum.at(preds, i), Enum.at(truths, j))) end)
        end
        |> Enum.max_by(&LP.to_float/1)

      assert total == best
    end
  end

  test "a greedy matching (best IoU first) loses a true positive the optimal one keeps" do
    # P1 overlaps T1 by 0.74 and T2 by 0.6; P2 overlaps T1 by 0.54 and T2 by 0.18
    {t1, t2, p1, p2} = {[0, 0, 10, 10], [4, 0, 14, 10], [1.5, 0, 11.5, 10], [-3, 0, 7, 10]}
    {preds, truths} = {Enum.map([p1, p2], &r/1), Enum.map([t1, t2], &r/1)}
    {:ok, m, true} = Detect.match(preds, truths)
    assert m |> Enum.map(fn {i, j, _} -> {i, j} end) |> Enum.sort() == [{0, 1}, {1, 0}]

    # greedy: the best pair first (P1–T1), then what is left (P2–T2, below 0.5)
    pairs = for i <- 0..1, j <- 0..1, do: {i, j, Detect.iou(Enum.at(preds, i), Enum.at(truths, j))}
    {greedy, _, _} =
      pairs |> Enum.sort_by(fn {_, _, v} -> -LP.to_float(v) end)
      |> Enum.reduce({[], [], []}, fn {i, j, v}, {acc, ui, uj} -> if i in ui or j in uj, do: {acc, ui, uj}, else: {[v | acc], [i | ui], [j | uj]} end)

    assert Enum.count(greedy, &(LP.qcmp(&1, {1, 2}) >= 0)) == 1
    {:ok, out} = Detect.run(Vapor.JSON.encode([%{"truth" => [t1, t2], "pred" => [p1 ++ [0.9], p2 ++ [0.9]]}]), reps: 20)
    # optimal: both matches above 0.5, so F1 at τ = 0.5 is 1 (greedy would give 0.5)
    assert hd(out.f1_by_tau) == 1.0
  end

  test "the example detector is better than chance; shuffled across images, it is not" do
    {:ok, out} = Assay.run("detect", Assay.example("detect"))
    assert out.cg_f1 > 30 and hd(out.cg_f1_ci95) > 0
    assert Enum.all?(out.evidence, & &1.ok)
    assert out.choices |> Enum.join(" ") =~ "false negative"

    # control: every image gets another image's predictions
    lines = Assay.example("detect") |> String.split("\n") |> Enum.map(&Vapor.JSON.decode!/1)
    preds = Enum.map(lines, & &1["pred"])
    shuffled = Enum.zip(lines, tl(preds) ++ [hd(preds)]) |> Enum.map(fn {l, p} -> Vapor.JSON.encode(%{l | "pred" => p}) end) |> Enum.join("\n")
    {:ok, ctrl} = Assay.run("detect", shuffled)
    assert ctrl.cg_f1 < out.cg_f1 / 3
    refute Enum.all?(ctrl.evidence, & &1.ok)
  end

  test "confidence at 0.5 does not count; images without the object decide presence" do
    # one positive found, one negative with a false alarm at 0.5 (dropped) and one at 0.51 (kept)
    json = Vapor.JSON.encode([
      %{"truth" => [[0, 0, 10, 10]], "pred" => [[0, 0, 10, 10, 0.9]]},
      %{"truth" => [], "pred" => [[0, 0, 5, 5, 0.5]]},
      %{"truth" => [], "pred" => [[0, 0, 5, 5, 0.51]]}
    ])
    {:ok, out} = Detect.run(json, reps: 20)
    assert out.pm_f1 == 1.0
    assert out.presence == %{tp: 1, fn: 0, fp: 1, tn: 1}
    assert_in_delta out.il_mcc, 0.5, 1.0e-12
    # nothing predicted anywhere and no object anywhere: MCC's denominator is 0, read as 0 (the stated choice)
    {:ok, none} = Detect.run(Vapor.JSON.encode([%{"truth" => [], "pred" => []}]), reps: 10)
    assert none.il_mcc == 0.0 and none.cg_f1 == nil
  end
end
