defmodule Vapor.AssayTest do
  use ExUnit.Case, async: true
  alias Vapor.Assay
  alias Vapor.Assay.Stats

  @moduletag timeout: 600_000

  defp csv(header, rows), do: Enum.join([Enum.join(header, ",") | Enum.map(rows, &Enum.join(&1, ","))], "\n")
  defp bern(seed, i, p), do: if(Vapor.Alembic.Builtins.hash01([seed, i]) < p, do: 1, else: 0)

  describe "statistics, against known values" do
    test "normal quantile, Holm and Benjamini–Hochberg" do
      assert_in_delta Stats.qnorm(0.975), 1.959963984540054, 1.0e-12
      assert_in_delta Stats.qnorm(0.001), -3.090232306167813, 1.0e-10
      # R: p.adjust(c(0.01, 0.04, 0.03, 0.005), "holm") and "BH"
      assert Enum.map(Stats.holm([0.01, 0.04, 0.03, 0.005]), &Float.round(&1, 6)) == [0.03, 0.06, 0.06, 0.02]
      assert Enum.map(Stats.bh([0.01, 0.04, 0.03, 0.005]), &Float.round(&1, 6)) == [0.02, 0.04, 0.04, 0.02]
    end

    test "McNemar's exact test (binomial, two-sided)" do
      # 2 vs 10 discordant: 2 * P(X ≤ 2 | n = 12, ½) = 2 * 79/4096
      assert_in_delta Stats.binom_two_sided(2, 12), 2 * 79 / 4096, 1.0e-12
    end

    test "Fleiss' κ and Krippendorff's α on published examples" do
      # Fleiss (1971) as in the Wikipedia worked example: κ = 0.210
      counts = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
      rows = Enum.map(counts, fn cs -> cs |> Enum.with_index() |> Enum.flat_map(fn {n, c} -> List.duplicate("c#{c}", n) end) end)
      text = csv(for(k <- 1..14, do: "r#{k}"), rows)
      {:ok, r} = Assay.run("agreement", text)
      assert_in_delta r.fleiss_kappa, 0.210, 0.001
      # Krippendorff (2011), nominal example with missing values: α = 0.743
      k = [["1", "1", nil, "1"], ["2", "2", "3", "2"], ["3", "3", "3", "3"], ["3", "3", "3", "3"], ["2", "2", "2", "2"], ["1", "2", "3", "4"], ["4", "4", "4", "4"],
           ["1", "1", "2", "1"], ["2", "2", "2", "2"], [nil, "5", "5", "5"], [nil, nil, "1", "1"], [nil, nil, "3", nil]]
      text2 = csv(["A", "B", "C", "D"], Enum.map(k, fn r -> Enum.map(r, &(&1 || "")) end))
      {:ok, r2} = Assay.run("agreement", text2)
      assert_in_delta r2.krippendorff_alpha, 0.743, 0.001
    end
  end

  describe "signal or noise, calibrated" do
    test "the paired comparison holds its level on null data and finds a planted effect" do
      ps =
        for s <- 1..60 do
          rows = for i <- 1..80, do: [i, bern({:a, s}, i, 0.6), bern({:b, s}, i, 0.6)]
          {:ok, r} = Assay.run("compare", csv(["id", "a", "b"], rows), reps: 600, seed: s)
          r.p_permutation
        end
      rate = Enum.count(ps, &(&1 < 0.05)) / length(ps)
      assert rate <= 0.15, "type-I error #{rate} on null data"
      rows = for i <- 1..300, do: [i, bern(:a, i, 0.55), bern(:b, i, 0.75)]
      {:ok, r} = Assay.run("compare", csv(["id", "a", "b"], rows))
      assert r.significant and r.mcnemar.p < 0.001
    end

    test "the leaderboard separates a clear leader and refuses to separate ties" do
      rows = for i <- 1..400, do: [i, bern(:x, i, 0.70), bern(:y, i, 0.70), bern(:z, i, 0.50)]
      {:ok, r} = Assay.run("leaderboard", csv(["item", "x", "y", "z"], rows))
      leader = hd(r.systems).system
      other = if leader == "x", do: "y", else: "x"
      assert other in r.tied_with_leader
      assert "z" not in r.tied_with_leader
    end

    test "calibration: a calibrated model is not flagged; an overconfident one is" do
      cal = for i <- 1..800, do: (p = 0.5 + 0.5 * Vapor.Alembic.Builtins.hash01([:p, i]); [Float.round(p, 4), bern(:y, i, p)])
      {:ok, r} = Assay.run("calibration", csv(["p", "correct"], cal))
      assert r.p_miscalibrated > 0.05
      over = for i <- 1..800, do: (p = 0.5 + 0.5 * Vapor.Alembic.Builtins.hash01([:p, i]); [Float.round(p, 4), bern(:y, i, p - 0.2)])
      {:ok, r2} = Assay.run("calibration", csv(["p", "correct"], over))
      assert r2.p_miscalibrated < 0.01
      assert r2.recalibration.ece_after < r2.recalibration.ece_before
    end

    test "scaling: the planted exponents are recovered with intervals that contain them, and the largest runs are predicted" do
      {:ok, r} = Assay.run("scaling", Assay.example("scaling"))
      [lo, hi] = r.ci95.alpha
      assert lo <= 0.34 and 0.34 <= hi
      assert r.holdout.worst_relative_error < 0.01
    end

    test "scaling on structureless losses: no overflow, and the holdout says the law predicts nothing" do
      [h | rows] = String.split(Assay.example("scaling"), "\n")
      losses = rows |> Enum.map(&List.last(String.split(&1, ","))) |> Enum.sort_by(&Vapor.Alembic.Builtins.hash01([:shuffle, &1]))
      shuffled = Enum.zip_with(rows, losses, fn r, l -> (r |> String.split(",") |> Enum.drop(-1) |> Enum.join(",")) <> "," <> l end)
      {:ok, r} = Assay.run("scaling", Enum.join([h | shuffled], "\n"))
      assert r.holdout.worst_relative_error > 0.01
      refute Enum.find(r.evidence, &(&1.check == "predicts its largest runs")).ok
    end

    test "contamination finds the planted overlap and reports the clean subset" do
      {:ok, r} = Assay.run("contamination", Assay.example("contamination"))
      assert r.contaminated == 1 and r.clean_indices == [1, 2]
    end

    test "dedup clusters near-duplicates verified by exact Jaccard" do
      docs = ["the cat sat on the mat and looked out of the window at the rain", "the cat sat on the mat and looked out of the window at the rain!",
              "an entirely different sentence about prime numbers and the gaps between them", "the cat sat on the mat and looked out of the window at the rain."]
      {:ok, r} = Assay.run("dedup", Enum.join(docs, "\n"))
      assert r.keep == [0, 2] and r.removed == 2
    end

    test "an LLM judge that prefers the first slot is caught" do
      rows = for i <- 1..200, do: if(Vapor.Alembic.Builtins.hash01([:j, i]) < 0.7, do: ["A", "B"], else: ["A", "A"])
      {:ok, r} = Assay.run("judge", csv(["ab", "ba"], rows))
      assert r.p_position_bias < 0.001
    end
  end

  test "every example runs" do
    for %{tool: t, example: ex} <- Assay.tools(), do: assert({:ok, _} = Assay.run(t, ex), t)
  end
end
