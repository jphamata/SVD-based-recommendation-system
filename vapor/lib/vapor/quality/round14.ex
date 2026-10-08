defmodule Vapor.Quality.Round14 do
  @moduledoc """
  Quality checks for the 0.14 round (the open bench: Alembic, the Athanor,
  the Touchstone, the Crucible, the Assay), in the suite's discipline: a
  value, a **control** that a broken, naive or lucky implementation would
  produce, and a threshold that separates them. The bench takes arbitrary
  input, so the controls ask the question that matters for open input:
  *would the same answer have come out if there were nothing to find?*

  | check | value | control (must fail, or be caught) |
  |---|---|---|
  | furnace vs chance | Golomb 7 reaches 25 | random search, same budget: never |
  | exhaustive proof | max-cut 13 over 1024 cuts = brute force by hand | a forged value: refused by the Touchstone |
  | proof vs refutation | R(3,3) = 6: proved over 32 768 graphs | n = 5: the pentagon as counterexample |
  | holdout | planted AR(1) momentum: ρ ≥ 0.5 | random walk: ρ < 0.5 (selection bias exposed) |
  | replay | same seed: same journal root, replay verified | another seed: another root |
  | sandbox | a memory bomb killed, the caller alive | a small program: answered |
  | atoms | 2 000 fresh identifiers: < 50 atoms | — |
  | laws | 6 random Hamiltonians: each H rediscovered and proved | 6 random dissipative systems: no law claimed |
  | quantum | oscillator n + ½, observed order 2 | a box too small: evidence fails |
  | Kepler | Yoshida 4: error bounded over ~250 orbits | RK4 at the same dt: drifts 40× further |
  | molecule | H₂ RHF/STO-3G −1.1167 | stretched H₂ above two atoms (RHF's known failure, shown) |
  | phylogeny | true clades supported | shuffled columns: support < 0.7 |
  | regression | planted law, test R² > 0.9999 | shuffled target: R² < 0.5 |
  | paired test | type-I ≤ 0.15 on 60 null sets | a planted 20-point gap: p < 0.001 |
  | calibration | calibrated model: not flagged | overconfident by 0.2: flagged; Platt lowers ECE |
  | agreement | Krippendorff's α = 0.743 | random labels: α ≈ 0 |
  | scaling | α CI contains 0.34; largest runs predicted < 1 % | shuffled losses: holdout error > 1 % |
  | dedup | near-duplicates clustered, Jaccard exact | — |
  | games | tic-tac-toe: draw over 5 478 positions; MCTS beats random | random vs random ≈ ½ |
  | portable tree | noise() golden values (= the browser's) | a non-numeric call: refused |
  """
  alias Vapor.{Alembic, Assay, Athanor, Crucible}
  alias Vapor.Athanor.{Examples, Game, Touchstone}
  alias Vapor.Alembic.{Builtins, Tree}

  def run(_opts \\ []) do
    %{checks: List.flatten([furnace(), proofs(), holdout(), replay(), sandbox(), laws(), physics(), biology(), assay(), games(), tree()])}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}
  defp f(x) when is_float(x), do: Float.round(x, 6)
  defp f(x), do: x

  # ============================================================ Athanor

  defp furnace do
    {:ok, c} = Athanor.run(Examples.get("golomb").text, seconds: 120)
    [check("furnace vs chance: Golomb ruler with 7 marks (optimum 25)", c.best.value, "random reached it #{c.control.reached_best}× in #{c.control.evaluations}",
           "best = 25; random at the same budget: 0 hits", c.best.value == 25 and c.control.reached_best == 0)]
  end

  defp proofs do
    text = Examples.get("maxcut").text
    {:ok, c} = Athanor.run(text, control: false)
    {:ok, p} = Alembic.load(text, skip: ["space"])
    # an independent brute force, outside the furnace: every 0/1 vector of the same length
    n = length(c.best.data)
    by_hand = Enum.reduce(0..(Integer.pow(2, n) - 1), nil, fn k, acc ->
      x = for i <- 0..(n - 1), do: Bitwise.band(Bitwise.bsr(k, i), 1)
      case Alembic.call(p, "maximize", [x]) do {:ok, v} when is_number(v) -> if acc == nil or v > acc, do: v, else: acc; _ -> acc end
    end)
    ok = match?({:ok, %{verified: true}}, Touchstone.verify(text, Vapor.Main.jsonable(c), full: true))
    forged = Touchstone.verify(text, put_in(Vapor.Main.jsonable(c), ["best", "value"], c.best.value + 1))
    {:ok, r6} = Athanor.run(Examples.get("ramsey").text, control: false)
    {:ok, r5} = Athanor.run(String.replace(Examples.get("ramsey").text, "n = 6", "n = 5"), control: false)
    [check("exhaustive proof: max-cut by enumeration = an independent brute force", c.best.value, "forged #{c.best.value + 1}: #{if match?({:ok, %{verified: false}}, forged), do: "refused", else: "ACCEPTED"}",
           "equal, certificate verified, forgery refused", c.reason == "exhausted" and c.best.value == by_hand and ok and match?({:ok, %{verified: false}}, forged)),
     check("proof vs refutation: R(3,3) — every 2-colouring of K6 has a monochromatic triangle", "#{r6.reason} (#{r6.evaluations} graphs)", "K5: #{r5.reason}, #{length(r5.best.data)} edges",
           "n = 6 proved over 32 768; n = 5 refuted by the pentagon", r6.reason == "exhausted" and r6.verdict =~ "proved" and r5.reason == "counterexample" and length(r5.best.data) == 5)]
  end

  defp ar1(phi) do
    """
    n = 600
    rets = scan(range(n), 0.0, (r, t) => #{phi}*r + 0.01*(noise(t, 7) - 0.5))
    price = scan(rets, 100.0, (p, r) => p * (1 + r))
    ma(t, k) = mean(price[t-k+1:t+1])
    pnl(f, s, lo, hi) = [if ma(t, f) > ma(t, s) then price[t+1]/price[t] - 1 else 0 for t in lo..hi-1]
    sharpe(xs) = if std(xs) == 0 then 0 else mean(xs) / std(xs) * sqrt(252)
    space = ints(2, 2, 60)
    valid(x) = x[0] < x[1]
    maximize(x) = sharpe(pnl(x[0], x[1], 60, 400))
    holdout(x) = sharpe(pnl(x[0], x[1], 400, n - 1))
    budget = 400
    """
  end

  defp holdout do
    {:ok, walk} = Athanor.run(Examples.get("crossover").text, seconds: 200, control: false)
    {:ok, mom} = Athanor.run(ar1(0.5), seconds: 200, control: false)
    rw = walk.holdout.rank_correlation
    rm = mom.holdout.rank_correlation
    [check("holdout: crossover rules, in-sample rank vs out-of-sample rank (Spearman ρ)", "AR(1) φ = 0.5: ρ = #{f(rm)}", "random walk: ρ = #{f(rw)}",
           "planted ≥ 0.5; noise < 0.5", rm >= 0.5 and rw < 0.5)]
  end

  defp replay do
    text = String.replace(Examples.get("tsp").text, "budget = 20000\n", "") <> "\nbudget = 1500"
    {:ok, a} = Athanor.run(text, control: false)
    {:ok, b} = Athanor.run(text, control: false)
    {:ok, c} = Athanor.run(text, seed: 2, control: false)
    replayed = match?({:ok, %{verified: true}}, Touchstone.verify(text, Vapor.Main.jsonable(a), replay: true))
    [check("replay: a run is a function of its seed (journal SHA-256 root)", String.slice(a.journal_root, 0, 12), "seed 2: " <> String.slice(c.journal_root, 0, 12),
           "same seed same root, replay verified; another seed another root", a.journal_root == b.journal_root and a.journal_root != c.journal_root and replayed)]
  end

  # ============================================================ Alembic

  defp sandbox do
    bomb = Vapor.Hermetic.seal(fn -> Alembic.eval("[repeat(1, 1000000) for k in 1..200]", fuel: 10_000_000_000) end, heap_mb: 16)
    small = Vapor.Hermetic.seal(fn -> Alembic.eval("sum([x^2 for x in 1..10])") end, heap_mb: 16)
    names = for i <- 1..2_000, do: "q14_ident_#{i}_#{System.unique_integer([:positive])}"
    before = :erlang.system_info(:atom_count)
    {:ok, _} = Alembic.load(Enum.map_join(names, "\n", &"#{&1} = 1"))
    grew = :erlang.system_info(:atom_count) - before
    [check("sandbox: a memory bomb under a 16 MB heap", inspect(bomb), inspect(small), "bomb killed (:memory); small program answered 385", bomb == {:error, :memory} and small == {:ok, {:ok, 385}}),
     check("atoms: 2 000 fresh identifiers loaded", grew, 2000, "< 50 new atoms (identifiers stay binaries)", grew < 50)]
  end

  # ============================================================ Crucible

  defp rnd(seed, i, lo, hi), do: lo + trunc(Builtins.hash01([:q14, seed, i]) * (hi - lo + 1))

  defp laws do
    planted =
      for s <- 1..6 do
        {a, b, c, d} = {rnd(s, 1, 1, 4), rnd(s, 2, 1, 4), rnd(s, 3, -3, 3), rnd(s, 4, 1, 3)}
        # H = a x² + b y² + c x y + d x² y ; x' = ∂H/∂y, y' = −∂H/∂x
        sys = "x' = #{2 * b}*y + #{c}*x + #{d}*x^2\ny' = -(#{2 * a}*x + #{c}*y + #{2 * d}*x*y)\ndegree = 3"
        {:ok, r} = Crucible.run("laws", sys)
        Enum.any?(r.laws, &(&1.status == "proved" and &1.degree == 3))
      end
    dissipative =
      for s <- 1..6 do
        {a, b, c, k} = {rnd(s, 5, 1, 4), rnd(s, 6, 1, 4), rnd(s, 7, -3, 3), rnd(s, 8, 1, 3)}
        sys = "x' = #{b}*y + #{c}*x*y - #{k}*x\ny' = -#{a}*x - #{k}*y + x^2\ndegree = 3"
        {:ok, r} = Crucible.run("laws", sys)
        length(r.laws)
      end
    [check("laws: random cubic Hamiltonians, H rediscovered exactly over ℚ", "#{Enum.count(planted, & &1)}/6 found and proved", "dissipative systems: #{Enum.sum(dissipative)} laws claimed",
           "6/6 planted; 0 claimed on systems that have none", Enum.all?(planted) and Enum.sum(dissipative) == 0)]
  end

  defp physics do
    {:ok, ho} = Crucible.run("quantum", "V(x) = 0.5*x^2\nx = -9 .. 9\nn = 300\nstates = 4")
    {:ok, box} = Crucible.run("quantum", "V(x) = 0.5*x^2\nx = -1.5 .. 1.5\nn = 300\nstates = 4")
    e_err = ho.states |> Enum.with_index() |> Enum.map(fn {s, k} -> abs(s.energy - (k + 0.5)) end) |> Enum.max()
    orders = Enum.map(ho.states, & &1.observed_order)
    {:ok, kep} = Crucible.run("hamiltonian", "H = (p1^2 + p2^2)/2 - 1/sqrt(q1^2 + q2^2)\nq1(0) = 1; q2(0) = 0; p1(0) = 0; p2(0) = 0.8\nt = 0 .. 1000\ndt = 0.1")
    span = fn rows -> hs = Enum.map(rows, & &1.h); h0 = hd(hs); {abs(List.last(hs) - h0), Enum.max(Enum.map(hs, &abs(&1 - h0)))} end
    {sy_end, sy_max} = span.(kep.trajectory)
    {rk_end, _} = span.(kep.control)
    {:ok, h2} = Crucible.run("molecule", "H 0 0 0\nH 0 0 1.4")
    {:ok, far} = Crucible.run("molecule", "H 0 0 0\nH 0 0 8.0")
    two_atoms = 2 * -0.466582
    [check("quantum: harmonic oscillator by Sturm bisection, E = n + ½", "max |E − (n+½)| = #{f(e_err)}, orders #{inspect(Enum.map(orders, &f/1))}",
           "box ±1.5: #{Enum.count(box.evidence, & &1.ok)}/#{length(box.evidence)} evidence", "< 2·10⁻⁴, order 2 ± 0.2; the small box fails its evidence",
           e_err < 2.0e-4 and Enum.all?(orders, &(abs(&1 - 2) < 0.2)) and not Enum.all?(box.evidence, & &1.ok)),
     check("Kepler, e = 0.36 over ~250 orbits at a coarse dt = 0.1: energy error", "Yoshida 4: #{f(sy_end)} (bounded by #{f(sy_max)})", "RK4 same dt: #{f(rk_end)}",
           "symplectic: bounded, no secular drift; RK4's final drift ≥ 10× the symplectic maximum", kep.method == "yoshida4" and sy_max * 10 < rk_end),
     check("molecule: H₂ at 1.4 bohr, RHF/STO-3G", f(h2.energy), "R = 8 bohr: #{f(far.energy)} vs two atoms #{f(two_atoms)}",
           "−1.1167 ± 10⁻⁴; stretched RHF above two atoms (its known failure, reported not hidden)", abs(h2.energy + 1.1167) < 1.0e-4 and far.energy > two_atoms)]
  end

  defp biology do
    alias Vapor.Science.Biology
    seqs = Biology.evolve(Biology.tree(), 600, 3)
    fasta = Enum.map_join(seqs, "\n", fn {n, s} -> ">#{n}\n" <> Enum.map_join(s, &Enum.at(~w(A C G T), &1)) end)
    {:ok, r} = Crucible.run("phylogeny", fasta <> "\nbootstrap = 50")
    names = Enum.map(seqs, &elem(&1, 0))
    canon = fn set -> if "A" in set, do: Enum.sort(names -- set), else: Enum.sort(set) end
    truth = Biology.splits(Biology.tree()) |> Enum.map(&canon.(MapSet.to_list(&1)))
    found = r.splits |> Enum.filter(&(&1.support >= 0.7)) |> Enum.map(&canon.(&1.taxa))
    rows = for k <- 1..40, do: (x = k / 8; "#{x},#{Float.round(3 * x * x - 2 * x + 1, 6)}")
    {:ok, g} = Crucible.run("regress", "x, y\n" <> Enum.join(rows, "\n") <> "\ntarget = y\nops = + - *\nmax_size = 9\nbudget = 8000")
    [check("phylogeny: NJ + bootstrap on sequences evolved down a known tree", "#{length(found)} supported clades, all true", "shuffled columns: mean support #{f(r.control.mean_support)}",
           "every supported clade is true; control < 0.7", found != [] and Enum.all?(found, &(&1 in truth)) and r.control.mean_support < 0.7),
     check("symbolic regression: y = 3x² − 2x + 1", "test R² #{f(g.best.test_r2)}", "shuffled target: R² #{f(g.control.test_r2)}", "> 0.9999; control < 0.5",
           g.best.test_r2 > 0.9999 and g.control.test_r2 < 0.5)]
  end

  # ============================================================ Assay

  defp csv(header, rows), do: Enum.join([Enum.join(header, ",") | Enum.map(rows, &Enum.join(&1, ","))], "\n")
  defp bern(seed, i, p), do: if(Builtins.hash01([seed, i]) < p, do: 1, else: 0)

  defp assay do
    ps = for s <- 1..60 do
      {:ok, r} = Assay.run("compare", csv(["id", "a", "b"], for(i <- 1..80, do: [i, bern({:a, s}, i, 0.6), bern({:b, s}, i, 0.6)])), reps: 600, seed: s)
      r.p_permutation
    end
    type1 = Enum.count(ps, &(&1 < 0.05)) / 60
    {:ok, pw} = Assay.run("compare", csv(["id", "a", "b"], for(i <- 1..300, do: [i, bern(:a, i, 0.55), bern(:b, i, 0.75)])))

    cal = for i <- 1..800, do: (p = 0.5 + 0.5 * Builtins.hash01([:p, i]); [Float.round(p, 4), bern(:y, i, p)])
    over = for i <- 1..800, do: (p = 0.5 + 0.5 * Builtins.hash01([:p, i]); [Float.round(p, 4), bern(:y, i, p - 0.2)])
    {:ok, c1} = Assay.run("calibration", csv(["p", "correct"], cal))
    {:ok, c2} = Assay.run("calibration", csv(["p", "correct"], over))

    k = [["1", "1", "", "1"], ["2", "2", "3", "2"], ["3", "3", "3", "3"], ["3", "3", "3", "3"], ["2", "2", "2", "2"], ["1", "2", "3", "4"], ["4", "4", "4", "4"],
         ["1", "1", "2", "1"], ["2", "2", "2", "2"], ["", "5", "5", "5"], ["", "", "1", "1"], ["", "", "3", ""]]
    {:ok, a1} = Assay.run("agreement", csv(~w(A B C D), k))
    rand = for i <- 1..200, do: for(j <- 1..4, do: "c#{trunc(Builtins.hash01([:lab, i, j]) * 4)}")
    {:ok, a2} = Assay.run("agreement", csv(~w(A B C D), rand))

    {:ok, sc} = Assay.run("scaling", Assay.example("scaling"))
    [h | rows] = String.split(Assay.example("scaling"), "\n")
    losses = rows |> Enum.map(&List.last(String.split(&1, ","))) |> Enum.sort_by(&Builtins.hash01([:shuffle, &1]))
    shuffled = Enum.zip_with(rows, losses, fn r, l -> (r |> String.split(",") |> Enum.drop(-1) |> Enum.join(",")) <> "," <> l end)
    {:ok, sh} = Assay.run("scaling", Enum.join([h | shuffled], "\n"))
    [lo, hi] = sc.ci95.alpha

    docs = ["the cat sat on the mat and looked out of the window at the rain", "the cat sat on the mat and looked out of the window at the rain!",
            "an entirely different sentence about prime numbers and the gaps between them", "the cat sat on the mat and looked out of the window at the rain."]
    {:ok, dd} = Assay.run("dedup", Enum.join(docs, "\n"))

    [check("paired comparison: permutation p on 60 null data sets (80 items)", "type-I rate #{f(type1)}", "planted 0.55 vs 0.75: p = #{f(pw.p_permutation)}, McNemar #{f(pw.mcnemar.p)}",
           "≤ 0.15 at α = 0.05; planted p < 0.001", type1 <= 0.15 and pw.significant and pw.mcnemar.p < 0.001),
     check("calibration: ECE against its own sampling floor", "calibrated: p = #{f(c1.p_miscalibrated)}", "overconfident: p = #{f(c2.p_miscalibrated)}; Platt #{f(c2.recalibration.ece_before)} → #{f(c2.recalibration.ece_after)}",
           "calibrated not flagged (p > 0.05); overconfident flagged (p < 0.01), Platt helps", c1.p_miscalibrated > 0.05 and c2.p_miscalibrated < 0.01 and c2.recalibration.ece_after < c2.recalibration.ece_before),
     check("agreement: Krippendorff's α on his published example", f(a1.krippendorff_alpha), "random labels: α = #{f(a2.krippendorff_alpha)}", "0.743 ± 0.001; random |α| < 0.1",
           abs(a1.krippendorff_alpha - 0.743) < 0.001 and abs(a2.krippendorff_alpha) < 0.1),
     check("scaling law (Hoffmann approach 3): planted α = 0.34", "CI [#{f(lo)}, #{f(hi)}], largest runs off by #{f(sc.holdout.worst_relative_error)}", "shuffled losses: off by #{f(sh.holdout.worst_relative_error)}",
           "CI contains 0.34, error < 1 %; shuffled > 1 %", lo <= 0.34 and 0.34 <= hi and sc.holdout.worst_relative_error < 0.01 and sh.holdout.worst_relative_error > 0.01),
     check("dedup: MinHash LSH, verified by exact Jaccard", "keep #{inspect(dd.keep)}", "MinHash error #{f(dd.minhash_error)}", "keep [0, 2]; estimate within 0.1", dd.keep == [0, 2] and dd.minhash_error < 0.1)]
  end

  # ============================================================ games and the tree

  defp games do
    {:ok, g} = Game.load(Examples.get("tictactoe").text)
    {:ok, r} = Game.solve(g)
    m = Game.match(g, {:mcts, 64}, :random, 20)
    rr = Game.match(g, :random, :random, 60)
    [check("games: tic-tac-toe solved by negamax over the legal positions", "value #{r.value}, #{r.positions} positions; MCTS vs random #{f(m.score)}", "random vs random #{f(rr.score)}",
           "draw over 5 478; MCTS > 0.75; random ≈ ½ (0.3–0.8, first-mover edge)", r.value == 0 and r.positions == 5478 and m.score > 0.75 and rr.score > 0.3 and rr.score < 0.8)]
  end

  defp tree do
    golden = [Tree.noise([1.0, 2.0]), Tree.noise([-1.25, 7.0]), Tree.noise([123.4567, -0.0015])]
    refused = match?({:error, _}, Tree.parse("len([1])", ["t"]))
    [check("portable tree: noise() bit-identical to the browser's (golden values, test/js/scene_noise.mjs)", Enum.map(golden, &f/1), "len([1]) in a scene: #{if refused, do: "refused", else: "ACCEPTED"}",
           "the three goldens exactly; non-numeric calls refused", golden == [0.16636425908654928, 0.03921110928058624, 0.10643366817384958] and refused)]
  end
end
