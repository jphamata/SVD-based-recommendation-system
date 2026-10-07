defmodule Vapor.AthanorTest do
  use ExUnit.Case, async: true
  alias Vapor.Alembic
  alias Vapor.Athanor
  alias Vapor.Athanor.{Examples, Game, Session, Space, Spec, Touchstone}

  @moduletag timeout: 300_000

  describe "spaces" do
    test "finite spaces enumerate exactly their size, and every sample belongs" do
      rng = :rand.seed_s(:exsss, {1, 2, 3})
      for {kind, args} <- [{"bits", [5]}, {"ints", [2, -1, 3]}, {"perm", [4]}, {"subset", [Enum.to_list(1..7), 3]}, {"subsets", [Enum.to_list(1..5)]},
                           {"seq", [3, ["a", "b", "c"]]}, {"graph", [4]}, {"partition", [5, 3]}] do
        {:ok, s} = Space.build(kind, args)
        all = Enum.to_list(Space.enumerate(s))
        assert length(all) == s.size, "#{kind}: enumerated #{length(all)}, size #{s.size}"
        assert length(Enum.uniq_by(all, &Space.canon(s, &1))) == s.size
        Enum.reduce(1..50, rng, fn _, r ->
          {x, r} = Space.sample(s, r)
          assert {:ok, _} = Space.check(s, x)
          {y, r} = Space.mutate(s, x, r)
          assert {:ok, _} = Space.check(s, y), "#{kind}: a mutation left the space: #{inspect(y)}"
          {z, r} = Space.cross(s, x, y, r)
          assert {:ok, _} = Space.check(s, z), "#{kind}: a child left the space: #{inspect(z)}"
          r
        end)
      end
    end

    test "program spaces count, realise and parse their trees" do
      {:ok, s} = Space.build("program", [["x"], ["+", "*", "sin"], [1], 3])
      assert s.size == Enum.count(Space.enumerate(s))
      {:ok, t} = Space.parse_program(s, "x * x + 1")
      {:fn, _, 1, f} = Space.realize(s, t)
      assert f.([3.0]) == 10.0
      assert {:error, _} = Space.parse_program(s, "exp(x)")
    end

    test "an unknown space or a bad argument is refused with the list of spaces" do
      assert {:error, m} = Spec.parse("space = cube(3)\nminimize(x) = 0")
      assert m =~ "cube"
      assert {:error, m2} = Spec.parse("minimize(x) = 0")
      assert m2 =~ "no space"
    end
  end

  describe "searches and their certificates" do
    test "a small space is enumerated: the optimum is proved, and the Touchstone re-checks it" do
      text = Examples.get("maxcut").text
      {:ok, c} = Athanor.run(text)
      assert c.reason == "exhausted" and c.best.value == 13
      assert c.verdict =~ "optimal"
      assert {:ok, %{verified: true}} = Touchstone.verify(text, c, full: true)
    end

    test "a claim is refuted with a counterexample, or proved over the whole space" do
      {:ok, c} = Athanor.run(Examples.get("euler").text)
      assert c.reason == "counterexample" and c.best.candidate == "[40]"
      ramsey5 = String.replace(Examples.get("ramsey").text, "n = 6", "n = 5")
      {:ok, c5} = Athanor.run(ramsey5)
      assert c5.reason == "counterexample"
      # the counterexample on 5 vertices is the pentagon (or its complement): 5 edges
      assert length(c5.best.data) == 5
      {:ok, c4} = Athanor.run("space = graph(4)\nclaim(g) = len(g) <= 6")
      assert c4.reason == "exhausted" and c4.verdict =~ "proved"
    end

    test "a violation measure lets the search climb to validity (Schur colouring of 1..13)" do
      {:ok, c} = Athanor.run(Examples.get("schur").text, seconds: 120)
      assert c.reason == "found"
      {:ok, p} = Alembic.load(Examples.get("schur").text, skip: ["space"])
      assert Alembic.call(p, "violation", [c.best.data]) == {:ok, 0}
    end

    test "the search beats its random control on a structured problem, and the certificate says so" do
      {:ok, c} = Athanor.run(Examples.get("golomb").text, seconds: 120)
      assert c.reason == "target" and c.best.value == 25
      assert c.control.reached_best == 0
    end

    test "a run is a function of its seed: same seed, same journal; the replay reproduces it" do
      text = Examples.get("tsp").text <> "\nbudget = 1500"
      text = String.replace(text, "budget = 20000\n", "")
      {:ok, a} = Athanor.run(text, control: false)
      {:ok, b} = Athanor.run(text, control: false)
      {:ok, c} = Athanor.run(text, seed: 2, control: false)
      assert a.journal_root == b.journal_root and a.journal_root != c.journal_root
      assert {:ok, %{verified: true}} = Touchstone.verify(text, Vapor.Main.jsonable(a), replay: true)
    end

    test "a forged certificate fails the Touchstone" do
      text = Examples.get("maxcut").text
      {:ok, c} = Athanor.run(text, control: false)
      forged = put_in(Vapor.Main.jsonable(c), ["best", "value"], 14)
      assert {:ok, %{verified: false, checks: checks}} = Touchstone.verify(text, forged)
      assert Enum.any?(checks, &(&1.check == "value" and not &1.ok))
      other = put_in(Vapor.Main.jsonable(c), ["best", "candidate"], "[2, 0, 0, 1, 0, 0, 1, 1, 0, 0]")
      assert {:ok, %{verified: false}} = Touchstone.verify(text, other)
    end

    test "outside proposals are checked, recorded and evaluated like any other" do
      {:ok, spec} = Spec.parse("space = ints(2, 0, 50)\nminimize(v) = (v[0] - 37)^2 + (v[1] - 11)^2\nbudget = 200")
      r = Athanor.init(spec)
      r = Athanor.propose(r, [[37, 11], [99, 1], "nope"], :human)
      assert length(r.queue) == 1 and length(r.error_samples) == 2
      r = Athanor.step(r, :all)
      c = Athanor.certificate(r, control: false)
      assert c.best.value == 0 and c.best.by == "human"
      assert [%{source: :human, candidate: "[37, 11]"}] = c.outside_proposals
    end

    test "the holdout exposes selection bias in trading-rule mining on a random walk" do
      {:ok, c} = Athanor.run(Examples.get("crossover").text, seconds: 200)
      assert c.holdout.trials >= 300
      assert c.holdout.rank_correlation < 0.5
    end

    test "a measured objective takes measurements from outside" do
      truth = fn [a, b] -> (a - 0.3) ** 2 + (b - 0.7) ** 2 end
      {:ok, c} = Athanor.run("space = reals(2, 0, 1)\nmeasured = true\nbudget = 30", measure: fn x -> {:ok, truth.(x)} end)
      assert c.evaluations == 30
      assert c.best.value < 0.05
      assert Map.has_key?(c.strategies, "bayes")
    end

    test "every example parses and runs a short budget without errors" do
      for e <- Examples.all(), e.id != "lab", not Game.game?(e.text) do
        assert {:ok, c} = Athanor.run(e.text, budget: 60, control: false, seconds: 60), e.id
        assert c.errors == 0, "#{e.id}: #{inspect(c.error_samples)}"
      end
    end
  end

  describe "sessions: a person steering" do
    test "start, watch, propose, ban, certify" do
      {:ok, id} = Session.start("space = ints(3, 0, 20)\nminimize(v) = abs(v[0] - 3) + abs(v[1] - 14) + abs(v[2] - 9)\nbudget = 400")
      {:ok, _} = Session.call(id, {:propose, [[3, 14, 9]]})
      wait = fn wait, n -> {:ok, s} = Session.call(id, :snapshot); if s.status == "running" and n > 0, do: (Process.sleep(50); wait.(wait, n - 1)), else: s end
      s = wait.(wait, 200)
      assert s.best.value == 0
      :ok = Session.call(id, {:ban, "[3, 14, 9]"})
      {:ok, c} = Session.call(id, :certificate)
      assert Enum.any?(c.outside_proposals, &(&1.candidate == "[3, 14, 9]"))
      Session.close(id)
      assert {:error, _} = Session.call(id, :snapshot)
    end
  end

  describe "games" do
    setup do
      {:ok, ttt} = Game.load(Examples.get("tictactoe").text)
      %{ttt: ttt}
    end

    test "tic-tac-toe solved exactly: a draw over the 5478 legal positions", %{ttt: g} do
      {:ok, r} = Game.solve(g)
      assert r.value == 0 and r.positions == 5478
    end

    test "search finds the immediate win, and plays well against random", %{ttt: g} do
      # X to move with two in a row on the top line
      s = [1, 1, 0, 2, 2, 0, 0, 0, 0]
      assert Game.mcts(g, s, sims: 300).move == 2
      m = Game.match(g, {:mcts, 64}, :random, 20)
      assert m.score > 0.75
    end

    test "a subtraction game: the losing positions are those ≡ 0, 2 (mod 7)" do
      {:ok, g} = Game.load("init = (30, 1)\nplayer(s) = s[1]\nmoves(s) = [k for k in [1, 3, 4] if k <= s[0]]\nplay(s, m) = (s[0] - m, 3 - s[1])\nwinner(s) = if s[0] == 0 then 3 - s[1] else nil")
      for n <- 1..20 do
        {:ok, r} = Game.solve(g, state: {n, 1})
        assert (r.value == -1) == (rem(n, 7) in [0, 2]), "n = #{n}"
      end
    end

    test "a broken game says what is missing" do
      assert {:error, m} = Game.load("init = 0\nmoves(s) = [1]")
      assert m =~ "player" and m =~ "winner"
    end
  end
end
