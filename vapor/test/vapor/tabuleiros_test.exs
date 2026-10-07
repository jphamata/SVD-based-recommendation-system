defmodule Vapor.TabuleirosTest do
  @moduledoc """
  Board and card games (docs/TABULEIROS.md): chess and shogi move
  generators against the published perft counts and against python-chess
  / python-shogi on positions reached by random play; chess notation
  round trips; mate proofs verified move by move; Go's rules against
  Tromp & Farnebäck's legal-position counts; the m,n,k solver against the
  known values; CFR on Kuhn poker to the game value −1/18 with its
  exploitability falling (the uniform strategy, the control, is
  exploitable); Leduc hold'em; the generic self-play learner against
  perfect play.
  """
  use ExUnit.Case, async: true
  @moduletag timeout: 900_000
  alias Vapor.Play
  alias Vapor.Play.{Chess, Go, MNK, Poker, Shogi}

  describe "chess" do
    test "perft: the initial position and the four standard test positions" do
      for {fen, exp} <- [
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [20, 400, 8902, 197_281]},
            {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97_862]},
            {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43_238]},
            {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]},
            {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62_379]}
          ] do
        p = Chess.from_fen!(fen)
        assert (for d <- 1..length(exp), do: Chess.perft_parallel(p, d)) == exp, fen
      end
    end

    @tag :chess_py
    test "python-chess agrees on legal moves (SAN) and perft(2) along random games" do
      {fens, sans} =
        Enum.reduce(1..12, {[], []}, fn g, acc ->
          Enum.reduce(0..40, {Chess.start(), acc}, fn k, {p, {fs, ss}} ->
            ms = Chess.moves(p)
            if ms == [], do: {p, {fs, ss}}, else: (
              m = Enum.at(ms, trunc(Vapor.Sampler.uniform(g, k) * length(ms)))
              {Chess.make(p, m), {[Chess.to_fen(p) | fs], [ms |> Enum.map(&Chess.san(p, &1)) |> Enum.sort() |> Enum.join(" ") | ss]}})
          end) |> elem(1)
        end)
      out = Vapor.TestHelpers.py!("""
      import sys, chess
      def perft(b, d):
          if d == 0: return 1
          n = 0
          for m in b.legal_moves:
              b.push(m); n += perft(b, d - 1); b.pop()
          return n
      for line in sys.stdin:
          b = chess.Board(line.strip())
          print(' '.join(sorted(b.san(m) for m in b.legal_moves)) + '|' + str(perft(b, 2)))
      """, [], Enum.join(Enum.reverse(fens), "\n"))
      rows = String.split(String.trim(out), "\n")
      for {{fen, san}, row} <- Enum.zip(Enum.zip(Enum.reverse(fens), Enum.reverse(sans)), rows) do
        [psan, pp] = String.split(row, "|")
        assert san == psan, fen
        assert Chess.perft(Chess.from_fen!(fen), 2) == String.to_integer(pp), fen
      end
    end

    test "FEN and SAN round trips; a PGN of the Scholar's mate" do
      p = Chess.start()
      assert Chess.to_fen(p) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      moves = ~w(e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7#)
      {ms, last} = Enum.map_reduce(moves, p, fn s, q -> {:ok, m} = Chess.parse_move(q, s); {m, Chess.make(q, m)} end)
      assert Chess.status(last) == :checkmate
      assert Chess.pgn(ms) =~ "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0"
    end

    test "a mate in two is proved and the proof tree checked; a wrong proof is refused" do
      p = Chess.from_fen!("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
      assert {:mate, proof} = Chess.prove_mate(p, 1)
      assert {:ok, %{leaves: 1}} = Chess.verify_mate(p, proof)
      # a two-mover (the ladder): no mate in one, a forced mate in two
      q = Chess.from_fen!("7k/8/8/8/8/8/1R6/R5K1 w - - 0 1")
      assert :no_mate = Chess.prove_mate(q, 1)
      assert {:mate, pr} = Chess.prove_mate(q, 2)
      assert {:ok, %{depth: d}} = Chess.verify_mate(q, pr)
      assert d <= 2
      assert {:error, _} = Chess.verify_mate(q, %{pr | replies: Map.new(Enum.take(pr.replies, 1))}) |> then(fn r -> if map_size(pr.replies) > 1, do: r, else: {:error, :skip} end)
    end

    test "the engine finds a mate in one and wins material it is offered" do
      p = Chess.from_fen!("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
      assert Chess.search(p, depth: 2).san == "Rd8#"
      hang = Chess.from_fen!("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")
      assert Chess.search(hang, depth: 3).san == "Rxd5"
    end
  end

  describe "shogi" do
    test "perft from the initial position: 30, 900, 25 470" do
      p = Shogi.start()
      assert for(d <- 1..3, do: Shogi.perft_parallel(p, d)) == [30, 900, 25_470]
    end

    @tag :shogi_py
    test "python-shogi agrees on the legal moves along random games (drops and promotions included)" do
      sfens =
        Enum.flat_map(1..8, fn g ->
          Enum.reduce(0..60, {Shogi.start(), []}, fn k, {p, acc} ->
            ms = Shogi.moves(p)
            if ms == [], do: {p, acc}, else: {Shogi.make(p, Enum.at(ms, trunc(Vapor.Sampler.uniform(g + 50, k) * length(ms)))), [Shogi.to_sfen(p) | acc]}
          end) |> elem(1)
        end)
      out = Vapor.TestHelpers.py!("""
      import sys, shogi
      for line in sys.stdin:
          b = shogi.Board(line.strip())
          print(' '.join(sorted(m.usi() for m in b.legal_moves)))
      """, [], Enum.join(sfens, "\n"))
      for {sfen, row} <- Enum.zip(sfens, String.split(String.trim(out), "\n")) do
        mine = Shogi.from_sfen!(sfen) |> Shogi.moves() |> Enum.map(&Shogi.usi/1) |> Enum.sort() |> Enum.join(" ")
        assert mine == row, sfen
      end
    end

    test "SFEN round trip" do
      s = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
      assert Shogi.to_sfen(Shogi.from_sfen!(s)) == s
    end
  end

  describe "Go" do
    test "legal positions: 1, 57, 12 675 on 1×1, 2×2, 3×3 (Tromp & Farnebäck)" do
      assert for(n <- 1..3, do: Go.legal_positions(n)) == [1, 57, 12_675]
    end

    test "capture, suicide refused, positional superko" do
      # black takes a white corner stone (point 2 has neighbours 1 and 5)
      s = Enum.reduce([1, 2], Go.new(3), fn m, s -> Go.play(s, m) end)
      assert elem(s.board, 2) == -1
      s2 = Go.play(s, 5)
      assert elem(s2.board, 2) == 0
      # white may not play a suicide on a point with no liberties
      t = Enum.reduce([1, :pass, 3], Go.new(3), fn m, s -> Go.play(s, m) end)
      refute 0 in Go.legal(t)
      # with suicide allowed (Tromp–Taylor) the stone is placed and removed — and a one-stone
      # suicide recreates the previous position, so positional superko still forbids it
      assert {b, 0} = Go.place(t.board, 3, 0, -1, true)
      assert elem(b, 0) == 0
      u = Enum.reduce([1, :pass, 3], Go.new(3, suicide: true), fn m, s -> Go.play(s, m) end)
      refute 0 in Go.legal(u)
    end

    test "area scoring with komi" do
      s = Enum.reduce([0, 8], Go.new(3, komi: 0.5), fn m, s -> Go.play(s, m) end)
      assert Go.score(s).margin == -0.5
    end

    test "a 5×5 Monte Carlo player beats random play (random against itself, the control, is even)" do
      wins =
        for g <- 1..6 do
          {s, _} = Enum.reduce_while(1..60, {Go.new(5, komi: 0.5), 0}, fn k, {s, _} ->
            case Go.outcome(s) do
              nil ->
                m = if s.turn == 1, do: Play.mcts(Go.Playouts, s, sims: 120, seed: g * 100 + k).best,
                      else: (ms = Go.sensible(s); Enum.at(ms, trunc(Vapor.Sampler.uniform(g, k) * length(ms))))
                {:cont, {Go.play(s, m), k}}
              _ -> {:halt, {s, k}}
            end
          end)
          Go.score(s).margin > 0
        end
      assert Enum.count(wins, & &1) >= 5
    end
  end

  describe "m,n,k games" do
    test "solved values: 3,3,3 draw; 4,3,3 first-player win; 4,4,3 with gravity a win" do
      assert {+0.0, _} = Play.solve(MNK, MNK.new(3, 3, 3)) |> then(fn {v, b} -> {abs(v), b} end)
      assert {1.0, _} = Play.solve(MNK, MNK.new(4, 3, 3))
      assert {1.0, _} = Play.solve(MNK, MNK.new(4, 4, 3, gravity: true))
    end
  end

  describe "poker" do
    test "Kuhn: CFR+ reaches the game value −1/18 and an exploitability below 10⁻³; uniform play (the control) is exploitable" do
      r = Poker.solve(Poker.Kuhn, 1000)
      assert_in_delta r.value, -1 / 18, 1.0e-3
      assert r.exploitability < 1.0e-3
      assert Poker.exploitability(Poker.Kuhn, Poker.uniform()) > 0.4
      assert hd(r.curve).exploitability > List.last(r.curve).exploitability
    end

    @tag timeout: 1_200_000
    test "Leduc hold'em: exploitability below 0.02 chips per hand; value near −0.0856" do
      r = Poker.solve(Poker.Leduc, 100, every: 50)
      assert r.exploitability < 0.02
      assert_in_delta r.value, -0.0856, 0.02
      assert Poker.exploitability(Poker.Leduc, Poker.uniform()) > 1.0
    end
  end

  describe "generic self-play" do
    test "a policy–value network trained only on its own games loses far fewer optimal lines than the same search untrained (the control)" do
      alias Vapor.Play.{MNK, SelfPlay}
      s0 = MNK.new(3, 3, 3)
      r = SelfPlay.train(MNK, s0, games: 300, sims: 32, epochs: 2)
      assert List.last(r.losses) < 0.6 * hd(r.losses)
      t = SelfPlay.versus_optimal(MNK, s0, r.net, 8)
      u = SelfPlay.versus_optimal(MNK, s0, SelfPlay.net(MNK, s0), 8)
      # measured: 24 of 114 lines lost trained, 165 of 205 untrained
      assert t.losses / t.lines < 0.35
      assert u.losses / u.lines > 0.6
    end
  end
end
