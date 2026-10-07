defmodule Vapor.GamesTest do
  @moduledoc """
  Self-play and environment manipulation (docs/JOGOS.md): the shipped
  self-play policy–value network (PUCT search) (`priv/games/tictactoe.json`, trained only on its
  own games) against a perfect player, with the untrained search as the
  control; training replayable to the bit; and a policy trained on
  randomised cart-poles against one trained on the nominal cart-pole, on
  cart-poles neither saw.
  """
  use ExUnit.Case, async: true
  alias Vapor.Games, as: G

  @moduletag timeout: 900_000

  test "against every optimal line of perfect play: the network buys strength per simulation; at 128 it loses no line" do
    {:ok, n, recipe} = G.load()
    assert recipe["games"] == 400
    # exhaustive, not sampled: the agent is deterministic, so all of the perfect player's optimal choices are followed
    t8 = G.versus_every_optimal_line(n, 8)
    u8 = G.versus_every_optimal_line(G.net(1), 8)
    assert t8.lines > 100 and u8.lines > 100
    # with 8 simulations the trained search loses a small share of lines, the untrained (the control) nearly all
    assert t8.losses / t8.lines < 0.2 and u8.losses / u8.lines > 0.8
    assert G.versus_every_optimal_line(n, 128).losses == 0
    # the sampled measure agrees where it reaches, and misses what the exhaustive one finds (dito em JOGOS.md)
    assert G.versus_perfect(n, 64, 30, 5).losses == 0 and G.versus_every_optimal_line(n, 64).losses > 0
    # and it beats a random player almost always
    assert G.versus_random(n, 32, 20).wins >= 36
  end

  test "the perfect player is perfect: the empty board is a draw, and an X fork is a win" do
    assert G.solve(G.empty()) == 0
    # X at 0 and 8, O at 4 and 2: X to move wins by 6 (two threats)
    b = {1, 0, -1, 0, -1, 0, 0, 0, 1}
    assert G.solve(b) == 1 and 6 in G.optimal(b)
  end

  test "self-play training is a function of its seed: two runs, one digest" do
    a = G.train(games: 20, sims: 8, seed: 3)
    b = G.train(games: 20, sims: 8, seed: 3)
    assert G.digest(a.net) == G.digest(b.net)
    refute G.digest(a.net) == G.digest(G.train(games: 20, sims: 8, seed: 4).net)
  end

  test "domain randomisation: trained on varied cart-poles, the policy holds shifted ones up; trained on one, it often does not" do
    runs = for seed <- 1..3, do: {G.robustness(G.train_cartpole(seed: seed).w), G.robustness(G.train_cartpole(seed: seed, randomize: true).w)}
    nominal = Enum.sum(for {a, _} <- runs, do: a.shifted) / 3
    randomized = Enum.sum(for {_, b} <- runs, do: b.shifted) / 3
    # both balance the nominal pole
    assert Enum.all?(runs, fn {a, b} -> a.nominal == 500 and b.nominal == 500 end)
    assert randomized >= 400 and randomized > nominal + 100, "#{nominal} vs #{randomized}"
  end
end
