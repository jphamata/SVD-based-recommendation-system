defmodule Vapor.RLTest do
  @moduledoc """
  Reinforcement learning (`Vapor.RL`):

    * the environments are gymnasium's: CartPole trajectories equal to 1e-12
      from the same state and actions; FrozenLake's transition table entry
      for entry;
    * tabular Q-learning finds the optimal FrozenLake policy (value
      iteration's), success 74–75 %; the control (always left) 0 %;
    * the shipped CartPole policy (REINFORCE through `Vapor.Autodiff`)
      balances ≥ 450 of 500 steps on starts it never saw — an untrained one
      falls in about 20; the shipped reacher policy (behaviour cloning)
      reaches ≥ 70 % of held-out goals, random 1 %;
    * training is reproducible: the oracle and the worker reach the same
      weights; a replay (seed + actions) gives the same frames.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Learn, RL}
  import Vapor.TestHelpers

  setup_all do
    w = if Vapor.Runtime.Substrates.binary("vapor-worker", "native"), do: elem(Vapor.Runtime.Worker.start_link(exec: worker_exec(:host)), 1)
    {:ok, worker: w}
  end

  @tag :python
  test "CartPole = gymnasium's dynamics; FrozenLake = gymnasium's transition table" do
    if python?(["gymnasium"]) do
      # gymnasium returns float32 observations; its state (compared here) is binary64
      st = %{s: [0.01, -0.02, 0.03, 0.04], t: 0, done: false}
      actions = for i <- 0..60, do: rem(div(i * 7, 3), 2)

      ours =
        Enum.reduce_while(actions, {st, []}, fn a, {s, acc} ->
          if s.done, do: {:halt, {s, acc}}, else: (fn {s2, _} -> {:cont, {s2, [s2.s ++ [if(s2.done, do: 1.0, else: 0.0)] | acc]}} end).(RL.step(:cartpole, s, a))
        end)
        |> elem(1)
        |> Enum.reverse()

      out = py!("""
      import gymnasium as gym, json, sys
      import numpy as np
      env = gym.make('CartPole-v1').unwrapped
      env.reset(seed=0)
      env.state = np.array([0.01, -0.02, 0.03, 0.04])
      acts = json.loads(sys.argv[1]); rows = []
      for a in acts:
          s, r, term, trunc, _ = env.step(a)
          rows.append([float(v) for v in env.state] + [1.0 if term else 0.0])
          if term: break
      lake = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True).unwrapped
      P = {str(s): {str(a): [[p, n, r, d] for (p, n, r, d) in lake.P[s][a]] for a in range(4)} for s in range(16)}
      print(json.dumps({'cart': rows, 'lake': P}))
      """, [Vapor.JSON.encode(actions)])

      ref = Vapor.JSON.decode!(out)
      assert length(ref["cart"]) == length(ours)
      for {a, b} <- Enum.zip(ours, ref["cart"]), {x, y} <- Enum.zip(a, b), do: assert(abs(x - y) < 1.0e-12)

      table = RL.lake_table()
      for s <- 0..15, a <- 0..3 do
        theirs = ref["lake"]["#{s}"]["#{a}"]
        mine = table[s][a]
        for {{p, n, r, d}, [p2, n2, r2, d2]} <- Enum.zip(mine, theirs) do
          assert {n, r, d} == {n2, r2 * 1.0, d2} and abs(p - p2) < 1.0e-15
        end
      end
    end
  end

  test "Q-learning finds the optimal FrozenLake policy; its success rate is the optimum's; always-left never arrives" do
    {_, vi} = RL.value_iteration()
    {_, ql} = RL.q_learning(20_000)
    assert ql == vi
    rate = RL.lake_success(vi, 1000)
    assert rate > 0.7 and rate < 0.8
    assert RL.lake_success(List.duplicate(0, 16), 1000) == 0.0
  end

  @tag :native
  test "the shipped policies beat their controls on starts and goals never seen", %{worker: w} do
    {:ok, cp} = RL.load(:cartpole)
    assert RL.cartpole_score(cp.net, 20, 50_000, w) >= 450
    assert RL.cartpole_score(Learn.new([4, 32, 32, 2], 12_345), 20, 50_000, w, greedy: false) < 50
    {:ok, re} = RL.load(:reacher)
    assert RL.reacher_success(re.net, 100, 1000, w) >= 0.7
    assert RL.reacher_success(:random, 100, 1000) <= 0.05
    assert RL.reacher_success(:expert, 100, 1000) == 1.0
    for {m, dir} <- [{cp, "cartpole"}, {re, "reacher"}], do: assert(m.config["weights_sha256"] == Learn.digest(m.net), dir)
  end

  @tag :native
  test "training is reproducible (oracle = worker) and a replay re-simulates the same frames", %{worker: w} do
    {a, ia} = RL.reinforce(worker: w, updates: 2, envs: 4, batch: 64, seed: 3)
    {b, ib} = RL.reinforce(updates: 2, envs: 4, batch: 64, seed: 3)
    assert Learn.digest(a) == Learn.digest(b) and ia.returns == ib.returns

    {:ok, %{video: v1, actions: acts}} = Vapor.Studio.Nodes.RL.run("rl.episode", %{}, %{env: "reacher", policy: "expert", seed: 5, max_steps: 60, size: 64, fps: 25.0}, %{worker: w})
    v2 = RL.replay(:reacher, 5, acts, size: 64)
    assert v1.frames == v2.frames and length(v1.frames) == length(acts)
  end
end
