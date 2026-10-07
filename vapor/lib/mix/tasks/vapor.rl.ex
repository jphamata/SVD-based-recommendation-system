defmodule Mix.Tasks.Vapor.Rl do
  @shortdoc "Train and measure the reproducible RL policies (CartPole, FrozenLake, reacher)"
  @moduledoc """
      mix vapor.rl train [--out priv/rl]     # REINFORCE on CartPole, behaviour cloning on the reacher
      mix vapor.rl eval                      # the shipped policies against their controls
      mix vapor.rl replay ENV SEED OUT.gif   # an episode of the shipped policy, re-simulated, as a GIF

  Every number is a function of the seed: train twice and the weights are
  the same bits (their SHA-256 is in each `config.json`).
  """
  use Mix.Task
  alias Vapor.RL

  @impl true
  def run(argv) do
    {o, args, _} = OptionParser.parse(argv, strict: [out: :string])
    Mix.Task.run("app.start")
    w = Vapor.Vision.OCR.worker()

    case args do
      ["train"] -> train(o[:out] || "priv/rl", w)
      ["eval"] -> eval(w)
      ["replay", env, seed, out] -> replay(String.to_existing_atom(env), String.to_integer(seed), out, w)
      _ -> Mix.raise("usage: mix vapor.rl train | eval | replay ENV SEED OUT.gif")
    end
  end

  defp train(out, w) do
    t0 = System.monotonic_time(:millisecond)
    {pn, info} = RL.reinforce(worker: w, updates: 100, seed: 1)
    cfg = RL.save(pn, Path.join(out, "cartpole"), %{"env" => "cartpole", "algorithm" => "REINFORCE, mean-normalized returns", "updates" => 100, "envs" => 16,
                                                   "batch" => 1024, "lr" => 5.0e-3, "gamma" => 0.99, "seed" => 1, "returns" => info.returns,
                                                   "seconds" => div(System.monotonic_time(:millisecond) - t0, 1000)})
    Mix.shell().info("cartpole #{cfg["weights_sha256"]}: last returns #{inspect(Enum.take(info.returns, -5))}")

    t1 = System.monotonic_time(:millisecond)
    demos = RL.demonstrations(300, 7)
    {bc, binfo} = RL.clone(demos, worker: w, steps: 6000, seed: 1)
    cfg = RL.save(bc, Path.join(out, "reacher"), %{"env" => "reacher", "algorithm" => "behaviour cloning of the analytic expert", "episodes" => 300,
                                                  "demo_seed" => 7, "steps" => 6000, "seed" => 1, "dataset_sha256" => RL.dataset_digest(demos),
                                                  "rows" => length(List.flatten(demos)), "loss" => elem(List.last(binfo.losses), 1),
                                                  "seconds" => div(System.monotonic_time(:millisecond) - t1, 1000)})
    Mix.shell().info("reacher #{cfg["weights_sha256"]}")
  end

  defp eval(w) do
    {:ok, cp} = RL.load(:cartpole)
    {:ok, re} = RL.load(:reacher)
    random = Vapor.Learn.new([4, 32, 32, 2], 12345)
    Mix.shell().info("CartPole mean return (20 held-out starts): trained #{RL.cartpole_score(cp.net, 20, 50_000, w)}, untrained #{RL.cartpole_score(random, 20, 50_000, w, greedy: false)}")
    {_, vi} = RL.value_iteration()
    {_, ql} = RL.q_learning(20_000)
    Mix.shell().info("FrozenLake success (1000 episodes): value iteration #{RL.lake_success(vi, 1000)}, Q-learning #{RL.lake_success(ql, 1000)} (same policy: #{vi == ql}), always-left #{RL.lake_success(List.duplicate(0, 16), 1000)}")
    Mix.shell().info("Reacher success (100 held-out goals): expert #{RL.reacher_success(:expert, 100, 1000)}, cloned #{RL.reacher_success(re.net, 100, 1000, w)}, random #{RL.reacher_success(:random, 100, 1000)}")
  end

  defp replay(env, seed, out, w) do
    v = Vapor.Studio.Nodes.RL.episode(env, "trained", seed, 500, w)
    File.write!(out, Vapor.Media.GIF.encode(v.frames, fps: v.fps))
    Mix.shell().info("#{out}: #{length(v.frames)} frames")
  end
end
