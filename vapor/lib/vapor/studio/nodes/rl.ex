defmodule Vapor.Studio.Nodes.RL do
  @moduledoc "Reinforcement-learning nodes: an episode of an environment under a policy, as a video (`Vapor.RL`) — a replay is the seed plus the actions."
  @behaviour Vapor.Studio.Node
  alias Vapor.{Learn, RL}

  @impl true
  def nodes do
    [{__MODULE__, %{type: "rl.episode", version: 1, category: "rl", title: "Episode",
                    doc: "Run CartPole, FrozenLake or the two-link reacher under a policy — the shipped trained one (REINFORCE / behaviour cloning / value iteration), random, or the reacher's expert — and render it. Deterministic: the seed and the actions are the replay.",
                    inputs: [], outputs: [video: :video, return: :number, actions: :json],
                    params: [env: {:enum, ~w(cartpole frozenlake reacher), "cartpole"}, policy: {:enum, ~w(trained random expert), "trained"},
                             seed: {:int, 0, 1_000_000, 0}, max_steps: {:int, 1, 500, 200}, size: {:int, 48, 480, 160}, fps: {:float, 1.0, 60.0, 25.0}]}}]
  end

  @impl true
  def run("rl.episode", _, p, ctx) do
    env = String.to_existing_atom(p.env)
    {actions, ret} = actions(env, p.policy, p.seed, p.max_steps, ctx.worker)
    video = RL.replay(env, p.seed, actions, size: p.size, fps: p.fps)
    {:ok, %{video: video, return: ret, actions: actions}}
  end

  @doc "An episode of the given policy as a video (for `mix vapor.rl replay`)."
  def episode(env, policy, seed, max_steps, w) do
    {actions, _} = actions(env, policy, seed, max_steps, w)
    RL.replay(env, seed, actions, size: 200)
  end

  defp actions(env, policy, seed, max_steps, w) do
    pick = picker(env, policy, w)

    Stream.unfold({RL.reset(env, seed), 0.0}, fn {st, ret} ->
      if st.done or st.t >= max_steps do
        nil
      else
        a = pick.(st)
        {st2, r} = RL.step(env, st, a, seed)
        {{a, ret + r}, {st2, ret + r}}
      end
    end)
    |> Enum.to_list()
    |> then(fn steps -> {Enum.map(steps, &elem(&1, 0)), steps |> List.last({nil, 0.0}) |> elem(1)} end)
  end

  defp picker(:cartpole, "trained", w) do
    {:ok, m} = RL.load(:cartpole)
    fn st -> [[l0, l1]] = Learn.predict(m.net, [RL.observe(:cartpole, st)], worker: w, rows: 16); if l1 > l0, do: 1, else: 0 end
  end

  defp picker(:reacher, "trained", w) do
    {:ok, m} = RL.load(:reacher)
    fn st -> hd(Learn.predict(m.net, [RL.observe(:reacher, st)], worker: w, rows: 16)) end
  end

  defp picker(:reacher, "expert", _), do: &RL.expert/1

  defp picker(:frozenlake, p, _) when p in ["trained", "expert"] do
    {_, pol} = RL.value_iteration()
    pt = List.to_tuple(pol)
    fn st -> elem(pt, st.s) end
  end

  defp picker(env, _random, _) do
    fn st ->
      u = Vapor.Sampler.uniform(4242, st.t)
      case env do
        :cartpole -> if u < 0.5, do: 0, else: 1
        :frozenlake -> min(3, trunc(u * 4))
        :reacher -> [u * 0.2 - 0.1, Vapor.Sampler.uniform(4243, st.t) * 0.2 - 0.1]
      end
    end
  end
end
