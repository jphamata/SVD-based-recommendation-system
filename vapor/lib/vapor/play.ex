defmodule Vapor.Play do
  @moduledoc """
  Games (docs/BOARDS.md): the rules of chess, shogi, Go, the m,n,k
  family (tic-tac-toe, Connect Four, gomoku) and two poker games, each
  checked against the published counts that pin a rule set down, and the
  searches that play them — one generic Monte Carlo tree search for any
  game that implements the four functions below, exact solvers where the
  game is small enough to solve, alpha–beta for chess, counterfactual
  regret minimisation for poker.

  A game module implements:

      legal(state) :: [move]
      play(state, move) :: state
      outcome(state) :: nil | float     # value for the player to move, game over
      key(state) :: term                 # for transpositions

  and `mcts/3` plays it (UCT with random playouts, or PUCT with a
  `prior:` function — the self-play network of 0.11 generalised).
  """

  @doc "Map over a list on every scheduler, keeping the order (deterministic results)."
  def pmap(list, fun), do: list |> Task.async_stream(fun, max_concurrency: System.schedulers_online(), timeout: :infinity, ordered: true) |> Enum.map(fn {:ok, v} -> v end)

  # ================================================================= MCTS

  @doc """
  Monte Carlo tree search from `state` with `sims` simulations:
  `%{best, visits, value}`. Options: `prior: fn state -> %{move => p} end`
  and `value: fn state -> v end` (PUCT, the network's guidance); without
  them, UCT with random playouts seeded by `seed:` (reproducible).
  """
  def mcts(game, state, opts \\ []) do
    sims = Keyword.get(opts, :sims, 400)
    seed = Keyword.get(opts, :seed, 1)
    c = Keyword.get(opts, :c, if(opts[:prior], do: 1.5, else: 1.4))
    ctx = %{game: game, prior: opts[:prior], value: opts[:value], c: c, seed: seed, playout_cap: Keyword.get(opts, :playout_cap, 400), noise: opts[:noise]}

    tree = Enum.reduce(1..sims, %{}, fn i, tree -> elem(simulate(ctx, state, tree, i, 0), 1) end)
    node = tree[game.key(state)]
    visits = node.n
    {best, _} = Enum.max_by(visits, fn {m, n} -> {n, -:erlang.phash2(m)} end)
    total = visits |> Map.values() |> Enum.sum()
    %{best: best, visits: visits, value: Enum.sum(Map.values(node.w)) / max(total, 1), simulations: sims}
  end

  # returns {value for the player to move at `s`, tree}
  defp simulate(ctx, s, tree, i, depth) do
    g = ctx.game
    case g.outcome(s) do
      nil ->
        k = g.key(s)
        case Map.fetch(tree, k) do
          :error ->
            moves = g.legal(s)
            p = if ctx.prior, do: ctx.prior.(s), else: Map.new(moves, &{&1, 1.0 / length(moves)})
            # exploration noise at the root (self-play): a quarter of the prior from normalised exponentials
            p = if depth == 0 and ctx.noise, do: noisy(p, ctx.noise), else: p
            v = if ctx.value, do: ctx.value.(s), else: playout(ctx, s, i, depth)
            {v, Map.put(tree, k, %{p: p, n: Map.new(moves, &{&1, 0}), w: Map.new(moves, &{&1, 0.0})})}

          {:ok, node} ->
            total = node.n |> Map.values() |> Enum.sum()
            m = Enum.max_by(Map.keys(node.n), fn m ->
              n = node.n[m]
              q = if n == 0, do: (if ctx.prior, do: 0.0, else: 1.0e9), else: node.w[m] / n
              u = if ctx.prior, do: ctx.c * Map.get(node.p, m, 0.0) * :math.sqrt(total + 1) / (1 + n), else: ctx.c * :math.sqrt(:math.log(total + 1) / max(n, 1))
              {q + u, -:erlang.phash2(m)}
            end)
            {cv, tree} = simulate(ctx, g.play(s, m), tree, i, depth + 1)
            v = -cv
            node = tree[k]
            {v, Map.put(tree, k, %{node | n: Map.update!(node.n, m, &(&1 + 1)), w: Map.update!(node.w, m, &(&1 + v))})}
        end

      v ->
        {v, tree}
    end
  end

  defp noisy(p, {seed, frac}) do
    raw = p |> Map.keys() |> Enum.sort_by(&:erlang.phash2/1) |> Enum.with_index() |> Map.new(fn {m, k} -> {m, -:math.log(max(Vapor.Sampler.uniform(seed, k), 1.0e-12))} end)
    tot = raw |> Map.values() |> Enum.sum()
    Map.new(p, fn {m, x} -> {m, (1 - frac) * x + frac * raw[m] / tot} end)
  end

  # a random game to the end (bounded), its value for the player to move at s
  defp playout(ctx, s, i, depth) do
    g = ctx.game
    Enum.reduce_while(0..ctx.playout_cap, {s, 1.0}, fn k, {st, sign} ->
      case g.outcome(st) do
        nil ->
          ms = g.legal(st)
          m = Enum.at(ms, trunc(Vapor.Sampler.uniform(ctx.seed * 7919 + i, depth * 1000 + k) * length(ms)) |> min(length(ms) - 1))
          {:cont, {g.play(st, m), -sign}}
        v -> {:halt, {:done, v * sign}}
      end
    end)
    |> case do
      {:done, v} -> v
      _ -> 0.0
    end
  end

  # ================================================================ solving

  @doc """
  Exact game value for the player to move by negamax with alpha–beta and
  a transposition table (for games small enough): `{value, best_moves}`.
  """
  def solve(game, state) do
    Process.put(:play_tt, %{})
    v = negamax(game, state, -2.0, 2.0)
    best = for m <- game.legal(state), -negamax(game, game.play(state, m), -2.0, 2.0) == v, do: m
    {v, best}
  end

  defp negamax(game, s, a, b) do
    case game.outcome(s) do
      nil ->
        k = game.key(s)
        tt = Process.get(:play_tt)
        case Map.get(tt, k) do
          {:exact, v} -> v
          {:lower, v} when v >= b -> v
          {:upper, v} when v <= a -> v
          _ ->
            moves = order_for(game, s)
            a0 = a
            {v, _} = Enum.reduce_while(moves, {-2.0, a}, fn m, {best, a} ->
              x = -negamax(game, game.play(s, m), -b, -a)
              best = max(best, x)
              a = max(a, x)
              if a >= b, do: {:halt, {best, a}}, else: {:cont, {best, a}}
            end)
            # bounds against the window the node was searched with
            flag = cond do v <= a0 -> :upper; v >= b -> :lower; true -> :exact end
            Process.put(:play_tt, Map.put(Process.get(:play_tt), k, {flag, v}))
            v
        end
      v -> v
    end
  end

  defp order_for(game, s), do: if(function_exported?(game, :order, 2), do: game.order(s, game.legal(s)), else: game.legal(s))
end
