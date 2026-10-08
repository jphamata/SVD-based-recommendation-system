defmodule Vapor.Play.SelfPlay do
  @moduledoc """
  Learning a game by playing it against itself (docs/BOARDS.md §6):
  the 0.11 tic-tac-toe learner generalised to **any** game that gives
  `features/1`, `actions/1` and `index/2` beside the four rule functions
  — a policy-and-value network (features → tanh hidden layer → move
  logits and a value), PUCT search guided by it (`Vapor.Play.mcts/3`),
  training only on its own games with the search's visit counts as the
  policy target and the outcome as the value target.

  The judge is not another program but **perfect play**: for games the
  exact solver settles, `versus_optimal/4` follows every optimal line of
  the perfect player from both sides and counts the lines the learner
  loses — exhaustive, not sampled. The control is the same search with
  the untrained network.
  """
  alias Vapor.{Play, Sampler}

  @doc "A network with seeded small random weights for `game` from `state`."
  def net(game, state, hidden \\ 48, seed \\ 1) do
    f = length(game.features(state))
    a = game.actions(state)
    r = fn salt, k, scale -> (Sampler.uniform(seed * 1000 + salt, k) - 0.5) * 2 * scale end
    %{w1: for(h <- 0..(hidden - 1), do: for(i <- 0..(f - 1), do: r.(1, h * f + i, 0.3))), b1: List.duplicate(0.0, hidden),
      wp: for(x <- 0..(a - 1), do: for(h <- 0..(hidden - 1), do: r.(2, x * hidden + h, 0.1))), bp: List.duplicate(0.0, a),
      wv: for(h <- 0..(hidden - 1), do: r.(3, h, 0.1)), bv: 0.0}
  end

  defp dot(a, b), do: Enum.zip_with(a, b, &*/2) |> Enum.sum()

  @doc "Policy over the legal moves and value for the mover."
  def evaluate(game, n, s) do
    x = game.features(s)
    h = Enum.zip_with(n.w1, n.b1, fn w, b -> :math.tanh(dot(w, x) + b) end)
    logits = Enum.zip_with(n.wp, n.bp, fn w, b -> dot(w, h) + b end) |> List.to_tuple()
    moves = game.legal(s)
    ls = Enum.map(moves, &elem(logits, game.index(s, &1)))
    mx = Enum.max(ls)
    ex = Enum.map(ls, &:math.exp(&1 - mx))
    z = Enum.sum(ex)
    {Map.new(Enum.zip(moves, Enum.map(ex, &(&1 / z)))), :math.tanh(dot(n.wv, h) + n.bv), %{x: x, h: h}}
  end

  defp guided(game, n), do: [prior: fn s -> elem(evaluate(game, n, s), 0) end, value: fn s -> elem(evaluate(game, n, s), 1) end]

  @doc "The learner's move: the most visited after `sims` guided simulations."
  def best_move(game, n, s, sims), do: Play.mcts(game, s, [sims: sims] ++ guided(game, n)).best

  @doc """
  Train: `games` self-play games in batches of `batch` (root noise, the
  first `explore` moves sampled by visits), SGD on a replay buffer.
  `%{net, losses, positions}`.
  """
  def train(game, s0, opts \\ []) do
    seed = Keyword.get(opts, :seed, 1)
    games = Keyword.get(opts, :games, 200)
    sims = Keyword.get(opts, :sims, 24)
    batch = Keyword.get(opts, :batch, 10)
    lr = Keyword.get(opts, :lr, 0.03)
    explore = Keyword.get(opts, :explore, 2)
    n0 = net(game, s0, Keyword.get(opts, :hidden, 48), seed)

    {n, buf, losses} =
      Enum.reduce(1..max(div(games, batch), 1), {n0, [], []}, fn round, {n, buf, losses} ->
        fresh = Play.pmap(1..batch, fn g -> self_play(game, n, s0, sims, explore, seed * 100_000 + round * 1000 + g) end) |> List.flatten()
        # every position in all the board's symmetries, when the game has them
        fresh = if Keyword.get(opts, :symmetries, true) and function_exported?(game, :symmetries, 1), do: Enum.flat_map(fresh, fn {s, pi, z} -> for perm <- game.symmetries(s), do: (({s2, pi2} = game.transform(s, pi, perm)); {s2, pi2, z}) end), else: fresh
        buf = Enum.take(fresh ++ buf, Keyword.get(opts, :buffer, 6000))
        {n, l} = Enum.reduce(1..Keyword.get(opts, :epochs, 1), {n, 0.0}, fn e, {n, _} -> sgd(game, n, buf, lr, seed * 7 + round * 31 + e) end)
        {n, buf, [l | losses]}
      end)

    %{net: n, losses: Enum.reverse(losses), positions: length(buf)}
  end

  @doc false
  def self_play(game, n, s0, sims, explore, salt) do
    {hist, z, last_turn_value} = play_out(game, n, s0, sims, explore, salt, 0, [])
    _ = last_turn_value
    # z: value for the player to move at the final state; propagate back with alternating signs
    {out, _} = Enum.map_reduce(hist, -z, fn {s, pi}, v -> {{s, pi, v}, -v} end)
    out
  end

  defp play_out(game, n, s, sims, explore, salt, k, hist) do
    case game.outcome(s) do
      nil ->
        r = Play.mcts(game, s, [sims: sims, seed: salt + k, noise: {salt * 31 + k, 0.25}] ++ guided(game, n))
        total = r.visits |> Map.values() |> Enum.sum() |> max(1)
        pi = Map.new(r.visits, fn {m, c} -> {m, c / total} end)
        m = if k < explore, do: sample(pi, Sampler.uniform(salt + 1, k)), else: r.best
        play_out(game, n, game.play(s, m), sims, explore, salt, k + 1, [{s, pi} | hist])
      z -> {hist, z, nil}
    end
  end

  defp sample(pi, u) do
    pi |> Enum.sort_by(fn {m, _} -> :erlang.phash2(m) end) |> Enum.reduce_while(0.0, fn {m, p}, acc -> if acc + p >= u, do: {:halt, {:pick, m}}, else: {:cont, acc + p} end)
    |> case do {:pick, m} -> m; _ -> pi |> Map.keys() |> hd() end
  end

  defp sgd(game, n, buf, lr, seed) do
    data = Vapor.Modal.Rng.permute(buf, seed)
    hidden = length(n.b1)
    a_n = length(n.bp)

    Enum.reduce(data, {n, 0.0}, fn {s, pi, z}, {n, loss} ->
      {p, v, %{x: x, h: h}} = evaluate(game, n, s)
      idx = Map.new(p, fn {m, pm} -> {game.index(s, m), {pm, Map.get(pi, m, 0.0)}} end)
      dlog = for a <- 0..(a_n - 1), do: (case Map.get(idx, a) do {pm, t} -> pm - t; nil -> 0.0 end)
      dv = 2 * (v - z) * (1 - v * v)
      wpt = Enum.zip_with(n.wp, & &1)
      dh = for {j, col} <- Enum.zip(0..(hidden - 1), wpt), do: (dot(dlog, col) + dv * Enum.at(n.wv, j)) * (1 - Enum.at(h, j) ** 2)
      l = (v - z) ** 2 - Enum.sum(for {_, {pm, t}} <- idx, do: t * :math.log(max(pm, 1.0e-12)))
      n = %{n |
        wp: Enum.zip_with(n.wp, dlog, fn row, g -> if g == 0.0, do: row, else: Enum.zip_with(row, h, fn w, hj -> w - lr * g * hj end) end),
        bp: Enum.zip_with(n.bp, dlog, fn b, g -> b - lr * g end),
        wv: Enum.zip_with(n.wv, h, fn w, hj -> w - lr * dv * hj end),
        bv: n.bv - lr * dv,
        w1: Enum.zip_with(n.w1, dh, fn row, g -> Enum.zip_with(row, x, fn w, xi -> if xi == 0.0, do: w, else: w - lr * g * xi end) end),
        b1: Enum.zip_with(n.b1, dh, fn b, g -> b - lr * g end)}
      {n, loss + l / length(data)}
    end)
  end

  # ---------------------------------------------------------------- judge

  @doc """
  The learner (search with `sims`, network `n`) against **every** optimal
  line of the perfect player, from both sides: `%{lines, wins, draws,
  losses}`. The learner is deterministic, so the tree it faces branches
  only where the perfect player has several optimal moves — all followed.
  """
  def versus_optimal(game, s0, n, sims) do
    Process.put(:sp_solve, %{})
    for side <- [1, -1], reduce: %{lines: 0, wins: 0, draws: 0, losses: 0} do
      acc -> Map.merge(acc, lines(game, s0, n, sims, side), fn _, a, b -> a + b end)
    end
  end

  defp lines(game, s, n, sims, side) do
    case game.outcome(s) do
      nil ->
        if s.turn == side do
          lines(game, game.play(s, best_move(game, n, s, sims)), n, sims, side)
        else
          Enum.reduce(optimal(game, s), %{lines: 0, wins: 0, draws: 0, losses: 0}, fn m, acc -> Map.merge(acc, lines(game, game.play(s, m), n, sims, side), fn _, a, b -> a + b end) end)
        end
      v ->
        # v is for the player to move; the learner's result:
        r = if s.turn == side, do: v, else: -v
        k = cond do r > 0 -> :wins; r < 0 -> :losses; true -> :draws end
        %{lines: 1, wins: 0, draws: 0, losses: 0} |> Map.put(k, 1)
    end
  end

  defp optimal(game, s) do
    memo = Process.get(:sp_solve)
    case Map.fetch(memo, game.key(s)) do
      {:ok, ms} -> ms
      :error ->
        {_, ms} = Play.solve(game, s)
        Process.put(:sp_solve, Map.put(Process.get(:sp_solve), game.key(s), ms))
        ms
    end
  end
end
