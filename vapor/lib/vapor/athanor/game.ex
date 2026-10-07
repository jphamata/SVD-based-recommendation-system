defmodule Vapor.Athanor.Game do
  @moduledoc """
  Any two-player game written in Alembic, solved, searched, learned and
  played (docs/ATHANOR.md §6):

      init = [0, 0, 0, 0, 0, 0, 0, 0, 0]
      player(s) = if count(s, 1) == count(s, 2) then 1 else 2
      moves(s) = [i for i in 0..8 if s[i] == 0]
      play(s, m) = set_at(s, m, player(s))
      winner(s) = …        # nil while playing, 0 for a draw, else the winner
      show(s) = …          # optional: the board as text
      features(s) = […]    # optional: numbers describing s, for a learned value

  `solve/2` settles the game exactly by negamax with a transposition table
  (the value, the distance to the end, the best moves) — a proof by
  exhaustion, refused politely when the game is too large; `mcts/3` is
  Monte-Carlo tree search (UCT) for games that are; `learn/2` fits a value
  function on `features(s)` from self-play and uses it at the leaves of
  the search — measured against plain search at the same number of
  simulations, the control; `match/4` plays agents against each other
  with a Wilson interval on the score.

  Players are whatever values `player(s)` returns; a position with no
  moves and no winner is a draw.
  """
  alias Vapor.Alembic

  defstruct [:prog, :init, :features, :show]

  @doc "Load a game; `{:ok, game}` or `{:error, why}`."
  def load(text) do
    with {:ok, p} <- (case Alembic.load(text) do {:ok, p} -> {:ok, p}; {:error, e} -> {:error, Alembic.format_error(e)} end) do
      missing = Enum.reject([{"player", 1}, {"moves", 1}, {"play", 2}, {"winner", 1}], fn {n, a} -> Alembic.defined?(p, n, a) end)
      cond do
        not Alembic.defined?(p, "init") -> {:error, "a game needs init (the starting position)"}
        missing != [] -> {:error, "a game needs " <> Enum.map_join(missing, ", ", fn {n, a} -> "#{n}(#{Enum.join(Enum.take(~w(s m), a), ", ")})" end)}
        true -> {:ok, %__MODULE__{prog: p, init: Alembic.const(p, "init"), features: Alembic.defined?(p, "features", 1), show: Alembic.defined?(p, "show", 1)}}
      end
    end
  end

  @doc "Is this text a game (has init, moves, play, winner)?"
  def game?(text), do: text =~ ~r/^\s*init\s*=/m and text =~ ~r/^\s*moves\s*\(/m and text =~ ~r/^\s*winner\s*\(/m

  defp call!(g, name, args) do
    case Alembic.call(g.prog, name, args) do
      {:ok, v} -> v
      {:error, m} -> throw({:game_error, "#{name}: #{m}"})
    end
  end

  def player(g, s), do: call!(g, "player", [s])
  def moves(g, s), do: (case call!(g, "moves", [s]) do l when is_list(l) -> l; v -> throw({:game_error, "moves returned #{Alembic.show(v)}, not a list"}) end)
  def play(g, s, m), do: call!(g, "play", [s, m])
  def winner(g, s), do: call!(g, "winner", [s])

  @doc "The board as text (show(s) if defined)."
  def render(g, s) do
    if g.show do
      case Alembic.call(g.prog, "show", [s]) do
        {:ok, t} when is_binary(t) -> t
        {:ok, v} -> Alembic.show(v)
        _ -> Alembic.show(s)
      end
    else
      Alembic.show(s)
    end
  end

  # value for the player to move at a terminal (nil if not terminal)
  defp terminal(g, s) do
    case winner(g, s) do
      nil ->
        if moves(g, s) == [], do: 0, else: nil
      0 -> 0
      w -> if Vapor.Alembic.Builtins.eq?(w, player(g, s)), do: 1, else: -1
    end
  end

  # ================================================================ solve

  @doc """
  Solve exactly from `state` (default `init`). `{:ok, %{value, distance,
  best, moves, positions}}` — value for the player to move (1 win, 0 draw,
  −1 loss), every move's value — or `{:error, why}` past `max_positions`.
  """
  def solve(g, opts \\ []) do
    s = Keyword.get(opts, :state, g.init)
    limit = Keyword.get(opts, :max_positions, 400_000)
    Process.put(:game_tt, %{})
    Process.put(:game_limit, limit)

    try do
      {v, d} = negamax(g, s)
      ms = for m <- moves(g, s) do
        {cv, cd} = negamax(g, play(g, s, m))
        %{move: Alembic.show(m), value: -cv, distance: cd + 1}
      end
      best = ms |> Enum.sort_by(&score_key({&1.value, &1.distance}), :desc) |> Enum.take_while(&(&1.value == v)) |> Enum.filter(&(&1.distance == d))
      {:ok, %{value: v, distance: d, best: Enum.map(best, & &1.move), moves: ms, positions: map_size(Process.get(:game_tt)),
              says: words(v, d)}}
    catch
      :too_large -> {:error, "more than #{limit} positions: too large to solve exactly — use search (mcts) instead"}
      {:game_error, m} -> {:error, m}
    after
      Process.delete(:game_tt)
    end
  end

  defp words(1, d), do: "the player to move wins with best play, in #{d} moves"
  defp words(-1, d), do: "the player to move loses with best play (holding out #{d} moves)"
  defp words(0, _), do: "a draw with best play"

  # win sooner, lose later
  defp score_key({1, d}), do: {2, -d}
  defp score_key({0, d}), do: {1, d}
  defp score_key({-1, d}), do: {0, d}

  defp negamax(g, s) do
    key = Alembic.show(s)
    tt = Process.get(:game_tt)
    case Map.fetch(tt, key) do
      {:ok, r} -> r
      :error ->
        if map_size(tt) >= Process.get(:game_limit), do: throw(:too_large)
        r =
          case terminal(g, s) do
            nil ->
              moves(g, s)
              |> Enum.map(fn m -> {v, d} = negamax(g, play(g, s, m)); {-v, d + 1} end)
              |> Enum.max_by(&score_key/1)
            v -> {v, 0}
          end
        Process.put(:game_tt, Map.put(Process.get(:game_tt), key, r))
        r
    end
  end

  # ================================================================ MCTS

  @doc """
  Monte-Carlo tree search (UCT) from `state`: `%{move, stats}`. Options:
  `sims:` (default 400), `seed:`, `value:` a value function `fn s -> v in
  [−1, 1] for the player to move end` used at the leaves instead of random
  playouts, `c:` exploration.
  """
  def mcts(g, s, opts \\ []) do
    sims = Keyword.get(opts, :sims, 400)
    rng = :rand.seed_s(:exsss, {Keyword.get(opts, :seed, 1), 31, 7})
    c = Keyword.get(opts, :c, 1.4)
    value = Keyword.get(opts, :value)
    root = Alembic.show(s)
    tree = %{}

    {tree, _rng} =
      Enum.reduce(1..sims, {tree, rng}, fn _, {t, r} ->
        {t, r, _v} = simulate(g, s, root, t, r, c, value, 0)
        {t, r}
      end)

    node = Map.fetch!(tree, root)
    stats = Enum.map(node.children, fn {m, ck} -> {m, Map.get(tree, ck, %{n: 0, w: 0.0})} end)
    {best, _} = Enum.max_by(stats, fn {_, st} -> st.n end)
    %{move: best, stats: Enum.map(stats, fn {m, st} -> %{move: Alembic.show(m), visits: st.n, q: if(st.n > 0, do: -st.w / st.n, else: 0.0)} end) |> Enum.sort_by(& &1.visits, :desc)}
  end

  # returns the value for the player to move at s
  defp simulate(g, s, key, tree, rng, c, value, depth) do
    case Map.fetch(tree, key) do
      :error ->
        case terminal(g, s) do
          nil ->
            ms = moves(g, s)
            children = Enum.map(ms, fn m -> {m, Alembic.show(play(g, s, m))} end)
            {v, rng} = if value, do: {value.(s), rng}, else: rollout(g, s, rng, 0)
            {Map.put(tree, key, %{n: 1, w: v, children: children, terminal: nil}), rng, v}
          tv ->
            {Map.put(tree, key, %{n: 1, w: tv * 1.0, children: [], terminal: tv}), rng, tv * 1.0}
        end

      {:ok, %{terminal: tv} = node} when tv != nil ->
        {Map.put(tree, key, %{node | n: node.n + 1, w: node.w + tv}), rng, tv * 1.0}

      {:ok, node} ->
        {m, ck} = select(node, tree, c)
        {tree, rng, cv} = simulate(g, play(g, s, m), ck, tree, rng, c, value, depth + 1)
        v = -cv
        {Map.put(tree, key, %{node | n: node.n + 1, w: node.w + v}), rng, v}
    end
  end

  defp select(node, tree, c) do
    logn = :math.log(max(node.n, 1))
    Enum.max_by(node.children, fn {_m, ck} ->
      case Map.get(tree, ck) do
        nil -> 1.0e9
        st -> -st.w / st.n + c * :math.sqrt(logn / st.n)
      end
    end)
  end

  defp rollout(g, s, rng, depth) do
    case terminal(g, s) do
      nil when depth < 400 ->
        ms = moves(g, s)
        {i, rng} = :rand.uniform_s(length(ms), rng)
        {v, rng} = rollout(g, play(g, s, Enum.at(ms, i - 1)), rng, depth + 1)
        {-v, rng}
      nil -> {0.0, rng}
      v -> {v * 1.0, rng}
    end
  end

  # ================================================================ learning

  @doc """
  Learn a value function v(s) = tanh(w·features(s) + b) from self-play
  games played by search with that same function (`games:`, `sims:`,
  `seed:`); returns `%{weights, games, loss}` — Monte-Carlo targets, SGD.
  """
  def learn(%{features: false}, _opts), do: {:error, "learning needs features(s): a list of numbers describing a position"}

  def learn(g, opts) do
    games = Keyword.get(opts, :games, 40)
    sims = Keyword.get(opts, :sims, 64)
    lr = Keyword.get(opts, :lr, 0.05)
    seed = Keyword.get(opts, :seed, 1)
    nf = length(features(g, g.init))
    w0 = %{w: List.duplicate(0.0, nf), b: 0.0}

    {w, losses} =
      Enum.reduce(1..games, {w0, []}, fn k, {w, losses} ->
        vf = fn s -> vhat(w, features(g, s)) end
        traj = selfplay(g, sims, seed * 1000 + k, if(k > 3, do: vf, else: nil))
        {w, l} = Enum.reduce(traj, {w, 0.0}, fn {f, z}, {w, l} -> sgd(w, f, z, lr) |> then(fn {w2, e} -> {w2, l + e} end) end)
        {w, [l / max(length(traj), 1) | losses]}
      end)

    {:ok, %{weights: w, games: games, sims: sims, loss: Enum.reverse(losses)}}
  end

  defp features(g, s) do
    case Alembic.call(g.prog, "features", [s]) do
      {:ok, f} when is_list(f) -> Enum.map(f, fn x when is_number(x) -> x * 1.0; true -> 1.0; _ -> 0.0 end)
      _ -> throw({:game_error, "features(s) must return a list of numbers"})
    end
  end

  defp vhat(w, f), do: :math.tanh(Enum.zip_with(w.w, f, &(&1 * &2)) |> Enum.sum() |> Kernel.+(w.b))

  defp sgd(w, f, z, lr) do
    y = vhat(w, f)
    e = y - z
    gfac = e * (1 - y * y)
    {%{w: Enum.zip_with(w.w, f, &(&1 - lr * gfac * &2)), b: w.b - lr * gfac}, e * e}
  end

  defp selfplay(g, sims, seed, vf) do
    selfplay_loop(g, g.init, sims, seed, vf, [], 0)
  end

  defp selfplay_loop(g, s, sims, seed, vf, acc, ply) do
    case terminal(g, s) do
      nil when ply < 400 ->
        opts = [sims: sims, seed: seed + ply] ++ if(vf, do: [value: vf], else: [])
        %{stats: stats} = mcts(g, s, opts)
        # sample in proportion to visits for the first plies (exploration), else the most visited
        m = if ply < 2 do
          tot = Enum.sum(Enum.map(stats, & &1.visits))
          r = Vapor.Alembic.Builtins.hash01([seed, ply]) * tot
          pick_visit(stats, r)
        else
          hd(stats).move
        end
        {:ok, mv} = Alembic.literal(m)
        selfplay_loop(g, play(g, s, mv), sims, seed, vf, [{features(g, s), player(g, s)} | acc], ply + 1)

      res ->
        # outcome from each recorded position's mover's view
        win = winner(g, s)
        Enum.map(acc, fn {f, p} ->
          z = cond do res == nil or win in [nil, 0] -> 0.0; Vapor.Alembic.Builtins.eq?(win, p) -> 1.0; true -> -1.0 end
          {f, z}
        end)
    end
  end

  defp pick_visit([m], _r), do: m.move
  defp pick_visit([m | rest], r), do: if(r < m.visits, do: m.move, else: pick_visit(rest, r - m.visits))

  # ================================================================ matches

  @doc """
  Play `n` games between agents `a` and `b` (alternating who starts):
  `:random`, `:solver`, `{:mcts, sims}`, `{:learned, weights, sims}`.
  `%{a_wins, b_wins, draws, score, interval}` — score = (wins + draws/2)/n
  for `a`, with a 95 % Wilson interval.
  """
  def match(g, a, b, n, seed \\ 1) do
    results =
      for k <- 1..n do
        {first, second} = if rem(k, 2) == 1, do: {a, b}, else: {b, a}
        r = game(g, first, second, seed * 7919 + k)
        # r: 1 first wins, -1 second wins, 0 draw → from a's view
        if rem(k, 2) == 1, do: r, else: -r
      end

    aw = Enum.count(results, &(&1 == 1))
    bw = Enum.count(results, &(&1 == -1))
    d = n - aw - bw
    score = (aw + d / 2) / n
    %{a_wins: aw, b_wins: bw, draws: d, games: n, score: score, interval: wilson(score, n)}
  end

  defp game(g, first, second, seed) do
    p1 = player(g, g.init)
    loop = fn loop, s, ply ->
      case winner(g, s) do
        nil ->
          ms = moves(g, s)
          if ms == [] or ply > 400 do
            0
          else
            agent = if Vapor.Alembic.Builtins.eq?(player(g, s), p1), do: first, else: second
            m = choose(g, agent, s, seed + ply)
            loop.(loop, play(g, s, m), ply + 1)
          end
        0 -> 0
        w -> if Vapor.Alembic.Builtins.eq?(w, p1), do: 1, else: -1
      end
    end
    loop.(loop, g.init, 0)
  end

  @doc "An agent's move at `s`."
  def choose(g, :random, s, seed) do
    ms = moves(g, s)
    Enum.at(ms, trunc(Vapor.Alembic.Builtins.hash01([seed, :r]) * length(ms)))
  end

  def choose(g, :solver, s, _seed) do
    {:ok, r} = solve(g, state: s)
    {:ok, m} = Alembic.literal(hd(r.best))
    m
  end

  def choose(g, {:mcts, sims}, s, seed), do: mcts(g, s, sims: sims, seed: seed).move
  def choose(g, {:learned, w, sims}, s, seed), do: mcts(g, s, sims: sims, seed: seed, value: fn x -> vhat(w, features(g, x)) end).move

  @doc "Wilson score interval (95 %) for a proportion p over n trials."
  def wilson(p, n) do
    z = 1.959964
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * :math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    [max(mid - half, 0.0), min(mid + half, 1.0)]
  end

  @doc "Run a game-level operation, turning rule errors into `{:error, why}`."
  def safely(fun) do
    fun.()
  catch
    {:game_error, m} -> {:error, m}
  end
end
