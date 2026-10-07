defmodule Vapor.Games do
  @moduledoc """
  Reinforcement learning by self-play and by manipulating environments
  (docs/JOGOS.md).

  **Self-play with a policy–value network and PUCT search** (the method of
  Silver et al. 2018, at the scale of tic-tac-toe): a network with
  a policy head and a value head (18 inputs → 64 tanh → 9 + 1), Monte
  Carlo tree search guided by it (PUCT), and training only on the games
  it plays against itself — no human games, no rules beyond legality and
  the end. Judged against a **perfect player** (negamax over the whole
  game tree): `versus_every_optimal_line/2` follows **every** optimal line
  of the perfect player from both sides. Measured: with 8 simulations the
  trained search loses 17 of 129 lines, the untrained (the control) 169 of
  175; at 64 both lose 4 lines (a sampled 60 games had shown 0 — the
  exhaustive count is the honest one); at 128 the trained search loses
  none.

  **Domain randomisation** (Tobin et al. 2017): the cart-pole of
  gymnasium with its physical constants exposed; a linear policy trained
  by random search on the nominal cart-pole, and the same training on
  cart-poles drawn from a range (pole length, masses, motor force),
  both tested on cart-poles neither saw. The randomised policy must hold
  the shifted poles up longer.

  Everything is a function of its seed (`Vapor.Sampler`): a training run
  is replayable to the bit (`Vapor.Archive`).
  """
  alias Vapor.Sampler

  # ============================================================ tic-tac-toe

  @lines [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

  def empty, do: List.duplicate(0, 9) |> List.to_tuple()
  def to_move(b), do: if(Enum.count(Tuple.to_list(b), &(&1 != 0)) |> rem(2) == 0, do: 1, else: -1)
  def legal(b), do: for(i <- 0..8, elem(b, i) == 0, do: i)
  def play(b, i), do: put_elem(b, i, to_move(b))

  @doc "1 if X has won, −1 if O, 0 for a draw, nil if the game goes on."
  def winner(b) do
    case Enum.find(@lines, fn [x, y, z] -> elem(b, x) != 0 and elem(b, x) == elem(b, y) and elem(b, y) == elem(b, z) end) do
      [x | _] -> elem(b, x)
      nil -> if legal(b) == [], do: 0, else: nil
    end
  end

  # ---------------------------------------------------------- the perfect player

  @doc "Negamax value of a position for the player to move (+1 win, 0 draw, −1 loss), memoised."
  def solve(b), do: elem(negamax(b, %{}), 0)

  defp negamax(b, memo) do
    case Map.fetch(memo, b) do
      {:ok, v} -> {v, memo}
      :error ->
        {v, memo} =
          case winner(b) do
            nil ->
              Enum.reduce(legal(b), {-2, memo}, fn i, {best, memo} ->
                {v, memo} = negamax(play(b, i), memo)
                {max(best, -v), memo}
              end)

            0 -> {0, memo}
            w -> {w * to_move(b), memo}
          end

        {v, Map.put(memo, b, v)}
    end
  end

  @doc "The optimal moves of a position."
  def optimal(b) do
    vals = for i <- legal(b), do: {i, -solve(play(b, i))}
    best = vals |> Enum.map(&elem(&1, 1)) |> Enum.max()
    for {i, v} <- vals, v == best, do: i
  end

  # ------------------------------------------------------------- the network

  @hidden 64

  @doc "A network with seeded small random weights."
  def net(seed \\ 1) do
    r = fn salt, k, scale -> (Sampler.uniform(seed * 1000 + salt, k) - 0.5) * 2 * scale end
    %{w1: for(h <- 0..(@hidden - 1), do: for(i <- 0..17, do: r.(1, h * 18 + i, 0.3))), b1: List.duplicate(0.0, @hidden),
      wp: for(a <- 0..8, do: for(h <- 0..(@hidden - 1), do: r.(2, a * @hidden + h, 0.1))), bp: List.duplicate(0.0, 9),
      wv: for(h <- 0..(@hidden - 1), do: r.(3, h, 0.1)), bv: 0.0}
  end

  # the position from the mover's side: 9 own stones, 9 opponent's
  defp features(b) do
    p = to_move(b)
    l = Tuple.to_list(b)
    Enum.map(l, &if(&1 == p, do: 1.0, else: 0.0)) ++ Enum.map(l, &if(&1 == -p, do: 1.0, else: 0.0))
  end

  defp dot(a, b), do: Enum.zip_with(a, b, &*/2) |> Enum.sum()

  @doc "Policy over the legal moves (softmax of the logits) and value for the mover."
  def evaluate(n, b) do
    x = features(b)
    h = Enum.zip_with(n.w1, n.b1, fn w, bb -> :math.tanh(dot(w, x) + bb) end)
    logits = Enum.zip_with(n.wp, n.bp, fn w, bb -> dot(w, h) + bb end)
    moves = legal(b)
    m = moves |> Enum.map(&Enum.at(logits, &1)) |> Enum.max()
    ex = Map.new(moves, fn i -> {i, :math.exp(Enum.at(logits, i) - m)} end)
    z = ex |> Map.values() |> Enum.sum()
    {Map.new(ex, fn {i, e} -> {i, e / z} end), :math.tanh(dot(n.wv, h) + n.bv), %{x: x, h: h, logits: logits}}
  end

  # ------------------------------------------------------------------- MCTS

  @c_puct 1.5

  @doc """
  PUCT search from `b` with `sims` simulations: the visit counts of the
  root's moves. `noise: {seed, k}` mixes uniform noise into the root
  priors (exploration during self-play).
  """
  def mcts(n, b, sims, opts \\ []) do
    tree = Enum.reduce(1..sims, %{}, fn _, tree -> elem(simulate(n, b, tree, true, opts), 1) end)
    %{n: visits} = tree[b]
    visits
  end

  # returns {value for the player to move at b, tree}
  defp simulate(n, b, tree, root?, opts) do
    case winner(b) do
      nil ->
        case Map.fetch(tree, b) do
          :error ->
            {p, v, _} = evaluate(n, b)
            p = if root? and opts[:noise], do: noisy(p, opts[:noise]), else: p
            {v, Map.put(tree, b, %{p: p, n: Map.new(Map.keys(p), &{&1, 0}), w: Map.new(Map.keys(p), &{&1, 0.0})})}

          {:ok, node} ->
            total = node.n |> Map.values() |> Enum.sum()
            a = Enum.max_by(Map.keys(node.p), fn a ->
              q = if node.n[a] == 0, do: 0.0, else: node.w[a] / node.n[a]
              q + @c_puct * node.p[a] * :math.sqrt(total + 1) / (1 + node.n[a])
            end)

            {child_v, tree} = simulate(n, play(b, a), tree, false, opts)
            v = -child_v
            node = tree[b]
            {v, Map.put(tree, b, %{node | n: Map.update!(node.n, a, &(&1 + 1)), w: Map.update!(node.w, a, &(&1 + v))})}
        end

      0 -> {0.0, tree}
      w -> {w * to_move(b) * 1.0, tree}
    end
  end

  defp noisy(p, {seed, k}) do
    raw = Map.new(p, fn {a, _} -> {a, -:math.log(max(Sampler.uniform(seed, k * 16 + a), 1.0e-12))} end)
    s = raw |> Map.values() |> Enum.sum()
    Map.new(p, fn {a, pa} -> {a, 0.75 * pa + 0.25 * raw[a] / s} end)
  end

  @doc "The agent's move: the most visited after `sims` simulations."
  def best_move(n, b, sims), do: n |> mcts(b, sims) |> Enum.max_by(fn {a, c} -> {c, -a} end) |> elem(0)

  # --------------------------------------------------------------- training

  @doc """
  Self-play training: `games` games, each move searched with `sims`
  simulations (root noise; the first two moves sampled by visit counts),
  every position stored with the search's visit distribution and the
  game's outcome; after each batch of games, `epochs` passes of SGD on the
  replay buffer. Returns `%{net, games, positions, losses}`.
  """
  def train(opts \\ []) do
    seed = Keyword.get(opts, :seed, 1)
    games = Keyword.get(opts, :games, 300)
    sims = Keyword.get(opts, :sims, 32)
    batch = Keyword.get(opts, :batch_games, 20)
    lr = Keyword.get(opts, :lr, 0.05)

    {n, buf, losses} =
      Enum.reduce(1..div(games, batch), {net(seed), [], []}, fn round, {n, buf, losses} ->
        # the board's eight symmetries: each position seen in all of them
        new = for g <- 1..batch, pos <- self_play(n, sims, seed * 100_000 + round * 1000 + g), sym <- symmetries(), do: transform(pos, sym)
        buf = Enum.take(new ++ buf, Keyword.get(opts, :buffer, 12_000))
        {n, l} = sgd(n, buf, Keyword.get(opts, :epochs, 1), lr, seed * 7 + round)
        {n, buf, [l | losses]}
      end)

    %{net: n, games: games, positions: length(buf), losses: Enum.reverse(losses)}
  end

  defp self_play(n, sims, salt) do
    {hist, z} =
      Enum.reduce_while(0..9, {empty(), []}, fn k, {b, hist} ->
        case winner(b) do
          nil ->
            visits = mcts(n, b, sims, noise: {salt, k})
            total = visits |> Map.values() |> Enum.sum()
            pi = Map.new(visits, fn {a, c} -> {a, c / total} end)
            a = if k < 2, do: sample(pi, Sampler.uniform(salt + 1, k)), else: visits |> Enum.max_by(fn {a, c} -> {c, -a} end) |> elem(0)
            {:cont, {play(b, a), [{b, pi} | hist]}}

          w ->
            {:halt, {hist, w}}
        end
      end)
      |> case do
        {hist, w} when is_integer(w) -> {hist, w}
        {b, hist} -> {hist, winner(b)}
      end

    for {b, pi} <- hist, do: {b, pi, z * to_move(b) * 1.0}
  end

  # the dihedral group of the square, as index permutations of the 3×3 board
  defp symmetries do
    rot = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    flip = [2, 1, 0, 5, 4, 3, 8, 7, 6]
    id = Enum.to_list(0..8)
    compose = fn a, b -> Enum.map(b, &Enum.at(a, &1)) end
    rots = Enum.scan(1..3, id, fn _, acc -> compose.(acc, rot) end)
    base = [id | rots]
    base ++ Enum.map(base, &compose.(&1, flip))
  end

  defp transform({b, pi, z}, perm) do
    # new board cell i takes old cell perm[i]; a move to old cell a becomes the new cell holding it
    inv = perm |> Enum.with_index() |> Map.new()
    {List.to_tuple(Enum.map(perm, &elem(b, &1))), Map.new(pi, fn {a, p} -> {inv[a], p} end), z}
  end

  defp sample(pi, u) do
    pi |> Enum.sort() |> Enum.reduce_while(0.0, fn {a, p}, acc -> if acc + p >= u, do: {:halt, {:pick, a}}, else: {:cont, acc + p} end)
    |> case do
      {:pick, a} -> a
      _ -> pi |> Map.keys() |> Enum.max()
    end
  end

  # plain SGD on (π − p)·log-loss + (z − v)², by hand-written backpropagation
  defp sgd(n, buf, epochs, lr, seed) do
    data = Vapor.Modal.Rng.permute(buf, seed)

    Enum.reduce(1..epochs, {n, 0.0}, fn _, {n, _} ->
      Enum.reduce(data, {n, 0.0}, fn {b, pi, z}, {n, loss} ->
        {p, v, %{x: x, h: h}} = evaluate(n, b)
        moves = Map.keys(p)
        # dL/dlogit_a = p_a − π_a on legal moves (0 elsewhere); dL/dv = 2(v − z)
        dlog = for a <- 0..8, do: if(a in moves, do: p[a] - Map.get(pi, a, 0.0), else: 0.0)
        dv = 2 * (v - z) * (1 - v * v)
        dh = for j <- 0..(@hidden - 1), do: (Enum.sum(for a <- 0..8, do: Enum.at(dlog, a) * (n.wp |> Enum.at(a) |> Enum.at(j))) + dv * Enum.at(n.wv, j)) * (1 - Enum.at(h, j) ** 2)
        l = (v - z) ** 2 - Enum.sum(for a <- moves, do: Map.get(pi, a, 0.0) * :math.log(max(p[a], 1.0e-12)))

        n = %{n |
          wp: Enum.zip_with(n.wp, dlog, fn row, g -> Enum.zip_with(row, h, fn w, hj -> w - lr * g * hj end) end),
          bp: Enum.zip_with(n.bp, dlog, fn bb, g -> bb - lr * g end),
          wv: Enum.zip_with(n.wv, h, fn w, hj -> w - lr * dv * hj end),
          bv: n.bv - lr * dv,
          w1: Enum.zip_with(n.w1, dh, fn row, g -> Enum.zip_with(row, x, fn w, xi -> w - lr * g * xi end) end),
          b1: Enum.zip_with(n.b1, dh, fn bb, g -> bb - lr * g end)}

        {n, loss + l / length(data)}
      end)
    end)
  end

  # ------------------------------------------------------------- evaluation

  @doc """
  The agent (search with `sims`) against the perfect player, `games` per
  side; the perfect player picks among its optimal moves with a seeded
  draw, so many optimal lines are tried. `%{losses, draws, wins}` summed
  over both sides.
  """
  def versus_perfect(n, sims, games, seed \\ 1) do
    results =
      for side <- [1, -1], g <- 1..games do
        Enum.reduce_while(0..9, empty(), fn k, b ->
          case winner(b) do
            nil ->
              a = if to_move(b) == side, do: best_move(n, b, sims), else: (opt = optimal(b); Enum.at(opt, trunc(Sampler.uniform(seed * 977 + g * 2 + max(side, 0), k) * length(opt)) |> min(length(opt) - 1)))
              {:cont, play(b, a)}

            w ->
              {:halt, w * side}
          end
        end)
      end

    %{wins: Enum.count(results, &(&1 == 1)), draws: Enum.count(results, &(&1 == 0)), losses: Enum.count(results, &(&1 == -1)), games: length(results)}
  end

  @doc """
  The agent against **every** optimal line of the perfect player, from
  both sides: the agent's search is deterministic, so the game tree it
  faces branches only where the perfect player has several optimal moves
  — and all of them are followed. `%{lines, losses, draws, wins}`. This
  is the exhaustive form of `versus_perfect/4` (tic-tac-toe is small
  enough for it).
  """
  def versus_every_optimal_line(n, sims) do
    {tally, _memo} =
      Enum.reduce([1, -1], {%{lines: 0, losses: 0, draws: 0, wins: 0}, %{}}, fn side, {t, memo} ->
        {t2, memo} = lines(n, sims, empty(), side, memo)
        {Map.merge(t, t2, fn _, a, b -> a + b end), memo}
      end)

    tally
  end

  defp lines(n, sims, b, side, memo) do
    case winner(b) do
      nil ->
        if to_move(b) == side do
          {a, memo} =
            case memo do
              %{^b => a} -> {a, memo}
              _ -> (a = best_move(n, b, sims); {a, Map.put(memo, b, a)})
            end

          lines(n, sims, play(b, a), side, memo)
        else
          Enum.reduce(optimal(b), {%{lines: 0, losses: 0, draws: 0, wins: 0}, memo}, fn a, {t, memo} ->
            {t2, memo} = lines(n, sims, play(b, a), side, memo)
            {Map.merge(t, t2, fn _, x, y -> x + y end), memo}
          end)
        end

      w ->
        k = case w * side do 1 -> :wins; 0 -> :draws; _ -> :losses end
        {%{lines: 1, losses: 0, draws: 0, wins: 0} |> Map.put(k, 1), memo}
    end
  end

  @doc "The agent against a uniformly random player, `games` per side."
  def versus_random(n, sims, games, seed \\ 1) do
    results =
      for side <- [1, -1], g <- 1..games do
        Enum.reduce_while(0..9, empty(), fn k, b ->
          case winner(b) do
            nil ->
              moves = legal(b)
              a = if to_move(b) == side, do: best_move(n, b, sims), else: Enum.at(moves, trunc(Sampler.uniform(seed * 311 + g * 2 + max(side, 0), k) * length(moves)) |> min(length(moves) - 1))
              {:cont, play(b, a)}

            w ->
              {:halt, w * side}
          end
        end)
      end

    %{wins: Enum.count(results, &(&1 == 1)), draws: Enum.count(results, &(&1 == 0)), losses: Enum.count(results, &(&1 == -1)), games: length(results)}
  end

  # ===================================================== domain randomisation

  @nominal %{gravity: 9.8, masscart: 1.0, masspole: 0.1, length: 0.5, force: 10.0, tau: 0.02}

  @doc "gymnasium's cart-pole constants."
  def nominal, do: @nominal

  @doc "One cart-pole step with physical parameters `p` (gymnasium's equations, Euler)."
  def cart_step([x, xd, th, thd], a, p) do
    f = if a == 1, do: p.force, else: -p.force
    total = p.masscart + p.masspole
    pml = p.masspole * p.length
    {c, s} = {:math.cos(th), :math.sin(th)}
    temp = (f + pml * thd * thd * s) / total
    thacc = (p.gravity * s - c * temp) / (p.length * (4.0 / 3.0 - p.masspole * c * c / total))
    xacc = temp - pml * thacc * c / total
    [x + p.tau * xd, xd + p.tau * xacc, th + p.tau * thd, thd + p.tau * thacc]
  end

  @doc "Steps a linear policy a = [w·s > 0] keeps the pole up (≤ 500), from a seeded start."
  def episode(w, p, seed, k) do
    s0 = for i <- 0..3, do: Sampler.uniform(seed, k * 4 + i) * 0.1 - 0.05

    Enum.reduce_while(1..500, s0, fn t, s ->
      s2 = cart_step(s, if(dot(w, s) > 0, do: 1, else: 0), p)
      [x, _, th, _] = s2
      if abs(x) > 2.4 or abs(th) > 0.2095, do: {:halt, t}, else: {:cont, s2}
    end)
    |> case do
      t when is_integer(t) -> t
      _ -> 500
    end
  end

  @doc "A cart-pole drawn from the training range (pole length ×0.5–3, masses ×0.5–4, motor force ×0.3–1)."
  def random_params(seed, k) do
    u = fn i -> Sampler.uniform(seed, k * 8 + i) end
    %{@nominal | length: 0.25 + 1.25 * u.(0), masspole: 0.05 + 0.35 * u.(1), masscart: 0.5 + 1.5 * u.(2), force: 3.0 + 7.0 * u.(3)}
  end

  @doc """
  Held-out cart-poles neither training saw (the test of robustness): the
  first three lie outside the randomisation box (pole 1.6 and 0.45 kg; cart
  2.4 kg; pole 0.2 and 0.5 kg), the fourth in its corner (every parameter
  near its hardest edge at once — inside the box, rarely drawn).
  """
  def shifted do
    [%{@nominal | length: 1.6, masspole: 0.45, force: 3.5}, %{@nominal | length: 1.4, masscart: 2.4, force: 3.0},
     %{@nominal | length: 0.2, masspole: 0.5, force: 4.0}, %{@nominal | length: 1.5, masspole: 0.3, masscart: 2.0, force: 3.2}]
  end

  @doc """
  Random search (ARS-style, two-sided) for a linear policy: `randomize:
  true` draws a new cart-pole per evaluation episode. `%{w, curve}`.
  """
  def train_cartpole(opts \\ []) do
    seed = Keyword.get(opts, :seed, 1)
    iters = Keyword.get(opts, :iters, 60)
    dirs = Keyword.get(opts, :dirs, 6)
    eps = Keyword.get(opts, :episodes, 6)
    rand = Keyword.get(opts, :randomize, false)
    params = fn k -> if rand, do: random_params(seed * 13, k), else: @nominal end
    score = fn w, it -> Enum.sum(for e <- 1..eps, do: episode(w, params.(it * 100 + e), seed + 5, it * 100 + e)) / eps end

    Enum.reduce(1..iters, {[0.0, 0.0, 0.0, 0.0], []}, fn it, {w, curve} ->
      grads =
        for d <- 1..dirs do
          delta = for i <- 0..3, do: Sampler.uniform(seed * 31 + it, d * 4 + i) - 0.5
          plus = score.(Enum.zip_with(w, delta, &(&1 + 0.5 * &2)), it)
          minus = score.(Enum.zip_with(w, delta, &(&1 - 0.5 * &2)), it)
          Enum.map(delta, &(&1 * (plus - minus)))
        end

      g = Enum.zip_with(grads, fn col -> Enum.sum(col) / dirs end)
      w = Enum.zip_with(w, g, &(&1 + 0.02 * &2))
      {w, [score.(w, it) | curve]}
    end)
    |> then(fn {w, curve} -> %{w: w, curve: Enum.reverse(curve)} end)
  end

  @doc "Mean steps up on the nominal pole and on the shifted ones, over `n` starts each."
  def robustness(w, n \\ 10) do
    nom = Enum.sum(for k <- 1..n, do: episode(w, @nominal, 77, k)) / n
    sh = Enum.sum(for p <- shifted(), k <- 1..n, do: episode(w, p, 78, k)) / (n * length(shifted()))
    %{nominal: nom, shifted: sh}
  end

  # ============================================================== save/load

  @doc "Save a trained network with its recipe and digest (JSON)."
  def save(n, path, recipe) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Vapor.JSON.encode(%{"recipe" => recipe, "digest" => digest(n), "net" => Map.new(n, fn {k, v} -> {Atom.to_string(k), v} end)}))
  end

  @doc "The shipped tic-tac-toe network (`priv/games/tictactoe.json`) or one at `path`."
  def load(path \\ Path.join([to_string(:code.priv_dir(:vapor)), "games", "tictactoe.json"])) do
    {:ok, m} = Vapor.JSON.decode(File.read!(path))
    n = Map.new(m["net"], fn {k, v} -> {String.to_existing_atom(k), v} end)
    if digest(n) == m["digest"], do: {:ok, n, m["recipe"]}, else: {:error, :digest_mismatch}
  end

  @doc "The canonical digest of a network's weights."
  def digest(n), do: Vapor.Canonical.hex_digest(Enum.map([:w1, :b1, :wp, :bp, :wv, :bv], &n[&1]))

  # ================================================================= replay

  @doc false
  # bounded: the shipped recipe (400 games, 32 simulations) is the ceiling
  # "games.selfplay" is the kind; "games.alphazero" is kept so archives saved by 0.11 still replay
  def replay(kind, %{"games" => g, "sims" => s, "seed" => seed}) when kind in ["games.selfplay", "games.alphazero"] do
    with {:ok, g} <- Vapor.Discover.bounded(g, 1..400), {:ok, s} <- Vapor.Discover.bounded(s, 1..64), {:ok, seed} <- Vapor.Discover.bounded(seed, 0..1_000_000_000) do
      %{net: n} = train(games: g, sims: s, seed: seed)
      {:ok, %{versus_perfect: versus_perfect(n, s, 10, seed)}}
    end
  end

  def replay(_, _), do: {:error, :bad_recipe}
end
