defmodule Vapor.RL do
  @moduledoc """
  **Reinforcement learning whose runs are reproducible to the last bit** —
  environments, learners and replays, from first principles.

  RL results are notoriously hard to reproduce: the same code and seed give
  different curves on another GPU, another thread count, another library
  version. Here an environment is a pure function of its state and action
  (binary64 arithmetic plus the correctly rounded `sin`/`cos` of
  `Vapor.CR`), the randomness is a counter-based generator of `(seed,
  step)`, and the policy is a certified program (`Vapor.Learn`) — so a
  training run is a function of its seed, and a **replay is a seed and a
  list of actions**: re-simulated anywhere, it gives the same frames.

  Environments (`reset/2`, `step/3`, `observe/2`, `render/3`):

    * `:cartpole` — gymnasium's CartPole-v1 dynamics (Barto, Sutton &
      Anderson): the same equations, Euler integration, thresholds and
      500-step limit; trajectories equal gymnasium's to 1e-12 from the
      same state and actions (`test/vapor/rl_test.exs`).
    * `:frozenlake` — gymnasium's FrozenLake 4×4, slippery: its transition
      table, entry for entry.
    * `:reacher` — a planar two-link arm (links 0.5 + 0.5) reaching a goal:
      the robotics exercise (behaviour cloning from an analytic expert).

  Learners: tabular Q-learning (`q_learning/2`), exact value iteration
  (`value_iteration/2`, the optimum to compare with), REINFORCE with a
  baseline (`reinforce/2`, the gradient through `Vapor.Autodiff`) and
  behaviour cloning (`clone/2`).
  """
  alias Vapor.{CR, Learn}
  alias Vapor.Modal.Image
  alias Vapor.Sampler
  alias Vapor.Studio.Video

  # ------------------------------------------------------------ cartpole --

  @g 9.8
  @mc 1.0
  @mp 0.1
  @total @mc + @mp
  @len 0.5
  @pml @mp * @len
  @force 10.0
  @tau 0.02
  @theta_lim 12 * 2 * :math.pi() / 360
  @x_lim 2.4

  @doc "Initial state from a seed (gymnasium: each of x, ẋ, θ, θ̇ uniform in ±0.05)."
  def reset(:cartpole, seed), do: %{s: Enum.map(0..3, fn i -> Sampler.uniform(seed, i) * 0.1 - 0.05 end), t: 0, done: false}

  def reset(:frozenlake, _seed), do: %{s: 0, t: 0, done: false}

  def reset(:reacher, seed) do
    # a goal in the reachable annulus, the arm at rest
    r = 0.25 + 0.7 * Sampler.uniform(seed, 0)
    a = 2 * :math.pi() * Sampler.uniform(seed, 1)
    %{q: [0.3, 0.6], goal: {r * CR.cos_f64(a), r * CR.sin_f64(a)}, t: 0, done: false}
  end

  @doc "One step: `{state, reward}` (the state carries `done`). Stochastic environments draw from `(seed, t)`."
  def step(env, state, action, seed \\ 0)

  def step(:cartpole, %{s: [x, xd, th, thd], t: t} = st, a, _seed) do
    f = if a == 1, do: @force, else: -@force
    {c, s} = {CR.cos_f64(th), CR.sin_f64(th)}
    temp = (f + @pml * thd * thd * s) / @total
    thacc = (@g * s - c * temp) / (@len * (4.0 / 3.0 - @mp * c * c / @total))
    xacc = temp - @pml * thacc * c / @total
    s2 = [x + @tau * xd, xd + @tau * xacc, th + @tau * thd, thd + @tau * thacc]
    [x2, _, th2, _] = s2
    term = x2 < -@x_lim or x2 > @x_lim or th2 < -@theta_lim or th2 > @theta_lim
    {%{st | s: s2, t: t + 1, done: term or t + 1 >= 500}, 1.0}
  end

  @lake ~w(SFFF FHFH FFFH HFFG)

  def step(:frozenlake, %{s: s, t: t} = st, a, seed) do
    # slippery: the intended move or one of the two perpendicular ones, a third each
    u = Sampler.uniform(seed, t)
    move = rem(a + 3 + min(trunc(u * 3), 2), 4)
    s2 = lake_move(s, move)
    tile = lake_tile(s2)
    {%{st | s: s2, t: t + 1, done: tile in [?H, ?G] or t + 1 >= 100}, if(tile == ?G, do: 1.0, else: 0.0)}
  end

  def step(:reacher, %{q: [q1, q2], goal: g, t: t} = st, [d1, d2], _seed) do
    clip = fn v -> min(0.1, max(-0.1, v)) end
    q = [q1 + clip.(d1), q2 + clip.(d2)]
    d = dist(effector(q), g)
    {%{st | q: q, t: t + 1, done: d < 0.02 or t + 1 >= 100}, -d}
  end

  defp lake_tile(s), do: :binary.at(Enum.at(@lake, div(s, 4)), rem(s, 4))

  defp lake_move(s, a) do
    {r, c} = {div(s, 4), rem(s, 4)}
    {r, c} = case a do
      0 -> {r, max(c - 1, 0)}
      1 -> {min(r + 1, 3), c}
      2 -> {r, min(c + 1, 3)}
      3 -> {max(r - 1, 0), c}
    end
    r * 4 + c
  end

  @doc "FrozenLake's transition table: `%{state => %{action => [{prob, next, reward, done}]}}` (gymnasium's `P`)."
  def lake_table do
    for s <- 0..15, into: %{} do
      acts =
        for a <- 0..3, into: %{} do
          if lake_tile(s) in [?H, ?G] do
            {a, [{1.0, s, 0.0, true}, {1.0, s, 0.0, true}, {1.0, s, 0.0, true}]}
          else
            {a, for(b <- [rem(a + 3, 4), a, rem(a + 1, 4)], do: (fn s2 -> {1 / 3, s2, if(lake_tile(s2) == ?G, do: 1.0, else: 0.0), lake_tile(s2) in [?H, ?G]} end).(lake_move(s, b)))}
          end
        end

      {s, acts}
    end
  end

  @doc "Observation vector (floats) of a state."
  def observe(:cartpole, %{s: s}), do: s
  def observe(:frozenlake, %{s: s}), do: for(i <- 0..15, do: if(i == s, do: 1.0, else: 0.0))

  def observe(:reacher, %{q: [q1, q2] = q, goal: {gx, gy}}) do
    {ex, ey} = effector(q)
    [CR.cos_f64(q1), CR.sin_f64(q1), CR.cos_f64(q2), CR.sin_f64(q2), gx, gy, ex - gx, ey - gy]
  end

  @doc "End-effector position of the two-link arm."
  def effector([q1, q2]), do: {0.5 * CR.cos_f64(q1) + 0.5 * CR.cos_f64(q1 + q2), 0.5 * CR.sin_f64(q1) + 0.5 * CR.sin_f64(q1 + q2)}

  defp dist({a, b}, {c, d}), do: :math.sqrt((a - c) * (a - c) + (b - d) * (b - d))

  @doc """
  The reacher's analytic expert: inverse kinematics (elbow down) to the
  goal, then a proportional step towards those angles (clipped like any
  action). It uses the host's `acos`/`atan2` (there is no correctly
  rounded one here): its actions are *data* — hashed into the dataset's
  digest — not a certified computation.
  """
  def expert(%{q: [q1, q2], goal: {gx, gy}}) do
    d2 = min(1.0, max(-1.0, (gx * gx + gy * gy - 0.5) / 0.5))
    t2 = :math.acos(d2)
    t1 = :math.atan2(gy, gx) - :math.atan2(0.5 * :math.sin(t2), 0.5 + 0.5 * :math.cos(t2))
    wrap = fn a -> a - 2 * :math.pi() * Float.round(a / (2 * :math.pi())) end
    clip = fn v -> min(0.1, max(-0.1, v)) end
    [clip.(0.5 * wrap.(t1 - q1)), clip.(0.5 * wrap.(t2 - q2))]
  end

  # ------------------------------------------------------------- tabular --

  @doc "Exact value iteration on FrozenLake (γ = `gamma`): `{values, greedy policy}`."
  def value_iteration(gamma \\ 0.99, iters \\ 2000) do
    p = lake_table()
    v = Enum.reduce(1..iters, List.duplicate(0.0, 16), fn _, v ->
      vt = List.to_tuple(v)
      for s <- 0..15, do: Enum.max(for(a <- 0..3, do: q_of(p, vt, s, a, gamma)))
    end)
    vt = List.to_tuple(v)
    {v, for(s <- 0..15, do: Enum.max_by(0..3, &q_of(p, vt, s, &1, gamma)))}
  end

  defp q_of(p, vt, s, a, gamma), do: p[s][a] |> Enum.map(fn {pr, s2, r, d} -> pr * (r + if(d, do: 0.0, else: gamma * elem(vt, s2))) end) |> Enum.sum()

  @doc """
  Tabular Q-learning on FrozenLake: ε-greedy (ε decaying linearly to 0.05),
  step size `alpha`, discount `gamma`. Returns `{q_table, greedy policy}`.
  """
  def q_learning(episodes, opts \\ []) do
    {alpha, gamma, seed} = {Keyword.get(opts, :alpha, 0.1), Keyword.get(opts, :gamma, 0.99), Keyword.get(opts, :seed, 1)}
    q0 = :array.new(64, default: 0.0)

    q =
      Enum.reduce(0..(episodes - 1), q0, fn ep, q ->
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))
        run_episode(q, reset(:frozenlake, 0), ep, seed, eps, alpha, gamma)
      end)

    table = for s <- 0..15, do: for(a <- 0..3, do: :array.get(s * 4 + a, q))
    {table, Enum.map(table, fn row -> row |> Enum.with_index() |> Enum.max_by(fn {v, a} -> {v, -a} end) |> elem(1) end)}
  end

  defp run_episode(q, st, ep, seed, eps, alpha, gamma) do
    if st.done do
      q
    else
      k = ep * 1000 + st.t
      a = if Sampler.uniform(seed * 31 + 7, k) < eps, do: trunc(Sampler.uniform(seed * 31 + 8, k) * 4) |> min(3), else: greedy(q, st.s)
      {st2, r} = step(:frozenlake, st, a, seed * 1_000_003 + ep)
      best = if st2.done, do: 0.0, else: Enum.max(for b <- 0..3, do: :array.get(st2.s * 4 + b, q))
      old = :array.get(st.s * 4 + a, q)
      q = :array.set(st.s * 4 + a, old + alpha * (r + gamma * best - old), q)
      run_episode(q, st2, ep, seed, eps, alpha, gamma)
    end
  end

  defp greedy(q, s), do: Enum.max_by(0..3, fn a -> {:array.get(s * 4 + a, q), -a} end)

  @doc "Success rate of a FrozenLake policy (a list of 16 actions) over `n` seeded episodes."
  def lake_success(policy, n, seed \\ 99) do
    pt = List.to_tuple(policy)
    wins = Enum.count(0..(n - 1), fn ep -> play(reset(:frozenlake, 0), fn st -> elem(pt, st.s) end, seed * 100_003 + ep) > 0 end)
    wins / n
  end

  defp play(st, pol, seed, ret \\ 0.0) do
    if st.done, do: ret, else: (fn {st2, r} -> play(st2, pol, seed, ret + r) end).(step(:frozenlake, st, pol.(st), seed))
  end

  # -------------------------------------------------------- policy gradient --

  @pg_batch 1024

  @doc """
  REINFORCE with a mean baseline on CartPole: a softmax policy
  `MLP(4 → 32 → 32 → 2)`, `envs` episodes per update (run in lockstep, one
  certified forward pass per step for all of them), returns normalized per
  update; one AdamW step per update on the gradient of −Σ Â·log π(a|s) over
  a seeded sample of 1024 rows. Options: `updates` (100), `envs` (16), `lr` (5e-3),
  `gamma` (0.99), `seed`, `worker` (nil: the oracle). Returns
  `{net, %{returns: [mean return per update]}}`.
  """
  def reinforce(opts \\ []) do
    {updates, envs, seed} = {Keyword.get(opts, :updates, 100), Keyword.get(opts, :envs, 16), Keyword.get(opts, :seed, 1)}
    gamma = Keyword.get(opts, :gamma, 0.99)
    w = Keyword.get(opts, :worker)
    net = Learn.new([4, 32, 32, 2], seed)
    batch = Keyword.get(opts, :batch, @pg_batch)
    {:ok, comp} = Vapor.Compile.Lower.lower(pg_program(net, Keyword.get(opts, :lr, 5.0e-3), batch))

    {net, _state, _t, curve} =
      Enum.reduce(0..(updates - 1), {net, nil, 0, []}, fn u, {net, state, t, curve} ->
        trajs = rollouts(net, envs, seed * 10_000 + u * envs, w)
        mean_ret = (trajs |> Enum.map(&length/1) |> Enum.sum()) / envs
        {xs, as, advs} = batch(trajs, gamma)
        # one gradient step per update, on a seeded sample of the batch (several steps on the same
        # episodes would overfit them: the classic instability of on-policy updates)
        pick = Vapor.Modal.Rng.permute(Enum.to_list(0..(length(xs) - 1)), seed * 31 + u) |> Enum.take(batch) |> Enum.sort()
        sub = fn l -> lt = List.to_tuple(l); Enum.map(pick, &elem(lt, &1)) end
        {net, state, t} = pg_update(comp, net, state, t, sub.(xs), sub.(as), sub.(advs), w, batch)
        {net, state, t, [mean_ret | curve]}
      end)

    {net, %{returns: Enum.reverse(curve)}}
  end

  @doc "Run `n` CartPole episodes in lockstep with a softmax policy (`greedy: true` takes the argmax). Returns the trajectories `[[{obs, action}]]`."
  def rollouts(%Learn{} = net, n, seed, w, opts \\ []) do
    greedy = Keyword.get(opts, :greedy, false)
    states = for i <- 0..(n - 1), do: reset(:cartpole, seed + i)
    go(net, states, List.duplicate([], n), seed, w, greedy)
  end

  defp go(net, states, acc, seed, w, greedy) do
    live = for {st, i} <- Enum.with_index(states), not st.done, do: i

    if live == [] do
      Enum.map(acc, &Enum.reverse/1)
    else
      obs = for i <- live, do: observe(:cartpole, Enum.at(states, i))
      logits = Learn.predict(net, obs, worker: w, rows: 16 * div(length(live) + 15, 16))
      st = List.to_tuple(states)
      ac = List.to_tuple(acc)

      {st, ac} =
        Enum.zip([live, obs, logits])
        |> Enum.reduce({st, ac}, fn {i, o, [l0, l1]}, {st, ac} ->
          s = elem(st, i)
          p1 = 1.0 / (1.0 + CR.exp_f64(min(700.0, max(-700.0, l0 - l1))))
          a = if greedy, do: (if l1 > l0, do: 1, else: 0), else: (if Sampler.uniform(seed + i * 7919, s.t) < p1, do: 1, else: 0)
          {s2, _} = step(:cartpole, s, a)
          {put_elem(st, i, s2), put_elem(ac, i, [{o, a} | elem(ac, i)])}
        end)

      go(net, Tuple.to_list(st), Tuple.to_list(ac), seed, w, greedy)
    end
  end

  # discounted returns, normalized over the batch (the baseline)
  defp batch(trajs, gamma) do
    rows =
      Enum.flat_map(trajs, fn tr ->
        {rets, _} = tr |> Enum.reverse() |> Enum.map_reduce(0.0, fn {o, a}, g -> g2 = 1.0 + gamma * g; {{o, a, g2}, g2} end)
        Enum.reverse(rets)
      end)

    gs = Enum.map(rows, &elem(&1, 2))
    mean = Enum.sum(gs) / length(gs)
    sd = :math.sqrt(Enum.reduce(gs, 0.0, fn g, s -> s + (g - mean) * (g - mean) end) / length(gs)) + 1.0e-8
    {Enum.map(rows, &elem(&1, 0)), Enum.map(rows, &elem(&1, 1)), Enum.map(gs, &((&1 - mean) / sd))}
  end


  # the policy-gradient step: seed ∂/∂z = (softmax z − onehot a)·Â / B over valid rows, AdamW
  defp pg_program(net, lr, b) do
    alias Vapor.Algebra.Term, as: T
    {din, dout} = {hd(net.padded), List.last(net.padded)}
    x = T.input(:x, :f32, [b, din])
    y = T.input(:y, :f32, [b, dout])
    adv = T.input(:adv, :f32, [b, 1])
    c1 = T.input(:c1, :f32, [1, 1])
    c2 = T.input(:c2, :f32, [1, 1])
    p = Map.new(net.params, fn {k, t} -> {k, T.input(k, :f32, t.shape)} end)

    z =
      Enum.reduce(0..2, x, fn l, h ->
        zl = T.add(T.linear(h, p[:"w#{l}"]), p[:"b#{l}"])
        if l == 2, do: zl, else: T.silu(zl)
      end)

    # padded logits pushed far below the real ones (exp underflows to 0)
    mask = T.const(Vapor.Tensor.from_list(:f32, [1, dout], for(i <- 0..(dout - 1), do: if(i < 2, do: 0.0, else: -1.0e4))))
    zm = T.add(z, mask)
    e = T.exp(T.sub(zm, T.reduce(:max, zm)))
    q = T.mul(e, T.rcp(T.reduce(:sum, e)))
    seed = T.mul(T.mul(T.sub(q, y), adv), T.splat(1.0 / b))
    keys = Enum.map(net.params, &elem(&1, 0))
    {:ok, grads} = Vapor.Autodiff.grad(z, seed, Enum.map(keys, &p[&1]))

    updates =
      Enum.zip(keys, grads)
      |> Enum.flat_map(fn {name, g} ->
        pv = p[name]
        {:input, _, _, shape} = pv
        m = T.input(:"#{name}_m", :f32, shape)
        v = T.input(:"#{name}_v", :f32, shape)
        m2 = T.add(T.mul(m, T.splat(0.9)), T.mul(g, T.splat(0.1)))
        v2 = T.add(T.mul(v, T.splat(0.999)), T.mul(T.mul(g, g), T.splat(0.001)))
        stp = T.mul(T.mul(m2, c1), T.rsqrt(T.add(T.mul(v2, c2), T.splat(1.0e-16))))
        [{:"#{name}_next", T.sub(pv, T.mul(stp, T.splat(lr)))}, {:"#{name}_m_next", m2}, {:"#{name}_v_next", v2}]
      end)

    state = Enum.flat_map(keys, fn n -> [{n, :"#{n}_next"}, {:"#{n}_m", :"#{n}_m_next"}, {:"#{n}_v", :"#{n}_v_next"}] end)
    Vapor.Program.new([probs: q] ++ updates, state: state)
  end

  # one update per chunk of 256 rows (the last chunk zero-padded: zero advantage → zero gradient)
  defp pg_update(comp, net, state, t0, xs, as, advs, w, b) do
    zeros = fn t -> Vapor.Tensor.new(:f32, t.shape, :binary.copy(<<0::32>>, Enum.product(t.shape))) end
    state = state || Map.new(Enum.flat_map(net.params, fn {k, t} -> [{k, t}, {:"#{k}_m", zeros.(t)}, {:"#{k}_v", zeros.(t)}] end))
    {din, dout} = {hd(net.padded), List.last(net.padded)}
    f32 = fn v -> <<Vapor.F32.from_float(v * 1.0)::32-little>> end

    rows = Enum.zip([xs, as, advs]) |> Enum.chunk_every(b)

    {state, t} =
      rows
      |> Enum.reduce({state, t0}, fn ch, {state, t} ->
        pad = b - length(ch)
        x = IO.iodata_to_binary([Enum.map(ch, fn {o, _, _} -> [Enum.map(o, f32), :binary.copy(<<0::32>>, din - 4)] end), :binary.copy(<<0::32>>, pad * din)])
        y = IO.iodata_to_binary([Enum.map(ch, fn {_, a, _} -> for(i <- 0..(dout - 1), do: f32.(if(i == a, do: 1.0, else: 0.0))) end), :binary.copy(<<0::32>>, pad * dout)])
        ad = IO.iodata_to_binary([Enum.map(ch, fn {_, _, g} -> f32.(g) end), :binary.copy(<<0::32>>, pad)])
        sched = Learn.schedule(t, 1, lr: 0.0)
        env = Map.merge(state, %{x: Vapor.Tensor.new(:f32, [b, din], x), y: Vapor.Tensor.new(:f32, [b, dout], y),
                                 adv: Vapor.Tensor.new(:f32, [b, 1], ad),
                                 c1: Vapor.Tensor.new(:f32, [1, 1], sched.c1.data), c2: Vapor.Tensor.new(:f32, [1, 1], sched.c2.data)})
        {:ok, r} = if w, do: Vapor.Runtime.Native.run(w, comp, env, isa: Vapor.Runtime.Substrates.host_isa(), mode: :native), else: Vapor.Runtime.Native.run_oracle(comp, env)
        {Map.new(state, fn {name, _} -> {name, r.outputs[:"#{name}_next"]} end), t + 1}
      end)

    {%{net | params: Enum.map(net.params, fn {k, _} -> {k, state[k]} end)}, state, t}
  end

  @doc "Save a policy network and its receipt (`model.safetensors`, `config.json`)."
  def save(%Learn{} = net, dir, info) do
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "model.safetensors"), Vapor.Ingest.Safetensors.encode(Learn.tensors(net)))
    cfg = Map.merge(%{"sizes" => net.sizes, "weights_sha256" => Learn.digest(net)}, info)
    File.write!(Path.join(dir, "config.json"), Vapor.JSON.encode(cfg))
    cfg
  end

  @doc "Load a shipped policy: `:cartpole` or `:reacher` (priv/rl), or a directory."
  def load(which) when which in [:cartpole, :reacher], do: load(Path.join([to_string(:code.priv_dir(:vapor)), "rl", Atom.to_string(which)]))

  def load(dir) do
    with {:ok, ts} <- Vapor.Ingest.Safetensors.read(Path.join(dir, "model.safetensors")),
         {:ok, cfg} <- File.read(Path.join(dir, "config.json")),
         {:ok, cfg} <- Vapor.JSON.decode(cfg) do
      {:ok, %{net: Learn.from_tensors(cfg["sizes"], ts), config: cfg}}
    else
      _ -> {:error, Vapor.Rejection.new({:rl, dir}, "a trained policy", "mix vapor.rl train")}
    end
  end

  @doc "Mean return of a CartPole policy over `n` episodes (greedy unless `greedy: false`)."
  def cartpole_score(net, n, seed, w, opts \\ []) do
    trajs = rollouts(net, n, seed, w, greedy: Keyword.get(opts, :greedy, true))
    (trajs |> Enum.map(&length/1) |> Enum.sum()) / n
  end

  # ------------------------------------------------------ behaviour cloning --

  @doc """
  Expert demonstrations on the reacher: `episodes` episodes, each a list of
  `{observation, action}` — the dataset of a behaviour-cloning run, hashed
  like any evidence (`dataset_digest/1`).
  """
  def demonstrations(episodes, seed) do
    for e <- 0..(episodes - 1) do
      Stream.unfold(reset(:reacher, seed + e), fn st ->
        if st.done, do: nil, else: (fn a -> {{observe(:reacher, st), a}, elem(step(:reacher, st, a, 0), 0)} end).(expert(st))
      end)
      |> Enum.to_list()
    end
  end

  def dataset_digest(eps), do: Vapor.Canonical.hex_digest(Enum.map(eps, fn ep -> Enum.map(ep, fn {o, a} -> [o, a] end) end))

  @doc "Behaviour cloning: an MLP (8 → 64 → 64 → 2) regressed on the expert's actions (`Vapor.Learn`)."
  def clone(demos, opts \\ []) do
    rows = List.flatten(demos)
    net = Learn.new([8, 64, 64, 2], Keyword.get(opts, :seed, 1))
    Learn.train(net, Enum.map(rows, &elem(&1, 0)), Enum.map(rows, &elem(&1, 1)), Keyword.get(opts, :steps, 3000),
                Keyword.merge([batch: 128, lr: 3.0e-3, lr_end: 0.05, chunk: 500], opts))
  end

  @doc "Success rate (goal within 0.02 in 100 steps) of a reacher policy over `n` goals; `policy` is `:expert`, `:random` or a `Vapor.Learn` net."
  def reacher_success(policy, n, seed, w \\ nil) do
    states = for i <- 0..(n - 1), do: reset(:reacher, seed + i)
    final = reach(policy, states, seed, w)
    Enum.count(final, fn st -> dist(effector(st.q), st.goal) < 0.02 end) / n
  end

  defp reach(policy, states, seed, w) do
    live = for {st, i} <- Enum.with_index(states), not st.done, do: i

    if live == [] do
      states
    else
      acts =
        case policy do
          :expert -> Enum.map(live, &expert(Enum.at(states, &1)))
          :random -> Enum.map(live, fn i -> t = Enum.at(states, i).t; [Sampler.uniform(seed + i, 2 * t) * 0.2 - 0.1, Sampler.uniform(seed + i, 2 * t + 1) * 0.2 - 0.1] end)
          %Learn{} = net -> Learn.predict(net, Enum.map(live, &observe(:reacher, Enum.at(states, &1))), worker: w, rows: 16 * div(length(live) + 15, 16))
        end

      st = Enum.zip(live, acts) |> Enum.reduce(List.to_tuple(states), fn {i, a}, t -> put_elem(t, i, elem(step(:reacher, elem(t, i), a, 0), 0)) end)
      reach(policy, Tuple.to_list(st), seed, w)
    end
  end

  # -------------------------------------------------------------- render --

  @doc "A frame of an environment's state (`size` × `size` pixels)."
  def render(env, st, size \\ 160)

  def render(:cartpole, %{s: [x, _, th, _]}, size) do
    {w, h} = {size, div(size * 2, 3)}
    scale = w / 4.8
    cx = w / 2 + x * scale
    cy = h * 0.72
    tip = {cx + CR.sin_f64(th) * scale * 1.0, cy - CR.cos_f64(th) * scale * 1.0}
    canvas(w, h, {0.96, 0.97, 0.98}, fn px, py ->
      cond do
        abs(py - cy) <= h * 0.06 and abs(px - cx) <= scale * 0.25 -> {0.16, 0.2, 0.24}
        seg_dist({px, py}, {cx, cy}, tip) <= max(2.0, size / 80) -> {0.8, 0.45, 0.25}
        abs(py - (cy + h * 0.06)) < 1.0 -> {0.55, 0.6, 0.62}
        true -> nil
      end
    end)
  end

  def render(:frozenlake, %{s: s}, size) do
    cell = size / 4
    canvas(size, size, {0.9, 0.95, 1.0}, fn px, py ->
      {c, r} = {trunc(px / cell), trunc(py / cell)}
      tile = lake_tile(r * 4 + c)
      {ix, iy} = {px - (c + 0.5) * cell, py - (r + 0.5) * cell}
      cond do
        r * 4 + c == s and ix * ix + iy * iy < (cell * 0.3) * (cell * 0.3) -> {0.85, 0.3, 0.2}
        tile == ?H -> {0.15, 0.25, 0.4}
        tile == ?G -> {0.35, 0.7, 0.35}
        rem(trunc(px), trunc(cell)) == 0 or rem(trunc(py), trunc(cell)) == 0 -> {0.7, 0.8, 0.9}
        true -> nil
      end
    end)
  end

  def render(:reacher, %{q: [q1, _] = q, goal: {gx, gy}}, size) do
    to_px = fn {x, y} -> {size / 2 + x * size * 0.45, size / 2 - y * size * 0.45} end
    base = to_px.({0.0, 0.0})
    elbow = to_px.({0.5 * CR.cos_f64(q1), 0.5 * CR.sin_f64(q1)})
    hand = to_px.(effector(q))
    goal = to_px.({gx, gy})
    canvas(size, size, {0.97, 0.97, 0.95}, fn px, py ->
      cond do
        seg_dist({px, py}, goal, goal) <= size / 40 -> {0.3, 0.65, 0.35}
        seg_dist({px, py}, base, elbow) <= size / 60 -> {0.2, 0.3, 0.45}
        seg_dist({px, py}, elbow, hand) <= size / 70 -> {0.3, 0.45, 0.65}
        seg_dist({px, py}, hand, hand) <= size / 45 -> {0.85, 0.4, 0.2}
        true -> nil
      end
    end)
  end

  defp canvas(w, h, bg, f) do
    vals = for y <- 0..(h - 1), x <- 0..(w - 1), c = f.(x + 0.5, y + 0.5) || bg, k <- 0..2, do: elem(c, k)
    %Image{w: w, h: h, c: 3, px: List.to_tuple(vals)}
  end

  defp seg_dist({px, py}, {ax, ay}, {bx, by}) do
    {dx, dy} = {bx - ax, by - ay}
    l2 = dx * dx + dy * dy
    t = if l2 == 0.0, do: 0.0, else: min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    {qx, qy} = {ax + t * dx, ay + t * dy}
    :math.sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy))
  end

  @doc "Replay an episode as a video: environment, seed, and the policy's actions are re-simulated from scratch."
  def replay(env, seed, actions, opts \\ []) do
    {states, _} =
      Enum.map_reduce(actions, reset(env, seed), fn a, st -> {st, if(st.done, do: st, else: elem(step(env, st, a, seed), 0))} end)

    frames = Enum.map(states, &render(env, &1, Keyword.get(opts, :size, 160)))
    %Video{fps: Keyword.get(opts, :fps, 25) * 1.0, frames: frames}
  end

end
