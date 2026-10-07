defmodule Vapor.Physics do
  @moduledoc """
  **A physics engine whose trajectories are facts** — for reinforcement
  learning and digital twins.

  Simulators disagree with themselves: the same model, seed and actions
  give another trajectory on another GPU, thread count or library version,
  because floating-point sums are reordered. For a smooth system that is a
  nuisance; for a chaotic one (a double pendulum, a walking robot, a
  turbulent flow) it is total — two runs part ways within seconds, so an
  RL result cannot be replayed and a digital twin cannot be audited
  against its plant. Here a world's step is an ordinary vapor program, so
  it runs under canonical semantics: **the same bits on every substrate**
  (the oracle, the native workers, the GPU daemons), for any batch of
  worlds.

  The method is extended position-based dynamics (XPBD, Macklin, Müller &
  Chentanez 2016) with many small substeps and one constraint pass each
  (Müller et al. 2020, *Detailed rigid body simulation with extended
  position based dynamics*): particles with inverse masses, rods (distance
  constraints with a compliance), rails (a coordinate held fixed — a cart
  on its track), a floor, and actuators (forces from the action vector).
  The constraint graph is a signed incidence matrix `D` (rod c: +1 at i,
  −1 at j), so one Jacobi pass over every rod of every world is two
  matrix products:

      d = p·Dᵀ                       rod vectors, every world at once
      λ = −(|d| − ℓ) / (wᵢ + wⱼ + α/h²)
      Δp = ((λ/|d|)·d)·D ∘ w ∘ 1/deg  each particle's share

  — `linear` operations the airlock already certifies. Worlds are rows
  (`f32[B, N]` per axis): a thousand cart-poles are one program run.

  Because the step is a program of differentiable operators (including,
  piecewise, the floor's `max`), `Vapor.Autodiff` differentiates whole
  trajectories: `sysid/4` fits a twin's unknown parameters (rod lengths,
  damping) to a plant's measured motion by gradient descent through the
  simulator.

  Measured (`test/vapor/physics_test.exs`): a pendulum's period against
  the exact elliptic-integral period, converging as the substep shrinks; a
  double pendulum bit-identical on oracle and native worker while a
  one-ulp perturbation of its start diverges; trajectory gradients against
  finite differences; parameters recovered from noisy observations (the
  control: time-shuffled observations recover nothing).
  """
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Runtime.{Native, Session, Substrates}

  defstruct axes: [:x, :y], n: 0, m: 0, np: 16, mp: 16, pos0: [], w: [], mobile: [], rods: [], rest: [], rails: [], ground: [],
            actuators: [], gravity: -9.81, dt: 0.02, substeps: 8, damping: 0.0, compliance: 0.0

  @type t :: %__MODULE__{}

  @doc """
  A world. Options:

    * `particles` — `[{position, mass}]`, a position a list (one value per
      axis), a mass a positive number or `:fixed` (an anchor);
    * `rods` — `[{i, j}]` (rest length: the initial distance) or `[{i, j, length}]`;
    * `rails` — `[{i, axis, value}]`: particle i's coordinate on `axis` held at `value`;
    * `ground` — particle indices that collide with the floor `y ≥ 0`;
    * `actuators` — `[{i, axis}]`: action component k pushes particle i along `axis`;
    * `axes` (`[:x, :y]`; `[:x, :y, :z]` in 3-D), `gravity` (−9.81, on `y`),
      `dt` (0.02 s per step), `substeps` (8), `damping` (0, per second),
      `compliance` (0: rigid rods).
  """
  def new(opts) do
    parts = Keyword.fetch!(opts, :particles)
    axes = Keyword.get(opts, :axes, [:x, :y])
    n = length(parts)
    pos0 = Enum.map(parts, fn {p, _} -> Enum.map(p, &(&1 * 1.0)) end)
    w = Enum.map(parts, fn {_, m} -> if m == :fixed, do: 0.0, else: 1.0 / m end)

    rods =
      Keyword.get(opts, :rods, [])
      |> Enum.map(fn
        {i, j} -> {i, j, dist(Enum.at(pos0, i), Enum.at(pos0, j))}
        {i, j, l} -> {i, j, l * 1.0}
      end)

    %__MODULE__{axes: axes, n: n, m: length(rods), np: pad16(n), mp: pad16(max(length(rods), 1)), pos0: pos0, w: w,
                mobile: Enum.map(w, &(&1 > 0)), rods: Enum.map(rods, fn {i, j, _} -> {i, j} end), rest: Enum.map(rods, &elem(&1, 2)),
                rails: Keyword.get(opts, :rails, []), ground: Keyword.get(opts, :ground, []), actuators: Keyword.get(opts, :actuators, []),
                gravity: Keyword.get(opts, :gravity, -9.81) * 1.0, dt: Keyword.get(opts, :dt, 0.02) * 1.0,
                substeps: Keyword.get(opts, :substeps, 8), damping: Keyword.get(opts, :damping, 0.0) * 1.0,
                compliance: Keyword.get(opts, :compliance, 0.0) * 1.0}
  end

  defp pad16(k), do: max(16, div(k + 15, 16) * 16)
  defp dist(a, b), do: :math.sqrt(Enum.zip_with(a, b, fn u, v -> (u - v) * (u - v) end) |> Enum.sum())

  # ------------------------------------------------------------- worlds --

  @doc "A simple pendulum: an anchor at the origin, a bob at angle `theta` (from the downward vertical), rod `length`."
  def pendulum(opts \\ []) do
    {l, th} = {Keyword.get(opts, :length, 1.0), Keyword.get(opts, :theta, 0.5)}
    new([particles: [{[0.0, 0.0], :fixed}, {[l * :math.sin(th), -l * :math.cos(th)], 1.0}], rods: [{0, 1}]] ++ Keyword.drop(opts, [:length, :theta]))
  end

  @doc "A double pendulum (two unit masses, two rods of `length`), released from angles `theta1`, `theta2`."
  def double_pendulum(opts \\ []) do
    {l, t1, t2} = {Keyword.get(opts, :length, 1.0), Keyword.get(opts, :theta1, 2.0), Keyword.get(opts, :theta2, 2.5)}
    p1 = [l * :math.sin(t1), -l * :math.cos(t1)]
    p2 = [hd(p1) + l * :math.sin(t2), List.last(p1) - l * :math.cos(t2)]
    new([particles: [{[0.0, 0.0], :fixed}, {p1, 1.0}, {p2, 1.0}], rods: [{0, 1}, {1, 2}]] ++ Keyword.drop(opts, [:length, :theta1, :theta2]))
  end

  @doc """
  A cart-pole: a cart (mass 1) on a rail at `y = 0`, a pole of `length` 1
  carrying a point mass 0.1 at its tip, the cart pushed by action 0.
  """
  def cartpole(opts \\ []) do
    {l, th} = {Keyword.get(opts, :length, 1.0), Keyword.get(opts, :theta, 0.0)}
    new([particles: [{[0.0, 0.0], 1.0}, {[l * :math.sin(th), l * :math.cos(th)], 0.1}], rods: [{0, 1}], rails: [{0, :y, 0.0}],
         actuators: [{0, :x}]] ++ Keyword.drop(opts, [:length, :theta]))
  end

  @doc "A chain of `links` rods (`length` each) held horizontally from an anchor at `height`, released; its links collide with the floor."
  def chain(opts \\ []) do
    k = Keyword.get(opts, :links, 6)
    l = Keyword.get(opts, :length, 0.25)
    y0 = Keyword.get(opts, :height, 2.0)
    parts = for i <- 0..k, do: {[i * l, y0], if(i == 0, do: :fixed, else: 0.2)}
    new([particles: parts, rods: for(i <- 0..(k - 1), do: {i, i + 1}), ground: Enum.to_list(1..k)] ++ Keyword.drop(opts, [:links, :length, :height]))
  end

  # ------------------------------------------------------------- program --

  @doc """
  The program of one control step (`substeps` XPBD substeps) for `batch`
  worlds. Inputs: per axis `p_<a>`, `v_<a>` (`f32[B, N]`, N padded to 16),
  `u` (`f32[B, 16]`, the actions), and the parameters `w` (`f32[1, N]`),
  `rest` (`f32[1, M]`), `damping` and `gravity` (`f32[1, 1]`). Outputs
  `p_<a>_next`, `v_<a>_next` (the session's state). Option `steps` (1):
  control steps per run, unrolled (the actions held).
  """
  def program(%__MODULE__{} = wd, batch, opts \\ []) do
    steps = Keyword.get(opts, :steps, 1)
    ins = inputs(wd, batch)
    {lets, state} = Enum.reduce(1..steps, {[], ins.state}, fn k, {lets, st} -> control_step(wd, ins, st, lets, "s#{k}") end)
    outs = for a <- wd.axes, {kind, t} <- [{"p", state[{:p, a}]}, {"v", state[{:v, a}]}], do: {:"#{kind}_#{a}_next", t}
    Program.new(outs, lets: Enum.reverse(lets), state: for(a <- wd.axes, k <- ["p", "v"], do: {:"#{k}_#{a}", :"#{k}_#{a}_next"}))
  end

  defp inputs(wd, b) do
    state = for a <- wd.axes, k <- [:p, :v], into: %{}, do: {{k, a}, T.input(:"#{k}_#{a}", :f32, [b, wd.np])}

    %{state: state, u: T.input(:u, :f32, [b, 16]), w: T.input(:w, :f32, [1, wd.np]), rest: T.input(:rest, :f32, [1, wd.mp]),
      damping: T.input(:damping, :f32, [1, 1]), gravity: T.input(:gravity, :f32, [1, 1])}
  end

  defp row(vals, n, pad), do: T.const(Tensor.from_list(:f32, [1, n], vals ++ List.duplicate(pad, n - length(vals))))

  defp mat(rows, cols, f), do: T.const(Tensor.from_list(:f32, [rows, cols], for(r <- 0..(rows - 1), c <- 0..(cols - 1), do: f.(r, c) * 1.0)))

  defp control_step(wd, ins, st, lets, tag) do
    {np, mp} = {wd.np, wd.mp}
    h = wd.dt / wd.substeps
    rods = List.to_tuple(wd.rods)
    sign = fn c, i -> (if c < wd.m, do: (case elem(rods, c) do {^i, _} -> 1.0; {_, ^i} -> -1.0; _ -> 0.0 end), else: 0.0) end
    d = mat(mp, np, sign)
    dt = mat(np, mp, fn i, c -> sign.(c, i) end)
    dabs = mat(mp, np, fn c, i -> abs(sign.(c, i)) end)
    deg = for i <- 0..(wd.n - 1), do: Enum.count(wd.rods, fn {a, b} -> a == i or b == i end)
    relax = row(Enum.map(deg, &(1.0 / max(&1, 1))), np, 0.0)
    mobile = row(Enum.map(wd.mobile, &if(&1, do: 1.0, else: 0.0)), np, 0.0)
    padrod = row(List.duplicate(0.0, wd.m), mp, 1.0)
    acts = List.to_tuple(wd.actuators)

    bind = fn {lets, name}, term -> {T.ref(name, term), [{name, term} | lets]} end

    # forces of the actions, per axis (held through the substeps)
    {force, lets} =
      Enum.reduce(wd.axes, {%{}, lets}, fn a, {f, lets} ->
        m = mat(np, 16, fn i, k -> (if k < tuple_size(acts) and elem(acts, k) == {i, a}, do: 1.0, else: 0.0) end)
        {r, lets} = bind.({lets, :"#{tag}.f_#{a}"}, T.mul(T.linear(ins.u, m), ins.w))
        {Map.put(f, a, r), lets}
      end)

    wsum = T.linear(ins.w, dabs)
    denom = T.add(T.add(wsum, T.splat(wd.compliance / (h * h))), padrod)
    keep = T.sub(T.splat(1.0), T.mul(ins.damping, T.splat(h)))

    Enum.reduce(1..wd.substeps, {lets, st}, fn s, {lets, st} ->
      pre = "#{tag}.#{s}"

      # integrate: v += h·(F·w + g), p += h·v
      {vs, lets} =
        Enum.reduce(wd.axes, {%{}, lets}, fn a, {acc, lets} ->
          g = if a == :y, do: T.mul(ins.gravity, mobile), else: nil
          dv = if g, do: T.add(force[a], g), else: force[a]
          v = T.mul(T.add(st[{:v, a}], T.mul(dv, T.splat(h))), keep)
          {r, lets} = bind.({lets, :"#{pre}.v_#{a}"}, v)
          {Map.put(acc, a, r), lets}
        end)

      {ps, lets} =
        Enum.reduce(wd.axes, {%{}, lets}, fn a, {acc, lets} ->
          {r, lets} = bind.({lets, :"#{pre}.q_#{a}"}, T.add(st[{:p, a}], T.mul(vs[a], T.splat(h))))
          {Map.put(acc, a, r), lets}
        end)

      # one Jacobi pass over every rod of every world
      {ds, lets} =
        Enum.reduce(wd.axes, {%{}, lets}, fn a, {acc, lets} ->
          {r, lets} = bind.({lets, :"#{pre}.d_#{a}"}, T.linear(ps[a], d))
          {Map.put(acc, a, r), lets}
        end)

      l2 = wd.axes |> Enum.map(&T.mul(ds[&1], ds[&1])) |> Enum.reduce(&T.add(&2, &1)) |> T.add(padrod)
      {inv, lets} = bind.({lets, :"#{pre}.inv"}, T.rsqrt(l2))
      c = T.sub(T.mul(l2, inv), ins.rest)
      {s_, lets} = bind.({lets, :"#{pre}.s"}, T.mul(T.neg(T.divide(c, denom)), inv))

      # positions, and velocities as v + (corrections)/h — never as
      # (p − p_before)/h, whose cancellation in single precision costs
      # three digits once h is small (found in 0.10: the pendulum's period
      # got *worse* past 8 substeps)
      {lets, st2} =
        Enum.reduce(wd.axes, {lets, %{}}, fn a, {lets, acc} ->
          {dp, lets} = bind.({lets, :"#{pre}.dp_#{a}"}, T.mul(T.mul(T.linear(T.mul(s_, ds[a]), dt), ins.w), relax))
          {pc, lets} = bind.({lets, :"#{pre}.pc_#{a}"}, T.add(ps[a], dp))
          pf = pc |> then(&rails(wd, a, &1, np))
          pf = if a == :y and wd.ground != [], do: T.max(pf, row(for(i <- 0..(wd.n - 1), do: if(i in wd.ground, do: 0.0, else: -1.0e30)), np, -1.0e30)), else: pf
          {p, lets} = bind.({lets, :"#{pre}.p_#{a}"}, pf)
          v = T.add(vs[a], T.mul(T.add(dp, T.sub(p, pc)), T.splat(1.0 / h)))
          v = case railmask(wd, a, np) do nil -> v; m -> T.mul(v, T.sub(T.splat(1.0), m)) end
          {r, lets} = bind.({lets, :"#{pre}.w_#{a}"}, v)
          {lets, acc |> Map.put({:p, a}, p) |> Map.put({:v, a}, r)}
        end)

      {lets, st2}
    end)
  end

  defp railmask(wd, a, np) do
    case for({i, ^a, _} <- wd.rails, do: i) do
      [] -> nil
      is -> row(for(i <- 0..(wd.n - 1), do: if(i in is, do: 1.0, else: 0.0)), np, 0.0)
    end
  end

  defp rails(wd, a, p, np) do
    case for({i, ^a, v} <- wd.rails, do: {i, v * 1.0}) do
      [] -> p
      rs ->
        mask = row(for(i <- 0..(wd.n - 1), do: if(List.keymember?(rs, i, 0), do: 1.0, else: 0.0)), np, 0.0)
        val = row(for(i <- 0..(wd.n - 1), do: (case List.keyfind(rs, i, 0) do {_, v} -> v; nil -> 0.0 end)), np, 0.0)
        T.add(T.mul(p, T.sub(T.splat(1.0), mask)), T.mul(val, mask))
    end
  end

  # --------------------------------------------------------------- state --

  @doc """
  The initial state of `batch` worlds: `%{{:p | :v, axis} => [[value]]}`
  (one row per world). Option `perturb: fn world, particle, axis -> delta end`.
  """
  def state(%__MODULE__{} = wd, batch, opts \\ []) do
    perturb = Keyword.get(opts, :perturb, fn _, _, _ -> 0.0 end)

    for {a, k} <- Enum.with_index(wd.axes), kind <- [:p, :v], into: %{} do
      rows =
        for b <- 0..(batch - 1) do
          vals = for {p, i} <- Enum.with_index(wd.pos0), do: if(kind == :p, do: Enum.at(p, k) + perturb.(b, i, a), else: 0.0)
          vals ++ List.duplicate(0.0, wd.np - wd.n)
        end

      {{kind, a}, rows}
    end
  end

  @doc "The parameters as program inputs (overrides: `rest`, `damping`, `gravity`, `w`)."
  def params(%__MODULE__{} = wd, over \\ []) do
    pad = fn vals, n, v -> vals ++ List.duplicate(v, n - length(vals)) end

    %{w: Tensor.from_list(:f32, [1, wd.np], pad.(Keyword.get(over, :w, wd.w), wd.np, 0.0)),
      rest: Tensor.from_list(:f32, [1, wd.mp], pad.(Keyword.get(over, :rest, wd.rest), wd.mp, 1.0)),
      damping: Tensor.from_list(:f32, [1, 1], [Keyword.get(over, :damping, wd.damping) * 1.0]),
      gravity: Tensor.from_list(:f32, [1, 1], [Keyword.get(over, :gravity, wd.gravity) * 1.0])}
  end

  defp tensors(st, batch, np), do: Map.new(st, fn {{k, a}, rows} -> {:"#{k}_#{a}", Tensor.from_list(:f32, [batch, np], List.flatten(rows))} end)

  defp actions(nil, batch), do: Tensor.from_list(:f32, [batch, 16], List.duplicate(0.0, batch * 16))
  defp actions(us, batch), do: Tensor.from_list(:f32, [batch, 16], Enum.flat_map(us, fn u -> u ++ List.duplicate(0.0, 16 - length(u)) end) |> Enum.take(batch * 16))

  # ---------------------------------------------------------- simulating --

  defmodule Sim do
    @moduledoc "A running batch of worlds: the compiled step, and a session on a native worker (or the oracle)."
    defstruct [:world, :batch, :comp, :session, :state, :params, t: 0]
  end

  @doc """
  Start `batch` worlds: `%Sim{}`. Options: `worker` (a native worker: the
  state then lives in its session; default the oracle), `state` (from
  `state/3`), `params` (overrides for `params/2`), `steps` (control steps
  per call, unrolled in the program).
  """
  def start(%__MODULE__{} = wd, batch, opts \\ []) do
    {:ok, comp} = Vapor.Compile.Lower.lower(program(wd, batch, Keyword.take(opts, [:steps])))
    st = Keyword.get_lazy(opts, :state, fn -> state(wd, batch) end)
    params = params(wd, Keyword.get(opts, :params, []))

    session =
      case opts[:worker] do
        nil -> nil
        w -> elem(Session.open(w, comp, isa: Substrates.host_isa()), 1)
      end

    %Sim{world: wd, batch: batch, comp: comp, session: session, state: tensors(st, batch, wd.np), params: params}
  end

  @doc "Advance one call (`steps` control steps) with actions `us` (`[[force]]` per world, or nil). Returns `{sim, %{p_x: rows, …}}`."
  def advance(%Sim{} = sim, us \\ nil) do
    wd = sim.world
    u = actions(us, sim.batch)
    want = for a <- wd.axes, k <- ["p", "v"], do: :"#{k}_#{a}_next"

    out =
      case sim.session do
        nil ->
          {:ok, r} = Native.run_oracle(sim.comp, sim.state |> Map.merge(sim.params) |> Map.put(:u, u))
          r.outputs

        s ->
          ins = if sim.t == 0, do: sim.state |> Map.merge(sim.params) |> Map.put(:u, u), else: %{u: u}
          {:ok, o, _} = Session.step(s, ins, want)
          o
      end

    next = Map.new(wd.axes |> Enum.flat_map(fn a -> [{:"p_#{a}", out[:"p_#{a}_next"]}, {:"v_#{a}", out[:"v_#{a}_next"]}] end))
    {%{sim | state: next, t: sim.t + 1}, next}
  end

  @doc "Rows of a state tensor: `[[float]]` (world × particle, padding dropped)."
  def rows(%Sim{world: wd}, %Tensor{} = t), do: t |> Tensor.to_floats() |> Enum.chunk_every(wd.np) |> Enum.map(&Enum.take(&1, wd.n))

  @doc "Close the session."
  def stop(%Sim{session: nil}), do: :ok
  def stop(%Sim{session: s}), do: Session.close(s)

  @doc "Total energy (kinetic + gravitational) of each world of a state."
  def energy(%Sim{world: wd} = sim, state) do
    ys = rows(sim, state.p_y)
    vel = for a <- wd.axes, do: rows(sim, state[:"v_#{a}"])

    for b <- 0..(sim.batch - 1) do
      Enum.sum(
        for i <- 0..(wd.n - 1), w = Enum.at(wd.w, i), w > 0 do
          m = 1.0 / w
          v2 = vel |> Enum.map(&(Enum.at(Enum.at(&1, b), i) ** 2)) |> Enum.sum()
          0.5 * m * v2 - m * wd.gravity * Enum.at(Enum.at(ys, b), i)
        end
      )
    end
  end

  # ------------------------------------------------------ exact references --

  @doc """
  The exact period of a pendulum of `length` released at rest from
  amplitude `theta0` under gravity `g`: `4·√(L/g)·K(sin(θ₀/2))`, the
  complete elliptic integral by the arithmetic–geometric mean.
  """
  def pendulum_period(length, theta0, g \\ 9.81) do
    k = :math.sin(theta0 / 2)
    4 * :math.sqrt(length / g) * (:math.pi() / (2 * agm(1.0, :math.sqrt(1 - k * k))))
  end

  defp agm(a, b) when abs(a - b) < 1.0e-15, do: a
  defp agm(a, b), do: agm((a + b) / 2, :math.sqrt(a * b))

  # ------------------------------------------------- system identification --

  @doc """
  The program of a twin's fit: `horizon` control steps from a known start
  (inputs `p0_<a>`, `v0_<a>`, one world), the squared distance of every
  mobile particle to its observed positions (`o<t>_<a>`, `f32[1, N]`)
  summed over the horizon as `loss`, and its gradient with respect to the
  parameters `rest` and `damping` (`grad_rest`, `grad_damping`).
  """
  def sysid_program(%__MODULE__{} = wd, horizon) do
    ins = inputs(wd, 1)
    st0 = for a <- wd.axes, k <- [:p, :v], into: %{}, do: {{k, a}, T.input(:"#{k}0_#{a}", :f32, [1, wd.np])}
    mobile = row(Enum.map(wd.mobile, &if(&1, do: 1.0, else: 0.0)), wd.np, 0.0)

    {lets, _st, terms} =
      Enum.reduce(1..horizon, {[], st0, []}, fn t, {lets, st, terms} ->
        {lets, st} = control_step(wd, ins, st, lets, "t#{t}")

        sq =
          for a <- wd.axes do
            e = T.mul(T.sub(st[{:p, a}], T.input(:"o#{t}_#{a}", :f32, [1, wd.np])), mobile)
            T.reduce(:sum, T.mul(e, e))
          end

        {lets, st, terms ++ sq}
      end)

    loss_term = Enum.reduce(terms, &T.add(&2, &1))
    lets = Enum.reverse(lets) ++ [{:loss, loss_term}]
    loss = T.ref(:loss, loss_term)
    {:ok, [g_rest, g_damp], all} = Vapor.Autodiff.grad_lets(lets, loss, T.add(T.mul(loss, T.splat(0.0)), T.splat(1.0)), [ins.rest, ins.damping])
    Program.new([loss: loss, grad_rest: g_rest, grad_damping: g_damp], lets: all)
  end

  @doc """
  Fit a twin's rod lengths and damping to a plant's observed positions
  (`obs`: `[%{x: [...], y: [...]}]` per control step, one value per
  particle) from a known start, by Adam on the gradient through the
  simulator. Options: `iters` (150), `lr` (0.01), `worker`, `rest` and
  `damping` (the initial guesses). Returns `%{rest, damping, loss: [..]}`.
  """
  def sysid(%__MODULE__{} = wd, obs, opts \\ []) do
    horizon = length(obs)
    {:ok, comp} = Vapor.Compile.Lower.lower(sysid_program(wd, horizon))
    run = runner(comp, opts[:worker])
    st0 = tensors(state(wd, 1), 1, wd.np) |> Map.new(fn {k, v} -> {String.to_atom(String.replace(to_string(k), "_", "0_", global: false)), v} end)
    pad = fn vals, v -> vals ++ List.duplicate(v, wd.np - length(vals)) end

    ob =
      for {o, t} <- Enum.with_index(obs, 1), a <- wd.axes, into: %{} do
        {:"o#{t}_#{a}", Tensor.from_list(:f32, [1, wd.np], pad.(Map.fetch!(o, a), 0.0))}
      end

    x0 = Keyword.get(opts, :rest, wd.rest) ++ [Keyword.get(opts, :damping, wd.damping)]
    lr = Keyword.get(opts, :lr, 0.01)

    {x, _m, _v, losses} =
      Enum.reduce(1..Keyword.get(opts, :iters, 150), {x0, nil, nil, []}, fn k, {x, m, v, losses} ->
        {rest, [damp]} = Enum.split(x, wd.m)
        env = st0 |> Map.merge(ob) |> Map.merge(params(wd, rest: rest, damping: damp)) |> Map.put(:u, actions(nil, 1))
        out = run.(env)
        g = Enum.take(Tensor.to_floats(out.grad_rest), wd.m) ++ Tensor.to_floats(out.grad_damping)
        {x, m, v} = adam(x, g, m, v, k, lr)
        {x, m, v, [hd(Tensor.to_floats(out.loss)) | losses]}
      end)

    {rest, [damp]} = Enum.split(x, wd.m)
    %{rest: rest, damping: damp, loss: Enum.reverse(losses)}
  end

  defp runner(comp, nil), do: fn env -> elem(Native.run_oracle(comp, env), 1).outputs end
  defp runner(comp, w), do: fn env -> elem(Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native), 1).outputs end

  # Adam in binary64 on the host (deterministic: IEEE operations only)
  defp adam(x, g, m, v, k, lr) do
    {b1, b2, eps} = {0.9, 0.999, 1.0e-8}
    m = Enum.zip_with(m || Enum.map(x, fn _ -> 0.0 end), g, &(b1 * &1 + (1 - b1) * &2))
    v = Enum.zip_with(v || Enum.map(x, fn _ -> 0.0 end), g, &(b2 * &1 + (1 - b2) * &2 * &2))
    {c1, c2} = {1 - :math.pow(b1, k), 1 - :math.pow(b2, k)}
    x = Enum.zip_with([x, m, v], fn [xi, mi, vi] -> xi - lr * (mi / c1) / (:math.sqrt(vi / c2) + eps) end)
    {x, m, v}
  end

  # ------------------------------------------------- reinforcement learning --

  @doc """
  Cart-pole episodes for a batch of linear policies, in lockstep — one
  world per policy, one program run per control step for all of them.
  `policies` are `[[k_x, k_v, k_θ, k_ω]]`: force = clamp(k·obs, ±10) with
  obs = (cart x, cart velocity, sin θ, θ̇). A world earns 1 per step while
  |x| < 2.4 and |θ| < 12° (gymnasium's thresholds). Returns the returns.
  Options: `horizon` (200), `seed` (initial tilts, ±0.05 rad), `worker`.
  """
  def cartpole_returns(policies, opts \\ []) do
    b = length(policies)
    horizon = Keyword.get(opts, :horizon, 200)
    seed = Keyword.get(opts, :seed, 1)
    wd = cartpole(dt: 0.02, substeps: 4)
    tilt = fn wld -> (Vapor.Sampler.uniform(seed, wld) - 0.5) * 0.1 end
    st = state(wd, b, perturb: fn wld, i, a -> if i == 1 and a == :x, do: :math.sin(tilt.(wld)), else: 0.0 end)
    sim = start(wd, b, worker: opts[:worker], state: st)
    lim = :math.sin(12 * :math.pi() / 180)

    {sim, _obs, alive, ret} =
      Enum.reduce(1..horizon, {sim, observe(sim, sim.state), List.duplicate(true, b), List.duplicate(0, b)}, fn _, {sim, obs, alive, ret} ->
        us = Enum.zip_with(policies, obs, fn k, o -> [k |> Enum.zip_with(o, &(&1 * &2)) |> Enum.sum() |> max(-10.0) |> min(10.0)] end)
        {sim, next} = advance(sim, us)
        obs = observe(sim, next)
        alive = Enum.zip_with(alive, obs, fn a, [x, _, s, _] -> a and abs(x) < 2.4 and abs(s) < lim end)
        {sim, obs, alive, Enum.zip_with(ret, alive, fn r, a -> if a, do: r + 1, else: r end)}
      end)

    stop(sim)
    _ = alive
    ret
  end

  defp observe(sim, st) do
    xs = rows(sim, st.p_x)
    ys = rows(sim, st.p_y)
    vs = rows(sim, st.v_x)

    Enum.zip_with([xs, ys, vs], fn [[xc, xt | _], [_, yt | _], [vc, vt | _]] ->
      l = :math.sqrt((xt - xc) * (xt - xc) + yt * yt)
      [xc, vc, (xt - xc) / l, (vt - vc) / l]
    end)
  end

  @doc """
  Augmented random search (Mania, Guy & Recht 2018) for the cart-pole: a
  linear policy, `dirs` random sign directions per iteration, each
  evaluated as +σ and −σ in one batch of worlds, the step scaled by the
  returns' standard deviation. The directions are counter-based
  (`Vapor.Sampler`), so a run is a function of its seed. Returns `{policy,
  [mean return of ± pairs per iteration]}`.
  """
  def cartpole_ars(opts \\ []) do
    {iters, dirs, sigma, lr, seed} = {Keyword.get(opts, :iters, 25), Keyword.get(opts, :dirs, 8), Keyword.get(opts, :sigma, 1.0), Keyword.get(opts, :lr, 1.0), Keyword.get(opts, :seed, 1)}

    Enum.reduce(1..iters, {[0.0, 0.0, 0.0, 0.0], []}, fn it, {k, curve} ->
      ds = for d <- 1..dirs, do: for(j <- 0..3, do: if(Vapor.Sampler.uniform(seed * 1000 + it, d * 4 + j) < 0.5, do: -1.0, else: 1.0))
      pols = Enum.flat_map(ds, fn d -> [Enum.zip_with(k, d, &(&1 + sigma * &2)), Enum.zip_with(k, d, &(&1 - sigma * &2))] end)
      rets = cartpole_returns(pols, Keyword.merge(opts, seed: seed * 7919 + it))
      pairs = Enum.chunk_every(rets, 2)
      mean = Enum.sum(rets) / length(rets)
      sd = :math.sqrt(Enum.sum(Enum.map(rets, &((&1 - mean) ** 2))) / length(rets))
      step = Enum.zip_with(pairs, ds, fn [rp, rm], d -> Enum.map(d, &(&1 * (rp - rm))) end) |> Enum.zip_with(&Enum.sum/1)
      k = Enum.zip_with(k, step, fn ki, si -> ki + lr / (dirs * max(sd, 1.0e-6)) * si end)
      {k, [mean | curve]}
    end)
    |> then(fn {k, curve} -> {k, Enum.reverse(curve)} end)
  end

  # ---------------------------------------------------------- digital twin --

  defmodule Twin do
    @moduledoc """
    A digital twin: the model run beside the plant, step for step, with a
    tamper-evident ledger of what it predicted.

    Each `observe/3` advances the model with the plant's actions, compares
    its predicted positions with the measured ones (residual in units of
    the sensor noise σ) and runs a CUSUM on the residuals: the alarm is
    raised when the accumulated excess over `k` passes `h` (Page 1954) —
    a slow drift (a rod stretching, a bearing wearing) adds up where a
    single-step threshold would miss it. Every step is appended to a hash
    chain of `(t, actions, prediction digest, measurement digest, residual)`:
    since the simulation is bit-exact, anyone holding the model and the
    actions can `replay/1` the predictions and check the chain — the twin's
    record of "what we expected at t" cannot be rewritten after the fact.
    """
    defstruct [:sim, :sigma, :k, :h, :start, cusum: 0.0, t: 0, alarm: nil, ledger: [], head: "", actions: []]
  end

  @doc "A twin of `world` (one instance). Options: `sigma` (sensor noise, 0.002), `k` (0.5), `h` (8.0), `worker`, `params`."
  def twin(%__MODULE__{} = wd, opts \\ []) do
    sim = start(wd, 1, Keyword.take(opts, [:worker, :params, :state]))
    %Twin{sim: sim, start: sim.state, sigma: Keyword.get(opts, :sigma, 0.002), k: Keyword.get(opts, :k, 0.5), h: Keyword.get(opts, :h, 8.0)}
  end

  @doc """
  One step of the twin: the actions applied to the plant (`[force]` or
  nil) and the plant's measured positions (`%{x: [...], y: [...]}`).
  Returns the twin, with `alarm` set to the first step at which the CUSUM
  passed `h`.
  """
  def observe(%Twin{} = tw, us, measured) do
    {sim, st} = advance(tw.sim, if(us, do: [us]))
    wd = sim.world

    pred = Map.new(wd.axes, fn a -> {a, hd(rows(sim, st[:"p_#{a}"]))} end)
    sq = for a <- wd.axes, {p, m, i} <- Enum.zip([pred[a], Map.fetch!(measured, a), 0..(wd.n - 1)]), Enum.at(wd.w, i) > 0, do: (p - m) * (p - m)
    r = :math.sqrt(Enum.sum(sq) / max(length(sq), 1)) / tw.sigma
    cusum = max(0.0, tw.cusum + r - 1.0 - tw.k)
    t = tw.t + 1
    entry = [t, us || [], Vapor.Canonical.hex_digest(Enum.map(wd.axes, &pred[&1])), Vapor.Canonical.hex_digest(Enum.map(wd.axes, &Map.fetch!(measured, &1))), Float.round(r, 6)]
    head = Vapor.Canonical.hex_digest([tw.head, entry])

    %{tw | sim: sim, t: t, cusum: cusum, alarm: tw.alarm || if(cusum > tw.h, do: t), ledger: [entry | tw.ledger], head: head, actions: [us | tw.actions]}
  end

  @doc """
  Check a twin's ledger: re-simulate its predictions from the start and
  the recorded actions (bit-exact), and recompute the hash chain. `:ok`,
  or `{:error, {:tampered, t}}` at the first entry that disagrees.
  """
  def replay(%Twin{} = tw) do
    wd = tw.sim.world
    sim = %{tw.sim | state: tw.start, t: 0, session: nil}

    tw.ledger
    |> Enum.reverse()
    |> Enum.reduce_while({sim, ""}, fn [t, us, pd, _md, _r] = entry, {sim, head} ->
      {sim, st} = advance(sim, if(us != [], do: [us]))
      ok = Vapor.Canonical.hex_digest(Enum.map(wd.axes, fn a -> hd(rows(sim, st[:"p_#{a}"])) end)) == pd
      if ok, do: {:cont, {sim, Vapor.Canonical.hex_digest([head, entry])}}, else: {:halt, {:error, {:tampered, t}}}
    end)
    |> case do
      {:error, _} = e -> e
      {_, head} -> if head == tw.head, do: :ok, else: {:error, {:tampered, :chain}}
    end
  end
end
