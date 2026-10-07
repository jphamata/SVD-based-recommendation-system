defmodule Vapor.PhysicsTest do
  @moduledoc """
  `Vapor.Physics` measured against exact references and controls: a
  pendulum's period against the elliptic-integral period, converging with
  the substep; a chaotic double pendulum bit-identical on the oracle and
  the native worker while a one-ulp change of its start diverges;
  trajectory gradients against finite differences; a twin's parameters
  recovered from noisy measurements (time-shuffled measurements, the
  control, recover nothing); a cart-pole balanced by random search (the
  zero policy, the control, falls); a twin that raises its alarm on a
  fault and not without one, whose ledger replays and catches tampering.
  """
  use ExUnit.Case, async: false
  alias Vapor.Physics, as: P

  @moduletag timeout: 1_200_000

  defp worker, do: Vapor.Vision.OCR.worker()

  defp period(wd, steps, w) do
    sim = P.start(wd, 1, worker: w)

    {xs, _} =
      Enum.map_reduce(1..steps, sim, fn _, s ->
        {s, st} = P.advance(s)
        {s |> P.rows(st.p_x) |> hd() |> Enum.at(1), s}
      end)

    cross =
      xs
      |> Enum.with_index(1)
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [{a, _}, {b, _}] -> a > 0 and b <= 0 end)
      |> Enum.map(fn [{a, i}, {b, _}] -> (i + a / (a - b)) * wd.dt end)

    (List.last(cross) - hd(cross)) / (length(cross) - 1)
  end

  @tag :native
  test "a pendulum's period: the exact elliptic period, the error halving as the substep halves" do
    exact = P.pendulum_period(1.0, 0.5)
    errs = for sub <- [4, 8, 16], do: abs(period(P.pendulum(theta: 0.5, substeps: sub, dt: 0.01), 400, worker()) - exact) / exact
    assert Enum.at(errs, 2) < 2.0e-4, inspect(errs)
    # first order: each halving of the substep at least 1.6× better
    assert Enum.at(errs, 0) > 1.6 * Enum.at(errs, 1) and Enum.at(errs, 1) > 1.6 * Enum.at(errs, 2), inspect(errs)
  end

  @tag :native
  test "a double pendulum: oracle and native agree to the bit; a one-ulp start diverges" do
    wd = P.double_pendulum(substeps: 8, dt: 0.01)
    x2 = wd.pos0 |> Enum.at(2) |> hd()
    <<i::32>> = <<x2::float-32>>
    <<x2u::float-32>> = <<i + 1::32>>
    st = P.state(wd, 2, perturb: fn b, p, a -> if b == 1 and p == 2 and a == :x, do: x2u - x2, else: 0.0 end)
    n = P.start(wd, 2, worker: worker(), state: st, steps: 10)
    o = P.start(wd, 2, state: st, steps: 10)

    {n, o} =
      Enum.reduce(1..20, {n, o}, fn _, {n, o} ->
        {n, sn} = P.advance(n)
        {o, so} = P.advance(o)
        assert sn == so
        {n, o}
      end)

    _ = o
    # 2000 steps (20 s): the perturbed world has parted from the other
    {n, st} = Enum.reduce(1..180, {n, nil}, fn _, {n, _} -> P.advance(n) end)
    [a, b] = P.rows(n, st.p_x)
    assert abs(Enum.at(a, 2) - Enum.at(b, 2)) > 1.0e-2
    P.stop(n)
  end

  @tag :native
  test "trajectory gradients equal finite differences of the simulated loss" do
    wd = P.pendulum(theta: 0.8, substeps: 4, dt: 0.02, damping: 0.2)
    sim = P.start(wd, 1, worker: worker())
    {obs, _} = Enum.map_reduce(1..15, sim, fn _, s -> {s, st} = P.advance(s); {%{x: hd(P.rows(s, st.p_x)), y: hd(P.rows(s, st.p_y))}, s} end)
    {:ok, comp} = Vapor.Compile.Lower.lower(P.sysid_program(wd, 15))

    loss = fn rest, damp ->
      st0 = P.state(wd, 1) |> Map.new(fn {{k, a}, rows} -> {:"#{k}0_#{a}", Vapor.Tensor.from_list(:f32, [1, wd.np], List.flatten(rows))} end)
      ob = for {o, t} <- Enum.with_index(obs, 1), a <- wd.axes, into: %{}, do: {:"o#{t}_#{a}", Vapor.Tensor.from_list(:f32, [1, wd.np], o[a] ++ List.duplicate(0.0, wd.np - wd.n))}
      env = st0 |> Map.merge(ob) |> Map.merge(P.params(wd, rest: [rest], damping: damp)) |> Map.put(:u, Vapor.Tensor.from_list(:f32, [1, 16], List.duplicate(0.0, 16)))
      {:ok, r} = Vapor.Runtime.Native.run(worker(), comp, env, isa: Vapor.Runtime.Substrates.host_isa(), mode: :native)
      {hd(Vapor.Tensor.to_floats(r.outputs.loss)), hd(Vapor.Tensor.to_floats(r.outputs.grad_rest)), hd(Vapor.Tensor.to_floats(r.outputs.grad_damping))}
    end

    {_, gr, gd} = loss.(0.9, 0.0)
    e = 1.0e-3
    fr = (elem(loss.(0.9 + e, 0.0), 0) - elem(loss.(0.9 - e, 0.0), 0)) / (2 * e)
    fd = (elem(loss.(0.9, e), 0) - elem(loss.(0.9, -e), 0)) / (2 * e)
    assert abs(gr - fr) / abs(fr) < 0.03, "rest: #{gr} vs #{fr}"
    assert abs(gd - fd) / abs(fd) < 0.03, "damping: #{gd} vs #{fd}"
  end

  @tag :native
  test "a twin's rod length and damping recovered from noisy measurements; shuffled measurements recover nothing" do
    w = worker()
    plant = P.pendulum(theta: 0.8, substeps: 4, dt: 0.02, damping: 0.3)
    sim = P.start(plant, 1, worker: w)

    {obs, _} =
      Enum.map_reduce(1..40, sim, fn t, s ->
        {s, st} = P.advance(s)
        noise = fn k -> (Vapor.Sampler.uniform(7, t * 10 + k) - 0.5) * 0.004 end
        [x] = P.rows(s, st.p_x)
        [y] = P.rows(s, st.p_y)
        {%{x: Enum.with_index(x, fn v, i -> v + noise.(i) end), y: Enum.with_index(y, fn v, i -> v + noise.(i + 5) end)}, s}
      end)

    twin = %{plant | damping: 0.0}
    fit = P.sysid(twin, obs, worker: w, rest: [0.85], damping: 0.0, iters: 100, lr: 0.02)
    assert abs(hd(fit.rest) - 1.0) < 0.01 and abs(fit.damping - 0.3) < 0.05, inspect(Map.drop(fit, [:loss]))
    assert List.last(fit.loss) < 0.01 * hd(fit.loss)

    control = P.sysid(twin, Vapor.Modal.Rng.permute(obs, 3), worker: w, rest: [0.85], damping: 0.0, iters: 100, lr: 0.02)
    assert abs(hd(control.rest) - 1.0) > 0.05 or abs(control.damping - 0.3) > 0.2
    assert List.last(control.loss) > 100 * List.last(fit.loss)
  end

  @tag :native
  test "random search balances the cart-pole on unseen starts; the zero policy falls" do
    w = worker()
    {k, curve} = P.cartpole_ars(worker: w)
    assert List.last(curve) > 2 * hd(curve)
    assert P.cartpole_returns(List.duplicate(k, 8), worker: w, seed: 999) == List.duplicate(200, 8)
    assert Enum.max(P.cartpole_returns(List.duplicate([0.0, 0.0, 0.0, 0.0], 8), worker: w, seed: 999)) < 100
  end

  @tag :native
  test "a twin: silent while the plant is the model, alarmed by a fault; its ledger replays and catches tampering" do
    w = worker()
    wd = P.pendulum(theta: 0.6, substeps: 4, dt: 0.02)
    noise = fn t, k -> (Vapor.Sampler.uniform(11, t * 10 + k) - 0.5) * 0.004 end
    meas = fn s, st, t -> %{x: Enum.with_index(hd(P.rows(s, st.p_x)), fn v, i -> v + noise.(t, i) end), y: Enum.with_index(hd(P.rows(s, st.p_y)), fn v, i -> v + noise.(t, i + 5) end)} end

    # the plant on the oracle, the twin on the worker (a worker holds one session)
    run = fn fault_at ->
      plant = P.start(wd, 1)

      Enum.reduce(1..150, {plant, P.twin(wd, worker: w)}, fn t, {plant, tw} ->
        # the fault: the rod stretches by 0.5 % — 5 mm, a few times the sensor noise (a new plant from the same state)
        plant = if t == fault_at, do: reopen(plant), else: plant
        {plant, st} = P.advance(plant)
        {plant, P.observe(tw, nil, meas.(plant, st, t))}
      end)
      |> elem(1)
    end

    quiet = run.(nil)
    assert quiet.alarm == nil
    faulty = run.(60)
    assert faulty.alarm != nil and faulty.alarm >= 60 and faulty.alarm <= 90, inspect(faulty.alarm)

    assert P.replay(%{quiet | ledger: Enum.take(quiet.ledger, -40), head: chain(Enum.take(quiet.ledger, -40))}) == :ok
    [[t, us, _pd, md, r] | rest] = Enum.reverse(Enum.take(quiet.ledger, -40))
    forged = Enum.reverse([[t, us, Vapor.Canonical.hex_digest(:forged), md, r] | rest])
    assert {:error, {:tampered, 1}} = P.replay(%{quiet | ledger: forged, head: chain(forged)})
  end

  # the chain head of a ledger (entries newest first)
  defp chain(ledger), do: ledger |> Enum.reverse() |> Enum.reduce("", fn e, h -> Vapor.Canonical.hex_digest([h, e]) end)

  # a plant restarted from a given state on a fresh session
  defp reopen(sim), do: P.start(sim.world, 1, params: [rest: [1.005]], state: unstate(sim))

  defp unstate(sim), do: Map.new(sim.state, fn {k, t} -> [kind, a] = String.split(to_string(k), "_"); {{String.to_atom(kind), String.to_atom(a)}, Enum.chunk_every(Vapor.Tensor.to_floats(t), sim.world.np)} end)

  test "a chain falls onto the floor and rests on it (the floor is a constraint, not a suggestion)" do
    wd = P.chain(links: 4, height: 0.5, substeps: 8, dt: 0.02)
    sim = P.start(wd, 1)
    {sim, st} = Enum.reduce(1..60, {sim, nil}, fn _, {s, _} -> P.advance(s) end)
    ys = hd(P.rows(sim, st.p_y))
    assert Enum.all?(ys, &(&1 >= 0.0))
    # the links keep their length within 2 %
    xs = hd(P.rows(sim, st.p_x))
    for i <- 0..3, do: assert(abs(:math.sqrt((Enum.at(xs, i + 1) - Enum.at(xs, i)) ** 2 + (Enum.at(ys, i + 1) - Enum.at(ys, i)) ** 2) - 0.25) < 0.005)
  end
end
