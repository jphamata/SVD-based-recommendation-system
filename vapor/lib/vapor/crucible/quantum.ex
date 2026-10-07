defmodule Vapor.Crucible.Quantum do
  @moduledoc """
  The 1-D Schrödinger equation for **any** potential the user writes
  (docs/CRUCIBLE.md §2), with evidence that needs no reference solution:

    * **bound states** of H = −(ħ²/2m) d²/dx² + V(x) by finite differences
      (Dirichlet walls at the ends of the box): the lowest eigenvalues of
      the tridiagonal matrix by **Sturm-sequence bisection** (exact to the
      tolerance for that matrix) and the states by inverse iteration;
    * **convergence**: the same at n/2, n and 2n points — the observed order
      log₂((Eₙ/₂ − Eₙ)/(Eₙ − E₂ₙ)) should be 2, the scheme's order, and the
      Richardson-extrapolated energies come with the error estimate;
    * the **virial theorem**, true for every bound state of every
      potential: 2⟨T⟩ = ⟨x V′(x)⟩ — with V′ derived symbolically from the
      user's V. A wrong Hamiltonian (a sign, a factor) fails it;
    * with an initial state ψ₀(x), **time evolution** by split-step Fourier
      (unitary, second order), with the norm, the energy and **Ehrenfest's
      theorem**, d⟨p⟩/dt = −⟨V′(x)⟩, checked along the way.

  Units: ħ = 1 and the mass `mass` (default 1).
  """
  alias Vapor.Expr
  alias Vapor.Crucible.Sheet
  alias Vapor.Science.Quantum, as: Q

  def run(text) do
    s = Sheet.parse(text)

    with [] <- s.errors,
         {:ok, v, vt, var} <- Sheet.fun1(s, "V") do
      {a, b} = Map.get(s.ranges, var, Map.get(s.ranges, "x", {-10.0, 10.0}))
      n = s |> Sheet.const("n", 400.0) |> round() |> max(40) |> min(4000)
      k = s |> Sheet.const("states", 5.0) |> round() |> max(1) |> min(30)
      m = Sheet.const(s, "mass", 1.0)
      dv = Expr.diff(vt, var)
      vprime = fn x -> Expr.eval(dv, %{var => x}) end

      levels = for nn <- [div(n, 2), n, 2 * n], do: eigen(v, a, b, nn, k, m)
      [e_half, e_n, e_2n] = Enum.map(levels, & &1.values)
      orders = Enum.zip_with([e_half, e_n, e_2n], fn [x, y, z] -> if abs(y - z) > 1.0e-14, do: :math.log2(abs((x - y) / (y - z))), else: nil end)
      extrap = Enum.zip_with(e_n, e_2n, fn y, z -> z + (z - y) / 3 end)
      errs = Enum.zip_with(e_2n, extrap, &abs(&1 - &2))
      fine = Enum.at(levels, 2)
      virial = Enum.map(fine.states, &virial(&1, fine.x, v, vprime, m, fine.dx))

      states =
        for i <- 0..(k - 1) do
          %{index: i, energy: Enum.at(extrap, i), energy_2n: Enum.at(e_2n, i), error_estimate: Enum.at(errs, i), observed_order: Enum.at(orders, i),
            virial_residual: Enum.at(virial, i)}
        end

      plot = %{x: Enum.take_every(fine.x, max(div(length(fine.x), 300), 1)), v: Enum.take_every(Enum.map(fine.x, v), max(div(length(fine.x), 300), 1)),
               psi: Enum.map(fine.states, &Enum.take_every(&1, max(div(length(&1), 300), 1)))}

      dyn = dynamics(s, v, vprime, a, b, m)
      orders_ok = Enum.count(orders, &(&1 != nil and abs(&1 - 2) < 0.3))
      vir_ok = Enum.count(virial, &(&1 < 1.0e-3))

      {:ok, %{kind: "quantum", potential: Expr.to_text(vt), dv: Expr.to_text(dv), box: [a, b], n: n, mass: m, states: states, plot: plot, dynamics: dyn,
              evidence: [
                %{check: "observed order", ok: orders_ok == k, detail: "#{orders_ok} of #{k} levels converge at order ≈ 2 (the scheme's) between n/2, n and 2n"},
                %{check: "virial theorem", ok: vir_ok == k, detail: "2⟨T⟩ = ⟨x V′⟩ to < 10⁻³ (relative) for #{vir_ok} of #{k} states — V′ = #{Expr.to_text(dv)}, derived symbolically"}
              ] ++ (if dyn, do: dyn.evidence, else: []),
              says: "#{k} bound states; energies extrapolated with their error estimates" <> if(dyn, do: "; ψ₀ evolved #{dyn.steps} split steps", else: "")}}
    else
      errs when is_list(errs) -> {:error, Enum.join(errs, "; ")}
      {:error, e} -> {:error, e <> " — write V(x) = … (and optionally x = a .. b, n = 400, states = 5, mass = 1, psi0(x) = …, t = 0 .. T)"}
    end
  end

  # ------------------------------------------------------------ eigenproblem

  defp eigen(v, a, b, n, k, m) do
    dx = (b - a) / (n + 1)
    xs = for i <- 1..n, do: a + i * dx
    off = -1 / (2 * m * dx * dx)
    d = Enum.map(xs, fn x -> 1 / (m * dx * dx) + v.(x) end) |> List.to_tuple()
    lo = Enum.min(Tuple.to_list(d)) - 2 * abs(off)
    hi = Enum.max(Tuple.to_list(d)) + 2 * abs(off)
    vals = for i <- 0..(k - 1), do: bisect(d, off, n, i, lo, hi)
    states = Enum.map(vals, fn lam -> inverse_iteration(d, off, n, lam, dx) end)
    %{values: vals, states: states, x: xs, dx: dx}
  end

  # the number of eigenvalues below lam (Sturm sequence)
  defp count_below(d, off, n, lam) do
    e2 = off * off
    {c, _} =
      Enum.reduce(0..(n - 1), {0, nil}, fn i, {c, q} ->
        q = if q == nil, do: elem(d, i) - lam, else: elem(d, i) - lam - e2 / (if q == 0.0, do: 1.0e-300, else: q)
        {if(q < 0, do: c + 1, else: c), q}
      end)
    c
  end

  defp bisect(d, off, n, i, lo, hi) do
    Enum.reduce_while(1..200, {lo, hi}, fn _, {l, h} ->
      mid = (l + h) / 2
      if h - l <= 1.0e-14 * max(abs(mid), 1.0), do: {:halt, {l, h}}, else: {:cont, if(count_below(d, off, n, mid) > i, do: {l, mid}, else: {mid, h})}
    end)
    |> then(fn {l, h} -> (l + h) / 2 end)
  end

  defp inverse_iteration(d, off, n, lam, dx) do
    shift = lam + 1.0e-10 * max(abs(lam), 1.0)
    y0 = for i <- 0..(n - 1), do: 1.0 + 0.01 * :math.sin(i * 1.7)
    y = Enum.reduce(1..3, y0, fn _, y -> thomas(d, off, n, shift, y) |> normalize(dx) end)
    # sign: positive lobe first
    first = Enum.find(y, &(abs(&1) > 1.0e-6)) || 1.0
    if first < 0, do: Enum.map(y, &(-&1)), else: y
  end

  defp normalize(y, dx) do
    s = :math.sqrt(Enum.reduce(y, 0.0, &(&1 * &1 + &2)) * dx)
    Enum.map(y, &(&1 / s))
  end

  defp thomas(d, off, n, shift, rhs) do
    r = List.to_tuple(rhs)
    {cp, dp} =
      Enum.reduce(0..(n - 1), {[], []}, fn i, {cs, ds} ->
        bi = elem(d, i) - shift
        case {cs, ds} do
          {[], []} -> {[off / bi], [elem(r, 0) / bi]}
          {[c | _], [dd | _]} ->
            den = bi - off * c
            den = if den == 0.0, do: 1.0e-300, else: den
            {[off / den | cs], [(elem(r, i) - off * dd) / den | ds]}
        end
      end)
    cp = cp |> Enum.reverse() |> List.to_tuple()
    dp = dp |> Enum.reverse() |> List.to_tuple()
    {xs, _} = Enum.reduce((n - 1)..0//-1, {[], nil}, fn i, {acc, nxt} ->
      x = if nxt == nil, do: elem(dp, i), else: elem(dp, i) - elem(cp, i) * nxt
      {[x | acc], x}
    end)
    xs
  end

  defp virial(psi, xs, v, vprime, m, dx) do
    p = List.to_tuple(psi)
    n = tuple_size(p)
    # ⟨T⟩ by the same finite differences, ⟨V⟩ and ⟨x V′⟩ by quadrature
    t =
      Enum.reduce(0..(n - 1), 0.0, fn i, s ->
        l = if i > 0, do: elem(p, i - 1), else: 0.0
        r = if i < n - 1, do: elem(p, i + 1), else: 0.0
        s + elem(p, i) * -(l - 2 * elem(p, i) + r) / (2 * m * dx * dx)
      end) * dx
    xv = Enum.zip(xs, psi) |> Enum.reduce(0.0, fn {x, y}, s -> s + y * y * x * vprime.(x) end) |> Kernel.*(dx)
    ev = Enum.zip(xs, psi) |> Enum.reduce(0.0, fn {x, y}, s -> s + y * y * v.(x) end) |> Kernel.*(dx)
    abs(2 * t - xv) / max(abs(t + ev), 1.0e-12)
  end

  # ------------------------------------------------------------ dynamics

  defp dynamics(s, v, vprime, a, b, m) do
    case Sheet.fun1(s, "psi0") do
      {:ok, f, _t, _var} ->
        {t0, t1} = Map.get(s.ranges, "t", {0.0, 2 * :math.pi()})
        nfft = s |> Sheet.const("nfft", 256.0) |> round() |> pow2()
        l = b - a
        g = Q.grid(nfft, l) |> Map.update!(:x, fn xs -> Enum.map(xs, &(&1 + (a + b) / 2)) end)
        psi0 = Enum.map(g.x, fn x -> {f.(x) * 1.0, 0.0} end) |> Q.normalize(g.dx)
        vs = Enum.map(g.x, v)
        dt = Sheet.const(s, "dt", min((t1 - t0) / 400, 0.02))
        steps = max(round((t1 - t0) / dt), 1)
        every = max(div(steps, 200), 1)
        obs = fn tt, p -> observe(g, p, vs, vprime, m, tt) end
        {psi, trace} = evolve(g, psi0, vs, dt, steps, obs, every, m)
        trace = [observe(g, psi0, vs, vprime, m, 0.0) | trace]
        norm_drift = abs(Q.norm(psi, g.dx) - 1.0)
        e0 = hd(trace).energy
        e_drift = trace |> Enum.map(&abs(&1.energy - e0)) |> Enum.max()
        ehr = ehrenfest(trace)

        %{steps: steps, dt: dt, n: nfft, t: Enum.map(trace, & &1.t), mean_x: Enum.map(trace, & &1.x), mean_p: Enum.map(trace, & &1.p), energy: Enum.map(trace, & &1.energy),
          density: psi |> Enum.map(fn {re, im} -> re * re + im * im end) |> Enum.take_every(max(div(nfft, 256), 1)),
          evidence: [
            %{check: "unitarity", ok: norm_drift < 1.0e-10, detail: "norm drift #{fmt(norm_drift)}"},
            %{check: "energy", ok: e_drift / max(abs(e0), 1.0e-12) < 1.0e-2, detail: "⟨H⟩ moves by #{fmt(e_drift)} (relative #{fmt(e_drift / max(abs(e0), 1.0e-12))}) — split-step conserves a nearby energy, not H exactly"},
            %{check: "Ehrenfest", ok: ehr < 2.0e-2, detail: "d⟨p⟩/dt = −⟨V′⟩ holds to #{fmt(ehr)} (relative, finite differences of the trace)"}
          ]}

      _ -> nil
    end
  end

  defp pow2(n), do: Enum.find([64, 128, 256, 512, 1024, 2048], 2048, &(&1 >= n))

  defp evolve(g, psi, v, dt, steps, observe, every, m) do
    half = Enum.map(v, fn vx -> {:math.cos(-vx * dt / 2), :math.sin(-vx * dt / 2)} end)
    kin = Enum.map(g.k, fn k -> {:math.cos(-k * k * dt / (2 * m)), :math.sin(-k * k * dt / (2 * m))} end)
    mul = fn a, b -> Enum.zip_with(a, b, fn {ar, ai}, {br, bi} -> {ar * br - ai * bi, ar * bi + ai * br} end) end

    Enum.reduce(1..steps, {psi, []}, fn st, {psi, obs} ->
      psi = psi |> mul.(half) |> Q.fft() |> mul.(kin) |> Q.fft(true) |> mul.(half)
      obs = if rem(st, every) == 0, do: [observe.(st * dt, psi) | obs], else: obs
      {psi, obs}
    end)
    |> then(fn {p, o} -> {p, Enum.reverse(o)} end)
  end

  defp observe(g, psi, vs, vprime, m, t) do
    dens = Enum.map(psi, fn {re, im} -> re * re + im * im end)
    x = Enum.zip(g.x, dens) |> Enum.reduce(0.0, fn {xx, d}, s -> s + xx * d end) |> Kernel.*(g.dx)
    pot = Enum.zip(vs, dens) |> Enum.reduce(0.0, fn {vv, d}, s -> s + vv * d end) |> Kernel.*(g.dx)
    force = Enum.zip(g.x, dens) |> Enum.reduce(0.0, fn {xx, d}, s -> s + vprime.(xx) * d end) |> Kernel.*(g.dx)
    phi = Q.fft(psi)
    pk = Enum.map(phi, fn {re, im} -> re * re + im * im end)
    tot = Enum.sum(pk)
    p = Enum.zip(g.k, pk) |> Enum.reduce(0.0, fn {k, w}, s -> s + k * w end) |> Kernel./(tot)
    kin = Enum.zip(g.k, pk) |> Enum.reduce(0.0, fn {k, w}, s -> s + k * k / (2 * m) * w end) |> Kernel./(tot)
    %{t: t, x: x, p: p, energy: kin + pot, force: force}
  end

  defp ehrenfest(trace) do
    pairs = Enum.chunk_every(trace, 2, 1, :discard)
    diffs = Enum.map(pairs, fn [a, b] -> {(b.p - a.p) / (b.t - a.t), -(a.force + b.force) / 2} end)
    scale = diffs |> Enum.map(fn {_, f} -> abs(f) end) |> Enum.max(fn -> 1.0 end) |> max(1.0e-9)
    diffs |> Enum.map(fn {d, f} -> abs(d - f) end) |> Enum.max(fn -> 0.0 end) |> Kernel./(scale)
  end

  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, [{:scientific, 2}])
  defp fmt(x), do: to_string(x)
end
