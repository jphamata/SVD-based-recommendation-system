defmodule Vapor.Crucible.Fields do
  @moduledoc """
  A charged particle in **any** electric and magnetic fields the user
  writes, relativistic (docs/CRUCIBLE.md §10):

      Ex = 0; Ey = 0.1*cos(t); Ez = 0
      Bx = 0; By = 0; Bz = 1 + 0.1*x       # functions of x, y, z, t
      q = 1; m = 1; c = 1
      x(0) = 0; y(0) = 0; z(0) = 0
      ux(0) = 0.5; uy(0) = 0; uz(0) = 0.1  # u = γv
      t = 0 .. 100; dt = 0.01

  Integrated by the **Boris pusher** (the integrator of particle-in-cell
  codes: E half-kick, B rotation, E half-kick — volume-preserving, exact
  |u| in a pure magnetic field). Evidence without a reference:

    * the **work–energy theorem**: Δ(γmc²) must equal ∫ qE·v dt (the
      magnetic force does no work) — checked along the run;
    * in a field with E ≡ 0, |u| must not move (Boris rotates exactly);
    * the **observed order** from dt, dt/2, dt/4 (2 for Boris);
    * the **control**: explicit Euler at the same dt — its energy error.
  """
  alias Vapor.Expr
  alias Vapor.Crucible.Sheet

  @comps ~w(Ex Ey Ez Bx By Bz)

  def run(text) do
    s = Sheet.parse(text)
    with [] <- s.errors, {:ok, fields} <- fields(s) do
      q = Sheet.const(s, "q", 1.0)
      m = Sheet.const(s, "m", 1.0)
      c = Sheet.const(s, "c", 1.0)
      {t0, t1} = Map.get(s.ranges, "t", {0.0, 50.0})
      dt = Sheet.const(s, "dt", (t1 - t0) / 5000)
      init = fn n -> case Map.get(s.inits, n) do {_, v} -> v; nil -> 0.0 end end
      x0 = {init.("x"), init.("y"), init.("z")}
      u0 = {init.("ux"), init.("uy"), init.("uz")}
      steps = max(round((t1 - t0) / dt), 1)
      every = max(div(steps, 800), 1)
      eval = fn {x, y, z}, t -> Enum.map(fields, fn {f, _} -> f.(%{"x" => x, "y" => y, "z" => z, "t" => t}) end) end
      push = fn st, h, t -> boris(st, h, t, eval, q, m, c) end
      {traj, work, energy} = integrate(push, {x0, u0}, t0, dt, steps, every, eval, q, m, c)
      {ctraj, _, cenergy} = integrate(fn st, h, t -> euler(st, h, t, eval, q, m, c) end, {x0, u0}, t0, dt, steps, every, eval, q, m, c)
      e0 = hd(energy)
      wl = List.last(work)
      el = List.last(energy)
      work_err = Enum.zip(energy, work) |> Enum.map(fn {e, w} -> abs(e - e0 - w) end) |> Enum.max()
      e_only_b = Enum.all?(Enum.take(fields, 3), fn {_, zero} -> zero end)
      unorm = Enum.map(traj, fn %{u: {a, b, cc}} -> :math.sqrt(a * a + b * b + cc * cc) end)
      u_drift = Enum.max(unorm) - Enum.min(unorm)
      ord = order(push, {x0, u0}, t0, dt)
      cerr = cenergy |> Enum.map(&abs(&1 - e0)) |> Enum.max()
      scale = max(abs(e0), 1.0e-12)

      evidence =
        [%{check: "work–energy theorem", ok: work_err / scale < 1.0e-3, detail: "|Δ(γmc²) − ∫qE·v dt| ≤ #{fm(work_err)} (energy moved by #{fm(el - e0)}, work done #{fm(wl)})"},
         e_only_b && %{check: "|u| in a pure magnetic field", ok: u_drift < 1.0e-10 * max(Enum.max(unorm), 1.0), detail: "|u| varies by #{fm(u_drift)}"},
         %{check: "observed order", ok: ord != nil and abs(ord - 2) < 0.4, detail: "#{fm(ord)} (Boris: 2)"},
         %{check: "control", ok: true, detail: "explicit Euler at the same dt: energy error #{fm(cerr)} (Boris: #{fm(Enum.map(energy, &abs(&1 - e0)) |> Enum.max())} — with E ≠ 0 energy changes legitimately; the theorem above is the test)"}]
        |> Enum.filter(& &1)

      {:ok, %{kind: "fields", steps: steps, dt: dt, trajectory: Enum.map(traj, fn %{t: t, x: {a, b, cc}} -> [t, a, b, cc] end),
              control: Enum.map(ctraj, fn %{t: t, x: {a, b, cc}} -> [t, a, b, cc] end), energy: energy, work: work, evidence: evidence,
              says: "#{steps} Boris steps; work–energy residual #{fm(work_err)}"}}
    else
      errs when is_list(errs) -> {:error, Enum.join(errs, "; ")}
      e -> e
    end
  end

  defp fields(s) do
    fs =
      for name <- @comps do
        case Map.get(s.funs, name) do
          {_args, t, _} -> {:ok, t}
          nil -> case Map.get(s.consts, name) do nil -> {:ok, {:n, 0.0}}; v -> {:ok, {:n, v}} end
        end
      end

    bad = fs |> Enum.flat_map(fn {:ok, t} -> Expr.vars(t) end) |> Enum.uniq() |> Enum.reject(&(&1 in ~w(x y z t)))
    if bad != [], do: {:error, "unknown name(s) in the fields: #{Enum.join(bad, ", ")}"},
      else: {:ok, Enum.map(fs, fn {:ok, t} -> mark(t) end)}
  end

  defp mark({:n, v}) when v == 0, do: {fn _ -> 0.0 end, true}
  defp mark(t), do: {fn env -> Expr.eval(t, env) end, false}

  defp gamma({a, b, c}, cc), do: :math.sqrt(1 + (a * a + b * b + c * c) / (cc * cc))

  defp boris({x, u}, h, t, eval, q, m, c) do
    g0 = gamma(u, c)
    xm = add(x, mul(u, h / 2 / g0))
    [ex, ey, ez, bx, by, bz] = eval.(xm, t + h / 2)
    e = {ex, ey, ez}
    b = {bx, by, bz}
    k = q * h / (2 * m)
    um = add(u, mul(e, k))
    g = gamma(um, c)
    tv = mul(b, k / g)
    up = add(um, cross(um, tv))
    sv = mul(tv, 2 / (1 + dot(tv, tv)))
    uplus = add(um, cross(up, sv))
    un = add(uplus, mul(e, k))
    xn = add(xm, mul(un, h / 2 / gamma(un, c)))
    {xn, un}
  end

  defp euler({x, u}, h, t, eval, q, m, c) do
    g = gamma(u, c)
    v = mul(u, 1 / g)
    [ex, ey, ez, bx, by, bz] = eval.(x, t)
    f = mul(add({ex, ey, ez}, cross(v, {bx, by, bz})), q / m)
    {add(x, mul(v, h)), add(u, mul(f, h))}
  end

  defp integrate(step, st0, t0, dt, steps, every, eval, q, m, c) do
    {x0, u0} = st0
    e0 = m * c * c * gamma(u0, c)
    acc0 = {[%{t: t0, x: x0, u: u0}], [0.0], [e0]}
    {rows, works, energies, _, _} =
      Enum.reduce(1..steps, {elem(acc0, 0), elem(acc0, 1), elem(acc0, 2), st0, 0.0}, fn k, {rows, ws, es, {x, u} = st, w} ->
        {xn, un} = step.(st, dt, t0 + (k - 1) * dt)
        # work by the trapezoid rule on q E·v
        [ex, ey, ez | _] = eval.(x, t0 + (k - 1) * dt)
        [fx, fy, fz | _] = eval.(xn, t0 + k * dt)
        p1 = q * dot({ex, ey, ez}, mul(u, 1 / gamma(u, c)))
        p2 = q * dot({fx, fy, fz}, mul(un, 1 / gamma(un, c)))
        w = w + dt * (p1 + p2) / 2
        if rem(k, every) == 0 do
          {[%{t: t0 + k * dt, x: xn, u: un} | rows], [w | ws], [m * c * c * gamma(un, c) | es], {xn, un}, w}
        else
          {rows, ws, es, {xn, un}, w}
        end
      end)
    {Enum.reverse(rows), Enum.reverse(works), Enum.reverse(energies)}
  end

  defp order(push, st0, t0, dt) do
    run = fn h, n -> Enum.reduce(1..n, st0, fn k, st -> push.(st, h, t0 + (k - 1) * h) end) end
    n = 50
    a = run.(dt, n)
    b = run.(dt / 2, 2 * n)
    cc = run.(dt / 4, 4 * n)
    e1 = dist(a, b)
    e2 = dist(b, cc)
    if e1 > 0 and e2 > 0, do: :math.log2(e1 / e2), else: nil
  end

  defp dist({x1, u1}, {x2, u2}), do: :math.sqrt(dot(sub(x1, x2), sub(x1, x2)) + dot(sub(u1, u2), sub(u1, u2)))
  defp add({a, b, c}, {d, e, f}), do: {a + d, b + e, c + f}
  defp sub({a, b, c}, {d, e, f}), do: {a - d, b - e, c - f}
  defp mul({a, b, c}, k), do: {a * k, b * k, c * k}
  defp dot({a, b, c}, {d, e, f}), do: a * d + b * e + c * f
  defp cross({a, b, c}, {d, e, f}), do: {b * f - c * e, c * d - a * f, a * e - b * d}

  defp fm(nil), do: "—"
  defp fm(x) when is_float(x), do: :erlang.float_to_binary(x, [{:scientific, 2}])
  defp fm(x), do: to_string(x)
end
