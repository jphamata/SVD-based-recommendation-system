defmodule Vapor.Science.Relativity do
  @moduledoc """
  A charged particle in electromagnetic fields, special-relativistic
  (c = 1, q = m = 1), by the Boris pusher (Boris 1970) — the integrator of
  particle-in-cell plasma codes: half an electric kick, an exact-norm
  rotation in the magnetic field, half a kick, on the momentum
  u = γv.

  Measured (docs/SCIENCE.md): the gyration period 2πγ/B (time dilation —
  a fast particle turns slower), the conservation of |u| in a pure
  magnetic field to rounding, and the E×B drift velocity E/B. The control
  is the explicit Euler integrator of the same equations, whose energy
  grows every turn.
  """

  defp cross({a1, a2, a3}, {b1, b2, b3}), do: {a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1}
  defp add({a1, a2, a3}, {b1, b2, b3}), do: {a1 + b1, a2 + b2, a3 + b3}
  defp scale({a1, a2, a3}, s), do: {a1 * s, a2 * s, a3 * s}
  defp norm2({a1, a2, a3}), do: a1 * a1 + a2 * a2 + a3 * a3
  defp gamma(u), do: :math.sqrt(1 + norm2(u))

  @doc "One Boris step: `{x, u}` → `{x, u}` in fields `e`, `b` (3-vectors)."
  def boris({x, u}, e, b, dt) do
    um = add(u, scale(e, dt / 2))
    t = scale(b, dt / (2 * gamma(um)))
    s = scale(t, 2 / (1 + norm2(t)))
    up = add(um, cross(add(um, cross(um, t)), s))
    u2 = add(up, scale(e, dt / 2))
    {add(x, scale(u2, dt / gamma(u2))), u2}
  end

  @doc "The control: explicit Euler on du/dt = E + (u/γ)×B."
  def euler({x, u}, e, b, dt) do
    g = gamma(u)
    {add(x, scale(u, dt / g)), add(u, scale(add(e, cross(scale(u, 1 / g), b)), dt))}
  end

  @doc """
  Gyration in B = (0, 0, b) from speed `v` (as a fraction of c): the
  measured period (from the angle of u turned), the theory 2πγ/b, and the
  drift of |u| — for Boris and for the Euler control.
  """
  def gyration(v, opts \\ []) do
    b = Keyword.get(opts, :b, 1.0)
    dt = Keyword.get(opts, :dt, 0.01)
    turns = Keyword.get(opts, :turns, 5)
    g = 1 / :math.sqrt(1 - v * v)
    u0 = {g * v, 0.0, 0.0}
    period = 2 * :math.pi() * g / b
    steps = round(turns * period / dt)

    run = fn step ->
      {state, angle} =
        Enum.reduce(1..steps, {{{0.0, 0.0, 0.0}, u0}, 0.0}, fn _, {{x, u}, ang} ->
          {x2, u2} = step.({x, u}, {0.0, 0.0, 0.0}, {0.0, 0.0, b}, dt)
          {{x2, u2}, ang + angle_between(u, u2)}
        end)

      {_, u} = state
      %{period: 2 * :math.pi() * steps * dt / angle, u_drift: abs(:math.sqrt(norm2(u)) / (g * v) - 1)}
    end

    %{gamma: g, theory: period, boris: run.(&boris/4), euler: run.(&euler/4)}
  end

  defp angle_between({a1, a2, _}, {b1, b2, _}), do: abs(:math.atan2(a1 * b2 - a2 * b1, a1 * b1 + a2 * b2))

  @doc "E×B drift with E = (0, e, 0), B = (0, 0, b), e < b: the mean x-velocity over many gyrations against e/b."
  def drift(e, b, opts \\ []) do
    dt = Keyword.get(opts, :dt, 0.01)
    steps = Keyword.get(opts, :steps, 40_000)
    {{x, _, _}, _} = Enum.reduce(1..steps, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, fn _, st -> boris(st, {0.0, e, 0.0}, {0.0, 0.0, b}, dt) end)
    %{measured: x / (steps * dt), theory: e / b}
  end
end
