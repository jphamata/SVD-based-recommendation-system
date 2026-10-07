defmodule Vapor.Science.Plasma do
  @moduledoc """
  Tokamak equilibrium: the Grad–Shafranov equation for the poloidal flux
  ψ(R, Z) of an axisymmetric plasma,

      Δ*ψ = R ∂_R (R⁻¹ ∂_R ψ) + ∂²_Z ψ = −μ₀R² p′(ψ) − F F′(ψ),

  with the Solov'ev profiles (p′ and FF′ constant, Solov'ev 1968), whose
  exact solutions are polynomials: ψ = A·R⁴/8 + B·Z²/2 + c₁ + c₂R² +
  c₃(R⁴ − 4R²Z²) solves Δ*ψ = A·R² + B.

  Solved by second-order finite differences on (R, Z) with successive
  over-relaxation, the boundary from the exact solution (docs/CIENCIA.md).
  The conservative scheme has **no truncation error on these polynomials**
  (checked by hand: its R-flux differences are exact on R², R⁴ and R²Z²),
  so the flux matches to the SOR tolerance — this tests the solver and the
  discretisation's form, not an h² rate. The **magnetic axis** (where
  ∇ψ = 0), located by a parabola through the grid maximum, converges as h²
  (that rate is the locator's). The control: the same solver with the
  Cartesian Laplacian (the R⁻¹∂_R term of toroidal geometry dropped) does
  not converge to the solution — the error stays as the mesh is refined.
  """

  @doc "The exact Solov'ev flux with these coefficients."
  def solovev(%{a: a, b: b, c1: c1, c2: c2, c3: c3}, r, z), do: a * r ** 4 / 8 + b * z * z / 2 + c1 + c2 * r * r + c3 * (r ** 4 - 4 * r * r * z * z)

  @doc "A default equilibrium: its magnetic axis (a maximum of ψ) at R = 1, Z = 0."
  def params, do: %{a: -1.0, b: -1.5, c1: 0.0, c2: 0.5, c3: -0.125}

  @doc """
  Solve on [r0, r1] × [z0, z1] with n×n interior points: `%{psi (rows of
  Z), max_error, axis, axis_exact, iters}`. `toroidal: false` is the
  control (Cartesian Laplacian).
  """
  def solve(n, opts \\ []) do
    p = Keyword.get(opts, :params, params())
    {r0, r1, z0, z1} = {0.6, 1.4, -0.4, 0.4}
    toroidal = Keyword.get(opts, :toroidal, true)
    hr = (r1 - r0) / (n + 1)
    hz = (z1 - z0) / (n + 1)
    rs = for i <- 0..(n + 1), do: r0 + i * hr
    zs = for j <- 0..(n + 1), do: z0 + j * hz
    exact = fn i, j -> solovev(p, Enum.at(rs, i), Enum.at(zs, j)) end

    # grid with boundary values, interior zeros
    psi0 =
      for j <- 0..(n + 1), into: %{} do
        {j, (for i <- 0..(n + 1), into: %{}, do: {i, if(i in [0, n + 1] or j in [0, n + 1], do: exact.(i, j), else: 0.0)})}
      end

    omega = 2 / (1 + :math.sin(:math.pi() / (n + 1)))
    rt = List.to_tuple(rs)

    sweep = fn psi ->
      Enum.reduce(1..n, {psi, 0.0}, fn j, {psi, res} ->
        zj = Enum.at(zs, j)
        Enum.reduce(1..n, {psi, res}, fn i, {psi, res} ->
          r = elem(rt, i)
          src = p.a * r * r + p.b + 0 * zj
          # R ∂_R(R⁻¹ ∂_R ψ) ≈ (ψ_{i+1} − ψ_i)/(h² · (1 + h/2R)⁻¹)… in conservative form:
          {cw, ce} = if toroidal, do: {r / (r - hr / 2), r / (r + hr / 2)}, else: {1.0, 1.0}
          row = psi[j]
          new = ((cw * row[i - 1] + ce * row[i + 1]) / (hr * hr) + (psi[j - 1][i] + psi[j + 1][i]) / (hz * hz) - src) / ((cw + ce) / (hr * hr) + 2 / (hz * hz))
          upd = row[i] + omega * (new - row[i])
          {put_in(psi[j][i], upd), max(res, abs(upd - row[i]))}
        end)
      end)
    end

    {psi, iters} =
      Enum.reduce_while(1..20_000, {psi0, 0}, fn it, {psi, _} ->
        {psi, res} = sweep.(psi)
        if res < 1.0e-12, do: {:halt, {psi, it}}, else: {:cont, {psi, it}}
      end)

    err = for(j <- 1..n, i <- 1..n, do: abs(psi[j][i] - exact.(i, j))) |> Enum.max()

    # the magnetic axis: the interior extremum of ψ, refined by a parabola in each direction
    {ai, aj} = for(j <- 1..n, i <- 1..n, do: {i, j}) |> Enum.max_by(fn {i, j} -> psi[j][i] end)
    axis = {Enum.at(rs, ai) + hr * vertex(psi[aj][ai - 1], psi[aj][ai], psi[aj][ai + 1]), Enum.at(zs, aj) + hz * vertex(psi[aj - 1][ai], psi[aj][ai], psi[aj + 1][ai])}
    %{max_error: err, axis: axis, axis_exact: axis_exact(p), iters: iters, h: hr}
  end

  defp vertex(a, b, c), do: if(a - 2 * b + c == 0, do: 0.0, else: 0.5 * (a - c) / (a - 2 * b + c))

  @doc "Where ∇ψ = 0 for the exact solution (Z = 0 by symmetry; R from ∂_R ψ = 0)."
  def axis_exact(p) do
    # ∂_R ψ at Z = 0: A R³/2 + 2 c₂ R + 4 c₃ R³ = 0 → R² = −2c₂ / (A/2 + 4c₃)
    {:math.sqrt(-2 * p.c2 / (p.a / 2 + 4 * p.c3)), 0.0}
  end
end
