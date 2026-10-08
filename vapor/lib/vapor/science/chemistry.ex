defmodule Vapor.Science.Chemistry do
  @moduledoc """
  Quantum chemistry from first principles: restricted Hartree–Fock for
  two-electron diatomics in the STO-3G basis, with every integral in
  closed form over s-type Gaussians (Boys function F₀ through `erf`), the
  self-consistent field by symmetric orthogonalisation — the algorithm of
  Szabo & Ostlund, *Modern Quantum Chemistry* (1982), §3.5.

  Measured against the book's published energies (docs/SCIENCE.md): H₂ at
  R = 1.4 bohr (−1.1167 hartree) and HeH⁺ at R = 1.4632 bohr
  (−2.86066). And the method's known failure, shown, not hidden: the RHF
  energy of H₂ pulled apart stays far above two hydrogen atoms (a single
  closed-shell determinant cannot describe two separate electrons) — why
  configuration interaction exists.

  Materials: a Lennard-Jones liquid (`lj/1`), integrated by velocity
  Verlet — energy conserved to 10⁻⁴ over thousands of steps, while the
  explicit Euler control heats up — with its radial distribution function.
  """

  # STO-3G contraction for a 1s Slater function of exponent ζ = 1
  @alpha [0.109818, 0.405771, 2.22766]
  @coef [0.444635, 0.535328, 0.154329]

  defp f0(t) when t < 1.0e-8, do: 1.0 - t / 3
  defp f0(t), do: 0.5 * :math.sqrt(:math.pi() / t) * :math.erf(:math.sqrt(t))

  # contracted 1s on centre x (on the z axis), Slater exponent zeta
  defp basis(x, zeta), do: Enum.zip_with(@alpha, @coef, fn a, d -> {a * zeta * zeta, d * :math.pow(2 * a * zeta * zeta / :math.pi(), 0.75), x} end)

  defp s_prim({a, _, xa}, {b, _, xb}), do: :math.pow(:math.pi() / (a + b), 1.5) * :math.exp(-a * b / (a + b) * (xa - xb) ** 2)
  defp t_prim({a, _, xa}, {b, _, xb}) do
    r2 = (xa - xb) ** 2
    a * b / (a + b) * (3 - 2 * a * b / (a + b) * r2) * :math.pow(:math.pi() / (a + b), 1.5) * :math.exp(-a * b / (a + b) * r2)
  end

  defp v_prim({a, _, xa}, {b, _, xb}, {z, xc}) do
    p = (a * xa + b * xb) / (a + b)
    -2 * :math.pi() / (a + b) * z * :math.exp(-a * b / (a + b) * (xa - xb) ** 2) * f0((a + b) * (p - xc) ** 2)
  end

  defp eri({a, _, xa}, {b, _, xb}, {c, _, xc}, {d, _, xd}) do
    p = (a * xa + b * xb) / (a + b)
    q = (c * xc + d * xd) / (c + d)
    2 * :math.pow(:math.pi(), 2.5) / ((a + b) * (c + d) * :math.sqrt(a + b + c + d)) *
      :math.exp(-a * b / (a + b) * (xa - xb) ** 2 - c * d / (c + d) * (xc - xd) ** 2) * f0((a + b) * (c + d) / (a + b + c + d) * (p - q) ** 2)
  end

  defp contract2(mu, nu, f), do: (for pa <- mu, pb <- nu, reduce: 0.0, do: (acc -> acc + elem(pa, 1) * elem(pb, 1) * f.(pa, pb)))

  defp contract4(m, n, l, s) do
    for pa <- m, pb <- n, pc <- l, pd <- s, reduce: 0.0 do
      acc -> acc + elem(pa, 1) * elem(pb, 1) * elem(pc, 1) * elem(pd, 1) * eri(pa, pb, pc, pd)
    end
  end

  @doc """
  RHF for two nuclei (charges za, zb, Slater exponents ζa, ζb) at distance
  r with two electrons: `%{energy, electronic, iterations, orbital_energies}`.
  """
  def rhf(za, zb, zeta_a, zeta_b, r, opts \\ []) do
    bs = [basis(0.0, zeta_a), basis(r, zeta_b)]
    nuclei = [{za, 0.0}, {zb, r}]
    idx = [0, 1]
    s = for i <- idx, do: for(j <- idx, do: contract2(Enum.at(bs, i), Enum.at(bs, j), &s_prim/2))
    h = for i <- idx, do: for(j <- idx, do: contract2(Enum.at(bs, i), Enum.at(bs, j), fn p, q -> t_prim(p, q) + Enum.sum(for c <- nuclei, do: v_prim(p, q, c)) end))
    two = for i <- idx, j <- idx, k <- idx, l <- idx, into: %{}, do: {{i, j, k, l}, contract4(Enum.at(bs, i), Enum.at(bs, j), Enum.at(bs, k), Enum.at(bs, l))}
    x = inv_sqrt(s)

    {p, e_el, it, eps} =
      Enum.reduce_while(1..Keyword.get(opts, :max_iter, 100), {[[0.0, 0.0], [0.0, 0.0]], 0.0, 0, nil}, fn it, {p, e_old, _, _} ->
        g = for i <- idx, do: for(j <- idx, do: Enum.sum(for k <- idx, l <- idx, do: at(p, k, l) * (two[{i, j, l, k}] - 0.5 * two[{i, k, l, j}])))
        f = madd(h, g)
        {vals, vecs} = eig2(mmul(mmul(transpose(x), f), x))
        c = mmul(x, vecs)
        # the lowest orbital doubly occupied
        p_new = for i <- idx, do: for(j <- idx, do: 2 * at(c, i, 0) * at(c, j, 0))
        e = 0.5 * Enum.sum(for i <- idx, j <- idx, do: at(p_new, i, j) * (at(h, i, j) + at(f, i, j)))
        if it > 1 and abs(e - e_old) < 1.0e-11, do: {:halt, {p_new, e, it, vals}}, else: {:cont, {p_new, e, it, vals}}
      end)

    _ = p
    %{energy: e_el + za * zb / r, electronic: e_el, iterations: it, orbital_energies: eps}
  end

  @doc "H₂ at R (bohr), ζ = 1.24 (the book's)."
  def h2(r \\ 1.4), do: rhf(1.0, 1.0, 1.24, 1.24, r)
  @doc "HeH⁺ at R, ζ_He = 2.0925, ζ_H = 1.24 (the book's)."
  def heh(r \\ 1.4632), do: rhf(2.0, 1.0, 2.0925, 1.24, r)
  @doc "A hydrogen atom in the same basis: −⟨T + V⟩ of the normalised contraction."
  def h_atom(zeta \\ 1.24) do
    b = basis(0.0, zeta)
    contract2(b, b, fn p, q -> t_prim(p, q) + v_prim(p, q, {1.0, 0.0}) end) / contract2(b, b, &s_prim/2)
  end

  defp at(m, i, j), do: m |> Enum.at(i) |> Enum.at(j)
  defp transpose(m), do: Enum.zip_with(m, & &1)
  defp madd(a, b), do: Enum.zip_with(a, b, fn r1, r2 -> Enum.zip_with(r1, r2, &+/2) end)
  defp mmul(a, b), do: for(r <- a, do: for(c <- transpose(b), do: Enum.sum(Enum.zip_with(r, c, &*/2))))

  # eigen-decomposition of a symmetric 2×2: values ascending, vectors as columns
  defp eig2([[a, b], [_, d]]) do
    tr = a + d
    disc = :math.sqrt(((a - d) / 2) ** 2 + b * b)
    {l1, l2} = {tr / 2 - disc, tr / 2 + disc}
    v = fn l -> if abs(b) > 1.0e-14, do: unit([b, l - a]), else: (if abs(l - a) < abs(l - d), do: [1.0, 0.0], else: [0.0, 1.0]) end
    [v1, v2] = [v.(l1), v.(l2)]
    {[l1, l2], [[Enum.at(v1, 0), Enum.at(v2, 0)], [Enum.at(v1, 1), Enum.at(v2, 1)]]}
  end

  defp unit([x, y]), do: (n = :math.sqrt(x * x + y * y); [x / n, y / n])

  defp inv_sqrt(s) do
    {[l1, l2], u} = eig2(s)
    d = [[1 / :math.sqrt(l1), 0.0], [0.0, 1 / :math.sqrt(l2)]]
    mmul(mmul(u, d), transpose(u))
  end

  # ---------------------------------------------------------- Lennard-Jones

  @doc """
  A Lennard-Jones liquid (reduced units, cut-off 2.5σ, periodic cube): n³
  particles on a lattice at density `rho`, velocities from a seeded
  draw at temperature `t`, integrated `steps` steps by velocity Verlet
  (`integrator: :euler` is the control). Returns the energy series
  `%{energies, drift (relative), rdf: [{r, g}]}`.
  """
  def lj(opts \\ []) do
    {k, rho, temp, dt, steps} = {Keyword.get(opts, :cells, 4), Keyword.get(opts, :rho, 0.8), Keyword.get(opts, :t, 1.0), Keyword.get(opts, :dt, 0.004), Keyword.get(opts, :steps, 500)}
    integ = Keyword.get(opts, :integrator, :verlet)
    n = k * k * k
    box = :math.pow(n / rho, 1 / 3)
    a = box / k
    pos = for i <- 0..(k - 1), j <- 0..(k - 1), l <- 0..(k - 1), do: {(i + 0.5) * a, (j + 0.5) * a, (l + 0.5) * a}
    vel = for i <- 0..(n - 1), do: (for d <- 0..2, do: (Vapor.Sampler.uniform(7 + d, i) - 0.5)) |> List.to_tuple()
    # remove drift, scale to temperature
    mean = vel |> Enum.reduce({0.0, 0.0, 0.0}, &v_add/2) |> v_scale(1 / n)
    vel = Enum.map(vel, &v_sub(&1, mean))
    ke = Enum.reduce(vel, 0.0, fn v, s -> s + v_dot(v, v) end) / 2
    vel = Enum.map(vel, &v_scale(&1, :math.sqrt(1.5 * n * temp / ke)))

    {f0, u0} = forces(pos, box)

    {final, _, _, es} =
      Enum.reduce(1..steps, {pos, vel, f0, [u0 + kinetic(vel)]}, fn _, {p, v, f, es} ->
        case integ do
          :verlet ->
            v_half = Enum.zip_with(v, f, fn vi, fi -> v_add(vi, v_scale(fi, dt / 2)) end)
            p2 = Enum.zip_with(p, v_half, fn pi, vi -> wrap(v_add(pi, v_scale(vi, dt)), box) end)
            {f2, u2} = forces(p2, box)
            v2 = Enum.zip_with(v_half, f2, fn vi, fi -> v_add(vi, v_scale(fi, dt / 2)) end)
            {p2, v2, f2, [u2 + kinetic(v2) | es]}

          :euler ->
            p2 = Enum.zip_with(p, v, fn pi, vi -> wrap(v_add(pi, v_scale(vi, dt)), box) end)
            v2 = Enum.zip_with(v, f, fn vi, fi -> v_add(vi, v_scale(fi, dt)) end)
            {f2, u2} = forces(p2, box)
            {p2, v2, f2, [u2 + kinetic(v2) | es]}
        end
      end)

    es = Enum.reverse(es)
    %{energies: es, drift: abs(List.last(es) - hd(es)) / abs(hd(es)), fluctuation: (Enum.max(es) - Enum.min(es)) / abs(hd(es)), n: n, box: box, rdf: rdf(final, box)}
  end

  defp kinetic(v), do: Enum.reduce(v, 0.0, fn x, s -> s + v_dot(x, x) end) / 2

  defp forces(pos, box) do
    t = List.to_tuple(pos)
    n = tuple_size(t)
    rc2 = 6.25
    ec = 4 * (1 / rc2 ** 6 - 1 / rc2 ** 3)

    {f, u} =
      for i <- 0..(n - 2), j <- (i + 1)..(n - 1), reduce: {%{}, 0.0} do
        {f, u} ->
          d = min_image(v_sub(elem(t, i), elem(t, j)), box)
          r2 = v_dot(d, d)

          if r2 < rc2 do
            ir6 = 1 / (r2 * r2 * r2)
            fm = 24 * ir6 * (2 * ir6 - 1) / r2
            fv = v_scale(d, fm)
            {f |> Map.update(i, fv, &v_add(&1, fv)) |> Map.update(j, v_scale(fv, -1.0), &v_sub(&1, fv)), u + 4 * ir6 * (ir6 - 1) - ec}
          else
            {f, u}
          end
      end

    {for(i <- 0..(n - 1), do: Map.get(f, i, {0.0, 0.0, 0.0})), u}
  end

  defp rdf(pos, box) do
    t = List.to_tuple(pos)
    n = tuple_size(t)
    bins = 30
    rmax = min(box / 2, 3.0)
    dr = rmax / bins
    counts = for i <- 0..(n - 2), j <- (i + 1)..(n - 1), d = min_image(v_sub(elem(t, i), elem(t, j)), box), r = :math.sqrt(v_dot(d, d)), r < rmax, reduce: %{} do
      acc -> Map.update(acc, trunc(r / dr), 2, &(&1 + 2))
    end
    rho = n / box ** 3
    for b <- 0..(bins - 1), do: (r = (b + 0.5) * dr; {r, Map.get(counts, b, 0) / (n * rho * 4 * :math.pi() * r * r * dr)})
  end

  defp min_image({x, y, z}, box), do: {x - box * Float.round(x / box), y - box * Float.round(y / box), z - box * Float.round(z / box)}
  defp wrap({x, y, z}, box), do: {x - box * Float.floor(x / box), y - box * Float.floor(y / box), z - box * Float.floor(z / box)}
  defp v_add({a, b, c}, {d, e, f}), do: {a + d, b + e, c + f}
  defp v_sub({a, b, c}, {d, e, f}), do: {a - d, b - e, c - f}
  defp v_scale({a, b, c}, s), do: {a * s, b * s, c * s}
  defp v_dot({a, b, c}, {d, e, f}), do: a * d + b * e + c * f
end
