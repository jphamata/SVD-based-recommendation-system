defmodule Vapor.Crucible.Molecule do
  @moduledoc """
  Restricted Hartree–Fock for **any** arrangement of hydrogen and helium
  nuclei the user writes (docs/CRUCIBLE.md §8) — H₂, HeH⁺, H₃⁺, a chain of
  H atoms, He₂ — in the STO-3G basis (one contracted 1s per atom), every
  integral in closed form over s-type Gaussians in three dimensions.

      H 0 0 0
      H 0 0 1.4
      units = bohr          # or angstrom
      charge = 0
      scan = 0.6 .. 4.0     # optional: move the last atom along its axis to the first and scan the distance

  Evidence that needs no reference value:

    * **self-consistency**: the commutator ‖FPS − SPF‖ — zero exactly at a
      converged solution, whatever the molecule;
    * **electron count**: tr(PS) equals the number of electrons;
    * the **virial ratio** −V/T, reported (2 for an exact wavefunction at
      equilibrium; a minimal basis with fixed exponents misses it — which
      is itself information about the basis);
    * and, when the input is one of the textbook molecules (H₂ at 1.4 bohr,
      HeH⁺ at 1.4632 bohr), the published value as a reference.

  Atoms beyond helium need p orbitals: refused, with the reason.
  """
  alias Vapor.Crucible.Sheet

  @alpha [0.109818, 0.405771, 2.22766]
  @coef [0.444635, 0.535328, 0.154329]
  @zeta %{"H" => 1.24, "He" => 2.0925}
  @z %{"H" => 1.0, "He" => 2.0}
  @angstrom 1.8897261246

  def run(text) do
    s = Sheet.parse(text)
    atoms = for {_, line} <- s.raw, a = atom(line), a != nil, do: a
    bad = for {_, line} <- s.raw, atom(line) == nil, do: line
    units = Sheet.word(s, "units", "bohr")
    charge = s |> Sheet.const("charge", 0.0) |> round()
    k = if units in ["angstrom", "A", "Å", "ang"], do: @angstrom, else: 1.0
    atoms = Enum.map(atoms, fn {el, x, y, z} -> {el, {x * k, y * k, z * k}} end)

    cond do
      bad != [] -> {:error, "not understood: #{Enum.join(Enum.take(bad, 3), " | ")} — atoms are lines `H x y z` (H or He; heavier atoms need p orbitals, not in this basis)"}
      atoms == [] -> {:error, "no atoms: write lines like `H 0 0 0`"}
      length(atoms) > 12 -> {:error, "at most 12 atoms"}
      true ->
        n_el = round(Enum.sum(Enum.map(atoms, fn {el, _} -> @z[el] end))) - charge
        cond do
          n_el <= 0 -> {:error, "no electrons left (charge #{charge})"}
          rem(n_el, 2) == 1 -> {:error, "#{n_el} electrons: restricted Hartree–Fock needs a closed shell (an even count; change the charge)"}
          div(n_el, 2) > length(atoms) -> {:error, "#{n_el} electrons do not fit in #{length(atoms)} 1s orbitals"}
          true -> compute(atoms, n_el, s, k)
        end
    end
  end

  defp atom(line) do
    case String.split(line) do
      [el, x, y, z] when el in ["H", "He"] ->
        with {xf, _} <- Float.parse(x), {yf, _} <- Float.parse(y), {zf, _} <- Float.parse(z), do: {el, xf, yf, zf}, else: (_ -> nil)
      _ -> nil
    end
  end

  defp compute(atoms, n_el, s, k) do
    r = rhf(atoms, n_el)
    ref = reference(atoms, n_el)
    scan = case Map.get(s.ranges, "scan") do
      {a, b} when length(atoms) >= 2 -> scan(atoms, n_el, a * k, b * k)
      _ -> nil
    end

    evidence =
      [%{check: "self-consistency", ok: r.commutator < 1.0e-6, detail: "‖FPS − SPF‖ = #{f(r.commutator)} after #{r.iterations} iterations"},
       %{check: "electron count", ok: abs(r.trace - n_el) < 1.0e-8, detail: "tr(PS) = #{f(r.trace)} (#{n_el} electrons)"},
       %{check: "virial ratio", ok: true, detail: "−V/T = #{f(r.virial)} (2 for an exact wavefunction at equilibrium)"}] ++
        if(ref, do: [%{check: "textbook value", ok: abs(r.energy - ref.value) < 1.0e-4, detail: "#{ref.name}: #{f(r.energy)} against #{ref.value} hartree (Szabo & Ostlund)"}], else: [])

    {:ok, Map.merge(r, %{kind: "molecule", atoms: Enum.map(atoms, fn {el, {x, y, z}} -> [el, x, y, z] end), electrons: n_el, scan: scan, evidence: evidence,
                         says: "RHF/STO-3G energy #{f(r.energy)} hartree (#{n_el} electrons, #{length(atoms)} atoms)" <>
                           if(scan && scan.minimum, do: "; scan minimum at R = #{f(scan.minimum.r)} bohr, E = #{f(scan.minimum.energy)}", else: "")})}
  end

  defp f(x) when is_float(x), do: :erlang.float_to_binary(x, [{:decimals, 6}, :compact])
  defp f(x), do: to_string(x)

  defp reference([{"H", a}, {"H", b}], 2), do: if(abs(dist(a, b) - 1.4) < 1.0e-9, do: %{name: "H₂ at 1.4 bohr", value: -1.1167})
  defp reference([{x, a}, {y, b}], 2) when x != y, do: if(abs(dist(a, b) - 1.4632) < 1.0e-9, do: %{name: "HeH⁺ at 1.4632 bohr", value: -2.860662})
  defp reference(_, _), do: nil

  defp scan(atoms, n_el, a, b) do
    {head, [{el, last}]} = Enum.split(atoms, -1)
    {_, first} = hd(atoms)
    dir = norm_vec(sub(last, first), {0.0, 0.0, 1.0})
    pts =
      for i <- 0..24 do
        rr = a + (b - a) * i / 24
        pos = add(first, scale(dir, rr))
        e = rhf(head ++ [{el, pos}], n_el).energy
        %{r: rr, energy: e}
      end
    min = Enum.min_by(pts, & &1.energy)
    i = Enum.find_index(pts, &(&1 == min))
    refined =
      if i > 0 and i < 24 do
        [p0, p1, p2] = Enum.slice(pts, i - 1, 3)
        h = p1.r - p0.r
        den = p0.energy - 2 * p1.energy + p2.energy
        if den > 0, do: (rmin = p1.r + h * (p0.energy - p2.energy) / (2 * den); %{r: rmin, energy: rhf(head ++ [{el, add(first, scale(dir, rmin))}], n_el).energy}), else: min
      end
    %{points: pts, minimum: refined}
  end

  defp norm_vec({0.0, 0.0, 0.0}, d), do: d
  defp norm_vec({x, y, z} = v, _), do: (n = :math.sqrt(x * x + y * y + z * z); if(n == 0, do: {0.0, 0.0, 1.0}, else: scale(v, 1 / n)))
  defp sub({a, b, c}, {d, e, f}), do: {a - d, b - e, c - f}
  defp add({a, b, c}, {d, e, f}), do: {a + d, b + e, c + f}
  defp scale({a, b, c}, s), do: {a * s, b * s, c * s}
  defp dist(a, b), do: (({x, y, z} = sub(a, b)); :math.sqrt(x * x + y * y + z * z))
  defp d2(a, b), do: (({x, y, z} = sub(a, b)); x * x + y * y + z * z)

  # ------------------------------------------------------------ integrals

  defp f0(t) when t < 1.0e-8, do: 1.0 - t / 3
  defp f0(t), do: 0.5 * :math.sqrt(:math.pi() / t) * :math.erf(:math.sqrt(t))

  defp basis(el, pos) do
    z = @zeta[el]
    Enum.zip_with(@alpha, @coef, fn a, d -> (aa = a * z * z; {aa, d * :math.pow(2 * aa / :math.pi(), 0.75), pos}) end)
  end

  defp gp({a, _, pa}, {b, _, pb}), do: scale(add(scale(pa, a), scale(pb, b)), 1 / (a + b))

  defp s_prim({a, _, pa} = x, {b, _, pb} = y), do: (_ = {x, y}; :math.pow(:math.pi() / (a + b), 1.5) * :math.exp(-a * b / (a + b) * d2(pa, pb)))

  defp t_prim({a, _, pa}, {b, _, pb}) do
    r2 = d2(pa, pb)
    a * b / (a + b) * (3 - 2 * a * b / (a + b) * r2) * :math.pow(:math.pi() / (a + b), 1.5) * :math.exp(-a * b / (a + b) * r2)
  end

  defp v_prim({a, _, pa} = x, {b, _, pb} = y, {zc, pc}) do
    p = gp(x, y)
    -2 * :math.pi() / (a + b) * zc * :math.exp(-a * b / (a + b) * d2(pa, pb)) * f0((a + b) * d2(p, pc))
  end

  defp eri({a, _, pa} = w, {b, _, pb} = x, {c, _, pc} = y, {d, _, pd} = z) do
    p = gp(w, x)
    q = gp(y, z)
    2 * :math.pow(:math.pi(), 2.5) / ((a + b) * (c + d) * :math.sqrt(a + b + c + d)) *
      :math.exp(-a * b / (a + b) * d2(pa, pb) - c * d / (c + d) * d2(pc, pd)) * f0((a + b) * (c + d) / (a + b + c + d) * d2(p, q))
  end

  defp c2(m, n, f), do: (for pa <- m, pb <- n, reduce: 0.0, do: (acc -> acc + elem(pa, 1) * elem(pb, 1) * f.(pa, pb)))

  defp c4(m, n, l, s) do
    for pa <- m, pb <- n, pc <- l, pd <- s, reduce: 0.0 do
      acc -> acc + elem(pa, 1) * elem(pb, 1) * elem(pc, 1) * elem(pd, 1) * eri(pa, pb, pc, pd)
    end
  end

  @doc "RHF energy and diagnostics for atoms [{element, {x, y, z}}] (bohr) and an even electron count."
  def rhf(atoms, n_el) do
    bs = Enum.map(atoms, fn {el, pos} -> basis(el, pos) end)
    nucs = Enum.map(atoms, fn {el, pos} -> {@z[el], pos} end)
    n = length(bs)
    ix = 0..(n - 1)
    sm = for i <- ix, do: for(j <- ix, do: c2(Enum.at(bs, i), Enum.at(bs, j), &s_prim/2))
    tm = for i <- ix, do: for(j <- ix, do: c2(Enum.at(bs, i), Enum.at(bs, j), &t_prim/2))
    vm = for i <- ix, do: for(j <- ix, do: Enum.reduce(nucs, 0.0, fn nuc, acc -> acc + c2(Enum.at(bs, i), Enum.at(bs, j), &v_prim(&1, &2, nuc)) end))
    h = madd(tm, vm)
    tw = for i <- ix, j <- ix, k <- ix, l <- ix, into: %{}, do: {{i, j, k, l}, c4(Enum.at(bs, i), Enum.at(bs, j), Enum.at(bs, k), Enum.at(bs, l))}
    enuc = for({{za, a}, i} <- Enum.with_index(nucs), {{zb, b}, j} <- Enum.with_index(nucs), i < j, do: za * zb / dist(a, b)) |> Enum.sum()
    {u, sv} = Vapor.Athanor.Strategy.jacobi(sm)
    x = mmul(u, mmul(diag(Enum.map(sv, &(1 / :math.sqrt(&1)))), transpose(u)))
    occ = div(n_el, 2)
    p0 = for(_ <- ix, do: for(_ <- ix, do: 0.0))

    {p, f, _e_el, eps, iters} =
      Enum.reduce_while(1..200, {p0, h, 0.0, [], 0}, fn it, {p, _f, e_old, _eps, _} ->
        g = for i <- ix, do: for(j <- ix, do: Enum.reduce(for(k <- ix, l <- ix, do: {k, l}), 0.0, fn {k, l}, acc -> acc + at(p, k, l) * (tw[{i, j, l, k}] - 0.5 * tw[{i, k, l, j}]) end))
        fm = madd(h, g)
        fp = mmul(transpose(x), mmul(fm, x))
        {cp, e} = Vapor.Athanor.Strategy.jacobi(fp)
        order = e |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
        c = mmul(x, cp)
        pn = for i <- ix, do: for(j <- ix, do: Enum.reduce(Enum.take(order, occ), 0.0, fn o, acc -> acc + 2 * at(c, i, o) * at(c, j, o) end))
        pn = if it > 1, do: madd(mscale(pn, 0.7), mscale(p, 0.3)), else: pn
        e_el = 0.5 * Enum.sum(for i <- ix, j <- ix, do: at(pn, j, i) * (at(h, i, j) + at(fm, i, j)))
        eps = Enum.map(order, &Enum.at(e, &1))
        if it > 3 and abs(e_el - e_old) < 1.0e-13, do: {:halt, {pn, fm, e_el, eps, it}}, else: {:cont, {pn, fm, e_el, eps, it}}
      end)

    # final Fock from the converged density for the diagnostics
    g = for i <- ix, do: for(j <- ix, do: Enum.reduce(for(k <- ix, l <- ix, do: {k, l}), 0.0, fn {k, l}, acc -> acc + at(p, k, l) * (tw[{i, j, l, k}] - 0.5 * tw[{i, k, l, j}]) end))
    fm = madd(h, g)
    _ = f
    e_el = 0.5 * Enum.sum(for i <- ix, j <- ix, do: at(p, j, i) * (at(h, i, j) + at(fm, i, j)))
    comm = msub(mmul(fm, mmul(p, sm)), mmul(sm, mmul(p, fm))) |> List.flatten() |> Enum.map(&abs/1) |> Enum.max()
    trace = Enum.sum(for i <- ix, j <- ix, do: at(p, i, j) * at(sm, j, i))
    kin = Enum.sum(for i <- ix, j <- ix, do: at(p, i, j) * at(tm, j, i))
    total = e_el + enuc

    %{energy: total, electronic: e_el, nuclear_repulsion: enuc, orbital_energies: eps, iterations: iters, commutator: comm, trace: trace,
      kinetic: kin, virial: -(total - kin) / kin}
  end

  defp at(m, i, j), do: m |> Enum.at(i) |> Enum.at(j)
  defp madd(a, b), do: Enum.zip_with(a, b, fn r, s -> Enum.zip_with(r, s, &+/2) end)
  defp msub(a, b), do: Enum.zip_with(a, b, fn r, s -> Enum.zip_with(r, s, &-/2) end)
  defp mscale(a, k), do: Enum.map(a, fn r -> Enum.map(r, &(&1 * k)) end)
  defp transpose(m), do: m |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
  defp mmul(a, b), do: (bt = transpose(b); Enum.map(a, fn r -> Enum.map(bt, fn c -> Enum.zip_with(r, c, &*/2) |> Enum.sum() end) end))
  defp diag(v), do: for({x, i} <- Enum.with_index(v), do: for(j <- 0..(length(v) - 1), do: if(i == j, do: x, else: 0.0)))
end
