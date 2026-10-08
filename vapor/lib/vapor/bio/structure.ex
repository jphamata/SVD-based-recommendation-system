defmodule Vapor.Bio.Structure do
  @moduledoc """
  Protein structure — measured, compared, folded from contacts, and the
  contacts read from evolution (docs/PROTEINS.md).

  **Comparison**, the metrics of CASP and of every structure-prediction
  paper: optimal superposition (Horn's quaternion solution of the Kabsch
  problem), RMSD, **TM-score** (Zhang & Skolnick 2004, the search of the
  TM-score program: seed fragments, iterative re-superposition on the
  residues within d₀), **GDT-TS** and the superposition-free **lDDT-Cα**
  (Mariani et al. 2013). TM-score is checked against TM-align
  (`tmtools`) in the tests.

  **Folding from contacts** (`fold/2`): distance geometry — bounds from
  the chain (Cα–Cα 3.8 Å), the contacts (< 8 Å) and the non-contacts
  (≥ 8 Å, when the map is complete), a classical multidimensional-scaling
  embedding of the shortest-path completed bounds, refinement by
  gradient descent on the bound violations, and the mirror image chosen
  by the handedness of the helices (α-helices are right-handed: the Cα
  virtual dihedral of a helical turn is near +50°). Contacts do not fix
  chirality; the helices do.

  **Contacts from evolution** (`Vapor.Bio.Coevolution`): mutual
  information with the average-product correction, and mean-field direct
  coupling analysis, on alignments sampled from a Potts model planted on
  a real protein's contact map — so the truth is known exactly.
  """
  alias Vapor.Dense

  @three %{"ALA" => "A", "ARG" => "R", "ASN" => "N", "ASP" => "D", "CYS" => "C", "GLN" => "Q", "GLU" => "E", "GLY" => "G", "HIS" => "H", "ILE" => "I",
           "LEU" => "L", "LYS" => "K", "MET" => "M", "PHE" => "F", "PRO" => "P", "SER" => "S", "THR" => "T", "TRP" => "W", "TYR" => "Y", "VAL" => "V", "MSE" => "M"}

  # ================================================================== PDB

  @doc """
  The Cα trace of a PDB file: `%{sequence, ca: [{x, y, z}], residues: [number], chain}`
  (first model, first chain unless `chain:` is given; alternate locations: the first).
  """
  def read_pdb(text, opts \\ []) do
    lines = String.split(text, "\n")
    first_model = Enum.take_while(lines, &(not String.starts_with?(&1, "ENDMDL")))
    atoms =
      for l <- first_model, String.starts_with?(l, "ATOM") or (String.starts_with?(l, "HETATM") and String.slice(l, 17, 3) == "MSE"),
          String.length(l) >= 54, String.trim(String.slice(l, 12, 4)) == "CA", String.slice(l, 16, 1) in [" ", "A"] do
        %{res: String.slice(l, 17, 3), chain: String.slice(l, 21, 1), num: String.slice(l, 22, 5) |> String.trim(),
          xyz: {f(String.slice(l, 30, 8)), f(String.slice(l, 38, 8)), f(String.slice(l, 46, 8))}}
      end

    chain = opts[:chain] || (case atoms do [a | _] -> a.chain; [] -> nil end)
    atoms = Enum.filter(atoms, &(&1.chain == chain)) |> Enum.uniq_by(& &1.num)
    if atoms == [], do: {:error, "no Cα atoms"}, else: {:ok, %{sequence: Enum.map_join(atoms, &Map.get(@three, &1.res, "X")), ca: Enum.map(atoms, & &1.xyz), residues: Enum.map(atoms, & &1.num), chain: chain}}
  end

  @doc "Every model of a multi-model (NMR) file, as Cα traces."
  def read_models(text) do
    text |> String.split(~r/^ENDMDL.*$/m) |> Enum.map(&read_pdb/1) |> Enum.filter(&match?({:ok, _}, &1)) |> Enum.map(&elem(&1, 1))
  end

  defp f(s), do: (case Float.parse(String.trim(s)) do {v, _} -> v; :error -> 0.0 end)

  @doc "A Cα trace as PDB text."
  def to_pdb(ca, sequence \\ nil) do
    one_to_three = Map.new(@three, fn {k, v} -> {v, k} end)
    body =
      ca |> Enum.with_index(1) |> Enum.map_join("", fn {{x, y, z}, i} ->
        res = if sequence, do: Map.get(one_to_three, String.at(sequence, i - 1), "GLY"), else: "GLY"
        :io_lib.format("ATOM  ~5w  CA  ~3s A~4w    ~8.3f~8.3f~8.3f  1.00  0.00           C~n", [i, res, i, x * 1.0, y * 1.0, z * 1.0]) |> to_string()
      end)
    body <> "END\n"
  end

  # ========================================================== superposition

  defp centroid(ps), do: (n = length(ps); {sx, sy, sz} = Enum.reduce(ps, {0.0, 0.0, 0.0}, fn {x, y, z}, {a, b, c} -> {a + x, b + y, c + z} end); {sx / n, sy / n, sz / n})
  defp sub({a, b, c}, {x, y, z}), do: {a - x, b - y, c - z}
  defp add({a, b, c}, {x, y, z}), do: {a + x, b + y, c + z}
  defp d2({a, b, c}, {x, y, z}), do: (a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2

  @doc """
  The rotation and translation that best superpose `mobile` on `target`
  (lists of points, paired): `{rot (3×3 rows), t}` with x ↦ R·x + t.
  Horn's quaternion method: the largest eigenvector of a 4×4 symmetric matrix.
  """
  def superpose(mobile, target) do
    cm = centroid(mobile)
    ct = centroid(target)
    {sxx, sxy, sxz, syx, syy, syz, szx, szy, szz} =
      Enum.zip(mobile, target) |> Enum.reduce({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, fn {p, q}, {a, b, c, d, e, f, g, h, i} ->
        {px, py, pz} = sub(p, cm)
        {qx, qy, qz} = sub(q, ct)
        {a + px * qx, b + px * qy, c + px * qz, d + py * qx, e + py * qy, f + py * qz, g + pz * qx, h + pz * qy, i + pz * qz}
      end)
    n = [[sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
         [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
         [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
         [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz]]
    {_vals, vecs} = Dense.eigh(n)
    [q0, q1, q2, q3] = List.last(vecs)
    r = [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
         [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
         [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]]
    rcm = apply_rot(r, cm)
    {r, sub(ct, rcm)}
  end

  defp apply_rot([[a, b, c], [d, e, f], [g, h, i]], {x, y, z}), do: {a * x + b * y + c * z, d * x + e * y + f * z, g * x + h * y + i * z}

  @doc "Apply `{rot, t}` to points."
  def transform(ps, {r, t}), do: Enum.map(ps, &add(apply_rot(r, &1), t))

  @doc "RMSD after optimal superposition."
  def rmsd(a, b) do
    moved = transform(a, superpose(a, b))
    :math.sqrt(Enum.zip_with(moved, b, &d2/2) |> Enum.sum() |> Kernel./(length(a)))
  end

  # ============================================================== TM-score

  @doc "The TM-score's d₀ for a target of length L."
  def d0(l) when l > 21, do: 1.24 * :math.pow(l - 15, 1 / 3) - 1.8
  def d0(_), do: 0.5

  @doc """
  TM-score of `model` against `native` (same residue correspondence),
  normalised by the native's length, with the superposition that
  maximises it: `%{tm, rmsd, gdt_ts, gdt_ha, superposition}`.
  """
  def tm_score(model, native) do
    l = length(native)
    d0 = d0(l)
    pairs = Enum.zip(model, native) |> List.to_tuple()
    n = tuple_size(pairs)
    lens = Stream.iterate(n, &div(&1, 2)) |> Enum.take_while(&(&1 >= 4)) |> Enum.take(6)
    seeds = for len <- lens, start <- seed_starts(n, len), do: Enum.to_list(start..(start + len - 1))

    {best_tm, best_sup, gdt} =
      Enum.reduce(seeds, {0.0, nil, %{1 => 0, 2 => 0, 4 => 0, 8 => 0, 0.5 => 0}}, fn idx, {bt, bs, gdt} ->
        {tm, sup, counts} = refine(pairs, idx, d0, l, 0)
        gdt = Map.merge(gdt, counts, fn _, a, b -> max(a, b) end)
        if tm > bt, do: {tm, sup, gdt}, else: {bt, bs, gdt}
      end)

    moved = transform(model, best_sup)
    rmsd = :math.sqrt(Enum.zip_with(moved, native, &d2/2) |> Enum.sum() |> Kernel./(l))
    %{tm: best_tm, rmsd_at_tm: rmsd, rmsd: rmsd(model, native), d0: d0,
      gdt_ts: (gdt[1] + gdt[2] + gdt[4] + gdt[8]) / (4 * l), gdt_ha: (gdt[0.5] + gdt[1] + gdt[2] + gdt[4]) / (4 * l), superposition: best_sup}
  end

  defp seed_starts(n, len) do
    step = max(div(len, 2), 1)
    Enum.uniq(Enum.to_list(0..(n - len)//step) ++ [n - len])
  end

  # superpose on idx, score, re-select the residues within d0 (and GDT cutoffs), repeat
  defp refine(pairs, idx, d0, l, iter) do
    ps = Enum.map(idx, &elem(pairs, &1))
    sup = superpose(Enum.map(ps, &elem(&1, 0)), Enum.map(ps, &elem(&1, 1)))
    n = tuple_size(pairs)
    ds = for i <- 0..(n - 1), do: (({m, t} = elem(pairs, i)); :math.sqrt(d2(hd(transform([m], sup)), t)))
    tm = Enum.reduce(ds, 0.0, fn d, acc -> acc + 1 / (1 + (d / d0) ** 2) end) / l
    counts = for c <- [0.5, 1, 2, 4, 8], into: %{}, do: {c, Enum.count(ds, &(&1 <= c))}
    cutoff = max(d0 + 1.0, 4.5)
    next = for {d, i} <- Enum.with_index(ds), d <= cutoff, do: i
    next = if length(next) < 3, do: idx, else: next
    if next == idx or iter >= 20 do
      {tm, sup, counts}
    else
      {tm2, sup2, c2} = refine(pairs, next, d0, l, iter + 1)
      c = Map.merge(counts, c2, fn _, a, b -> max(a, b) end)
      if tm2 >= tm, do: {tm2, sup2, c}, else: {tm, sup, c}
    end
  end

  # ================================================================== lDDT

  @doc "lDDT on Cα (superposition-free): pairs within 15 Å in the reference, fraction of distances preserved within 0.5/1/2/4 Å."
  def lddt(model, ref, radius \\ 15.0) do
    m = List.to_tuple(model)
    r = List.to_tuple(ref)
    n = tuple_size(r)
    pairs = for i <- 0..(n - 1), j <- 0..(n - 1), i != j, (dr = :math.sqrt(d2(elem(r, i), elem(r, j)))) < radius, do: {dr, :math.sqrt(d2(elem(m, i), elem(m, j)))}
    if pairs == [], do: 0.0, else: (Enum.sum(for t <- [0.5, 1.0, 2.0, 4.0], do: Enum.count(pairs, fn {a, b} -> abs(a - b) < t end) / length(pairs)) / 4)
  end

  # ============================================================ contacts

  @doc "Cα contacts (< `cut` Å) with sequence separation ≥ `min_sep`."
  def contacts(ca, cut \\ 8.0, min_sep \\ 6) do
    t = List.to_tuple(ca)
    n = tuple_size(t)
    for i <- 0..(n - 1), j <- (i + min_sep)..(n - 1)//1, d2(elem(t, i), elem(t, j)) < cut * cut, do: {i, j}
  end

  @doc "Precision of the top-k predicted pairs against true contacts."
  def precision(predicted, truth, k) do
    t = MapSet.new(truth)
    top = Enum.take(predicted, k)
    if top == [], do: 0.0, else: Enum.count(top, &MapSet.member?(t, &1)) / length(top)
  end

  @doc """
  Secondary structure from Cα geometry alone (a P-SEA-like rule): helix
  where i→i+3 is 4.6–5.6 Å and the virtual dihedral is +30…+70°; strand
  where i→i+2 ≥ 6.2 Å and the dihedral is −170…−120° or beyond 160°.
  Returns a string of H, E and -.
  """
  def secondary(ca) do
    t = List.to_tuple(ca)
    n = tuple_size(t)
    raw =
      for i <- 0..(n - 1) do
        dh = if i + 3 < n, do: dihedral(elem(t, i), elem(t, i + 1), elem(t, i + 2), elem(t, i + 3)), else: nil
        d3 = if i + 3 < n, do: :math.sqrt(d2(elem(t, i), elem(t, i + 3))), else: nil
        d2_ = if i + 2 < n, do: :math.sqrt(d2(elem(t, i), elem(t, i + 2))), else: nil
        cond do
          dh != nil and dh > 30 and dh < 75 and d3 > 4.4 and d3 < 5.8 -> "H"
          dh != nil and d2_ > 6.0 and (dh < -110 or dh > 160) -> "E"
          true -> "-"
        end
      end
    # a helix covers the four residues of each helical turn; runs shorter than 3 are dropped
    hs = for {"H", i} <- Enum.with_index(raw), k <- i..min(i + 3, n - 1), into: MapSet.new(), do: k
    raw |> Enum.with_index() |> Enum.map(fn {c, i} -> if MapSet.member?(hs, i), do: "H", else: (if c == "E", do: "E", else: "-") end) |> Enum.join()
  end

  @doc "The dihedral angle (degrees) of four points."
  def dihedral(p0, p1, p2, p3) do
    b0 = sub(p0, p1); b1 = sub(p2, p1); b2 = sub(p3, p2)
    nb1 = :math.sqrt(dot3(b1, b1))
    b1n = scale(b1, 1 / nb1)
    v = sub(b0, scale(b1n, dot3(b0, b1n)))
    w = sub(b2, scale(b1n, dot3(b2, b1n)))
    x = dot3(v, w)
    y = dot3(cross(b1n, v), w)
    :math.atan2(y, x) * 180 / :math.pi()
  end

  defp dot3({a, b, c}, {x, y, z}), do: a * x + b * y + c * z
  defp scale({a, b, c}, s), do: {a * s, b * s, c * s}
  defp cross({a, b, c}, {x, y, z}), do: {b * z - c * y, c * x - a * z, a * y - b * x}

  # ================================================================ folding

  @doc """
  Fold a chain of `n` residues from contacts (`[{i, j}]`). Options:
  `complete: true` (the map lists every contact: non-contacts get the
  ≥ 8 Å bound), `helices:` a secondary-structure string (for chirality),
  `restarts:` (3), `seed:`. Returns `%{ca, violation, mirrored}`.
  """
  def fold(n, contacts, opts \\ []) do
    complete = Keyword.get(opts, :complete, false)
    seed = Keyword.get(opts, :seed, 1)
    cset = MapSet.new(Enum.flat_map(contacts, fn {i, j} -> [{i, j}, {j, i}] end))
    min_sep = contacts |> Enum.map(fn {i, j} -> abs(j - i) end) |> Enum.min(fn -> 6 end)
    ss = opts[:helices]
    helix? = fn i, j -> ss != nil and String.slice(ss, i, j - i + 1) |> String.graphemes() |> Enum.all?(&(&1 == "H")) end
    bounds =
      for i <- 0..(n - 1), j <- (i + 1)..(n - 1)//1, into: %{} do
        b = cond do
          j == i + 1 -> {3.8, 3.8}
          # α-helix geometry (Cα i→i+2 5.4, i→i+3 5.0, i→i+4 6.2 Å) where the secondary structure says helix
          j - i in 2..4 and helix?.(i, j) -> elem({nil, nil, {5.2, 5.7}, {4.8, 5.4}, {5.9, 6.5}}, j - i)
          j == i + 2 -> {5.0, 7.3}
          MapSet.member?(cset, {i, j}) -> {4.0, 8.0}
          complete and j - i >= min_sep -> {8.0, 3.8 * (j - i)}
          # a predicted map is incomplete: an unlisted pair is probably apart — a soft bound (weight 0.05)
          not complete and Keyword.get(opts, :soft_apart, true) and j - i >= 6 -> {8.0, 3.8 * (j - i), 0.05}
          true -> {4.0, 3.8 * (j - i)}
        end
        {{i, j}, b}
      end

    upper = shortest_upper(n, bounds)
    runs =
      Vapor.Play.pmap(Enum.to_list(1..Keyword.get(opts, :restarts, 3)), fn r ->
        x0 = embed(n, upper, bounds, seed * 97 + r)
        x = refine_coords(x0, bounds, n, 1500)
        {x, violation(x, bounds)}
      end)
    {best, viol} = Enum.min_by(runs, &elem(&1, 1))
    {best, mirrored} = chirality(best, opts[:helices])
    %{ca: best, violation: viol, mirrored: mirrored, bounds: map_size(bounds)}
  end

  # Floyd–Warshall on the upper bounds (triangle inequality)
  defp shortest_upper(n, bounds) do
    m = for i <- 0..(n - 1), into: %{}, do: {i, for(j <- 0..(n - 1), into: %{}, do: {j, cond do i == j -> 0.0; i < j -> elem(bounds[{i, j}], 1); true -> elem(bounds[{j, i}], 1) end})}
    Enum.reduce(0..(n - 1), m, fn k, m ->
      mk = m[k]
      Map.new(m, fn {i, row} ->
        dik = row[k]
        {i, Map.new(row, fn {j, dij} -> {j, min(dij, dik + mk[j])} end)}
      end)
    end)
  end

  # metric matrix from distances (random between bounds, the upper ones triangle-smoothed), top-3 eigenvectors by power iteration
  defp embed(n, upper, bounds, seed) do
    d = for i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: (cond do
      i == j -> 0.0
      true ->
        {a, b} = if i < j, do: {i, j}, else: {j, i}
        lo = elem(bounds[{a, b}], 0) |> min(upper[a][b])
        hi = max(upper[a][b], lo)
        lo + (hi - lo) * Vapor.Sampler.uniform(seed, a * 1000 + b)
    end))
    d2m = Enum.map(d, fn r -> Enum.map(r, &(&1 * &1)) end)
    rm = Enum.map(d2m, &(Enum.sum(&1) / n))
    tot = Enum.sum(rm) / n
    g = for {r, i} <- Enum.with_index(d2m), do: for({x, j} <- Enum.with_index(r), do: -0.5 * (x - Enum.at(rm, i) - Enum.at(rm, j) + tot))
    vecs = top_eigen(g, 3, seed)
    for i <- 0..(n - 1), do: List.to_tuple(Enum.map(vecs, fn {lam, v} -> Enum.at(v, i) * :math.sqrt(max(lam, 1.0e-6)) end))
  end

  defp top_eigen(g, k, seed) do
    n = length(g)
    Enum.reduce(1..k, {g, []}, fn kk, {m, acc} ->
      v0 = for i <- 0..(n - 1), do: Vapor.Sampler.uniform(seed + kk, i) - 0.5
      v = Enum.reduce(1..300, normalize(v0), fn _, v -> normalize(Dense.matvec(m, v)) end)
      lam = Dense.dot(v, Dense.matvec(m, v))
      deflated = for {r, i} <- Enum.with_index(m), do: for({x, j} <- Enum.with_index(r), do: x - lam * Enum.at(v, i) * Enum.at(v, j))
      {deflated, acc ++ [{lam, v}]}
    end)
    |> elem(1)
  end

  defp normalize(v), do: (s = :math.sqrt(Dense.dot(v, v)); Enum.map(v, &(&1 / max(s, 1.0e-300))))

  @doc false
  def violation(x, bounds) do
    t = List.to_tuple(x)
    Enum.reduce(bounds, 0.0, fn {{i, j}, b}, acc ->
      {lo, hi, w} = bw(b)
      d = :math.sqrt(d2(elem(t, i), elem(t, j)))
      cond do d < lo -> acc + w * (lo - d) ** 2; d > hi -> acc + w * (d - hi) ** 2; true -> acc end
    end)
  end

  defp bw({lo, hi}), do: {lo, hi, 1.0}
  defp bw({_, _, _} = b), do: b

  # gradient descent with momentum on the violations (bonds weighted 10×)
  defp refine_coords(x, bounds, n, iters) do
    bl = Map.to_list(bounds)
    {x, _} =
      Enum.reduce(1..iters, {List.to_tuple(x), Tuple.duplicate({0.0, 0.0, 0.0}, n)}, fn it, {x, vel} ->
        g = Enum.reduce(bl, Tuple.duplicate({0.0, 0.0, 0.0}, n), fn {{i, j}, b}, g ->
          {lo, hi, bwt} = bw(b)
          {pi, pj} = {elem(x, i), elem(x, j)}
          diff = sub(pi, pj)
          d = :math.sqrt(max(dot3(diff, diff), 1.0e-12))
          w = if j == i + 1, do: 10.0, else: bwt
          e = cond do d < lo -> d - lo; d > hi -> d - hi; true -> 0.0 end
          if e == 0.0, do: g, else: (
            f = scale(diff, 2 * w * e / d)
            g |> put_elem(i, add(elem(g, i), f)) |> put_elem(j, sub(elem(g, j), f)))
        end)
        lr = if it < iters / 2, do: 0.01, else: 0.004
        vel = for k <- 0..(n - 1), do: sub(scale(elem(vel, k), 0.8), scale(elem(g, k), lr))
        vel = List.to_tuple(vel)
        {List.to_tuple(for k <- 0..(n - 1), do: add(elem(x, k), elem(vel, k))), vel}
      end)
    Tuple.to_list(x)
  end

  # helices are right-handed: positive virtual dihedrals over helical turns
  defp chirality(ca, nil), do: {ca, false}
  defp chirality(ca, ss) do
    t = List.to_tuple(ca)
    n = tuple_size(t)
    s = for i <- 0..(n - 4), String.slice(ss, i, 4) == "HHHH", reduce: 0.0, do: (acc -> acc + dihedral(elem(t, i), elem(t, i + 1), elem(t, i + 2), elem(t, i + 3)))
    if s < 0, do: {Enum.map(ca, fn {x, y, z} -> {-x, y, z} end), true}, else: {ca, false}
  end
end
