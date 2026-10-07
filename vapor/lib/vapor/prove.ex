defmodule Vapor.Prove do
  @moduledoc """
  Machine proof and discovery in geometry and topology, every claim with
  a certificate a reader can check without trusting the search
  (docs/MATEMATICA.md).

  **Geometry, by the algebraic method** (the family of Wu 1978 and of
  Gröbner-basis provers): a figure is a construction — free parameters,
  then points defined from earlier ones (midpoint, intersection of two
  lines, foot of a perpendicular, circumcenter, a rational point of the
  unit circle…). Every coordinate is then an exact **rational function**
  of the parameters, with integer coefficients, and a claim (collinear,
  parallel, perpendicular, equal lengths, concyclic, the same point) is a
  polynomial in the coordinates. The theorem holds for every figure in
  general position iff the claim's numerator is the **zero polynomial**
  (`prove/1`) — the certificate is that numerator, expanded, and the
  non-degeneracy conditions are the denominators that appeared; a
  construction that is 0/0 for every parameter value is reported as
  degenerate, never proved. A second check evaluates the construction in
  exact rationals at random integer points (`check/2`, Schwartz–Zippel: a
  non-zero polynomial of degree D vanishes at a random point of a set of
  size N with probability ≤ D/N) — independent of the polynomial algebra,
  not of the construction formulas, which both arithmetics share. False
  statements are the control: their numerators are not zero and the
  evaluation fails.

  **Conjecture and prove** (`discover/2`): every triple of points of a
  figure tested for collinearity and every quadruple for concyclicity,
  first in floating point, then each survivor proved — so the Euler line
  and the nine-point circle come out of a triangle without being asked.
  What learned geometry provers add (readable synthetic proofs, auxiliary
  constructions proposed by a language model) is not done here; the
  logic desk (`Vapor.Logic.check/2`, MCP `logic_check`) is where an
  outside proposer's candidate is checked.

  **Topology**: Betti numbers of simplicial complexes over GF(2) and over
  ℚ (`betti/2`) — the difference is torsion, which is how the torus and
  the Klein bottle are told apart; and persistent homology of point clouds
  (`persistence/2`, Vietoris–Rips over GF(2)) — a loop sampled with noise
  has one long-lived H₁ class, a blob has none.
  """

  # ============================================================ polynomials
  # %{exponent tuple => integer coefficient}; zero is %{}

  defp pc(0, _n), do: %{}
  defp pc(c, n), do: %{List.to_tuple(List.duplicate(0, n)) => c}
  defp pv(i, n), do: %{List.to_tuple(for(k <- 0..(n - 1), do: if(k == i, do: 1, else: 0))) => 1}

  defp padd(a, b), do: Map.merge(a, b, fn _, x, y -> x + y end) |> Map.reject(fn {_, c} -> c == 0 end)
  defp pneg(a), do: Map.new(a, fn {e, c} -> {e, -c} end)

  defp pmul(a, b) do
    for {ea, ca} <- a, {eb, cb} <- b, reduce: %{} do
      acc ->
        e = List.to_tuple(Enum.zip_with(Tuple.to_list(ea), Tuple.to_list(eb), &+/2))
        Map.update(acc, e, ca * cb, &(&1 + ca * cb))
    end
    |> Map.reject(fn {_, c} -> c == 0 end)
  end

  defp pdeg(a), do: a |> Map.keys() |> Enum.map(fn e -> e |> Tuple.to_list() |> Enum.sum() end) |> Enum.max(fn -> 0 end)

  # ================================================================= fields
  # Three fields with the same operations: symbolic rational functions,
  # exact rationals at a point, and floats at a point.

  defp f_add(:sym, {n1, d}, {n2, d}), do: {padd(n1, n2), d}
  defp f_add(:sym, {n1, d1}, {n2, d2}), do: {padd(pmul(n1, d2), pmul(n2, d1)), pmul(d1, d2)}
  defp f_add(:q, {a, b}, {c, d}), do: qnorm({a * d + c * b, b * d})
  defp f_add(:f, x, y), do: x + y

  defp f_neg(:sym, {n, d}), do: {pneg(n), d}
  defp f_neg(:q, {a, b}), do: {-a, b}
  defp f_neg(:f, x), do: -x

  defp f_sub(k, x, y), do: f_add(k, x, f_neg(k, y))

  defp f_mul(:sym, {n1, d1}, {n2, d2}), do: {pmul(n1, n2), pmul(d1, d2)}
  defp f_mul(:q, {a, b}, {c, d}), do: qnorm({a * c, b * d})
  defp f_mul(:f, x, y), do: x * y

  defp f_div(:sym, {n1, d1}, {n2, d2}), do: {pmul(n1, d2), pmul(d1, n2)}
  defp f_div(:q, {a, b}, {c, d}), do: qnorm({a * d, b * c})
  defp f_div(:f, x, y), do: x / y

  defp qnorm({_, 0}), do: throw(:degenerate)
  defp qnorm({a, b}) do
    g = Integer.gcd(a, b)
    {a, b} = {div(a, g), div(b, g)}
    if b < 0, do: {-a, -b}, else: {a, b}
  end

  defp f_const(:sym, c, n), do: {pc(c, n), pc(1, n)}
  defp f_const(:q, c, _), do: {c, 1}
  defp f_const(:f, c, _), do: c * 1.0

  defp f_zero?(:sym, {n, _}), do: n == %{}
  defp f_zero?(:q, {a, _}), do: a == 0
  defp f_zero?(:f, x), do: abs(x) < 1.0e-7

  # ========================================================== constructions

  @doc """
  The built-in theorems (true) and their controls (false statements of
  the same shape): `%{name => %{params, points, claim, true?: bool, doc}}`.
  Points: `{:coords, x, y}` with expressions over the parameters (integers,
  parameter atoms, `{:+ | :- | :* | :/, a, b}`), `{:midpoint, p, q}`,
  `{:intersect, a, b, c, d}` (lines ab and cd), `{:foot, p, a, b}`,
  `{:circumcenter, a, b, c}`, `{:circle, t}` (the unit circle's rational
  point), `{:on_line, a, b, t}`, `{:on_circle, center, through, t}` (the
  point of the circle about `center` through `through`, rotated by the
  rational angle parameter `t`).
  """
  def theorems do
    tri = [a: {:coords, 0, 0}, b: {:coords, :u, 0}, c: {:coords, :v, :w}]
    p3 = [:u, :v, :w]
    centroid = [ma: {:midpoint, :b, :c}, mb: {:midpoint, :c, :a}, mc: {:midpoint, :a, :b}, g: {:intersect, :a, :ma, :b, :mb}]
    ortho = [ha: {:foot, :a, :b, :c}, hb: {:foot, :b, :c, :a}, hc: {:foot, :c, :a, :b}, h: {:intersect, :a, :ha, :b, :hb}]
    circ = [o: {:circumcenter, :a, :b, :c}]
    unit = [a: {:circle, :p}, b: {:circle, :q}, c: {:circle, :r}, o: {:coords, 0, 0}]

    %{
      "medians_concurrent" => %{params: p3, points: tri ++ centroid, claim: {:collinear, :c, :g, :mc}, true?: true, doc: "the three medians meet (the centroid)"},
      "altitudes_concurrent" => %{params: p3, points: tri ++ ortho, claim: {:collinear, :c, :h, :hc}, true?: true, doc: "the three altitudes meet (the orthocenter)"},
      "bisectors_concurrent" => %{params: p3, points: tri ++ circ ++ [mc: {:midpoint, :a, :b}], claim: {:perpendicular, :o, :mc, :a, :b}, true?: true, doc: "the perpendicular bisectors meet (the circumcenter)"},
      "euler_line" => %{params: p3, points: tri ++ centroid ++ ortho ++ circ, claim: {:collinear, :o, :g, :h}, true?: true, doc: "circumcenter, centroid and orthocenter are collinear"},
      "euler_ratio" => %{params: p3, points: tri ++ centroid ++ ortho ++ circ ++ [m: {:on_ratio, :o, :h, 1, 3}], claim: {:same, :g, :m}, true?: true, doc: "the centroid divides OH as 1 : 2"},
      "nine_point_circle" => %{params: p3, points: tri ++ centroid ++ ortho ++ circ ++ [n: {:midpoint, :o, :h}], claim: {:equal_length, :n, :ma, :n, :ha}, true?: true, doc: "the nine-point center is as far from a side's midpoint as from an altitude's foot"},
      "nine_point_concyclic" => %{params: p3, points: tri ++ centroid ++ ortho, claim: {:concyclic, :ma, :mb, :ha, :hb}, true?: true, doc: "two midpoints and two feet of altitudes are concyclic"},
      "midline" => %{params: p3, points: tri ++ [mb: {:midpoint, :c, :a}, mc: {:midpoint, :a, :b}], claim: {:parallel, :mb, :mc, :b, :c}, true?: true, doc: "the segment joining two midpoints is parallel to the third side"},
      "varignon" => %{params: [:u, :v, :w, :s, :t], points: [a: {:coords, 0, 0}, b: {:coords, :u, 0}, c: {:coords, :v, :w}, d: {:coords, :s, :t},
                     p: {:midpoint, :a, :b}, q: {:midpoint, :b, :c}, r: {:midpoint, :c, :d}, s2: {:midpoint, :d, :a}], claim: {:parallel, :p, :q, :s2, :r}, true?: true,
                     doc: "the midpoints of any quadrilateral form a parallelogram"},
      "thales" => %{params: [:p, :q], points: [a: {:coords, -1, 0}, b: {:coords, 1, 0}, c: {:circle, :p}], claim: {:perpendicular, :c, :a, :c, :b}, true?: true, doc: "an angle inscribed in a semicircle is right"},
      "simson_line" => %{params: [:p, :q, :r, :t], points: unit ++ [d: {:circle, :t}, x: {:foot, :d, :b, :c}, y: {:foot, :d, :c, :a}, z: {:foot, :d, :a, :b}], claim: {:collinear, :x, :y, :z}, true?: true,
                         doc: "the feet of the perpendiculars from a point of the circumcircle to the sides are collinear (Simson)"},
      "pappus" => %{params: [:a1, :a2, :a3, :b1, :b2, :b3, :k],
                    points: [o1: {:coords, 0, 0}, o2: {:coords, 1, 0}, a: {:coords, :a1, 0}, b: {:coords, :a2, 0}, c: {:coords, :a3, 0},
                             d: {:coords, :b1, {:*, :k, :b1}}, e: {:coords, :b2, {:*, :k, :b2}}, f: {:coords, :b3, {:*, :k, :b3}},
                             x: {:intersect, :a, :e, :b, :d}, y: {:intersect, :a, :f, :c, :d}, z: {:intersect, :b, :f, :c, :e}], claim: {:collinear, :x, :y, :z}, true?: true,
                    doc: "Pappus: points on two lines, cross-joined, meet on a line"},
      # ---- the controls: false statements of the same shapes
      "false_centroid_on_circle" => %{params: p3, points: tri ++ centroid ++ circ, claim: {:equal_length, :o, :g, :o, :a}, true?: false, doc: "(false) the centroid lies on the circumcircle"},
      "false_orthocenter_is_circumcenter" => %{params: p3, points: tri ++ ortho ++ circ, claim: {:same, :h, :o}, true?: false, doc: "(false) the orthocenter is the circumcenter"},
      "false_midline_perpendicular" => %{params: p3, points: tri ++ [mb: {:midpoint, :c, :a}, mc: {:midpoint, :a, :b}], claim: {:perpendicular, :mb, :mc, :b, :c}, true?: false, doc: "(false) the midline is perpendicular to the third side"},
      "false_simson_off_circle" => %{params: [:p, :q, :r, :s, :t], points: unit ++ [d: {:coords, :s, :t}, x: {:foot, :d, :b, :c}, y: {:foot, :d, :c, :a}, z: {:foot, :d, :a, :b}], claim: {:collinear, :x, :y, :z}, true?: false,
                                     doc: "(false) Simson's line for a point anywhere — the hypothesis that the point is on the circle dropped"},
      "false_nine_point_vertex" => %{params: p3, points: tri ++ centroid ++ ortho ++ circ ++ [n: {:midpoint, :o, :h}], claim: {:equal_length, :n, :ma, :n, :a}, true?: false, doc: "(false) the nine-point circle passes through a vertex"}
    }
  end

  @doc """
  Prove a statement (a `theorems/0` name or map): `{:proved, cert}` with
  `cert = %{numerator_terms: 0, degree, nondegenerate: [polynomial
  text…]}`, or `{:refuted, %{numerator_terms, degree}}` — the claim's
  numerator is a non-zero polynomial: false in general position.
  """
  def prove(name) when is_binary(name), do: prove(Map.fetch!(theorems(), name))

  def prove(%{params: params, points: points, claim: claim}) do
    n = length(params)
    env = Map.new(Enum.with_index(params), fn {p, i} -> {p, {pv(i, n), pc(1, n)}} end)
    {pts, dens} = build(:sym, points, env, n)
    vals = claim_values(:sym, claim, pts, n)
    nums = Enum.map(vals, fn {num, _} -> num end)
    deg = nums |> Enum.map(&pdeg/1) |> Enum.max()
    terms = nums |> Enum.map(&map_size/1) |> Enum.sum()
    nondeg = dens |> Enum.uniq() |> Enum.reject(&(map_size(&1) <= 1)) |> Enum.map(&show(&1, params)) |> Enum.uniq() |> Enum.take(8)

    if terms == 0, do: {:proved, %{numerator_terms: 0, degree: deg, nondegenerate: nondeg}}, else: {:refuted, %{numerator_terms: terms, degree: deg}}
  catch
    :degenerate -> {:degenerate, :construction}
  end

  @doc """
  The independent check: the construction in exact rationals at `k`
  seeded random integer points in [−N, N]; `{:holds | :fails, points
  tested}`. A degenerate draw (a zero denominator) is skipped.
  """
  def check(name, opts \\ [])
  def check(name, opts) when is_binary(name), do: check(Map.fetch!(theorems(), name), opts)

  def check(%{params: params, points: points, claim: claim}, opts) do
    k = Keyword.get(opts, :points, 5)
    big = Keyword.get(opts, :range, 1_000_000_007)
    seed = Keyword.get(opts, :seed, 1)

    results =
      for t <- 1..(k * 3), reduce: [] do
        acc when length(acc) >= k ->
          acc

        acc ->
          env = Map.new(Enum.with_index(params), fn {p, i} -> {p, {trunc((Vapor.Sampler.uniform(seed * 131 + i, t) - 0.5) * 2 * big), 1}} end)

          try do
            {pts, _} = build(:q, points, env, length(params))
            [Enum.all?(claim_values(:q, claim, pts, 0), &f_zero?(:q, &1)) | acc]
          catch
            :degenerate -> acc
          end
      end

    {if(results != [] and Enum.all?(results), do: :holds, else: :fails), length(results)}
  end

  @doc "Coordinates of every point of a figure at given parameter values (floats), for drawing."
  def coordinates(%{params: params, points: points}, values) do
    env = Map.new(Enum.zip(params, values), fn {p, v} -> {p, v * 1.0} end)
    {pts, _} = build(:f, points, env, 0)
    Map.new(pts, fn {k, {x, y}} -> {k, {x, y}} end)
  catch
    :degenerate -> %{}
  end

  # builds every point; returns {points, denominators seen (symbolic only)}
  defp build(k, points, env, n) do
    Enum.reduce(points, {%{}, []}, fn {name, spec}, {pts, dens} ->
      {xy, ds} = point(k, spec, pts, env, n)
      {Map.put(pts, name, xy), ds ++ dens}
    end)
  end

  defp expr(k, c, _env, n) when is_integer(c), do: f_const(k, c, n)
  defp expr(_k, p, env, _n) when is_atom(p), do: Map.fetch!(env, p)
  defp expr(k, {op, a, b}, env, n) do
    {x, y} = {expr(k, a, env, n), expr(k, b, env, n)}
    case op do
      :+ -> f_add(k, x, y)
      :- -> f_sub(k, x, y)
      :* -> f_mul(k, x, y)
      :/ -> f_div(k, x, y)
    end
  end

  defp den(:sym, {_, d}), do: [d]
  defp den(_, _), do: []

  defp point(k, {:coords, x, y}, _pts, env, n), do: {{expr(k, x, env, n), expr(k, y, env, n)}, []}

  defp point(k, {:midpoint, p, q}, pts, _env, n) do
    {{x1, y1}, {x2, y2}} = {pts[p], pts[q]}
    two = f_const(k, 2, n)
    {{f_div(k, f_add(k, x1, x2), two), f_div(k, f_add(k, y1, y2), two)}, []}
  end

  defp point(k, {:on_ratio, p, q, a, b}, pts, _env, n) do
    {{x1, y1}, {x2, y2}} = {pts[p], pts[q]}
    t = f_div(k, f_const(k, a, n), f_const(k, b, n))
    {{f_add(k, x1, f_mul(k, t, f_sub(k, x2, x1))), f_add(k, y1, f_mul(k, t, f_sub(k, y2, y1)))}, []}
  end

  defp point(k, {:on_line, p, q, t}, pts, env, n) do
    {{x1, y1}, {x2, y2}} = {pts[p], pts[q]}
    tv = expr(k, t, env, n)
    {{f_add(k, x1, f_mul(k, tv, f_sub(k, x2, x1))), f_add(k, y1, f_mul(k, tv, f_sub(k, y2, y1)))}, []}
  end

  defp point(k, {:circle, t}, _pts, env, n) do
    tv = expr(k, t, env, n)
    one = f_const(k, 1, n)
    t2 = f_mul(k, tv, tv)
    d = f_add(k, one, t2)
    {{f_div(k, f_sub(k, one, t2), d), f_div(k, f_mul(k, f_const(k, 2, n), tv), d)}, den(k, d)}
  end

  defp point(k, {:intersect, a, b, c, d}, pts, _env, _n) do
    {{xa, ya}, {xb, yb}, {xc, yc}, {xd, yd}} = {pts[a], pts[b], pts[c], pts[d]}
    m = &f_mul(k, &1, &2)
    s = &f_sub(k, &1, &2)
    dd = s.(m.(s.(xa, xb), s.(yc, yd)), m.(s.(ya, yb), s.(xc, xd)))
    if f_zero?(k, dd), do: throw(:degenerate)
    e1 = s.(m.(xa, yb), m.(ya, xb))
    e2 = s.(m.(xc, yd), m.(yc, xd))
    px = f_div(k, s.(m.(e1, s.(xc, xd)), m.(s.(xa, xb), e2)), dd)
    py = f_div(k, s.(m.(e1, s.(yc, yd)), m.(s.(ya, yb), e2)), dd)
    {{px, py}, den(k, dd)}
  end

  defp point(k, {:foot, p, a, b}, pts, _env, _n) do
    {{xp, yp}, {xa, ya}, {xb, yb}} = {pts[p], pts[a], pts[b]}
    {dx, dy} = {f_sub(k, xb, xa), f_sub(k, yb, ya)}
    len2 = f_add(k, f_mul(k, dx, dx), f_mul(k, dy, dy))
    if f_zero?(k, len2), do: throw(:degenerate)
    t = f_div(k, f_add(k, f_mul(k, f_sub(k, xp, xa), dx), f_mul(k, f_sub(k, yp, ya), dy)), len2)
    {{f_add(k, xa, f_mul(k, t, dx)), f_add(k, ya, f_mul(k, t, dy))}, den(k, len2)}
  end

  defp point(k, {:circumcenter, a, b, c}, pts, _env, n) do
    {{ax, ay}, {bx, by}, {cx, cy}} = {pts[a], pts[b], pts[c]}
    m = &f_mul(k, &1, &2)
    s = &f_sub(k, &1, &2)
    ad = &f_add(k, &1, &2)
    sq = fn x, y -> ad.(m.(x, x), m.(y, y)) end
    d = m.(f_const(k, 2, n), ad.(ad.(m.(ax, s.(by, cy)), m.(bx, s.(cy, ay))), m.(cx, s.(ay, by))))
    if f_zero?(k, d), do: throw(:degenerate)
    {a2, b2, c2} = {sq.(ax, ay), sq.(bx, by), sq.(cx, cy)}
    ux = ad.(ad.(m.(a2, s.(by, cy)), m.(b2, s.(cy, ay))), m.(c2, s.(ay, by)))
    uy = ad.(ad.(m.(a2, s.(cx, bx)), m.(b2, s.(ax, cx))), m.(c2, s.(bx, ax)))
    {{f_div(k, ux, d), f_div(k, uy, d)}, den(k, d)}
  end

  # the values that must all vanish
  defp claim_values(k, claim, pts, n) do
    m = &f_mul(k, &1, &2)
    s = &f_sub(k, &1, &2)
    ad = &f_add(k, &1, &2)
    d2 = fn {x1, y1}, {x2, y2} -> ad.(m.(s.(x1, x2), s.(x1, x2)), m.(s.(y1, y2), s.(y1, y2))) end
    cross = fn {x1, y1}, {x2, y2}, {x3, y3}, {x4, y4} -> s.(m.(s.(x2, x1), s.(y4, y3)), m.(s.(y2, y1), s.(x4, x3))) end
    dot = fn {x1, y1}, {x2, y2}, {x3, y3}, {x4, y4} -> ad.(m.(s.(x2, x1), s.(x4, x3)), m.(s.(y2, y1), s.(y4, y3))) end

    case claim do
      {:collinear, a, b, c} -> [cross.(pts[a], pts[b], pts[a], pts[c])]
      {:parallel, a, b, c, d} -> [cross.(pts[a], pts[b], pts[c], pts[d])]
      {:perpendicular, a, b, c, d} -> [dot.(pts[a], pts[b], pts[c], pts[d])]
      {:equal_length, a, b, c, d} -> [s.(d2.(pts[a], pts[b]), d2.(pts[c], pts[d]))]
      {:same, a, b} -> [s.(elem(pts[a], 0), elem(pts[b], 0)), s.(elem(pts[a], 1), elem(pts[b], 1))]
      {:concyclic, a, b, c, d} -> [concyclic(k, Enum.map([a, b, c, d], &pts[&1]), n)]
    end
  end

  # det [[x²+y², x, y, 1] …] by cofactor expansion on the last column
  defp concyclic(k, rows, _n) do
    m = &f_mul(k, &1, &2)
    s = &f_sub(k, &1, &2)
    ad = &f_add(k, &1, &2)
    r = Enum.map(rows, fn {x, y} -> [ad.(m.(x, x), m.(y, y)), x, y] end)
    det3 = fn [[a, b, c], [d, e, f], [g, h, i]] -> ad.(s.(m.(a, s.(m.(e, i), m.(f, h))), m.(b, s.(m.(d, i), m.(f, g)))), m.(c, s.(m.(d, h), m.(e, g)))) end
    minors = for skip <- 0..3, do: det3.(List.delete_at(r, skip))
    [m0, m1, m2, m3] = minors
    # signs of the cofactors of column 4: -, +, -, +
    ad.(s.(m1, m0), s.(m3, m2))
  end

  @doc "A polynomial as text over the parameters."
  def show(p, params) do
    p
    |> Enum.sort_by(fn {e, _} -> e end, :desc)
    |> Enum.map_join(" + ", fn {e, c} ->
      mono = e |> Tuple.to_list() |> Enum.zip(params) |> Enum.reject(fn {x, _} -> x == 0 end) |> Enum.map_join("·", fn {1, v} -> "#{v}"; {x, v} -> "#{v}^#{x}" end)
      cond do
        mono == "" -> "#{c}"
        c == 1 -> mono
        true -> "#{c}·#{mono}"
      end
    end)
    |> String.replace("+ -", "- ")
  end

  # =============================================================== discovery

  @doc """
  Conjecture and prove, on a figure (`theorems/0` name or map): every
  triple of its points tested for collinearity and every quadruple for
  concyclicity at two random float instances; survivors that are not
  true by construction (three points of one defining line) are proved
  symbolically by `prove/1` and checked by `check/2` in exact rationals. `%{collinear: [...], concyclic: [...],
  candidates, survivors}`.
  """
  def discover(fig, opts \\ [])
  def discover(name, opts) when is_binary(name), do: discover(Map.fetch!(theorems(), name), opts)

  def discover(%{params: params, points: points} = fig, opts) do
    names = Keyword.get(opts, :only, Keyword.keys(points))
    inst = for s <- 1..2, do: coordinates(fig, for(i <- 0..(length(params) - 1), do: Vapor.Sampler.uniform(s * 97 + i, 1) * 4 - 2))
    by_def = definitional(points)

    col =
      for [a, b, c] <- combos(names, 3), not MapSet.member?(by_def, MapSet.new([a, b, c])),
          Enum.all?(inst, fn p -> collinear_f(p[a], p[b], p[c]) end), do: {:collinear, a, b, c}

    cyc =
      for [a, b, c, d] <- combos(names, 4),
          not Enum.any?(combos([a, b, c, d], 3), fn t -> Enum.all?(inst, fn p -> apply_t(p, t) end) end),
          Enum.all?(inst, fn p -> concyclic_f(p[a], p[b], p[c], p[d]) end), do: {:concyclic, a, b, c, d}

    # each survivor proved symbolically (the numerator is the zero polynomial) and checked in exact rationals
    proved = fn claims -> for c <- claims, match?({:proved, _}, prove(%{fig | claim: c})), check(%{fig | claim: c}, points: 3) |> elem(0) == :holds, do: c end
    %{collinear: proved.(col), concyclic: proved.(cyc), candidates: length(combos(names, 3)) + length(combos(names, 4)), survivors: length(col) + length(cyc)}
  end

  defp apply_t(p, [a, b, c]), do: collinear_f(p[a], p[b], p[c])

  # triples on one line by construction: every line through two base points
  # collects the points defined on it (midpoints, feet, intersections, ratios)
  defp definitional(points) do
    lines =
      for {name, spec} <- points, reduce: %{} do
        acc ->
          on = case spec do
            {:midpoint, p, q} -> [[p, q]]
            {:foot, _, a, b} -> [[a, b]]
            {:on_ratio, p, q, _, _} -> [[p, q]]
            {:on_line, p, q, _} -> [[p, q]]
            {:intersect, a, b, c, d} -> [[a, b], [c, d]]
            _ -> []
          end

          Enum.reduce(on, acc, fn [p, q], acc -> Map.update(acc, MapSet.new([p, q]), MapSet.new([p, q, name]), &MapSet.put(&1, name)) end)
      end

    lines |> Map.values() |> Enum.flat_map(fn set -> set |> MapSet.to_list() |> combos(3) |> Enum.map(&MapSet.new/1) end) |> MapSet.new()
  end

  defp combos(_, 0), do: [[]]
  defp combos([], _), do: []
  defp combos([h | t], k), do: Enum.map(combos(t, k - 1), &[h | &1]) ++ combos(t, k)

  defp collinear_f({x1, y1}, {x2, y2}, {x3, y3}) do
    abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) < 1.0e-7 * (1 + abs(x2 - x1) + abs(x3 - x1) + abs(y2 - y1) + abs(y3 - y1)) ** 2
  end

  defp collinear_f(_, _, _), do: false

  defp concyclic_f(p1, p2, p3, p4) do
    rows = Enum.map([p1, p2, p3, p4], fn {x, y} -> [x * x + y * y, x, y, 1.0] end)
    abs(det4(rows)) < 1.0e-7 * (1 + Enum.sum(for [a, _, _, _] <- rows, do: a)) ** 2
  end

  defp det4([r0 | rest]) do
    Enum.with_index(r0) |> Enum.reduce(0.0, fn {v, j}, acc ->
      minor = Enum.map(rest, &List.delete_at(&1, j))
      acc + (if rem(j, 2) == 0, do: 1, else: -1) * v * det3f(minor)
    end)
  end

  defp det3f([[a, b, c], [d, e, f], [g, h, i]]), do: a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

  # ================================================================ topology

  @doc """
  Simplicial complexes for the classics, as lists of top simplices (vertex
  index lists): `:sphere` (the boundary of a tetrahedron), `:torus` (3×3
  grid), `:klein` (the same grid, one side glued reversed), `:rp2` (six
  vertices), `:mobius`, `:disk`.
  """
  def complex(:sphere), do: [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
  def complex(:torus), do: grid_surface(fn i, j -> {rem(i, 3), rem(j, 3)} end)
  def complex(:klein), do: grid_surface(fn i, j -> if i == 3, do: {0, rem(3 - rem(j, 3), 3)}, else: {i, rem(j, 3)} end)
  def complex(:rp2), do: [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], [1, 2, 4], [2, 3, 5], [3, 4, 1], [4, 5, 2], [5, 1, 3]]
  def complex(:mobius), do: [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]]
  def complex(:disk), do: [[0, 1, 2], [0, 2, 3]]

  # a 3×3 grid of squares, each two triangles, with the vertex identification `id`
  defp grid_surface(id) do
    v = fn i, j -> {a, b} = id.(i, j); a * 3 + b end
    for i <- 0..2, j <- 0..2, t <- [[{i, j}, {i + 1, j}, {i + 1, j + 1}], [{i, j}, {i, j + 1}, {i + 1, j + 1}]], do: Enum.map(t, fn {a, b} -> v.(a, b) end)
  end

  @doc """
  Betti numbers b₀, b₁, b₂ of the complex generated by `tops`, over
  `:gf2` or `:q`, and the Euler characteristic: `%{betti, euler}`. Ranks
  of boundary matrices by exact elimination (bitsets for GF(2), fraction-
  free integers for ℚ).
  """
  def betti(tops, field \\ :gf2) do
    simplices = for d <- 0..2, into: %{}, do: {d, tops |> Enum.flat_map(&faces(Enum.sort(&1), d + 1)) |> Enum.uniq() |> Enum.sort()}
    count = Map.new(simplices, fn {d, s} -> {d, length(s)} end)
    r1 = rank(boundary(simplices[1], simplices[0]), field)
    r2 = rank(boundary(simplices[2], simplices[1]), field)
    b = [count[0] - r1, count[1] - r1 - r2, count[2] - r2]
    %{betti: b, euler: count[0] - count[1] + count[2], counts: [count[0], count[1], count[2]]}
  end

  defp faces(s, k), do: combos(s, k)

  # boundary rows: one per simplex, entries {column index, sign}
  defp boundary(higher, lower) do
    idx = lower |> Enum.with_index() |> Map.new()
    for s <- higher, do: for({f, i} <- Enum.with_index(faces_minus_one(s)), do: {idx[f], if(rem(i, 2) == 0, do: 1, else: -1)})
  end

  defp faces_minus_one(s), do: for(i <- 0..(length(s) - 1), do: List.delete_at(s, i))

  defp rank(rows, :gf2) do
    rows |> Enum.map(fn r -> Enum.reduce(r, 0, fn {c, _}, acc -> Bitwise.bxor(acc, Bitwise.bsl(1, c)) end) end) |> gf2_rank(0)
  end

  defp rank(rows, :q) do
    width = rows |> List.flatten() |> Enum.map(&elem(&1, 0)) |> Enum.max(fn -> 0 end)
    dense = for r <- rows, do: (m = Map.new(r); for(c <- 0..width, do: Map.get(m, c, 0)))
    bareiss_rank(dense)
  end

  defp gf2_rank([], r), do: r
  defp gf2_rank([0 | rest], r), do: gf2_rank(rest, r)

  defp gf2_rank([v | rest], r) do
    low = Bitwise.band(v, -v)
    gf2_rank(Enum.map(rest, fn w -> if Bitwise.band(w, low) != 0, do: Bitwise.bxor(w, v), else: w end), r + 1)
  end

  defp bareiss_rank([]), do: 0

  defp bareiss_rank(rows) do
    case Enum.find_index(rows, fn r -> Enum.any?(r, &(&1 != 0)) end) do
      nil -> 0
      _ ->
        {piv_row, piv_col} =
          rows |> Enum.with_index() |> Enum.find_value(fn {r, i} -> case Enum.find_index(r, &(&1 != 0)) do nil -> nil; c -> {i, c} end end)

        p = Enum.at(rows, piv_row)
        pv = Enum.at(p, piv_col)
        rest = List.delete_at(rows, piv_row)
        reduced = Enum.map(rest, fn r -> f = Enum.at(r, piv_col); Enum.zip_with(r, p, fn a, b -> a * pv - f * b end) |> content_divide() end)
        1 + bareiss_rank(Enum.reject(reduced, fn r -> Enum.all?(r, &(&1 == 0)) end))
    end
  end

  defp content_divide(r) do
    g = Enum.reduce(r, 0, &Integer.gcd/2)
    if g > 1, do: Enum.map(r, &div(&1, g)), else: r
  end

  @doc """
  Persistent homology of a point cloud (Vietoris–Rips up to triangles,
  GF(2)): `%{h0: [{birth, death}], h1: [{birth, death}]}` with deaths
  `:inf` for classes that never die. The standard column reduction.
  """
  def persistence(points, opts \\ []) do
    max_r = Keyword.get(opts, :max_radius, :infinity)
    n = length(points)
    pt = List.to_tuple(points)
    dist = fn i, j -> {x1, y1} = elem(pt, i); {x2, y2} = elem(pt, j); :math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) end
    edges = for(i <- 0..(n - 1), j <- (i + 1)..(n - 1)//1, d = dist.(i, j), max_r == :infinity or d <= max_r, do: {d, [i, j]}) |> Enum.sort()
    elen = Map.new(edges, fn {d, e} -> {e, d} end)

    tris =
      for i <- 0..(n - 1), j <- (i + 1)..(n - 1)//1, k <- (j + 1)..(n - 1)//1,
          Map.has_key?(elen, [i, j]) and Map.has_key?(elen, [i, k]) and Map.has_key?(elen, [j, k]),
          do: {Enum.max([elen[[i, j]], elen[[i, k]], elen[[j, k]]]), [i, j, k]}

    # filtration order: vertices (0), then edges and triangles by value, faces first
    simplices = for(v <- 0..(n - 1), do: {0.0, 0, [v]}) ++ Enum.map(edges, fn {d, e} -> {d, 1, e} end) ++ Enum.map(Enum.sort(tris), fn {d, t} -> {d, 2, t} end)
    simplices = Enum.sort_by(simplices, fn {d, dim, s} -> {d, dim, s} end)
    index = simplices |> Enum.with_index() |> Map.new(fn {{_, _, s}, i} -> {s, i} end)
    value = simplices |> Enum.with_index() |> Map.new(fn {{d, dim, _}, i} -> {i, {d, dim}} end)

    # columns as index lists sorted high to low; "low" is the highest index
    cols = for {_, dim, s} <- simplices, do: (if dim == 0, do: [], else: faces_minus_one(s) |> Enum.map(&index[&1]) |> Enum.sort(:desc))

    {pairs, _} =
      cols
      |> Enum.with_index()
      |> Enum.reduce({[], %{}}, fn {col, j}, {pairs, lows} ->
        col = reduce_col(col, lows)
        if col == [], do: {pairs, lows}, else: {[{hd(col), j} | pairs], Map.put(lows, hd(col), col)}
      end)

    killed = MapSet.new(pairs, &elem(&1, 0))
    dead = MapSet.new(pairs, &elem(&1, 1))
    intervals = for {b, d} <- pairs, {bv, bdim} = value[b], {dv, _} = value[d], dv > bv, do: {bdim, bv, dv}
    essential = for i <- 0..(length(simplices) - 1), not MapSet.member?(killed, i), not MapSet.member?(dead, i), {v, dim} = value[i], do: {dim, v, :inf}
    all = intervals ++ essential
    %{h0: for({0, b, d} <- all, do: {b, d}), h1: for({1, b, d} <- all, do: {b, d})}
  end

  defp reduce_col([], _lows), do: []

  defp reduce_col([low | _] = col, lows) do
    case Map.fetch(lows, low) do
      {:ok, other} -> reduce_col(symdiff(col, other), lows)
      :error -> col
    end
  end

  # symmetric difference of two lists sorted high to low
  defp symdiff([], b), do: b
  defp symdiff(a, []), do: a
  defp symdiff([x | a], [x | b]), do: symdiff(a, b)
  defp symdiff([x | a], [y | _] = b) when x > y, do: [x | symdiff(a, b)]
  defp symdiff(a, [y | b]), do: [y | symdiff(a, b)]

  # ================================================================= replay

  @doc false
  # recipes are data from an archive or a request: only named theorems and complexes
  def replay("prove.geometry", %{"name" => n}) when is_binary(n) do
    cond do
      not Map.has_key?(theorems(), n) -> {:error, {:unknown_theorem, n}}
      # Simson's symbolic proof takes ~100 s (no polynomial GCD): checked in exact rationals only, as in the console
      n =~ "simson" -> {:ok, %{"result" => "skipped_symbolic", "check" => inspect(check(n))}}
      true -> {:ok, %{"result" => inspect(prove(n)), "check" => inspect(check(n))}}
    end
  end

  def replay("prove.homology", %{"complex" => c}) when c in ~w(sphere torus klein rp2 mobius disk) do
    k = complex(String.to_existing_atom(c))
    {:ok, %{gf2: betti(k, :gf2), q: betti(k, :q)}}
  end
  def replay(_, _), do: {:error, :bad_recipe}
end
