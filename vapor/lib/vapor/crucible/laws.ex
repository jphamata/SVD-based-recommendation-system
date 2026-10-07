defmodule Vapor.Crucible.Laws do
  @moduledoc """
  Conservation laws of **any** dynamical system, discovered and — when the
  system is polynomial — **proved** (docs/CRUCIBLE.md §4).

  A quantity φ(x) is conserved by x' = f(x) exactly when its Lie
  derivative ∇φ·f vanishes identically. Take φ as an unknown combination
  of monomials up to a degree (and of ln xᵢ where fᵢ is divisible by xᵢ,
  which is how the Lotka–Volterra invariant appears); ∇φ·f is then a
  polynomial whose every coefficient is a linear form in the unknowns.
  Setting them all to zero is a linear system over **ℚ** — solved exactly,
  its null space is the space of conserved quantities, and each basis
  vector is a polynomial identity checked symbolically: a proof, not a
  fit. Products of laws already found are set aside, so what is reported
  is a set of generators.

  Two more checks travel with the answer. The laws are evaluated along a
  numerical solution (their drift should be at the integrator's error) —
  a test of the parser and the arithmetic, independent of the algebra.
  And the **control**: the same search on the system with each
  coefficient nudged by a different 1 % — a generic system has no
  conservation laws, so laws that survive the nudge are structural (like
  S + I + R), and the rest are the delicate ones.

  For systems that are not polynomial the same idea runs numerically
  (least squares over sampled points), and what it returns is labelled
  *verified at N random points*, never *proved*.
  """
  alias Vapor.{Expr, Solve}
  alias Vapor.Crucible.Poly

  @doc """
  `{:ok, result}` for an ODE system written as in the workbench (`x' = …`,
  parameters, initial values, `t = a .. b`). Options in the text:
  `degree = d` (default 2, at most 6).
  """
  def run(text) do
    {degree, text} = take_option(text, "degree", 2)
    {extra, text} = take_basis(text)
    text = ensure_defaults(text)

    with {:ok, sys} <- Solve.parse_ode(text) do
      degree = degree |> round() |> max(1) |> min(6)
      vars = sys.states
      params = sys.params
      raw = Enum.map(vars, &sys.rhs[&1])

      polys = Enum.map(raw, &Poly.from_expr(&1, vars, params))

      if extra == [] and Enum.all?(polys, &match?({:ok, _}, &1)) do
        fs = Enum.map(polys, &elem(&1, 1))
        exact(sys, vars, fs, degree)
      else
        numeric(sys, vars, degree, extra)
      end
    end
  end

  defp take_option(text, name, default) do
    case Regex.run(~r/^\s*#{name}\s*=\s*([\d.]+)\s*$/m, text) do
      [line, v] -> {elem(Float.parse(v), 0), String.replace(text, line, "")}
      nil -> {default, text}
    end
  end

  defp take_basis(text) do
    case Regex.run(~r/^\s*basis\s*=\s*\[(.*)\]\s*$/m, text) do
      [line, items] ->
        exprs = items |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))
        {exprs, String.replace(text, line, "")}
      nil -> {[], text}
    end
  end

  # laws need no span or initial values to be found; the numerical check does
  defp ensure_defaults(text) do
    states = Regex.scan(~r/^\s*([\p{L}_][\w]*)'\s*=/mu, text) |> Enum.map(&Enum.at(&1, 1))
    states = states ++ (Regex.scan(~r/^\s*d([\p{L}_][\w]*)\/dt\s*=/mu, text) |> Enum.map(&Enum.at(&1, 1)))
    inits = for s <- states, not Regex.match?(~r/^\s*#{Regex.escape(s)}\(\s*0\s*\)\s*=/m, text), do: "#{s}(0) = #{0.3 + 0.17 * :erlang.phash2(s, 7) / 7}"
    span = if Regex.match?(~r/^\s*t\s*=.*\.\./m, text), do: [], else: ["t = 0 .. 10"]
    Enum.join([text | inits ++ span], "\n")
  end

  # ================================================================ exact

  defp exact(sys, vars, fs, degree) do
    n = length(vars)
    {laws, basis_size} = generators(fs, n, degree)
    sol = solve_quiet(sys)

    laws_out =
      Enum.map(laws, fn l ->
        drift = drift(l, vars, sol)
        %{law: law_text(l, vars), degree: l.degree, status: "proved", proof: "∇φ·f ≡ 0 as a polynomial over ℚ (#{l.terms} terms cancel)",
          drift: drift.max, relative_drift: drift.rel, series: drift.series}
      end)

    control = control(fs, n, degree, vars)

    {:ok, %{kind: "laws", method: "exact (polynomial system over ℚ)", variables: vars, degree: degree, candidates: basis_size,
            vector_field: Enum.map(Enum.zip(vars, fs), fn {v, f} -> "#{v}' = #{Poly.text(f, vars)}" end),
            laws: laws_out, count: length(laws_out), control: control, t: sol && sol.t,
            says: says(laws_out, control, degree)}}
  end

  defp says([], control, d), do: "no conserved quantity of degree ≤ #{d} (ln terms included where they apply) — proved: the null space is empty" <> ctl(control)
  defp says(laws, control, _d), do: "#{length(laws)} independent conserved quantit#{if length(laws) == 1, do: "y", else: "ies"}, each proved by exact cancellation" <> ctl(control)

  defp ctl(%{count: c, survivors: s, generic: g}) do
    nudge = if c == 0, do: "none survive nudging every coefficient by a different 1 % (they depend on the exact values)", else: "#{c} survive nudging every coefficient by a different 1 % (#{Enum.join(s, ", ")}) — structural, not tied to the coefficients' values"
    gen = if g == 0, do: "the generic control (small generic terms added) has none: the method does not invent laws", else: "the generic control still shows #{g} — inspect them"
    "; " <> nudge <> "; " <> gen
  end

  # basis: monomials of degree 1..d, plus ln(x_i) when f_i/x_i is a polynomial
  defp basis(fs, n, d) do
    monos = Poly.monomials(n, d) |> Enum.map(&{:mono, &1})
    logs = for {f, i} <- Enum.with_index(fs), match?({:ok, _}, Poly.div_var(f, i)), do: {:log, i}
    monos ++ logs
  end

  defp lie({:mono, e}, fs, _n) do
    m = %{e => {1, 1}}
    Enum.with_index(fs) |> Enum.reduce(Poly.zero(), fn {f, i}, acc -> Poly.add(acc, Poly.mul(Poly.diff(m, i), f)) end)
  end

  defp lie({:log, i}, fs, _n), do: elem(Poly.div_var(Enum.at(fs, i), i), 1)

  defp null_laws(fs, n, d) do
    b = basis(fs, n, d)
    lies = Enum.map(b, &lie(&1, fs, n))
    rows_keys = lies |> Enum.flat_map(&Map.keys/1) |> Enum.uniq()
    rows = for k <- rows_keys, do: Enum.map(lies, &Map.get(&1, k, {0, 1}))
    vecs = if rows == [], do: identity_vecs(length(b)), else: Poly.null_space(rows, length(b))
    {b, vecs}
  end

  defp identity_vecs(k), do: for(i <- 0..(k - 1), do: for(j <- 0..(k - 1), do: if(i == j, do: 1, else: 0)))

  # generators degree by degree: keep a law only if it is not in the span of products of laws of lower degree
  defp generators(fs, n, degree) do
    {gens, size} =
      Enum.reduce(1..degree, {[], 0}, fn d, {gens, _} ->
        {b, vecs} = null_laws(fs, n, d)
        span = products(gens, b, n, d)
        new = Enum.reduce(vecs, {gens, span}, fn v, {gs, sp} ->
          if independent?(v, sp) do
            l = %{basis: b, coeffs: v, degree: d, terms: Enum.count(v, &(&1 != 0))}
            {gs ++ [l], sp ++ [v]}
          else
            {gs, sp}
          end
        end)
        {elem(new, 0), length(b)}
      end)

    {gens, size}
  end

  # the products of generators that stay within degree d, expressed in the basis b (log terms only alone)
  defp products(gens, b, n, d) do
    index = b |> Enum.with_index() |> Map.new()
    polys = for g <- gens, Enum.all?(Enum.zip(g.basis, g.coeffs), fn {e, c} -> c == 0 or match?({:mono, _}, e) end), do: {to_poly(g, n), g.degree}

    combos = monomial_products(polys, d)
    lifted = for g <- gens, do: lift(g, index, length(b))

    prods =
      for p <- combos do
        vec = List.duplicate(0, length(b))
        Enum.reduce(p, vec, fn {e, {num, den}}, acc ->
          case Map.fetch(index, {:mono, e}) do
            {:ok, i} -> if den == 1, do: List.update_at(acc, i, &(&1 + num)), else: throw(:frac)
            :error -> acc
          end
        end)
      end

    lifted ++ prods
  catch
    :frac -> for g <- gens, do: lift(g, Map.new(Enum.with_index(b)), length(b))
  end

  defp lift(g, index, size) do
    Enum.zip(g.basis, g.coeffs) |> Enum.reduce(List.duplicate(0, size), fn {e, c}, acc ->
      case Map.fetch(index, e) do {:ok, i} -> List.replace_at(acc, i, c); :error -> acc end
    end)
  end

  defp to_poly(g, _n) do
    Enum.zip(g.basis, g.coeffs) |> Enum.reduce(%{}, fn {_, 0}, acc -> acc; {{:mono, e}, c}, acc -> Map.put(acc, e, {c, 1}) end)
  end

  defp monomial_products(polys, d) do
    # every product of two or more generators (with repetition) whose degree ≤ d
    for {p, dp} <- polys, {q, dq} <- polys, dp + dq <= d, reduce: [] do
      acc ->
        pq = Poly.mul(p, q)
        triples = for {r, dr} <- polys, dp + dq + dr <= d, do: Poly.mul(pq, r)
        acc ++ [pq | triples]
    end
  end

  defp independent?(v, span), do: rank(span ++ [v], length(v)) > rank(span, length(v))

  defp rank([], _n), do: 0
  defp rank(rows, n), do: rows |> Enum.map(fn r -> Enum.map(r, &{&1, 1}) end) |> Poly.rref(n) |> elem(1) |> length()

  defp law_text(l, vars) do
    terms =
      Enum.zip(l.basis, l.coeffs)
      |> Enum.reject(fn {_, c} -> c == 0 end)
      |> Enum.map(fn
        {{:mono, e}, c} -> {c, e |> Tuple.to_list() |> Enum.zip(vars) |> Enum.filter(fn {k, _} -> k > 0 end) |> Enum.map_join("·", fn {1, v} -> v; {k, v} -> "#{v}^#{k}" end)}
        {{:log, i}, c} -> {c, "ln(#{Enum.at(vars, i)})"}
      end)

    terms
    |> Enum.with_index()
    |> Enum.map_join("", fn {{c, m}, i} ->
      coef = if abs(c) == 1, do: "", else: "#{abs(c)}·"
      sign = cond do i == 0 and c < 0 -> "−"; i == 0 -> ""; c < 0 -> " − "; true -> " + " end
      sign <> coef <> m
    end)
  end

  defp eval_law(l, xs) do
    Enum.zip(l.basis, l.coeffs)
    |> Enum.reduce(0.0, fn
      {_, 0}, s -> s
      {{:mono, e}, c}, s -> s + c * (e |> Tuple.to_list() |> Enum.zip(xs) |> Enum.reduce(1.0, fn {k, x}, m -> m * :math.pow(x, k) end))
      {{:log, i}, c}, s -> s + c * :math.log(max(Enum.at(xs, i), 1.0e-300))
    end)
  end

  defp solve_quiet(sys) do
    case Solve.integrate(sys) do
      {:ok, sol} -> sol
      _ -> nil
    end
  rescue
    _ -> nil
  end

  defp drift(_l, _vars, nil), do: %{max: nil, rel: nil, series: []}

  defp drift(l, vars, sol) do
    pts = Enum.map(0..(length(sol.t) - 1), fn k -> Enum.map(vars, &Enum.at(sol.series[&1], k)) end)
    vals =
      try do
        Enum.map(pts, &eval_law(l, &1))
      rescue
        _ -> []
      end
    case vals do
      [] -> %{max: nil, rel: nil, series: []}
      [v0 | _] ->
        mx = vals |> Enum.map(&abs(&1 - v0)) |> Enum.max()
        %{max: mx, rel: mx / max(abs(v0), 1.0e-300), series: Enum.take_every(vals, max(div(length(vals), 120), 1))}
    end
  end

  # ================================================================ control

  defp control(fs, n, degree, vars) do
    nudged =
      fs
      |> Enum.with_index()
      |> Enum.map(fn {f, i} ->
        f |> Enum.with_index() |> Map.new(fn {{e, c}, j} -> {e, Poly.qmul(c, {100 + 1 + rem(7 * i + 13 * j + 3, 17), 100})} end)
      end)

    {laws, _} = generators(nudged, n, degree)
    generic = fs |> Enum.with_index() |> Enum.map(fn {f, i} -> f |> Poly.add(Poly.scale(Poly.var(i, n), {i + 1, 97})) |> Poly.add(Poly.scale(Poly.var(rem(i + 1, n), n), {1, 89})) |> Poly.add(Poly.const({i + 2, 83}, n)) end)
    {glaws, _} = generators(generic, n, degree)
    %{count: length(laws), survivors: Enum.map(laws, &law_text(&1, vars)), generic: length(glaws),
      generic_text: "each fᵢ plus (i+1)/97·xᵢ + xᵢ₊₁/89 + (i+2)/83, a generic perturbation"}
  end

  # ================================================================ numeric

  defp numeric(sys, vars, degree, extra) do
    n = length(vars)
    monos = Poly.monomials(n, degree) |> Enum.map(&mono_expr(&1, vars))
    extras = Enum.flat_map(extra, fn e -> case Expr.parse(e) do {:ok, t} -> [{e, t}]; _ -> [] end end)
    basis = monos ++ extras
    grads = Enum.map(basis, fn {_, t} -> Enum.map(vars, &Expr.diff(t, &1)) end)
    f = Expr.compile(sys.rhs_list, ["t" | vars])
    sol = solve_quiet(sys)
    eval_f = fn x -> f.(List.to_tuple([0.0 | x])) |> to_list() end
    lie_row = fn x ->
      env = Map.new(Enum.zip(vars, x))
      fx = eval_f.(x)
      Enum.map(grads, fn g -> Enum.zip(g, fx) |> Enum.reduce(0.0, fn {gi, fi}, s -> s + Expr.eval(gi, env) * fi end) end)
    end

    rows =
      sample_points(sys, sol, vars, 500)
      |> Enum.flat_map(fn x -> try do [lie_row.(x)] rescue _ -> [] end end)
      |> Enum.reject(fn r -> Enum.any?(r, &(abs(&1) > 1.0e12)) end)

    if length(rows) < length(basis) + 5 do
      {:error, "could not evaluate the system at enough points"}
    else
      k = length(basis)
      ata = for i <- 0..(k - 1), do: for(j <- 0..(k - 1), do: Enum.reduce(rows, 0.0, fn r, s -> s + Enum.at(r, i) * Enum.at(r, j) end))
      {vecs, vals} = Vapor.Athanor.Strategy.jacobi(ata)
      top = Enum.max(Enum.map(vals, &abs/1))
      cands = for {lam, c} <- Enum.with_index(vals), abs(lam) <= 1.0e-12 * max(top, 1.0e-300), do: Enum.map(vecs, &Enum.at(&1, c))
      fresh = sample_points(sys, nil, vars, 1500) |> Enum.flat_map(fn x -> try do [lie_row.(x)] rescue _ -> [] end end)
      names = Enum.map(basis, &elem(&1, 0))

      laws =
        cands
        |> Enum.map(fn c ->
          c = normalize(c)
          worst = fresh |> Enum.map(fn r -> abs(Enum.zip(c, r) |> Enum.reduce(0.0, fn {a, b}, s -> s + a * b end)) end) |> Enum.max(fn -> 1.0 end)
          scale = fresh |> Enum.map(fn r -> Enum.zip(c, r) |> Enum.reduce(0.0, fn {a, b}, s -> s + abs(a * b) end) end) |> Enum.max(fn -> 1.0 end)
          text = Enum.zip(c, names) |> Enum.reject(fn {ci, _} -> abs(ci) < 1.0e-9 end) |> Enum.map_join(" + ", fn {ci, nm} -> "#{nice(ci)}·#{nm}" end)
          %{law: text, degree: degree, residual: worst / max(scale, 1.0e-300),
            status: if(worst <= 1.0e-8 * max(scale, 1.0e-300), do: "verified numerically at #{length(fresh)} fresh points (not a proof)", else: "rejected at fresh points")}
        end)
        |> Enum.filter(&String.starts_with?(&1.status, "verified"))

      {:ok, %{kind: "laws", method: "numeric (not a polynomial system" <> if(extras != [], do: ", or extra basis functions", else: "") <> ")", variables: vars, degree: degree,
              candidates: k, basis: names, laws: laws, count: length(laws), control: nil, t: sol && sol.t,
              says: if(laws == [], do: "no conserved combination of the basis found numerically (add functions with basis = [cos(q), …])", else: "#{length(laws)} candidate law(s), verified numerically at fresh points — not proved")}}
    end
  end

  defp nice(x) do
    r = Float.round(x, 6)
    if r == Float.round(r), do: trunc(r), else: r
  end

  defp mono_expr(e, vars) do
    factors = e |> Tuple.to_list() |> Enum.zip(vars) |> Enum.filter(fn {k, _} -> k > 0 end)
    t = factors |> Enum.map(fn {1, v} -> {:v, v}; {k, v} -> {:^, {:v, v}, {:n, k * 1.0}} end) |> Enum.reduce(fn a, b -> {:*, b, a} end)
    {mono_text(e, vars), t}
  end

  defp to_list(t) when is_tuple(t), do: Tuple.to_list(t)
  defp to_list(l), do: l

  defp normalize(c) do
    m = Enum.max_by(c, &abs/1)
    Enum.map(c, &(&1 / m))
  end

  defp mono_text(e, vars), do: e |> Tuple.to_list() |> Enum.zip(vars) |> Enum.filter(fn {k, _} -> k > 0 end) |> Enum.map_join("·", fn {1, v} -> v; {k, v} -> "#{v}^#{k}" end)

  defp sample_points(sys, sol, vars, n) do
    {lo, hi} =
      case sol do
        nil -> {Enum.map(vars, fn v -> sys.init[v] - 1.0 end), Enum.map(vars, fn v -> sys.init[v] + 1.0 end)}
        s ->
          {Enum.map(vars, fn v -> Enum.min(s.series[v]) - 0.5 end), Enum.map(vars, fn v -> Enum.max(s.series[v]) + 0.5 end)}
      end
    for k <- 1..n, do: Enum.zip_with([lo, hi, 0..(length(vars) - 1) |> Enum.to_list()], fn [a, b, j] -> a + (b - a) * Vapor.Alembic.Builtins.hash01([k, j, :laws]) end)
  end
end
