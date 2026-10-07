defmodule Vapor.Solve.PDE do
  @moduledoc """
  Partial differential equations typed in as text, with **code
  verification by manufactured solutions** (docs/BANCADA.md §5).

  Three families, each the workhorse of a field:

    * **parabolic in one dimension** — `u_t = a(x,t)·u_xx + b(x,t)·u_x + R(x,t,u)`:
      heat and mass diffusion, consolidation of soils (Terzaghi), the
      cable equation, Black–Scholes in log-price, reaction–diffusion
      (Fisher–KPP). Crank–Nicolson for the linear part (a tridiagonal
      solve per step); a reaction nonlinear in `u` goes explicitly by
      Adams–Bashforth 2, so the scheme stays second order. Dirichlet
      (`u(0,t) = …`) or Neumann (`u_x(1,t) = …`, by a ghost node) at
      each end. `scheme = implicit` is backward Euler (first order).
    * **hyperbolic in one dimension** — `u_tt = a(x,t)·u_xx + s(x,t)`:
      strings, rods, acoustics, by leapfrog; refused, with the step
      count that would satisfy it, when the CFL condition fails.
    * **elliptic in two dimensions** — `poisson -(u_xx + u_yy) + c·u = f`
      on a rectangle with Dirichlet data: electrostatics, steady heat,
      torsion (Prandtl's stress function), groundwater; five-point
      stencil, conjugate gradients.

  With `exact u = …` the error against it is reported; with `verify`
  too, the solver **manufactures** a problem whose solution is that
  expression — the source term is derived symbolically, the initial and
  boundary data taken from it — runs it on three grids, and reports the
  **observed order of accuracy**. A scheme that claims second order and
  shows first has a bug; this is the check ASME V&V 10/20 and Roache's
  method of manufactured solutions ask of scientific codes, and it runs
  on whatever equation the user typed.
  """
  alias Vapor.{Dense, Expr, Solve}

  @doc "Solve a PDE given as text. `{:ok, result}` or `{:error, why}`."
  def run(text) do
    with {:ok, spec} <- parse(text) do
      case spec.kind do
        :poisson -> with_verify(spec, &poisson/1)
        :parabolic -> with_verify(spec, &parabolic/1)
        :wave -> with_verify(spec, &wave/1)
      end
    end
  end

  # ================================================================ parsing

  @id "[\\p{L}_][\\p{L}\\p{N}_]*"

  defp parse(text) do
    st = Solve.statements(text)
    base = %{kind: nil, rhs: nil, params: %{}, x: nil, y: nil, t: nil, ic: nil, ic_t: nil, bc: %{}, nx: 101, ny: 41, nt: 200, exact: nil, verify: false,
             scheme: "cn", c: {:n, 0.0}, f: nil, boundary: nil, snapshots: 60}

    Enum.reduce_while(st, {:ok, base}, fn {s, line}, {:ok, acc} ->
      case pline(s, acc) do
        {:ok, acc} -> {:cont, {:ok, acc}}
        {:error, why} -> {:halt, {:error, "line #{line}: #{why}"}}
      end
    end)
    |> case do
      {:ok, %{kind: nil}} -> {:error, "write the equation: u_t = …, u_tt = …, or poisson -(u_xx + u_yy) = f"}
      {:ok, acc} -> finish(acc)
      e -> e
    end
  end

  defp pline(s, acc) do
    cond do
      m = Regex.run(~r/^u_t\s*=\s*(.+)$/u, s) -> eq(acc, :parabolic, Enum.at(m, 1))
      m = Regex.run(~r/^u_tt\s*=\s*(.+)$/u, s) -> eq(acc, :wave, Enum.at(m, 1))
      m = Regex.run(~r/^poisson\s+-\s*\(\s*u_xx\s*\+\s*u_yy\s*\)\s*(?:\+\s*(.+?)\s*\*?\s*u)?\s*=\s*(.+)$/iu, s) ->
        with {:ok, f} <- Expr.parse(Enum.at(m, 2)), {:ok, c} <- (if Enum.at(m, 1) in [nil, ""], do: {:ok, {:n, 0.0}}, else: Expr.parse(Enum.at(m, 1))) do
          {:ok, %{acc | kind: :poisson, f: f, c: c}}
        end
      m = Regex.run(~r/^poisson\b/iu, s) -> (_ = m; {:error, "write: poisson -(u_xx + u_yy) = f  (or  + c*u  before the =)"})
      m = Regex.run(~r/^(x|y|t)\s*=\s*(.+?)\s*\.\.\s*(.+)$/u, s) ->
        with {:ok, a} <- val(Enum.at(m, 2), acc), {:ok, b} <- val(Enum.at(m, 3), acc) do
          if b <= a, do: {:error, "#{Enum.at(m, 1)} = a .. b needs b > a"}, else: {:ok, Map.put(acc, String.to_atom(Enum.at(m, 1)), {a, b})}
        end
      m = Regex.run(~r/^u\s*\(\s*x\s*,\s*0\s*\)\s*=\s*(.+)$/u, s) -> tree(acc, :ic, Enum.at(m, 1))
      m = Regex.run(~r/^u_t\s*\(\s*x\s*,\s*0\s*\)\s*=\s*(.+)$/u, s) -> tree(acc, :ic_t, Enum.at(m, 1))
      m = Regex.run(~r/^u(_x)?\s*\(\s*([^,]+?)\s*,\s*t\s*\)\s*=\s*(.+)$/u, s) ->
        with {:ok, at} <- val(Enum.at(m, 2), acc), {:ok, g} <- Expr.parse(Enum.at(m, 3)) do
          {:ok, %{acc | bc: Map.put(acc.bc, at, {if(Enum.at(m, 1) == "_x", do: :neumann, else: :dirichlet), g})}}
        end
      m = Regex.run(~r/^(?:boundary|contorno)\s+u\s*=\s*(.+)$/iu, s) -> tree(acc, :boundary, Enum.at(m, 1))
      m = Regex.run(~r/^u\s*=\s*(.+?)\s+on\s+(?:the\s+)?boundary$/iu, s) -> tree(acc, :boundary, Enum.at(m, 1))
      m = Regex.run(~r/^(?:exact|exata)\s+u\s*=\s*(.+)$/iu, s) -> tree(acc, :exact, Enum.at(m, 1))
      s =~ ~r/^(verify|verificar|mms)$/iu -> {:ok, %{acc | verify: true}}
      m = Regex.run(~r/^(nx|ny|nt|snapshots)\s*=\s*(\d+)$/u, s) -> {:ok, Map.put(acc, String.to_atom(Enum.at(m, 1)), String.to_integer(Enum.at(m, 2)))}
      m = Regex.run(~r/^scheme\s*=\s*(cn|crank-nicolson|implicit|euler)$/iu, s) -> {:ok, %{acc | scheme: if(Enum.at(m, 1) =~ ~r/^(cn|crank)/i, do: "cn", else: "implicit")}}
      m = Regex.run(~r/^(#{@id})\s*=\s*(.+)$/u, s) -> with({:ok, v} <- val(Enum.at(m, 2), acc), do: {:ok, %{acc | params: Map.put(acc.params, Enum.at(m, 1), v)}})
      true -> {:error, "not understood: #{inspect(s)}"}
    end
  end

  defp eq(acc, kind, rhs), do: with({:ok, t} <- Expr.parse(rhs), do: {:ok, %{acc | kind: kind, rhs: t}})
  defp tree(acc, k, s), do: with({:ok, t} <- Expr.parse(s), do: {:ok, Map.put(acc, k, t)})

  defp val(s, acc) do
    with {:ok, t} <- Expr.parse(s) do
      case Expr.vars(t) -- Map.keys(acc.params) do
        [] -> (try do {:ok, Expr.eval(t, acc.params)} rescue _ -> {:error, "cannot evaluate #{s}"} end)
        m -> {:error, "unknown name(s) #{Enum.join(m, ", ")}"}
      end
    end
  end

  defp finish(acc) do
    sub = Map.new(acc.params, fn {k, v} -> {k, {:n, v}} end)
    s = fn nil -> nil; t -> t |> Expr.subst(sub) |> Expr.simplify() end
    acc = %{acc | rhs: s.(acc.rhs), ic: s.(acc.ic), ic_t: s.(acc.ic_t), exact: s.(acc.exact), f: s.(acc.f), c: s.(acc.c), boundary: s.(acc.boundary),
                  bc: Map.new(acc.bc, fn {k, {kind, g}} -> {k, {kind, s.(g)}} end), nx: min(max(acc.nx, 5), 4001), ny: min(max(acc.ny, 5), 401), nt: min(max(acc.nt, 1), 200_000)}

    cond do
      acc.x == nil -> {:error, "the domain is missing: x = a .. b"}
      acc.kind == :poisson and acc.y == nil -> {:error, "the domain is missing: y = c .. d"}
      acc.kind == :poisson and acc.boundary == nil and acc.exact == nil -> {:error, "the boundary data is missing: boundary u = g(x, y)  (or exact u = …)"}
      acc.kind != :poisson and acc.t == nil -> {:error, "the time span is missing: t = 0 .. T"}
      acc.kind != :poisson and acc.ic == nil and not (acc.verify and acc.exact != nil) -> {:error, "the initial condition is missing: u(x, 0) = …"}
      acc.verify and acc.exact == nil -> {:error, "verify needs the exact solution: exact u = …"}
      true -> check_vars(acc)
    end
  end

  defp check_vars(acc) do
    allowed = case acc.kind do
      :poisson -> ["x", "y"]
      :parabolic -> ["x", "t", "u", "u_x", "u_xx"]
      :wave -> ["x", "t", "u", "u_x", "u_xx"]
    end

    trees = [acc.rhs, acc.f, acc.c] |> Enum.reject(&is_nil/1)
    bad = trees |> Enum.flat_map(&Expr.vars/1) |> Enum.uniq() |> Kernel.--(allowed)
    if bad == [], do: {:ok, acc}, else: {:error, "unknown name(s) in the equation: #{Enum.join(bad, ", ")} (allowed: #{Enum.join(allowed, ", ")} and parameters)"}
  end

  # ================================================================ MMS

  defp with_verify(%{verify: true, exact: ex} = spec, solver) do
    m = manufactured(spec, ex)
    levels =
      for k <- 0..2 do
        r = Integer.pow(2, k)
        s = %{m | nx: (spec.nx - 1) * r + 1, ny: (spec.ny - 1) * r + 1, nt: spec.nt * r, snapshots: 2}
        {:ok, out} = solver.(s)
        %{nx: s.nx, ny: if(spec.kind == :poisson, do: s.ny), nt: if(spec.kind != :poisson, do: s.nt), error: out.error}
      end

    orders = levels |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> :math.log2(max(a.error, 1.0e-300) / max(b.error, 1.0e-300)) end)
    expected = if spec.kind == :parabolic and spec.scheme == "implicit", do: 1, else: 2
    {:ok, base} = solver.(m)

    {:ok, Map.merge(base, %{verification: %{levels: levels, orders: orders, expected_order: expected, source: Expr.to_text(m.source),
                                             verdict: if(List.last(orders) > expected - 0.25, do: "observed order #{Float.round(List.last(orders), 2)} (expected #{expected}): verified", else: "observed order #{Float.round(List.last(orders), 2)}, #{expected} expected — the scheme or the problem is not what it claims")}})}
  rescue
    e in [ArithmeticError, MatchError] -> {:error, "verification failed to run: #{Exception.message(e)}"}
  end

  defp with_verify(spec, solver) do
    solver.(spec)
  rescue
    e in [ArithmeticError] -> {:error, "arithmetic error (a division by zero or a function outside its domain): #{Exception.message(e)}"}
  end

  @doc false
  # a problem whose exact solution is `ex`: source derived symbolically, data from ex
  def manufactured(%{kind: :poisson} = spec, ex) do
    lap = Expr.simplify({:+, Expr.diff(Expr.diff(ex, "x"), "x"), Expr.diff(Expr.diff(ex, "y"), "y")})
    f = Expr.simplify({:+, {:neg, lap}, {:*, spec.c, ex}})
    %{spec | f: f, boundary: ex} |> Map.put(:source, f)
  end

  def manufactured(spec, ex) do
    ux = Expr.diff(ex, "x")
    uxx = Expr.diff(ux, "x")
    lhs = if spec.kind == :wave, do: Expr.diff(Expr.diff(ex, "t"), "t"), else: Expr.diff(ex, "t")
    op = Expr.subst(spec.rhs, %{"u" => ex, "u_x" => ux, "u_xx" => uxx})
    src = Expr.simplify({:-, lhs, op})
    {x0, x1} = spec.x
    bc = for at <- [x0, x1], into: %{} do
      case Enum.find(spec.bc, fn {k, _} -> abs(k - at) < 1.0e-12 end) do
        {_, {:neumann, _}} -> {at, {:neumann, Expr.subst(ux, %{"x" => {:n, at}}) |> Expr.simplify()}}
        _ -> {at, {:dirichlet, Expr.subst(ex, %{"x" => {:n, at}}) |> Expr.simplify()}}
      end
    end
    ic = Expr.subst(ex, %{"t" => {:n, elem(spec.t, 0)}}) |> Expr.simplify()
    ict = Expr.subst(Expr.diff(ex, "t"), %{"t" => {:n, elem(spec.t, 0)}}) |> Expr.simplify()
    %{spec | rhs: Expr.simplify({:+, spec.rhs, src}), bc: bc, ic: ic, ic_t: ict} |> Map.put(:source, src)
  end

  # ============================================================ parabolic

  defp parabolic(spec) do
    {x0, x1} = spec.x
    {t0, t1} = spec.t
    n = spec.nx
    h = (x1 - x0) / (n - 1)
    dt = (t1 - t0) / spec.nt
    xs = for i <- 0..(n - 1), do: x0 + i * h
    theta = if spec.scheme == "implicit", do: 1.0, else: 0.5

    # split the right side: a·u_xx + b·u_x + c·u + s (linear) and N(u) (the rest, explicit)
    a = Expr.diff(spec.rhs, "u_xx")
    b = Expr.diff(spec.rhs, "u_x")
    for {nm, coef} <- [{"u_xx", a}, {"u_x", b}], v <- ["u", "u_x", "u_xx"], v in Expr.vars(coef),
      do: throw({:pde_error, "the coefficient of #{nm} depends on #{v}: only a reaction term may be nonlinear in u"})
    # a and b do not depend on u or its derivatives (checked above), so the rest is the right side at u_xx = u_x = 0
    rest = spec.rhs |> Expr.subst(%{"u_xx" => {:n, 0.0}, "u_x" => {:n, 0.0}}) |> Expr.simplify()
    c = Expr.diff(rest, "u")
    linear = not ("u" in Expr.vars(c))
    {c, s, nl} = if linear, do: {c, Expr.simplify(Expr.subst(rest, %{"u" => {:n, 0.0}})), nil}, else: {{:n, 0.0}, {:n, 0.0}, rest}

    cf = Expr.compile([a, b, c, s], ["x", "t"])
    nlf = nl && Expr.compile(nl, ["x", "t", "u"])
    left = boundary(spec.bc, x0)
    right = boundary(spec.bc, x1)
    bcf = fn {kind, g} -> {kind, Expr.compile(g, ["t"])} end
    {left, right} = {bcf.(left), bcf.(right)}
    u0 = Enum.map(xs, &Expr.eval(spec.ic, %{"x" => &1, "t" => t0}))
    u0 = apply_dirichlet(u0, left, right, t0)

    every = max(1, div(spec.nt, max(spec.snapshots - 1, 1)))

    {u, _nprev, snaps} =
      Enum.reduce(1..spec.nt, {u0, nil, [{t0, u0}]}, fn k, {u, nprev, snaps} ->
        tn = t0 + (k - 1) * dt
        tp = tn + dt
        {ln, kn} = operator(cf, xs, h, tn, left, right)
        {lp, kp} = operator(cf, xs, h, tp, left, right)
        nnow = nlf && Enum.zip_with(xs, u, fn x, ui -> nlf.({x, tn, ui}) end)
        expl = case {nnow, nprev} do
          {nil, _} -> List.duplicate(0.0, n)
          {now, nil} -> Enum.map(now, &(dt * &1))
          {now, prev} -> Enum.zip_with(now, prev, &(dt * (1.5 * &1 - 0.5 * &2)))
        end

        lu = tri_apply(ln, u)
        rhs = for i <- 0..(n - 1), do: Enum.at(u, i) + (1 - theta) * dt * (Enum.at(lu, i) + Enum.at(kn, i)) + theta * dt * Enum.at(kp, i) + Enum.at(expl, i)
        {lo, di, up} = lp
        lo = Enum.map(lo, &(-theta * dt * &1))
        up = Enum.map(up, &(-theta * dt * &1))
        di = Enum.map(di, &(1 - theta * dt * &1))
        # Dirichlet rows: u = g
        {lo, di, up, rhs} = dirichlet_rows({lo, di, up, rhs}, left, right, tp, n)
        un = Dense.tridiag(lo, di, up, rhs)
        snaps = if rem(k, every) == 0 or k == spec.nt, do: [{tp, un} | snaps], else: snaps
        {un, nnow, snaps}
      end)

    error = spec.exact && (u |> Enum.zip(xs) |> Enum.map(fn {ui, x} -> abs(ui - Expr.eval(spec.exact, %{"x" => x, "t" => t1})) end) |> Enum.max())
    snaps = Enum.reverse(snaps)

    {:ok, %{family: "parabolic", scheme: if(theta == 0.5, do: "Crank–Nicolson" <> if(nl, do: " + Adams–Bashforth 2 (reaction)", else: ""), else: "backward Euler"),
            x: xs, t: Enum.map(snaps, &elem(&1, 0)), u: Enum.map(snaps, &elem(&1, 1)), final: u, error: error, dx: h, dt: dt, nx: n, nt: spec.nt,
            terms: %{diffusion: Expr.to_text(a), advection: Expr.to_text(b), reaction: if(nl, do: Expr.to_text(nl), else: Expr.to_text(Expr.simplify({:+, {:*, c, {:v, "u"}}, s})))},
            boundaries: %{left: elem(left, 0), right: elem(right, 0)}}}
  catch
    {:pde_error, why} -> {:error, why}
  end

  defp boundary(bc, at) do
    case Enum.find(bc, fn {k, _} -> abs(k - at) < 1.0e-12 end) do
      {_, v} -> v
      nil -> {:neumann, {:n, 0.0}}
    end
  end

  defp apply_dirichlet(u, {lk, lg}, {rk, rg}, t) do
    u = if lk == :dirichlet, do: List.replace_at(u, 0, lg.({t})), else: u
    if rk == :dirichlet, do: List.replace_at(u, -1, rg.({t})), else: u
  end

  defp dirichlet_rows({lo, di, up, rhs}, {lk, lg}, {rk, rg}, t, n) do
    {lo, di, up, rhs} = if lk == :dirichlet, do: {lo, List.replace_at(di, 0, 1.0), List.replace_at(up, 0, 0.0), List.replace_at(rhs, 0, lg.({t}))}, else: {lo, di, up, rhs}
    if rk == :dirichlet, do: {List.replace_at(lo, n - 1, 0.0), List.replace_at(di, n - 1, 1.0), up, List.replace_at(rhs, n - 1, rg.({t}))}, else: {lo, di, up, rhs}
  end

  # the discrete operator L (tridiagonal) and its constant part k at time t; ghost nodes for Neumann ends
  defp operator(cf, xs, h, t, {lk, lg}, {rk, rg}) do
    n = length(xs)
    rows =
      for {x, i} <- Enum.with_index(xs) do
        [a, b, c, s] = cf.({x, t})
        cond do
          i == 0 and lk == :neumann ->
            g = lg.({t})
            {0.0, -2 * a / (h * h) + c, 2 * a / (h * h), -2 * a * g / h + b * g + s}
          i == n - 1 and rk == :neumann ->
            g = rg.({t})
            {2 * a / (h * h), -2 * a / (h * h) + c, 0.0, 2 * a * g / h + b * g + s}
          true ->
            {a / (h * h) - b / (2 * h), -2 * a / (h * h) + c, a / (h * h) + b / (2 * h), s}
        end
      end

    {{Enum.map(rows, &elem(&1, 0)), Enum.map(rows, &elem(&1, 1)), Enum.map(rows, &elem(&1, 2))}, Enum.map(rows, &elem(&1, 3))}
  end

  defp tri_apply({lo, di, up}, u) do
    ut = List.to_tuple(u)
    n = tuple_size(ut)
    for i <- 0..(n - 1) do
      Enum.at(di, i) * elem(ut, i) + (if i > 0, do: Enum.at(lo, i) * elem(ut, i - 1), else: 0.0) + (if i < n - 1, do: Enum.at(up, i) * elem(ut, i + 1), else: 0.0)
    end
  end

  # ================================================================= wave

  defp wave(spec) do
    {x0, x1} = spec.x
    {t0, t1} = spec.t
    n = spec.nx
    h = (x1 - x0) / (n - 1)
    dt = (t1 - t0) / spec.nt
    xs = for i <- 0..(n - 1), do: x0 + i * h
    a = Expr.diff(spec.rhs, "u_xx")
    rest = spec.rhs |> Expr.subst(%{"u_xx" => {:n, 0.0}}) |> Expr.simplify()
    if Enum.any?(["u", "u_x", "u_xx"], &(&1 in Expr.vars(a) or &1 in Expr.vars(rest))), do: throw({:pde_error, "the wave equation here is u_tt = a(x,t)·u_xx + s(x,t)"})
    cf = Expr.compile([a, rest], ["x", "t"])
    amax = xs |> Enum.flat_map(fn x -> for t <- [t0, (t0 + t1) / 2, t1], do: hd(cf.({x, t})) end) |> Enum.max()
    if amax < 0, do: throw({:pde_error, "u_tt = a·u_xx needs a ≥ 0"})
    cfl = :math.sqrt(max(amax, 0.0)) * dt / h
    if cfl > 1.0, do: throw({:pde_error, "the CFL condition fails (c·dt/dx = #{Float.round(cfl, 3)} > 1): use nt ≥ #{ceil(spec.nt * cfl)}"})
    left = boundary(spec.bc, x0) |> then(fn {k, g} -> {k, Expr.compile(g, ["t"])} end)
    right = boundary(spec.bc, x1) |> then(fn {k, g} -> {k, Expr.compile(g, ["t"])} end)

    lap = fn u, t ->
      ut = List.to_tuple(u)
      for {x, i} <- Enum.with_index(xs) do
        [ai, si] = cf.({x, t})
        uxx = cond do
          i == 0 -> (if elem(left, 0) == :neumann, do: (2 * elem(ut, 1) - 2 * elem(ut, 0)) / (h * h) - 2 * elem(left, 1).({t}) / h, else: 0.0)
          i == n - 1 -> (if elem(right, 0) == :neumann, do: (2 * elem(ut, n - 2) - 2 * elem(ut, n - 1)) / (h * h) + 2 * elem(right, 1).({t}) / h, else: 0.0)
          true -> (elem(ut, i + 1) - 2 * elem(ut, i) + elem(ut, i - 1)) / (h * h)
        end
        ai * uxx + si
      end
    end

    u0 = Enum.map(xs, &Expr.eval(spec.ic, %{"x" => &1, "t" => t0})) |> apply_dirichlet(left, right, t0)
    v0 = Enum.map(xs, &Expr.eval(spec.ic_t || {:n, 0.0}, %{"x" => &1, "t" => t0}))
    u1 = Enum.zip_with([u0, v0, lap.(u0, t0)], fn [u, v, l] -> u + dt * v + dt * dt / 2 * l end) |> apply_dirichlet(left, right, t0 + dt)
    every = max(1, div(spec.nt, max(spec.snapshots - 1, 1)))

    {u, _, snaps} =
      Enum.reduce(2..spec.nt//1, {u1, u0, [{t0 + dt, u1}, {t0, u0}]}, fn k, {u, up, snaps} ->
        t = t0 + (k - 1) * dt
        un = Enum.zip_with([u, up, lap.(u, t)], fn [a1, b1, l] -> 2 * a1 - b1 + dt * dt * l end) |> apply_dirichlet(left, right, t + dt)
        snaps = if rem(k, every) == 0 or k == spec.nt, do: [{t + dt, un} | snaps], else: snaps
        {un, u, snaps}
      end)

    error = spec.exact && (u |> Enum.zip(xs) |> Enum.map(fn {ui, x} -> abs(ui - Expr.eval(spec.exact, %{"x" => x, "t" => t1})) end) |> Enum.max())
    snaps = Enum.reverse(snaps) |> Enum.uniq_by(&elem(&1, 0))
    {:ok, %{family: "hyperbolic", scheme: "leapfrog (second order)", cfl: cfl, x: xs, t: Enum.map(snaps, &elem(&1, 0)), u: Enum.map(snaps, &elem(&1, 1)), final: u, error: error, dx: h, dt: dt, nx: n, nt: spec.nt}}
  catch
    {:pde_error, why} -> {:error, why}
  end

  # ============================================================== poisson

  defp poisson(spec) do
    {x0, x1} = spec.x
    {y0, y1} = spec.y
    {nx, ny} = {spec.nx, spec.ny}
    hx = (x1 - x0) / (nx - 1)
    hy = (y1 - y0) / (ny - 1)
    g = spec.boundary || spec.exact
    gf = Expr.compile(g, ["x", "y"])
    ff = Expr.compile([spec.f, spec.c], ["x", "y"])
    {mx, my} = {nx - 2, ny - 2}
    idx = fn i, j -> (j - 1) * mx + (i - 1) end
    xy = fn i, j -> {x0 + i * hx, y0 + j * hy} end
    cx = 1 / (hx * hx)
    cy = 1 / (hy * hy)
    cvals = for j <- 1..my, i <- 1..mx, do: Enum.at(ff.(xy.(i, j)), 1)
    if Enum.any?(cvals, &(&1 < 0)), do: throw({:pde_error, "c(x, y) must be ≥ 0 (the operator must stay positive definite)"})
    ct = List.to_tuple(cvals)

    b =
      for j <- 1..my, i <- 1..mx do
        [f, _] = ff.(xy.(i, j))
        bnd = fn ii, jj, w ->
          if ii == 0 or jj == 0 or ii == nx - 1 or jj == ny - 1 do
            {bx, by} = xy.(ii, jj)
            w * gf.({bx, by})
          else
            0.0
          end
        end
        f + bnd.(i - 1, j, cx) + bnd.(i + 1, j, cx) + bnd.(i, j - 1, cy) + bnd.(i, j + 1, cy)
      end

    apply_a = fn u ->
      ut = List.to_tuple(u)
      at = fn i, j -> if i < 1 or j < 1 or i > mx or j > my, do: 0.0, else: elem(ut, idx.(i, j)) end
      for j <- 1..my, i <- 1..mx do
        (2 * cx + 2 * cy + elem(ct, idx.(i, j))) * at.(i, j) - cx * (at.(i - 1, j) + at.(i + 1, j)) - cy * (at.(i, j - 1) + at.(i, j + 1))
      end
    end

    {u, its, rel} = Dense.cg(apply_a, b, tol: 1.0e-12, max_iter: 20 * (mx + my) + 200)
    ut = List.to_tuple(u)
    cell = fn i, j ->
      if i == 0 or j == 0 or i == nx - 1 or j == ny - 1 do
        {x, y} = xy.(i, j)
        gf.({x, y})
      else
        elem(ut, idx.(i, j))
      end
    end

    grid = for j <- 0..(ny - 1), do: for(i <- 0..(nx - 1), do: cell.(i, j))

    error =
      if spec.exact do
        ef = Expr.compile(spec.exact, ["x", "y"])
        for(j <- 0..(ny - 1), i <- 0..(nx - 1), do: abs(cell.(i, j) - ef.(xy.(i, j)))) |> Enum.max()
      end

    {:ok, %{family: "elliptic", scheme: "five-point stencil, conjugate gradients", grid: grid, x: for(i <- 0..(nx - 1), do: x0 + i * hx), y: for(j <- 0..(ny - 1), do: y0 + j * hy),
            iterations: its, residual: rel, error: error, nx: nx, ny: ny, unknowns: mx * my}}
  catch
    {:pde_error, why} -> {:error, why}
  end
end
