defmodule Vapor.Solve do
  @moduledoc """
  The workbench's solvers for problems typed in as text (docs/BANCADA.md):
  ordinary differential equations (adaptive Dormand–Prince 5(4) with its
  dense output, a Rosenbrock 2(3) method for stiff systems with the
  Jacobian derived symbolically, classical RK4), and — in the submodules —
  nonlinear systems, parameter fitting, minimisation
  (`Vapor.Solve.Algebra`) and partial differential equations with
  verification by manufactured solutions (`Vapor.Solve.PDE`).

  `run/1` recognises what a text asks for and dispatches; every answer
  carries the numbers that let a reader judge it — steps accepted and
  rejected, function evaluations, residuals, the observed order of
  accuracy — and the units are checked before anything runs.

  An ODE system is written as it is on paper:

      x' = v
      v' = -k/m*x - c/m*v + F0*cos(w*t)
      k = 4[N/m]; m = 1[kg]; c = 0.1[N*s/m]; F0 = 1[N]; w = 2[1/s]
      x(0) = 1[m]; v(0) = 0[m/s]
      t = 0 .. 20[s]
      E := 0.5*m*v^2 + 0.5*k*x^2        # an output computed along the way
      stop when x < -2[m]                 # an event (optional)
      method = auto                       # rk45 | stiff | rk4 | auto
  """
  alias Vapor.{Dense, Expr, Units}

  @options ~w(method rtol atol samples max_steps h)

  # ============================================================== dispatch

  @doc """
  Solve whatever the text describes: an ODE system (lines with `'` or
  `d…/dt`), a PDE (`u_t = …` or `poisson`), a minimisation (`minimize`),
  a fit (`fit`), a nonlinear system (`unknowns …` and equations) or else a
  worksheet of formulas with units. `{:ok, %{kind, …}}` or `{:error, why}`.
  """
  def run(text) when is_binary(text) do
    cond do
      text =~ ~r/^\s*(poisson|u_t\s*=|u_tt\s*=)/mi -> tag(Vapor.Solve.PDE.run(text), "pde")
      text =~ ~r/^\s*[\p{L}_][\w]*'\s*=|^\s*d[\p{L}_]\w*\/dt\s*=/mu -> tag(ode(text), "ode")
      text =~ ~r/^\s*(minimi[sz]e|minimi[sz]ar|maximi[sz]e|maximi[sz]ar)\b/mi -> tag(Vapor.Solve.Algebra.minimize(text), "minimize")
      text =~ ~r/^\s*(fit|ajuste|ajustar)\b/mi -> tag(Vapor.Solve.Algebra.fit(text), "fit")
      text =~ ~r/^\s*(unknowns?|inc[oó]gnitas?|solve|resolva)\b/mi -> tag(Vapor.Solve.Algebra.nonlinear(text), "nonlinear")
      true -> {:ok, %{kind: "worksheet", lines: Expr.worksheet(text)}}
    end
  end

  defp tag({:ok, r}, k), do: {:ok, Map.put(r, :kind, k)}
  defp tag(e, _), do: e

  # ================================================================ parsing

  @doc false
  # split a problem text into trimmed, comment-free statements (newline or ;)
  def statements(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.flat_map(fn {l, n} ->
      l |> String.split("#", parts: 2) |> hd() |> String.split(";") |> Enum.map(&{String.trim(&1), n})
    end)
    |> Enum.reject(fn {s, _} -> s == "" end)
  end

  @doc """
  Parse an ODE system: `{:ok, system}` with the states in order of
  appearance, their right-hand sides (parameters substituted), initial
  values, the span, outputs, the event and options — or `{:error, why}`
  naming the line.
  """
  def parse_ode(text) do
    acc = %{states: [], rhs: %{}, init: %{}, params: %{}, pdims: %{}, span: nil, outputs: [], event: nil, opts: %{}, idims: %{}, tdim: Units.none()}

    statements(text)
    |> Enum.reduce_while({:ok, acc}, fn {s, line}, {:ok, acc} ->
      case ode_line(s, acc) do
        {:ok, acc} -> {:cont, {:ok, acc}}
        {:error, why} -> {:halt, {:error, "line #{line}: #{why}"}}
      end
    end)
    |> case do
      {:ok, acc} -> finish_ode(acc)
      e -> e
    end
  end

  @id "[\\p{L}_][\\p{L}\\p{N}_]*"

  defp ode_line(s, acc) do
    cond do
      m = Regex.run(~r/^(#{@id})'\s*=\s*(.+)$/u, s) -> deriv(acc, Enum.at(m, 1), Enum.at(m, 2))
      m = Regex.run(~r/^d(#{@id})\/dt\s*=\s*(.+)$/u, s) -> deriv(acc, Enum.at(m, 1), Enum.at(m, 2))
      m = Regex.run(~r/^(#{@id})\(\s*[^)]*\)\s*=\s*(.+)$/u, s) -> initial(acc, Enum.at(m, 1), Enum.at(m, 2))
      m = Regex.run(~r/^t\s*=\s*(.+?)\s*\.\.\s*(.+)$/u, s) -> span(acc, Enum.at(m, 1), Enum.at(m, 2))
      m = Regex.run(~r/^(?:stop|pare)\s+(?:when|quando)\s+(.+)$/iu, s) -> with({:ok, t} <- Expr.parse(Enum.at(m, 1)), do: {:ok, %{acc | event: t}})
      m = Regex.run(~r/^(#{@id})\s*:=\s*(.+)$/u, s) -> with({:ok, t} <- Expr.parse(Enum.at(m, 2)), do: {:ok, %{acc | outputs: acc.outputs ++ [{Enum.at(m, 1), t}]}})
      m = Regex.run(~r/^(#{Enum.join(@options, "|")})\s*=\s*(.+)$/u, s) -> {:ok, put_in(acc, [:opts, Enum.at(m, 1)], String.trim(Enum.at(m, 2)))}
      m = Regex.run(~r/^(#{@id})\s*=\s*(.+)$/u, s) -> param(acc, Enum.at(m, 1), Enum.at(m, 2))
      true -> {:error, "not understood: #{inspect(s)} (expected x' = …, x(0) = …, t = a .. b, name = value, name := output, stop when …)"}
    end
  end

  defp deriv(acc, name, rhs) do
    with {:ok, t} <- Expr.parse(rhs) do
      if Map.has_key?(acc.rhs, name), do: {:error, "#{name}' is given twice"}, else: {:ok, %{acc | states: acc.states ++ [name], rhs: Map.put(acc.rhs, name, t)}}
    end
  end

  defp const_value(rhs, acc) do
    with {:ok, t} <- Expr.parse(rhs),
         [] <- Expr.vars(t) -- Map.keys(acc.params) |> (fn [] -> []; m -> {:error, "unknown name(s) #{Enum.join(m, ", ")} in a constant"} end).(),
         {:ok, d} <- Expr.dim(t, acc.pdims) do
      try do
        {:ok, Expr.eval(t, acc.params), d}
      rescue
        ArithmeticError -> {:error, "arithmetic error in #{rhs}"}
      end
    end
  end

  defp initial(acc, name, rhs) do
    with {:ok, v, d} <- const_value(rhs, acc), do: {:ok, %{acc | init: Map.put(acc.init, name, v), idims: Map.put(acc.idims, name, d)}}
  end

  defp param(acc, name, rhs) do
    if name == "t", do: {:error, "t is the independent variable"}, else: (with {:ok, v, d} <- const_value(rhs, acc), do: {:ok, %{acc | params: Map.put(acc.params, name, v), pdims: Map.put(acc.pdims, name, d)}})
  end

  defp span(acc, a, b) do
    with {:ok, t0, d0} <- const_value(a, acc), {:ok, t1, d1} <- const_value(b, acc) do
      d = if Units.none?(d0), do: d1, else: d0
      cond do
        not Units.none?(d0) and not Units.none?(d1) and not Units.same?(d0, d1) -> {:error, "the span's ends have different units"}
        t1 <= t0 -> {:error, "t = a .. b needs b > a"}
        true -> {:ok, %{acc | span: {t0, t1}, tdim: d}}
      end
    end
  end

  defp finish_ode(acc) do
    missing = Enum.reject(acc.states, &Map.has_key?(acc.init, &1))
    cond do
      acc.states == [] -> {:error, "no equation x' = …"}
      missing != [] -> {:error, "initial value missing for #{Enum.join(missing, ", ")} (write #{hd(missing)}(0) = …)"}
      acc.span == nil -> {:error, "the span is missing (write t = 0 .. 10)"}
      true -> check_ode(acc)
    end
  end

  defp check_ode(acc) do
    sub = Map.new(acc.params, fn {k, v} -> {k, {:n, v}} end)
    rhs = for s <- acc.states, do: acc.rhs[s] |> Expr.subst(sub) |> Expr.simplify()
    outs = for {n, t} <- acc.outputs, do: {n, t |> Expr.subst(sub) |> Expr.simplify()}
    event = acc.event && acc.event |> Expr.subst(sub) |> Expr.simplify()
    known = ["t" | acc.states]
    bad = (rhs ++ Enum.map(outs, &elem(&1, 1)) ++ List.wrap(event)) |> Enum.flat_map(&Expr.vars/1) |> Enum.uniq() |> Enum.reject(&(&1 in known))

    if bad != [] do
      {:error, "unknown name(s): #{Enum.join(bad, ", ")} (not a state, not a parameter, not t)"}
    else
      dims = Map.new(acc.states, &{&1, Map.get(acc.idims, &1, Units.none())}) |> Map.put("t", acc.tdim)
      raw_dims = Map.merge(acc.pdims, dims)

      checks =
        for s <- acc.states do
          want = Units.sub(dims[s], acc.tdim)
          case Expr.dim(acc.rhs[s], raw_dims) do
            {:ok, d} -> if Units.same?(d, want), do: :ok, else: {:error, "#{s}' has #{Units.label(d)} but #{s}/t is #{Units.label(want)}"}
            {:error, why} -> {:error, "#{s}': #{why}"}
          end
        end

      case Enum.find(checks, &(&1 != :ok)) do
        nil -> {:ok, Map.merge(acc, %{rhs_list: rhs, outputs_s: outs, event_s: event, dims: dims})}
        err -> err
      end
    end
  end

  # ================================================================== ODEs

  @doc """
  Solve an ODE system given as text (see the moduledoc): `{:ok, result}`
  with `t`, `series` (per state and output, sampled at `samples` points by
  the method's dense output), `steps`, `rejected`, `evals`, `method`, the
  `event` if one fired, and the units of each series.
  """
  def ode(text) when is_binary(text) do
    with {:ok, sys} <- parse_ode(text), do: integrate(sys)
  end

  @doc false
  def integrate(sys) do
    names = ["t" | sys.states]
    f = Expr.compile(sys.rhs_list, names)
    {t0, t1} = sys.span
    y0 = Enum.map(sys.states, &sys.init[&1])
    opts = sys.opts
    rtol = num_opt(opts["rtol"], 1.0e-6)
    atol = num_opt(opts["atol"], 1.0e-9)
    samples = opts["samples"] |> num_opt(400) |> trunc() |> max(2) |> min(20_000)
    max_steps = opts["max_steps"] |> num_opt(200_000) |> trunc() |> min(2_000_000)
    method = opts["method"] || "auto"
    rhs = fn t, y -> f.(List.to_tuple([t | y])) end
    event = sys.event_s && Expr.compile(sys.event_s, names)
    ctx = %{rhs: rhs, t0: t0, t1: t1, rtol: rtol, atol: atol, max_steps: max_steps, event: event, sys: sys, names: names}

    started = System.monotonic_time(:microsecond)

    result =
      case method do
        "rk4" -> rk4(ctx, y0, num_opt(opts["h"], (t1 - t0) / 1000))
        "stiff" -> rosenbrock(ctx, y0)
        "rk45" -> dopri(ctx, y0)
        _ ->
          # explicit first; a stiff system shows itself by exhausting the step budget — then switch, and say so
          case dopri(%{ctx | max_steps: min(max_steps, 20_000)}, y0) do
            {:error, {:max_steps, _}} -> with {:ok, r} <- rosenbrock(ctx, y0), do: {:ok, Map.put(r, :switched, "the explicit method exhausted 20 000 steps: the system is stiff; solved with Rosenbrock 2(3)")}
            other -> other
          end
      end

    with {:ok, r} <- result do
      {:ok, finish(r, sys, samples, Map.put(ctx, :us, System.monotonic_time(:microsecond) - started))}
    end
  rescue
    e in ArithmeticError -> {:error, "arithmetic error while integrating (a division by zero or a function outside its domain): #{Exception.message(e)}"}
  end

  defp num_opt(nil, d), do: d * 1.0
  defp num_opt(s, d) when is_binary(s), do: (case Float.parse(s) do {v, _} -> v; :error -> d * 1.0 end)
  defp num_opt(v, _) when is_number(v), do: v * 1.0

  # sample the dense pieces uniformly, add outputs, units
  defp finish(r, sys, samples, ctx) do
    tend = if r.event, do: r.event.t, else: ctx.t1
    ts = for i <- 0..(samples - 1), do: ctx.t0 + (tend - ctx.t0) * i / (samples - 1)
    ys = sample(r.pieces, ts)
    states = Map.new(Enum.with_index(sys.states), fn {s, i} -> {s, Enum.map(ys, &Enum.at(&1, i))} end)
    outs_f = for {n, t} <- sys.outputs_s, do: {n, Expr.compile(t, ctx.names)}
    outs = Map.new(outs_f, fn {n, f} -> {n, Enum.zip_with(ts, ys, fn t, y -> f.(List.to_tuple([t | y])) end)} end)
    units = Map.new(sys.states, &{&1, Units.format(sys.dims[&1])})

    %{t: ts, series: Map.merge(states, outs), states: sys.states, outputs: Enum.map(sys.outputs, &elem(&1, 0)), units: units, time_unit: Units.format(sys.tdim),
      method: r.method, steps: r.steps, rejected: r.rejected, evals: r.evals, event: r.event, switched: Map.get(r, :switched),
      final: Map.new(Enum.with_index(sys.states), fn {s, i} -> {s, Enum.at(r.y_end, i)} end), t_end: r.t_end,
      rtol: ctx.rtol, atol: ctx.atol, microseconds: ctx.us, equations: Enum.zip_with(sys.states, sys.rhs_list, fn s, t -> "#{s}' = #{Expr.to_text(t)}" end)}
  end

  # pieces: [{t_a, t_b, interp_fun(θ) -> y}] ascending
  defp sample(pieces, ts) do
    pieces = List.to_tuple(pieces)
    n = tuple_size(pieces)
    {ys, _} =
      Enum.map_reduce(ts, 0, fn t, k ->
        k = advance_piece(pieces, n, k, t)
        {ta, tb, interp} = elem(pieces, k)
        th = if tb > ta, do: min(max((t - ta) / (tb - ta), 0.0), 1.0), else: 1.0
        {interp.(th), k}
      end)
    ys
  end

  defp advance_piece(p, n, k, t) do
    if k < n - 1 and elem(elem(p, k), 1) < t, do: advance_piece(p, n, k + 1, t), else: k
  end

  defp err_norm(e, y0, y1, atol, rtol) do
    s = Enum.zip([e, y0, y1]) |> Enum.reduce(0.0, fn {ei, a, b}, acc -> (q = ei / (atol + rtol * max(abs(a), abs(b))); acc + q * q) end)
    :math.sqrt(s / max(length(e), 1))
  end

  defp lin(y, terms), do: Enum.reduce(terms, y, fn {c, k}, acc -> if c == 0, do: acc, else: Enum.zip_with(acc, k, &(&1 + c * &2)) end)

  defp initial_h(ctx, y0, f0, order) do
    sc = fn y -> Enum.map(y, &(ctx.atol + ctx.rtol * abs(&1))) end
    nrm = fn v, s -> :math.sqrt(Enum.zip_with(v, s, &((&1 / &2) ** 2)) |> Enum.sum() |> Kernel./(max(length(v), 1))) end
    s0 = sc.(y0)
    d0 = nrm.(y0, s0)
    d1 = nrm.(f0, s0)
    h0 = if d0 < 1.0e-5 or d1 < 1.0e-5, do: 1.0e-6, else: 0.01 * d0 / d1
    span = ctx.t1 - ctx.t0
    h0 = min(h0, span)
    y1 = lin(y0, [{h0, f0}])
    f1 = ctx.rhs.(ctx.t0 + h0, y1)
    d2 = nrm.(Enum.zip_with(f1, f0, &(&1 - &2)), s0) / h0
    h1 = if max(d1, d2) <= 1.0e-15, do: max(1.0e-6, h0 * 1.0e-3), else: :math.pow(0.01 / max(d1, d2), 1 / (order + 1))
    min(min(100 * h0, h1), span)
  end

  # ------------------------------------------------------- Dormand–Prince

  @a [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
  @c [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
  @e [71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40]
  # Hairer's dense output (contd5)
  @dd [-12715105075 / 11282082432, 0.0, 87487479700 / 32700410799, -10690763975 / 1880347072, 701980252875 / 199316789632, -1453857185 / 822651844, 69997945 / 29380423]

  defp dopri(ctx, y0) do
    f0 = ctx.rhs.(ctx.t0, y0)
    h = initial_h(ctx, y0, f0, 5)
    dp_loop(ctx, ctx.t0, y0, f0, h, %{steps: 0, rejected: 0, evals: 2, pieces: [], ev0: event_val(ctx, ctx.t0, y0)})
  end

  defp event_val(%{event: nil}, _, _), do: nil
  defp event_val(%{event: e}, t, y), do: e.(List.to_tuple([t | y]))

  defp dp_loop(ctx, t, y, f0, h, st) do
    cond do
      t >= ctx.t1 - 1.0e-12 * abs(ctx.t1) ->
        {:ok, %{method: "Dormand–Prince 5(4)", steps: st.steps, rejected: st.rejected, evals: st.evals, pieces: Enum.reverse(st.pieces), y_end: y, t_end: t, event: nil}}

      st.steps + st.rejected >= ctx.max_steps ->
        {:error, {:max_steps, t}}

      h < 1.0e-14 * max(abs(t), 1.0) ->
        {:error, "the step size underflowed at t = #{Expr.num(t)} (a singularity, or the tolerance is below what binary64 can hold)"}

      true ->
        h = min(h, ctx.t1 - t)
        ks = Enum.reduce(1..5, [f0], fn i, ks ->
          yi = lin(y, Enum.zip(Enum.map(Enum.at(@a, i), &(&1 * h)), Enum.reverse(ks)))
          [ctx.rhs.(t + Enum.at(@c, i) * h, yi) | ks]
        end) |> Enum.reverse()
        y1 = lin(y, Enum.zip(Enum.map(Enum.at(@a, 6), &(&1 * h)), Enum.take(ks, 6)))
        f1 = ctx.rhs.(t + h, y1)
        ks = ks ++ [f1]
        e = lin(List.duplicate(0.0, length(y)), Enum.zip(Enum.map(@e, &(&1 * h)), ks))
        err = err_norm(e, y, y1, ctx.atol, ctx.rtol)
        fac = if err == 0, do: 5.0, else: min(5.0, max(0.2, 0.9 * :math.pow(err, -0.2)))

        if err <= 1.0 do
          interp = dense5(y, y1, h, ks)
          piece = {t, t + h, interp}
          st = %{st | steps: st.steps + 1, evals: st.evals + 6, pieces: [piece | st.pieces]}

          case check_event(ctx, st.ev0, t, h, interp) do
            {:fired, te, ye} ->
              pieces = [{t, te, fn th -> interp.(th * (te - t) / h) end} | tl(st.pieces)]
              {:ok, %{method: "Dormand–Prince 5(4)", steps: st.steps, rejected: st.rejected, evals: st.evals, pieces: Enum.reverse(pieces), y_end: ye, t_end: te, event: %{t: te, y: ye}}}

            ev ->
              dp_loop(ctx, t + h, y1, f1, h * fac, %{st | ev0: ev})
          end
        else
          dp_loop(ctx, t, y, f0, h * max(0.2, fac), %{st | rejected: st.rejected + 1, evals: st.evals + 6})
        end
    end
  end

  defp dense5(y0, y1, h, ks) do
    [k1, _k2, _k3, _k4, _k5, _k6, k7] = ks
    r2 = Enum.zip_with(y1, y0, &(&1 - &2))
    r3 = Enum.zip_with(k1, r2, &(h * &1 - &2))
    r4 = Enum.zip_with([r2, k7, r3], fn [a, b, c] -> a - h * b - c end)
    r5 = lin(List.duplicate(0.0, length(y0)), Enum.zip(Enum.map(@dd, &(&1 * h)), ks))

    fn th ->
      t1 = 1 - th
      Enum.zip_with([y0, r2, r3, r4, r5], fn [a, b, c, d, e] -> a + th * (b + t1 * (c + th * (d + t1 * e))) end)
    end
  end

  # an event fires when its condition turns true inside the step: located by bisection on the dense output
  defp check_event(%{event: nil}, _, _, _, _), do: nil

  defp check_event(ctx, before, t, h, interp) do
    now = event_val(ctx, t + h, interp.(1.0))
    if before == 0.0 and now != 0.0 do
      th = Enum.reduce(1..50, {0.0, 1.0}, fn _, {lo, hi} ->
        mid = (lo + hi) / 2
        if event_val(ctx, t + mid * h, interp.(mid)) != 0.0, do: {lo, mid}, else: {mid, hi}
      end) |> elem(1)
      {:fired, t + th * h, interp.(th)}
    else
      now
    end
  end

  # ------------------------------------------------------------ Rosenbrock

  @doc false
  # ode23s (Shampine & Reichelt 1997): L-stable, order 2 with an order-3 error estimate, one LU per step
  def rosenbrock(ctx, y0) do
    sys = ctx.sys
    n = length(sys.states)
    jac = Expr.compile(for(r <- sys.rhs_list, s <- sys.states, do: Expr.diff(r, s)), ctx.names)
    dfdt = Expr.compile(Enum.map(sys.rhs_list, &Expr.diff(&1, "t")), ctx.names)
    f0 = ctx.rhs.(ctx.t0, y0)
    h = initial_h(ctx, y0, f0, 2)
    tools = %{jac: fn t, y -> jac.(List.to_tuple([t | y])) |> Enum.chunk_every(n) end, dfdt: fn t, y -> dfdt.(List.to_tuple([t | y])) end}
    rb_loop(ctx, tools, ctx.t0, y0, f0, h, %{steps: 0, rejected: 0, evals: 1, lus: 0, pieces: [], ev0: event_val(ctx, ctx.t0, y0)})
  end

  @d 1 / (2 + :math.sqrt(2))
  @e32 6 + :math.sqrt(2)

  defp rb_loop(ctx, tools, t, y, f0, h, st) do
    cond do
      t >= ctx.t1 - 1.0e-12 * abs(ctx.t1) ->
        {:ok, %{method: "Rosenbrock 2(3) (stiff)", steps: st.steps, rejected: st.rejected, evals: st.evals, lu: st.lus, pieces: Enum.reverse(st.pieces), y_end: y, t_end: t, event: nil}}

      st.steps + st.rejected >= ctx.max_steps ->
        {:error, "the stiff solver exhausted #{ctx.max_steps} steps at t = #{Expr.num(t)}"}

      h < 1.0e-14 * max(abs(t), 1.0) ->
        {:error, "the step size underflowed at t = #{Expr.num(t)}"}

      true ->
        h = min(h, ctx.t1 - t)
        j = tools.jac.(t, y)
        tt = tools.dfdt.(t, y)
        nn = length(y)
        w = for {row, i} <- Enum.with_index(j), do: for({x, k} <- Enum.with_index(row), do: (if i == k, do: 1.0, else: 0.0) - h * @d * x)

        with {:ok, k1} <- Dense.solve(w, Enum.zip_with(f0, tt, &(&1 + h * @d * &2))) do
          f1 = ctx.rhs.(t + 0.5 * h, lin(y, [{0.5 * h, k1}]))
          {:ok, k2a} = Dense.solve(w, Enum.zip_with(f1, k1, &(&1 - &2)))
          k2 = Enum.zip_with(k2a, k1, &(&1 + &2))
          y1 = lin(y, [{h, k2}])
          f2 = ctx.rhs.(t + h, y1)
          rhs3 = for i <- 0..(nn - 1), do: Enum.at(f2, i) - @e32 * (Enum.at(k2, i) - Enum.at(f1, i)) - 2 * (Enum.at(k1, i) - Enum.at(f0, i)) + h * @d * Enum.at(tt, i)
          {:ok, k3} = Dense.solve(w, rhs3)
          e = for i <- 0..(nn - 1), do: h / 6 * (Enum.at(k1, i) - 2 * Enum.at(k2, i) + Enum.at(k3, i))
          err = err_norm(e, y, y1, ctx.atol, ctx.rtol)
          fac = if err == 0, do: 5.0, else: min(5.0, max(0.2, 0.8 * :math.pow(err, -1 / 3)))

          if err <= 1.0 do
            interp = fn th -> lin(y, [{h * th * (1 - th) / (1 - 2 * @d), k1}, {h * th * (th - 2 * @d) / (1 - 2 * @d), k2}]) end
            st = %{st | steps: st.steps + 1, evals: st.evals + 3, lus: st.lus + 1, pieces: [{t, t + h, interp} | st.pieces]}
            case check_event(ctx, st.ev0, t, h, interp) do
              {:fired, te, ye} ->
                pieces = [{t, te, fn th -> interp.(th * (te - t) / h) end} | tl(st.pieces)]
                {:ok, %{method: "Rosenbrock 2(3) (stiff)", steps: st.steps, rejected: st.rejected, evals: st.evals, lu: st.lus, pieces: Enum.reverse(pieces), y_end: ye, t_end: te, event: %{t: te, y: ye}}}
              ev -> rb_loop(ctx, tools, t + h, y1, f2, h * fac, %{st | ev0: ev})
            end
          else
            rb_loop(ctx, tools, t, y, f0, h * max(0.2, fac), %{st | rejected: st.rejected + 1, evals: st.evals + 2, lus: st.lus + 1})
          end
        else
          {:error, {:singular, _}} -> rb_loop(ctx, tools, t, y, f0, h * 0.25, %{st | rejected: st.rejected + 1})
        end
    end
  end

  # ------------------------------------------------------------------ RK4

  defp rk4(ctx, y0, h) do
    n = max(1, ceil((ctx.t1 - ctx.t0) / h))
    h = (ctx.t1 - ctx.t0) / n

    {y, pieces} =
      Enum.reduce(0..(n - 1), {y0, []}, fn i, {y, pieces} ->
        t = ctx.t0 + i * h
        k1 = ctx.rhs.(t, y)
        k2 = ctx.rhs.(t + h / 2, lin(y, [{h / 2, k1}]))
        k3 = ctx.rhs.(t + h / 2, lin(y, [{h / 2, k2}]))
        k4 = ctx.rhs.(t + h, lin(y, [{h, k3}]))
        y1 = lin(y, [{h / 6, k1}, {h / 3, k2}, {h / 3, k3}, {h / 6, k4}])
        # cubic Hermite between the step's ends
        f1 = ctx.rhs.(t + h, y1)
        interp = fn th ->
          h00 = 2 * th ** 3 - 3 * th ** 2 + 1; h10 = th ** 3 - 2 * th ** 2 + th; h01 = -2 * th ** 3 + 3 * th ** 2; h11 = th ** 3 - th ** 2
          Enum.zip_with([y, k1, y1, f1], fn [a, b, c, d] -> h00 * a + h10 * h * b + h01 * c + h11 * h * d end)
        end
        {y1, [{t, t + h, interp} | pieces]}
      end)

    {:ok, %{method: "classical Runge–Kutta 4 (fixed step #{Expr.num(h)})", steps: n, rejected: 0, evals: 5 * n, pieces: Enum.reverse(pieces), y_end: y, t_end: ctx.t1, event: nil}}
  end
end
