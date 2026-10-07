defmodule Vapor.Crucible.Hamiltonian do
  @moduledoc """
  Any Hamiltonian system the user writes (docs/CRUCIBLE.md §3):

      H = (p1^2 + p2^2)/2 - 1/sqrt(q1^2 + q2^2)
      q1(0) = 1; q2(0) = 0; p1(0) = 0; p2(0) = 1
      t = 0 .. 100
      dt = 0.01
      method = auto            # verlet | yoshida4 | midpoint | rk4

  Coordinates are `q` or `q1…qn`, momenta `p` or `p1…pn`. Hamilton's
  equations come from **symbolic** derivatives of H. If H separates into
  T(p) + V(q) (checked symbolically: every ∂²H/∂q∂p is 0) the default is
  Yoshida's 4th-order symplectic composition of leapfrog; otherwise the
  implicit midpoint rule (symplectic for any H).

  Evidence, none of which needs the true solution:

    * **energy**: a symplectic method keeps H within a bounded band for
      exponentially long times; the same run by classical RK4 (the
      control) drifts. The drift's slope is fitted and compared;
    * **observed order**: three runs at dt, dt/2, dt/4 over a short span —
      log₂ of the ratio of successive differences must match the method;
    * **time reversibility**: forward, then backward with −dt, back to the
      start (to round-off for a symmetric method);
    * **conservation laws**: when Hamilton's equations are polynomial, the
      Crucible's exact search (`Vapor.Crucible.Laws`) runs on them — and
      must find H itself, plus whatever else is conserved (angular
      momentum…).
  """
  alias Vapor.Expr
  alias Vapor.Crucible.{Laws, Sheet}

  def run(text) do
    s = Sheet.parse(text)

    with [] <- s.errors,
         {:ok, h} <- fetch_h(s),
         {:ok, qs, ps} <- coords(h) do
      dhdq = Enum.map(qs, &Expr.diff(h, &1))
      dhdp = Enum.map(ps, &Expr.diff(h, &1))
      names = qs ++ ps
      separable = Enum.all?(for q <- qs, p <- ps, do: Expr.diff(Expr.diff(h, q), p) |> Expr.simplify() |> zero?())
      fq = compile(dhdq, names)
      fp = compile(dhdp, names)
      hf = compile([h], names)
      energy = fn z -> hd(hf.(z)) end
      {t0, t1} = Map.get(s.ranges, "t", {0.0, 50.0})
      dt = Sheet.const(s, "dt", (t1 - t0) / 2000)
      method = case Sheet.word(s, "method", "auto") do "auto" -> if(separable, do: "yoshida4", else: "midpoint"); m -> m end

      cond do
        method in ["verlet", "yoshida4"] and not separable -> {:error, "#{method} needs a separable H = T(p) + V(q); this one is not (use method = midpoint)"}
        method not in ["verlet", "yoshida4", "midpoint", "rk4"] -> {:error, "method: verlet, yoshida4, midpoint or rk4"}
        Enum.any?(names, &(not Map.has_key?(s.inits, &1))) -> {:error, "initial values: " <> Enum.map_join(names, ", ", &"#{&1}(0) = …")}
        true ->
          z0 = Enum.map(names, &elem(s.inits[&1], 1))
          n = length(qs)
          stepper = stepper(method, fq, fp, n)
          steps = max(round((t1 - t0) / dt), 1)
          every = max(div(steps, 600), 1)
          {traj, zf} = integrate(stepper, z0, dt, steps, every, energy, t0)
          control = if method != "rk4", do: integrate(stepper("rk4", fq, fp, n), z0, dt, steps, every, energy, t0) |> elem(0), else: nil
          e0 = energy.(z0)
          err = fn tr -> tr |> Enum.map(&abs(&1.h - e0)) |> Enum.max() end
          slope = fn tr -> drift_slope(tr, e0) end
          ord = observed_order(method, fq, fp, n, z0, dt)
          back = reverse_error(stepper, zf, z0, dt, steps)
          laws = laws(qs, ps, dhdq, dhdp, s)
          expected = %{"verlet" => 2, "yoshida4" => 4, "midpoint" => 2, "rk4" => 4}[method]

          evidence = [
            %{check: "energy", ok: secular_ok(traj, control, e0, t1 - t0, slope, err), detail: "max |H − H₀| = #{fmt(err.(traj))}, drift slope #{fmt(slope.(traj))}/time" <>
              if(control, do: "; RK4 at the same dt (the control): #{fmt(err.(control))}, slope #{fmt(slope.(control))}", else: "")},
            %{check: "observed order", ok: ord != nil and abs(ord - expected) < 0.5, detail: "#{fmt(ord)} (the method's: #{expected})"},
            %{check: "reversibility", ok: method == "rk4" or back < 1.0e-6, detail: "forward then backward returns within #{fmt(back)}"}
          ] ++ if(laws, do: [%{check: "conservation laws", ok: laws.count > 0, detail: laws.says}], else: [])

          {:ok, %{kind: "hamiltonian", h: Expr.to_text(h), coordinates: qs, momenta: ps, separable: separable, method: method, dt: dt, steps: steps,
                  equations: Enum.map(Enum.zip(qs, dhdp), fn {q, d} -> "#{q}' = #{nice(d, names, 1)}" end) ++ Enum.map(Enum.zip(ps, dhdq), fn {p, d} -> "#{p}' = #{nice(d, names, -1)}" end),
                  trajectory: Enum.map(traj, fn r -> %{t: r.t, z: r.z, h: r.h} end), control: control && Enum.map(control, &%{t: &1.t, h: &1.h}),
                  laws: laws && laws.laws, evidence: evidence,
                  says: "#{method} (#{if separable, do: "separable", else: "non-separable"} H), #{steps} steps; energy error #{fmt(err.(traj))}" <> if(control, do: " vs RK4 #{fmt(err.(control))}", else: "")}}
      end
    else
      errs when is_list(errs) -> {:error, Enum.join(errs, "; ")}
      e -> e
    end
  end

  # a symplectic method's energy error oscillates without a trend: its fitted drift over the run stays
  # below its own oscillation, or far below the control's drift
  defp secular_ok(traj, nil, _e0, span, slope, err), do: abs(slope.(traj)) * span <= err.(traj)
  defp secular_ok(traj, control, _e0, span, slope, err), do: abs(slope.(traj)) * span <= err.(traj) or abs(slope.(traj)) * 10 < abs(slope.(control))

  defp zero?({:n, x}), do: x == 0
  defp zero?(_), do: false

  defp fetch_h(s) do
    case Map.fetch(s.funs, "H") do
      {:ok, {_args, t, _}} -> {:ok, t}
      :error -> {:error, "write the Hamiltonian as H = … in q, p (or q1…qn, p1…pn)"}
    end
  end

  defp coords(h) do
    vs = Expr.vars(h)
    qs = vs |> Enum.filter(&Regex.match?(~r/^q\d*$/, &1)) |> Enum.sort_by(&idx/1)
    ps = vs |> Enum.filter(&Regex.match?(~r/^p\d*$/, &1)) |> Enum.sort_by(&idx/1)
    other = vs -- (qs ++ ps)
    cond do
      other != [] -> {:error, "unknown name(s) in H: #{Enum.join(other, ", ")} (define them as constants, or use q/p, q1…qn/p1…pn)"}
      qs == [] -> {:error, "H has no coordinate (q or q1…)"}
      Enum.map(qs, &idx/1) != Enum.map(ps, &idx/1) -> {:error, "coordinates and momenta must pair up: #{Enum.join(qs, ", ")} with #{Enum.join(Enum.map(qs, &String.replace(&1, "q", "p")), ", ")}"}
      true -> {:ok, qs, ps}
    end
  end

  defp idx("q"), do: 0
  defp idx("p"), do: 0
  defp idx(<<_, n::binary>>), do: String.to_integer(n)

  defp compile(trees, names) do
    f = Expr.compile(trees, names)
    fn z -> f.(List.to_tuple(z)) |> then(fn t when is_tuple(t) -> Tuple.to_list(t); l -> l end) end
  end

  # ------------------------------------------------------------ integrators

  defp stepper("verlet", fq, fp, n), do: fn z, h -> leapfrog(z, h, fq, fp, n) end

  defp stepper("yoshida4", fq, fp, n) do
    c = :math.pow(2, 1 / 3)
    w1 = 1 / (2 - c)
    w0 = -c / (2 - c)
    fn z, h -> z |> leapfrog(w1 * h, fq, fp, n) |> leapfrog(w0 * h, fq, fp, n) |> leapfrog(w1 * h, fq, fp, n) end
  end

  defp stepper("midpoint", fq, fp, n) do
    fn z, h ->
      Enum.reduce_while(1..60, z, fn _, w ->
        mid = Enum.zip_with(z, w, &((&1 + &2) / 2))
        d = rhs(mid, fq, fp, n)
        w2 = Enum.zip_with(z, d, &(&1 + h * &2))
        if Enum.zip(w, w2) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max() < 1.0e-15 * (1 + Enum.max(Enum.map(w2, &abs/1))), do: {:halt, w2}, else: {:cont, w2}
      end)
    end
  end

  defp stepper("rk4", fq, fp, n) do
    fn z, h ->
      k1 = rhs(z, fq, fp, n)
      k2 = rhs(axpy(z, k1, h / 2), fq, fp, n)
      k3 = rhs(axpy(z, k2, h / 2), fq, fp, n)
      k4 = rhs(axpy(z, k3, h), fq, fp, n)
      Enum.zip_with([z, k1, k2, k3, k4], fn [y, a, b, c, d] -> y + h / 6 * (a + 2 * b + 2 * c + d) end)
    end
  end

  defp rhs(z, fq, fp, _n), do: fp.(z) ++ Enum.map(fq.(z), &(-&1))
  defp axpy(z, k, h), do: Enum.zip_with(z, k, &(&1 + h * &2))

  defp leapfrog(z, h, fq, fp, n) do
    {q, p} = Enum.split(z, n)
    p = Enum.zip_with(p, fq.(q ++ p), &(&1 - h / 2 * &2))
    q = Enum.zip_with(q, fp.(q ++ p), &(&1 + h * &2))
    p = Enum.zip_with(p, fq.(q ++ p), &(&1 - h / 2 * &2))
    q ++ p
  end

  defp integrate(step, z0, dt, steps, every, energy, t0) do
    {rows, z} =
      Enum.reduce(1..steps, {[%{t: t0, z: z0, h: energy.(z0)}], z0}, fn k, {rows, z} ->
        z = step.(z, dt)
        rows = if rem(k, every) == 0, do: [%{t: t0 + k * dt, z: z, h: energy.(z)} | rows], else: rows
        {rows, z}
      end)
    {Enum.reverse(rows), z}
  rescue
    ArithmeticError -> {[%{t: t0, z: z0, h: energy.(z0)}], z0}
  end

  defp drift_slope(rows, e0) do
    ts = Enum.map(rows, & &1.t)
    ys = Enum.map(rows, &(&1.h - e0))
    n = length(ts)
    mt = Enum.sum(ts) / n
    my = Enum.sum(ys) / n
    sxx = Enum.reduce(ts, 0.0, &(&2 + (&1 - mt) * (&1 - mt)))
    if sxx == 0, do: 0.0, else: Enum.zip(ts, ys) |> Enum.reduce(0.0, fn {t, y}, s -> s + (t - mt) * (y - my) end) |> Kernel./(sxx)
  end

  defp observed_order(method, fq, fp, n, z0, dt) do
    st = stepper(method, fq, fp, n)
    run = fn h, k -> Enum.reduce(1..k, z0, fn _, z -> st.(z, h) end) end
    k = 40
    a = run.(dt, k)
    b = run.(dt / 2, 2 * k)
    c = run.(dt / 4, 4 * k)
    e1 = dist(a, b)
    e2 = dist(b, c)
    if e2 > 0 and e1 > 0, do: :math.log2(e1 / e2), else: nil
  rescue
    _ -> nil
  end

  defp reverse_error(step, zf, z0, dt, steps) do
    back = Enum.reduce(1..steps, zf, fn _, z -> step.(z, -dt) end)
    dist(back, z0)
  rescue
    _ -> 1.0e300
  end

  defp dist(a, b), do: Enum.zip_with(a, b, &((&1 - &2) * (&1 - &2))) |> Enum.sum() |> :math.sqrt()

  defp laws(qs, ps, dhdq, dhdp, s) do
    eqq = Enum.map(Enum.zip(qs, dhdp), fn {q, d} -> "#{q}' = #{Expr.to_text(d)}" end)
    eqp = Enum.map(Enum.zip(ps, dhdq), fn {p, d} -> "#{p}' = -(#{Expr.to_text(d)})" end)
    inits = Enum.map(qs ++ ps, fn v -> "#{v}(0) = #{elem(s.inits[v], 1)}" end)
    hdeg = case Vapor.Crucible.Poly.from_expr(h_of(s), qs ++ ps) do {:ok, poly} -> Vapor.Crucible.Poly.degree(poly); _ -> 2 end
    lines = eqq ++ eqp ++ inits ++ ["t = 0 .. 5", "degree = #{s |> Sheet.const("degree", max(hdeg, 2) * 1.0) |> round() |> min(6)}"]
    case Laws.run(Enum.join(lines, "\n")) do
      {:ok, %{method: "exact" <> _} = r} -> r
      _ -> nil
    end
  rescue
    _ -> nil
  end

  # an equation's right-hand side, expanded exactly when it is a polynomial
  defp nice(d, names, sign) do
    case Vapor.Crucible.Poly.from_expr(d, names) do
      {:ok, p} -> p |> Vapor.Crucible.Poly.scale({sign, 1}) |> Vapor.Crucible.Poly.text(names)
      _ -> if sign == 1, do: Expr.to_text(d), else: "−(" <> Expr.to_text(d) <> ")"
    end
  end

  defp h_of(s), do: s.funs |> Map.fetch!("H") |> elem(1)

  defp fmt(nil), do: "—"
  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, [{:scientific, 2}])
  defp fmt(x), do: to_string(x)
end
