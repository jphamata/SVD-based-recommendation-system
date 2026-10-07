defmodule Vapor.Engineering.Process do
  @moduledoc """
  Chemical process calculations (docs/ENGENHARIA.md §6).

  **Reaction networks** written as chemists write them —

      A + B -> C ; k = 0.5
      C <-> D ; kf = 1, kb = 0.2
      2 A -> E ; k = 0.01*exp(-Ea/(R*T)) ; order A=2
      Ea = 50e3; R = 8.314; T = 350
      A0 = 1; B0 = 0.8
      reactor cstr tau=5 ; feed A=1, B=0.8     # batch (default) | cstr tau= | (pfr: batch in residence time)
      t = 0 .. 20

  — become mass-action ODEs solved by `Vapor.Solve` (switching to the
  stiff method by itself when the kinetics demand it). The
  **certificate** is computed from the stoichiometric matrix alone: its
  left null space gives the conserved combinations (elements, moieties,
  total moles where they hold); in a batch reactor each must stay
  constant along the solution, and their drift is reported.

  **Flash** (`flash/1`): an isothermal flash of an ideal mixture by
  Rachford–Rice, vapour pressures from Antoine's equation
  (log₁₀ P[mmHg] = A − B/(C + T[°C])); bubble and dew pressures; the
  material balance closed to the last digit is the certificate.

  **Distillation** (`distill/1`): a binary column at constant relative
  volatility by McCabe–Thiele, with Fenske's minimum stages, Underwood's
  minimum reflux and Gilliland's estimate beside the stage count — and,
  as the control, McCabe–Thiele at very large reflux, which must fall to
  Fenske.
  """
  alias Vapor.{Dense, Solve}

  # ============================================================ reactions

  @doc "Parse and integrate a reaction network. `{:ok, result}` or `{:error, why}`."
  def reactions(text) do
    with {:ok, net} <- parse_reactions(text), {:ok, ode} <- to_ode(net), {:ok, sol} <- Solve.ode(ode.text) do
      inv = invariants(net)
      species = net.species
      drift =
        if net.reactor == :batch do
          for v <- inv do
            series = Enum.map(0..(length(sol.t) - 1), fn k -> Enum.zip(species, v) |> Enum.reduce(0.0, fn {s, c}, acc -> acc + c * Enum.at(sol.series[s], k) end) end)
            %{combination: describe(species, v), initial: hd(series), drift: Enum.max(Enum.map(series, &abs(&1 - hd(series))))}
          end
        else
          []
        end

      {:ok, %{species: species, reactions: Enum.map(net.reactions, & &1.text), odes: ode.lines, t: sol.t, series: Map.take(sol.series, species),
              method: sol.method, switched: sol.switched, steps: sol.steps, reactor: Atom.to_string(net.reactor),
              stoichiometry: net.matrix, invariants: drift, final: Map.take(sol.final, species)}}
    end
  end

  @doc false
  def parse_reactions(text) do
    # a reaction keeps its ';' options on its line; any other line splits into statements
    stmts =
      text |> String.split("\n") |> Enum.with_index(1)
      |> Enum.flat_map(fn {l, n} ->
        l = l |> String.split("#", parts: 2) |> hd() |> String.trim()
        if l =~ ~r/(->|<->|⇌|→)/u, do: [{l, n}], else: l |> String.split(";") |> Enum.map(&{String.trim(&1), n}) |> Enum.reject(fn {x, _} -> x == "" end)
      end)

    Enum.reduce_while(stmts, {:ok, %{reactions: [], params: [], init: %{}, reactor: :batch, tau: nil, feed: %{}, span: "t = 0 .. 10", extra: []}}, fn {s, line}, {:ok, acc} ->
      cond do
        s =~ ~r/(->|<->|⇌|→)/u ->
          case reaction(s, text, line) do
            {:ok, r} -> {:cont, {:ok, %{acc | reactions: acc.reactions ++ [r]}}}
            e -> {:halt, e}
          end
        m = Regex.run(~r/^reactor\s+(batch|cstr|pfr)\s*(?:tau\s*=\s*(\S+))?$/iu, s) ->
          {:cont, {:ok, %{acc | reactor: String.to_atom(String.downcase(Enum.at(m, 1))), tau: Enum.at(m, 2)}}}
        m = Regex.run(~r/^feed\s+(.+)$/iu, s) ->
          f = Enum.at(m, 1) |> String.split(",") |> Enum.map(&String.split(&1, "=", parts: 2)) |> Enum.filter(&(length(&1) == 2)) |> Map.new(fn [k, v] -> {String.trim(k), String.trim(v)} end)
          {:cont, {:ok, %{acc | feed: f}}}
        s =~ ~r/^t\s*=/ -> {:cont, {:ok, %{acc | span: s}}}
        s =~ ~r/^(rtol|atol|method|samples)\s*=/ -> {:cont, {:ok, %{acc | extra: acc.extra ++ [s]}}}
        m = Regex.run(~r/^([\p{L}_][\p{L}\p{N}_]*)0\s*=\s*(.+)$/u, s) -> {:cont, {:ok, %{acc | init: Map.put(acc.init, Enum.at(m, 1), Enum.at(m, 2)), params: acc.params ++ [s]}}}
        s =~ ~r/^[\p{L}_][\p{L}\p{N}_]*\s*=/u -> {:cont, {:ok, %{acc | params: acc.params ++ [s]}}}
        true -> {:halt, {:error, "line #{line}: not understood: #{inspect(s)}"}}
      end
    end)
    |> case do
      {:ok, %{reactions: []}} -> {:error, "no reaction (write A + B -> C ; k = …)"}
      {:ok, acc} ->
        species = acc.reactions |> Enum.flat_map(&(Map.keys(&1.lhs) ++ Map.keys(&1.rhs))) |> Enum.uniq()
        matrix = for s <- species, do: for(r <- acc.reactions, do: Map.get(r.rhs, s, 0) - Map.get(r.lhs, s, 0))
        {:ok, Map.merge(acc, %{species: species, matrix: matrix})}
      e -> e
    end
  end

  # a reaction line is the whole source line (the ';' options follow it)
  defp reaction(_s, text, line) do
    src = text |> String.split("\n") |> Enum.at(line - 1) |> String.split("#", parts: 2) |> hd()
    [eq | opts] = String.split(src, ";") |> Enum.map(&String.trim/1)
    {arrow, rev} = cond do
      String.contains?(eq, "<->") -> {"<->", true}
      String.contains?(eq, "⇌") -> {"⇌", true}
      String.contains?(eq, "->") -> {"->", false}
      true -> {"→", false}
    end
    [l, r] = String.split(eq, arrow, parts: 2)
    o = opts |> Enum.flat_map(&String.split(&1, ",")) |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))
    kv = for item <- o, [k, v] <- [String.split(item, "=", parts: 2)], into: %{}, do: {String.trim(k), String.trim(v)}
    orders = for item <- o, String.starts_with?(item, "order"), m <- Regex.scan(~r/([\p{L}_][\p{L}\p{N}_]*)\s*=\s*([\d.]+)/u, item), into: %{}, do: {Enum.at(m, 1), Enum.at(m, 2)}
    orders = Map.merge(orders, for({"order " <> k, v} <- kv, into: %{}, do: {String.trim(k), v}))

    with {:ok, lhs} <- side(l), {:ok, rhs} <- side(r) do
      kf = kv["k"] || kv["kf"]
      cond do
        kf == nil -> {:error, "line #{line}: the rate constant is missing (; k = …)"}
        rev and kv["kb"] == nil -> {:error, "line #{line}: a reversible reaction needs kf = … and kb = …"}
        true -> {:ok, %{lhs: lhs, rhs: rhs, kf: kf, kb: kv["kb"], rev: rev, orders: orders, text: String.trim(eq)}}
      end
    end
  end

  defp side(s) do
    s |> String.split("+") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 in ["", "0", "∅"]))
    |> Enum.reduce_while({:ok, %{}}, fn term, {:ok, m} ->
      case Regex.run(~r/^(\d+)?\s*([\p{L}_][\p{L}\p{N}_]*)$/u, term) do
        [_, "", sp] -> {:cont, {:ok, Map.update(m, sp, 1, &(&1 + 1))}}
        [_, n, sp] -> {:cont, {:ok, Map.update(m, sp, String.to_integer(n), &(&1 + String.to_integer(n)))}}
        [_, sp] -> {:cont, {:ok, Map.update(m, sp, 1, &(&1 + 1))}}
        _ -> {:halt, {:error, "not a species term: #{inspect(term)}"}}
      end
    end)
  end

  defp to_ode(net) do
    rate = fn r ->
      fwd = "(#{r.kf})" <> Enum.map_join(r.lhs, "", fn {s, n} -> "*#{s}^#{Map.get(r.orders, s, n)}" end)
      if r.rev, do: fwd <> " - (#{r.kb})" <> Enum.map_join(r.rhs, "", fn {s, n} -> "*#{s}^#{n}" end), else: fwd
    end

    rates = Enum.map(net.reactions, rate)
    lines =
      for {s, row} <- Enum.zip(net.species, net.matrix) do
        terms = for {nu, rt} <- Enum.zip(row, rates), nu != 0, do: "#{if nu > 0, do: "+", else: "-"} #{abs(nu)}*(#{rt})"
        flow = case net.reactor do
          :cstr -> " + (#{Map.get(net.feed, s, "0")} - #{s})/(#{net.tau || "1"})"
          _ -> ""
        end
        "#{s}' = 0 #{Enum.join(terms, " ")}#{flow}"
      end

    if net.reactor == :cstr and net.tau == nil, do: throw({:proc, "a CSTR needs tau="})
    inits = for s <- net.species, do: "#{s}(0) = #{Map.get(net.init, s, "0")}"
    params = Enum.reject(net.params, &(&1 =~ ~r/^[\p{L}_][\p{L}\p{N}_]*0\s*=/u))
    {:ok, %{text: Enum.join(lines ++ params ++ inits ++ [net.span] ++ net.extra, "\n"), lines: lines}}
  catch
    {:proc, w} -> {:error, w}
  end

  @doc false
  # a basis of the left null space of the stoichiometric matrix (rational row reduction, integer vectors)
  def invariants(net) do
    n = net.matrix
    s = length(net.species)
    # solve vᵀ N = 0: v in the null space of Nᵀ
    nt = Dense.transpose(n) |> Enum.map(fn r -> Enum.map(r, &{&1, 1}) end)
    null_space(nt, s)
  end

  defp null_space(rows, ncols) do
    {rref, pivots} = rref(rows, ncols)
    free = Enum.reject(0..(ncols - 1), &(&1 in pivots))
    for f <- free do
      v = for c <- 0..(ncols - 1) do
        cond do
          c == f -> {1, 1}
          c in pivots -> (r = Enum.at(rref, Enum.find_index(pivots, &(&1 == c))); qneg(Enum.at(r, f)))
          true -> {0, 1}
        end
      end
      l = Enum.reduce(v, 1, fn {_, d}, acc -> lcm(acc, d) end)
      ints = Enum.map(v, fn {a, d} -> div(a * l, d) end)
      g = Enum.reduce(ints, 0, &Integer.gcd(&1, &2)) |> max(1)
      ints = Enum.map(ints, &div(&1, g))
      if Enum.find(ints, &(&1 != 0)) < 0, do: Enum.map(ints, &(-&1)), else: ints
    end
  end

  defp lcm(a, b), do: div(a * b, Integer.gcd(a, b))
  defp qnorm({a, b}) when b < 0, do: qnorm({-a, -b})
  defp qnorm({0, _}), do: {0, 1}
  defp qnorm({a, b}), do: (g = Integer.gcd(a, b); {div(a, g), div(b, g)})
  defp qneg({a, b}), do: qnorm({-a, b})
  defp qsub({a, b}, {c, d}), do: qnorm({a * d - c * b, b * d})
  defp qmul({a, b}, {c, d}), do: qnorm({a * c, b * d})
  defp qdiv({a, b}, {c, d}), do: qnorm({a * d, b * c})

  defp rref(rows, ncols) do
    Enum.reduce(0..(ncols - 1), {rows, [], 0}, fn c, {rows, piv, r} ->
      case Enum.find_index(Enum.drop(rows, r), fn row -> elem(Enum.at(row, c), 0) != 0 end) do
        nil -> {rows, piv, r}
        k ->
          rows = swap(rows, r, r + k)
          pr = Enum.at(rows, r)
          pv = Enum.at(pr, c)
          pr = Enum.map(pr, &qdiv(&1, pv))
          rows = rows |> List.replace_at(r, pr) |> Enum.with_index() |> Enum.map(fn {row, i} ->
            if i == r, do: row, else: (f = Enum.at(row, c); Enum.zip_with(row, pr, fn a, b -> qsub(a, qmul(f, b)) end))
          end)
          {rows, piv ++ [c], r + 1}
      end
    end)
    |> then(fn {rows, piv, _} -> {rows, piv} end)
  end

  defp swap(l, i, i), do: l
  defp swap(l, i, j), do: (a = Enum.at(l, i); b = Enum.at(l, j); l |> List.replace_at(i, b) |> List.replace_at(j, a))

  defp describe(species, v) do
    Enum.zip(v, species) |> Enum.reject(fn {c, _} -> c == 0 end) |> Enum.with_index()
    |> Enum.map_join("", fn {{c, s}, i} ->
      term = if abs(c) == 1, do: s, else: "#{abs(c)}·#{s}"
      cond do
        i == 0 and c < 0 -> "−" <> term
        i == 0 -> term
        c < 0 -> " − " <> term
        true -> " + " <> term
      end
    end)
  end

  # ================================================================= flash

  @doc """
  Isothermal flash:

      component benzene z=0.4 A=6.90565 B=1211.033 C=220.79
      component toluene z=0.6 A=6.95464 B=1344.8 C=219.482
      T = 95        # °C
      P = 760       # mmHg

  `{:ok, %{phase, vapour_fraction, x, y, K, bubble_p, dew_p, balance}}`.
  """
  def flash(text) do
    with {:ok, f} <- parse_flash(text) do
      psat = Enum.map(f.comps, fn c -> :math.pow(10, c.a - c.b / (c.c + f.t)) end)
      k = Enum.map(psat, &(&1 / f.p))
      z = f.comps |> Enum.map(& &1.z) |> normalise()
      bubble = Enum.zip(z, psat) |> Enum.map(fn {zi, p} -> zi * p end) |> Enum.sum()
      dew = 1 / (Enum.zip(z, psat) |> Enum.map(fn {zi, p} -> zi / p end) |> Enum.sum())
      rr = fn b -> Enum.zip(z, k) |> Enum.map(fn {zi, ki} -> zi * (ki - 1) / (1 + b * (ki - 1)) end) |> Enum.sum() end

      {phase, beta} =
        cond do
          f.p >= bubble -> {"liquid (subcooled)", 0.0}
          f.p <= dew -> {"vapour (superheated)", 1.0}
          true ->
            # rr is monotone decreasing on (0, 1) between the asymptotes: bisection to the last bit, then report
            b = Enum.reduce(1..200, {0.0, 1.0}, fn _, {lo, hi} -> (m = (lo + hi) / 2; if rr.(m) > 0, do: {m, hi}, else: {lo, m}) end) |> then(fn {lo, hi} -> (lo + hi) / 2 end)
            {"two phases", b}
        end

      x = Enum.zip(z, k) |> Enum.map(fn {zi, ki} -> zi / (1 + beta * (ki - 1)) end)
      y = Enum.zip(x, k) |> Enum.map(fn {xi, ki} -> xi * ki end)
      # one phase: the other is the incipient one (the first bubble, the first drop), normalised
      {x, y} = case phase do "two phases" -> {x, y}; "liquid (subcooled)" -> {z, normalise(y)}; _ -> {normalise(x), z} end
      bal = Enum.zip([z, x, y]) |> Enum.map(fn {zi, xi, yi} -> abs(zi - (1 - beta) * xi - beta * yi) end) |> Enum.max()
      names = Enum.map(f.comps, & &1.name)
      {:ok, %{phase: phase, vapour_fraction: beta, x: Map.new(Enum.zip(names, x)), y: Map.new(Enum.zip(names, y)), k: Map.new(Enum.zip(names, k)),
              bubble_p: bubble, dew_p: dew, p: f.p, t: f.t, certificate: %{balance: bal, sum_x: Enum.sum(x), sum_y: Enum.sum(y), rachford_rice: if(phase == "two phases", do: rr.(beta), else: nil)}}}
    end
  end

  defp normalise(z), do: (s = Enum.sum(z); Enum.map(z, &(&1 / s)))

  defp parse_flash(text) do
    Solve.statements(text)
    |> Enum.reduce_while({:ok, %{comps: [], t: nil, p: nil}}, fn {s, line}, {:ok, acc} ->
      cond do
        m = Regex.run(~r/^component\s+(\S+)\s+(.+)$/iu, s) ->
          o = for kvp <- String.split(Enum.at(m, 2)), [k, v] <- [String.split(kvp, "=", parts: 2)], {f, _} <- [Float.parse(v)], into: %{}, do: {String.downcase(k), f}
          if Enum.all?(~w(z a b c), &Map.has_key?(o, &1)), do: {:cont, {:ok, %{acc | comps: acc.comps ++ [%{name: Enum.at(m, 1), z: o["z"], a: o["a"], b: o["b"], c: o["c"]}]}}},
            else: {:halt, {:error, "line #{line}: a component needs z= A= B= C="}}
        m = Regex.run(~r/^T\s*=\s*([-\d.eE+]+)$/u, s) -> {:cont, {:ok, %{acc | t: elem(Float.parse(Enum.at(m, 1)), 0)}}}
        m = Regex.run(~r/^P\s*=\s*([-\d.eE+]+)$/u, s) -> {:cont, {:ok, %{acc | p: elem(Float.parse(Enum.at(m, 1)), 0)}}}
        true -> {:halt, {:error, "line #{line}: not understood: #{inspect(s)}"}}
      end
    end)
    |> case do
      {:ok, %{comps: c}} when length(c) < 2 -> {:error, "at least two components"}
      {:ok, %{t: nil}} -> {:error, "T = … (°C) is missing"}
      {:ok, %{p: nil}} -> {:error, "P = … (mmHg) is missing"}
      other -> other
    end
  end

  # =========================================================== distillation

  @doc """
  A binary column at constant relative volatility:

      alpha = 2.5; xF = 0.45; xD = 0.95; xB = 0.05; q = 1; R = 1.5   # or Rfactor = 1.3 (R = 1.3·Rmin)

  `{:ok, %{stages, feed_stage, fenske, rmin, gilliland, steps, control}}`.
  """
  def distill(text) do
    vals = for {s, _} <- Solve.statements(text), [k, v] <- [String.split(s, "=", parts: 2)], {f, _} <- [Float.parse(String.trim(v))], into: %{}, do: {String.trim(k), f}
    {a, xf, xd, xb, q} = {vals["alpha"], vals["xF"], vals["xD"], vals["xB"], vals["q"] || 1.0}

    cond do
      Enum.any?([a, xf, xd, xb], &is_nil/1) -> {:error, "give alpha, xF, xD, xB (and q, R or Rfactor)"}
      not (xb < xf and xf < xd) -> {:error, "need xB < xF < xD"}
      a <= 1 -> {:error, "alpha must exceed 1"}
      true ->
        nmin = :math.log(xd / (1 - xd) * (1 - xb) / xb) / :math.log(a)
        rmin = underwood(a, xf, xd, q)
        r = vals["R"] || (vals["Rfactor"] || 1.3) * rmin
        if r <= rmin, do: throw({:proc, "R = #{Float.round(r, 4)} is below the minimum reflux #{Float.round(rmin, 4)}: the column cannot make the split"})
        {n, feed, steps} = mccabe(a, xf, xd, xb, q, r)
        {nc, _, _} = mccabe(a, xf, xd, xb, q, 1.0e4)
        xg = (r - rmin) / (r + 1)
        y = 1 - :math.exp((1 + 54.4 * xg) / (11 + 117.2 * xg) * (xg - 1) / :math.sqrt(xg))
        {:ok, %{stages: n, feed_stage: feed, reflux: r, rmin: rmin, fenske: nmin, gilliland: (nmin + y) / (1 - y), steps: steps,
                control: %{total_reflux_stages: nc, fenske: nmin, note: "McCabe–Thiele at R = 10⁴ must step off ⌈Fenske⌉ (#{Float.round(nmin, 3)}) stages"}}}
    end
  catch
    {:proc, w} -> {:error, w}
  end

  defp eq_y(a, x), do: a * x / (1 + (a - 1) * x)
  defp eq_x(a, y), do: y / (a - (a - 1) * y)

  defp underwood(a, xf, xd, q) do
    # pinch: intersection of the q-line with the equilibrium curve
    xp = if abs(q - 1) < 1.0e-12, do: xf, else: bisect(fn x -> eq_y(a, x) - (q / (q - 1) * x - xf / (q - 1)) end, 1.0e-12, 1 - 1.0e-12)
    yp = eq_y(a, xp)
    (xd - yp) / (yp - xp)
  end

  defp bisect(f, lo, hi), do: Enum.reduce(1..200, {lo, hi}, fn _, {l, h} -> (m = (l + h) / 2; if f.(l) * f.(m) <= 0, do: {l, m}, else: {m, h}) end) |> then(fn {l, h} -> (l + h) / 2 end)

  # step off stages from the top; switch to the stripping line once past the feed
  defp mccabe(a, xf, xd, xb, q, r) do
    rect = fn x -> r / (r + 1) * x + xd / (r + 1) end
    # intersection of the operating lines (on the q-line)
    xi = if abs(q - 1) < 1.0e-12, do: xf, else: (xd / (r + 1) + xf / (q - 1)) / (q / (q - 1) - r / (r + 1))
    yi = rect.(xi)
    strip = fn x -> yi + (yi - xb) / (xi - xb) * (x - xi) end
    Enum.reduce_while(1..500, {xd, xd, [[xd, xd]], nil}, fn k, {_x, y, steps, feed} ->
      xn = eq_x(a, y)
      steps = steps ++ [[xn, y]]
      if xn <= xb do
        {:halt, {k, feed || k, steps}}
      else
        {feed, yn} = if xn > xi and feed == nil, do: {nil, rect.(xn)}, else: {feed || k, strip.(xn)}
        {:cont, {xn, yn, steps ++ [[xn, yn]], feed}}
      end
    end)
    |> case do
      {k, f, steps} -> {k, f, steps}
      _ -> {500, nil, []}
    end
  end
end
