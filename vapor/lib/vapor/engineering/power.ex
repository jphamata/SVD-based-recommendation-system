defmodule Vapor.Engineering.Power do
  @moduledoc """
  AC power flow of a transmission or distribution network
  (docs/ENGENHARIA.md §2): Newton–Raphson in polar coordinates with the
  full Jacobian, the classic of every utility's planning study; and
  Gauss–Seidel, the textbook method, as the control that must reach the
  same voltages in many more iterations.

      base 100
      bus 1 slack V=1.06 a=0
      bus 2 pv P=40 V=1.045         # MW injected (generation − load)
      bus 3 pq P=-45 Q=-15          # MW, Mvar (a load is negative)
      line 1 2 r=0.02 x=0.06 b=0.06 # per unit on the base; b the total charging
      line 2 3 r=0.06 x=0.18 b=0.04
      shunt 3 b=0.1                 # optional, per unit

  The result: every bus's |V| and angle, P and Q injections, each line's
  flows at both ends and its losses — and the **certificate**: the
  power mismatch at every bus, recomputed from the admittances, and the
  balance (generation − load = losses).
  """
  alias Vapor.Dense

  @doc "Parse and solve. `{:ok, result}` or `{:error, why}`."
  def run(text, opts \\ []) do
    with {:ok, net} <- parse(text) do
      solve(net, Keyword.get(opts, :method, :newton))
    end
  end

  # =============================================================== parsing

  def parse(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.reduce_while({:ok, %{base: 100.0, buses: [], lines: [], shunts: []}}, fn {raw, n}, {:ok, acc} ->
      l = raw |> String.split("#", parts: 2) |> hd() |> String.trim()
      toks = String.split(l)
      res =
        case toks do
          [] -> {:ok, acc}
          ["base", b] -> with({:ok, v} <- num(b), do: {:ok, %{acc | base: v}})
          ["bus", id, kind | kvs] when kind in ["slack", "pv", "pq"] ->
            with {:ok, o} <- kvs(kvs), do: {:ok, %{acc | buses: acc.buses ++ [%{id: id, kind: String.to_atom(kind), v: o["v"] || 1.0, a: (o["a"] || 0.0) * :math.pi() / 180, p: o["p"] || 0.0, q: o["q"] || 0.0}]}}
          ["line", a, b | kvs] -> with({:ok, o} <- kvs(kvs), do: (if (o["r"] || 0.0) == 0.0 and (o["x"] || 0.0) == 0.0, do: {:error, "a line needs r or x"}, else: {:ok, %{acc | lines: acc.lines ++ [%{from: a, to: b, r: o["r"] || 0.0, x: o["x"] || 0.0, b: o["b"] || 0.0, tap: o["tap"] || 1.0}]}}))
          ["shunt", id | kvs] -> with({:ok, o} <- kvs(kvs), do: {:ok, %{acc | shunts: acc.shunts ++ [%{id: id, g: o["g"] || 0.0, b: o["b"] || 0.0}]}})
          _ -> {:error, "not understood: #{inspect(l)}"}
        end
      case res do
        {:ok, acc} -> {:cont, {:ok, acc}}
        {:error, w} -> {:halt, {:error, "line #{n}: #{w}"}}
      end
    end)
    |> case do
      {:ok, net} -> check(net)
      e -> e
    end
  end

  defp check(net) do
    ids = Enum.map(net.buses, & &1.id)
    slack = Enum.count(net.buses, &(&1.kind == :slack))
    unknown = (Enum.flat_map(net.lines, &[&1.from, &1.to]) ++ Enum.map(net.shunts, & &1.id)) |> Enum.uniq() |> Enum.reject(&(&1 in ids))
    cond do
      slack != 1 -> {:error, "exactly one slack bus is needed (#{slack} given)"}
      unknown != [] -> {:error, "unknown bus(es): #{Enum.join(unknown, ", ")}"}
      length(Enum.uniq(ids)) != length(ids) -> {:error, "a bus id repeats"}
      true -> {:ok, net}
    end
  end

  defp num(s), do: (case Float.parse(s) do {v, ""} -> {:ok, v}; _ -> {:error, "not a number: #{s}"} end)

  defp kvs(list) do
    Enum.reduce_while(list, {:ok, %{}}, fn kv, {:ok, m} ->
      case String.split(kv, "=", parts: 2) do
        [k, v] -> (case num(v) do {:ok, x} -> {:cont, {:ok, Map.put(m, String.downcase(k), x)}}; e -> {:halt, e} end)
        _ -> {:halt, {:error, "expected key=value, got #{kv}"}}
      end
    end)
  end

  # ============================================================== admittance

  @doc false
  def ybus(net) do
    idx = net.buses |> Enum.map(& &1.id) |> Enum.with_index() |> Map.new()
    n = map_size(idx)
    y0 = for _ <- 1..n, do: for(_ <- 1..n, do: {0.0, 0.0})
    put = fn y, i, j, v -> List.update_at(y, i, fn r -> List.update_at(r, j, &Dense.cadd(&1, v)) end) end

    y = Enum.reduce(net.lines, y0, fn l, y ->
      {i, j} = {idx[l.from], idx[l.to]}
      ys = Dense.cdiv({1.0, 0.0}, {l.r, l.x})
      bsh = {0.0, l.b / 2}
      t = l.tap
      y |> put.(i, i, Dense.cadd({elem(ys, 0) / (t * t), elem(ys, 1) / (t * t)}, bsh)) |> put.(j, j, Dense.cadd(ys, bsh))
      |> put.(i, j, {-elem(ys, 0) / t, -elem(ys, 1) / t}) |> put.(j, i, {-elem(ys, 0) / t, -elem(ys, 1) / t})
    end)

    y = Enum.reduce(net.shunts, y, fn s, y -> put.(y, idx[s.id], idx[s.id], {s.g, s.b}) end)
    {y, idx}
  end

  # injections S = V conj(Y V) at every bus
  defp injections(y, v, a) do
    n = length(v)
    vt = List.to_tuple(v)
    at = List.to_tuple(a)
    for i <- 0..(n - 1) do
      row = Enum.at(y, i)
      Enum.reduce(0..(n - 1), {0.0, 0.0}, fn k, {p, q} ->
        {g, b} = Enum.at(row, k)
        th = elem(at, i) - elem(at, k)
        vv = elem(vt, i) * elem(vt, k)
        {p + vv * (g * :math.cos(th) + b * :math.sin(th)), q + vv * (g * :math.sin(th) - b * :math.cos(th))}
      end)
    end
  end

  # ================================================================== solve

  defp solve(net, method) do
    {y, idx} = ybus(net)
    base = net.base
    buses = net.buses
    v0 = Enum.map(buses, &(if &1.kind == :pq, do: 1.0, else: &1.v))
    a0 = Enum.map(buses, &(if &1.kind == :slack, do: &1.a, else: 0.0))
    psp = Enum.map(buses, &(&1.p / base))
    qsp = Enum.map(buses, &(&1.q / base))

    result =
      case method do
        :newton -> newton(y, buses, v0, a0, psp, qsp, 0)
        :gauss_seidel -> gauss_seidel(y, buses, v0, a0, psp, qsp, 0)
      end

    with {:ok, v, a, its} <- result do
      s = injections(y, v, a)
      mism =
        for {{p, q}, bus, ps, qs} <- Enum.zip([s, buses, psp, qsp]) do
          case bus.kind do
            :slack -> 0.0
            :pv -> abs(p - ps)
            :pq -> max(abs(p - ps), abs(q - qs))
          end
        end

      flows =
        for l <- net.lines do
          {i, j} = {idx[l.from], idx[l.to]}
          {sij, sji} = line_flow(l, Enum.at(v, i), Enum.at(a, i), Enum.at(v, j), Enum.at(a, j))
          %{from: l.from, to: l.to, p_from: elem(sij, 0) * base, q_from: elem(sij, 1) * base, p_to: elem(sji, 0) * base, q_to: elem(sji, 1) * base,
            loss_mw: (elem(sij, 0) + elem(sji, 0)) * base}
        end

      gen = s |> Enum.map(&elem(&1, 0)) |> Enum.sum()
      losses = flows |> Enum.map(& &1.loss_mw) |> Enum.sum()
      shunt_p = Enum.reduce(net.shunts, 0.0, fn sh, acc -> acc + sh.g * Enum.at(v, idx[sh.id]) ** 2 end) * base

      {:ok, %{method: if(method == :newton, do: "Newton–Raphson (polar)", else: "Gauss–Seidel"), iterations: its, base_mva: base,
              buses: for({bus, vi, ai, {p, q}} <- Enum.zip([buses, v, a, s]), do: %{id: bus.id, kind: bus.kind, v: vi, angle_deg: ai * 180 / :math.pi(), p_mw: p * base, q_mvar: q * base}),
              lines: flows, losses_mw: losses,
              certificate: %{max_mismatch_pu: Enum.max(mism), balance_mw: abs(gen * base - losses - shunt_p)}}}
    end
  end

  defp line_flow(l, vi, ai, vj, aj) do
    ys = Dense.cdiv({1.0, 0.0}, {l.r, l.x})
    t = l.tap
    ei = {vi * :math.cos(ai) / t, vi * :math.sin(ai) / t}
    ej = {vj * :math.cos(aj), vj * :math.sin(aj)}
    iij = Dense.cadd(Dense.cmul(ys, Dense.csub(ei, ej)), Dense.cmul({0.0, l.b / 2}, ei))
    iji = Dense.cadd(Dense.cmul(ys, Dense.csub(ej, ei)), Dense.cmul({0.0, l.b / 2}, ej))
    conj = fn {r, i} -> {r, -i} end
    {Dense.cmul(ei, conj.(iij)), Dense.cmul(ej, conj.(iji))}
  end

  defp newton(y, buses, v, a, psp, qsp, its) do
    n = length(buses)
    s = injections(y, v, a)
    pv_pq = for {b, i} <- Enum.with_index(buses), b.kind != :slack, do: i
    pq = for {b, i} <- Enum.with_index(buses), b.kind == :pq, do: i
    dp = for i <- pv_pq, do: Enum.at(psp, i) - elem(Enum.at(s, i), 0)
    dq = for i <- pq, do: Enum.at(qsp, i) - elem(Enum.at(s, i), 1)
    mis = Dense.norm_inf(dp ++ dq)

    cond do
      mis < 1.0e-12 -> {:ok, v, a, its}
      its >= 40 -> {:error, "Newton–Raphson did not converge (mismatch #{mis} p.u. after 40 iterations): the case may have no solution (voltage collapse)"}
      true ->
        g = fn i, k -> elem(Enum.at(Enum.at(y, i), k), 0) end
        b = fn i, k -> elem(Enum.at(Enum.at(y, i), k), 1) end
        vt = List.to_tuple(v); at = List.to_tuple(a)
        {pi, qi} = {fn i -> elem(Enum.at(s, i), 0) end, fn i -> elem(Enum.at(s, i), 1) end}
        th = fn i, k -> elem(at, i) - elem(at, k) end
        vv = fn i -> elem(vt, i) end
        # ∂P/∂θ, ∂P/∂V, ∂Q/∂θ, ∂Q/∂V (polar, with ∂/∂V not scaled by V)
        dpdt = fn i, k -> if i == k, do: -qi.(i) - b.(i, i) * vv.(i) ** 2, else: vv.(i) * vv.(k) * (g.(i, k) * :math.sin(th.(i, k)) - b.(i, k) * :math.cos(th.(i, k))) end
        dpdv = fn i, k -> if i == k, do: pi.(i) / vv.(i) + g.(i, i) * vv.(i), else: vv.(i) * (g.(i, k) * :math.cos(th.(i, k)) + b.(i, k) * :math.sin(th.(i, k))) end
        dqdt = fn i, k -> if i == k, do: pi.(i) - g.(i, i) * vv.(i) ** 2, else: -vv.(i) * vv.(k) * (g.(i, k) * :math.cos(th.(i, k)) + b.(i, k) * :math.sin(th.(i, k))) end
        dqdv = fn i, k -> if i == k, do: qi.(i) / vv.(i) - b.(i, i) * vv.(i), else: vv.(i) * (g.(i, k) * :math.sin(th.(i, k)) - b.(i, k) * :math.cos(th.(i, k))) end
        j = (for i <- pv_pq, do: (for(k <- pv_pq, do: dpdt.(i, k)) ++ for(k <- pq, do: dpdv.(i, k)))) ++
            (for i <- pq, do: (for(k <- pv_pq, do: dqdt.(i, k)) ++ for(k <- pq, do: dqdv.(i, k))))
        case Dense.solve(j, dp ++ dq) do
          {:ok, dx} ->
            {da, dv} = Enum.split(dx, length(pv_pq))
            a = Enum.reduce(Enum.zip(pv_pq, da), a, fn {i, d}, a -> List.update_at(a, i, &(&1 + d)) end)
            v = Enum.reduce(Enum.zip(pq, dv), v, fn {i, d}, v -> List.update_at(v, i, &(&1 + d)) end)
            _ = n
            newton(y, buses, v, a, psp, qsp, its + 1)
          _ -> {:error, "the power-flow Jacobian is singular (an islanded bus, or the operating point is at the nose of the PV curve)"}
        end
    end
  end

  defp gauss_seidel(y, buses, v, a, psp, qsp, its) do
    e = Enum.zip_with(v, a, fn m, th -> {m * :math.cos(th), m * :math.sin(th)} end)
    gs_loop(y, buses, e, psp, qsp, its)
  end

  defp gs_loop(y, buses, e, psp, qsp, its) do
    n = length(buses)
    e2 =
      Enum.reduce(0..(n - 1), e, fn i, e ->
        bus = Enum.at(buses, i)
        if bus.kind == :slack do
          e
        else
          row = Enum.at(y, i)
          sum = Enum.reduce(0..(n - 1), {0.0, 0.0}, fn k, acc -> if k == i, do: acc, else: Dense.cadd(acc, Dense.cmul(Enum.at(row, k), Enum.at(e, k))) end)
          ei = Enum.at(e, i)
          q = if bus.kind == :pv do
            # Q from the current estimate: Im{ conj(E_i) · (Y_ii E_i + Σ) }, sign per S = E conj(I)
            ii = Dense.cadd(Dense.cmul(Enum.at(row, i), ei), sum)
            -elem(Dense.cmul({elem(ei, 0), -elem(ei, 1)}, ii), 1)
          else
            Enum.at(qsp, i)
          end
          s_conj = {Enum.at(psp, i), -q}
          new = Dense.cdiv(Dense.csub(Dense.cdiv(s_conj, {elem(ei, 0), -elem(ei, 1)}), sum), Enum.at(row, i))
          new = if bus.kind == :pv, do: (m = Dense.cabs(new); {elem(new, 0) * bus.v / m, elem(new, 1) * bus.v / m}), else: new
          List.replace_at(e, i, new)
        end
      end)

    d = Enum.zip(e, e2) |> Enum.map(fn {x, z} -> Dense.cabs(Dense.csub(x, z)) end) |> Enum.max()
    cond do
      d < 1.0e-13 -> {:ok, Enum.map(e2, &Dense.cabs/1), Enum.map(e2, &Dense.carg/1), its + 1}
      its >= 20_000 -> {:error, "Gauss–Seidel did not converge in 20 000 sweeps"}
      true -> gs_loop(y, buses, e2, psp, qsp, its + 1)
    end
  end
end
