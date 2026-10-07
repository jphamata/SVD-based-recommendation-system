defmodule Vapor.Engineering.Pipes do
  @moduledoc """
  Pressurised pipe networks — water supply, fire mains, district
  heating, process piping (docs/ENGENHARIA.md §5): Darcy–Weisbach losses
  with the Colebrook–White friction factor solved exactly (laminar 64/Re
  below Re 2000), minor losses, and the network solved by Newton on the
  joint system of flows and heads — the formulation of Todini and Pilati's
  global gradient algorithm, the one inside EPANET.

      fluid nu=1.0e-6[m^2/s]                       # kinematic viscosity (water at 20 °C)
      reservoir R head=60[m]
      junction A elev=10[m] demand=15[L/s]
      junction B elev=12[m] demand=20[L/s]
      pipe 1 R A L=800[m] D=250[mm] eps=0.1[mm]
      pipe 2 A B L=600[m] D=200[mm] eps=0.1[mm] K=2   # K: sum of minor-loss coefficients
      pipe 3 R B L=1200[m] D=200[mm] eps=0.1[mm]

  The answer: every pipe's flow, velocity, Reynolds number, friction
  factor and head loss; every junction's head and pressure — with the
  **certificate** recomputed from the answer, not from the solver: the
  continuity residual at each junction and the energy residual around
  every independent loop (found from a spanning tree).
  """
  alias Vapor.{Dense, Expr}

  @g 9.80665

  def run(text) do
    with {:ok, n} <- parse(text), do: solve(n)
  end

  def parse(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.reduce_while({:ok, %{nu: 1.0e-6, reservoirs: %{}, junctions: %{}, pipes: []}}, fn {raw, n}, {:ok, acc} ->
      l = raw |> String.split("#", parts: 2) |> hd() |> String.trim()
      res =
        case String.split(l) do
          [] -> {:ok, acc}
          ["fluid" | kv] -> with({:ok, o} <- kv(kv), do: {:ok, %{acc | nu: o["nu"] || acc.nu}})
          ["reservoir", id | kv] -> with({:ok, o} <- kv(kv), do: (if o["head"], do: {:ok, %{acc | reservoirs: Map.put(acc.reservoirs, id, o["head"])}}, else: {:error, "reservoir needs head="}))
          ["junction", id | kv] -> with({:ok, o} <- kv(kv), do: {:ok, %{acc | junctions: Map.put(acc.junctions, id, %{elev: o["elev"] || 0.0, demand: o["demand"] || 0.0})}})
          ["pipe", id, a, b | kv] ->
            with {:ok, o} <- kv(kv) do
              if o["L"] && o["D"], do: {:ok, %{acc | pipes: acc.pipes ++ [%{id: id, from: a, to: b, l: o["L"], d: o["D"], eps: o["eps"] || 0.0, k: o["K"] || 0.0}]}}, else: {:error, "pipe needs L= and D="}
            end
          [w | _] -> {:error, "not understood: #{w} (fluid, reservoir, junction, pipe)"}
        end
      case res do
        {:ok, acc} -> {:cont, {:ok, acc}}
        {:error, w} -> {:halt, {:error, "line #{n}: #{w}"}}
      end
    end)
    |> case do
      {:ok, n} ->
        known = Map.keys(n.reservoirs) ++ Map.keys(n.junctions)
        bad = n.pipes |> Enum.flat_map(&[&1.from, &1.to]) |> Enum.uniq() |> Enum.reject(&(&1 in known))
        cond do
          map_size(n.reservoirs) == 0 -> {:error, "at least one reservoir (a fixed head) is needed"}
          bad != [] -> {:error, "undefined node(s): #{Enum.join(bad, ", ")}"}
          n.pipes == [] -> {:error, "no pipes"}
          true -> {:ok, n}
        end
      e -> e
    end
  end

  defp kv(list) do
    Enum.reduce_while(list, {:ok, %{}}, fn item, {:ok, m} ->
      with [k, v] <- String.split(item, "=", parts: 2), {:ok, t} <- Expr.parse(v), [] <- Expr.vars(t) do
        {:cont, {:ok, Map.put(m, k, Expr.eval(t))}}
      else
        _ -> {:halt, {:error, "expected key=value (with optional [unit]): #{item}"}}
      end
    end)
  end

  @doc "Colebrook–White friction factor (exact, by Newton on 1/√f); 64/Re in laminar flow."
  def friction(re, rel) do
    cond do
      re < 1.0e-9 -> 0.0
      re < 2000 -> 64 / re
      true ->
        # Swamee–Jain as the starting point, then Newton on x = 1/√f
        f0 = 0.25 / :math.pow(:math.log10(rel / 3.7 + 5.74 / :math.pow(re, 0.9)), 2)
        x = Enum.reduce_while(1..50, 1 / :math.sqrt(f0), fn _, x ->
          g = x + 2 * :math.log10(rel / 3.7 + 2.51 * x / re)
          dg = 1 + 2 * 2.51 / (re * :math.log(10) * (rel / 3.7 + 2.51 * x / re))
          xn = x - g / dg
          if abs(xn - x) < 1.0e-14 * x, do: {:halt, xn}, else: {:cont, xn}
        end)
        1 / (x * x)
    end
  end

  # head loss of a pipe at flow q (signed), and its derivative (the friction factor frozen, as in GGA)
  defp loss(p, q, nu) do
    a = :math.pi() * p.d * p.d / 4
    v = abs(q) / a
    re = v * p.d / nu
    f = friction(re, p.eps / p.d)
    r = (f * p.l / p.d + p.k) / (2 * @g * a * a)
    h = r * q * abs(q)
    {h, max(2 * r * abs(q), 1.0e-9), %{v: v, re: re, f: f}}
  end

  defp solve(n) do
    jids = n.junctions |> Map.keys() |> Enum.sort()
    jidx = jids |> Enum.with_index() |> Map.new()
    np = length(n.pipes)
    nj = length(jids)
    head = fn h, id -> (case Map.fetch(n.reservoirs, id) do {:ok, v} -> v; :error -> Enum.at(h, jidx[id]) end) end
    q0 = Enum.map(n.pipes, &(0.5 * :math.pi() * &1.d * &1.d / 4))
    h0 = Enum.map(jids, fn _ -> n.reservoirs |> Map.values() |> Enum.max() end)

    case newton(n, jids, jidx, head, q0, h0, 0) do
      {:ok, q, h, its} -> report(n, jids, jidx, head, q, h, its, np, nj)
      e -> e
    end
  end

  defp newton(n, jids, jidx, head, q, h, its) do
    np = length(n.pipes)
    nj = length(jids)
    ls = Enum.zip_with(n.pipes, q, &loss(&1, &2, n.nu))
    # pipe rows: H_from − H_to − h(Q) = 0 ; junction rows: Σin − Σout − demand = 0
    rp = for {p, {hl, _, _}} <- Enum.zip(n.pipes, ls), do: head.(h, p.from) - head.(h, p.to) - hl
    rj = for j <- jids, do: Enum.reduce(Enum.zip(n.pipes, q), 0.0, fn {p, qq}, s -> s + (if p.to == j, do: qq, else: 0.0) - (if p.from == j, do: qq, else: 0.0) end) - n.junctions[j].demand
    res = rp ++ rj

    if Dense.norm_inf(res) < 1.0e-11 or its > 60 do
      if its > 60, do: {:error, "the network did not converge (a junction with no path to a reservoir?)"}, else: {:ok, q, h, its}
    else
      jac =
        (for {pp, {_, dh, _}} <- Enum.zip(n.pipes, ls) do
          qrow = for p2 <- n.pipes, do: if(p2.id == pp.id, do: -dh, else: 0.0)
          hrow = for j <- jids, do: (if pp.from == j, do: 1.0, else: 0.0) - (if pp.to == j, do: 1.0, else: 0.0)
          qrow ++ hrow
        end) ++
        (for j <- jids do
          (for p <- n.pipes, do: (if p.to == j, do: 1.0, else: 0.0) - (if p.from == j, do: 1.0, else: 0.0)) ++ List.duplicate(0.0, nj)
        end)

      case Dense.solve(jac, Enum.map(res, &(-&1))) do
        {:ok, dx} ->
          {dq, dh} = Enum.split(dx, np)
          newton(n, jids, jidx, head, Enum.zip_with(q, dq, &(&1 + &2)), Enum.zip_with(h, dh, &(&1 + &2)), its + 1)
        _ -> {:error, "the network matrix is singular: some junction is not connected to a reservoir"}
      end
    end
  end

  defp report(n, jids, _jidx, head, q, h, its, _np, _nj) do
    rho_g = 998.2 * @g
    pipes = for {p, qq} <- Enum.zip(n.pipes, q) do
      {hl, _, st} = loss(p, qq, n.nu)
      %{id: p.id, from: p.from, to: p.to, flow: qq, velocity: st.v, reynolds: st.re, friction: st.f, headloss: hl,
        regime: cond do st.re < 2000 -> "laminar"; st.re < 4000 -> "transitional"; true -> "turbulent" end}
    end
    junctions = for {j, i} <- Enum.with_index(jids), do: (hh = Enum.at(h, i); %{id: j, head: hh, pressure: (hh - n.junctions[j].elev) * rho_g, pressure_head: hh - n.junctions[j].elev})
    continuity = for j <- jids, do: abs(Enum.reduce(pipes, 0.0, fn p, s -> s + (if p.to == j, do: p.flow, else: 0.0) - (if p.from == j, do: p.flow, else: 0.0) end) - n.junctions[j].demand)
    loops = loops(n)
    hl = Map.new(pipes, &{&1.id, &1.headloss})
    energy = for lp <- loops, do: abs(Enum.reduce(lp, 0.0, fn {pid, s}, acc -> acc + s * hl[pid] end))
    # loops through two reservoirs: the head difference closes the path
    {:ok, %{pipes: pipes, junctions: junctions, iterations: its, loops: length(loops),
            certificate: %{continuity_max: Enum.max(continuity, fn -> 0.0 end), loop_energy_max: Enum.max(energy, fn -> 0.0 end),
                           reservoir_paths: path_energy(n, pipes, head, h)}}}
  end

  # independent loops: each pipe a spanning tree leaves out, closed by the tree path back (signed pipe lists)
  defp loops(n) do
    {tree, extra, _} =
      Enum.reduce(n.pipes, {[], [], %{}}, fn p, {tree, extra, uf} ->
        {ra, uf} = find(uf, p.from)
        {rb, uf} = find(uf, p.to)
        if ra == rb, do: {tree, [p | extra], uf}, else: {[p | tree], extra, Map.put(uf, ra, rb)}
      end)

    adj = Enum.reduce(tree, %{}, fn p, a -> a |> Map.update(p.from, [{p.to, p, 1.0}], &[{p.to, p, 1.0} | &1]) |> Map.update(p.to, [{p.from, p, -1.0}], &[{p.from, p, -1.0} | &1]) end)
    for p <- extra, do: [{p.id, 1.0} | tree_path(adj, p.to, p.from)]
  end

  defp find(uf, x) do
    case Map.fetch(uf, x) do
      {:ok, y} when y != x -> (({r, uf} = find(uf, y)); {r, Map.put(uf, x, r)})
      _ -> {x, uf}
    end
  end

  # breadth-first path in the tree from a to b: [{pipe, +1 if walked from → to}]
  defp tree_path(adj, a, b), do: bfs([{a, []}], MapSet.new([a]), adj, b)

  defp bfs([], _seen, _adj, _b), do: []
  defp bfs([{x, path} | rest], seen, adj, b) do
    if x == b do
      Enum.reverse(path)
    else
      nexts = for {y, p, s} <- Map.get(adj, x, []), not MapSet.member?(seen, y), do: {y, [{p.id, s} | path]}
      bfs(rest ++ nexts, Enum.reduce(nexts, seen, fn {y, _}, acc -> MapSet.put(acc, y) end), adj, b)
    end
  end

  # for every pipe path between two reservoirs: Δ(reservoir heads) = Σ head losses; checked pipe by pipe as H_from − H_to − h
  defp path_energy(n, pipes, head, h) do
    pipes |> Enum.map(fn p -> abs(head.(h, p.from) - head.(h, p.to) - p.headloss) end) |> Enum.max(fn -> 0.0 end) |> then(&(&1 + 0.0 * map_size(n.reservoirs)))
  end
end
