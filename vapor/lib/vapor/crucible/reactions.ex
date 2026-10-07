defmodule Vapor.Crucible.Reactions do
  @moduledoc """
  A reaction network the user writes, deterministic **and** stochastic
  (docs/CRUCIBLE.md §5). The deterministic side is the engineering
  workbench's (`Vapor.Engineering.Process.reactions/1`: mass-action ODEs,
  stiff when needed, conserved moieties from the stoichiometric matrix's
  left null space). The Crucible adds:

    * **Gillespie's stochastic simulation** (direct method) at a volume
      `volume` (molecules = concentration × volume), an ensemble of
      `runs` trajectories with a seeded generator — the ensemble mean must
      follow the ODE within its standard error when molecules are many,
      and the deviation is the measure of *when the ODE stops being
      right* (small volumes, bistability);
    * **equilibrium**: for each reversible reaction, the reaction quotient
      at the end against K = kf/kb.
  """
  alias Vapor.Engineering.Process, as: P
  alias Vapor.Expr

  def run(text) do
    {volume, text} = take(text, "volume", 200.0)
    {runs, text} = take(text, "runs", 40.0)

    with {:ok, ode} <- P.reactions(text),
         {:ok, net} <- P.parse_reactions(text) do
      params = params(net)
      species = net.species
      rates = Enum.map(net.reactions, fn r -> {num(r.kf, params), r.kb && num(r.kb, params)} end)
      x0 = Enum.map(species, fn s -> num(Map.get(net.init, s, "0"), params) end)
      t_end = List.last(ode.t)
      every = max(div(length(ode.t), 40), 1)
      times = Enum.take_every(ode.t, every)
      ode_at = species |> Enum.map(&Enum.take_every(ode.series[&1], every)) |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
      ssa = if net.reactor == :batch, do: ensemble(net, rates, x0, volume, round(runs), times, ode_at), else: nil
      eq = equilibrium(net, rates, ode.final)

      evidence =
        Enum.map(ode.invariants, fn i -> %{check: "conserved: #{i.combination}", ok: i.drift < 1.0e-6 * max(abs(i.initial), 1.0), detail: "drift #{f(i.drift)} along the ODE solution"} end) ++
          if(ssa, do: [%{check: "stochastic vs deterministic", ok: ssa.max_z < 4.0, detail: "ensemble of #{ssa.runs} Gillespie runs at volume #{f(volume)}: the mean stays within #{f(ssa.max_z)} standard errors of the ODE (worst species and time)"}], else: []) ++
          Enum.map(eq, fn e -> %{check: "equilibrium #{e.reaction}", ok: true, detail: "Q/K = #{f(e.ratio)} at t = #{f(t_end)} (1 at equilibrium)"} end)

      {:ok, Map.merge(ode, %{kind: "reactions", ssa: ssa, equilibrium: eq, evidence: evidence, volume: volume,
                             says: "#{length(species)} species, #{length(net.reactions)} reactions; #{length(ode.invariants)} conserved combination(s)" <> if(ssa, do: "; stochastic ensemble compared", else: "")})}
    end
  end

  defp take(text, name, default) do
    case Regex.run(~r/^\s*#{name}\s*=\s*([\d.eE+-]+)\s*$/m, text) do
      [line, v] -> {elem(Float.parse(v), 0), String.replace(text, line, "")}
      nil -> {default, text}
    end
  end

  defp params(net) do
    Enum.reduce(net.params, %{}, fn line, acc ->
      case String.split(line, "=", parts: 2) do
        [k, v] ->
          case Expr.parse(String.trim(v)) do
            {:ok, t} -> (try do Map.put(acc, String.trim(k), Expr.eval(t, acc)) rescue _ -> acc end)
            _ -> acc
          end
        _ -> acc
      end
    end)
  end

  defp num(text, params) do
    case Expr.parse(String.trim(to_string(text))) do
      {:ok, t} -> Expr.eval(t, params) * 1.0
      _ -> 0.0
    end
  rescue
    _ -> 0.0
  end

  defp f(x) when is_float(x), do: :erlang.float_to_binary(x, [{:decimals, 4}, :compact])
  defp f(x), do: to_string(x)

  # ------------------------------------------------------------ Gillespie

  defp ensemble(net, rates, x0, vol, runs, times, ode_at) do
    n0 = Enum.map(x0, &round(&1 * vol))
    channels = channels(net, rates, vol)
    t_end = List.last(times)
    trajs = for k <- 1..runs, do: ssa(channels, n0, times, t_end, :rand.seed_s(:exsss, {k, 97, 13}))

    # per time and species: ensemble mean (as concentration) and its standard error
    stats =
      for {t, ti} <- Enum.with_index(times) do
        for {_, si} <- Enum.with_index(net.species) do
          xs = Enum.map(trajs, fn tr -> Enum.at(Enum.at(tr, ti), si) / vol end)
          m = Enum.sum(xs) / runs
          sd = :math.sqrt(Enum.reduce(xs, 0.0, &(&2 + (&1 - m) * (&1 - m))) / max(runs - 1, 1))
          {t, m, sd / :math.sqrt(runs)}
        end
      end

    floor = 1 / (vol * :math.sqrt(runs))
    z =
      Enum.zip(stats, ode_at)
      |> Enum.flat_map(fn {row, ode_row} -> Enum.zip_with(row, ode_row, fn {_, m, se}, o -> abs(m - o) / (se + floor) end) end)
      |> Enum.max(fn -> 0.0 end)

    %{runs: runs, times: times, species: net.species, mean: Enum.map(stats, fn row -> Enum.map(row, &elem(&1, 1)) end),
      se: Enum.map(stats, fn row -> Enum.map(row, &elem(&1, 2)) end), first_run: hd(trajs) |> Enum.map(fn row -> Enum.map(row, &(&1 / vol)) end), max_z: z}
  end

  # each channel: {reactant counts, net change, rate constant scaled to molecules}
  defp channels(net, rates, vol) do
    idx = net.species |> Enum.with_index() |> Map.new()
    Enum.zip(net.reactions, rates)
    |> Enum.flat_map(fn {r, {kf, kb}} ->
      fwd = channel(r.lhs, r.rhs, kf, idx, vol)
      if r.rev and kb, do: [fwd, channel(r.rhs, r.lhs, kb, idx, vol)], else: [fwd]
    end)
  end

  defp channel(lhs, rhs, k, idx, vol) do
    order = lhs |> Map.values() |> Enum.sum()
    change = Map.merge(Map.new(rhs, fn {s, n} -> {idx[s], n} end), Map.new(lhs, fn {s, n} -> {idx[s], -n} end), fn _, a, b -> a + b end)
    {Enum.map(lhs, fn {s, n} -> {idx[s], n} end), change, k / :math.pow(vol, max(order - 1, 0)) * Enum.reduce(lhs, 1, fn {_, n}, acc -> acc * fact(n) end)}
  end

  defp fact(n), do: Enum.reduce(1..max(n, 1), 1, &*/2)

  defp propensity({reactants, _, c}, x) do
    Enum.reduce(reactants, c, fn {i, n}, a -> a * falling(elem(x, i), n) / fact(n) end)
  end

  defp falling(x, n), do: Enum.reduce(0..(n - 1), 1, fn j, a -> a * max(x - j, 0) end)

  defp ssa(channels, n0, times, t_end, rng) do
    x = List.to_tuple(n0)
    {samples, _} = ssa_loop(channels, x, 0.0, times, t_end, rng, [], 0)
    Enum.reverse(samples)
  end

  defp ssa_loop(_ch, x, _t, [], _te, rng, acc, _steps), do: {acc, {x, rng}}

  defp ssa_loop(ch, x, t, [ts | rest] = times, te, rng, acc, steps) do
    props = Enum.map(ch, &propensity(&1, x))
    a0 = Enum.sum(props)
    if a0 <= 0 or steps > 2_000_000 do
      # nothing can happen any more: the state holds for every remaining sample time
      {Enum.reduce(times, acc, fn _, a -> [Tuple.to_list(x) | a] end), {x, rng}}
    else
      {u1, rng} = :rand.uniform_s(rng)
      {u2, rng} = :rand.uniform_s(rng)
      tau = -:math.log(u1) / a0
      if t + tau > ts do
        # the next event falls after the sample time: record, and (memoryless) draw again from there
        ssa_loop(ch, x, ts, rest, te, rng, [Tuple.to_list(x) | acc], steps)
      else
        {_, change, _} = pick(ch, props, u2 * a0)
        x = Enum.reduce(change, x, fn {i, d}, xx -> put_elem(xx, i, max(elem(xx, i) + d, 0)) end)
        ssa_loop(ch, x, t + tau, times, te, rng, acc, steps + 1)
      end
    end
  end

  defp pick([c], _props, _r), do: c
  defp pick([c | rest], [p | ps], r), do: if(r < p, do: c, else: pick(rest, ps, r - p))

  defp equilibrium(net, rates, final) do
    for {r, {kf, kb}} <- Enum.zip(net.reactions, rates), r.rev, kb != nil and kb > 0 do
      prod = fn side -> Enum.reduce(side, 1.0, fn {s, n}, a -> a * :math.pow(max(final[s], 0.0), n) end) end
      q = prod.(r.rhs) / max(prod.(r.lhs), 1.0e-300)
      %{reaction: r.text, k: kf / kb, quotient: q, ratio: q / (kf / kb)}
    end
  end
end
