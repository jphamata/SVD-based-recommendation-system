defmodule Vapor.Finance.Arbitrage do
  @moduledoc """
  Arbitrage decided exactly (docs/FINANCE.md §8).

  The fundamental theorem of asset pricing is a theorem of the
  alternative — Farkas' lemma: **either** a portfolio costs nothing (or
  less) and pays something (and never loses), **or** there are strictly
  positive state prices that reproduce every quote within its bid–ask.
  Never both, never neither. Here both sides are found by the exact
  rational simplex of `Vapor.Logic.LP` and each answer comes with the
  object that proves it:

  * an **arbitrage**: the portfolio (buy at the ask, sell at the bid), its
    cost and its payoff in every state — checked by exact arithmetic;
  * **no arbitrage**: the state-price vector ψ > 0 with
    bid ≤ Σₛ Xᵢₛψₛ ≤ ask for every asset — checked the same way.

  Three inputs:

      states = up, down
      bond  bid=0.95 ask=0.96 payoff = 1, 1
      stock bid=100 ask=100.5 payoff = 120, 90
      call  bid=10 ask=10.5 payoff = 20, 0

      calls T=1 r=5% S=100            # call quotes across strikes; states at 0, every strike, 2·K_max
      90  bid=14.1 ask=14.4           # and the slope beyond it (piecewise-linear payoffs: checking
      100 bid=7.9 ask=8.2             # the kinks and the slope checks every terminal price)
      110 bid=3.6 ask=3.9

      fx                              # a cycle whose product of rates exceeds 1
      USD/BRL bid=5.40 ask=5.41
      EUR/USD bid=1.08 ask=1.081
      EUR/BRL bid=5.90 ask=5.91
  """
  alias Vapor.Logic.LP
  import Vapor.Logic.LP, only: [qadd: 2, qsub: 2, qmul: 2, qsign: 1, rat: 1, show: 1, to_float: 1]

  def run(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#") |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    case lines do
      [] -> {:error, "empty"}
      [first | rest] ->
        case model(first, rest) do
          {:model, assets, names} -> decide(assets, names)
          {:model, assets, names, extra} -> with({:ok, res} <- decide(assets, names), do: {:ok, Map.merge(res, extra)})
          other -> other
        end
    end
  end

  defp model(first, rest) do
    cond do
      first =~ ~r/^states?\s*=/i -> states(first, rest)
      first =~ ~r/^(calls|opções|opcoes)\b/i -> calls(first, rest)
      first =~ ~r/^(fx|câmbio|cambio)\b/i -> fx(rest)
      true -> {:error, "first line: states = …, calls T=… r=… [S=…], or fx"}
    end
  end

  @doc """
  Check an outside **proposal** against quotes (states or calls format) —
  the finance twin of `Vapor.Logic.check/2`: the proposer may be a person,
  a search or a language model (MCP `arbitrage_check`); only exact
  arithmetic decides.

  * `%{"portfolio" => %{asset => quantity}}` (positive: bought at the ask;
    negative: sold at the bid) is accepted as an **arbitrage** when its
    payoff is ≥ 0 in every state and it costs < 0 (or ≤ 0 with a positive
    payoff somewhere);
  * `%{"state_prices" => [ψ…]}` is accepted as a **no-arbitrage
    certificate** when every ψ > 0 and bid ≤ Σ Xψ ≤ ask for every asset.
  """
  def check(text, proposal) when is_binary(text) and is_map(proposal) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#") |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    with [first | rest] <- lines, m when elem(m, 0) == :model <- model(first, rest) do
      {assets, names} = {elem(m, 1), elem(m, 2)}
      ratq = fn v -> try do {:ok, rat(if is_number(v), do: v, else: to_string(v))} rescue _ -> :error end end
      cond do
        is_map(proposal["portfolio"]) ->
          known = MapSet.new(assets, & &1.name)
          bad = Enum.find(Map.keys(proposal["portfolio"]), &(not MapSet.member?(known, &1)))
          qs = Map.new(proposal["portfolio"], fn {k, v} -> {k, ratq.(v)} end)
          cond do
            bad -> {:error, "unknown asset #{inspect(bad)} (known: #{Enum.join(Enum.map(assets, & &1.name), ", ")})"}
            Enum.any?(qs, fn {_, v} -> v == :error end) -> {:error, "quantities are numbers or \"p/q\""}
            true ->
              qs = Map.new(qs, fn {k, {:ok, v}} -> {k, v} end)
              legs = for a <- assets, n = Map.get(qs, a.name, {0, 1}), qsign(n) != 0, do: {a, n}
              priced = Enum.all?(legs, fn {a, n} -> if qsign(n) > 0, do: a.ask != nil, else: a.bid != nil end)
              cost = Enum.reduce(legs, {0, 1}, fn {a, n}, acc -> qadd(acc, if(qsign(n) > 0, do: qmul(n, a.ask), else: qmul(n, a.bid))) end)
              pays = for st <- 0..(length(names) - 1), do: Enum.reduce(legs, {0, 1}, fn {a, n}, acc -> qadd(acc, qmul(n, Enum.at(a.payoff, st))) end)
              ok = priced and Enum.all?(pays, &(qsign(&1) >= 0)) and (qsign(cost) < 0 or (qsign(cost) <= 0 and Enum.any?(pays, &(qsign(&1) > 0))))
              worst = pays |> Enum.zip(names) |> Enum.min_by(fn {p, _} -> to_float(p) end)
              {:ok, %{accepted: ok, claim: "arbitrage", cost: show(cost), payoffs: Enum.zip(names, Enum.map(pays, &show/1)) |> Map.new(),
                      reason: cond do
                        not priced -> "a leg has no quote on the side it trades (buying needs an ask, selling a bid)"
                        ok -> "payoff ≥ 0 in every state and cost #{show(cost)}: an arbitrage, checked exactly"
                        qsign(elem(worst, 0)) < 0 -> "loses #{show(LP.qneg(elem(worst, 0)))} in state #{elem(worst, 1)}"
                        qsign(cost) > 0 -> "costs #{show(cost)} > 0: it pays, but it is paid for — not an arbitrage"
                        true -> "costs nothing and pays nothing in any state: not an arbitrage"
                      end}}
          end
        is_list(proposal["state_prices"]) and length(proposal["state_prices"]) == length(names) ->
          ps = Enum.map(proposal["state_prices"], ratq)
          if Enum.any?(ps, &(&1 == :error)) do
            {:error, "state prices are numbers or \"p/q\""}
          else
            ps = Enum.map(ps, fn {:ok, v} -> v end)
            vals = for a <- assets, do: {a, Enum.zip(a.payoff, ps) |> Enum.reduce({0, 1}, fn {x, p}, acc -> qadd(acc, qmul(x, p)) end)}
            out = Enum.find(vals, fn {a, v} -> (a.bid != nil and qsign(qsub(v, a.bid)) < 0) or (a.ask != nil and qsign(qsub(a.ask, v)) < 0) end)
            pos = Enum.all?(ps, &(qsign(&1) > 0))
            {:ok, %{accepted: pos and out == nil, claim: "no arbitrage",
                    reason: cond do
                      not pos -> "a state price is not strictly positive"
                      out -> "#{elem(out, 0).name} is worth #{show(elem(out, 1))} under these prices, outside its bid–ask"
                      true -> "every ψ > 0 and every asset inside its bid–ask: no arbitrage (fundamental theorem), checked exactly"
                    end}}
          end
        true -> {:error, "a proposal is {portfolio: {asset: quantity}} or {state_prices: [one per state: #{Enum.join(names, ", ")}]}"}
      end
    else
      {:error, _} = e -> e
      _ -> {:error, "proposals are checked against states = … or calls … quotes"}
    end
  end

  defp kv(l), do: Regex.scan(~r/(\w+)\s*=\s*([-\d.\/]+%?)/u, l) |> Map.new(fn [_, k, v] -> {String.downcase(k), v} end)

  defp prat(v) do
    if String.ends_with?(v, "%"), do: LP.qdiv(rat(String.trim_trailing(v, "%")), {100, 1}), else: rat(v)
  end

  # ---------------------------------------------------------- finite states

  defp states(first, rest) do
    names = first |> String.split("=", parts: 2) |> List.last() |> String.split(~r/[\s,]+/, trim: true)
    assets =
      Enum.map(rest, fn l ->
        [name | _] = String.split(l)
        m = kv(l)
        payoff = case Regex.run(~r/payoff\s*=\s*(.+)$/i, l) do [_, p] -> p |> String.split(~r/[\s,;]+/, trim: true) |> Enum.map(&rat/1); _ -> nil end
        %{name: name, bid: m["bid"] && rat(m["bid"]), ask: m["ask"] && rat(m["ask"]), payoff: payoff}
      end)
    cond do
      Enum.any?(assets, &(&1.payoff == nil or length(&1.payoff) != length(names))) -> {:error, "every asset needs payoff = one number per state (#{length(names)} states)"}
      Enum.any?(assets, &(&1.bid == nil and &1.ask == nil)) -> {:error, "every asset needs a bid, an ask, or both"}
      true -> {:model, assets, names}
    end
  end

  # -------------------------------------------------- calls across strikes

  defp calls(first, rest) do
    m = kv(first)
    t = rat(m["t"] || "1"); r = prat(m["r"] || "0")
    quotes =
      for l <- rest, [_, k] <- [Regex.run(~r/^\s*([\d.]+)\b/, l)] do
        mm = kv(l); %{k: rat(k), bid: mm["bid"] && rat(mm["bid"]), ask: mm["ask"] && rat(mm["ask"])}
      end
    if length(quotes) < 2 do
      {:error, "at least two strikes: one line per strike — K bid=… ask=…"}
    else
      ks = quotes |> Enum.map(& &1.k) |> Enum.sort_by(&to_float/1)
      top = qmul({2, 1}, List.last(ks))
      grid = [{0, 1}] ++ ks ++ [top]
      names = Enum.map(grid, &"S=#{show(&1)}") ++ ["slope beyond #{show(top)}"]
      # the discount factor e^(−rT) is irrational: a rational bracket is used, exact to 10⁻¹⁵, and said
      df = rat(:erlang.float_to_binary(:math.exp(-to_float(r) * to_float(t)), [{:decimals, 15}]))
      call_payoff = fn k -> Enum.map(grid, fn s -> if qsign(qsub(s, k)) > 0, do: qsub(s, k), else: {0, 1} end) ++ [{1, 1}] end
      assets =
        Enum.map(quotes, fn qq -> %{name: "C(#{show(qq.k)})", bid: qq.bid, ask: qq.ask, payoff: call_payoff.(qq.k)} end) ++
          [%{name: "bond (pays 1)", bid: df, ask: df, payoff: List.duplicate({1, 1}, length(grid)) ++ [{0, 1}]}] ++
          if(m["s"], do: [%{name: "underlying", bid: rat(m["s"]), ask: rat(m["s"]), payoff: grid ++ [{1, 1}]}], else: [])
      {:model, assets, names, %{discount_factor: %{exact: show(df), note: "e^(−rT) rounded to 15 decimals"}, strikes: Enum.map(ks, &to_float/1)}}
    end
  end

  # -------------------------------------------------------------- the LPs

  @doc """
  Decide arbitrage among `assets` (`%{name, bid, ask, payoff}` with
  rationals; a missing bid means it cannot be sold, a missing ask that it
  cannot be bought) over `states`.
  """
  def decide(assets, states) do
    ns = length(states)
    # variables: b_i (bought at ask), s_i (sold at bid), each in [0, 1]
    vars = Enum.flat_map(Enum.with_index(assets), fn {a, i} -> (if a.ask, do: ["b#{i}"], else: []) ++ (if a.bid, do: ["s#{i}"], else: []) end)
    pay_row = fn st -> Enum.reduce(Enum.with_index(assets), %{}, fn {a, i}, m ->
      x = Enum.at(a.payoff, st)
      m = if a.ask, do: Map.put(m, "b#{i}", x), else: m
      if a.bid, do: Map.put(m, "s#{i}", LP.qneg(x)), else: m
    end) end
    cost = Enum.reduce(Enum.with_index(assets), %{}, fn {a, i}, m ->
      m = if a.ask, do: Map.put(m, "b#{i}", a.ask), else: m
      if a.bid, do: Map.put(m, "s#{i}", LP.qneg(a.bid)), else: m
    end)
    bounds = for v <- vars, do: {%{v => {1, 1}}, :le, {1, 1}}
    payoffs = for st <- 0..(ns - 1), do: {pay_row.(st), :ge, {0, 1}}
    lp1 = %{sense: :min, vars: vars, c: cost, c0: {0, 1}, rows: payoffs ++ bounds, free: []}
    {:ok, r1} = LP.solve(lp1)
    cond do
      r1.status == :optimal and qsign(r1.objective) < 0 ->
        {:ok, arbitrage(assets, states, r1.x, "a portfolio that pays at least zero in every state and is paid #{show(LP.qneg(r1.objective))} to enter")}
      true ->
        total = Enum.reduce(0..(ns - 1), %{}, fn st, m -> Map.merge(m, pay_row.(st), fn _, a, b -> qadd(a, b) end) end)
        lp2 = %{sense: :max, vars: vars, c: total, c0: {0, 1}, rows: payoffs ++ bounds ++ [{cost, :le, {0, 1}}], free: []}
        {:ok, r2} = LP.solve(lp2)
        if r2.status == :optimal and qsign(r2.objective) > 0 do
          {:ok, arbitrage(assets, states, r2.x, "a portfolio that costs nothing, never loses, and pays in some state")}
        else
          state_prices(assets, states)
        end
    end
  end

  defp arbitrage(assets, states, x, why) do
    pos = for {a, i} <- Enum.with_index(assets), do: {a, qsub(Map.get(x, "b#{i}", {0, 1}), Map.get(x, "s#{i}", {0, 1})), Map.get(x, "b#{i}", {0, 1}), Map.get(x, "s#{i}", {0, 1})}
    cost = Enum.reduce(pos, {0, 1}, fn {a, _, b, s}, acc -> acc |> qadd(if(a.ask, do: qmul(b, a.ask), else: {0, 1})) |> qsub(if(a.bid, do: qmul(s, a.bid), else: {0, 1})) end)
    pays = for st <- 0..(length(states) - 1), do: Enum.reduce(pos, {0, 1}, fn {a, n, _, _}, acc -> qadd(acc, qmul(n, Enum.at(a.payoff, st))) end)
    proven = Enum.all?(pays, &(qsign(&1) >= 0)) and (qsign(cost) < 0 or (qsign(cost) <= 0 and Enum.any?(pays, &(qsign(&1) > 0))))
    %{arbitrage: true, why: why,
      portfolio: for({a, n, _, _} <- pos, qsign(n) != 0, do: %{asset: a.name, quantity: %{exact: show(n), value: to_float(n)}, side: if(qsign(n) > 0, do: "buy at ask", else: "sell at bid")}),
      cost: %{exact: show(cost), value: to_float(cost)},
      payoffs: Enum.zip(states, pays) |> Enum.map(fn {s, p} -> %{state: s, payoff: %{exact: show(p), value: to_float(p)}} end),
      certificate: %{checked_exactly: proven, rule: "payoff ≥ 0 in every state, and cost < 0 (or cost ≤ 0 with a positive payoff somewhere)"}}
  end

  defp state_prices(assets, states) do
    ns = length(states)
    psi = for s <- 0..(ns - 1), do: "p#{s}"
    vars = ["t" | psi]
    rows =
      (for p <- psi, do: {%{p => {1, 1}, "t" => {-1, 1}}, :ge, {0, 1}}) ++ [{%{"t" => {1, 1}}, :le, {1, 1}}] ++
        Enum.flat_map(assets, fn a ->
          val = Map.new(Enum.with_index(a.payoff), fn {x, s} -> {"p#{s}", x} end) |> Map.reject(fn {_, x} -> qsign(x) == 0 end)
          (if a.bid, do: [{val, :ge, a.bid}], else: []) ++ (if a.ask, do: [{val, :le, a.ask}], else: [])
        end)
    {:ok, r} = LP.solve(%{sense: :max, vars: vars, c: %{"t" => {1, 1}}, c0: {0, 1}, rows: rows, free: []})
    cond do
      r.status == :optimal and qsign(r.objective) > 0 ->
        ps = Enum.map(psi, &r.x[&1])
        fair = for a <- assets, do: Enum.zip(a.payoff, ps) |> Enum.reduce({0, 1}, fn {x, p}, acc -> qadd(acc, qmul(x, p)) end)
        ok = Enum.all?(ps, &(qsign(&1) > 0)) and Enum.all?(Enum.zip(assets, fair), fn {a, v} -> (a.bid == nil or qsign(qsub(v, a.bid)) >= 0) and (a.ask == nil or qsign(qsub(a.ask, v)) >= 0) end)
        {:ok, %{arbitrage: false, why: "strictly positive state prices reproduce every quote inside its bid–ask (fundamental theorem)",
                state_prices: Enum.zip(states, ps) |> Enum.map(fn {s, p} -> %{state: s, price: %{exact: show(p), value: to_float(p)}} end),
                model_values: Enum.zip(assets, fair) |> Enum.map(fn {a, v} -> %{asset: a.name, value: to_float(v), bid: a.bid && to_float(a.bid), ask: a.ask && to_float(a.ask)} end),
                certificate: %{checked_exactly: ok, rule: "ψ > 0 and bid ≤ Σ Xψ ≤ ask for every asset"}}}
      true ->
        # unreachable by the theorem of the alternative — reported if it ever happens, never hidden
        {:ok, %{arbitrage: nil, why: "neither an arbitrage nor positive state prices were found (weak arbitrage at the boundary: a quote equal to its bound)", certificate: %{checked_exactly: false}}}
    end
  end

  # ------------------------------------------------------------------- FX

  defp fx(rest) do
    pairs =
      for l <- rest, [_, a, b] <- [Regex.run(~r/^([A-Za-z]{3})\s*\/\s*([A-Za-z]{3})/, l)] do
        m = kv(l); {String.upcase(a), String.upcase(b), m["bid"] && rat(m["bid"]), m["ask"] && rat(m["ask"])}
      end
    # edges: selling 1 A gets bid B (A→B at bid); 1 B buys 1/ask A (B→A at 1/ask)
    edges = Enum.flat_map(pairs, fn {a, b, bid, ask} ->
      (if bid, do: [{a, b, bid, "sell #{a}/#{b} at #{show(bid)}"}], else: []) ++ (if ask, do: [{b, a, LP.qdiv({1, 1}, ask), "buy #{a}/#{b} at #{show(ask)}"}], else: [])
    end)
    nodes = edges |> Enum.flat_map(fn {a, b, _, _} -> [a, b] end) |> Enum.uniq()
    cycles = for start <- nodes, c <- simple_cycles(edges, start, start, [], MapSet.new(), 5), do: c
    best =
      cycles |> Enum.map(fn c -> {c, Enum.reduce(c, {1, 1}, fn {_, _, r, _}, acc -> qmul(acc, r) end)} end)
      |> Enum.max_by(fn {_, p} -> to_float(p) end, fn -> nil end)
    case best do
      nil -> {:error, "no cycle among the quotes (at least three currencies, linked)"}
      {c, p} ->
        if qsign(qsub(p, {1, 1})) > 0 do
          {:ok, %{arbitrage: true, why: "a cycle of conversions returns more than it started with",
                  cycle: Enum.map(c, fn {a, b, r, how} -> %{from: a, to: b, rate: to_float(r), how: how} end),
                  gross: %{exact: show(p), value: to_float(p)}, profit_per_unit: to_float(p) - 1,
                  certificate: %{checked_exactly: true, rule: "the product of the cycle's rates, in rationals, exceeds 1"}}}
        else
          {:ok, %{arbitrage: false, why: "every cycle of up to five conversions returns at most what it started with",
                  best_cycle: Enum.map(c, fn {a, b, r, how} -> %{from: a, to: b, rate: to_float(r), how: how} end), gross: %{exact: show(p), value: to_float(p)},
                  certificate: %{checked_exactly: true, rule: "all simple cycles of length ≤ 5 enumerated; each product ≤ 1 in rationals"}}}
        end
    end
  end

  defp simple_cycles(_edges, _start, _cur, _path, _seen, 0), do: []
  defp simple_cycles(edges, start, cur, path, seen, depth) do
    Enum.flat_map(edges, fn {a, b, _, _} = e ->
      cond do
        a != cur -> []
        b == start and length(path) >= 1 -> [Enum.reverse([e | path])]
        MapSet.member?(seen, b) or b == start -> []
        true -> simple_cycles(edges, start, b, [e | path], MapSet.put(seen, b), depth - 1)
      end
    end)
  end
end
