defmodule Vapor.Finance do
  @moduledoc """
  The finance and trading desk (docs/FINANCE.md): one front door for the
  console, the MCP server and the CLI. Every task takes the text a
  practitioner would write and answers with what lets it be judged.

  | kind | input | module |
  |---|---|---|
  | `calendar` | `du 2025-01-02 2026-01-02`, `holidays anbima 2026`, `adjust …`, `yf act/360 …`, `allocate 100.00 1 1 1`, `factor 13.65% 21` | `Calendar`, `Money` |
  | `curve` | instruments (DI1, LTN, NTN-F, deposits, bonds, swaps) | `Curve` |
  | `options` | `price/iv/american/heston/smile …` | `Options` |
  | `mc` | `S=100 K=100 T=1 r=5% vol=20% paths=8192 steps=64 barrier=85` | `MonteCarlo` (native worker) |
  | `risk` | a P&L series and a VaR method | `Risk` |
  | `portfolio` | asset returns | `Risk` |
  | `backtest` | data, sweep, signal, cost | `Backtest` |
  | `arbitrage` | states / calls / fx | `Arbitrage` |
  | `book` | orders (`buy 100 @ 101.25 gtc owner=a`, `cancel 3`, …) | `Book`, `Book.Check`, `Itch`, `Fix` |
  | `exchange` | `steps=1500 seed=3 makers=2 …` | `Exchange` |
  | `micro` | `hawkes …`, `mm gamma=0.1`, `execution X=1e6 N=5 …` | `Micro` |
  """
  alias Vapor.Finance.{Arbitrage, Backtest, Book, Calendar, Curve, Exchange, Fix, Itch, Micro, Money, MonteCarlo, Num, Options, Risk}

  @kinds ~w(arbitrage backtest book calendar curve exchange mc micro options portfolio risk)
  def kinds, do: @kinds

  def run(kind, text, opts \\ [])
  def run("curve", t, _), do: Curve.run(t)
  def run("options", t, _), do: Options.run(t)
  def run("backtest", t, _), do: Backtest.run(t)
  def run("arbitrage", t, _), do: Arbitrage.run(t)
  def run("calendar", t, _), do: calendar(t)
  def run("mc", t, o), do: mc(t, o)
  def run("risk", t, _), do: risk(t)
  def run("portfolio", t, _), do: portfolio(t)
  def run("book", t, _), do: book(t)
  def run("exchange", t, _), do: exchange(t)
  def run("micro", t, _), do: micro(t)
  def run(k, _, _), do: {:error, "kind: #{Enum.join(@kinds, ", ")} (got #{inspect(k)})"}

  @timing ~w(ms latency_us book_check_ms native_ms lowering_ms beam_f64_ms_estimate speedup)

  @doc false
  # an archive's recipe run again (Vapor.Archive): the same text, the same result — wall-clock fields scrubbed
  def replay("finance." <> kind, %{"text" => t}) when kind in @kinds and kind != "mc" and is_binary(t) do
    case run(kind, t) do
      {:ok, r} -> {:ok, scrub(r)}
      e -> e
    end
  end
  def replay(_, _), do: {:error, :bad_recipe}

  @doc "A result with its wall-clock fields removed (what an archive keeps and a replay compares)."
  def scrub(%{} = m) when not is_struct(m), do: m |> Map.reject(fn {k, _} -> to_string(k) in @timing end) |> Map.new(fn {k, v} -> {k, scrub(v)} end)
  def scrub(l) when is_list(l), do: Enum.map(l, &scrub/1)
  def scrub(x), do: x

  defp kv(t), do: Regex.scan(~r/([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-\w.,\/%]+)/u, t) |> Map.new(fn [_, k, v] -> {String.downcase(k), v} end)

  defp num(nil, d), do: d
  defp num(v, d) do
    v = String.replace(v, ",", ".")
    {pct, v} = if String.ends_with?(v, "%"), do: {true, String.trim_trailing(v, "%")}, else: {false, v}
    case Float.parse(v) do
      {x, _} -> if pct, do: x / 100, else: x
      :error -> d
    end
  end

  defp int(v, d, lo, hi), do: v |> num(d * 1.0) |> trunc() |> max(lo) |> min(hi)

  # ============================================================ calendar

  defp calendar(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#") |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    results = Enum.map(lines, &cal_line/1)
    case Enum.find(results, &match?({:error, _}, &1)) do
      nil -> {:ok, %{lines: Enum.zip(lines, Enum.map(results, &elem(&1, 1))) |> Enum.map(fn {l, r} -> Map.put(r, :input, l) end)}}
      e -> e
    end
  end

  defp date!(s), do: (case Date.from_iso8601(s) do {:ok, d} -> d; _ -> throw({:bad, "not a date (YYYY-MM-DD): #{s}"}) end)
  defp cal!(s), do: (case Calendar.parse_calendar(s || "anbima") do {:ok, c} -> c; {:error, w} -> throw({:bad, w}) end)

  defp cal_line(l) do
    w = String.split(l)
    try do
      case Enum.map(w, &String.downcase/1) do
        ["du", a, b | c] -> (cal = cal!(List.first(c)); {:ok, %{kind: "business days", value: Calendar.business_days(cal, date!(a), date!(b)), calendar: cal}})
        ["holidays", c, y] -> {:ok, %{kind: "holidays", calendar: cal!(c), value: Enum.map(Calendar.holidays(cal!(c), String.to_integer(y)), fn {d, n} -> %{date: Date.to_iso8601(d), name: n} end)}}
        ["feriados", c, y] -> cal_line("holidays #{c} #{y}")
        ["adjust", d, conv | c] ->
          conv = String.to_existing_atom(conv)
          {:ok, %{kind: "adjusted date", value: Date.to_iso8601(Calendar.adjust(cal!(List.first(c)), date!(d), conv))}}
        ["add", d, n | c] -> {:ok, %{kind: "business days added", value: Date.to_iso8601(Calendar.add_business_days(cal!(List.first(c)), date!(d), String.to_integer(n)))}}
        ["yf", basis, a, b | c] ->
          {:ok, bs} = Calendar.parse_basis(basis)
          {:ok, %{kind: "year fraction", basis: bs, value: Calendar.year_fraction(bs, date!(a), date!(b), cal!(List.first(c)))}}
        ["di1", code] -> {:ok, %{kind: "DI1 maturity", value: Date.to_iso8601(elem(Calendar.di1_maturity(code), 1))}}
        ["allocate", total | ws] ->
          {:ok, parts, cert} = Money.allocate(Money.parse!(total), ws, Money.parse!(total).e)
          {:ok, %{kind: "allocation (largest remainder)", value: Enum.map(parts, &Money.to_string/1), certificate: cert}}
        ["round", x, places, mode] -> {:ok, %{kind: "rounding", value: Money.to_string(Money.round(Money.parse!(x), String.to_integer(places), String.to_existing_atom(mode)))}}
        ["factor", rate, du] ->
          {:ok, f} = Money.factor_252(Money.parse!(String.trim_trailing(rate, "%")) |> then(fn m -> if String.ends_with?(rate, "%"), do: Money.divide(m, Money.parse!("100"), m.e + 2, :down), else: m end), String.to_integer(du))
          {:ok, %{kind: "factor (1 + r)^(du/252), truncated at 8 places", value: Money.to_string(f)}}
        ["sum" | xs] ->
          exact = xs |> Enum.map(&Money.parse!/1) |> Money.sum()
          {:ok, %{kind: "exact sum (binary64 for contrast)", value: Money.to_string(exact), binary64: xs |> Enum.map(&num(&1, 0.0)) |> Enum.sum() |> :erlang.float_to_binary([:short])}}
        _ -> {:error, "not understood: #{inspect(l)} — du, holidays, adjust, add, yf, di1, allocate, round, factor, sum"}
      end
    rescue
      e -> {:error, "#{inspect(l)}: #{Exception.message(e)}"}
    catch
      {:bad, w} -> {:error, w}
    end
  end

  # ================================================================== MC

  defp mc(t, o) do
    m = kv(t)
    args = [s0: num(m["s"], 100.0), k: num(m["k"], 100.0), t: num(m["t"], 1.0), r: num(m["r"], 0.0), q: num(m["q"], 0.0), sigma: num(m["vol"] || m["sigma"], 0.2),
            type: if(String.contains?(String.downcase(t), "put"), do: :put, else: :call), paths: int(m["paths"], 8192, 64, 65_536), steps: int(m["steps"], 64, 1, 512),
            barrier: m["barrier"] && num(m["barrier"], nil), seed: int(m["seed"], 1, 0, 1_000_000_000), threads: int(m["threads"], 2, 1, 8),
            rng: if(m["rng"] == "host", do: :host, else: :device), drift: if(m["drift"] == "no_ito", do: :no_ito, else: :ito)] ++ o
    MonteCarlo.price(args)
  end

  # ================================================================ risk

  defp series_from(m, rows) do
    case m["data"] do
      "csv" ->
        xs = for l <- rows, [_, v | _] <- [String.split(l, ~r/\s*[,;\s]\s*/)], {x, _} <- [Float.parse(v)], do: x
        if m["prices"] == "true" or Enum.all?(xs, &(&1 > 0)) and Num.mean(xs) > 1, do: Enum.zip(xs, tl(xs)) |> Enum.map(fn {a, b} -> b / a - 1 end), else: xs
      kind ->
        spec = %{data: {kind || "t", Map.new(Enum.filter(m, fn {k, _} -> k in ~w(n sigma seed nu mu phi) end), fn {k, v} -> {k, num(v, 0.0)} end)}, csv: []}
        case Backtest.data(spec) do
          {:ok, d} -> Enum.zip(d.close, tl(d.close)) |> Enum.map(fn {a, b} -> b / a - 1 end)
          {:error, w} -> throw({:bad, w})
        end
    end
  end

  defp risk(text) do
    lines = String.split(text, "\n")
    m = kv(Enum.join(Enum.reject(lines, &(&1 =~ ~r/^\s*[\d\-]/)), " "))
    rows = Enum.filter(lines, &(&1 =~ ~r/^\s*[\d\-]/))
    try do
      rets = series_from(m, rows)
      alpha = num(m["alpha"], 0.99); w = int(m["window"], 250, 30, 2000)
      method = case m["method"] do "normal" -> :normal; "cornish_fisher" -> :cornish_fisher; "ewma" -> :ewma; _ -> :historical end
      if length(rets) < w + 50, do: throw({:bad, "the series needs at least window + 50 points (#{length(rets)} < #{w + 50})"})
      fc = Risk.rolling_var(rets, w, alpha, method)
      bt = Risk.backtest(fc, alpha)
      last = Enum.take(fc, -250)
      basel = if length(last) == 250 and abs(alpha - 0.99) < 1.0e-9, do: Risk.backtest(last, 0.99), else: nil
      full = for mm <- [:historical, :normal, :cornish_fisher, :ewma], do: Map.put(Risk.var_es(rets, alpha, mm), :method, mm)
      exc = fc |> Enum.with_index() |> Enum.filter(fn {{v, p}, _} -> -p > v end) |> Enum.map(fn {{v, p}, i} -> %{i: i, var: v, pnl: p} end)
      {:ok, %{observations: length(rets), alpha: alpha, window: w, method: method, backtest: bt, basel_last_250: basel, full_sample: full, exceptions: exc,
              forecasts: fc |> Enum.with_index() |> Enum.take_every(max(1, div(length(fc), 500))) |> Enum.map(fn {{v, p}, i} -> %{i: i, var: v, pnl: p, exception: -p > v} end)}}
    catch
      {:bad, w} -> {:error, w}
    end
  end

  defp portfolio(text) do
    lines = String.split(text, "\n")
    m = kv(Enum.join(Enum.reject(lines, &(&1 =~ ~r/^\s*[\d\-]/)), " "))
    rows = lines |> Enum.filter(&(&1 =~ ~r/^\s*[\d\-]/)) |> Enum.map(fn l -> l |> String.split(~r/\s*[,;\s]\s*/, trim: true) |> Enum.map(&num(&1, 0.0)) end)
    rets =
      if rows != [] do
        rows
      else
        # a two-factor model: n assets, t days
        n = int(m["assets"], 8, 2, 40); t = int(m["n"], 750, 60, 5000); seed = int(m["seed"], 2, 0, 1_000_000)
        z = Num.normals(seed, t * (n + 2)) |> Enum.chunk_every(n + 2)
        loads = for i <- 0..(n - 1), do: {0.5 + 0.8 * Num.u01(seed + 9, i), (if rem(i, 2) == 0, do: 0.6, else: -0.3) * Num.u01(seed + 11, i), 0.6 + 1.2 * Num.u01(seed + 13, i)}
        for [f1, f2 | es] <- z, do: Enum.zip_with(loads, es, fn {a, b, s}, e -> 0.006 * (a * f1 + b * f2) + 0.008 * s * e end)
      end
    cond do
      length(rets) < 30 -> {:error, "at least 30 rows of returns"}
      length(hd(rets)) < 2 -> {:error, "at least two assets"}
      true ->
        {sample, _} = Risk.covariance(rets)
        lw = Risk.ledoit_wolf(rets)
        cov = lw.covariance
        {:ok, mv} = Risk.min_variance(cov, cap: num(m["cap"], 1.0))
        rp = Risk.risk_parity(cov)
        hrp = Risk.hrp(cov)
        ev = fn c -> {vals, _} = Vapor.Dense.eigh(c); Enum.max(vals) / max(Enum.min(vals), 1.0e-300) end
        vol = fn w -> :math.sqrt(Vapor.Dense.dot(w, Vapor.Dense.matvec(cov, w)) * 252) end
        n = length(cov)
        {:ok, %{assets: n, observations: length(rets), shrinkage: lw.shrinkage, condition_sample: ev.(sample), condition_shrunk: ev.(cov),
                min_variance: Map.put(mv, :vol_annual, vol.(mv.weights)), risk_parity: Map.put(rp, :vol_annual, vol.(rp.weights)),
                hrp: Map.put(hrp, :vol_annual, vol.(hrp.weights)), equal_weight: %{vol_annual: vol.(List.duplicate(1 / n, n))},
                correlation: for(i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: Enum.at(Enum.at(cov, i), j) / :math.sqrt(Enum.at(Enum.at(cov, i), i) * Enum.at(Enum.at(cov, j), j))))}}
    end
  end

  # ================================================================ book

  @doc false
  def parse_orders(text) do
    lines = text |> String.split("\n") |> Enum.map(&(&1 |> String.split("#") |> hd() |> String.trim())) |> Enum.reject(&(&1 == ""))
    {cfg, rest} = Enum.split_with(lines, &(&1 =~ ~r/^(stp|tick)\s*=/i))
    c = kv(Enum.join(cfg, " "))
    tick = num(c["tick"], 0.01)
    stp = case c["stp"] do "cancel_resting" -> :cancel_resting; "off" -> :off; _ -> :cancel_taker end
    ticks = fn p -> round(num(p, 0.0) / tick) end
    {evs, _} =
      Enum.map_reduce(Enum.with_index(rest, 1), 1, fn {l, ln}, next ->
        low = String.downcase(l)
        o = kv(l)
        owner = o["owner"] || o["dono"] || "anon"
        tif = cond do low =~ ~r/\bioc\b/ -> :ioc; low =~ ~r/\bfok\b/ -> :fok; true -> :gtc end
        ts = ln * 1_000_000
        cond do
          m = Regex.run(~r/^(buy|sell|compra|venda)\s+(\d+)\s*(?:@\s*([\d.,]+)|(mkt|market|mercado))/i, l) ->
            side = if String.downcase(Enum.at(m, 1)) in ["buy", "compra"], do: :buy, else: :sell
            price = if Enum.at(m, 3) not in [nil, ""], do: ticks.(Enum.at(m, 3)), else: nil
            {%{type: :new, id: next, owner: owner, side: side, price: price, qty: String.to_integer(Enum.at(m, 2)), tif: tif, post_only: low =~ ~r/post/, ts: ts}, next + 1}
          m = Regex.run(~r/^(cancel|cancelar)\s+(\d+)/i, l) -> {%{type: :cancel, id: String.to_integer(Enum.at(m, 2)), ts: ts}, next}
          m = Regex.run(~r/^(modify|alterar)\s+(\d+)/i, l) ->
            {%{type: :modify, id: String.to_integer(Enum.at(m, 2)), qty: int(o["qty"], 1, 1, 1_000_000_000), price: o["price"] && ticks.(o["price"]), ts: ts}, next}
          m = Regex.run(~r/^(kill)\s+(\S+)/i, l) -> {%{type: :kill, owner: Enum.at(m, 2), ts: ts}, next}
          true -> {{:error, "line #{ln}: #{inspect(l)} — buy/sell QTY @ PRICE [ioc|fok|post] owner=…, buy QTY mkt, cancel ID, modify ID qty=… [price=…], kill OWNER"}, next}
        end
      end)
    case Enum.find(evs, &match?({:error, _}, &1)) do
      nil -> {:ok, evs, %{tick: tick, stp: stp}}
      e -> e
    end
  end

  defp book(text) do
    with {:ok, evs, cfg} <- parse_orders(text) do
      if length(evs) > 5000 do
        {:error, "at most 5 000 orders here"}
      else
        s = Book.session(evs, stp: cfg.stp)
        chk = Book.Check.check(s.journal, stp: cfg.stp)
        feed = Itch.from_session(s)
        {:ok, back} = Itch.unframe(Itch.frame(feed))
        price = fn p -> p && p * cfg.tick end
        fix = s.journal |> Enum.take(12) |> Enum.flat_map(& &1.reports) |> Fix.exec_reports(%{}, seq: 1, tick: cfg.tick) |> Enum.take(16) |> Enum.map(&Fix.readable/1)
        first_fill = Enum.find(s.journal, fn e -> Enum.any?(e.reports, &match?({:fill, _}, &1)) end)
        proof = if first_fill do
          pr = Book.prove(s.book, first_fill.seq)
          %{seq: first_fill.seq, leaf: Base.encode16(pr.leaf, case: :lower), path: Enum.map(pr.proof, fn {side, h} -> %{side: side, hash: Base.encode16(h, case: :lower)} end), root: Base.encode16(pr.root, case: :lower),
            verified: Vapor.Merkle.verify(pr.leaf, pr.proof, pr.root)}
        end
        {:ok, %{events: length(evs), trades: length(s.trades), volume: s.book.volume, head: s.head, merkle_root: s.merkle_root, stp: cfg.stp, tick: cfg.tick,
                depth: %{bids: Enum.map(s.depth.bids, fn {p, q, n} -> %{price: price.(p), qty: q, orders: n} end), asks: Enum.map(s.depth.asks, fn {p, q, n} -> %{price: price.(p), qty: q, orders: n} end)},
                journal: Enum.map(s.journal, fn e -> %{seq: e.seq, event: show_event(e.event, price), reports: Enum.map(e.reports, &show_report(&1, price)), hash: Base.encode16(e.hash, case: :lower) |> binary_part(0, 16)} end),
                check: Map.take(chk, [:ok, :checked, :verdict, :failures]) |> Map.update!(:failures, fn fs -> Enum.map(fs, &inspect/1) end),
                itch: Map.merge(Itch.consistent?(s, back), %{messages: length(feed), bytes: byte_size(Itch.frame(feed)), sample: feed |> Enum.drop(2) |> Enum.take(6) |> Enum.map(&(&1 |> Itch.encode() |> Base.encode16(case: :lower)))}),
                fix: fix, proof: proof,
                latency_us: s.latency_ns |> Enum.sort() |> then(fn l -> %{p50: Enum.at(l, div(length(l), 2)) / 1000, max: List.last(l) / 1000} end)}}
      end
    end
  end

  defp show_event(%{type: :new} = e, price), do: "#{e.side} #{e.qty} #{if e.price, do: "@ #{fmt(price.(e.price))}", else: "mkt"} #{e.tif}#{if e.post_only, do: " post", else: ""} (#{e.owner}) ##{e.id}"
  defp show_event(%{type: :cancel, id: id}, _), do: "cancel ##{id}"
  defp show_event(%{type: :modify} = e, price), do: "modify ##{e.id} qty=#{e.qty}#{if e.price, do: " price=#{fmt(price.(e.price))}", else: ""}"
  defp show_event(%{type: :kill, owner: o}, _), do: "kill #{o}"
  defp show_event(%{type: :risk_reject} = e, _), do: "risk reject ##{e.id}: #{e.reason}"
  defp show_event(e, _), do: inspect(e)

  defp show_report({:fill, f}, price), do: "fill ##{f.taker} × ##{f.maker} #{f.qty} @ #{fmt(price.(f.price))}"
  defp show_report({:rested, id, p, q}, price), do: "rest ##{id} #{q} @ #{fmt(price.(p))}"
  defp show_report({:cancelled, id, q, why}, _), do: "cancel ##{id} #{q} (#{why})"
  defp show_report({:accepted, id}, _), do: "ack ##{id}"
  defp show_report({:modified, id, q}, _), do: "modified ##{id} → #{q}"
  defp show_report({:rejected, id, why}, _), do: "reject ##{id} (#{why})"
  defp show_report({:risk_rejected, id, why}, _), do: "risk reject ##{id} (#{why})"
  defp show_report(r, _), do: inspect(r)

  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, decimals: 2)
  defp fmt(x), do: to_string(x)

  # ============================================================ exchange

  defp exchange(text) do
    m = kv(text)
    r = Exchange.simulate(steps: int(m["steps"], 1500, 100, 6000), seed: int(m["seed"], 3, 0, 1_000_000_000), makers: int(m["makers"], 2, 1, 6),
                          half_spread: int(m["half_spread"], 3, 1, 50), gamma: num(m["gamma"], 0.02), sigma: num(m["sigma"], 2.0),
                          hawkes: {num(m["mu"], 1.0), num(m["alpha"], 0.8), num(m["beta"], 2.0)}, informed: (if m["informed"] == "off", do: nil, else: int(m["informed"], 4, 1, 1000)))
    {:ok, r |> Map.drop([:final_journal]) |> Map.update!(:depth, fn d -> %{bids: Enum.map(d.bids, &Tuple.to_list/1), asks: Enum.map(d.asks, &Tuple.to_list/1)} end)}
  end

  # =============================================================== micro

  defp micro(text) do
    m = kv(text)
    case text |> String.trim() |> String.split() |> List.first() |> to_string() |> String.downcase() do
      "hawkes" ->
        {mu, al, be, tt} = {num(m["mu"], 1.0), num(m["alpha"], 0.6), num(m["beta"], 1.5), num(m["t"], 2000.0) |> min(20_000.0)}
        if al >= be, do: throw({:bad, "α < β is needed (branching ratio below 1)"})
        ts = Micro.hawkes_simulate(mu, al, be, tt, int(m["seed"], 3, 0, 1_000_000_000))
        fit = Micro.hawkes_fit(ts, tt)
        bins = 120; w = tt / bins
        counts = Enum.reduce(ts, List.duplicate(0, bins), fn t, c -> List.update_at(c, min(trunc(t / w), bins - 1), &(&1 + 1)) end)
        {:ok, %{task: "hawkes", planted: %{mu: mu, alpha: al, beta: be, branching_ratio: al / be}, fit: fit, counts: counts, bin: w}}
      w when w in ["mm", "avellaneda", "stoikov"] ->
        {:ok, Map.put(Micro.avellaneda_stoikov(gamma: num(m["gamma"], 0.1), runs: int(m["runs"], 1000, 50, 5000), seed: int(m["seed"], 5, 0, 1_000_000)), :task, "market making")}
      w when w in ["execution", "execução", "execucao", "ac"] ->
        o = %{x: num(m["x"], 1.0e6), n: int(m["n"], 5, 1, 400), t: num(m["t"], 5.0), sigma: num(m["sigma"], 0.95), eta: num(m["eta"], 2.5e-6), gamma: num(m["gamma"], 2.5e-7),
              epsilon: num(m["epsilon"], 0.0625), lambda: num(m["lambda"], 1.0e-6)}
        r = Micro.almgren_chriss(o)
        fr = Micro.frontier(o, for(e <- -9..-4, x <- [1.0, 3.0], do: x * :math.pow(10, e)))
        {:ok, Map.merge(r, %{task: "execution", frontier: fr, twap: Micro.almgren_chriss(Map.put(o, :lambda, 0.0)) |> Map.take([:expected_cost, :variance])})}
      _ -> {:error, "first word: hawkes, mm or execution"}
    end
  catch
    {:bad, w} -> {:error, w}
  end
end
