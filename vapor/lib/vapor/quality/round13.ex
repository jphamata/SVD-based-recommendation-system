defmodule Vapor.Quality.Round13 do
  @moduledoc """
  Quality checks for the 0.13 round (finance and trading; the TODO items
  closed), in the suite's discipline: a value, a **control** that a broken
  or naive implementation would produce, and a threshold that separates
  them. Each line answers "is this signal or noise?" with a number that
  could have come out otherwise.

  | check | value | control (must fail, or be caught) |
  |---|---|---|
  | calendar | ANBIMA 2025: 252 business days, the moveable feasts by the computus | the fixed holidays only: 256 |
  | money | R$ 100,00 in three: 33,34 + 33,33 + 33,33 | binary64 floor to cents: a cent lost |
  | curve | every DI1/LTN/NTN-F repriced < 10⁻¹² | the DI1 counted in calendar days: mispriced > 10⁻⁴ |
  | options | parity < 10⁻¹², Greeks = finite differences < 10⁻⁶ | a price under the intrinsic: refused before solving |
  | American | Leisen–Reimer = 4.486 (S 36, K 40) | the European: lower by the early-exercise premium |
  | SVI | a calm smile: g ≥ 0 | Vogt's slice: g < 0 found |
  | Monte Carlo | the oracle's bits, the 2-thread bits, BSM inside the 99 % interval | Itô forgotten: BSM outside (z > 5) |
  | VaR | historical on t(3): Kupiec not rejected; size ≈ 5 % | normal on t(3): rejected |
  | backtest | a planted AR(1): every gate | 30 crossovers on noise: DSR < 0.95; a peek at tomorrow: caught |
  | LP | an optimum with its dual, an infeasible problem with its Farkas vector | a wrong proposal: rejected |
  | arbitrage | quotes with state prices ψ > 0 | a butterfly that pays: found, checked exactly |
  | order book | 6 000 random events: the naive engine agrees | one forged fill: caught |
  | ITCH | the feed rebuilds the engine's book | — |
  | pre-trade | four refusals, each with its cause | a refusal claimed without cause: caught |
  | Hawkes | branching ratio recovered; time-rescaling passes | Poisson on the same arrivals: rejected |
  | market making | inventory strategy: σ(P&L) ≈ half the symmetric | — |
  | execution | Almgren–Chriss closed form = numeric optimum | — |
  | exchange | replay to the same head; every certificate | another seed: another head |
  | archives | a signature by the trusted key | the manifest rewritten: refused |
  | units | 98,6 °F = 37 °C | J/(kg·°C): refused (degC suggested) |
  | transistors | the square law by hand, KCL < 10⁻⁹ | — |
  """
  alias Vapor.Finance.{Arbitrage, Backtest, Book, Calendar, Curve, Exchange, Itch, Micro, Money, MonteCarlo, Num, Options, PreTrade, Risk}
  alias Vapor.Logic.LP

  def run(opts \\ []) do
    %{checks: List.flatten([money(), curve(), options(), mc(opts), risk(), backtest(), lp(), book(), micro(), todo()])}
  end

  defp check(name, value, control, threshold, pass), do: %{name: name, value: value, control: control, threshold: threshold, pass: pass}
  defp f(x) when is_float(x) and x != 0.0 and abs(x) < 1.0e-3, do: String.to_float(:erlang.float_to_binary(x, [{:scientific, 3}]))
  defp f(x), do: if(is_float(x), do: Float.round(x, 6), else: x)

  # ============================================================ money

  defp money do
    du = Calendar.business_days(:anbima, ~D[2025-01-02], ~D[2026-01-02])
    fixed_only = Enum.count(0..(Date.diff(~D[2026-01-02], ~D[2025-01-02]) - 1), fn k ->
      d = Date.add(~D[2025-01-02], k)
      not Calendar.weekend?(d) and {d.month, d.day} not in [{1, 1}, {4, 21}, {5, 1}, {9, 7}, {10, 12}, {11, 2}, {11, 15}, {11, 20}, {12, 25}]
    end)
    {:ok, parts, cert} = Money.allocate(Money.parse!("100.00"), [1, 1, 1], 2)
    float_sum = 3 * (Float.floor(100.0 / 3 * 100) / 100)
    [check("calendar: ANBIMA business days 2025-01-02 → 2026-01-02 (Carnival, Good Friday, Corpus Christi by the computus)", du, fixed_only, "252; fixed holidays only ≠ 252", du == 252 and fixed_only != 252),
     check("money: R$ 100,00 allocated in three by largest remainder", Enum.map_join(parts, " + ", &Money.to_string/1), float_sum, "parts sum to 100.00 exactly; binary64 floor loses a cent", cert.sum_equals_total and float_sum < 100.0)]
  end

  # ============================================================ curve

  defp curve do
    {:ok, r} = Curve.run("date = 2025-01-02\ncalendar = anbima\nbasis = du252\ndi1 G25 = 12.33%\ndi1 J25 = 13.05%\ndi1 F26 = 15.02%\ndi1 F27 = 15.35%\nltn 2028-01-01 price = 652.30\nntnf 2031-01-01 price = 800.00")
    # control: the same F27 node priced with calendar days (act/365) instead of business days
    f27 = Enum.find(r.nodes, &(&1.instrument =~ "F27"))
    act = Date.diff(~D[2027-01-04], ~D[2025-01-02]) / 365
    pu_true = 100_000 * :math.pow(1.1535, -f27.t)
    pu_act = 100_000 * :math.pow(1.1535, -act)
    rel = abs(pu_act - pu_true) / pu_true
    [check("curve: DI1 + LTN + NTN-F bootstrapped (flat-forward, DU/252), every instrument repriced", f(r.certificate.max_rel_error), f(rel), "< 10⁻¹²; DI1 in calendar days off by > 10⁻⁴",
           r.certificate.max_rel_error < 1.0e-12 and rel > 1.0e-4)]
  end

  # ========================================================== options

  defp options do
    {:ok, p} = Options.run("price call S=100 K=105 T=0.5 r=5% q=2% vol=25%")
    refused = Options.implied_vol(:call, 0.5, 100.0, 50.0, 1.0, 0.05, 0.0)
    am = Options.binomial(:put, :american, 36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 801, :lr)
    eu = Options.bsm(:put, 36.0, 40.0, 1.0, 0.06, 0.0, 0.2)
    {:ok, smile} = Options.run("smile T=0.5 F=100\n70 32%\n80 28%\n90 25%\n100 23%\n110 22%\n120 22.5%\n130 23.5%")
    vogt = Options.svi_arbitrage(%{a: -0.0410, b: 0.1331, rho: 0.3060, m: 0.3586, sigma: 0.4153})
    [check("options: BSM parity and the five Greeks against central differences", f(abs(p.certificate.parity_residual)), elem(refused, 0), "parity < 10⁻¹², Greeks < 10⁻⁶; a price under intrinsic refused",
           abs(p.certificate.parity_residual) < 1.0e-12 and p.certificate.greeks_max_rel_diff < 1.0e-6 and match?({:error, _}, refused)),
     check("options: American put (Longstaff & Schwartz 2001, table 1) by Leisen–Reimer, 801 steps", f(am), f(eu), "4.486 ± 0.001; European 3.844 ± 0.001 below it", abs(am - 4.486) < 0.001 and abs(eu - 3.844) < 0.001),
     check("options: SVI fit of a calm smile is butterfly-free; Vogt's slice is not", f(smile.certificate.g_min), f(vogt.g_min), "fit: g ≥ 0 (RMSE of w < 2·10⁻⁴); Vogt: g < 0",
           smile.certificate.free and smile.rmse_total_variance < 2.0e-4 and not vogt.free)]
  end

  # ====================================================== Monte Carlo

  defp mc(opts) do
    if Keyword.get(opts, :worker) == nil and Vapor.Runtime.Substrates.binary("vapor-worker", "native") == nil do
      []
    else
      {:ok, r} = MonteCarlo.price(s0: 100, k: 100, t: 1.0, r: 0.05, sigma: 0.2, paths: 8192, steps: 64, barrier: 85, threads: 2)
      {:ok, c} = MonteCarlo.price(s0: 100, k: 100, t: 1.0, r: 0.05, sigma: 0.2, paths: 8192, steps: 64, drift: :no_ito)
      [check("Monte Carlo on the native worker: oracle bits, 2-thread bits, BSM in the 99 % interval", "#{r.certificate.oracle_parity} · #{r.certificate.threads.identical_bits} · z #{f(r.european.z)}", "z #{f(c.european.z)}",
             "parity and thread bits identical, |z| < 2.58; Itô forgotten: |z| > 5",
             r.certificate.oracle_parity and r.certificate.threads.identical_bits and r.european.covers and r.geometric_asian.covers and abs(c.european.z) > 5),
       check("Monte Carlo: arithmetic Asian with the geometric (Kemna–Vorst) as control variate", f(r.arithmetic_asian_cv.variance_reduction), f(r.speedup), "variance ÷ > 100", r.arithmetic_asian_cv.variance_reduction > 100)]
    end
  end

  # ============================================================== risk

  defp risk do
    {:ok, d} = Backtest.data(%{data: {"t", %{"n" => 3000.0, "sigma" => 0.01, "seed" => 4.0, "nu" => 3.0}}, csv: []})
    rets = Enum.zip(d.close, tl(d.close)) |> Enum.map(fn {a, b} -> b / a - 1 end)
    bad = Risk.backtest(Risk.rolling_var(rets, 500, 0.99, :normal), 0.99)
    good = Risk.backtest(Risk.rolling_var(rets, 500, 0.99, :historical), 0.99)
    size = Enum.count(1..400, fn s -> Risk.backtest(for(k <- 0..499, do: if(Num.u01(s, k) < 0.01, do: {0.0, -1.0}, else: {0.0, 1.0})), 0.99).kupiec.p_value < 0.05 end) / 400
    [check("VaR backtest on Student-t(3) returns: Kupiec p of the historical VaR (and the test's size)", f(good.kupiec.p_value), f(bad.kupiec.p_value), "historical p > 0.05; normal p < 0.05; size 1.5–10 %",
           good.kupiec.p_value > 0.05 and bad.kupiec.p_value < 0.05 and size > 0.015 and size < 0.10)]
  end

  # ========================================================= backtest

  defp backtest do
    {:ok, planted} = Backtest.run("data = ar1 n=5040 phi=0.15 sigma=0.01 seed=3\nsweep k = 1..3\nsignal = sign(sma(ret(close), k))\ncost = 1bp")
    {:ok, noise} = Backtest.run("data = gbm n=2520 sigma=0.01 seed=5\nsweep fast = 5..30 step 5\nsweep slow = 40..120 step 20\nsignal = sign(ema(close, fast) - ema(close, slow))\ncost = 2bp")
    {:ok, peek} = Backtest.run("data = gbm n=800 sigma=0.01 seed=2\nsignal = sign(lead(close) - close)")
    [check("backtest: planted AR(1) momentum passes every gate; the best of 30 crossovers on noise does not", f(planted.deflated_sharpe.dsr), f(noise.deflated_sharpe.dsr), "planted DSR > 0.95, PBO < 0.5, RC p < 0.05; noise DSR < 0.95",
           Enum.all?(planted.gates, & &1.pass) and noise.deflated_sharpe.dsr < 0.95),
     check("backtest: a peek at tomorrow is caught by prefix invariance (Sharpe of the cheat)", f(peek.best.stats.sharpe_annual), "day #{peek.lookahead[:day]}", "look-ahead flagged with its day", not peek.lookahead.clean)]
  end

  # ======================================================= LP, arbitrage

  defp lp do
    {:ok, o} = LP.solve("maximize 3x + 2y\nx + y <= 4\nx + 3y <= 6\nx <= 3")
    {:ok, i} = LP.solve("maximize x + y\nx + y <= 1\nx + y >= 2")
    {:ok, p} = LP.parse("maximize 3x + 2y\nx + y <= 4\nx + 3y <= 6\nx <= 3")
    wrong = LP.check(p, %{status: :optimal, x: %{"x" => {2, 1}, "y" => {2, 1}}, y: [{2, 1}, {0, 1}, {1, 1}]})
    {:ok, bf} = Arbitrage.run("calls T=1 r=5%\n90 bid=14.1 ask=14.4\n100 bid=9.4 ask=9.6\n110 bid=3.6 ask=3.9")
    {:ok, ok} = Arbitrage.run("calls T=1 r=5% S=100\n90 bid=14.1 ask=14.4\n100 bid=7.9 ask=8.2\n110 bid=3.6 ask=3.9")
    [check("LP in rationals: optimum with a dual (zero gap); infeasibility with a Farkas vector", LP.show(o.objective), wrong.reason |> String.slice(0, 40), "both certificates accepted; a wrong proposal rejected",
           o.check.accepted and i.check.accepted and not wrong.accepted),
     check("arbitrage: a butterfly found in call quotes (exact cost); consistent quotes get state prices", bf.cost.exact, length(ok.state_prices), "arbitrage with cost < 0, checked; ψ > 0 for the others",
           bf.arbitrage and bf.certificate.checked_exactly and bf.cost.value < 0 and ok.arbitrage == false and ok.certificate.checked_exactly)]
  end

  # ======================================================== order book

  defp events(seed, n) do
    {evs, _} =
      Enum.map_reduce(0..(n - 1), [], fn i, live ->
        u = fn k -> Num.u01(seed, i * 6 + k) end
        cond do
          u.(0) < 0.15 and live != [] -> id = Enum.at(live, trunc(u.(1) * length(live))); {%{type: :cancel, id: id}, List.delete(live, id)}
          u.(0) < 0.22 and live != [] -> id = Enum.at(live, trunc(u.(1) * length(live))); {%{type: :modify, id: id, price: (if u.(2) < 0.5, do: nil, else: 95 + trunc(u.(3) * 10)), qty: 1 + trunc(u.(4) * 12)}, live}
          true ->
            side = if u.(1) < 0.5, do: :buy, else: :sell
            tif = cond do u.(5) < 0.1 -> :ioc; u.(5) < 0.17 -> :fok; true -> :gtc end
            price = if u.(2) < 0.05, do: nil, else: (if side == :buy, do: 96 + trunc(u.(3) * 8), else: 97 + trunc(u.(3) * 8))
            {%{type: :new, id: i + 1, owner: trunc(u.(4) * 5), side: side, price: price, qty: 2 + trunc(u.(4) * 37 * u.(3)), tif: tif, post_only: u.(5) > 0.93}, [i + 1 | live]}
        end
      end)
    evs
  end

  defp book do
    s = Book.session(events(21, 6000))
    c = Book.Check.check(s.journal)
    i = Enum.find_index(s.journal, fn e -> Enum.any?(e.reports, &match?({:fill, _}, &1)) end)
    e = Enum.at(s.journal, i)
    forged = %{e | reports: Enum.map(e.reports, fn {:fill, fl} -> {:fill, %{fl | qty: fl.qty + 1}}; r -> r end)}
    caught = not Book.Check.check(List.replace_at(s.journal, i, forged)).ok
    {:ok, back} = Itch.unframe(Itch.frame(Itch.from_session(s)))
    itch = Itch.consistent?(s, back)
    pt_evs = [%{type: :new, id: 1, owner: "mm", side: :sell, price: 10010, qty: 100, ts: 1}, %{type: :new, id: 2, owner: "mm", side: :buy, price: 9990, qty: 100, ts: 2},
              %{type: :new, id: 3, owner: "algo", side: :buy, price: 10010, qty: 50_000, ts: 3}, %{type: :new, id: 4, owner: "algo", side: :buy, price: 13000, qty: 10, ts: 4},
              %{type: :kill, owner: "mm", ts: 5}, %{type: :new, id: 5, owner: "mm", side: :sell, price: 10010, qty: 10, ts: 6}]
    limits = %{default: %{max_qty: 1000, collar: 0.05, max_position: 120}}
    pts = PreTrade.session(pt_evs, limits)
    lie = Enum.map(pts.journal, fn x -> if x.event.type == :risk_reject and x.event.reason == :max_qty, do: put_in(x, [:event, :reason], :price_collar), else: x end)
    [check("order book: 6 000 random events — the naive engine reproduces every report; invariants hold", c.checked, if(caught, do: "forged fill caught", else: "missed"), "all reports equal; a forged fill caught", c.ok and caught),
     check("order book: the ITCH 5.0 feed rebuilds the engine's book and volume", "#{itch.levels} levels · #{itch.executed} executed", "—", "book and volume equal", itch.book_equal and itch.volume_equal),
     check("pre-trade gate: fat finger, collar, kill switch — each refusal with its cause, re-derived from the journal", pts.rejected, PreTrade.check(lie, limits).ok, "3 refusals verified; a misattributed refusal caught",
           pts.rejected == 3 and PreTrade.check(pts.journal, limits).ok and not PreTrade.check(lie, limits).ok)]
  end

  # ===================================================== microstructure

  defp micro do
    ts = Micro.hawkes_simulate(1.0, 0.6, 1.5, 2000.0, 3)
    h = Micro.hawkes_fit(ts, 2000.0)
    as = Micro.avellaneda_stoikov(runs: 500)
    ac = Micro.almgren_chriss(%{x: 1.0e6, n: 5, t: 5.0, sigma: 0.95, eta: 2.5e-6, gamma: 2.5e-7, epsilon: 0.0625, lambda: 1.0e-6})
    ex = Exchange.simulate(steps: 900, seed: 3)
    ex2 = Exchange.simulate(steps: 900, seed: 3)
    ex3 = Exchange.simulate(steps: 900, seed: 4)
    [check("Hawkes: branching ratio of a planted process (0.4) recovered; time-rescaling KS", f(h.branching_ratio), f(h.poisson_time_rescaling.p_value), "|n − 0.4| < 0.08, KS p > 0.05; Poisson p < 10⁻⁶",
           abs(h.branching_ratio - 0.4) < 0.08 and h.time_rescaling.p_value > 0.05 and h.poisson_time_rescaling.p_value < 1.0e-6),
     check("Avellaneda–Stoikov (γ = 0.1): σ(P&L) of the inventory strategy over the symmetric", f(as.certificate.pnl_dispersion_ratio), f(as.symmetric.std_pnl), "< 0.6 (paper 5.89/13.43 = 0.44); symmetric σ 13.43 ± 2.5",
           as.certificate.pnl_dispersion_ratio < 0.6 and abs(as.symmetric.std_pnl - 13.43) < 2.5),
     check("Almgren–Chriss: closed-form trajectory against the numeric optimum (tridiagonal)", f(ac.certificate.relative), f(ac.half_life), "relative gap < 10⁻¹²", ac.certificate.relative < 1.0e-12),
     check("exchange session: naive replay, pre-trade and ITCH certificates; replay to the same head", String.slice(ex.head, 0, 12), String.slice(ex3.head, 0, 12), "all certificates; same seed → same head; another seed → another",
           ex.certificate.book_check.ok and ex.certificate.pre_trade.ok and ex.certificate.itch.book_equal and ex.head == ex2.head and ex.head != ex3.head)]
  end

  # ======================================================= TODO closed

  defp todo do
    k = Vapor.Certificate.keygen()
    {:ok, r} = Vapor.Finance.replay("finance.calendar", %{"text" => "du 2025-01-02 2026-01-02"})
    a = Vapor.Archive.pack("finance.calendar", %{"text" => "du 2025-01-02 2026-01-02"}, r)
    {:ok, z} = Vapor.Archive.sign(a.zip, k)
    signed = match?({:ok, %{signature: %{trusted: true}}}, Vapor.Archive.verify(z, trusted: [k.public]))
    {:ok, ent} = :zip.unzip(z, [:memory])
    m = Map.new(ent, fn {n, b} -> {to_string(n), b} end)
    mj = String.replace(m["manifest.json"], "finance.calendar", "finance.calendaR")
    {:ok, {_, forged}} = :zip.create(~c"x.zip", [{~c"manifest.json", mj} | for({n, b} <- m, n != "manifest.json", do: {String.to_charlist(n), b})], [:memory])
    forged_r = Vapor.Archive.verify(forged, trusted: [k.public])
    {:ok, u} = Vapor.Solve.run("T = 98.6[°F]\nx = T in [°C]")
    {:ok, bad} = Vapor.Solve.run("c = 4186[J/(kg*°C)]")
    x = Enum.find(u.lines, &(&1.name == "x")).shown
    {:ok, cs} = Vapor.Engineering.Circuit.run("VDD vdd 0 DC 5\nVG g 0 DC 1.5\nRD vdd d 10k\nM1 d g 0 0 NMOS KP=50u VTO=0.7 LAMBDA=0.02 W=10u L=1u\n.op")
    vd = cs.op.nodes["d"]
    hand = abs((5 - vd) / 10_000 - 250.0e-6 * 0.64 * (1 + 0.02 * vd))
    [check("archives: signed by the operator's key, the identity unchanged", signed, inspect(forged_r), "trusted signature verified; a rewritten manifest refused", signed and forged_r == {:error, :bad_signature}),
     check("units: an affine reading converted (98.6 °F in °C)", f(x), hd(bad.lines).error |> to_string() |> String.slice(0, 40), "37 ± 10⁻⁹; °C inside a compound refused", abs(x - 37.0) < 1.0e-9 and hd(bad.lines).error =~ "degC"),
     check("circuits: a MOSFET common-source stage against the square law by hand (KCL from the device's own equation)", f(hand), "—", "< 10⁻¹¹ (the GMIN current); KCL < 10⁻⁹", hand < 1.0e-11 and cs.op.certificate.kcl_max < 1.0e-9)]
  end
end
