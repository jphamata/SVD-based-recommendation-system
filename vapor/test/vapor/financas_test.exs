defmodule Vapor.FinancasTest do
  @moduledoc "Round 0.13: finance and trading (docs/FINANCAS.md). External oracles: QuantLib, SciPy, simplefix."
  use ExUnit.Case, async: true
  alias Vapor.Finance.{Arbitrage, Backtest, Book, Calendar, Curve, Exchange, Fix, Itch, Micro, Money, MonteCarlo, Num, Options, PreTrade, Risk}
  alias Vapor.Logic.LP

  # ------------------------------------------------------------------ money

  describe "exact money" do
    test "0.1 + 0.2 = 0.3 exactly; every rounding mode on the ties and around them" do
      assert Money.sum([Money.parse!("0.1"), Money.parse!("0.2")]) |> Money.to_string() == "0.3"
      cases = [{"2.675", :half_even, "2.68"}, {"2.665", :half_even, "2.66"}, {"2.665", :half_up, "2.67"}, {"2.665", :half_down, "2.66"},
               {"-2.665", :half_up, "-2.67"}, {"-2.665", :half_down, "-2.66"}, {"-2.661", :down, "-2.66"}, {"-2.661", :floor, "-2.67"},
               {"2.661", :ceiling, "2.67"}, {"2.661", :up, "2.67"}, {"-2.669", :truncate, "-2.66"}]
      for {x, mode, want} <- cases, do: assert(Money.to_string(Money.round(Money.parse!(x), 2, mode)) == want, "#{x} #{mode}")
    end

    test "allocation by largest remainder sums to the total exactly" do
      {:ok, parts, cert} = Money.allocate(Money.parse!("100.00"), [1, 1, 1], 2)
      assert Enum.map(parts, &Money.to_string/1) == ["33.34", "33.33", "33.33"]
      assert cert.sum_equals_total and cert.within_one_unit
      {:ok, parts, cert} = Money.allocate(Money.parse!("1000.01"), ["0.1", "0.3", "0.6"], 2)
      assert Money.sum(parts, 2) |> Money.to_string() == "1000.01" and cert.within_one_unit
    end

    test "the 252-day factor: exact on whole years, truncated at 8 places" do
      {:ok, f} = Money.factor_252(Money.parse!("0.1365"), 252)
      assert Money.to_string(f) == "1.13650000"
      {:ok, f} = Money.factor_252(Money.parse!("0.1365"), 1)
      # (1.1365)^(1/252) = 1.00050788037…: truncated (Python decimal at 60 digits agrees)
      assert Money.to_string(f) == "1.00050788"
    end
  end

  # --------------------------------------------------------------- calendars

  describe "calendars" do
    test "Easter by the computus" do
      assert Calendar.easter(2024) == ~D[2024-03-31] and Calendar.easter(2025) == ~D[2025-04-20] and Calendar.easter(2026) == ~D[2026-04-05]
      assert Calendar.easter(2038) == ~D[2038-04-25] and Calendar.easter(2285) == ~D[2285-03-22]
    end

    test "ANBIMA 2025: the holidays on weekdays, 252 business days" do
      days = Calendar.holidays(:anbima, 2025) |> Enum.map(&elem(&1, 0))
      assert days == [~D[2025-01-01], ~D[2025-03-03], ~D[2025-03-04], ~D[2025-04-18], ~D[2025-04-21], ~D[2025-05-01], ~D[2025-06-19], ~D[2025-11-20], ~D[2025-12-25]]
      assert Calendar.business_days(:anbima, ~D[2025-01-02], ~D[2026-01-02]) == 252
      assert Calendar.di1_maturity("F26") == {:ok, ~D[2026-01-02]}
    end

    @tag :quantlib
    test "ANBIMA, NYSE and TARGET equal QuantLib's calendars day by day, 1990–2078" do
      out = Vapor.TestHelpers.py!("""
      import QuantLib as ql
      cals = [ql.Brazil(ql.Brazil.Settlement), ql.UnitedStates(ql.UnitedStates.NYSE), ql.TARGET()]
      for c in cals:
          hs = ql.Calendar.holidayList(c, ql.Date(1,1,1990), ql.Date(31,12,2078))
          print(' '.join('%04d-%02d-%02d' % (d.year(), d.month(), d.dayOfMonth()) for d in hs))
      """)
      [br, us, eu] = String.split(out, "\n", trim: true)
      for {cal, line} <- [anbima: br, nyse: us, target: eu] do
        mine = for y <- 1990..2078, {d, _} <- Calendar.holidays(cal, y), do: Date.to_iso8601(d)
        assert mine == String.split(line), "#{cal}"
      end
    end

    @tag :quantlib
    test "day counts and business days equal QuantLib's" do
      pairs = [{~D[2024-01-31], ~D[2024-03-31]}, {~D[2023-02-28], ~D[2024-02-29]}, {~D[2024-12-31], ~D[2027-07-15]}, {~D[2025-05-31], ~D[2025-08-31]}, {~D[2020-02-29], ~D[2021-03-01]}]
      out = Vapor.TestHelpers.py!("""
      import QuantLib as ql, sys
      dcs = [ql.Actual360(), ql.Actual365Fixed(), ql.Thirty360(ql.Thirty360.BondBasis), ql.Thirty360(ql.Thirty360.European), ql.ActualActual(ql.ActualActual.ISDA), ql.Business252(ql.Brazil(ql.Brazil.Settlement))]
      for line in sys.stdin:
          a, b = line.split()
          d1 = ql.Date(a, '%Y-%m-%d'); d2 = ql.Date(b, '%Y-%m-%d')
          print(' '.join(repr(dc.yearFraction(d1, d2)) for dc in dcs))
      """, [], Enum.map_join(pairs, "\n", fn {a, b} -> "#{a} #{b}" end) <> "\n")
      for {{a, b}, line} <- Enum.zip(pairs, String.split(out, "\n", trim: true)) do
        want = line |> String.split() |> Enum.map(&String.to_float/1)
        got = for basis <- [:act360, :act365f, :thirty360, :thirty_e360, :act_act_isda, :bus252], do: Calendar.year_fraction(basis, a, b, :anbima)
        for {w, g, basis} <- Enum.zip([want, got, [:act360, :act365f, :thirty360, :thirty_e360, :act_act_isda, :bus252]]), do: assert_in_delta(g, w, 1.0e-14, "#{basis} #{a} #{b}")
      end
    end
  end

  # ------------------------------------------------------------------ curves

  describe "curves" do
    @di """
    date = 2025-01-02
    calendar = anbima
    basis = du252
    di1 G25 = 12.33%
    di1 J25 = 13.05%
    di1 F26 = 15.02%
    di1 F27 = 15.35%
    ltn 2028-01-01 price = 652.30
    ntnf 2031-01-01 price = 800.00
    fit nss
    """

    test "a DI curve: every instrument repriced; the DI1 nodes are their own rates" do
      {:ok, r} = Curve.run(@di)
      assert r.certificate.repriced and r.certificate.max_rel_error < 1.0e-12
      f26 = Enum.find(r.nodes, &(&1.instrument =~ "F26"))
      assert_in_delta f26.zero, 0.1502, 1.0e-12
      assert f26.t == 1.0
      assert r.nss.rmse < 0.002
    end

    test "an inconsistent quote is refused with the instrument named" do
      {:error, why} = Curve.run("date = 2025-01-02\ndi1 F26 = 15%\nltn 2026-01-02 price = 1200")
      assert why =~ "same point" or why =~ "ltn"
    end

    @tag :quantlib
    test "deposits and fixed-rate bonds: the discount factors equal QuantLib's log-linear bootstrap" do
      text = """
      date = 2025-03-03
      calendar = weekends
      basis = act/365f
      compounding = simple
      deposit 2025-06-03 = 4.30% simple
      deposit 2025-09-03 = 4.25% simple
      deposit 2026-03-03 = 4.10% simple
      """
      {:ok, r} = Curve.run(text)
      out = Vapor.TestHelpers.py!("""
      import QuantLib as ql
      today = ql.Date(3,3,2025); ql.Settings.instance().evaluationDate = today
      dc = ql.Actual365Fixed(); cal = ql.NullCalendar()
      def dep(d, r):
          q = ql.QuoteHandle(ql.SimpleQuote(r))
          return ql.DepositRateHelper(q, ql.Period(ql.Date(*d) - today, ql.Days), 0, cal, ql.Unadjusted, False, dc)
      hs = [dep((3,6,2025), 0.043), dep((3,9,2025), 0.0425), dep((3,3,2026), 0.041)]
      c = ql.PiecewiseLogLinearDiscount(today, hs, dc)
      for d in [(3,6,2025), (3,9,2025), (3,3,2026), (15,7,2025), (1,1,2026)]:
          print(repr(c.discount(ql.Date(*d))))
      """)
      want = out |> String.split() |> Enum.map(&String.to_float/1)
      nodes = [{0.0, 1.0} | Enum.map(r.nodes, &{&1.t, &1.df})]
      ts = for d <- [~D[2025-06-03], ~D[2025-09-03], ~D[2026-03-03], ~D[2025-07-15], ~D[2026-01-01]], do: Date.diff(d, ~D[2025-03-03]) / 365
      for {t, w} <- Enum.zip(ts, want), do: assert_in_delta(Curve.df(nodes, t, :flat_forward), w, 1.0e-13)
    end
  end

  # ----------------------------------------------------------------- options

  describe "options" do
    test "parity, Greeks against finite differences, bounds before the implied volatility" do
      {:ok, r} = Options.run("price call S=100 K=105 T=0.5 r=5% q=2% vol=25%")
      assert r.certificate.ok
      {:ok, iv} = Options.implied_vol(:call, 6.0, 100.0, 105.0, 0.5, 0.05, 0.02)
      assert_in_delta Options.bsm(:call, 100.0, 105.0, 0.5, 0.05, 0.02, iv.sigma), 6.0, 1.0e-12
      {:error, why} = Options.implied_vol(:call, 0.5, 100.0, 50.0, 1.0, 0.05, 0.0)
      assert why =~ "lower no-arbitrage bound"
    end

    test "the American put of Longstaff & Schwartz (2001, table 1): lattice 4.486, LSM ≈ 4.47" do
      am = Options.binomial(:put, :american, 36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 801, :lr)
      assert_in_delta am, 4.486, 0.002
      lsm = MonteCarlo.lsm(:put, 36.0, 40.0, 1.0, 0.06, 0.0, 0.2, paths: 10_000, steps: 50)
      assert abs(lsm.price - am) < 4 * lsm.stderr + 0.02
      assert_in_delta lsm.european, Options.bsm(:put, 36.0, 40.0, 1.0, 0.06, 0.0, 0.2), 0.05
    end

    test "Heston: parity and the ξ → 0 limit (BSM at √v₀)" do
      p = %{v0: 0.04, kappa: 1.5, theta: 0.04, xi: 0.5, rho: -0.7}
      c = Options.heston(:call, 100.0, 95.0, 1.0, 0.03, 0.01, p); pu = Options.heston(:put, 100.0, 95.0, 1.0, 0.03, 0.01, p)
      assert_in_delta c - pu, 100 * :math.exp(-0.01) - 95 * :math.exp(-0.03), 1.0e-9
      lim = Options.heston(:call, 100.0, 100.0, 1.0, 0.0, 0.0, %{v0: 0.04, kappa: 1.0, theta: 0.04, xi: 1.0e-4, rho: 0.0})
      assert_in_delta lim, Options.bsm(:call, 100.0, 100.0, 1.0, 0.0, 0.0, 0.2), 1.0e-6
    end

    @tag :quantlib
    test "BSM, Leisen–Reimer American and Heston equal QuantLib" do
      out = Vapor.TestHelpers.py!("""
      import QuantLib as ql
      today = ql.Date(1,1,2025); ql.Settings.instance().evaluationDate = today
      dc = ql.Actual365Fixed(); exp = today + 365
      def proc(S, r, q, v):
          return ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S)), ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)), ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)), ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), v, dc)))
      e = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 105), ql.EuropeanExercise(exp)); e.setPricingEngine(ql.AnalyticEuropeanEngine(proc(100, 0.05, 0.02, 0.25)))
      a = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, 40), ql.AmericanExercise(today, exp)); a.setPricingEngine(ql.BinomialVanillaEngine(proc(36, 0.06, 0.0, 0.2), 'lr', 801))
      hp = ql.HestonProcess(ql.YieldTermStructureHandle(ql.FlatForward(today, 0.02, dc)), ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, dc)), ql.QuoteHandle(ql.SimpleQuote(100)), 0.04, 1.5, 0.04, 0.5, -0.7)
      h = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 90), ql.EuropeanExercise(exp)); h.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(hp), 1e-14, 100000))
      print(repr(e.NPV()), repr(e.delta()), repr(e.gamma()), repr(e.vega()), repr(a.NPV()), repr(h.NPV()))
      """)
      [ep, ed, eg, ev, ap, hp] = out |> String.split() |> Enum.map(&String.to_float/1)
      g = Options.greeks(:call, 100.0, 105.0, 1.0, 0.05, 0.02, 0.25)
      assert_in_delta Options.bsm(:call, 100.0, 105.0, 1.0, 0.05, 0.02, 0.25), ep, 1.0e-12
      assert_in_delta g.delta, ed, 1.0e-12
      assert_in_delta g.gamma, eg, 1.0e-12
      assert_in_delta g.vega, ev, 1.0e-10
      assert_in_delta Options.binomial(:put, :american, 36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 801, :lr), ap, 1.0e-10
      assert_in_delta Options.heston(:call, 100.0, 90.0, 1.0, 0.02, 0.01, %{v0: 0.04, kappa: 1.5, theta: 0.04, xi: 0.5, rho: -0.7}), hp, 1.0e-9
    end

    test "SVI: the smile fits; Vogt's slice has butterfly arbitrage, a fitted calm smile has none" do
      vogt = %{a: -0.0410, b: 0.1331, rho: 0.3060, m: 0.3586, sigma: 0.4153}
      a = Options.svi_arbitrage(vogt)
      refute a.free
      assert a.g_min < 0
      {:ok, r} = Options.run("smile T=0.5 F=100\n70 32%\n80 28%\n90 25%\n100 23%\n110 22%\n120 22.5%\n130 23.5%")
      assert r.rmse_total_variance < 2.0e-4
      assert r.certificate.free
    end

    test "static arbitrage across strikes: a convexity violation becomes a butterfly that pays to enter" do
      r = Options.static_arbitrage([{90.0, 14.0}, {100.0, 9.5}, {110.0, 3.0}], 0.0, 1.0)
      refute r.arbitrage_free
      [v | _] = r.violations
      assert v.kind =~ "convexity" and v.cost < 0 and v.payoff_nonnegative
      assert Options.static_arbitrage([{90.0, 14.0}, {100.0, 7.5}, {110.0, 3.0}], 0.0, 1.0).arbitrage_free
    end
  end

  # -------------------------------------------------------- Monte Carlo

  describe "Monte Carlo on the native worker" do
    @tag :native
    test "the same bits as the oracle and across thread counts; the interval covers Black–Scholes; Itô forgotten is caught" do
      {:ok, r} = MonteCarlo.price(s0: 100, k: 100, t: 1.0, r: 0.05, sigma: 0.2, paths: 4096, steps: 32, barrier: 85, threads: 2)
      assert r.certificate.oracle_parity == true
      assert r.certificate.threads.identical_bits
      assert r.european.covers and r.geometric_asian.covers
      assert r.arithmetic_asian_cv.variance_reduction > 100
      assert r.certificate.binary64_agreement.max_abs < 1.0e-3
      {:ok, c} = MonteCarlo.price(s0: 100, k: 100, t: 1.0, r: 0.05, sigma: 0.2, paths: 4096, steps: 32, drift: :no_ito)
      refute c.european.covers
    end

    test "the Wichmann–Hill stream on the BEAM is exact integer arithmetic (period-1 sanity)" do
      [a, b, c] = MonteCarlo.wh_seeds(1, 0) |> Enum.map(&trunc/1)
      assert a in 1..30268 and b in 1..30306 and c in 1..30322
      us = MonteCarlo.wh_uniforms(1, 0, 1000)
      assert Enum.all?(us, &(&1 > 0 and &1 < 1))
      assert_in_delta Num.mean(us), 0.5, 0.03
    end

    @tag :scipy
    test "PPND7 and the full-precision Φ⁻¹ against scipy.special.ndtri" do
      ps = [1.0e-9, 1.0e-4, 0.01, 0.2, 0.5, 0.77, 0.99, 0.999999]
      out = Vapor.TestHelpers.py!("from scipy.special import ndtri\nprint(' '.join(repr(float(ndtri(p))) for p in #{inspect(ps)}))")
      for {p, w} <- Enum.zip(ps, out |> String.split() |> Enum.map(&String.to_float/1)) do
        assert_in_delta Num.ninv(p), w, 1.0e-9 * max(1.0, abs(w))
        assert_in_delta MonteCarlo.ppnd7_f64(p), w, 1.0e-6
      end
    end
  end

  # ------------------------------------------------------------------ risk

  describe "risk" do
    test "Kupiec: a normal VaR on fat tails is rejected; historical passes; the test's size is near 5 %" do
      {:ok, d} = Backtest.data(%{data: {"t", %{"n" => 3000.0, "sigma" => 0.01, "seed" => 4.0, "nu" => 3.0}}, csv: []})
      rets = Enum.zip(d.close, tl(d.close)) |> Enum.map(fn {a, b} -> b / a - 1 end)
      bad = Risk.backtest(Risk.rolling_var(rets, 500, 0.99, :normal), 0.99)
      good = Risk.backtest(Risk.rolling_var(rets, 500, 0.99, :historical), 0.99)
      assert bad.kupiec.p_value < 0.05 and bad.exceptions > bad.expected
      assert good.kupiec.p_value > 0.05
      # size: exceptions drawn at exactly 1 % — the test rejects about 5 % of the time
      rej = Enum.count(1..400, fn s ->
        hits = for k <- 0..499, do: (if Num.u01(s, k) < 0.01, do: {0.0, -1.0}, else: {0.0, 1.0})
        Risk.backtest(hits, 0.99).kupiec.p_value < 0.05
      end)
      assert rej in 6..40
    end

    test "Basel zones for 250 days at 99 %" do
      mk = fn x -> for k <- 1..250, do: (if k <= x, do: {1.0, -2.0}, else: {1.0, 0.0}) end
      assert Risk.backtest(mk.(4), 0.99).zone == :green
      assert Risk.backtest(mk.(5), 0.99).zone == :yellow
      assert Risk.backtest(mk.(10), 0.99).zone == :red
    end

    test "minimum variance: KKT holds; risk parity: contributions equal; HRP sums to one" do
      sigma = [[0.04, 0.006, 0.002], [0.006, 0.09, 0.009], [0.002, 0.009, 0.0225]]
      {:ok, mv} = Risk.min_variance(sigma)
      assert_in_delta mv.certificate.budget, 1.0, 1.0e-12
      assert mv.certificate.stationarity < 1.0e-12 and mv.certificate.bounds_ok
      rp = Risk.risk_parity(sigma)
      assert rp.certificate.max_budget_error < 1.0e-12
      assert_in_delta Enum.sum(Risk.hrp(sigma).weights), 1.0, 1.0e-12
    end
  end

  # -------------------------------------------------------------- backtests

  describe "backtest noise gates" do
    test "look-ahead is caught by prefix invariance, with the day" do
      {:ok, r} = Backtest.run("data = gbm n=800 sigma=0.01 seed=2\nsignal = sign(lead(close) - close)")
      refute r.lookahead.clean
      assert r.best.stats.sharpe_annual > 5
      assert r.verdict =~ "look-ahead"
      {:ok, c} = Backtest.run("data = gbm n=800 sigma=0.01 seed=2\nsignal = sign(center(close))")
      refute c.lookahead.clean
    end

    test "noise: the best of 30 moving-average crossovers fails the deflated Sharpe" do
      {:ok, r} = Backtest.run("data = gbm n=2520 sigma=0.01 seed=5\nsweep fast = 5..30 step 5\nsweep slow = 40..120 step 20\nsignal = sign(ema(close, fast) - ema(close, slow))\ncost = 2bp")
      assert r.lookahead.clean
      assert r.deflated_sharpe.dsr < 0.95
      refute r.verdict =~ "every gate passed"
    end

    test "a planted signal passes every gate" do
      {:ok, r} = Backtest.run("data = ar1 n=5040 phi=0.15 sigma=0.01 seed=3\nsweep k = 1..3\nsignal = sign(sma(ret(close), k))\ncost = 1bp")
      assert r.lookahead.clean and r.deflated_sharpe.dsr > 0.95 and r.pbo.pbo < 0.5 and r.reality_check.p_value < 0.05
      assert r.verdict =~ "every gate passed"
    end
  end

  # ---------------------------------------------------- LP and arbitrage

  describe "exact LP and arbitrage" do
    test "optimal with dual, infeasible with Farkas, unbounded with a ray — each checked" do
      {:ok, o} = LP.solve("maximize 3x + 2y\nx + y <= 4\nx + 3y <= 6\nx <= 3")
      assert o.status == :optimal and o.objective == {11, 1} and o.check.accepted
      {:ok, i} = LP.solve("maximize x + y\nx + y <= 1\nx + y >= 2")
      assert i.status == :infeasible and i.check.accepted
      {:ok, u} = LP.solve("maximize x - y\nx - y >= 1\nfree y")
      assert u.status == :unbounded and u.check.accepted
      # a wrong proposal is rejected by the same check
      {:ok, p} = LP.parse("maximize 3x + 2y\nx + y <= 4\nx + 3y <= 6\nx <= 3")
      refute LP.check(p, %{status: :optimal, x: %{"x" => {2, 1}, "y" => {2, 1}}, y: [{2, 1}, {0, 1}, {1, 1}]}).accepted
    end

    @tag :scipy
    test "random LPs: the exact optimum equals SciPy's HiGHS to 10⁻⁹" do
      for seed <- 1..6 do
        n = 5; m = 4
        u = fn k -> Num.u01(seed, k) end
        a = for i <- 0..(m - 1), do: (for j <- 0..(n - 1), do: trunc(u.(i * 10 + j) * 9) + 1)
        b = for i <- 0..(m - 1), do: trunc(u.(100 + i) * 40) + 10
        c = for j <- 0..(n - 1), do: trunc(u.(200 + j) * 7) + 1
        vars = for j <- 0..(n - 1), do: "x#{j}"
        txt = "maximize " <> Enum.map_join(Enum.zip(c, vars), " + ", fn {cc, v} -> "#{cc}#{v}" end) <> "\n" <>
          Enum.map_join(Enum.zip(a, b), "\n", fn {row, bi} -> Enum.map_join(Enum.zip(row, vars), " + ", fn {x, v} -> "#{x}#{v}" end) <> " <= #{bi}" end)
        {:ok, r} = LP.solve(txt)
        out = Vapor.TestHelpers.py!("from scipy.optimize import linprog\nr = linprog(#{inspect(Enum.map(c, &(-&1)), charlists: :as_lists)}, A_ub=#{inspect(a, charlists: :as_lists)}, b_ub=#{inspect(b, charlists: :as_lists)}, method='highs')\nprint(repr(-r.fun))")
        assert_in_delta LP.to_float(r.objective), String.to_float(String.trim(out)), 1.0e-9
        assert r.check.accepted
      end
    end

    test "the logic desk decides LPs and checks proposals" do
      {:ok, r} = Vapor.Logic.run("minimize 2x + 3y\nx + y >= 10\nx - y = 2")
      assert r.kind == "linear" and r.verdict == "optimal" and r.certified
      {:ok, c} = Vapor.Logic.check("maximize x + y\nx + y <= 1\nx + y >= 2", %{"farkas" => ["1", "-1"]})
      assert c.accepted
      {:ok, c} = Vapor.Logic.check("maximize x + y\nx + y <= 1\nx + y >= 2", %{"farkas" => ["1", "1"]})
      refute c.accepted
    end

    test "fundamental theorem: state prices or an arbitrage, never neither" do
      {:ok, ok} = Arbitrage.run("states = up, down\nbond bid=0.95 ask=0.96 payoff = 1, 1\nstock bid=100 ask=100.5 payoff = 120, 90\ncall bid=10 ask=10.5 payoff = 20, 0")
      assert ok.arbitrage == false and ok.certificate.checked_exactly
      {:ok, arb} = Arbitrage.run("states = up, down\nbond bid=0.95 ask=0.96 payoff = 1, 1\nstock bid=100 ask=100.5 payoff = 120, 90\ncall bid=14 ask=14.5 payoff = 20, 0")
      assert arb.arbitrage and arb.certificate.checked_exactly and arb.cost.value < 0
      {:ok, bf} = Arbitrage.run("calls T=1 r=5%\n90 bid=14.1 ask=14.4\n100 bid=9.4 ask=9.6\n110 bid=3.6 ask=3.9")
      assert bf.arbitrage and Enum.map(bf.portfolio, & &1.asset) |> Enum.sort() == ["C(100)", "C(110)", "C(90)"]
      {:ok, fx} = Arbitrage.run("fx\nUSD/BRL bid=5.40 ask=5.41\nEUR/USD bid=1.08 ask=1.081\nEUR/BRL bid=5.90 ask=5.91")
      assert fx.arbitrage and fx.gross.value > 1
      {:ok, nfx} = Arbitrage.run("fx\nUSD/BRL bid=5.40 ask=5.41\nEUR/USD bid=1.08 ask=1.081\nEUR/BRL bid=5.833 ask=5.847")
      refute nfx.arbitrage
    end
  end

  # ----------------------------------------------------------- order book

  defp random_events(seed, n) do
    {evs, _} =
      Enum.map_reduce(0..(n - 1), [], fn i, live ->
        u = fn k -> Num.u01(seed, i * 6 + k) end
        cond do
          u.(0) < 0.15 and live != [] -> id = Enum.at(live, trunc(u.(1) * length(live))); {%{type: :cancel, id: id}, List.delete(live, id)}
          u.(0) < 0.22 and live != [] -> id = Enum.at(live, trunc(u.(1) * length(live))); {%{type: :modify, id: id, price: (if u.(2) < 0.5, do: nil, else: 95 + trunc(u.(3) * 10)), qty: 1 + trunc(u.(4) * 12)}, live}
          u.(0) < 0.23 -> {%{type: :kill, owner: trunc(u.(1) * 5)}, live}
          true ->
            side = if u.(1) < 0.5, do: :buy, else: :sell
            tif = cond do u.(5) < 0.1 -> :ioc; u.(5) < 0.17 -> :fok; true -> :gtc end
            price = if u.(2) < 0.05, do: nil, else: (if side == :buy, do: 96 + trunc(u.(3) * 8), else: 97 + trunc(u.(3) * 8))
            {%{type: :new, id: i + 1, owner: trunc(u.(4) * 5), side: side, price: price, qty: 2 + trunc(u.(4) * 37 * u.(3)), tif: tif, post_only: u.(5) > 0.93}, [i + 1 | live]}
        end
      end)
    evs
  end

  describe "order book" do
    test "price–time priority, partial fills, IOC remainder, FOK all-or-none, post-only, STP" do
      evs = [%{type: :new, id: 1, owner: :a, side: :sell, price: 101, qty: 5},
             %{type: :new, id: 2, owner: :b, side: :sell, price: 101, qty: 5},
             %{type: :new, id: 3, owner: :c, side: :sell, price: 100, qty: 2},
             %{type: :new, id: 4, owner: :d, side: :buy, price: 101, qty: 8, tif: :ioc},
             %{type: :new, id: 5, owner: :d, side: :buy, price: 101, qty: 10, tif: :fok},
             %{type: :new, id: 6, owner: :d, side: :buy, price: 101, qty: 1, post_only: true},
             %{type: :new, id: 7, owner: :b, side: :buy, price: 101, qty: 3}]
      s = Book.session(evs)
      j = s.journal
      fills = for {:fill, f} <- Enum.at(j, 3).reports, do: {f.maker, f.price, f.qty}
      assert fills == [{3, 100, 2}, {1, 101, 5}, {2, 101, 1}]
      assert {:cancelled, 5, 10, :fok} in Enum.at(j, 4).reports
      assert {:cancelled, 6, 1, :post_only} in Enum.at(j, 5).reports
      # b's own resting sell is first in line: self-trade prevention cancels the taker
      assert {:cancelled, 7, 3, :stp} in Enum.at(j, 6).reports
      assert Book.Check.check(j).ok
    end

    test "differential fuzzing: 6 000 random events per policy, the naive engine agrees and every invariant holds" do
      for {stp, seed} <- [cancel_taker: 1, cancel_resting: 2, off: 3] do
        s = Book.session(random_events(seed, 6000), stp: stp)
        c = Book.Check.check(s.journal, stp: stp)
        assert c.ok, "#{stp}: #{inspect(c.failures)}"
      end
    end

    test "a tampered journal is caught: a changed fill, a reordered queue, a broken chain" do
      s = Book.session(random_events(9, 800))
      i = Enum.find_index(s.journal, fn e -> Enum.any?(e.reports, &match?({:fill, _}, &1)) end)
      e = Enum.at(s.journal, i)
      forged = %{e | reports: Enum.map(e.reports, fn {:fill, f} -> {:fill, %{f | price: f.price + 1}}; r -> r end)}
      c = Book.Check.check(List.replace_at(s.journal, i, forged))
      refute c.ok
      assert Enum.any?(c.failures, &(elem(&1, 1) in [:hash_chain_broken, :reports_differ]))
      # forged consistently (hash recomputed): the chain holds, the replay does not
      h_prev = if i == 0, do: <<0::256>>, else: Enum.at(s.journal, i - 1).hash
      rehashed = %{forged | hash: :crypto.hash(:sha256, h_prev <> Vapor.Canonical.encode(Map.delete(forged, :hash)))}
      c2 = Book.Check.check(List.replace_at(s.journal, i, rehashed) |> Enum.take(i + 1))
      refute c2.ok
      assert Enum.any?(c2.failures, &(elem(&1, 1) == :reports_differ))
    end

    test "Merkle inclusion proof of a fill" do
      s = Book.session(random_events(4, 300))
      e = Enum.find(s.journal, fn e -> Enum.any?(e.reports, &match?({:fill, _}, &1)) end)
      pr = Book.prove(s.book, e.seq)
      assert Vapor.Merkle.verify(pr.leaf, pr.proof, pr.root)
      assert Base.encode16(pr.root, case: :lower) == s.merkle_root
    end

    test "ITCH 5.0: every message type round-trips at its specified length; the feed rebuilds the engine's book" do
      msgs = [%{type: :system, ts: 1, event: ?O}, %{type: :directory, ts: 2, stock: "PETR4"}, %{type: :add, ts: 3, ref: 7, side: :buy, shares: 100, stock: "PETR4", price: 381_200},
              %{type: :add_mpid, ts: 4, ref: 8, side: :sell, shares: 50, stock: "VALE3", price: 612_300, mpid: "XPIN"}, %{type: :executed, ts: 5, ref: 7, shares: 10, match: 99},
              %{type: :executed_price, ts: 6, ref: 7, shares: 5, match: 100, printable: true, price: 381_100}, %{type: :cancel, ts: 7, ref: 8, shares: 20},
              %{type: :delete, ts: 8, ref: 8}, %{type: :replace, ts: 9, ref: 7, new_ref: 70, shares: 85, price: 381_000},
              %{type: :trade, ts: 10, ref: 0, side: :buy, shares: 300, stock: "ITUB4", price: 330_000, match: 101}]
      for m <- msgs do
        b = Itch.encode(m)
        assert byte_size(b) == Itch.lengths()[:binary.first(b)]
        {:ok, d} = Itch.decode(b)
        assert Itch.encode(d) == b
      end
      s = Book.session(random_events(12, 2000))
      {:ok, back} = Itch.unframe(Itch.frame(Itch.from_session(s)))
      assert Itch.consistent?(s, back) |> Map.take([:book_equal, :volume_equal]) == %{book_equal: true, volume_equal: true}
    end

    test "pre-trade gate: fat finger, collar, position, kill switch — refused, journaled, re-checked" do
      evs = [%{type: :new, id: 1, owner: "mm", side: :sell, price: 10010, qty: 100, ts: 1}, %{type: :new, id: 2, owner: "mm", side: :buy, price: 9990, qty: 100, ts: 2},
             %{type: :new, id: 3, owner: "algo", side: :buy, price: 10010, qty: 50, ts: 3}, %{type: :new, id: 4, owner: "algo", side: :buy, price: 10010, qty: 50_000, ts: 4},
             %{type: :new, id: 5, owner: "algo", side: :buy, price: 13000, qty: 10, ts: 5}, %{type: :new, id: 6, owner: "algo", side: :buy, price: 10010, qty: 50, ts: 6},
             %{type: :new, id: 7, owner: "algo", side: :buy, price: 10000, qty: 60, ts: 7}, %{type: :kill, owner: "mm", ts: 8},
             %{type: :new, id: 8, owner: "mm", side: :sell, price: 10010, qty: 10, ts: 9}]
      limits = %{default: %{max_qty: 1000, collar: 0.05, max_position: 120, max_rate: 100}}
      s = PreTrade.session(evs, limits)
      reasons = for e <- s.journal, {:risk_rejected, _, why} <- e.reports, do: why
      assert reasons == [:max_qty, :price_collar, :max_position, :kill_switch]
      assert PreTrade.check(s.journal, limits).ok
      assert Book.Check.check(s.journal).ok
      # a journal claiming a refusal that had no cause is caught
      j2 = Enum.map(s.journal, fn e -> if e.seq == 4, do: put_in(e, [:event, :reason], :max_notional), else: e end)
      refute PreTrade.check(j2, limits).ok
    end

    @tag :simplefix
    test "FIX 4.4: our BodyLength and CheckSum are simplefix's, both ways" do
      ours = Fix.encode("D", [{11, "ord-1"}, {55, "PETR4"}, {54, "1"}, {38, "100"}, {40, "2"}, {44, "38.12"}, {59, "0"}], seq: 7, time: "20261005-15:00:00.000")
      out = Vapor.TestHelpers.py!("""
      import simplefix, sys
      m = simplefix.FixMessage()
      m.append_pair(8, 'FIX.4.4'); m.append_pair(35, 'D'); m.append_pair(49, 'CLIENT'); m.append_pair(56, 'VAPOR'); m.append_pair(34, 7)
      m.append_pair(52, '20261005-15:00:00.000')
      for t, v in [(11,'ord-1'),(55,'PETR4'),(54,'1'),(38,'100'),(40,'2'),(44,'38.12'),(59,'0')]: m.append_pair(t, v)
      sys.stdout.write(m.encode().decode().replace('\\x01', '|'))
      """)
      assert Fix.readable(ours) == out
      {:ok, fields} = Fix.parse(out)
      {:ok, ev} = Fix.to_event(fields)
      assert ev.side == :buy and ev.price == 3812 and ev.qty == 100
      assert {:error, why} = Fix.parse(String.replace(out, "PETR4", "PETR3"))
      assert why =~ "CheckSum"
    end
  end

  # ------------------------------------------------------ microstructure

  describe "microstructure" do
    test "Hawkes: the fit recovers a planted process; Poisson fails the time-rescaling test" do
      ts = Micro.hawkes_simulate(1.0, 0.6, 1.5, 2000.0, 3)
      f = Micro.hawkes_fit(ts, 2000.0)
      assert_in_delta f.branching_ratio, 0.4, 0.08
      assert f.time_rescaling.p_value > 0.05 and f.poisson_time_rescaling.p_value < 1.0e-6
    end

    test "Avellaneda–Stoikov: the inventory strategy halves the P&L dispersion and cuts inventory by two thirds (as in the paper)" do
      r = Micro.avellaneda_stoikov(runs: 400)
      assert r.certificate.pnl_dispersion_ratio < 0.6 and r.certificate.inventory_dispersion_ratio < 0.45
      assert_in_delta r.symmetric.std_pnl, 13.43, 2.5
    end

    test "Almgren–Chriss: the closed form is the numeric optimum; TWAP when λ = 0" do
      r = Micro.almgren_chriss(%{x: 1.0e6, n: 5, t: 5.0, sigma: 0.95, eta: 2.5e-6, gamma: 2.5e-7, epsilon: 0.0625, lambda: 1.0e-6})
      assert r.certificate.relative < 1.0e-12
      t = Micro.almgren_chriss(%{x: 1.0e6, n: 4, t: 4.0, sigma: 0.95, eta: 2.5e-6, gamma: 0.0, lambda: 0.0})
      assert Enum.map(t.trajectory, &round/1) == [1_000_000, 750_000, 500_000, 250_000, 0]
    end

    test "an exchange session audits itself: naive replay, pre-trade, ITCH, replay hash; the planted Hawkes is found" do
      r = Exchange.simulate(steps: 900, seed: 3)
      assert r.certificate.book_check.ok and r.certificate.pre_trade.ok and r.certificate.itch.book_equal
      assert r.measures.hawkes.poisson_time_rescaling.p_value < r.measures.hawkes.time_rescaling.p_value
      assert Exchange.simulate(steps: 900, seed: 3).head == r.head
      refute Exchange.simulate(steps: 900, seed: 4).head == r.head
    end
  end

  # ---------------------------------------------------------- front door

  test "the desk's front door answers every kind" do
    for {k, t} <- [{"calendar", "du 2025-01-02 2026-01-02\nallocate 10.00 1 2"}, {"curve", "date = 2025-01-02\ndi1 F26 = 15%"}, {"options", "price put S=100 K=100 T=1 vol=20%"},
                   {"risk", "data = t n=800 seed=2\nwindow = 250"}, {"portfolio", "assets = 5 n = 300"}, {"backtest", "data = gbm n=300\nsignal = sign(ret(close))"},
                   {"arbitrage", "fx\nUSD/BRL bid=5.40 ask=5.41\nEUR/USD bid=1.08 ask=1.081\nEUR/BRL bid=5.90 ask=5.91"}, {"book", "buy 10 @ 1.00 owner=a\nsell 10 @ 1.00 owner=b"},
                   {"micro", "execution X=1e5 N=10"}] do
      assert {:ok, _} = Vapor.Finance.run(k, t), k
    end
    assert {:error, _} = Vapor.Finance.run("book", "buy lots")
  end
end
