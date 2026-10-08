# Finance and the trading desk — every number with what lets you judge it

> Request (0.13), translated: "support for finance, HFT, attack the current limitations and
> TODO […] quality tests to guarantee that the answers are not
> just noise". Scrutiny: [DIRECTIVE.md §16](DIRECTIVE.md).

Finance is the domain where "the number came out of the computer" is worth least. A
price that changes when the calculation changes machines, a backtest that looked at
tomorrow by accident, the best of fifty strategies presented as if it
were the only one, an order book that no outsider can audit —
these are the industry's real pains, and all of them are pains of **verifiability**,
not of speed. vapor already had the answer in principle (a search
proposes, a verifier decides; the same bits on every substrate); this
round applies it to the market.

`Vapor.Finance.run/2` (the desk) · console *Markets → Finance* and
*Markets → Trading desk* · MCP `finance_run` and `arbitrage_check` ·
`mix vapor.finance KIND FILE` · tests `finance_test.exs` (with
QuantLib, SciPy and simplefix as external oracles) · §5i of
[bench/QUALITY.md](bench/QUALITY.md).

| task | what goes in (text) | the certificate that comes out |
|---|---|---|
| calendar and money | `du`, `holidays`, `adjust`, `yf`, `allocate`, `factor`… | holidays by rule (computus), checked against QuantLib; an allocation that sums exactly |
| curve | DI1, LTN, NTN-F, deposits, bonds, swaps | every instrument repriced; negative forwards flagged |
| options | `price`, `iv`, `american`, `heston`, `smile` | parity; Greeks against finite differences; no-arbitrage bounds **before** solving; Durrleman's g(k) |
| Monte Carlo | S, K, T, r, σ, paths, barrier | bits = exact oracle and = another thread count; 99 % interval against the closed form |
| risk | P&L series, VaR method | Kupiec, Christoffersen, Basel zone |
| portfolio | asset returns | KKT of the minimum variance; risk contributions |
| backtest | data, sweep, signal, cost | **four gates**: prefix invariance, deflated Sharpe, PBO, Reality Check |
| arbitrage | states, calls, FX | **either** the arbitrage portfolio **or** the state prices — in exact rationals |
| order book | orders as text | SHA-256 journal + Merkle root; independent naive engine; ITCH feed; FIX |
| exchange session | agents and seed | the three certificates + the same hash head on replay |
| microstructure | Hawkes, Avellaneda–Stoikov, Almgren–Chriss | time-rescaling test; the paper reproduced; closed form = numerical optimum |

## 1. Exact money

`Vapor.Finance.Money`: a value is an integer and a decimal scale
(`%{c: 12345, e: 2}` = 123.45). Sum, subtraction and product are exact; every
rounding is an explicit act with a named mode (`half_even`,
`half_up`, `half_down`, `down`/`truncate`, `up`, `floor`, `ceiling`) —
a single primitive, `div_round/3`, tested on ties on both sides of
zero.

- `allocate/3` allocates by the **largest remainder** (Hamilton): R$ 100.00 in three
  parts gives 33.34 + 33.33 + 33.33, and the sum is the total **by construction**;
  the certificate also says that each part is less than one unit from
  its exact fraction. The control in §5i: the floor in binary64 loses a
  cent (99.99).
- `factor_252/3` computes (1 + r)^(du/252) — the factor of the Brazilian
  convention — with an integer root in big-integer arithmetic and
  **truncates** at the 8th decimal place as ANBIMA does. (1.1365)^(1/252) = 1.00050788…:
  Python's `decimal` with 60 digits agrees; the decision at the 8th place is not
  a floating-point accident.

## 2. Calendars and day counts

`Vapor.Finance.Calendar`: each holiday is a **rule** — fixed date,
n-th weekday, or movable feast by the Gregorian computus (Easter
by the anonymous algorithm of 1876; Carnival = Easter − 48 and − 47; Good
Friday − 2; Corpus Christi + 60). The closures that no rule produces
(11 September 2001, Hurricane Sandy, presidential days of mourning) are
data, each with its reason.

| calendar | rules | check |
|---|---|---|
| `:anbima` (= B3, DU/252) | national + movable; 20 November since 2024 (Law 14.759/2023) | **903 holidays on business days, 1990–2078, identical to `ql.Brazil(Settlement)`** |
| `:nyse` | observed dates (Saturday → Friday, Sunday → Monday; 1 January on a Saturday is **not** moved back), MLK (1998–), Juneteenth (2022–), special closures | **848 identical to `ql.UnitedStates(NYSE)`** |
| `:target` | TARGET2: Good Friday, Easter Monday, 1 May and 26 December since 2000 | **401 identical to `ql.TARGET()`** |

In the first comparison NYSE diverged on a single day — 27/04/1994, the mourning
for President Nixon — and TARGET on dates before 2000 (when the
system did not exist and QuantLib closes only 1 January): the two
differences became data and rules, and the test now requires day-by-day
equality over 89 years.

Day counts (`year_fraction/4`): `:bus252` (DU/252), `:act360`, `:act365f`,
`:thirty360` (US bond basis, ISDA 4.16(f)), `:thirty_e360`,
`:act_act_isda` — all equal to QuantLib to 10⁻¹⁴ in the edge cases
(end of February, day 31, leap years). `adjust/3` (following,
modified following, preceding…), `add_business_days/3`, `di1_maturity/1`
(F26 → first business day of January 2026).

## 3. Curves

`Vapor.Finance.Curve` reads the instruments the way a desk writes them:

```
date = 2025-01-02
calendar = anbima
basis = du252
interpolation = flat_forward
di1 F26 = 15.02%
ltn 2028-01-01 price = 652.30
ntnf 2031-01-01 price = 800.00
swap 5y = 4.10%
fit nss
```

The *bootstrap* puts a node at the last cash flow of each instrument and finds (Brent)
the discount factor that reprices it, with the earlier nodes fixed. The
interpolation is `flat_forward` (log-linear in the discount — the convention of the
DI curve) or `linear_zero`. The cash flows follow the conventions: DI1 with PU
100,000/(1 + r)^(DU/252); LTN with face 1,000 paying on the next business day;
NTN-F with a semi-annual coupon 1,000·(√1.1 − 1) on 1 January and 1 July;
bonds with accrued interest for the clean price; single-curve par swaps.

**The certificate** reprices each instrument from the finished curve and
gives the worst relative error (4.5·10⁻¹⁶ on the example DI curve); it lists the
forwards between nodes and **flags the negative ones** — the "inconsistent
quote" example (January at 15 %, July at 9 %) is repriced but comes out with the
negative forward named. The control in §5i: the same DI1 counted in calendar
days gets the PU wrong by 2.5·10⁻³. `fit nss` fits Nelson–Siegel–Svensson
(exact β by least squares on a grid of τ, then Nelder–Mead).
Simple deposits match QuantLib's `PiecewiseLogLinearDiscount`
to 10⁻¹³ (`finance_test.exs`).

## 4. Options

`Vapor.Finance.Options`:

| function | model | certificate |
|---|---|---|
| `bsm/7`, `greeks/7` | Black–Scholes–Merton with dividends | parity (7·10⁻¹⁵); each Greek against the central difference (< 2·10⁻⁸ relative) |
| `black76/6`, `bachelier/6` | futures (lognormal / normal: negative rates, spreads) | parity |
| `implied_vol/8` | inversion by Brent on σ ∈ [10⁻⁶, 10] | the **no-arbitrage bounds checked first**: a price below the discounted intrinsic value is refused with the bound; the repricing error; the vega and how much a price *tick* moves σ |
| `binomial/10` | Cox–Ross–Rubinstein and Leisen–Reimer (Peizer–Pratt 2), European and American | the tree's European against BSM |
| `heston/7` | Heston by Lewis's single integral and Albrecher's "little trap" characteristic function | parity; the limit ξ → 0 = BSM at √v₀ |
| `svi_fit/2`, `svi_arbitrage/2`, `svi_calendar/3` | raw SVI | Durrleman's g(k) ≥ 0 on a fine grid (butterfly); total variance increasing in T (calendar); the risk-neutral density |
| `static_arbitrage/3` | model-free: calls across strikes | monotonicity, slope in [−e^(−rT), 0], convexity — each violation returned as the portfolio that exploits it, with the *payoff* checked at every vertex |

**Against QuantLib** (`finance_test.exs`): BSM and Δ, Γ to 10⁻¹², vega to
10⁻¹⁰; Leisen–Reimer American with 801 steps to 10⁻¹⁰ (4.486076);
Heston to 10⁻⁹ (QuantLib's adaptive Gauss–Kronrod integrator and the
composite Gauss–Legendre here agree up to the 9th decimal place). The CRR here is the
textbook one (p = (e^{rΔ} − d)/(u − d)); QuantLib's uses the
additive log approximation — they differ by 1.6·10⁻⁵ and the document says which is
which.

The American put from table 1 of Longstaff & Schwartz (S = 36, K = 40,
σ = 0.2, T = 1, r = 6 %): tree 4.4861, LSM here 4.4788 ± 0.0143 (20,000
antithetic paths, regression on {1, x, x²}); the paper gives 4.472 for
LSM. The SVI slice attributed to Vogt in Gatheral & Jacquier (2014) — the
classic butterfly-arbitrage example — is detected (minimum g
−0.033 for k ∈ [0.645, 1.255]); a calm fitted smile comes out with g ≥ 0.

## 5. Monte Carlo compiled to the worker — the same bits on every machine

The pain: a risk number that changes when the calculation moves to another machine,
another thread count or a GPU is a model-risk finding
waiting to happen (parallel reductions in another order, an FMA here
and not there, a different `exp`). `Vapor.Finance.MonteCarlo` writes the
path step as **terms of vapor's algebra** and compiles it for the native
worker:

- the **generator** is Wichmann–Hill (1982) inside the program: three Lehmer
  generators whose products stay below 2²³, so that each
  state update is **exact in binary32**; the modulo is done with the
  2²³ rounding trick and a correction step — all the generation
  runs in the worker, the BEAM only seeds;
- the normal deviate is Wichura's AS241 (PPND7, 7 digits: what binary32
  can hold), with canonical `log` and `rsqrt`;
- per step: L ← L + μΔ + σ√Δ·Φ⁻¹(u); A ← A + e^L; G ← G + L; alive ← alive·[L > ln(B/S₀)].

**Certificates**, all in the same answer: (i) the first call,
compiled for 64 lanes and run by the **exact oracle**, equal bit for bit
to the worker's first 64 lanes; (ii) the whole simulation repeated on a
worker with **2 threads**: the same bits; (iii) the same uniforms in
binary64 on the BEAM (full-precision Φ⁻¹): maximum difference 2·10⁻⁵ in the
*payoff*; (iv) the 99 % interval against the closed form — BSM for the
European, Kemna–Vorst for the discrete geometric Asian; (v) the arithmetic
Asian with the geometric one as a **control variate** (variance ÷
1,300). The **control**: forgetting Itô's −σ²/2 gives z = 8.7 — the biased
estimator is caught, not averaged away.

Measured (Xeon 2 vCPUs, AVX-512, one worker core): 8,192 paths ×
64 steps in 125 ms against 750 ms estimated for the BEAM in binary64 (6×);
16,384 × 32 in 86 ms against 1,950 ms (23×). Compilation costs ~0.6 s per
unrolled step and per ISA (the allocation verifier extracted from Lean is
quadratic); hence one step per call and only the machine's ISA
(`all_targets: true` compiles all four).

Limits, stated: Wichmann–Hill is an old generator (period ~7·10¹², it fails
modern batteries such as BigCrush); it serves pricing, not
cryptography nor extreme-tail studies — `rng: :host` uses splitmix64 on the
BEAM and compares.

## 6. Market risk

`Vapor.Finance.Risk`: historical, normal, Cornish–Fisher and EWMA VaR and ES
(λ = 0.94); one-step forecasts on a rolling window; **backtest** with
Kupiec (proportion of failures, χ²₁), Christoffersen (independence, χ²₁; and
conditional coverage, χ²₂) and the Basel zone (green 0–4, yellow 5–9,
red ≥ 10 exceptions in 250 days at 99 %). A risk test that never
rejects is as suspect as one that always rejects: the **size** of the
Kupiec test is measured (exceptions drawn at exactly 1 %: it rejects in 1.5–10 % of the
400 series) and its **power** too — on a Student t with ν = 3, the normal
VaR is rejected (p = 0.038) and the historical one is not (p = 0.43).

Portfolios: Ledoit–Wolf covariance (target μI), **long-only minimum
variance** by active set with the KKT conditions as certificate
(stationarity < 10⁻¹², multipliers with the right sign), **risk
parity** by Newton on Spinu's convex formulation (contributions equal
to the budgets to 10⁻¹²), López de Prado's HRP.

## 7. Backtests with noise gates

A backtest is a machine for producing Sharpe ratios; try enough variants
and one will look good on pure noise. The industry's two silent failures
are **look-ahead** (the signal used what it would not have had) and
**selection** (the best of many trials presented as the only one).
`Vapor.Finance.Backtest`:

```
data = ar1 n=5040 phi=0.15 sigma=0.01 seed=3
sweep k = 1..3
signal = sign(sma(ret(close), k))
cost = 1bp
```

The signal language has causal operators (`lag`, `diff`, `ret`, `sma`,
`ema`, `std`, `zscore`, `rmax`, `rmin`, `rsi`, `sign`, `clip`, `if`…) and
accepts, because people write them, three that peek (`lead`, `center`,
`normalize` with the whole sample). The position over (t, t+1] is the signal at t;
the cost falls on |Δposition|.

1. **Prefix invariance — the certificate of absence of look-ahead.**
   The signal is recomputed over truncated histories `close[0..c]` at eight
   cut points and must be **equal, bit for bit**, to the one computed with the
   whole history up to c. It is a black-box test of the whole pipeline: it does not
   need to trust the operators. A peek (`lead`), a normalisation
   by the whole sample (`center`), a typo: caught, with the
   **day** on which the signal changes. (A first draft of the idea — checking
   only the operators — would have let through `center`, which is causal at
   each step and non-causal as a whole; that is why the test is about the result,
   not about the grammar.)
2. **Deflated Sharpe** (Bailey & López de Prado 2014): the
   probability that the true Sharpe exceeds the expected maximum of N
   trials without skill, corrected for skewness and kurtosis; N includes
   the trials declared before the file (`trials =`).
3. **PBO by CSCV** (Bailey, Borwein, López de Prado & Zhu 2017): the
   12,870 halves of 16 blocks; how often the in-sample champion
   falls below the median out of sample.
4. **White's Reality Check** with the stationary bootstrap of
   Politis–Romano.

Measured (§5i): the best of 30 moving-average crossovers on a random
walk has a positive Sharpe and **DSR 0.10** — failed; the planted signal
(momentum on AR(1) with φ = 0.15) passes the four gates (DSR 1.0; PBO <
0.5; RC p < 0.05); `sign(lead(close) − close)` has an annual Sharpe of 20 and is
caught on day 89.

## 8. Arbitrage decided exactly

The fundamental theorem of asset pricing is a theorem of the alternative —
Farkas's lemma: **either** a portfolio costs nothing (or less) and pays something
without ever losing, **or** there exist strictly positive state prices
that reproduce every quote within its bid–ask. Never both, never
neither. `Vapor.Finance.Arbitrage` finds both sides with the **exact
rational simplex** of `Vapor.Logic.LP` (two phases, Bland's rule) and each
answer comes with the object that proves it, checked by multiplication alone:

- an arbitrage: the portfolio (buy at the ask, sell at the bid), the cost and the
  *payoff* in each state — in the mispriced-call example, selling a
  bond, buying 1/90 of the stock and selling 1/60 of the call receives 1/15 today and
  pays 0 in both states;
- no arbitrage: the vector ψ > 0 with bid ≤ Σ Xψ ≤ ask — (1/2, 9/20) in the
  example's binomial.

For calls across strikes, the states are 0, each strike, 2·K_max and the
**slope** beyond it: since the *payoff* of any combination is piecewise
linear with vertices at the strikes, checking the vertices and the slope
checks every terminal price — the decision is complete for static
portfolios. The factor e^(−rT) is irrational and enters rounded to 15 decimal places (the
answer says so). FX: all simple cycles of up to five conversions,
the product of the rates in rationals.

**Human or AI in the loop**, as in logic: `Arbitrage.check/2` and the
MCP tool `arbitrage_check` take anyone's *proposal* — a
portfolio someone says is an arbitrage, or state prices someone says
prove there is none — and only exact arithmetic decides. The test's control:
"buy the stock and sell the call" pays in both states but costs 86.5 — it is not
an arbitrage, and the desk says why.

## 9. The order book as a verifiable object

`Vapor.Finance.Book`: price–time priority; limit and
market orders; GTC, IOC, FOK; post-only; cancellation; amendment (reducing the
quantity at the same price keeps priority; anything else is
cancel-and-replace); self-trade prevention (`:cancel_taker`,
`:cancel_resting`, `:off`); *kill switch* per owner. Prices in integer *ticks*,
integer quantities: nothing rounds. The engine is a pure
function (book, event) → (book, reports).

- **Chained journal**: hᵢ = SHA-256(hᵢ₋₁ ‖ canonical(event, reports));
  the session closes with a **Merkle root** (RFC 6962 leaves), and
  `prove/2` gives the inclusion proof of a trade without revealing the others —
  what a regulator or a client asks for without access to the whole book.
- **The independent judge** (`Vapor.Finance.Book.Check`) shares
  nothing with the engine beyond the specification: the book is a list, the
  priority is a sort by (price, time), each match scans the
  list. From the journal alone it redoes the hash chain, **re-executes each
  event on the naive engine and requires the same reports**, and checks the
  invariants: trade price = *maker* price and within the
  aggressor's limit; the *maker* is first in priority at that instant;
  quantities conserved (executed + resting + cancelled = ordered);
  book never crossed; FOK all-or-nothing; IOC and market orders never rest;
  post-only never aggresses; no trade between orders of the same owner under
  prevention.
- **Differential fuzzing**: 6,000 random events per policy,
  identical engines. In the first round the judge flagged `:fok_partial`:
  a **market FOK** order became IOC in both engines (the same
  misreading of the specification written twice) — only the invariant,
  which executes nothing, saw it. Fixed in both; the invariant stayed.
- A tampered journal is caught: a swapped price breaks the chain; the same
  forged trade **with the hash recomputed** (a coherent lie) passes
  the chain and falls at re-execution.

Measured: ~8.5 µs per event at the median (p99 ~100 µs) **including** the
SHA-256 and the canonical encoding of each entry; the judge re-executes
12,500 events in ~150 ms. The BEAM is soft real-time: these numbers
serve a desk's *matching*, an internaliser, a backtest
simulator — not a nanosecond exchange (see §14).

## 10. The market protocols: ITCH 5.0 and FIX 4.4

`Vapor.Finance.Itch`: encoder and decoder of the messages S, R, A,
F, E, C, X, D, U, P with the specification's lengths (12, 39, 36, 40,
31, 36, 23, 19, 35, 44 bytes; *big-endian*; price with 4 implied decimal places;
6-byte *timestamp* in ns), BinaryFILE framing, and
`from_session/2`: the feed an exchange would publish from the session (A when an
order rests, E when it is hit, X/D when it shrinks or leaves).
`rebuild/1` rebuilds the book **from the feed alone** — and it must be
equal to the engine's book, level by level, and the executed volume too: a
second independent certificate (the feed only sees what rests; the engine sees
everything).

`Vapor.Finance.Fix`: tag=value with BodyLength (9) and CheckSum (10)
checked on reading and computed on writing — **identical to `simplefix`**
byte for byte; NewOrderSingle (D), OrderCancelRequest (F) and
OrderCancelReplaceRequest (G) become book events; the reports become
ExecutionReports (8) with their own MsgSeqNum and ExecID.

## 11. Pre-trade risk

`Vapor.Finance.PreTrade` — what SEC Rule 15c3-5 and MiFID II RTS 6
require of anyone with market access, *and that they can show they
had*: maximum quantity (fat finger), maximum notional, price collar
against the reference (last trade, otherwise the mid), maximum position in the worst
case (executed + open orders on that side + this one), message rate
in a window, *kill switch*. A rejection never reaches the book and **goes into the
journal** with its cause. `check/2` rebuilds positions, open orders,
references and counts from the journal and checks that no accepted order
violated a limit **and that every rejection had its cause stated** — the
control in §5i attributes the cause "price collar" to a rejection for quantity,
and is caught.

## 12. Microstructure, each model with the measure that says whether it fits

`Vapor.Finance.Micro`:

- **Hawkes** (self-exciting arrivals, exponential kernel): simulation by
  Ogata's *thinning*, maximum likelihood by the O(n) recursion and the
  **time-rescaling test** — under the right model, the increments of the
  compensator are Exp(1), judged by Kolmogorov–Smirnov. Planted
  (μ, α, β) = (1, 0.6, 1.5), fitted (1.08, 0.61, 1.69), branching
  ratio 0.36 (planted 0.40), KS p = 0.73; the **control** — a
  Poisson fitted to the same arrivals — has p = 1.7·10⁻²⁰.
- **Avellaneda–Stoikov**: reservation price r = s − qγσ²(T − t), spread
  γσ²(T − t) + (2/γ)ln(1 + γ/k), simulated as in §4 of the paper (s₀ = 100,
  T = 1, σ = 2, dt = 0.005, k = 1.5, A = 140, 1,000 paths) against the
  symmetric quote with the same spread. Here / paper (γ = 0.1): σ(P&L)
  6.44 / 5.89 and 13.57 / 13.43; σ(q) 2.97 / 2.80 and 8.37 / 8.66; mean P&L
  65.0 / 62.9 and 68.0 / 67.2. The mean spread here (1.49) includes the term
  γσ²(T − t) that the paper's table seems to omit (1.29 = only the second
  term) — stated, not hidden.
- **Almgren–Chriss**: the closed-form trajectory xⱼ = X·sinh(κ(T − tⱼ))/sinh(κT)
  with κ from the discrete relation cosh(κτ) = 1 + κ̃²τ²/2, **certified**
  by solving the same mean-variance problem numerically (a tridiagonal
  system): relative difference 10⁻¹⁶. Half-life 1.65 days with the
  parameters of the paper's example; exact TWAP with λ = 0; the efficient
  frontier.
- On the tape: Roll's spread, Kyle's λ (with t-statistic), the
  realised-variance signature plot, Stoikov's *microprice*.

## 13. An exchange session that audits itself

`Vapor.Finance.Exchange`: market makers quoting *post-only* around
a reservation price skewed by inventory (the
Avellaneda–Stoikov rule) over the public price of the previous step; noise
aggressors whose arrivals follow an **exact continuous-time Hawkes**; an
informed trader who knows the fundamental and aggresses when the book drifts
away from it. Every order goes through the same `PreTrade` and the same `Book`
that a deployment would use — **the backtest is the exchange's code**, not a
model of it.

The session returns its own audit: the journal checked by the naive
engine, the limits checked from the journal, the ITCH feed
rebuilding the same book, the same seed reproducing **the same hash
head** (another seed, another head); and the microstructure measured on the
tape — the Hawkes fit of the arrivals **finds the planted self-excitation**
(branching 0.37 against 0.40; the Poisson rejected), Roll's spread
(5.66) against the quoted one (5.85), Kyle's λ, the variance signature.

## 14. What this document does not claim

- **Exchange-grade HFT latency.** The engine runs on the BEAM: microseconds per
  event with hashing, not nanoseconds. What is claimed is
  **verifiability** (journal, independent judge, feed, inclusion
  proof) and the identity between backtest and engine. The low-latency
  path — the engine as a compiled vapor program or in Zig in the worker
  — is in the [TODO](TODO.md).
- **Real market data.** No Nasdaq ITCH file, no ANBIMA
  curve was downloaded (no network for that on this machine); the
  prices in the examples are illustrative. The format is the specification's and
  the conventions are the published ones; the oracles are QuantLib, simplefix,
  SciPy and ngspice, not the data.
- **Investment advice.** The gates say when a backtest is
  indistinguishable from noise; passing them does not make a strategy
  profitable (real costs, capacity, regime).
- **Complete market models.** No XVA, no multi-factor interest-rate
  models (Hull–White, LMM), no local/stochastic volatility
  calibrated to a whole surface, no credit.

## 15. How to contest

```sh
mix test test/vapor/finance_test.exs test/vapor/console_markets_test.exs   # QuantLib, SciPy, simplefix when present
mix vapor.quality --only round13                                            # §5i: 23 checks with a control
mix vapor.finance backtest my_strategy.txt                                  # the four gates on your strategy
mix vapor.finance book my_orders.txt                                        # journal, naive judge, ITCH, FIX
node test/js/console_markets.mjs http://127.0.0.1:8000/ /tmp/capturas       # with `mix vapor.serve`
```
