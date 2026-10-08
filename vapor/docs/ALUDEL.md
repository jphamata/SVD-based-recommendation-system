# Aludel — claims about polynomials, decided in integers

> Since 0.15.0. Code: `lib/vapor/aludel.ex`. Tests: `test/vapor/aludel_test.exs`.
> Console: *Opus → Aludel*. Terminal: `vapor aludel decide 'x^2 - x + 1/4' --vars x --box '0,1'`.
> MCP: `aludel_decide`. Absorbed from the core of PALADIN (the round's attachment), rewritten over the BEAM's integers.

The aludel is the vessel in which a volatile substance is *fixed*.

## The pain

Every "provably safe controller" claim runs into the same point: a check by
samples says nothing between the samples, and floating point says nothing near zero. The
question "`p(x) ≥ 0` for every `x` in this box?" needs an exact, checkable answer.

## The procedure

- **Bernstein enclosure.** On a box, `p` is a combination of the Bernstein polynomials, which are
  nonnegative and sum to one; so `p` lies between the smallest and the largest Bernstein coefficient, and the
  coefficients at the vertices **are** `p` there. An exact conversion from the power basis (rationals,
  then integers over a common denominator; per axis, a Taylor shift and the
  Bernstein transform, over a dense array).
- **Three verdicts, never two.** All coefficients `≥ 0` (or `> 0` for the strict claim):
  **certified** on the cell. A vertex coefficient `< 0`: **refuted**, with the vertex — an exact
  point and the value. Otherwise the cell is split at the middle of its widest axis by de
  Casteljau's algorithm (only integer additions and shifts). A cell that reaches the budget is
  **exhausted**: a named verdict, with the cell, never a longer wait and never a guess.
- **The witness is the subdivision tree** (one bit per cell, depth-first). `check/4`
  replays it without searching and, at each leaf, recomputes the Bernstein coefficients **directly** from the
  polynomial on that leaf's box — a different computation from the search's subdivisions, with the same
  theorem behind it. Witnesses do not transfer: tampered bits, another polynomial, truncation
  — refused.

`enclose/3` gives a rigorous interval for `p` on the box (it contains the exact values, checked at
random rational points).

## Barrier certificates

For a polynomial field `ẋ = f(x)`, a function `B` with `B ≤ 0` on the initial set, `B > 0` on the
unsafe set and `λB − ∇B·f ≥ 0` on the domain proves that no trajectory from the initial set reaches the unsafe set
without first leaving the domain (Prajna & Jadbabaie, 2004). Each condition is a
positivity claim on a box — the same procedure. The candidate comes from the person, from a model or from
`synthesize/2`: an exact LP (`Vapor.Logic.LP`) over the coefficients of `B` whose rows are
Bernstein coefficients, grown by **constraint generation** (only the violated rows enter).
In every case the candidate is only accepted by the same decision. When the unsafe set meets the
initial set, no barrier exists, and the LP says so with an infeasibility certificate.

## Measured

| | |
|---|---|
| Motzkin + 1/1000 > 0 on [−2, 2]² (nonnegative, not a sum of squares) | certified in 159 cells; witness replayed |
| Motzkin ≥ 0 (touches zero at (±1, ±1)) | **exhausted** — said, not faked |
| 80 random polynomials | every "certified" holds at 60 random exact points; every "refuted" point is negative |
| damped oscillator, `B = x² + y² − 1` | three conditions certified; the unstable field `ẋ = x, ẏ = y`: refuted at an exact point |
| synthesis, nonlinear field `ẋ = −x + y², ẏ = −y` | barrier found by the LP and accepted by the decision |

## What it is not

It is not a model of an aircraft, a plasma or a cortex. It proves properties **of the polynomial
system it receives**; whether that system describes the world is another claim, and it is stated as such.
The cost of the conversion is exponential in the number of variables (the degree per axis enters as a product),
as in PALADIN: its place is systems with few variables.
