# Amalgam — sums that do not remember the order

> Since 0.15.0. Code: `lib/vapor/amalgam.ex`, `lib/vapor/train/lm.ex` (`reduce: :exact`).
> Tests: `test/vapor/amalgam_test.exs`, `test/vapor/train_exact_test.exs`. Console: *Opus → Amalgam*.
> Terminal: `echo "1e16 1 -1e16" | vapor amalgam -` (`--f32` for binary32). MCP: `amalgam_sum`.

## The pain

Floating-point addition is commutative and **not associative**. A distributed sum — the
gradient *all-reduce*, the joining of partial results from nodes that come and go — has bits
that depend on *who added what first*. The canonical policy up to 0.14 answered
by **fixing the shape** of the reduction (16 lanes, a fixed binary tree over the
micro-batch indices, a power-of-two count). Correct, but a constraint on everything around it: the
number of micro-batches, the scheduling, swapping a node on a failure, and the row split that
`Vapor.Shard` had to refuse.

## The first principle

Every value of a binary format is an integer multiple of its smallest subnormal, `2^qmin`. So the
sum of `n` values is an integer times `2^qmin`, and integer addition **is** associative. The
BEAM's arbitrary-precision integers make that integer a Kulisch accumulator with no limbs to
manage:

- a **cell** is `Σ mᵢ·2^(eᵢ − qmin)`, exact;
- `merge/2` adds cells — a commutative monoid with `:empty` as the identity;
- `round/1` rounds **once**, to nearest even, with gradual *underflow* and overflow to ±∞.

The result is the correctly rounded value of the true real sum — a definition that
names no order at all, so it is also the canonical one. Special values follow IEEE 754 §6.3 in
any order: NaN, or `+∞` with `−∞`, gives NaN; an infinity dominates the finite values; a sum of only `−0` is
`−0`, any cancellation gives `+0`. Formats: f16, bf16, f32, f64.

`dot/3` and `partial_dot/3` do the same with products (the product of two binary values is exact in a
larger integer): slices of a contraction, joined in any order, round to the unsliced dot
product — the row split made exact. `mean/2` is the correctly rounded
quotient (checked against both neighbors, in exact arithmetic). `to_wire/1` and `from_wire/1`
carry an amalgam over the network; reading refuses cells beyond `count × largest finite`.

## In training

`LM.start(…, reduce: :exact)` replaces the fixed tree with the exact mean: the step's gradient is the
correctly rounded mean of the exact sum of the micro-batches' gradients. The bits come to
depend **only on the set** of micro-batches — not on how many workers, on who computes what,
on the order of arrival, on a crash midway, nor on the count being a power of two. Measured:
three micro-batches with 1, 2 or 3 workers, arbitrary assignments, a worker killed
midway and the oracle — a single *digest*; the *checkpoint* and the resumption continue with the bits of the
uninterrupted run. It is **a different definition** from the tree (the *digests* differ — the test can fail), and
both learn (parameters within < 10⁻³ of each other).

## Measured

| | |
|---|---|
| 64 vectors × 4,096 f32 | 42 ms on the BEAM (≈ 6 M exact additions/s); rounding 4,096 cells: 6 ms |
| 2,048 values with cancellation, 30 orders and groupings | 1 result, equal to the exact sum rounded once; the left-to-right sum gave 30 different results |
| f64, `[1e16, 1, −1e16]` | `1.0` (the naive sum gives `0.0`) |

## What it is not

It is not free: one *bignum* addition per element (≈ 0.1–0.2 µs). It is a reduction for the control
plane — gradients across micro-batches and nodes, partial results in a *cluster* — and not for
the inner loop of a *kernel*, whose exact form calls for integer operations in the five emitters
([TODO](TODO.md)). Finite terms whose true sum is finite never overflow (left-to-right IEEE
can overflow along the way) — it is a difference in semantics, stated.
