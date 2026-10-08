# Cupel — silent corruption, caught

> Since 0.15.0. Code: `lib/vapor/cupel.ex`, `lib/vapor/cupel/sentinel.ex`.
> Tests: `test/vapor/cupel_test.exs`. Console: *Opus → Cupel*. Terminal: `vapor cupel`.
> MCP: `cupel_drill`.

The cupel is the assayer's porous dish: the base metal is absorbed, the noble metal stays.

## The pain

At fleet scale, a defective core returns **wrong numbers with no error at all** (Meta's and
Google's reports of *silent data corruption at scale*: about one machine in a thousand). Training
absorbs it as a loss spike days later; inference serves it. Replicating each product and
comparing bit for bit (what `Vapor.Cluster` already does) costs 2×. A check has to be **cheaper
than the product** and **never accuse a correct substrate**.

## The first principle

For `y = x·Wᵀ` and any vector `r`, `y·r = x·(Wᵀr)` — the adjoint identity
⟨Wx, r⟩ = ⟨x, Wᵀr⟩. `Wᵀr` depends only on the weights: computed once per matrix (the **probe**), it makes
each check `O(b·(n + k))` against the product's `O(b·n·k)` (Freivalds, 1977). Two choices
make this a verdict and not a heuristic:

- **exact comparison** — `ŷ·r` and `x·(Wᵀr)` in dyadic rationals (the cells of the
  [Amalgam](AMALGAM.md)): the only slack is the substrate's own rounding;
- **proved tolerance, not tuned** — Higham's lemma 3.1 (in `proofs/Vapor/Higham.lean`):
  any conforming substrate, in any summation order, with or without FMA, satisfies
  `|ŷᵢⱼ − yᵢⱼ| ≤ γₖ·Σₗ|xᵢₗ||Wⱼₗ| + 2k·η` (η = 2⁻¹²⁶ covers *flush-to-zero*), plus the operands
  that a DAZ substrate may read as zero. Projected onto `|r|`, the bound costs the same
  `O(b·(n + k))`. A row whose discrepancy exceeds the bound **cannot** have come from a
  correct substrate.

`r` has integer entries `±[1, 2²⁰]` drawn from a seed: a corrupted element is caught
whenever `|δ|·|rⱼ| > 2·tol`; several elements conspiring cancel out with probability ≈ 2⁻²⁰
per row, and not against someone who does not know the seed. For `s8` weights and activations (the int8 GEMM,
`s32` results) the check is exact: zero tolerance, all 32 bits caught.

Three verdicts per row: `:ok`, `:corrupt`, `:unchecked` (non-finite inputs, or outputs that
can legitimately overflow — never accused, never called "ok"). A non-finite output from
finite inputs is corruption.

## The sentinel

`Vapor.Cupel.Sentinel` guards a set of substrates that compute `x·Wᵀ` for a matrix:
each result goes through the cupel before being returned; a substrate whose result cannot have
come from correct arithmetic is put in **quarantine** with the evidence, the product is redone on the
next healthy one (or by the exact oracle), and the event enters a journal chained by SHA-256 and
closed by a Merkle root — the record an operator takes to the manufacturer. The work runs
in the caller's process (concurrent calls); the server only keeps the probe, the health and the
journal. A worker that dies is a journal entry, not a crash.

## Measured

| | |
|---|---|
| 256 × 256, batch 8 | probe 82 ms (once); product by the oracle 142 ms; check 3.2 ms |
| one bit flipped in an output (32 × 64, 12 batches) | bits 19–31: 100%; bit 0: 0% — below the envelope, and stated |
| 4 conforming summation orders × 6 scales (10⁻³⁰…10¹⁵) | 0 accusations in 24 |
| the forger who knows `r` | passes; the same forgery under another seed: caught |
| 20 concurrent calls, a core that flips an exponent bit | 20/20 correct answers, the core in quarantine |

The full per-bit profile (`sensitivity/3`, `vapor cupel`): sign, exponent and high mantissa
always caught; the low mantissa bits of large outputs are indistinguishable from rounding.

## Found along the way

The `dot16` oracle **silently truncated** contractions with `k` not a multiple of 16 (the tail was
ignored). It now refuses with an error; the test that found it is in `cupel_test.exs`.

## What it is not

Silicon does not become defective on request: the faults are injected at the boundary where a defective
core would emit them (the `runner` of the tests and of the drill). The check covers the
linear layer; the non-linearities between layers are verified by vapor's own ladder
(per-operator envelopes), not by this identity.
