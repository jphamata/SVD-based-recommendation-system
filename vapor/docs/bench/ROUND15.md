# round15 — checks against their controls

Written by `mix vapor.quality --only round15 --md …` (12/12 passed in 1624 ms,
substrate: the exact oracle). Each check has a value, a control that a broken, naive or lucky
implementation would produce, and the threshold that separates them; see `Vapor.Quality.Round15`.

| check | value | control | threshold | ok |
|---|---|---|---|---|
| order: 2 048 f32 values summed in 30 orders and groupings | 1 result(s), = exact sum rounded once | left-to-right f32: 30 different results | one result, equal to the exact rounding; the naive sum varies | ✓ |
| detection: a bit flip in one output of a 32×64 product (12 batches) | bits 19–31: 100 % | bit 0: 0.0 % (below the rounding envelope) | sign, exponent and high mantissa all caught; the lowest bit honestly not | ✓ |
| no false alarm: 4 summation orders × 6 scales (10⁻³⁰ … 10¹⁵) | 0 accused of 24 | forger with the seed: passes; another seed: caught | 0 accused; the seed is the secret | ✓ |
| quarantine: 20 concurrent callers, a pool with one core that flips an exponent bit | 20/20 answers correct | removed: %{"w0" => "quarantined"} | every answer correct; the bad core quarantined | ✓ |
| trojan: a 32-bit adder that misbehaves only when a = 0xDEADBEEF, b = 0 | trigger found exactly (SAT, shrunk) | 4 096 random patterns: missed (the miter was needed) | the trigger itself, after random simulation missed it | ✓ |
| algebra: a 16-bit array multiplier, m = a·b, by backward rewriting over ℤ | proved: 2748 substitutions, peak 522 terms | a wrong partial product: refuted, the point re-simulated | proved; the broken one refuted at a real point | ✓ |
| AES-GCM from GF(2⁸) and GF(2¹²⁸) arithmetic, 64 random keys, IVs, messages, AAD | 64/64 = OpenSSL | a flipped tag bit: 64/64 refused | all equal; all tampering refused | ✓ |
| stabilizers: a 400-qubit GHZ state measured qubit by qubit | 1 random, 399 determined, 1 distinct value(s) | 400 Hadamards: 400 random, 197 ones | GHZ: 1 random and all equal; Hadamards: 400 random, ones in 160–240 | ✓ |
| positivity: Motzkin's polynomial + 1/1000 > 0 on [−2, 2]² (non-negative, no sum of squares) | certified (159 cells), witness replayed | Motzkin ≥ 0: exhausted; the witness on another polynomial: refused | certified and replayed; the touching case not faked; witnesses do not transfer | ✓ |
| barrier: ẋ = y, ẏ = −x − y with B = x² + y² − 1 | proved: initial, unsafe, flow | ẋ = x, ẏ = y: refuted (flow condition refuted) | three conditions certified; the unstable field refuted at an exact point | ✓ |
| antinomy: a sale contract of six clauses over four facts | antinomies: C1 × C6 when delivered ∧ late ∧ ¬defective; C1 × C3 proved apart (DRUP) | with C4 > C5 and C6 > C1: 0 antinomies, 2 resolved | the clash found with its scenario, the safe pair proved; the overrides settle it | ✓ |
| alignment: a network averaged with a copy whose MLP units are shuffled | aligned: largest logit change 0.0 | unaligned: 2.581 | aligned = the network itself (0.0); unaligned > 10⁻³ | ✓ |
