# round17 — checks against their controls

Written by `mix vapor.quality --only round17 --md …` (12/12 passed in 63587 ms,
substrate: the exact oracle). Each check has a value, a control that a broken, naive or lucky
implementation would produce, and the threshold that separates them; see `Vapor.Quality.Round17`.

| check | value | control | threshold | ok |
|---|---|---|---|---|
| seal: 512 MB of off-heap binaries under a 64 MB seal | {:error, :memory} | the 0.16 heap-only cap: {:survived, 512} | stopped by the seal; the old cap lets them through | ✓ |
| integer programs: branch and bound against brute force, certificates checked | 40/40 | the relaxation rounded down: 12/40 right | 40/40, the control below | ✓ |
| integer programs: a forged optimum, and a worse incumbent with the honest tree | forged: false; worse: false | the naive check (feasible and integral) accepts the worse point: true | both refused; the naive check fooled | ✓ |
| causes: front-door and napkin estimands against the true intervention (exact rationals, 10 models) | 10/10 | the naive P(y \| x): 0/10 | 10/10; the naive estimate wrong somewhere | ✓ |
| causes: the bow is not identifiable (a checked hedge) | hedge checked: true | two models, one P(x, y): true; P(y=1 \| do(x=1)) = 1/2 and 1 | the hedge holds; the two effects differ | ✓ |
| mould: a 20-bit trojan trigger in a 21-input adder | found by SAT, re-simulated on both circuits | 4,096 random patterns: not found | found exactly; random simulation blind | ✓ |
| mould: a mapped adder read back (cells, NAND) | cells: equivalent, nand: equivalent | one pin moved: different | equivalent both ways; the moved pin told apart | ✓ |
| palingenesis: a plank that restores the ship (target gate) | admitted | a sham of the same norm: refused (target) | admitted; the sham refused at the target gate | ✓ |
| palingenesis: the brake at ε = 0.5 for the same plank | refused (drift), drift max 3.01e+00 | at ε = π: admitted | refused by the brake; admitted without it | ✓ |
| palingenesis: a block with its hidden units permuted | drift max 4.89e-07 | a random change (σ 0.05): drift max 5.94e-01 | < 10⁻³ (rounding); the random change well above | ✓ |
| recommendations: planted interactions (40 × 24, 60 % observed) | signal, RMSE 6.90e-02 vs biases 1.40e+00 | unstructured ratings: no evidence the interactions help (the biases explain what is predictable) | signal; the unstructured table not | ✓ |
| SVD: the smallest singular value of a matrix with condition 10⁸ | one-sided Jacobi: relative error 2.35e-09 | √eig(AᵀA): relative error 2.92e-02 | < 10⁻⁶; the Gram route loses it | ✓ |
