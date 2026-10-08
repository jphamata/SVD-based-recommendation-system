# Assurance — the ledger

> Generated from `lib/vapor/assurance.ex` by `mix vapor.assurance`; a test fails if cited evidence disappears or if
> this file drifts from the data. The discipline is the ASAS's (§11): each claim says whether it is **proved** (Lean 4,
> no axiom), **checked** (an independent checker runs on every answer or every build), tested,
> argued or *owed*.

**proved**: 5 · **checked**: 11 · tested: 15 · argued: 4 · *owed*: 4

## The core

| claim | basis | status | evidence | limits |
|---|---|---|---|---|
| Every substrate produces the same bits under the canonical policy | differential suites against the exact oracle on x86 AVX2/AVX-512, AArch64 and RVV (emulated), SPIR-V (lavapipe) | tested | `test/vapor/substrate_test.exs` · `test/vapor/fabric_test.exs` · `test/vapor/canonical_test.exs` | RVV and NEON run under emulation here; no discrete GPU |
| Higham's bound on any summation order (γₖ envelope) | Lean 4, no axiom | **proved** | `proofs/Vapor/Higham.lean` | — |
| Wilkinson's bound for the canonical reductions | Lean 4, no axiom | **proved** | `proofs/Vapor/Wilkinson.lean` | — |
| Rewrite rules preserve IEEE-754 binary32 semantics (finite values) | Lean 4 model of binary32 bits; NaN excluded by principle | **proved** | `proofs/Vapor/Binary32.lean` · `test/vapor/rewrite_soundness_test.exs` | constant folding is tested against the oracle, not proved |
| Register allocation is checked, not trusted | a checker proved in Lean, extracted to Elixir; extraction freshness audited | **proved** | `proofs/Vapor/RegAlloc.lean` · `test/vapor/regalloc_test.exs` · `test/vapor/extracted_conformance_test.exs` | — |
| Generated machine code never runs inside the BEAM (only isolated worker processes); no shell; no dependencies | source audit on every build | **checked** | `test/vapor/audit_test.exs` | — |
| One entropy boundary: OS randomness only through Vapor.Entropy, process generators seeded | source audit on every build | **checked** | `lib/vapor/entropy.ex` · `test/vapor/audit_test.exs` | — |

## Round 0.15

| claim | basis | status | evidence | limits |
|---|---|---|---|---|
| Amalgam: the correctly rounded exact sum, whatever the order or grouping | exact integer cells; rounding compared with exact rationals | tested | `test/vapor/amalgam_test.exs` · `test/vapor/train_exact_test.exs` | — |
| Cupel never accuses a conforming substrate | Higham's bound (proved) projected on |r|; four summation orders at six scales | **proved** | `proofs/Vapor/Higham.lean` · `test/vapor/cupel_test.exs` | detection below the rounding envelope is impossible and measured, not claimed |
| Cupel catches a corrupted element above the envelope | probability ≥ 1 − 2⁻²⁰ per row for a forger without the seed | argued | `lib/vapor/cupel.ex` · `test/vapor/cupel_test.exs` | — |
| Rebis 'equivalent' by SAT carries a DRUP proof checked by separate code | Vapor.Logic.DRUP on every UNSAT answer | **checked** | `lib/vapor/logic/drup.ex` · `test/vapor/rebis_test.exs` | — |
| Rebis word identities: remainder 0 ⇔ the identity holds | the gate polynomials form a Gröbner basis under reverse-topological lex order (Lv–Kalla–Enescu) | argued | `lib/vapor/rebis/ideal.ex` · `test/vapor/rebis_test.exs` | the substitution trace is not independently re-checked |
| Aludel 'certified' is replayed by direct per-leaf conversion | Aludel.check/4 on every answer; Bernstein enclosure is a classical theorem | **checked** | `lib/vapor/aludel.ex` · `test/vapor/aludel_test.exs` | — |
| Tabula consistency proofs are DRUP-checked | Vapor.Logic.DRUP on every pair proved apart | **checked** | `lib/vapor/tabula.ex` · `test/vapor/tabula_test.exs` | — |
| JBIG2 decoding equals jbig2dec bit for bit (or T.88 where jbig2dec departs) | 64 fixtures from an independent encoder | tested | `test/vapor/jbig2_test.exs` · `test/python/jbig2_streams.py` | — |

## Round 0.16

| claim | basis | status | evidence | limits |
|---|---|---|---|---|
| The khazāna opens at the old or the new root after a crash at any byte | fault injection at every byte of the pack append and of the root slot | tested | `lib/vapor/khazana.ex` · `test/vapor/khazana_test.exs` | assumes datasync is honest and a SHA-256 tag cannot collide (the platform's half of the contract) |
| A conversation message commits to its whole history | its hash covers its parent's hash (SHA-256) | argued | `lib/vapor/majlis.ex` · `test/vapor/majlis_test.exs` | — |
| An altered vapor export is refused on import | every message hash recomputed | **checked** | `lib/vapor/majlis/exchange.ex` · `test/vapor/majlis_test.exs` | — |
| Shared links cannot be forged and die on revocation | HMAC-SHA256 under the store's key; generation counter | argued | `lib/vapor/khazana.ex` · `test/vapor/hall_test.exs` | reduces to the PRF security of HMAC-SHA256 |
| The console terminal cannot read or write the server's files or run programs | jailed reads and writes, --measure refused, heap ceiling and deadline per command | tested | `lib/vapor/diwan.ex` · `test/vapor/diwan_test.exs` · `test/vapor/hall_test.exs` | rests on the BEAM's process isolation and on every verb reading through Vapor.Main.read_input |
| Almizan: the Latin and Arabic projections are bijective with the tree | read(print(t)) = t on 300 random programs in both scripts | tested | `lib/vapor/almizan/syntax.ex` · `test/vapor/almizan_test.exs` | — |
| Almizan burhān: identities and conservation laws decided exactly | exact polynomial normal form over ℚ; positivity by Aludel; invariants by SAT + DRUP | **checked** | `lib/vapor/almizan.ex` · `test/vapor/almizan_test.exs` | — |
| Almizan obligations exported to core Lean 4 close by grind / decide; a refuted claim's statement does not | Lower.lean/2 emits theorems over Rat and Bool with no Mathlib; the :lean tier runs Lean on the example modules' exports and on a refuted one (0.17) | tested | `lib/vapor/almizan/lower.ex` · `test/vapor/almizan_test.exs` | positivity on a box has no core-Lean proof: it is exported as a statement, its certificate is Aludel's witness |
| Fisher–Rao distances are a metric; natural gradient is invariant to feature scaling | property tests against controls (KL; plain gradient) | tested | `lib/vapor/info_geom.ex` · `test/vapor/info_geom_test.exs` | — |
| The language server answers editors over real stdio framing | a Node client against bin/vapor lsp | tested | `lib/vapor/lsp.ex` · `test/vapor/lsp_test.exs` · `test/js/lsp_client.mjs` | VS Code, Neovim and Emacs themselves are not run here |

## Round 0.17

| claim | basis | status | evidence | limits |
|---|---|---|---|---|
| The Lean development builds warning-free on Lean 4.34.1, and re-extraction is byte-identical | lake build; the extracted module compared with the sources' digest | **checked** | `proofs/lean-toolchain` · `test/vapor/audit_test.exs` | — |
| The hermetic seal stops a job whose heap and off-heap binaries together exceed its cap, or that misses its deadline; the caller survives | max_heap_size with include_shared_binaries; a 512 MB binary bomb under 64 MB against the 0.16 heap-only cap | tested | `lib/vapor/hermetic.ex` · `test/vapor/hermetic_test.exs` | bounds memory and time; does not resist code with access to the BEAM itself |
| An integer program's 'optimal' or 'infeasible' carries a branch-and-bound tree checked by code that shares nothing with the search | MIP.check/2 on every answer: the splits cover the integer points, every leaf's Farkas or dual certificate, the incumbent | **checked** | `lib/vapor/logic/mip.ex` · `test/vapor/mip_test.exs` | branching only (no cutting planes); a relaxation unbounded at the root is refused |
| Causal estimands equal the true intervention | exact rationals on random structural causal models with explicit hidden parents; the naive P(y | x) as control | tested | `lib/vapor/logic/causal.ex` · `test/vapor/causal_test.exs` | every verdict is conditional on the stated diagram; ID's completeness is a published theorem (Shpitser & Pearl, 2006), not mechanised |
| A causal identification that fails carries a hedge, checked | Causal.hedge?/5 on every failure | **checked** | `lib/vapor/logic/causal.ex` · `test/vapor/causal_test.exs` | — |
| Qālib: a netlist and its specification are called equivalent only with a truth table or a DRUP-checked SAT proof | Vapor.Rebis.equivalent/3; mapped netlists are read back and proved before they are printed | **checked** | `lib/vapor/qalib.ex` · `test/vapor/qalib_test.exs` | combinational only; each cell's function is tested against its formula, not read from the liberty files |
| Palingenesis: a published generation passed every gate, and its lineage re-derives from launch | the gates measured before publication; verify/2 recomputes the record chain and the root from the weights | tested | `lib/vapor/palingenesis.ex` · `test/vapor/palingenesis_test.exs` | drift is measured on the anchors only; measured on a tiny model here, not on production checkpoints |
| Palingenesis: a reader keeps its generation whole while new ones are published | immutable generations published by one persistent_term put; concurrent readers re-hash what they hold | tested | `lib/vapor/palingenesis.ex` · `test/vapor/palingenesis_test.exs` | at the BEAM level; a worker's shared-memory session sees the new generation at its next session |
| Recommendations are called signal only with a significant, minimum-size gain over the biases and a shuffled control at the mean | paired sign-flip test, a 1 % minimum gain, a shuffled-ratings control, nested selection | tested | `lib/vapor/recommend.ex` · `test/vapor/recommend_test.exs` | — |
| The one-sided Jacobi SVD comes with its residual and orthogonality, and keeps small singular values to relative precision | Dense.svd_residual/2; σ_min of a κ = 10⁸ matrix against the Gram-matrix route | tested | `lib/vapor/dense.ex` · `test/vapor/recommend_test.exs` | — |

## What is owed

| claim | basis | status | evidence | limits |
|---|---|---|---|---|
| Palingenesis measured on production checkpoints (Qwen 2.5, DeepSeek-V3) | vapor palingenesis try on real models | *owed* | `lib/vapor/palingenesis.ex` | this machine has neither the weights nor the memory |
| Sequential equivalence (latches, flip-flops) for Qālib | k-induction on the miter, IC3/PDR | *owed* | `lib/vapor/qalib.ex` | — |
| Weak-memory correctness of the worker's shared-memory protocol | litmus tests against a RVWMO/TSO model | *owed* | `native/src` | the worker uses pipes and /dev/shm with explicit synchronisation; no axiomatic-model check exists yet |
| Constant folding proved, not only tested | extend Binary32.lean | *owed* | `proofs/Vapor/Binary32.lean` | — |
