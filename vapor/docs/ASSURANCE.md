# Garantias — o livro-razão

> Gerado de `lib/vapor/assurance.ex` por `mix vapor.assurance`; um teste falha se a evidência citada sumir ou se
> este arquivo divergir dos dados. A disciplina é a do ASAS (§11): cada afirmação diz se é **provada** (Lean 4,
> sem axioma), **conferida** (um verificador independente roda a cada resposta ou a cada build), testada,
> argumentada ou *devida*.

**provado**: 5 · **conferido**: 7 · testado: 8 · argumentado: 4 · *devido*: 3

## O núcleo

| afirmação | base | estado | evidência | limites |
|---|---|---|---|---|
| Every substrate produces the same bits under the canonical policy | differential suites against the exact oracle on x86 AVX2/AVX-512, AArch64 and RVV (emulated), SPIR-V (lavapipe) | testado | `test/vapor/substrate_test.exs` · `test/vapor/fabric_test.exs` · `test/vapor/canonical_test.exs` | RVV and NEON run under emulation here; no discrete GPU |
| Higham's bound on any summation order (γₖ envelope) | Lean 4, no axiom | **provado** | `proofs/Vapor/Higham.lean` | — |
| Wilkinson's bound for the canonical reductions | Lean 4, no axiom | **provado** | `proofs/Vapor/Wilkinson.lean` | — |
| Rewrite rules preserve IEEE-754 binary32 semantics (finite values) | Lean 4 model of binary32 bits; NaN excluded by principle | **provado** | `proofs/Vapor/Binary32.lean` · `test/vapor/rewrite_soundness_test.exs` | constant folding is tested against the oracle, not proved |
| Register allocation is checked, not trusted | a checker proved in Lean, extracted to Elixir; extraction freshness audited | **provado** | `proofs/Vapor/RegAlloc.lean` · `test/vapor/regalloc_test.exs` · `test/vapor/extracted_conformance_test.exs` | — |
| Generated machine code never runs inside the BEAM (only isolated worker processes); no shell; no dependencies | source audit on every build | **conferido** | `test/vapor/audit_test.exs` | — |
| One entropy boundary: OS randomness only through Vapor.Entropy, process generators seeded | source audit on every build | **conferido** | `lib/vapor/entropy.ex` · `test/vapor/audit_test.exs` | — |

## Rodada 0.15

| afirmação | base | estado | evidência | limites |
|---|---|---|---|---|
| Amalgam: the correctly rounded exact sum, whatever the order or grouping | exact integer cells; rounding compared with exact rationals | testado | `test/vapor/amalgam_test.exs` · `test/vapor/train_exact_test.exs` | — |
| Cupel never accuses a conforming substrate | Higham's bound (proved) projected on |r|; four summation orders at six scales | **provado** | `proofs/Vapor/Higham.lean` · `test/vapor/cupel_test.exs` | detection below the rounding envelope is impossible and measured, not claimed |
| Cupel catches a corrupted element above the envelope | probability ≥ 1 − 2⁻²⁰ per row for a forger without the seed | argumentado | `lib/vapor/cupel.ex` · `test/vapor/cupel_test.exs` | — |
| Rebis 'equivalent' by SAT carries a DRUP proof checked by separate code | Vapor.Logic.DRUP on every UNSAT answer | **conferido** | `lib/vapor/logic/drup.ex` · `test/vapor/rebis_test.exs` | — |
| Rebis word identities: remainder 0 ⇔ the identity holds | the gate polynomials form a Gröbner basis under reverse-topological lex order (Lv–Kalla–Enescu) | argumentado | `lib/vapor/rebis/ideal.ex` · `test/vapor/rebis_test.exs` | the substitution trace is not independently re-checked |
| Aludel 'certified' is replayed by direct per-leaf conversion | Aludel.check/4 on every answer; Bernstein enclosure is a classical theorem | **conferido** | `lib/vapor/aludel.ex` · `test/vapor/aludel_test.exs` | — |
| Tabula consistency proofs are DRUP-checked | Vapor.Logic.DRUP on every pair proved apart | **conferido** | `lib/vapor/tabula.ex` · `test/vapor/tabula_test.exs` | — |
| JBIG2 decoding equals jbig2dec bit for bit (or T.88 where jbig2dec departs) | 64 fixtures from an independent encoder | testado | `test/vapor/jbig2_test.exs` · `test/python/jbig2_streams.py` | — |

## Rodada 0.16

| afirmação | base | estado | evidência | limites |
|---|---|---|---|---|
| The khazāna opens at the old or the new root after a crash at any byte | fault injection at every byte of the pack append and of the root slot | testado | `lib/vapor/khazana.ex` · `test/vapor/khazana_test.exs` | assumes datasync is honest and a SHA-256 tag cannot collide (the platform's half of the contract) |
| A conversation message commits to its whole history | its hash covers its parent's hash (SHA-256) | argumentado | `lib/vapor/majlis.ex` · `test/vapor/majlis_test.exs` | — |
| An altered vapor export is refused on import | every message hash recomputed | **conferido** | `lib/vapor/majlis/exchange.ex` · `test/vapor/majlis_test.exs` | — |
| Shared links cannot be forged and die on revocation | HMAC-SHA256 under the store's key; generation counter | argumentado | `lib/vapor/khazana.ex` · `test/vapor/hall_test.exs` | reduces to the PRF security of HMAC-SHA256 |
| The console terminal cannot read or write the server's files or run programs | jailed reads and writes, --measure refused, heap ceiling and deadline per command | testado | `lib/vapor/diwan.ex` · `test/vapor/diwan_test.exs` · `test/vapor/hall_test.exs` | rests on the BEAM's process isolation and on every verb reading through Vapor.Main.read_input |
| Al-Mizān: the Latin and Arabic projections are bijective with the tree | read(print(t)) = t on 300 random programs in both scripts | testado | `lib/vapor/mizan/syntax.ex` · `test/vapor/mizan_test.exs` | — |
| Al-Mizān burhān: identities and conservation laws decided exactly | exact polynomial normal form over ℚ; positivity by Aludel; invariants by SAT + DRUP | **conferido** | `lib/vapor/mizan.ex` · `test/vapor/mizan_test.exs` | — |
| Al-Mizān obligations exported to Lean 4 close by ring / decide | Lower.lean/2 emits the theorems | *devido* | `lib/vapor/mizan/lower.ex` | Lean is not installed on this machine: the export is generated, not checked here |
| Fisher–Rao distances are a metric; natural gradient is invariant to feature scaling | property tests against controls (KL; plain gradient) | testado | `lib/vapor/info_geom.ex` · `test/vapor/info_geom_test.exs` | — |
| The language server answers editors over real stdio framing | a Node client against bin/vapor lsp | testado | `lib/vapor/lsp.ex` · `test/vapor/lsp_test.exs` · `test/js/lsp_client.mjs` | VS Code, Neovim and Emacs themselves are not run here |

## O que é devido

| afirmação | base | estado | evidência | limites |
|---|---|---|---|---|
| Weak-memory correctness of the worker's shared-memory protocol | litmus tests against a RVWMO/TSO model | *devido* | `native/src` | the worker uses pipes and /dev/shm with explicit synchronisation; no axiomatic-model check exists yet |
| Constant folding proved, not only tested | extend Binary32.lean | *devido* | `proofs/Vapor/Binary32.lean` | — |
