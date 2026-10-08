# Assay — the AI research workbench: signal or noise

> Since 0.14.0. Code: `lib/vapor/assay.ex`, `lib/vapor/assay/` (`stats`, `data`, `scaling`).
> Tests: `test/vapor/assay_test.exs`. Quality: `mix vapor.quality --only round14`.

The *assay* is the test that says how much gold there really is in the metal. The pain it attacks
is the most common one in AI research and industry: **an evaluation difference that is noise**, a
*leaderboard* that ranks ties, a contaminated benchmark, a scaling law extrapolated without a
check. Each tool takes a CSV (or text) and answers with the number, the interval and the
sentence that says whether it is signal.

| tool | input | answer | the control |
|---|---|---|---|
| `compare` | columns a, b per item | paired difference, *bootstrap* CI, sign-flip permutation (exact up to n = 16), exact McNemar, d_z, **minimum detectable effect** and how many items would be needed | type I error rate measured on 60 null sets: 0.00 (≤ 0.15) |
| `leaderboard` | one column per system | ranks with a rank *bootstrap*, who ties with the leader, Holm and BH | two equal systems are not separated |
| `calibration` | p, correct | equal-mass ECE, **the ECE floor of a perfectly calibrated model**, p-value, Brier, NLL, Platt fitted on one half and judged on the other | the calibrated one is not accused (p = 0.45); the overconfident one is (p = 0.002) |
| `agreement` | one column per annotator | Krippendorff's α (nominal, with missing values), Fleiss's and Cohen's κ, CI | Krippendorff (2011): 0.743; random labels: α ≈ 0 |
| `judge` | order AB and BA | position bias of an LLM judge (exact binomial) | — |
| `contamination` | train / test | 13-gram overlap per item, the clean subset | — |
| `dedup` | documents | MinHash LSH (128 hashes, 16 × 8 bands) **verified by exact Jaccard** | error of the estimate |
| `scaling` | N, D, L | L = E + A/N^α + B/D^β (Hoffmann, approach 3: Huber in log, grid + Nelder–Mead), *bootstrap* CI, **prediction of the largest without them**, optimal N\*(C) | shuffled losses: the prediction misses by 33 % and the certificate says so |
| `geometry` (0.16) | model, item, p₁…p_k | mean Fisher–Rao distance between each pair of models with a *bootstrap* CI; each one's distance to the geometric consensus (Fréchet mean) — [GEOMETRY.md](GEOMETRY.md) | the same models with the items shuffled; the triangle inequality on every triple |

The scaling-law control found a real defect in this round: on a fit with no structure the
log parameters climbed until `exp` overflowed. Now everything stays finite and the *holdout* says
the law predicts nothing — which is the right answer.

```
vapor assay compare resultados.csv          # exits 0 if the difference is real, 1 if it is noise
vapor assay scaling corridas.csv --json | jq .holdout
vapor assay dedup corpus.jsonl --keep > limpo.jsonl
vapor assay contamination treino.txt teste.txt
```
