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
| `layers` (0.17) | labels, features per layer | which layer a linear probe should read: chosen on validation, reported on test against the output (exact McNemar) and chance — the Perception Encoder's finding as a protocol | a planted network folded at its output is read in the middle (31 of 32 discordant items, p = 1.5·10⁻⁸); shuffled labels find nothing |
| `detect` (0.17) | truth and predicted boxes per image | SAM 3's **cgF1** = 100 · pmF1 · IL_MCC: localisation by an **optimal matching, exact and certified** (rational simplex on a totally unimodular program), presence on images with and without the object, a bootstrap CI | a greedy matching loses a true positive the optimal one keeps; predictions shuffled across images fall to chance |
| `geometry` (0.16) | model, item, p₁…p_k | mean Fisher–Rao distance between each pair of models with a *bootstrap* CI; each one's distance to the geometric consensus (Fréchet mean) — [GEOMETRY.md](GEOMETRY.md) | the same models with the items shuffled; the triangle inequality on every triple |

The scaling-law control found a real defect in this round: on a fit with no structure the
log parameters climbed until `exp` overflowed. Now everything stays finite and the *holdout* says
the law predicts nothing — which is the right answer.

`detect` makes two choices SAM 3 leaves open, and reports them with every answer: a truth matched
below τ counts as a false negative (otherwise F1 is inflated), and IL_MCC is 0 when its
denominator is 0. Its matching maximises total IoU exactly. COCO-style greedy matching, best pair
first, can lose a true positive that the optimal matching keeps, and the test shows such a case.

```
vapor assay compare results.csv             # exits 0 if the difference is real, 1 if it is noise
vapor assay scaling runs.csv --json | jq .holdout
vapor assay dedup corpus.jsonl --keep > clean.jsonl
vapor assay contamination train.txt test.txt
vapor assay layers features.json            # which layer to probe, and whether it beats the output
vapor assay detect detections.jsonl         # cgF1 with its interval
```
