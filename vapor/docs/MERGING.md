# Model merging (`Vapor.Merge`, `mix vapor.merge`)

Closes item 6.3 of the TODO and, in this round (0.5.0), the two limitations that
0.4.0 declared: **throughput** (≈ 0.6 M parameters/s) and **"TIES and DARE made it worse"**.

## Methods

| method | result per tensor |
|---|---|
| `linear` | `Σ wᵢθᵢ / Σ wᵢ` |
| `task_arithmetic` | `base + λ Σ wᵢ(θᵢ − base)` |
| `slerp` | spherical interpolation of two models at `t` (linear if `|cos| > 0.9995`) |
| `ties` | task vectors pruned to the `density` of largest magnitude, sign elected by `sign(Σ wᵢτᵢ)`, disjoint mean of those that agree |
| `dare_linear`, `dare_ties` | each entry of the task vector kept with probability `density`, rescaled by `1/density`; then linear or TIES |
| **`regmean`** (new) | for each matrix whose inputs were measured: **least squares** `W = (Σ Wᵢ G̃ᵢ + λW̄)(Σ G̃ᵢ + λI)⁻¹`, `G = XᵀX` of each model's input activations on its own data, `G̃ = α·G + (1 − α)·diag G`; the rest `linear` |

And two tools that change the question from "which recipe should I use?" to "what do the
weights say, and what does the measurement show?":

- **`Merge.diagnose/2`** (`mix vapor.merge --diagnose`) — from the weights alone, before
  merging: how far each model moved from the base (`relative_delta`), how much of the
  delta's energy is in the entries TIES keeps (`concentration`), how much
  the signs fight (`sign_conflict`), the cosine between the deltas, and — what
  was missing — **whether the models have a common ancestor** (`weight_cosine`: ≈ 1 for
  fine-tunes of the same base, ≈ 0 for networks trained from different
  initialisations, whose units are not aligned).
- **`Merge.select/4`** (`mix vapor.merge --try "linear;ties:density=0.2;…" --eval heldout.txt`) —
  merges each candidate, measures bits per byte on held-out text through the certified
  substrate and keeps the best; the receipt `vapor.merge.select/1` stores all
  the scores and the digest of the evaluation text. The merged weights and the scores are
  deterministic: anyone can redo the table.

## Throughput: 0.6 → ≈ 11 M parameters/s end to end

0.4.0 merged with Elixir lists. Now each tensor is cut into blocks of
262,144 entries merged across all schedulers, each block by a kernel
that matches the binaries directly (`<<x::float-32-little, …>>`), with no intermediate
list. The DARE generator (splitmix64) is a **counter**: the state
after `n` draws is `key + n·γ`, so any block starts at its own
offset — parallelism does not change a bit.

| 26 M parameters (random Qwen2, 2 vCPUs) | 0.4.0 | 0.5.0 |
|---|---|---|
| linear (merge only) | ≈ 13 s | 1.2 s |
| SLERP (merge only) | ≈ 20 s | 1.9 s |
| open A + B, merge, Merkle root, write | ≈ 45 s | 2.4 s |

Output **bit-for-bit identical** to 0.4.0's in every method (checked against the
old module, receipts included). A 7 B model goes from hours to about
10 minutes — but it needs memory for three copies in f32; merging by
*streaming* straight from disk remains on the TODO.

## Why TIES and DARE made it worse — and when they do not

0.4.0 measured on two planted bigrams (Portuguese × English) whose "task
vectors" were the difference of two whole tables: deltas the size of the
weights. That is not the regime TIES and DARE were proposed for (fine-tunes
of the same base, small deltas). What was missing was measuring in the right regime, with
**genuinely trained** models.

`test/python/train_merge_models.py` trains, with PyTorch, character-level Llama decoders
(width 64, 2 layers) on the frozen corpora of `priv/quality`: a
**base** in Portuguese + English, two **fine-tunes** of it (`ft_pt`, `ft_en`) and two
**separately trained** models (`solo_pt`, `solo_en`, other seeds). They live
in `priv/quality/merge` with SHA-256. Bits per character, candidates chosen
on a validation slice and reported on a disjoint test slice
([bench/QUALITY.md §4b](bench/QUALITY.md)):

**Fine-tunes of one base** — diagnosis `:small_deltas` (deltas of 3% of the
weights, weight cosine 0.999, 62% of the energy in the largest 20%, signs in
conflict over 39% of the shared magnitude).

| | pt | en | mean |
|---|---|---|---|
| base | 3.316 | 3.282 | 3.299 |
| ft_pt / ft_en alone | 3.236 / 3.499 | 3.399 / 3.134 | 3.317 / 3.316 |
| **linear** (chosen on validation) | 3.289 | 3.199 | **3.244** |
| SLERP | 3.289 | 3.200 | 3.244 |
| RegMean | 3.295 | 3.190 | 3.243 |
| TIES 0.2 / 0.5 | 3.288 / 3.323 | 3.218 / 3.209 | 3.253 / 3.266 |
| task arithmetic λ = 1 / 0.7 | 3.475 / 3.336 | 3.269 / 3.205 | 3.372 / 3.271 |
| DARE linear / DARE-TIES (0.5) | 3.487 / 3.388 | 3.296 / 3.232 | 3.391 / 3.310 |

The linear merge of two fine-tunes **is better than the base in both languages and,
on average, better than each specialist** — each one remains the best in its
own language, but pays dearly in the other; the merge is a single model that serves
both. It is the promise of merging, measured, and at the size it actually has. TIES at
low density comes close; DARE and task arithmetic with λ = 1 make it worse.

**Trained separately** — diagnosis `:unrelated` (weight cosine 0.34).
The linear mean gives 5.08 bits/character against 3.3 for each specialist: worse
than either, as the diagnosis warned before merging. A network is the same
function under any permutation of its hidden units; two networks trained
separately are not aligned, and averaging misaligned weights is noise.

What the numbers say, from first principles:

- **DARE** perturbs each delta by `√((1 − p)/p)` *of the delta's own norm*
  (100% at p = 0.5). It is harmless only if the fine-tune is redundant — which the
  paper observed in billion-parameter models with deltas of 0.1%. In a small
  model, with no redundancy, the perturbation is damage. This cannot be read from the weights; it
  is measured.
- **Task arithmetic with λ = 1** applies each delta at full strength —
  including the part of each one that worsens the other's domain (tuning toward
  Portuguese pushes away from English). `λ = 1/n` is the linear mean.
- **TIES** only discards energy (38% at density 0.2) and elects signs; when the
  deltas are not sparse, that is loss, not cleanup.
- **RegMean** (least squares over each model's activations) ties with
  linear here. In a planted bigram the activations are *one-hot*: the Gram is
  diagonal and RegMean becomes, **exactly**, the mean weighted by the count of
  each context (tested in closed form). Its known gain appears with
  correlated activations and specialists in different tasks; here it was
  measured and did not win — it is recorded as such.

That is why `diagnose`'s recommendation ends in "measure": vapor does not promise a
winning method; it delivers the reproducible measurement that chooses.

## Disk to disk (0.7.0)

Merging two 7 B models through the in-memory API takes three copies of 28 GB.
`Merge.stream/3` (`mix vapor.merge --stream`) reads **one tensor at a time** from each
input (through the safetensors index, without loading the rest), merges it with the
**same kernels in the same block order**, and writes it straight into the output
file — the safetensors header is computed before any data, from
the declared shapes. Memory is that of the largest tensor times the number
of inputs.

- **The same bits**: the files are byte for byte those that `write_sharded/4`
  writes from `merge/2`, for linear, *task arithmetic*, SLERP, TIES,
  DARE and bf16 output, with sharding; the receipt carries the same Merkle roots
  for inputs and output (`merge_stream_test.exs`).
- **Measured**: three 20 MB models, binary memory peak **4 MB** when
  *streaming* against **83 MB** in memory; in the quality suite, peak/model
  size ≈ 0.1 with the same root, and a control (another `t`) that changes the root.
- **Found by measuring**: in the quality suite, called from a process with a
  large *heap*, the peak rose to 0.9× the size of the models — the tensors
  already written stayed around as garbage until that process's next collection. Now
  each written tensor is collected before the next one is read (0.08×, with the
  same caller).
- **What *streaming* does not do**: the airlock's full admission (it reads the
  weights). Compatibility is checked on what the files declare — the
  canonical *digest* of `config.json`, the adapter the airlock claims from
  the configuration and the tensor index, name, shape and type of
  each tensor — and the receipt's `config` field is that *digest*. RegMean stays
  out (it measures activations of the running models).

## What is different here

- **Compatibility checked at the airlock.** Same adapter, same contract, same
  names and shapes, and — unless `allow_config_mismatch: true` — the same configuration
  *digest*. Merging two topologies is a refusal; a NaN in a weight is a
  refusal that names the tensor.
- **The same bits on every host.** binary64 `+ − × ÷ √` (correctly
  rounded), `acos` without libm, a single rounding to binary32 per entry;
  RegMean's Grams come from the substrate (programs: the activations and `XᵀX`) and the
  system is solved by Cholesky in binary64 (`Vapor.Linalg`).
- **Receipts.** `vapor.merge/1` names the method, parameters, configuration
  *digests*, Merkle roots of the weights of each input, of the base and of the output and,
  for RegMean, the *digest* of the Grams; `vapor.merge.select/1` adds all the
  scores. Co-signable by independent nodes.
- **The airlock says what to measure.** The core does not know what a layer is: the
  adapter declares `taps/1` — the program's activations that feed each
  matrix (in the decoder: the normalised attention input for q, k, v; the MLP's
  for gate and up; the final norm for the head).

## Limits

- RegMean solves on the BEAM: `O(d³)` per matrix and `O(output·d²)` for the rows —
  seconds for widths of a few hundred, impractical for 4,096. The
  path is the product and the inverse as programs on the worker. It also does not measure
  `o_proj` and `down_proj` (their inputs are not named activations in the program).
- *Streaming* disk to disk (0.7.0): see "Disk to disk (0.7.0)" above; RegMean
  still needs the models in memory (it measures activations).
- Permutation alignment: only in SwiGLU blocks (§8); attention heads and
  the permutation of the residual stream are not aligned.

## 8. Align before merging (0.15)

A network is the same function under any permutation of its hidden units: in a SwiGLU block
`down(silu(gate·x) ⊙ up·x)`, reordering the rows of `gate` and `up` and the columns of `down` by the
same permutation changes nothing. Two networks trained separately fall into different orders, so
averaging the weights mixes unrelated units and destroys both — the case that `diagnose`
called "unrelated" and said not to merge.

`Merge.merge(models, align: true)` (and `Vapor.Merge.Align`) does the **weight matching** of
*Git Re-Basin* (Ainsworth, Hayase & Srinivasa, 2023): per block, the permutation of the second model
that maximises the total inner product with the first — a linear assignment problem, solved
**exactly** by the Hungarian method (shortest augmenting paths, `O(n³)` in the block width;
checked against all permutations up to n = 7). The merge receipt records the *digests* of the
permutations applied (`params.aligned`).

Measured: a network merged with its own shuffled copy — without alignment, the *logits* change by
2.58; with alignment, the merge **is** the network (difference 0.0). Two independently initialised
networks: alignment increases the matched similarity of every block. Scope: the
SwiGLU blocks of the Llama format (where the width is); width above 2,048 is refused — the
cubic assignment belongs on a native substrate, not on the BEAM.
