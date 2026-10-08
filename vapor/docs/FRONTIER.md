# Serving frontier models without losing the bits

The models the industry serves in 2026 are no longer a dense Llama: they are
mixtures of experts (Mixtral, Qwen3-MoE, DeepSeek-V3), latent attention
(MLA), sliding windows (Mistral, Gemma 3), state-space models (Mamba) — and
they are served with speculative decoding and spread across several
machines. Each of these tricks, as it is usually implemented,
**trades reproducibility for speed**: the result comes to depend
on the batch size, the order of the tokens, the number of GPUs.

This page shows how vapor implements each one **without** that trade — the
output remains a function of the terms, equal on every substrate — and
what each one costs. Numbers: [bench/FRONTIER.md](bench/FRONTIER.md).

## 1. Mixture of experts: predication, not permutation

**The pain.** A top-2-of-8-experts MoE computed densely costs 4× what is
needed. The usual GPU solution — sorting the tokens by expert
and dispatching in blocks — introduces gather/scatter and data-dependent
shapes, and (with a *capacity factor*) drops tokens depending on the batch.

**First principles.** In a *weight-stationary* GEMV on a CPU, the cost is
**reading the weights**. There is no need to move tokens: it is enough not to read the weight
rows that no token chose. `Term.linear_masked(x, W, m)` is the GEMV with
a per-row mask: rows with `m = 0` come out `+0` without touching `W`, and the
others execute **the same instructions** as the dense GEMV. Since the selection
(`sel`) discards the unchosen rows anyway, `moe: :sparse`
and `moe: :dense` give the same bits — checked on AVX2, AVX-512, RVV
(poisoned interpreter), Vulkan and QEMU.

| Reduced Mixtral, decode | dense | sparse |
|---|---|---|
| T = 1 | ≈ 13 ms | ≈ 5.1 ms (2.6×) |

With more tokens per step, more experts are chosen by someone and the
advantage drops (≈ 1.6× at T = 8 and 32) — the physics of the method, measured.

**In 4 bits (0.8).** The `sb4` weights (executed as `sb4x`, 4.75 bits/weight; 4.6875 as stored) ran dense: the
masked quantized GEMV was missing. `Term.qgemv_masked/3` and the KIR kernel
`gemv_sb4_masked` (emitted for x86, AVX-512, NEON, RVV and SPIR-V — on Vulkan
the mask zeroes the row's sub-block count) skip the unchosen
rows without reading their nibbles or their scales; the chosen ones
execute the dense GEMV's instructions. Bits = dense on every substrate; an
unchosen expert can have NaN scales without changing a bit (test).

| Reduced Mixtral in sb4 (width 512, top-2 of 8) | dense | sparse | retired instructions (RVV, exact) |
|---|---:|---:|---|
| T = 1 | 2.94 ms | 1.84 ms (1.6×) | 15.0 M → 5.3 M |
| T = 8 | 14.3 ms | 5.6 ms (2.6×) | 119 M → 41 M |

([bench/ROUND08.md](bench/ROUND08.md) §2.) The total gain is less than 4×
because attention, router and head do not change.

## 2. Latent attention: two programs, the operator chooses

DeepSeek's MLA stores, per token, a latent `c` (512) and a rotary
key (64) instead of full K and V per head (128 × 384): 85× less
memory. To attend without expanding, the key projection is **absorbed** into the
query and the value projection into the output (`Term.linear_grouped`, block-diagonal per
head), and attention becomes MQA over the latent.

The scrutiny: this is **not the same computation** in another order — it is another computation
(other bits, ~3× the attention FLOPs). So there are two programs: the latent
form (the default for MLA models) and the expanded one (`mla: :expanded`), both
checked against `transformers`, each with its own stable bits.

## 3. Sliding window and the circular cache

A row at position `p` with window `w` attends to `max(0, p − w + 1) … p` **in
canonical order**: only the range changes, so a windowed row is, bit for bit,
unwindowed attention over the same keys moved to the start (test).
The paged kernel starts walking the block table at the window's first
page.

**The circular cache.** If the window is on in *all* layers, no
query reads a position older than `w`. A ring in contiguous memory
would break the read order; a ring **in the block table** does not: logical
page `j` points to `mine[j mod R]` and the logical order stays intact. Position
`x` is overwritten by position `x + R·page`; a step writes
`p0 … p0+n−1` and then reads back to `p0 − w + 1`, so nothing readable is overwritten
if `R·page ≥ w + n − 1`:

    R = ⌈(w + step_tokens − 1) / page⌉

The test checks the bits against the full cache **and** that `R − 1` pages
change them (the bound is tight). A hybrid model (global layers among the
local ones, Gemma 2/3) keeps everything — the engine decides via `Config.ring_window/2`.

| context | window | pages per sequence (ring / full) |
|---|---|---|
| 32,768 | 4,096 | 260 / 2,048 → 7.9× more sequences |
| 131,072 | 4,096 | 260 / 8,192 → 31.5× |

## 4. State-space models (Mamba): the recurrence is the definition

An SSM trades the KV cache for a fixed-size state: the cost per token is
the same at token 10 and at token 10⁶. The parallel (associative) *scan* that
GPUs use for *prefill* reorders sums — the bits would depend on the degree of
parallelism. Here the program is **one step** (`t = 1`): *prefill* and decode
are the same instructions and give the same bits.

The state `s : f32[di, N]` and the causal convolution's window are the
program's `state`; the worker session feeds them back (`s ← s_next`) **inside the
worker** after each step — the `STEP` frame gained state copies
for this —, so a token costs one id in and one row of logits
out. Two new operators, both canonical: `log` (≤ 1 ulp from the
correctly rounded logarithm) and `softplus` (Mamba's Δ).

| d 256, 4 layers | step | memory per sequence |
|---|---|---|
| Mamba, any context | ≈ 0.5 ms | 38,912 floats, fixed |
| attention, context 8,192 | ≈ 1.9 ms | 4.2 M floats |

Checked against `transformers`'s `MambaForCausalLM` (3.4·10⁻⁷, identical greedy
output), with the control of a memoryless Mamba (state zeroed at every
token), which fails. `Vapor.Recurrent` generates; `Vapor.Engine` refuses
recurrent models by contract (its memory model is the paged cache).

**Mamba-2 (0.8).** `Vapor.Lock.Adapters.Mamba2` (`Mamba2ForCausalLM`):
multi-head with a scalar `A` per head, `B`/`C` shared by groups of
heads and passed through the same causal convolution as `x`, per-head Δ with
`time_step_limit`, and the gated RMSNorm before the output projection. The
head → channels expansion is exact: Δ reaches each head's channels through a
product with a *one-hot* matrix (`Δ·1 + Σ 0`: one nonzero term, no
rounding), `A` and `D` expanded at construction, and the group row of
`B`/`C` copied per channel with `gather_row`. Checked against
`transformers` (chunked scan, no cache): 7.8·10⁻⁷ relative error in the
logits, identical greedy output, native worker = oracle, GPU = CPU bit for bit.

Two divergences found along the way, and what was done about them:

- **The gated norm.** The training code (`mamba_ssm`, `RMSNormGated` with
  `group_size = d_inner / n_groups`) normalizes **each group**;
  `transformers`'s `MambaRMSNormGated` normalizes the whole width, whatever
  `n_groups` is. They only agree with one group. The default here is the
  training one (`gated_norm: :group`); `gated_norm: :whole` reproduces
  `transformers`. Both are checked against their own reference, and each one
  **fails** against the other's (error 0.74 and 0.66) — the comparison discriminates.
- **The Δ limit.** `transformers`'s cached decoding step
  does not apply the `time_step_limit` that its *chunked scan* applies; the step
  here always applies it (the trained semantics). The reference greedy output is
  therefore generated by the full pass, not by `generate()`.

## 5. Tree speculation over shared pages

Linear speculation was already exact (verifying `k + 1` rows gives the
bits of `k + 1` steps). The tree verifies **several** drafts in one step:

- branch 0 writes directly into the context's pages;
- branch `b ≥ 1` is a *slot* whose table lists the context's pages up to the
  last full page (shared, nobody writes to them) and then
  its own pages; it recomputes the tokens of the partial page (less than
  one page) before its own;
- the branch that agrees for longest wins; if it is a forked branch, its
  pages **become** the context's — ownership changes, nothing is copied.

Since a row's logits depend only on its token, its position and the
KV it reads, the output is the target's greedy output, token by token, for any
draft (tested on five tree shapes and with an always-wrong
draft). The default draft is not a model: it is **prompt lookup** — what
followed the previous occurrences of the last n-gram, copied with overlap
(a loop the text has entered is proposed to the full depth). It is
strong exactly where generation copies the input: answers with retrieval,
code editing, summaries that quote.

| target | no draft | lookup, linear | lookup, tree of 4 |
|---|---|---|---|
| planted bigram (the output follows the context) | 1.0 token/step | 6.0 | 6.0 |
| random weights (the output does not follow) | 1.0 | 1.1 | 1.2 |

The second row is here on purpose: on a target whose output does not copy the
context, lookup almost never hits, and the recomputed rows cost more than
they save. Measured, not hidden.

## 6. Tensor parallelism that does not change the bits

The canonical product accumulates 16 lanes in sequence and sums them in a
fixed tree; each output element is one of these products. Hence:

- **column-parallel** (each shard computes whole rows of `W`) is
  exact: each element is computed whole, in one shard, by the same
  instructions;
- **row-parallel** (splitting `k` and summing partials — the Megatron half with
  all-reduce) is **not**: accumulation restarts in each shard and the
  partials meet in another order (measured: 81% of the elements change).

The exact MLP (`Vapor.Shard.mlp/4`) replaces the all-reduce with an **all-gather**
of the intermediate activation and becomes column-parallel again: each layer stays
exact; the price is communication volume (`b·inter` instead of `b·d`, ≈ 2.7–4×
in SwiGLU MLPs). It is the trade a reproducible distributed runtime has to
make — stated with its cost.

**Across BEAM nodes (0.8).** `Vapor.Shard.Cluster` takes the exact form to the nodes
of an Erlang cluster (Distributed Erlang: authenticated by the *cookie*, and by
TLS with `-proto_dist inet_tls`). Each node receives its rows of `W` once,
with the SHA-256 — `Vapor.Shard.Host` recomputes it and **refuses** a shard that
does not match —, and keeps them **resident** (compiled with the shard as a
constant); a call carries only the activation and brings back the columns. Two
consequences that an inexact runtime cannot have, both tested with real `:peer`
nodes:

- **Failover without drift**: a node lost mid-run has its
  shards placed again on the survivors, the call is redone, and the
  answer has **the same bits** — nothing downstream notices.
- **Replica as verification**: with `replicas: 2` each shard is computed
  on two nodes and the results compared **bit for bit**; a node that corrupts
  a bit is caught, not averaged. Non-reproducible floating point could only
  compare with a tolerance — and a careful adversary stays within
  it.

## 7. GPU-resident sessions (0.8)

Up to 0.7 every run on Vulkan (`vapor-fabric`) recreated *pipelines*,
*buffers* and *command buffers* and carried the KV cache back and forth: the engine did not
serve on the GPU. Now the fabric has the sessions the CPU worker already had:

- `OPEN` creates the *pipelines* once, allocates the *buffers* at the program's maximum
  size — **direct memory** (`DEVICE_LOCAL | HOST_VISIBLE`, the case of
  integrated GPUs and of lavapipe) or ***staging*** (a copy *buffer*, the
  discrete-GPU path; forceable with `staging: true`, and chosen automatically
  if direct memory is lacking) — and uploads the constants;
- `STEP` writes only the inputs, dispatches, copies the state (`s ← s_next`)
  **inside the GPU** and reads only the requested outputs. The recorded *command buffers*
  sit in a cache whose key is **the exact bytes** of the step (writes,
  dispatches, copies, *push* constants): a step equal to an earlier one is
  just resubmitted — 31 of 32 decode steps reuse the recording;
- `CLOSE` releases; a fabric that dies invalidates the sessions (generation) and the
  engine reopens — never the BEAM.

| Reduced Llama (width 256, 4 layers), one token | ms/token | host↔GPU bytes per token |
|---|---:|---:|
| GPU, one `RUN` per token (0.7) | 72.4 | 1,056,776 |
| GPU, session, direct memory | 12.4 | 8,200 |
| GPU, session, *staging* | 10.5 | 8,200 |
| CPU, session in the worker | 1.3 | — |

The same bits on all four paths. `Vapor.Engine` serves entirely on the GPU
(`mix vapor.serve --gpu`), with the CPU's tokens. Here the "GPU" is lavapipe —
the CPU itself emulating Vulkan —, so the throughput (83 against 1,384 tokens/s
on the CPU) says nothing about a real GPU; what is proved is the protocol, the
residency and the equality of the bits.

## 8. What is not done

- Attention with *workgroup* memory in SPIR-V and the end of `@max_dh = 512`;
  measuring on a real GPU (only lavapipe here).
- FlashAttention: the blocked online softmax is another canonical order; only as a
  declared `:fast` policy.
- Mamba *prefill* in a single `RUN` frame with iterations (today: one `STEP`
  per token); Jamba/Zamba/Bamba (hybrids: KV cache **and** per-sequence state
  in the engine) and Falcon-Mamba.
- Attention sharded by heads across nodes (the MLP and the exact projections
  are done).
- Tree speculation without recomputing the partial page (copy-on-write of the
  page), and served by `Vapor.Engine` instead of a dedicated session.
