# Kimi K3 — the airlock, the checks, and what the report gets wrong

> `Vapor.Lock.Adapters.DeltaHybrid` (`lib/vapor/lock/adapters/delta_hybrid.ex`), `Vapor.Quant.MXFP4`,
> `Vapor.Train.Balance`. Tests: `kimi_test.exs`, `balance_test.exs`. Reference:
> `test/python/kimi_k3_reference.py`. Source read: *Kimi K3: Open Frontier Intelligence*
> (Kimi Team, technical report, August 2026).

## What K3 is, in the report's words

2.8 T parameters, 104 B active, 93 layers, a 1 M-token context. Four architectural changes over
Kimi K2:

| piece | what it does |
|---|---|
| **Kimi Delta Attention** (KDA) | a delta-rule recurrence with a per-channel forget gate; three KDA layers for every Gated MLA layer, plus a final MLA layer (69 + 24) |
| **Gated MLA, NoPE** | DeepSeek's latent attention without rotary positions (KDA carries position), with a full-rank sigmoid gate on its output |
| **Block Attention Residuals** | each module reads a softmax-weighted mix of the embedding and the sums of earlier blocks, instead of one running residual |
| **Stable LatentMoE** | 896 routed experts, 16 active, working in a 3584-wide latent space; an RMSNorm before the up-projection; SiTU-GLU (a soft-capped SwiGLU); Quantile Balancing of the router |

The routed experts ship in MXFP4. The vision tower is MoonViT-V2.

## What vapor admits, and how it runs

The whole text model as one step program (`Vapor.Recurrent`). The KDA state and its convolution
windows are a recurrence, as in Mamba. The MLA layers keep a cache of the compressed latent in the
session's state, written at `pos`. The key map is absorbed into the query, so the cache holds
`kv_lora_rank` numbers per token and layer. Prefill and decoding are the same instructions and give
the same bits. Each equation, and where it lives in the adapter, is in its moduledoc.

The configuration is **vapor's spelling** (`model_type: "kimi_k3"`). Moonshot's `config.json`
and tensor names cannot be read from the machine that built this (huggingface.co is refused by
its network policy). The names follow DeepSeek-V3's where the report says the module is
DeepSeek's (MLA, the router). Everything else is named in the adapter. A real checkpoint spelled
differently is refused with the field named, and reconciled by an alias file, not code.

## The checks

An independent reference (numpy, float64), written from the report's equations alone, builds two
tiny K3s (6 layers, 3 KDA + MLA + KDA + MLA, AttnRes blocks of 2 layers, 16 experts, top 3) and
computes them its own way: no caches, MLA with decompressed keys and values, the whole sequence
at every step.

| check | result |
|---|---|
| every tensor of the checkpoint read | yes, both variants |
| logits against the reference | relative error 4.4·10⁻⁶ (report's constants), 1.35·10⁻⁵ (soft caps biting, low-rank query, MXFP4 experts) |
| greedy decoding, 10 tokens | identical |
| native worker against the oracle | the same bits |
| control: forget the KDA state at every token | fails (another function) |
| control: forget the MLA cache at every token | fails |
| the report's chunkwise KDA (Eq. 4) against its recurrence (Eq. 1) | equal to 3·10⁻¹⁶. The UT transform the report defers to Kimi Linear was derived here and checked in the reference |
| the bounded decay's purpose (Eq. 5) | over a 16-token tile in float32, the reciprocal decay stays finite with `g ∈ (−5, 0)` and overflows with Kimi Linear's `−e^A·softplus(z)` |

## MXFP4 is exact

An MXFP4 value is `±{0, ½, 1, 1½, 2, 3, 4, 6} · 2^(s−127)`: a two-bit significand times a
power of two. Every such value inside binary32's range is a binary32 value, so decoding is exact,
and a model computes the same bits from MXFP4 experts as from any other copy of the same
values. An element that would overflow (6 under `s ≥ 253`), or a NaN scale (`s = 255`), is refused
by name. The rest of a block with a large scale is still read. The layout read is gpt-oss's: blocks
`u8[rows, k/32, 16]`, low nibble first, plus scales `u8[rows, k/32]`. K3's own layout is not verified.

## Quantile Balancing: the claim checked, and a flaw in Algorithm 1

Appendix C derives the router's bias from the balanced assignment `max Σ xᵢⱼsᵢⱼ` (each token k
experts, each expert `q = mk/n` tokens). Its relaxation is exact, and its dual is minimised one
block at a time, each block a quantile. `Vapor.Train.Balance` implements Eq. 14, Algorithm 1 and
the assignment itself, solved by the rational simplex with its dual certificate checked.
Measured on seeded batches (`balance_test.exs`):

- **The relaxation is integral** on every batch, as the report says (a bipartite b-matching: the
  matrix is totally unimodular). The simplex's optimum is a 0/1 assignment with a checked
  certificate.
- **One step of Eq. 14 lowers the worst load.** The mean of max load / q goes from 1.89 to 1.51 on
  8 tokens × 4 experts.
- **Algorithm 1 as written does not recover the balanced assignment.** It reaches a fixed point
  within a few rounds, but the top-k routing read from that fixed point is balanced in only a
  minority of batches (5 of 60 in a 200-round experiment). The reason is in the appendix's own
  convention. It sets each threshold *at* the (k+1)-th and (q+1)-th largest entry, which places
  margins exactly at zero. That creates, by construction, the ties the appendix says "have
  measure zero in practice", and top-k then breaks them by index. Every unbalanced fixed point
  the test sees has such a tie.
- **The midpoint fixes it.** The appendix itself notes that any threshold between the k-th and
  (k+1)-th entries is an exact coordinate minimiser. With midpoints, the same alternation reaches
  the certified optimum on every batch, within 10 rounds.

The training recipe is not affected. It applies a batch's bias to the *next* batch, where ties are
measure-zero again, and its histogram estimator interpolates within a bin. The flaw is in
Algorithm 1 as a statement about solving one batch's assignment.

## The real model, in numbers

Arithmetic from the report's Table 1. These are estimates from its shapes; nothing here was run at
this scale.

| quantity | value |
|---|---|
| routed-expert parameters | 92 layers × 896 × 3 × 3584 × 3072 ≈ 2.72 T |
| at MXFP4 (4.25 bits each, with scales) | ≈ 1.45 TB |
| everything else (attention, latent maps, shared experts, embeddings) | ≈ 60 B parameters, ≈ 120 GB in bf16 |
| read per decoded token at batch 1 | all non-expert weights + 16 experts per layer ≈ 120 GB + 26 GB ≈ 146 GB |
| time per token, bandwidth-bound | ≈ 3 s from RAM at 50 GB/s; ≈ 50 s from an NVMe SSD at 3 GB/s; ≈ 2 hours from a 20 MB/s network |

The proposal to stream weights from Hugging Face "a block at a time" therefore runs into
arithmetic, not engineering. Decoding reads every active weight once per token, so streaming
over a network costs hours per token for K3. For a 15 GB model at 20 MB/s, it costs about twelve
minutes per token, not the "3 to 5 seconds" proposed. What streaming can serve is work that reads
the weights **once** for a whole input: scoring, classifying, or embedding a long document, by one
pass of prefill. The network airlock (docs/SIPHON.md) is how bytes from outside come in at all.

## What is owed

- **The real spelling.** K3's `config.json`, tensor names and MXFP4 layout, read from Moonshot's
  release, written as an alias. The adapter then runs the real checkpoint's headers through the
  contract without its 1.5 TB of data (docs/SIPHON.md, header-only admission).
- **The vision tower** (MoonViT-V2) and the **MTP layer** (speculative drafts). Neither is read.
- **Serving the hybrid with `Vapor.Engine`**, so that the KDA state and the paged MLA cache are
  shared by concurrent sequences, with the report's checkpointed prefix cache. Today a K3 runs one
  sequence per session.
- **Speed.** The exact top-k router compares every pair of experts (896² per layer for the real
  model). It is exact and right for small models, not for K3's width. Lowering the tiny test model
  takes about a minute.

## Fable, Sol, Astra, DeepSeek V4

Closed models (the report's Claude Fable 5 and GPT-5.6 Sol, Google's Astra) have no weights to
admit. vapor reaches them only as proposers through the agent backends. What they propose is
checked like any other proposal, and they get no tools that act (docs/SIPHON.md). DeepSeek V4: no
report was supplied. A new family goes through the same steps as this one: read the report, write
an independent reference from its equations, then write the adapter.
