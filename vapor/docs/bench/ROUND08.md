# Round 0.8 measurements

Regenerable with `mix vapor.bench --round08`. Machine: Intel(R) Xeon(R) Processor @ 2.80GHz, 2 vCPUs, OTP 25.
The GPU is **lavapipe** (Mesa's Vulkan on the same CPU cores): the
GPU times measure the machinery — what the resident session takes out of
each step — not what a discrete GPU would deliver. Random weights:
measurements of the machine, not of the quality (that is `mix vapor.quality`).

## 1. Resident sessions on the GPU (P0 of 0.7)

Llama, width 256, 4 layers, 8 heads (2 KV), vocabulary 2048, context 256; a *prompt* of 32 tokens, then 32 one-token steps.
Wall-clock time per step, measured on the BEAM (includes the protocol), median:

| path | ms/token | host↔device bytes per token | recordings reused |
|---|---:|---:|---:|
| GPU, one `RUN` per token (no session: pipelines and KV cache at every step) | 72.36 | 1056776 | — |
| GPU, resident session, direct memory | 12.36 | 8200 | 31/32 |
| GPU, resident session, *staging* (the discrete-GPU path) | 10.54 | 8200 | 31/32 |
| CPU, session in the native worker | 1.29 | — | — |

Same bits on the four paths (logits of every step): **yes**.
The session cuts the time per token 5.86× and
the traffic 129×: the step moves the ids and one row of logits,
not the cache.

**Engine** (4 concurrent requests, 24 *prompt* tokens, 32 generated, greedy):

| substrate | tokens | tokens/s |
|---|---:|---:|
| CPU (native worker) | 128 | 1384.04 |
| GPU (resident session) | 128 | 83.38 |

Same tokens on both: **yes**. On lavapipe the GPU
is the CPU itself, so the throughput comparison says little about a real GPU; what
it proves is that the engine serves entirely on Vulkan, with the CPU's bits.

## 2. Sparse 4-bit experts (`qgemv_masked`)

Mixtral, width 512, 8 experts (top-2), 2 layers, sb4 weights (4.75 bits/weight); ISA `x86_64_avx512`. Time in the worker (median of 5) and instructions
retired, counted exactly by the RVV interpreter (VLEN 256):

| tokens | dense ms | sparse ms | × | dense instructions | sparse instructions | same bits |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 2.94 | 1.84 | 1.60 | 15033088 | 5273888 | yes |
| 8 | 14.26 | 5.59 | 2.55 | 119430325 | 41309909 | yes |

With top-2 of 8, each token reads 1/4 of the experts; the rest of the model
(attention, router, head) does not change — the total gain is less than 4×.
