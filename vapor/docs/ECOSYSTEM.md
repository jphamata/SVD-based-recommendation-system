# vapor ecosystem — scrutiny and plan

> **0.4.0:** models now enter through the **model airlock** (`Vapor.Lock`,
> [AIRLOCK.md](AIRLOCK.md)) — the engine, the embedder and the server read a contract,
> not a family. Image and audio: [ANY_TO_ANY.md](ANY_TO_ANY.md). Merging:
> [MERGING.md](MERGING.md). Output quality: [QUALITY.md](QUALITY.md).

This document examines the expansion proposal (tokenizer, Hugging Face
ingestion, LoRA/QLoRA, distillation, serving, benchmark apparatus) against the
guarantees the core already delivers, and fixes the build order. It is
revised at every phase: what is learned during implementation comes back here.

## Criterion

A new layer gets in if it preserves the core's invariants, or if it says
explicitly where it stops preserving them:

1. `deps: []` in the control plane; no generated code inside the BEAM.
2. Exact semantics in the oracle; `:canonical` policy bit-identical across
   substrates; `:fast` within an envelope.
3. Every accepted artifact has a certificate; every rejection has a counterexample.
4. Published numbers are measured here, with the source cited.

## Verdict per item

| Proposal | Verdict | Technical reason |
|---|---|---|
| OpenAI API | **yes**, HTTP/1.1 + SSE over `:gen_tcp` | it is what the OpenAI API uses for streaming; fits in pure OTP |
| HTTP/3 | **not in the core** | requires QUIC + TLS 1.3; terminating HTTP/3 at a proxy in front (Caddy, nginx, h2o) is safer and costs the model nothing |
| Phoenix WebSockets | **not as a dependency** | vapor is a library: a Phoenix app calls `Vapor.Serve` in the same BEAM |
| Zero-copy Python bridge | **later** | an HTTP client is enough; tensors already live in content-addressed `/dev/shm` |
| Pure BPE tokenizer | **yes** | byte-level BPE (GPT-2/Llama 3/Qwen) and SentencePiece-BPE with byte fallback (Llama 2/Mistral), reading `tokenizer.json`; differential oracle: HF's `tokenizers` library |
| Trie in Zig SIMD | **no** | it would be a NIF (breaks containment) or a process crossing more expensive than the tokenization itself |
| "20 M tokens/s per core" | **rejected a priori** | no measurement supports it; it will be measured and published |
| safetensors ingestion | **yes** | JSON header + offsets validated as an airlock; f32 weights mapped by the worker without copying |
| `config.json` → terms | **yes for Llama, Mistral, Qwen2** → later Qwen3, Qwen3-MoE, Mixtral, Gemma 3, DeepSeek-V3 (2026-10-01) | same structure (RMSNorm, RoPE, GQA, SwiGLU); MLA and MoE came in as exact contractions; Mamba in 0.6.0 |
| GGUF | **later** → done in P8 | secondary format; safetensors covers HF. Reading, dequantization and export (f32/q8_0) checked against gguf-py and llama.cpp itself |
| 70 B: 140 GB → 41 GB | **arithmetic correct** | 70·10⁹ × 4.6875 / 8 = 41.0 GB |
| "mathematically bounded perplexity loss" | **false** | the per-weight error is bounded and the kernels' rounding error has an envelope; perplexity can only be measured |
| QLoRA "adding B·A before emitting" | **corrected** | never materialize B·A: `y = W₀x + s·B(Ax)`, two thin GEMVs |
| Fusing B·A into `:sb4` | **with a caveat** | requantizing changes the quantization error; it is not lossless |
| Backprop "through Rung 3" | **yes, with the right reading** | the adjoint identity ⟨Wx,v⟩ = ⟨x,Wᵀv⟩ certifies the transposed kernels that reverse mode uses |
| Distillation "stabilized below the 2γ_K S_ij noise" | **reformulated** | stability comes from log-softmax with subtraction of the maximum; the envelope certifies the rounding error of the loss, it does not "stabilize gradients" |
| PagedKV with a dedicated OTP process | **structure yes, process no** | the page table is pure data in the engine's state; one more process only serializes messages |
| Continuous batching without padding | **yes** | and with a strong property: the canonical order does not depend on batch size ⇒ **batch invariance** (same tokens alone or in a batch) |
| Speculative decoding "accepted within the Wilkinson envelope" | **wrong concept** | acceptance is by token comparison (greedy) or rejection sampling; with batch invariance, the result is **identical** to decoding without speculation — that is the real guarantee |
| "up to 3.5×" | **rejected a priori** | depends on the acceptance rate; it will be measured |
| `perf_event_open` counters | **yes, when the kernel allows** | opened before seccomp; reports `:unavailable` otherwise |
| Joules/token via RAPL | **yes, where it exists** | on this machine `/sys/class/powercap` does not exist ⇒ not validatable here |
| "Vulkan energy interfaces" | **do not exist** | Vulkan core does not expose energy |
| ULP drift vs Float128 | **replaced** | the oracle is exact (dyadic rational), stronger than binary128 |
| Roofline SVG | **yes** | the data already comes from the arbiter |
| E-graph saturation | **later, low return** | almost no floating-point rule is exact |
| AVX-512 | **later** → done (HPC) | this CPU has AVX-512 (x86-64-v4); a performance gain, not a guarantee gain — the bits are the same |
| Training an 8B in LoRA "on corporate data" | **wrong scale for 1 core** | the machinery is demonstrated on small models; real scale needs multi-core/GPU |

## The prerequisite the proposal omits

A Llama needs `exp`, division, inverse square root, maximum, per-row
reductions, RoPE, softmax, attention over a variable-length KV cache and
embedding lookup. None of this exists in the core, and the way it is
introduced decides whether the central guarantee survives.

**Decision:** in the `:canonical` policy, every function is a program over
{correctly rounded +, −, ×, integer operations, comparison, selection}.

- `+ − ×` are correctly rounded on x86, AArch64, RVV **and** Vulkan
  (where division and square root are *not*: 2.5 ULP on Vulkan). Building `div`,
  `rsqrt` and `exp` from Newton–Raphson and polynomials in a fixed order gives
  identical bits on every substrate, GPU included, and the oracle computes them by
  composition, with no special case.
- `max`/`min` as `select(a < b, b, a)`: the NaN semantics of `maxps`,
  `fmax` and `vfmax` differ between ISAs; comparison + selection does not.
- `exp` flushes to zero by definition below 2⁻¹²⁶: the output is never subnormal, so
  GPUs that discard subnormals stay bit-identical.
- RoPE: cos/sin tables computed once in the BEAM and embedded as
  constants (bits in the certificate).
- Attention over the cache: the 16-lane reduction assigns element *i* to lane
  *i mod 16*; positions beyond `pos` contribute exact zeros. So stopping the
  loop at `pos + 1` produces **the same bits** as the masked definition over
  the maximum length: the run-time length is an optimization, not
  semantics.
- `:fast` may use the hardware's `vdivps`/`vrsqrtps`/FMA and is certified by
  envelope.

**Compositional certification.** The exact oracle is too slow for a whole
model. But each kernel is the same machine code whatever the
dimension (dimensions are arguments). A model certificate binds: the hash
of each kernel, its differential verification on probe instances, the
well-formedness of the schedule and admission at the maximum extents; full
equality with the oracle is checked on a small configuration of the same
architecture.

## Phases

Each phase ends with green tests at every level, a commit and an entry here.

| Phase | Deliverable | Done when |
|---|---|---|
| P1 | Numerics and operators: integer/comparison/selection primitives on the 3 CPU ISAs; `recip`, `div`, `rsqrt`, `exp`, `sigmoid`, `silu`, `max`; per-row reductions; broadcast; f32 GEMV; run-time gather and row write; worker FP environment checked | oracle = x86 = AArch64 = RVV (QEMU, VLEN 128–512) bit for bit; objdump decodes everything; ULP error of each function measured and published |
| P2 | JSON, safetensors (airlock), `config.json` → Llama/Mistral/Qwen2 program, quantization at ingestion | typed rejection for a malformed header; certified programs |
| P3 | Model oracle: `transformers` + `torch` (PyPI) on small models with random weights | logits within a declared tolerance; greedy generation identical outside ties |
| P4 | Tokenizer | identical to `tokenizers` on test corpora and on real tokenizers obtained offline |
| P5 | Generation, deterministic sampling, engine with continuous batching and PagedKV, OpenAI server + SSE | batch invariance tested; an OpenAI client talks to the server |
| P6 | Measurement apparatus: counters, ULP histogram vs exact, roofline SVG | reproducible numbers with the source recorded |
| P7 | Autodiff, LoRA, AdamW, distillation loss | gradients = finite differences in the oracle; certified adjoint; loss drops on a small model |
| P8 | Speculative decoding, SPIR-V for the new operators, GGUF, AVX-512, Mamba | each item with its own criterion |

## Log

- 2026-09-30 — initial plan.
- P2/P3 (f32) — `Vapor.JSON` (RFC 8259, duplicate keys refused, depth
  bounded; differential against Python's `json`), safetensors airlock
  (exact tiling of the data segment, `bf16`/`f16` widened exactly —
  the 65,536 `f16` patterns equal numpy's; reads bit for bit what the
  `safetensors` library writes), `config.json` for Llama/Mistral/Qwen2 (the
  `rope_scaling` and `rope_parameters` RoPE formats of transformers ≥ 5),
  sharded directory loader.
  **Finding:** BEAM terms are trees; the residual stream used ~3× per layer
  made the unshared tree grow like 3^L — and hashing/copying are
  linear in it. A 2-layer Qwen2 took 23.5 s to lower and 6.5 s in the
  oracle. **Fix:** *let-bindings* in `Program` (semantic substitution;
  weights as constant bindings, so no key contains weight bytes):
  0.19 s and 0.05 s, and the largest term does not grow with depth (tested with
  12 layers). The worker went from fixed tables (64 buffers/calls) to
  tables sized by the frame, with the count bounded by the bytes
  received.
  **Parity with transformers 5.18 / torch 2.14 (float32):** five variants
  (Llama with bias and untied head, Llama 3 RoPE, linear RoPE with MHA, Mistral,
  Qwen2 with tied head and 1 KV head): maximum relative |Δlogit|
  2.7·10⁻⁷ … 5.6·10⁻⁷ (declared tolerance 10⁻⁵); 16-token greedy decoding
  identical in all five, with no ties. Whole model bit for bit equal to the
  oracle on x86, the RVV interpreter, AArch64 and RVV (QEMU); batch invariance
  (prefill = decode steps) checked.
  **Reordered:** quantization at ingestion moves from P2 to after P3 — the
  error it introduces only has meaning when measured against the f32 reference.
- Quantization at ingestion — `Llama.program(..., quantize: :sb4)` stores every
  projection matrix and the head in 4-bit superblocks (the embedding stays
  f32: it is looked up, not multiplied). `qgemv` now accepts
  `x : f32[b, k]`: each activation row is an independent GEMV, with the
  same instructions in the same order as `b = 1` — batch invariance in the
  kernel, on the three ISAs, in the interpreter and in SPIR-V (`rows·b` dispatch).
  To fit in x86's 14 GPRs without losing the 2-rows-per-iteration
  variant, W's base register was eliminated (the next row starts
  where the last row pointer ended) and the sub-block count is
  re-read from the arguments into a new register (live ranges are
  envelopes). GEMV 4096² bench without regression (66–80 ms observed).
  **Measured quality** (Llama width 256, Gaussian weights, 24 positions,
  against transformers' float32 logits): `sb4` (4.6875 bits/weight) relative
  RMS error 0.161 and mean KL 0.0129; reference Q4_1 (5 bits/weight)
  0.158 / 0.0129; Q4_0 (4.5 bits/weight) 0.190 / 0.0178. The test requires
  `sb4` ≤ 1.25× Q4_1 on both measures. The high absolute error is inherent to a
  tiny model with random weights (no redundancy); it is not a
  prediction of perplexity on trained models.
- P4 (tokenizer) — `Vapor.Tokenizer`: byte-level BPE (GPT-2, Llama 3,
  Qwen2) and SentencePiece BPE with *byte fallback* (Llama 2), a single
  priority merge procedure in `O(n log n)` (heap + doubly linked
  list; by merge rank or by vocabulary score), tokens
  stored by their byte surface (decoding is concatenation).
  Sources: GGUF (`Vapor.Ingest.GGUF`, brought forward from P8: the HF hub is blocked
  by policy here; the real vocabularies arrive as llama.cpp's vocabulary-only
  GGUFs via raw.githubusercontent, `make fixtures`, SHA-256 pinned) and
  `tokenizer.json`.
  **Results:** the 4 real vocabularies (Llama 3, Qwen2, GPT-2, Llama 2)
  reproduce the 184 vectors Hugging Face generated for them; against the
  `tokenizers` library in 7 configurations (those converted by
  transformers and each release's original options: Llama 3's `ignore_merges`,
  Qwen2's NFC, Llama 2's Prepend+Replace normalizer) × 415
  adversarial texts: ids and decoded bytes identical.
  **Finding:** OTP's NFC (`:unicode.characters_to_nfc_binary`) composes
  across a class-0 mark (`и ๎ ̈` → `ӥ ๎`), violating UAX #15 — and
  changes token ids in Qwen2. `Vapor.Unicode` implements NFC/NFKC by the
  definition (table of 941 primary composites derived at compile time);
  equal to Python's `unicodedata` at every code point up to U+2FFFF and on
  20,000 random mark sequences.
  Speed (1 core, BEAM): ~330 thousand tokens/s encoding English text
  (Llama 3); loading the 11.6 MB `tokenizer.json` in ~4.5 s.

- 2026-09-30 — **new directive: "maximum HPC", end-to-end pipeline, final zip.**
  Scrutiny of the forms of parallelism relevant to this system and where
  each one enters (the guarantee to preserve: bit-identical results
  regardless of how many threads, how many sequences in the batch and
  where the KV lives):

  | form | where | invariant checked |
  |---|---|---|
  | SIMD | already: AVX2, NEON, RVV (VLEN 128–512); **AVX-512** new backend | same bits as the oracle |
  | threads (intra-op) | worker: thread pool with a barrier; each call carries a *partition descriptor* (which argument counts rows, which pointers advance how many bytes per row) | equal for 1…N threads (independent rows) |
  | continuous batching | engine: chunked prefill + decode in the same step | a sequence's tokens equal alone or in a batch |
  | PagedKV | `kv_write`/`attention` kernels with a block table | paged = contiguous bit for bit |
  | replicas (data) | session pool, one per worker, requests distributed | — |
  | BEAM concurrency | one process per HTTP connection, engine as a GenServer, SSE streaming | — |
  | memory | mmap'd weights shared between workers, resident sessions | — |
  | speculation | speculative decoding with a draft; acceptance by greedy comparison | output identical to that without speculation |
  | GPU | SPIR-V of the model operators (lavapipe here) | fabric = oracle |

  Every row of the table is implemented and tested; the status of
  each one is in the items below and in [ARCHITECTURE.md §4.6](ARCHITECTURE.md).

- P5 (generation and serving) — `Vapor.Sampler` (binary64 + SplitMix64, exact
  truncation; greedy and temperature sampling also as a `sample` operator
  on the substrate), `Vapor.Engine` (continuous batching over paged KV, pages
  reserved at admission — no deadlock and no preemption), `Vapor.Serve`
  (HTTP/1.1 + SSE on `:gen_tcp`, `/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, ChatML, Llama 3 and `[INST]` chat templates), CLI
  (`mix vapor.generate | serve | demo_model | export | bench | shm`).
  **Results:** batch invariance tested (9 concurrent requests =
  each one alone, under any prompt slicing and number of threads);
  the official `openai` client talks to the server, with and without streaming;
  `make e2e` runs airlocks → tokenizer → compiler → engine → HTTP →
  client. Sampling on the substrate: 1.8 → 316 tokens/s on a vocabulary of
  151,936 (sorting in the BEAM was the bottleneck).
  **Robustness:** a worker that dies during a step ends the sequences
  it held with `{:done, :error, …}` (HTTP 503, or an error event in
  SSE), the worker is reborn, the session is reopened and the engine carries on.

- P6 (measurement) — `mix vapor.bench` generates [bench/BENCH.md](bench/BENCH.md),
  `roofline.svg` and `engine.svg`, all measured on the spot: kernels per ISA and
  threads, engine by concurrency/threads/ISA/storage/replicas,
  prefill, ULP of the canonical functions against binary64, tokenizer. This VM
  (Xeon 2.1 GHz, 2 vCPUs) does not expose a PMU: CPU time comes from task-clock.
  Numbers from 2026-10-01: memory ceiling 53 GB/s; f32 GEMV 2048² 0.28 ms
  (60 GB/s); bf16 GEMV 2048² 0.17 ms; linear 512² × 64 rows 84 GFLOP/s
  (AVX-512, 2 threads); Llama engine width 256/4 layers/vocab 32,000:
  ~1,500–1,800 tokens/s with 8 sequences, prefill ~5,500 tokens/s;
  Llama 3 tokenizer ~535 thousand tokens/s on one BEAM core. They vary with the
  host's load (shared VM): the source is always the run's BENCH.md.

- P7 (autodiff and distillation) — `Vapor.Autodiff`: reverse mode from terms to
  terms (gradients are programs like any others: lowered,
  executed on every substrate, bit for bit equal and certified); a
  `transpose` operator. `Vapor.Train`: LoRA on the last MLP + head, KL loss against the
  teacher, AdamW — a training step is **one recurrent program**, the
  whole run executes in the worker behind one crossing. **Results:**
  gradients = finite differences (oracle); the adjoint of the transposed
  kernels is certified by Rung 3; KL 0.051 → 0.00093 in 120 steps
  on a small model (~20 ms of worker time); identical bits on 1–3 threads and on the
  fabric.

- P8 — **speculation** (`Vapor.Speculative`): the draft proposes `k`, the target
  verifies `k+1` rows in one step; by batch invariance the output is
  *identical* to the target alone (tested), acceptance only changes the speed
  (identical draft 1.0; similar 0.66; unrelated 0.0).
  **GGUF** (`Vapor.Ingest.GGUF`, `Vapor.Ingest.GGML`, `Vapor.Model.GGUF`):
  dequantization of F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K
  bit for bit equal to `gguf-py`'s; the q/k permutation of the llama.cpp converter
  undone; Llama 3's `rope_freqs` as factors. A GGUF produced by
  `convert_hf_to_gguf.py` itself (from the PyPI sdist) reproduces the
  transformers logits: f32 with relative error 5.6·10⁻⁷, q8_0 within its
  quantization. **Export** (`Vapor.Model.GGUF.write/4`, `mix vapor.export
  --out x.gguf`): f32 and q8_0 (quantization bit for bit equal to `gguf-py`'s,
  ties included); `load(write(m)) = m`; **llama.cpp** (libllama
  compiled from the same sdist) loads the file, tokenizes with our
  vocabulary exactly as vapor does and reproduces the logits (f32 ≤ 10⁻⁴,
  q8_0 ≤ 5% of the largest logit — llama.cpp also quantizes the activations) for
  Llama and Qwen2.
  **SPIR-V of the model operators**: gather, RoPE, KV write (copy, in
  place, paged, last write wins), contiguous and paged attention (three
  passes that recompute the scores — no scratch memory), sampling
  and transposition; the daemon moved to tables sized by the frame. A
  whole model, the engine's paged step with sampling and the recurrent
  decode run on lavapipe **bit for bit equal to the oracle**; a
  `storage: :bf16` model is certified with parity on fabric, AVX2, AVX-512,
  RVV and the oracle.
  **AVX-512**: complete EVEX backend (see ARCHITECTURE §3.2), bit for bit equal
  to the oracle on every canonical program, whole model, paged attention,
  sampling, `sb4` and gradients; binutils decodes every instruction of
  every kernel. Measured gain: `x·silu(x)` 2.3×, batched linear 1.4×, attention
  1.2×; bandwidth-bound kernels stay the same, as the roofline predicts.
  Mamba was left out of this phase (no requested architecture used it); it came in
  with 0.6.0 as a topology adapter ([FRONTIER.md §4](FRONTIER.md)).

- HPC — replicas (`Vapor.Engine.Pool`, `--replicas N`): one compilation,
  weights in the same pages (content-addressed `/dev/shm`), routing
  to the shortest queue, a dead replica replaced without affecting the others; same
  answer from any replica (tested). **Resident bf16**
  (`storage: :bf16`): a portable `vld_bf16` instruction in the four backends and
  in SPIR-V, `gemv_bf16` and `gather_row_bf16`; the program is *bit for bit* the
  f32 program over the rounded weights (tested on every substrate, in the
  engine and through the ladder); GEMV with half the bytes: 2× faster on this
  machine (0.59 vs 1.22 ms at 4096², 2 threads — bandwidth-bound in both
  cases). A Hugging Face bf16 checkpoint comes in without being widened.

- Complete safetensors — every dtype of the format recognized (sizes
  checked, sub-byte in bits): F64 rounded as torch/numpy do;
  F8_E4M3, F8_E4M3FNUZ, F8_E5M2, F8_E5M2FNUZ and F8_E8M0 widened exactly
  (the 256 patterns of each equal to torch's `.float()`, NaNs included);
  16/64-bit and unsigned integers brought to s32 when they fit (otherwise
  refused, naming the tensor); BOOL; F4/F6/C64 recognized and refused by
  name. Writing in F32 or narrowed to BF16/F16 (RNE; equal to torch on
  20,000 patterns with ties, subnormals and overflow), sharded layout and
  `model.safetensors.index.json`. `Config.to_map/1` (tested inverse of
  `from_map/1`) and `Vapor.Model.write/4` (`mix vapor.export --out DIR`):
  transformers loads the directory exported by vapor and reproduces **bit for
  bit** the original checkpoint's logits in the five variants; in BF16, vapor's
  logits over the same weights stay within 10⁻⁵. A model
  comes in as GGUF and leaves as a Hugging Face directory with bf16 shards —
  exercised in `make e2e`.

- 2026-10-01 — **new directive: frontier models, agents, RAG, "immutable
  agents"; then the Elixir ecosystem.** The full scrutiny and the
  design are in [AGENTS.md](AGENTS.md) and
  [ELIXIR_ECOSYSTEM.md](ELIXIR_ECOSYSTEM.md). The verdicts:

  | proposal | verdict | technical reason |
  |---|---|---|
  | frontier models (open architectures) | **yes**: Qwen3, Qwen3-MoE, Mixtral, Gemma 3, DeepSeek-V3 | each new piece reduced to exact contractions (0/1 selection, rank by counting, `sel`), with no "approximate" operator; parity with `transformers` ≤ 8·10⁻⁷ |
  | frontier models (hosted) | **yes, as observation** | OpenAI-compatible and the Messages API; the decision is recorded, not reproduced — saying so is part of the guarantee |
  | "one more agent framework" | **no** | the loop is a commodity; what only vapor has is determinism ⇒ execution as an object of proof (replaying = verifying) |
  | immutable agents | **yes, with five readings and their limits** | spec as a value; append-only journal; capabilities in the digest; replay as proof; erasure by key destruction |
  | "exactly once" | **reformulated** | at most once with an idempotent recipient (stable key per step); that is what exists |
  | RAG | **yes, verifiable** | corpus = Merkle root; retrieval = function (BM25 with CR log, certified dense scores, exact RRF); literal citations by construction |
  | structured output / tool calls | **yes, by construction** | byte grammar over the vocabulary trie; no *retry* |
  | chat templates | **yes, hermetic** | own Jinja with the semantics of the HF environment; byte for byte equal to `jinja2` on 240 real renderings |
  | elementary functions | **correctly rounded** for tables (RoPE/YaRN) | Ziv: binary64 + integer of increasing precision; validated against mpmath; digests pinned |
  | canonical encoding | **deterministic CBOR** (RFC 8949 §4.2) | `term_to_binary` is not a contract across OTP versions or across languages; certificates verifiable in Python |
  | Phoenix/Plug, Nx, Livebook | **yes, outside the core** | `integrations/` with their own dependencies; core `deps: []` |
  | Ecto, Oban, LiveView, Broadway | ***behaviours*/hooks + recipes** | `Store`, `on_event:`, cancellation by monitor |
  | AtomVM, Membrane, Riak | **no** | no *ports*/MMU; no media models; content-addressed storage serves better |

  **Findings** (each with a test):
  - Elementary functions: Ziv's fast path decided nothing for
    negative results (midpoints swapped): 27% fell into the slow
    path. `log(1)` looped forever (the dyadic exactness test
    was wrong).
  - A dead `let` loop with no outputs broke *lowering*. There is now
    dead-code elimination.
  - `transformers` 5 writes `num_local_experts` in Qwen3-MoE. DeepSeek's latent
    norms use ε = 10⁻⁶, not `rms_norm_eps`.
  - Server: a constrained request brought down the engine and hung
    forever. Now the engine is monitored and exceptions become 500s. Disconnected
    clients occupied the batch until `max_tokens`; now they are cancelled.
  - Grammar: three dead ends (an escape at the `maxLength` limit,
    lone surrogates, malformed UTF-8), found by property testing
    and by a random model in Livebook.
  - Mix ≥ 1.15 prunes the *code path*: `:inets`/`:ssl` had to be declared.
  - An old race, exposed by Elixir 1.18's *timing*: writing to a
    freshly dead worker (EPIPE) became an exit signal through the *port*'s *link*
    and brought down the engine. *Port* owners now trap exits.
  - **ZK/FHE** (proposal of 2026-10-01, scrutiny in [ZK_FHE.md](ZK_FHE.md)):
    zkVMs are RV32IM, so "RVV for zkVM" falls. Floating point is not
    field arithmetic, and the integer fragment is, with the bridge proved in
    Lean. With public weights, receipts already verify by re-execution, and
    ZK only adds privacy and succinctness. Built: fields/NTT,
    R1CS → Groth16 → EVM (209,922 execution gas measured, against the
    "< 200k" promised) and polynomials with proved error. Refused: CKKS/BFV
    without an audit. The independent review found two soundness holes in the
    circuit, both fixed and tested: the private input had no
    range check (any output could be reached), and a ReLU in a
    small field could have two decompositions.
  - The independent review found more defects, all fixed and
    tested. Resumption could continue from a prefix that did not verify and
    act. An erased history left resumption in a loop. A run
    stopped by `max_steps` did not verify. Two resumers could both
    act; now the action is announced beforehand. An erased data subject got a new key
    silently. The fast path's string length was counted
    in graphemes.
  - Canonical division is not IEEE division: up to 1 ulp of difference (this was known,
    never measured against an external reference; the Nx evaluator measured it).
