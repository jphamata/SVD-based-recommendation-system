# Training and endless context (0.10)

> Request, translated: "a complete training pipeline too, with HPC"; and, in the
> round's attachment, "invert RoPE" for infinite context. Scrutiny:
> [DIRECTIVE.md §13](DIRECTIVE.md). Tests: `train_lm_test.exs`,
> `streaming_test.exs`, `cluster_test.exs` (training across nodes).

## 1. Pre-training with the bits of a single machine (`Vapor.Train.LM`)

A byte-level Llama (vocabulary 256: no tokenizer to trust), pre-trained end
to end as **vapor programs**:

- **Gradients by reverse-mode differentiation with *let-bindings***
  (`Vapor.Autodiff.grad_lets/5`). Without them, a transformer's term tree
  grows exponentially with depth (a tiny model took more than 10 minutes to
  compile); with them, 0.8 s. Checked against PyTorch's *autograd* in
  binary64, parameter by parameter: maximum relative error 7.7·10⁻⁷.
- **Deterministic data parallelism.** Each step sums the micro-batch
  gradients in a **fixed binary tree over blocks**, each block the left fold
  `((0 + g₀) + g₁) + …`, accumulated **inside the worker's resident
  session** (the state `acc ← acc_next` never goes back to the BEAM).
  The step's bits depend neither on the number of workers, nor on who did
  which block, nor on a worker dying midway (tested, and with the exact
  oracle). The shape of the tree is part of the definition: another block
  size is another run (the test that can fail does fail).
- AdamW with *clipping* by the global norm, *warmup* and cosine; RoPE via a
  signed permutation (`linear`), packed sequences with a per-block causal
  mask, cross-entropy with a closed-form seed.
- **Checkpoint and resume** with the bits of an uninterrupted run;
  **export** to Hugging Face's Llama format (`transformers` loads it and
  computes the same *logits*; so does vapor's inference stack).

**The bundled model** (`priv/lm`, `mix vapor.train`): 492,160 parameters
(d = 128, 2 layers, 4 heads), 1000 steps × 1024 tokens over vapor's own
documentation. On held-out text:

| | bits/byte |
|---|---|
| untrained | 8.558 |
| byte frequency | 5.032 |
| Witten–Bell, order 3 | 3.913 |
| Witten–Bell, order 5 | 3.415 |
| **the model** | **2.919** |

The receipt (`priv/lm/receipt.json`) keeps the curve, the digests of each
checkpoint, the SHA-256 of the corpora and the *schedule*: redoing it gives
the same digests. Greedy samples still repeat ("de a prova de a prova…"): it
is a half-million-parameter model, honest about it.

**HPC.** d128/L2, 1024 tokens per step: ~2,700 tokens/s with 1 worker and
blocks of 4 micro-batches, ~3,600 with 2 workers (2-core machine), same
digests. Compiling each program takes ~27 s (once).

## 2. Endless context: the attachment, scrutinised

The attachment proposes "inverting RoPE" — measuring positions from the
current token, compressing frequencies (YaRN), or combining attention anchors
(*attention sinks*) with a sliding window (StreamingLLM) — and claims that
vapor "already has 90%". Checked:

- **Right:** the RoPE score depends only on the distance
  (`⟨R_m q, R_n k⟩ = qᵀ R_{n−m} k`), and two things break a model beyond
  its training length — never-seen angles and growing memory. vapor has
  YaRN (`rope_tables` with `{:yarn, …}`, checked against `transformers`),
  correctly rounded RoPE tables (`Vapor.CR`), `Vapor.Engine`'s ring of
  pages and Mamba/Mamba-2 (fixed state).
- **Imprecise:** the engine's ring only holds for models whose **window**
  covers every layer (Mistral), and the positions keep growing — the RoPE
  table has `max_seq` rows, so the ring's "infinity" was bounded by the
  declared context. Exact tables avoid drift *of the angle*; they do not
  solve out-of-distribution angles.

**What was done** (`Vapor.Streaming`, `kv: {:stream, sinks, window}`):
the cache keeps keys **unrotated**, `s = sinks + window` rows (the first
ones fixed, the others a ring); at each step the rows are gathered in
logical order (`gather_row`) and rotated at the **cache's own positions**
`0 … s−1`, the query at `s−1`. Since `R_a R_b = R_{a+b}`, re-basing the
positions changes no score inside the window — and the anchors sit right
before it, at a distance the model knows. No position goes past `s`, no
table grows, memory is constant: the stream has no end. It is
StreamingLLM's re-rotation, done only with operators the airlock already
certifies (no new kernel).

Measured on the bundled model (training length 64), 900 bytes of held-out
text — 14× the training length — in a 64-row cache:

| | bits/byte after byte 64 |
|---|---|
| stream (4 anchors + 60 of window) | **3.03** |
| window without anchors | 3.05 |
| control: positions growing up to 900 | 5.81 |
| up to byte 64 (reference) | 2.53 |

And until the cache fills, the stream computes **exactly** the causal
model's *logits* (same bits). The anchors help little in this model
(trained with packed sequences at random positions, it does not concentrate
attention on the first token); in large models StreamingLLM's effect is the
documented one — measuring it requires weights this machine does not
download.

**Limits:** decoding beyond the cache is one token per step (like any
generation); the path is the dense session — integrating it into the paged
engine (pinning the anchors' pages and re-rotating per *slot*) is in the
[TODO](TODO.md). Training models with long context (and the
*needle-in-a-haystack* evaluation) has not been done.
