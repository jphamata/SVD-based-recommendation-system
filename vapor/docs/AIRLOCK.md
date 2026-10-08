# The model airlock (`Vapor.Lock`)

> The core does not know what a Llama is. It knows what a **contract** is.

## 1. The problem

Up to the previous round, the engine, the embedder and the loader called
`Vapor.Model.Decoder.program/3` directly, and `Vapor.Model.Config` enumerated
eight families with their details. Each new model required editing the core. This is
the industry pattern — transformers' per-family `modeling_*.py`, llama.cpp's
`llm_build_*`, vLLM's `*_model.py` — and it is the real pain:

- **coupling**: the server, the scheduler and the cache know architecture
  details (tensor names, head *layouts*, MoE routing);
- **high marginal cost**: a family that is "Llama with other names"
  (Phi-3, EXAONE, …) costs a whole code file, review and release;
- **silent failure**: what the layer does not understand is often ignored
  instead of refused (see the GGUF exporter bug in §7).

## 2. The idea, from first principles

A program in vapor's algebra is already **self-describing**: its inputs, outputs
and state are typed terms. So what the core needs to know about a model
is not the family, it is the **shape of the program**. The airlock is the only boundary where
a family is known; on the inside there are only:

- a `Vapor.Lock.Spec` — the contract (`interface`), widths, vocabulary,
  EOS, input and output modalities, *features* the builder accepts, and a
  `digest` of the admitted configuration. The family-specific configuration goes
  in `config`, **opaque** to the core;
- a `Vapor.Program` built by the adapter and **checked against the contract
  in the airlock itself** (`Vapor.Lock.Contract`) before the core sees it.

```
checkpoint ─► manifest ─► claim (all adapters) ─► admit ─► Spec + weights
                                                              │
                core ◄── Contract.check ◄── build ◄───────────┘
```

| contract | inputs | outputs | state |
|---|---|---|---|
| `:causal_lm` | `tok, pos : s32[t]` (+ `last`, `sampling`, `table`, `slot`, `soft`, `soft_mask` by option) | `logits : f32[·, vocab]` and/or `hidden : f32[·, width]` | `{x, x_next}` pairs with equal sorts |
| `:encoder` | `rows : f32[T, k]`, `horizon : s32[T]` | `hidden : f32[T, width]` (+ `pooled`, `logits`) | — |
| `:codec` | `rows : f32[T, k]` ↔ `codes : s32[T]` | same | — |
| `:map` | `rows : f32[T, k]` | `out : f32[T, width]` | — |

## 3. Three properties enforced, not hoped for

1. **One owner per checkpoint.** Every adapter answers `claim/1` with
   `{:claim, score}`, `{:near, why}` or `:no`. The highest score wins;
   adapters registered at run time come before the built-in ones, so
   an override is explicit. Nobody claims → a typed refusal that lists
   **all the near-misses with the fix** (e.g.: "the tensors follow the
   decoder layout, but `model_type "phi9"` is not a known family — register an
   alias whose `like` is the closest family").
2. **Contract at the airlock.** A defective adapter is stopped at the boundary
   (`Rejection` with `{:contract, name}`), not inside the engine.
3. **No family in the core — tested.** `lock_test.exs` reads the atom table
   of the compiled BEAM of the core modules (engine, embedder, server,
   speculation, RAG, merging, hub, quality, compiler, runtime, ladder,
   emitters) and fails if any of them references `Vapor.Model.Decoder`,
   `Vapor.Model.Config`, `Vapor.Model.GGUF` or any adapter. The rule is
   a test, not a convention.

## 4. Adding a model: three levels, from cheapest to most expensive

### Level 1 — alias (data only, nothing compiled)

For families that *are* an existing topology under other names. A map (or
a JSON file) declares renamings of configuration keys, forced
values, required values, forbidden keys, tensor renamings by
*template* (`{l}` = layer index) and **splits of fused tensors** with
sizes expressed as products of fields of the admitted configuration:

```json
{"id": "phi3", "model_type": "phi3", "like": "llama",
 "config": {"set": {"attention_bias": false},
            "require": {"partial_rotary_factor": [null, 1, 1.0]}},
 "tensors": {"split": [
   {"from": "model.layers.{l}.self_attn.qkv_proj.weight",
    "into": [["model.layers.{l}.self_attn.q_proj.weight", "heads*head_dim"],
             ["model.layers.{l}.self_attn.k_proj.weight", "kv_heads*head_dim"],
             ["model.layers.{l}.self_attn.v_proj.weight", "kv_heads*head_dim"]]},
   {"from": "model.layers.{l}.mlp.gate_up_proj.weight",
    "into": [["model.layers.{l}.mlp.gate_proj.weight", "intermediate"],
             ["model.layers.{l}.mlp.up_proj.weight", "intermediate"]]}]}}
```

This is the built-in **Phi-3/Phi-4** alias. Registration:
`Vapor.Lock.register(map)`, `Vapor.Lock.register_json("arquivo.json")` or
`VAPOR_LOCK_ALIASES=a.json:b.json` in the environment. Admission runs the
base family's checks on the rewritten configuration, so an alias can **narrow**
what the base accepts, never **widen** it (LongRoPE and `partial_rotary_factor ≠ 1`
are refused by name). Splits that do not cover the rows exactly are
refused. Chains of `like` are followed (up to 8; loops are refused).

Verification: a Phi-3 checkpoint assembled by fusing Llama weights is admitted, the
tensors come back **identical** to the originals and the program gives **the same bits**
as Llama; and an independently written NumPy reference (slicing of
`qkv_proj` and `chunk(2)` of `gate_up_proj` as in HF) agrees with relative
error < 2·10⁻⁶ (`test/python/np_reference.py`).

### Level 2 — blueprint (a few lines of code)

For families that are an existing topology **with extra knobs**. The
adapter only translates the configuration into the knobs. The built-in example is
**IBM Granite 3.x** (`Vapor.Lock.Adapters.Multipliers`): `embedding_multiplier`,
`attention_multiplier`, `residual_multiplier` and `logits_scaling` become the
decoder's `embed_scale`, `attn_scale`, `residual_scale` and `logit_divisor` (the
last two are new knobs in this round, in `Vapor.Model.Config`). The
original configuration is kept, so `config.json` comes back as Granite.
Verified against an independent NumPy reference (< 2·10⁻⁶).

### Level 3 — topology (a new program, no new kernel)

For genuinely new program shapes. Built in this round:

- `Vapor.Lock.Adapters.Encoder` — bidirectional encoder over rows (patches,
  frames): Hugging Face ViT (`model_type: "vit"`, with or without the
  `vit.` prefix, classification head) and `vapor_encoder` (the same topology
  described directly, LayerNorm or RMSNorm). Verified against a NumPy
  reference that computes the **convolution as a convolution** (< 2·10⁻⁶) and bit for bit
  between native and oracle;
- `Vapor.Lock.Adapters.Codec` — VQ codec (`vapor_vq`): encode = argmax via
  greedy `sample`; decode = `gather_row`;
- `Vapor.Lock.Adapters.Linear` — affine projector (`vapor_linear`), the link between
  modalities.

For how each one is built only from existing operators, see
[ANY_TO_ANY.md](ANY_TO_ANY.md).

## 5. Diagnosis

```sh
mix vapor.lock ./MeuModelo            # who claims it, why, spec, missing/extra tensors
mix vapor.lock ./MeuModelo --alias meu_alias.json
mix vapor.lock --list
```

`Vapor.Lock.explain/2` returns each adapter's answer, the winner and —
when admitted — the tensors the builder expects and did not find (or
found with another shape) and the ones it will not read. It is the guide for writing
an alias: what was left over on one side and missing on the other is exactly the
renaming table.

## 6. Cost

Measured by `mix vapor.quality` ([QUALITY.md §6](bench/QUALITY.md)): admitting
costs 0.1–0.25 ms and building 0.05–14 ms on tiny models; *lowering*
(10–650 ms) dominates. The abstraction has no measurable cost on the hot path —
the program produced by the decoder adapter is **the same term** that the
direct builder produced (tested with `==`).

## 7. What this round fixed along the way

- **Silently wrong GGUF export.** `Vapor.Model.GGUF.write/4`
  wrote every family that was not `qwen2` as `llama`, silently discarding
  q/k-norm (Qwen3), experts (Mixtral, Qwen3-MoE), MLA
  (DeepSeek-V3), sandwich norms and GeGLU (Gemma 3). It is now a typed refusal
  that names what would be lost and suggests safetensors.
- **Phi-3 exported to HF** would lose the sliding window (the `to_map` of
  `llama` does not write it); now a Llama with a window and no biases is written with
  Mistral's spelling, which loads it.

## 8. Round 0.5.0: the oracle became `transformers` itself

- **Executed parity.** `test/vapor/lock_hf_test.exs` generates the checkpoints
  with `transformers` 5.18 (its own `save_pretrained`: the names, the tensor
  fusions and the configuration keys it writes) and compares against its
  forward: Phi-3, Phi-3 with partial rotary, Granite, ViT (classification
  and `ViTModel` with pooler), CLIP-vision with projection — ≤ 4.5·10⁻⁷ relative,
  identical greedy, no unused tensor.
- **The bug the NumPy reference could not find.** `transformers` ≥ 5 moves
  `partial_rotary_factor` inside `rope_parameters`. 0.4.0 checked the
  key at the top level — and admitted a Phi-4-mini-style checkpoint that then
  computed wrongly (relative error 0.5). Fix in two parts: partial
  rotary is now **built** (the rotating pairs go to the
  *rotate-half* layout through a permutation folded into the q/k rows — attention
  scores do not change under the same permutation of both — and the table has
  pass-through pairs with cos = 1, sin = 0, the same mechanism as MLA); and **every unknown
  key** of `rope_parameters` is refused by name: keys that change
  the mathematics are read or refused, never skipped.
- **New topologies:** `vapor_mlp` (the perceptron: classifier, diffusion
  denoiser, two-layer projector); the encoder gained CLIP's vision tower,
  ViT's pooler, the per-row head (`head: "rows"`, for OCR and speech
  with CTC) and shorter programs (`rows:`).
- **What to measure, stated by the airlock:** the adapter declares `taps/1` — the
  activations that feed each matrix — and least-squares merging reads
  that, without knowing what a layer is.

## 9. Round 0.6: state, parts, images

Five new topologies, all level 3 (a new program, no new
kernel), all checked against the reference implementation running:

| adapter | family | contract | checked against |
|---|---|---|---|
| `Mamba` | `MambaForCausalLM` | `:causal_lm` with the `:recurrent` *feature* (no `pos`: the state is the history) | `transformers` (3.4·10⁻⁷, identical greedy) |
| `Whisper` | `WhisperForConditionalGeneration` | `:causal_lm` (decoder) + the **part** `encoder: :encoder` | `transformers` (4.8·10⁻⁷ / 4.2·10⁻⁷, identical greedy) |
| `VAE` | `AutoencoderKL` (decoder) | `:map` | diffusers (8.7·10⁻⁷) |
| `DiT` | `DiTTransformer2DModel` | `:map` | diffusers (3.1·10⁻⁷) |
| `Encoder` (CLIP text) | `CLIPTextModel(WithProjection)`, the text half of a `CLIPModel` | `:encoder` (with `tok` and `pick`) | `transformers` (3.7·10⁻⁷) |

Two extensions of the contract, both small and checked at the boundary:

- **Parts.** A model of several programs (Whisper: encoder and decoder)
  declares in `spec.parts` the contract of each extra part;
  `Lock.build(spec, ws, part: :encoder)` checks the program against **that**
  contract, and an undeclared part is refused by name.
- **Window in every layer.** The optional `ring_window/2` *callback*
  tells `Vapor.Engine` when it can keep the pages in a ring
  (`Vapor.Lock.ring_window/2`); the engine does not read the family's configuration.
- **Recurrent.** A `:causal_lm` with the `:recurrent` *feature* does without `pos`;
  `Vapor.Engine` (whose memory model is the paged cache) refuses it by
  contract — `:paged` is missing — and `Vapor.Recurrent` serves it.

The near-miss refusals come along: `falcon_mamba` (normalises B, C and Δ
inside the mixer), `jamba`/`zamba`/`bamba`/`nemotron_h`/`falcon_h1`
(hybrids with attention; `mamba2` has its own adapter since 0.8), `PixArt`/`SD3`/`Flux` (DiTs with attention to text or
2-D rotary positions) — each one says what is missing.

## 10. What is still not guaranteed

- **EXAONE, InternLM2, Baichuan, OLMo2** and others are not built in: some
  are pure aliases (EXAONE seems to be only renamings), others are not (Baichuan2 has
  `NormHead`; OLMo2 normalises q/k over the whole projection and puts the norm afterwards).
  I did not build in what I could not verify.
- LLaVA, U-Net/ControlNet, Jamba: the path is designed and the pieces
  tested; the adapters are not (Mamba-2 came in with 0.8).
- Parity with random weights is not quality: VAE, DiT, Whisper and Mamba
  have not been measured with trained weights (they are not in this environment).
- `Vapor.Train` still speaks the decoder's language (LoRA over `%Config{}`): it is
  an adapter capability that should become an airlock *callback*.
- The manifest of an HF directory is assembled after reading the weights. For
  huge checkpoints there is now `Vapor.Lock.preflight/2` (0.17): admission
  from the configuration and the safetensors headers alone, fetched by
  ranged reads through the siphon (docs/SIPHON.md). `Lock.open` itself
  still reads the weights it builds from.

## 11. Where names live (0.17)

The question, asked of this airlock: are files like `llama.ex`, or an
adapter for Kimi K3, inevitable and correct, or should models (open or
proprietary, named by family or not) live only in the airlocks, with a
core that is pure mathematics?

Both, once two things are told apart. A checkpoint **computes a
function**, and someone has to write that function once in the algebra.
That code is inevitable, and it is mathematics. A checkpoint is also
**spelled** a certain way: a `model_type`, configuration keys, tensor
names, fused matrices, multipliers. A spelling is not mathematics, it is
data about a family, and it belongs to the airlock alone.

| tier | holds | names it may use | enforced by |
|---|---|---|---|
| **core** (algebra, compiler, emitters, runtime, engine, sampler, merge, quality, deciders) | mathematics and machinery | none: no family, no layout, no tensor name | `lock_test.exs`: the atom tables of the core's compiled modules reference no airlock module |
| **topologies** (`lib/vapor/model`, `lib/vapor/lock/adapters`) | the function, written once per *kind* of network, reading one canonical layout | named for what they compute: decoder, delta-rule hybrid, encoder–decoder, Mamba, ViT, DiT | `lock_test.exs`: no module name contains a product or family name |
| **spellings** (claims, aliases, blueprints) | how each family writes that function | product and family names: here, and only here | aliases and refusals are data (`Vapor.Lock.Alias`) |
| **agent protocols** (`Vapor.Agent.Backend.*`) | the wire format of a closed model's API | the vendor's protocol name | the one allowance in that test |

Applied in this round:

- The decoder module was correct code under the wrong name (it was named
  for Llama, and eight families read it). It is now `Vapor.Model.Decoder`.
- The Kimi K3 adapter is now the **delta-rule hybrid** topology
  (`Vapor.Lock.Adapters.DeltaHybrid`, claiming only vapor's spelling
  `vapor_delta_hybrid`). K3 itself is a built-in alias (`kimi_k3`), and
  Kimi Linear is a refusal written as data (`"refuse": "why"`, new in
  0.17) instead of a near miss hard-coded in a topology.
- Whisper's topology is `Vapor.Lock.Adapters.EncoderDecoder`, and
  Granite's blueprint is `Vapor.Lock.Adapters.Multipliers`. Each still
  claims its one family, in its own claim clause.

What remains, named so it is not forgotten: `Vapor.Model.Config` still
keys the decoder's knob readers by `model_type` (`family(%{arch:
"deepseek_v3"})`), so the decoder's eight spellings are code, not data.
The next step is blueprints as descriptors: each family's mapping from
its `config.json` onto the decoder's knobs, as an alias is today, with
`Config.families/0` becoming the registry of those descriptors.

Closed models (the K3 report's Claude Fable 5 and GPT-5.6 Sol, Google's
Astra) have no weights, so they have no topology and no spelling. They
appear only as a protocol at the agent boundary, and what they answer is
a proposal, checked like anyone's.
