# Palingenesis — renewing a model plank by plank

> `Vapor.Palingenesis` (`lib/vapor/palingenesis.ex`), `vapor palingenesis planks|try`. Tests:
> `palingenesis_test.exs`; §5m of the quality suite. The scrutiny of the request that asked for it:
> [DIRECTIVE §20](DIRECTIVE.md).

*Palingenesis* (παλιγγενεσία) is the alchemists' rebirth of a form from its own ashes. Here it
means Theseus' ship: a model renewed part by part, never retrained whole, and still the same ship,
because the record of every renewal says so.

## The hull and its planks

A model `%{spec, weights}` becomes a **hull**. Its tensors are grouped into **planks**: an
attention block (`…self_attn`: q, k, v, o and their biases), an MLP block (`…mlp`: gate, up,
down), one expert of a mixture (`…experts.3`), a norm, the embeddings. Each plank is named by the
Merkle root of its tensors (`name ‖ dtype ‖ shape ‖ SHA-256(data)`, the leaves the merge receipts
already use), and the hull by the Merkle root of its planks.

```sh
vapor palingenesis planks ./Qwen2-0.5B        # every plank, its tensor count and its root
```

## The gates

A plank is replaced only when the replacement passes every gate, measured against the
generation it would replace:

| gate | question | measure |
|---|---|---|
| contract | is it the same part? | the same tensor names, shapes and dtypes |
| **drift** (the brake) | does the ship still behave as it did where it must? | Fisher–Rao distance `2·arccos Σ√(pᵢqᵢ)` between the old and new next-token distributions at every position of the **anchor** sequences; the maximum must stay under `ε` |
| **target** | does the plank do what it was brought for? | bits per token on the **target** sequences, old against new, sequence by sequence; the paired sign-flip test of the mean difference (exact up to 16 sequences) must reject "no difference" at `alpha`, with the mean gain positive |
| invariants | does it still keep its promises? | checks the caller supplies, run on the candidate model |

The brake and the target pull in opposite directions, and the gates admit a plank only when it
improves what it claims within the stated drift budget. **The anchors are the only place the brake
looks.** Drift elsewhere is not measured, so this module does not promise "no forgetting". It
promises no more than ε of forgetting where you asked it to look.

Fisher–Rao is the right distance for the brake, for the reasons [GEOMETRY.md](GEOMETRY.md)
measured: it is a metric (KL is not), so "within ε" composes, and it is computed on output
distributions, so it does not care how the weights are parametrised (f32, bf16, or a quantised
store).

## Generations and readers

The current generation of a hull is published with `:persistent_term`: a reader takes it with one
lock-free, zero-copy lookup. A generation is an **immutable value**. A reader holding generation
N keeps exactly N's weights for as long as it holds it, while N + 1 is published beside it; N is
reclaimed by the garbage collector when its last reader lets go. That is read-copy-update, and on
the BEAM it needs nothing but immutability: there is no epoch counter and no hazard pointer.
Writers are serialised per hull (`:global.trans/3`), so two proposals cannot both build on N.

An engine serving from a native worker's shared memory sees the new generation when it opens its
next session, not in the middle of one; the swap is at the BEAM level.

## Lineage is identity

Every generation carries a record: its number, its root, its parent's root, the plank, the mode,
the alignment (if any), every gate's measurement, and the hash of the previous record. Records can
be signed (Ed25519). `Palingenesis.verify/2` re-derives the chain and recomputes the current root
from the weights:

- a record that does not name the one before it is refused;
- a record whose parent is not the previous root is refused;
- weights that do not hash to the recorded root are refused;
- with `trusted:`, every record must carry a trusted signature.

Every plank may be new, and the chain of records still says it is the same ship and how it became
this one. `vapor palingenesis try … --out DIR` writes the accepted model and `lineage.json`.

## Alignment, said precisely

The request that asked for this module called for Git Re-Basin (permutation alignment, Hungarian
method) "to eliminate destructive interference". For a **whole-block replacement** that is
unnecessary. Each hidden unit of a SwiGLU block is computed and consumed inside the block, so a
block with its units permuted is the same function. The tests measure it: a maximum drift of
4.9·10⁻⁷, rounding of the contraction order, against 0.59 for a random change of σ = 0.05.

Alignment matters when a plank is **blended** with the old one (`mode: {:blend, t}`). Averaging
unit i of one block with unit i of another averages unrelated units unless the second block is
first permuted onto the first. Then the new plank is aligned by the exact Hungarian method
(`Vapor.Merge.Align`). With `align: false`, the control, the unaligned blend's mean drift is more
than three times the aligned one's.

## Measured

On the tiny Qwen2 of the tests (random weights, 2 layers): the MLP plank of layer 1 is corrupted
(σ = 0.2), and its clean version is proposed back. Targets are 16 samples of the clean model at
temperature 1, which no model predicts better in expectation.

| proposal | gates | outcome |
|---|---|---|
| the clean plank, ε = π | target: mean gain positive, sign-flip p ≤ 0.05 | admitted |
| a sham plank, the same norm of change | target: no significant gain | refused at the target gate |
| the clean plank, ε = 0.5 | drift max ≈ 3.0 | refused at the brake |
| the clean block, units permuted | drift max 4.9·10⁻⁷ | admitted |

In a tiny random network every plank is entangled with everything, so restoring one costs a drift
near π on the anchors. This is why the brake's case and the target's case are measured
separately and not as one scenario that passes both. In a trained model, where planks specialise,
the two can coexist: this is what `vapor palingenesis try` measures on your checkpoints.

## What is not claimed

- Not measured on production models (Qwen 2.5 7B/14B, DeepSeek-V3). The request named them; this
  machine has neither their weights nor the memory, and nothing here is extrapolated to them.
- Not "the end of catastrophic forgetting". The brake bounds drift on the anchors only.
- Not a fine-tuning method. Planks come from somewhere else (a donor checkpoint, a LoRA merged into
  its block, a training run); Palingenesis decides whether they may come in.
- Not mixture-of-experts surgery yet. An expert is a plank, but adding experts (growing the router)
  changes the contract and is refused.
