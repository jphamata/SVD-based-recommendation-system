# ZK and FHE — scrutiny of the proposal and what was built

The proposal (attachment of 2026-10-01) asks to turn vapor “into the
cutting-edge compiler for zkML and fheML”, in a catalogue of ten items and a
table of priorities. Since the directive itself is subject to scrutiny,
this document separates three things: what is **fact**, checked against a source;
what is **analogy**, sometimes useful and sometimes misleading; and what was
**built and tested** here. The sources are at the end.

## 1. The first-principles question

**What does a zero-knowledge proof add to what vapor already gives?**

vapor already makes any output **re-derivable**. The receipt (`x-vapor-receipt`)
binds model, prompt, parameters and output, and whoever has the weights redoes the computation
and gets the same bits on any substrate. That is verifiability by
re-execution. A ZK proof adds exactly two things, and only they
justify the cost:

1. **Privacy for one of the parties**: the input (or the weights) is not revealed.
2. **Succinct verification**: verifying costs much less than re-executing,
   for example in a smart contract.

When neither of the two is needed, a ZK proof is cost without benefit.
The receipt is enough.

**What does FHE add?** A property that no other piece of vapor
gives: the server computes on data it **never sees**. Homomorphic
evaluation, however, is **exact modular integer** arithmetic. It is already
deterministic by nature, and so vapor's central differentiator
(identical bits on every substrate despite floating point) carries little weight
there. What vapor can offer FHE is something else: certified kernels
(NTT), polynomial approximations with **proved** error, and the discipline of
refusing what it cannot guarantee.

## 2. Verdict per item

| item of the proposal | verdict | why |
|---|---|---|
| 1.1 “optimise the RVV emission for SP1/RISC Zero” | **false premise** | zkVMs execute **RV32IM**: 32-bit integer, with no vectors and no floating point (RISC Zero's specification). vapor emits RV64GCV with RVV. It would be a **new** backend (RV32IM, without `V`, with f32 emulated in software, which is very expensive inside a proof). The idea that remains is good and points in the right direction: **separate witness and proof**. It was done, but by another path (§3.2). |
| 1.2 R1CS/Plonkish arithmetisation of `add`, `mul`, `fma`, `linear` | **done for the integer fragment** | Floating point is **not** field arithmetic. Each f32 rounding requires bit decomposition and a range proof, hundreds of constraints per operation. vapor's `gemm_i8`, on the other hand, is integer arithmetic, and its absence of *overflow* is **proved in Lean**. That is the fragment that becomes cheap constraints. |
| 1.3 BabyBear, Goldilocks, BN254 fields | **done (exact reference)** | `Vapor.Field`, with primality, 2-adicity and generators checked. SIMD kernels with Montgomery/Barrett are future work and will have this reference as their oracle. |
| 1.4 prove in Lean “Circuit(x,y)=1 ⟺ y=Oracle(x)”, “would eliminate 100% of the risks” | **partly done; the promise is overstated** | Proved: the lemma that links the integer bound to equality in the field (§3.3). The correctness of the *circuit* with respect to the specification is one part of the risk. The rest remains outside: the security of the proof system, the *setup* ceremony, bugs in the prover and the verifier, and the question “are these the right weights?”. No circuit proof eliminates 100% of that. |
| 1.5 Solidity verifier “< 200k gas” | **done and measured: the promise does not hold with the standard verifier** | The Groth16 verifier exported by snarkjs, deployed on an in-memory EVM, spent **209,922 execution gas**, about 237 k with the transaction, for 3 public inputs. The floor set by the *precompiles* is ≈ 181 k + 6 k per public input. Below 200 k only with a lean verifier and few inputs. |
| 2.1 CKKS/BFV as terms of the algebra | **not now** | Implementing an encryption scheme without an audit and without the parameter choices of the *Homomorphic Encryption Standard* would be a dangerous toy. Mature libraries (OpenFHE, Lattigo, TFHE-rs) exist. vapor's place is **beneath** them (kernels, approximations), not in their place. |
| 2.2 NTT on AVX-512 and RVV | **reference done; the kernel is the next step** | `Vapor.NTT` (cyclic and negacyclic, `Z_q[X]/(Xᴺ+1)`) matches the naive product over four fields. It is the **common denominator** of FHE and STARK provers, hence the kernel investment with the widest reach. The sentence “90% of the time is NTT” varies with the scheme and the operation; it was not measured here. |
| 2.3 Wilkinson envelope = noise budget | **analogy, not reuse** | The envelope bounds rounding error **deterministically**. CKKS noise is born **random**, from encryption, and is analysed with worst-case bounds (very pessimistic) or average-case bounds (heuristic). What can be reused is the *infrastructure* for propagating bounds through the term DAG, with new transfer functions. |
| 2.4 activations by Chebyshev; “Canon already decomposes into polynomials” | **done, with a certificate; the premise about `Canon` is false** | The canonical functions use comparison and selection (`sel`), range reduction and Newton. They are **not** pure polynomials. Approximating by Chebyshev is the right step. What was missing, and was done, is to **prove** the error (§3.4). |
| 2.5 *bootstrapping* inserted by the *cut sweep* | **analogy** | The *cut sweep* cuts regions by register pressure. Placing *bootstrapping* is level management (the problem of compilers such as EVA, HECATE or Fhelipe). The formalism of graph cuts may inspire; it is not the same algorithm. |
| 3 zkFHE for 70 B | **research, not a roadmap** | See §4 for the order of magnitude: just proving the in-the-clear inference of a 13 B model already takes minutes of GPU. |

## 3. What was built

### 3.1 Fields and NTT (`Vapor.Field`, `Vapor.NTT`)

BabyBear (2³¹ − 2²⁷ + 1), Goldilocks (2⁶⁴ − 2³² + 1), the BN254 scalar and
998,244,353 = 119·2²³ + 1. The tests check:
- primality (Miller–Rabin), 2-adicity and that the generator is a non-residue, which
  implies that the 2ˢ-th root has exact order;
- that the NTT inverts and that NTT products = naive products, cyclic and
  **negacyclic** (the ring of BFV, CKKS and ML-KEM);
- that the signed embedding is injective below p/2 and collides just above.

### 3.2 Integer inference as R1CS (`Vapor.ZK`)

An int8 network (`linear → ReLU → linear`) is compiled to R1CS over
BN254. The **first layer's witness is computed by vapor's certified
`gemm_i8` kernel**, on any substrate and with the same
bits; the circuit only checks it. This is the witness/proof separation that
item 1.1 was after, without needing a zkVM. The files come out in iden3's
binary formats (`.r1cs` and `.wtns`), accepted by the
circom/snarkjs ecosystem.

Three design choices:
- **Public weights are circuit constants.** Multiplying by a constant
  is free in R1CS (it is part of the linear combination). A hidden dense
  layer **costs no constraint at all**, and the last one costs one per output.
  The cost of the proof is in the non-linearities (ReLU costs a bit
  decomposition, `B + 3` constraints per neuron) and in the range check of the
  inputs (9 per private int8 input). It is the **inverse** of the CPU's cost
  structure.
- **The private input is int8 because the circuit says so.** Each `x + 128` is the
  sum of 8 boolean bits. The first version did not have this check, and the
  independent review showed the effect: the prover could choose
  any field value for `x`, reach any hidden activation and,
  with it, any output. The statement proved is now exactly
  **∃ x ∈ int8ᵏ : model(x) = y**.
- **Private weights are not offered.** With weights as a private
  witness, the proof says only that *there exist* weights that produce the output.
  That is useless without a commitment, inside the circuit, to a published
  model. And even a commitment says *which* weights, not that they are the
  announced model: the *Hollow-LLM* attack (2026) shows “hollow” weights that
  pass the verification of a larger model. Constants in the circuit
  pin down the model: the verification key derives from the circuit, and
  `ZK.digest/1` names the weights.
- **Refusal of what the field cannot represent unambiguously.** The compiler
  computes the exact bounds and refuses in two cases: when an intermediate
  value can reach p/2, and when the decomposition of a ReLU
  would need more bits than the field distinguishes (`2^(B+1) > p`, in which case
  a second representative of the same value would also fit). On BN254,
  int8 contractions stay very far from both limits. On BabyBear, an
  int8 contraction with K ≈ 70,000, or a ReLU over values near 10⁹,
  are already refused (tested).

Numbers from this machine (`zk_test`, level `:snarkjs`):
- 16 → 8 → 3, int8 with private input: **311 constraints, 304 wires**
  (144 of them are the range check of the 16 inputs);
- `snarkjs wtns check` accepts the witness;
- Groth16 proof generated and verified, and the same proof with a forged
  output is rejected;
- the Solidity verifier compiles (1,721 bytes of EVM) and spends **209,922 execution
  gas** (236,582 with the transaction), measured on an ethereumjs EVM;
- changing the value of any single wire breaks some constraint, and an
  input outside int8 has no witness. This is an empirical test, not a
  proof. The formal guarantee against under-constraint is that of the statement above:
  once the inputs are fixed, each internal wire is determined (unique bits below
  p, ReLU and linear layers as functions of the earlier wires). Different
  inputs with the same output remain possible, and that is inherent to the
  statement “there exists x”.

### 3.3 The lemma in Lean (`proofs/Vapor/FieldEmbedding.lean`)

```
embed_injective   : p ∣ a − b, |2a| < p, |2b| < p  ⟹  a = b
field_parity      : |dot| ≤ K·A·B, 2·K·A·B < p, p ∣ z − dot, |2z| < p  ⟹  z = dot
machine_field_agree : integer_parity ∧ field_parity  ⟹  32-bit machine = integers = field
```

The statements above are in mathematical notation. In Lean, each
`|2x| < p` is the conjunction `-p < 2 * x ∧ 2 * x < p`.

Together with the already existing integer parity theorem, this closes the chain
*certified kernel → integers → field*. The lemma's hypotheses (int8
inputs, bounds below p/2) are **facts of the circuit**: the range check
guarantees the first, and the compiler refuses what would violate the second. That is
why, for a given input, the value the worker computes is the one the
circuit enforces. The Lean core remains without
Mathlib, without `axiom`, without `sorry`, now with 45 theorems, and the
extracted module was regenerated with the new digest.

### 3.4 Polynomials with proved error (`Vapor.Poly`)

Under CKKS, every non-linearity becomes a polynomial. Common practice is to approximate
and **measure** the error on samples. `Vapor.Poly` approximates by Chebyshev and
**proves** the error in exact rational arithmetic:
- it evaluates `p` exactly on a dyadic grid;
- it brackets `f` with the correctly rounded exponential from `Vapor.CR`;
- between grid points, it uses the interpolation remainder `h²/8 · sup|e''|`,
  with `sup|p''|` bounded by Markov's inequality.

| function | interval | degree | **proved** error | observed error | depth |
|---|---|---|---|---|---|
| sigmoid | [−8, 8] | 7 | 2.9642·10⁻² | 2.9640·10⁻² | 3 |
| sigmoid | [−8, 8] | 15 | 1.388·10⁻³ | 1.382·10⁻³ | 4 |
| tanh | [−4, 4] | 15 | 2.776·10⁻³ | 2.765·10⁻³ | 4 |
| exp | [−1, 1] | 8 | 1.47·10⁻⁸ | 1.22·10⁻⁸ | 4 |

The bound is valid over the **whole** interval, not only at the samples, and is at
most 1.21× the observed error. The grid has 4,096 points, except in the
exp row, which uses 32,768 because a very precise approximation needs
a finer grid for the remainder term not to dominate. The “depth” is
⌈log₂(d+1)⌉, the minimum number of levels to multiply ciphertexts together. A
real CKKS evaluation usually spends one more level on the constants.

## 4. Order of magnitude (to calibrate the ambition)

- **zkLLM** (CCS 2024) proves the inference of LLaMA-2 13B in **803 s** on an
  A100. The proof is 188 kB and verification takes 3.95 s. The commitment to the
  weights takes 986 s and occupies 11 MB.
- **Hollow-LLM** (2026) shows that a ZK proof of inference certifies
  consistency with committed weights, not that the committed model is the
  announced one.
- **Groth16 on Ethereum**: the floor set by the cost of the *precompiles*
  (EIP-1108) is about 181 k + 6 k·ℓ gas, with ℓ public inputs. The
  snarkjs verifier measured 210 k of execution here.

“The world's definitive platform for confidential and verifiable AI” is not a
claim this repository can sustain. What it does sustain, with
tests, is smaller and more useful: an exact, proved, end-to-end path
from a certified int8 kernel to a Groth16 proof verifiable on the EVM, and
the base pieces (fields, NTT, polynomials with proved error) on which
the rest can be built without losing the guarantee.

## 5. Next steps, in order

1. **Certified NTT kernel** (AVX-512, NEON, RVV, SPIR-V), with
   `Vapor.NTT` as the bit-for-bit oracle. It has the widest reach: FHE and STARK.
2. **Requantisation** (`s32 → s8`) as a gadget and as a term of the algebra. It
   allows deeper integer networks with a certified witness in
   every layer.
3. **PLONK/STARK backend** (instead of Groth16, which requires a per-circuit
   *setup*) and a Poseidon commitment that binds the circuit to the digest of the
   model's certificate.
4. **Worst-case BFV noise bounds** propagated through the DAG, the honest part
   of item 2.3.
5. RV32IM as a zkVM target only if a concrete case calls for it. The direct
   constraints (§3.2) are cheaper than emulating a CPU.

## Sources

- RISC Zero, *zkVM Technical Specification* — “The zkVM implements the RV32IM instruction set”: https://dev.risczero.com/api/zkvm/zkvm-specification
- Sun et al., *zkLLM: Zero Knowledge Proofs for Large Language Models* (CCS 2024): https://arxiv.org/pdf/2404.16109
- Gong, Liu, Li, *Hollow-LLM Attack: Computationally Trivial Weights in Zero-Knowledge Verification of LLM Inference* (2026): https://arxiv.org/html/2607.28884v1
- Nebra, *Groth16 Verification Gas cost*: https://hackmd.io/@nebra-one/ByoMB8Zf6
- iden3, `r1cs`/`wtns` binary formats (circom/snarkjs), checked by `snarkjs r1cs info` and `wtns check` themselves in the tests.
