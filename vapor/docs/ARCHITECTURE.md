# vapor — architecture

This document describes the system as it is implemented and tested in this
repository: data flow, IPC protocols, memory model, numerical model,
verification ladder, and what is proved, what is tested and what is
trusted. The last section records where the original directive was
refined by technical scrutiny, and the limitations that remain.

## 1. Overview

```
                    ┌────────────────────────── BEAM (control plane, deps: []) ──────────────────────────┐
  Program ──▶ Rung 1 (sorts) ──▶ Rewrite (ε=0) ──▶ Lower + cut sweep ──▶ KIR ──▶ ISA selection
   (symbolic     │                                   │                          │
   terms)        │                                   ▼                          ▼
                 │                          SPIR-V (assembler)      liveness ─▶ linear scan (g groups)
                 │                                   │                          │
                 │                                   │              checker extracted from Lean (accepts?)
                 │                                   │                          │
                 │                                   ▼                          ▼
                 │                            .spv modules     x86-64 AVX2 · AVX-512 · AArch64 · RV64GCV (bits)
                 ▼                                   │                          │
   Rungs 2–5: admission · adjoint · oracle · parity + envelope (real execution on the substrates)
                 │
   Rung 6: Ed25519 certificate (deterministic, co-signable) ──▶ Bundle ──▶ edge node (verifies only)
                 │
   Vapor.run ──▶ arbiter (3 ceilings, declared profiles) ──▶ Dispatch with failover
                 └───────────┬───────────────────────────┬───────────────────────────┬───────────┘
                     {packet,4} stdio            {packet,4} stdio                   (pure)
                             ▼                           ▼                             ▼
                 vapor-worker (process)         vapor-fabric (process)            Oracle (BEAM)
                 seccomp · W^X · watchdog       full Vulkan compute               exact binary32
                 thread pool · sessions         lavapipe / real GPU
                 native | RVV interpreter
```

Nothing generated executes inside the BEAM. There is no NIF (the audit test
forbids `load_nif`/`@on_load`); the only native code the BEAM loads is the
VM's own.

| Component | Where | Lines (0.13) |
|---|---|---|
| Core: algebra, compilation, emitters, verification, runtime, certificate, substrate airlock | `lib/vapor/{algebra,compile,kir,emit,verify,runtime}` and top-level modules | 12,853 |
| Code extracted from Lean (generated) | `lib/vapor/extracted.ex` | 166 |
| Worker, thread pool, counters, RVV interpreter, sandbox, Vulkan and Metal daemons | `native/src/` | 4,717 |
| Lean 4 proofs + extractor | `proofs/` | 1,515 |
| All the layers (models, documents, studio, desks, finance…) | `lib/` | ~78,000 |
| Tests (Elixir, Python and Node scripts of the differential levels) | `test/` | ~21,000 |

The ecosystem built on the core (Llama/Mistral/Qwen2 models,
tokenizer, generation engine, OpenAI server, autodiff/LoRA, safetensors/GGUF
ingestion) and its scrutiny are in
[ECOSYSTEM.md](ECOSYSTEM.md); §4.6 below describes the forms of
parallelism and concurrency and the invariant each one preserves.

## 2. Algebra and programs

`Vapor.Algebra.Term` is a *symbolic* free algebra: every operator is a name
with fixed semantics (`Vapor.Runtime.Oracle`). The predecessor kept Elixir
closures inside the terms, which no emitter can lower to machine
code.

- Generators: `input`, `const`, `ew` (ι: `add sub mul fma neg relu`, with
  `splat`), `qgemv` (contraction with an `:sb4` matrix), `gemm_i8` (contraction in
  ℤ/2³²ℤ).
- **Semi-dynamic dimensions**: `{:dyn, :S, max}`. All bounds are
  established at the maximum; the monotonicity lemmas
  (`admissible_mono`, `withinEnvelope_mono`) guarantee that the certificate holds
  for every `S ≤ max`. `Compiled.dims/2` rejects extents above the maximum.
- **Recurrent programs** (`Vapor.Program`, `state: [h: :h_next]`)
  realise the `scan` generator over the affine monoid σ = 2 without materialising the
  sequence: the whole loop runs in the worker behind **one** BEAM
  crossing, with per-token streaming output.
- Numerical data live in contiguous binaries (`Vapor.Tensor`). There is no
  `Enum.at` in the code; where there is random access a tuple is used (O(1)); the
  lists appear only in transient sequential traversals of the oracle.

## 3. Compilation

### 3.1 Rewriting (ε = 0)
`Vapor.Compile.Rewrite` admits only identities that are valid bit for bit for all of
IEEE-754, signed zeros included: `neg(neg x) → x`, `x·1 → x`,
`x + (−0) → x`, `x − (+0) → x`, but **not** `x + (+0) → x` (because
`(−0) + (+0) = +0`). Constant folding evaluates the declared semantics
itself.

### 3.2 KIR, selection and the group factor
The kernels (`Vapor.KIR.Kernels`: fused `ew`, reductions, `gemv_f32` and
`gemv_bf16`, `sb_sums`, `gemv_sb4`, `gemm_i8`, and the model operators —
`gather_row`, `rope`, contiguous and paged `kv_write`, contiguous and paged
`attention`, `sample`, `transpose`) are written once in portable IR over virtual registers
with *types* (`:strip`, `:f16l`, `:i32acc`, …). Each backend sizes the
types. The **group factor `g` generalises RVV's LMUL to every ISA**:
an AVX2 strip with `g = 4` is four ymm in lockstep; an RVV strip with
`g = 8` is an m8 group.

| type | RVV | AVX2 | AVX-512 | NEON |
|---|---|---|---|---|
| `:strip` | m`g` | `g` ymm | `g` zmm | `g` q |
| `:f16l` (16 f32 lanes) | m4, vl=16 | 2 ymm | 1 zmm | 4 q |
| `:i32acc` | m`4g` | `g` ymm | `g` zmm | 2 q |

The AVX-512 backend (`Vapor.Emit.X86.AVX512`, level x86-64-v4) reuses the
integer side of AVX2 and encodes everything else in EVEX: 32 zmm registers
(zmm31 as scratch), the canonical 16-lane width in a single register,
strip tails done by one masked pass (`bzhi` → `k2…k5`, only masked
loads/stores: inactive lanes compute and are discarded) instead
of a scalar loop, and 4-byte constants read with embedded broadcast
`{1to16}`. Selects become `vcmpps → k1` + `vblendmps`; the reduction tree is
the same as AVX2's lane by lane, so the bits are the same.

Selection (`select/2` in each backend) happens **before** allocation, as
in production compilers; "lanewise" instructions are emitted as *bundles*
(one allocation unit), which lets destination and source share
registers when the source dies there.

### 3.3 Linear-scan allocation without spill, verified
`Vapor.KIR.Liveness` computes liveness by fixed point over the CFG (values
live on the back-edge cover the whole loop); `early clobber` models the overlap
rules of RVV's widening instructions. `Vapor.KIR.RegAlloc`
does linear scan with aligned blocks of size 1/2/4/8 (preference:
caller-saved, then occupied "buddy", then ISA order).

There is no spill path. If registers run out, the allocator returns the pressure
point and: (a) the compiler tries a smaller `g`; (b) `gemv_sb4` tries fewer
rows per iteration (R ∈ {4, 2, 1}); (c) the **cut sweep** closes the fused
region there. Every accepted allocation is revalidated by `check_alloc/3`,
**extracted from Lean** and proved correct (`checkAlloc_sound`): aligned
groups, inside the register file, outside the reserved ones, and no pair of
simultaneously live values shares a register. So each emitted region is
free of spill and of clobber by construction, not by assumption about the
heuristic.

### 3.4 Pure binary encoding
- x86-64: REX, 3-byte VEX, EVEX (AVX-512, always disp32 — never the
  compressed disp8·N), ModR/M, SIB; SysV ABI (`rdi = args`, callee-saved
  saved only if used, `vzeroupper; ret`).
- AArch64: A64/NEON words; AAPCS64 (`x0 = args`, `x19–x28` and `d8–d15`
  saved if used; `x16`/`v31` scratch; `x18` never allocated).
- RV64GCV: `vtype = vlmul[2:0] | vsew[5:3] | vta | vma`; vector
  loads/stores in LOAD-FP/STORE-FP with a width field; scalar FP with a static
  RNE mode; full psABI with `ret`; conditional branches as
  `b<inverse> +8; jal` (±1 MiB range regardless of size).
- SPIR-V: symbolic assembler that interns types/constants, orders the logical
  sections and emits words; `NoContraction` on every `FMul/FAdd/FSub` under the
  canonical policy.

The tests validate every emitted instruction — every kernel constructor, in the
four backends, at every group factor and policy — against GNU binutils
(x86 incl. EVEX, AArch64 and RISC-V with `rv64gcv`) and every SPIR-V module against
`spirv-val`; the product never invokes these tools.

## 4. Substrates, isolation and protocols

### 4.1 `vapor-worker` (Substrate I)
Static Zig executable, no libc (240–340 KB). HELLO fixes the number of
threads: the pool (`pool.zig`) and the event counters (`perf_event_open`:
cycles/instructions/cache when there is a PMU; task-clock, page faults and context
switches always) are created **before** the seccomp filter, installed with
`TSYNC` on all threads. Each call carries a *partition
descriptor* (count argument, grain, pointers that advance per unit,
per-thread scratch, optional guard) and the worker splits it among the threads;
since the units are independent rows, the result is the same for 1…N
threads. The baton hand-off is bounded busy-waiting (2 ms) and then futex.
Programs stay **resident** in sessions (OPEN/STEP/CLOSE): weights
mapped once, state (KV caches) kept between steps, each step
writes only the given inputs and returns only the requested outputs. Per RUN or STEP:

- **native**: the blob is copied to an RW page, the page becomes R+X (W^X),
  instruction cache synchronised (AArch64/RISC-V), called as
  `void k(const uint64_t *args)`;
- **emulated**: the RV64GCV blob is interpreted by `rvemu.zig`: every memory
  access is checked against the bound buffers, the fetch against the blob, and
  an unknown instruction becomes `IllegalInstruction` with pc and word.
  Faithful RVV 1.0 semantics (RNE, fused `vfmacc`, canonical NaN, NaN-boxing,
  group alignment, `vill`), with a *poison* mode that writes 1s into
  tail/mask-agnostic elements — proof that the kernels do not depend
  on them. Configurable VLEN (128–512).

Containment: seccomp-BPF (allowlist: read, write, openat, close, lseek, mmap,
munmap, mprotect, clock_gettime, setitimer, signals and exit; architecture
audited), `PR_SET_NO_NEW_PRIVS`, and a `SIGALRM` watchdog for native
code that does not terminate. The tests provoke the four classes of failure
(SIGILL, SIGSEGV, SIGALRM, SIGSYS): in all of them, only the worker dies, the
`GenServer` that owns the port reports `{:worker_crashed, {:signal, …}}`,
respawns and the next unit runs.

### 4.2 `vapor-fabric` (Substrate II)
Zig daemon (libc only for the loader's `dlopen`), hand-written Vulkan
bindings. Full headless pipeline: instance → physical device with a
compute queue → logical device → buffers → `vkCreateShaderModule` with the
SPIR-V emitted by the BEAM → descriptor set layouts → pipeline layouts with
push constants → `vkCreateComputePipelines` → descriptor pool/sets → one
command buffer per RUN (dispatches, barriers, per-iteration windows, state
copies, staging of the emissions) → submit → fence with deadline → readback.
A driver crash or `VK_ERROR_DEVICE_LOST` kills only the daemon; a fence
timeout is treated as device lost.

### 4.3 Protocol (both processes)
`{packet, 4}` frames on stdio (big-endian length); inner fields
little-endian. A process that dies in the middle of a frame cannot
desynchronise the BEAM.

```
HELLO   1 | flags:u32 | threads:u32      → 1 | arch | sandbox | version:u32 | threads:u32
RUN     2 | mode:u8 | vlen:u32 | flags:u32 | fuel:u64 | deadline_ms:u32
          | code_len:u32 | code
          | nbuf:u32 | (kind:u8 writable:u8 len:u64 [data | path_len:u16 path offset:u64])*
          | ncall:u32 | (entry:u32 nargs:u32 (0 imm:u64 | 1 buf:u32 off:u64
          |                                    | 2 buf:u32 base:u64 stride:u64)*)*
          | iters:u32 | nemit:u32 (buf base stride len)* | ncopy:u32 (src soff dst doff len)*
          | nret:u32 (buf)*
EMIT    3 | t:u32 | bytes                  (per iteration, streamed)
DONE    4 | elapsed_ns:u64 | retired:u64 | n:u8 (id:u8 value:u64)* | (len:u64 bytes)*
ERR     5 | code:u32 | pc:u64 | word:u32 | msg
OPEN    6 | (like RUN, without calls)     → resident session: code, buffers, constants
STEP    7 | deadline | writes (buf off bytes)* | calls | returns (buf off len)*
          | [ncopy:u32 (src soff dst doff len)*]   (optional, since 0.6)
CLOSE   8
```

STEP's optional copies are the feedback of state that is not
updated in place — the `s ← s_next` of a recurrent model (Mamba) —,
done inside the worker after the pass, like RUN's between iterations:
the state never crosses over to the BEAM. A STEP without them is byte for byte the one from
before.

Each call may carry up to two partition descriptors (`count, grain,
ptrs (arg, stride)*, scratch (arg, bytes)*, guard`); the first applicable one is
used. The worker's and the daemon's tables are sized by the
frame itself (a hostile count does not allocate more than the frame describes).

The fabric's RUN frame swaps `code/calls` for SPIR-V modules and dispatches
(`module, gx, gy, gz, push*, binds*`, with binds of buffer or *window* type
`buf, base, stride, len`). Plans are self-contained (no state in the worker):
a restarted process needs no replay.

### 4.4 Memory
- Weights ≥ 64 KiB go once to `/dev/shm/vapor-<sha256>`
  (content addressing, atomic write) and are mapped
  copy-on-write by the worker and **imported without copy** by the fabric via
  `VK_EXT_external_memory_host` when the device offers it.
- Per-token inputs are *windows*: the daemon copies slice `t` before
  iteration `t`; the worker passes `base + t·stride`.
- State feedback (`h ← h_next`) is a copy declared in the plan.
- Replicas and restarts map the same files: `n` workers with the same
  model occupy one copy of the weights in the page cache.
  `Vapor.Runtime.Shm.prune/0` (`mix vapor.shm`, and at the end of the test suite)
  removes the files that no process maps; whoever needs a removed name
  rewrites it (`put/1` checks every time).
- Weights can stay in **bfloat16** (`storage: :bf16`): half the bytes;
  `vld_bf16` widens 16 weights exactly on load (`vpmovzxwd` + shift,
  `SHLL #16`, `vle16`+`vzext.vf2`+`vsll`, word/half-word in SPIR-V),
  so the result is bit for bit that of the f32 program over the same values.

### 4.5 Failover
`Vapor.Runtime.Dispatch` tries the arbiter's choice and goes down the chain
`fabric → host AVX-512 → host base → RVV interpreter → oracle`,
recording the reason for each hop. The fabric also refuses by a static
limit (attention head above 512), and the unit goes down the chain. Under the canonical policy all links compute the same bits,
so rerouting never changes the answer; the oracle does not fail for a
certified program.

### 4.6 Parallelism and concurrency

The guarantee to preserve: the bits of each output do not depend on how many
threads, on how many sequences share the step, on where the KV lives, on which
replica serves nor on which substrate executes. Every form below draws
parallelism from **independent outputs**, never from reassociating a sum.

| form | where | invariant | test |
|---|---|---|---|
| SIMD | AVX2, AVX-512, NEON, RVV (VLEN 128–512), SPIR-V | same bits as the oracle | `canon_test`, `native_test`, `model_test` (QEMU, lavapipe) |
| ILP | R rows per iteration in the GEMV (4/2/1, chosen by the allocator) | same | same |
| intra-operation threads | pool in the worker, partition descriptor per call; decode attention split by KV head | 1 = 2 = 3 threads | `threads_test`, `model_ops_test` |
| continuous batching | `Vapor.Engine`: decode + prefill chunks in the same step | a sequence's tokens are equal alone or in a batch, under any chunking | `engine_test` |
| paged KV | `kv_write_paged` / `attention_paged` with a block table | paged = contiguous | `model_ops_test` |
| replicas (data) | `Vapor.Engine.Pool`: one compilation, shared weight pages, shortest queue | same answer from any replica; a dead replica takes only its own requests | `engine_test` |
| BEAM concurrency | process per HTTP connection, engine as a `GenServer`, SSE | — | `serve_test` (`openai` client) |
| memory bandwidth | shared f32 weights (mmap), resident `bf16`, *weight-stationary* GEMV (a block of rows of W meets the whole batch) | bf16 = f32 over rounded weights | `model_test`, `engine_test` |
| speculation | draft proposes, target verifies `k+1` rows in one step | output identical to the target alone | `speculative_test` |
| GPU | SPIR-V for all model operators | fabric = oracle (whole model, paged step with sampling, recurrent decode) | `model_test`, `autodiff_test` |

## 5. Numerical model

`Vapor.F32` is **exact** binary32 arithmetic on the BEAM: values as bit
patterns; `add/sub/mul` via binary64 + one rounding (correct, since
53 ≥ 2·24 + 2); `fma` in dyadic rationals with a single rounding
(RNE, gradual underflow, overflow to ∞). Validated against the hardware's `fmaf`/`+`/`*`
on 800,000 operations, including subnormals and signed zeros.

| Policy | Semantics | Guarantee |
|---|---|---|
| `:canonical` | mul and add rounded separately; reduction in a fixed 16-lane tree (`r8 → r4 → r2 → r`); `÷` correctly rounded (semantics v2) | **identical bits** on x86, ARM, RVV, interpreter and GPU (FNV-1a digest) |
| `:fast` | fused FMA where the operator allows | within the rigorous envelope on all substrates |

The central idea: reproducibility does not require slowness if the parallelism comes
from **independent outputs** (rows, elements) and not from reassociating a
sum. That is why each GEMV row has exactly 16 accumulators, and the
kernel gains throughput by processing R rows per iteration — chosen by the
allocator per ISA — without changing a bit.

`Vapor.Verify.Envelope` computes, for each output, the exact real value and a
rigorous bound valid for *any* conforming substrate (correct
rounding, fused FMA or not, any reduction order, gradual underflow
or flush-to-zero), in exact dyadic rationals: Wilkinson form
`γₙ·S + a` for contractions over exact inputs (decided by
`within_envelope/4` extracted from Lean) and running error analysis for the
rest. The verifier never rounds; only bounds are shortened, and only
upwards. Since 0.6 the Higham form (`γₙ` as a bound on products of
`(1 + δᵢ)`) is also proved in Lean (`proofs/Vapor/Higham.lean`).

**Semantics version.** A program's bits are a function of its terms
**and** of the definition of the canonical functions; the certificates record
`Vapor.Canon.version/0`. Version 1 (up to 0.5): `a/b = a·rcp(b)`, within ≤ 1 ulp.
Version 2 (since 0.6): `÷` **correctly rounded** (IEEE, with the GPUs'
DAZ/FTZ convention) — Markstein with Dekker's exact residual, without
FMA — and `log` (≤ 1 ulp from the correctly rounded value). With this
`+ − × ÷` are IEEE on every substrate.

**Substrate airlock (0.10).** A substrate only receives canonical
programs after it has been measured: the probes of `Vapor.Substrate` give the verdict
(`:canonical`, `:envelope` with the numerical fingerprint, `:refused`) and the
dispatcher respects it. The envelope also covers DAZ (input subnormals
read as zero), found by the kit in XLA. Metal, StableHLO/PJRT
(Tenstorrent), the cluster and FreeBSD: [SUBSTRATES.md](SUBSTRATES.md).

## 6. Verification ladder and certificate

| Rung | Establishes | How |
|---|---|---|
| 1 | sorts, shapes, feedback | `Program.check/1` |
| — | exact rewriting | `Rewrite` |
| 2 | admission | allocation accepted by the extracted checker on **every** ISA; int8 no-wrap via `admissible/3` at the maximum of the extents; SPIR-V limits |
| 3 | adjoint identity ⟨Wx, v⟩ = ⟨x, Wᵀv⟩ | compiled kernel vs. exact transpose (layout/transposition errors) |
| 4 | differential oracle | each CPU substrate bit for bit equal to the exact oracle |
| 5 | parity + envelope | equal FNV-1a digests across substrates; every output within the envelope |
| 6 | proof-carrying code | Ed25519 certificate with a quorum by co-signature |

The payload contains the SHA-256 of the program, of each machine blob and of each
SPIR-V module, the digest of the checkers' Lean sources, the evidence of each
rung and the **counted work** (FLOPs, bytes, instructions per ISA). It contains
nothing host-dependent (no timings, no dispatch decision), so
independent nodes that redo the ladder produce payloads **identical byte for
byte** and can co-sign (`Certificate.cosign/3`). `verify/3` requires
`quorum` valid signatures from distinct trusted keys. `Bundle`
packages artefacts + certificate; the edge node checks signatures and
hashes in ~2 ms and executes without redoing the ladder.

## 7. Arbiter: three-ceiling roofline

`t = Σ max(F/π, Q/β, I/ι) + overheads`, with `F`, `Q` counted from the schedule and
`I` = static count of the instructions in the body of each hot loop, *as
emitted*, times the number of iterations. The third ceiling is what makes the
model honest: the 4-bit GEMV is bound by instruction issue
(~1.1 instr/weight), not by bandwidth, and a two-ceiling model misses by ~5×.
With all three, the 4096×4096 GEMV is predicted at 71.7 ms and observed at
72–75 ms. The profiles are **declared** (a table per vendor; Mesa
lavapipe has a CPU-class profile). The decision is the argmin of the predicted time,
taken on each node from the certified work and its own profile.

## 8. Assurance model

**Proved in Lean 4 (core only, no `axiom`/`sorry`, no warnings):**
- ℤ/2³²ℤ: associativity/commutativity of sum and product; `wrap32` is a
  homomorphism; **every reduction tree with a wrapping adder computes
  the exact value** under `K·|A|max·|B|max < 2³¹` (`integer_parity`), for
  any permutation of the leaves; monotonicity of admissibility.
- Banks: Euclid's lemma from `gcd`; coprime stride ⇒ injective on the
  lanes (`bank_injective`); closed-form pad for 2ᵐ banks, sufficient
  and **minimal**.
- Affine monoid: associative, neutral (1, 0), and composition = application in
  sequence; associative segmented lifting for any monoid.
- Allocation checker: soundness (`checkAlloc_sound`).
- Envelope: the exact decision, monotonic in the extent; two-sided form.
- Estrin depth = 3 vs. Horner = 7, computed over the tree.

**Extracted to Elixir** (`lake exe vapor-extract`): `check_alloc`,
`admissible`, `wrap_s32`, `pad_stride`, `bank_of`, `affine_op`,
`seg_affine`, `within_envelope`. The extractor reads the *elaborated* terms — the
same ones the theorems talk about — and fails on any construct
outside the fragment. Lean computes 480+ conformance vectors that the
extracted Elixir must reproduce; the generated module carries the FNV-1a of the
Lean sources, checked by the audit test without needing the toolchain.

**Tested (not proved):** encoders (binutils, QEMU, lavapipe),
RVV interpreter vs. QEMU at VLEN 128/256/512, bit-for-bit equality across
all substrates, fault containment.

**Trusted base:** the BEAM and the Linux kernel; the extractor's printer (~200
lines) and its 6-line prelude; the IEEE-754 *standard model*
(`fl(x op y) = (x op y)(1 + δ)`, `|δ| ≤ 2⁻²⁴`, away from underflow — a
property of the hardware specification; the Higham lemma that starts from it
has been proved since 0.6); the Vulkan driver (contained in a process).

## 9. Scrutiny of the directive

1. **`VK_KHR_external_memory_fd` does not import an arbitrary memfd.** An
   OPAQUE_FD handle only accepts memory exported by the same driver. The
   mechanism that delivers zero copy from files written by the BEAM
   is `VK_EXT_external_memory_host` over a mapping — the one implemented.
2. **Lean→Elixir extraction happens at build time, not at run time.** Running the
   Lean toolchain in production would reintroduce the giant container that the
   directive wants to eliminate. Freshness is guaranteed by digest.
3. **The ridge condition of Eq. 3 was demoted to information.** With all
   the ceilings in the prediction, it can only pick the slowest substrate.
4. **The dispatch decision left the certificate**, which now carries the
   counted work (portable and co-signable); the decision is per node.
5. **A "safe" NIF was refused.** An interpreter in a NIF shares the
   BEAM's address space; a bug in it brings down the VM. The
   interpreter runs in the worker.
6. **Names:** modules `Vapor.Emit.*` (not `Aether.Emit.*`), consistent with the
   system's name.
7. **Predecessor bugs fixed:** `vtype` with SEW/LMUL swapped;
   `vle8.v` in the OP-V opcode; wrong cooperative matrix opcodes (4448 is
   a *capability*); `decode_i8` on floats; `axiom` and a "theorem" with
   conclusion `True`.

## 10. Limitations

- No physical RVV, ARM or discrete GPU hardware here: RVV/NEON validated under
  QEMU (and the interpreter against QEMU), Vulkan under lavapipe.
- The cooperative matrix path is emitted, validated by `spirv-val` and
  selected when the device advertises it (s8,s8,s32,16×16×16,subgroup),
  but lavapipe does not support it: **it has not been executed**.
- The worker is Linux (direct syscalls). The NEON code follows AAPCS64, which
  Apple's arm64 respects (x18 reserved, d8–d15 callee-saved), but a worker
  for macOS/Apple Silicon (`MAP_JIT`, `pthread_jit_write_protect_np`,
  isolation without seccomp) has not been written — no machine to run it on.
- This VM has 2 vCPUs and no PMU: scaling to many cores and the
  hardware counters have not been measured here (the code reads them when they
  exist). The 4-bit GEMV remains bound by instruction issue (the scalar computation
  of α/β per sub-block and the u8→f32 conversions dominate): AVX-512 barely
  speeds it up; vectorising that computation is the next step.
- The fabric executes self-contained units (no resident sessions): it serves
  whole models and certification, but the generation engine uses the worker.
- Attention in SPIR-V recomputes the scores in three passes (no shared
  memory): correct and bit for bit, not optimised for a real GPU.
- RoPE factors that arrive as a GGUF tensor have no spelling in
  `config.json`; exporting such a model to Hugging Face is refused.
- Fabric buffers are host-visible; discrete GPUs call for device-local +
  staging.
- The oracle and the envelope are exact and therefore slow: they limit the size of the
  probes of rungs 3–5 (the bounds hold at the maximum of the extents by
  monotonicity).
- seccomp is not available under `qemu-user` (reported as
  `:unsupported`); per-process isolation still holds.
- Frontier families (since 0.6: sparse MoE, latent MLA, window and ring,
  Mamba — [FRONTIER.md](FRONTIER.md)): attention *soft-capping* (Gemma 2),
  dynamic NTK, LongRoPE and SSM + attention hybrids are refused by
  name; Mamba-2 is in since 0.8 and 4-bit MoE is sparse since 0.8
  (`qgemv_masked`).
- The canonical transcendental functions (`exp`, `log`, `tanh`, `gelu`…) are
  identical on every substrate, but they are not correctly rounded (within ≤ 1–2
  ulps, measured); `+ − × ÷` match IEEE bit for bit (÷ since 0.6, with
  DAZ/FTZ).
- Normalised text depends on OTP's Unicode version. It is recorded
  in the agents' digest and tested separately, but it is not eliminated.

## 11. Layers over the core

Since 0.10: pre-training and endless context ([TRAINING.md](TRAINING.md)), physics
for RL and digital twins ([PHYSICS.md](PHYSICS.md)), CJK, Arabic, figures and formulas ([OCR.md §3g–§3k](OCR.md)).
Since 0.11: living scene, sketch and files ([SCENE.md](SCENE.md)), mathematics
([MATHEMATICS.md](MATHEMATICS.md)), science ([SCIENCE.md](SCIENCE.md)). (Complex
networks, algorithm discovery and the tic-tac-toe self-play left in
0.16: they were fixed demonstrations — [DIRECTIVE.md §19](DIRECTIVE.md).)
Since 0.12: workbench, engineering, logic, boards, proteins, render.
Since 0.13: finance and the trading desk ([FINANCE.md](FINANCE.md)) —
the first layer since the workbench to **compile to the core** again: the
Monte Carlo writes the trajectory step, the generator included, as terms
of the algebra (the generator chosen because it fits **exactly** in binary32: Lehmer
products below 2²³, modulo by the 2²³ rounding trick), and so
inherits the central guarantee — the same bits on every substrate and at every
thread count, checked against the oracle in the answer itself. The
other pieces (the order book and its judge, the rational simplex, the
noise gates) inherit from the core the canonical CBOR and the Merkle trees
of the journals.


The layers in this section use the core without altering it: everything they produce is a
certified program or canonical data.

| layer | modules | guarantee it inherits | document |
|---|---|---|---|
| constrained output | `Vapor.Grammar`, `Grammar.{JSONSchema, Vocab, Constraint}`, `Vapor.Tools` | the engine only chooses tokens the grammar admits; no accepted prefix is a dead end | [AGENTS.md §6](AGENTS.md) |
| templates | `Vapor.Template`, `Vapor.Chat` | rendering = HF's `jinja2`, byte for byte; explicit clock | [AGENTS.md §2](AGENTS.md) |
| agents | `Vapor.Agent`, `Agent.{Spec, Journal, Keys, Store, Backend.*, MCP}` | local decisions re-derivable on any substrate ⇒ replaying is verifying | [AGENTS.md §3](AGENTS.md) |
| RAG | `Vapor.RAG`, `Vapor.Merkle`, `Vapor.Embed` | dense scores are canonical `linear` ⇒ the same index on any node | [AGENTS.md §4](AGENTS.md) |
| canonical | `Vapor.Canonical` (CBOR RFC 8949 §4.2), `Vapor.CR` | certificates and journals verifiable outside the BEAM | [ECOSYSTEM.md](ECOSYSTEM.md) |
| transport | `Vapor.Serve.dispatch/3` + responders (`:gen_tcp`, `Vapor.Plug`) | the same handler, the same receipts | [ELIXIR_ECOSYSTEM.md](ELIXIR_ECOSYSTEM.md) |

The engine monitors the recipient of each request. A recipient that
dies removes its sequences from the batch at the next step boundary, and
`cancel/2` does the same on purpose. These two paths are the only way
for a sequence to leave the batch before the end, and neither of them alters the
bits of the others (batch invariance).
