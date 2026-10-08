# Substrates: hardware admitted by measurement (0.10)

> Round 0.10. Request, translated: "Metal (Apple) support + Tenstorrent + vapor
> cluster orchestration". Scrutiny of the request and what was left out:
> [DIRECTIVE.md §13](DIRECTIVE.md). Tests: `metal_test.exs`, `substrate_test.exs`,
> `stablehlo_test.exs`, `cluster_test.exs`, `freebsd_test.exs`.

## 1. The pain

A new accelerator (an Apple GPU, a Tenstorrent chip, the next NPU)
arrives with a compiler that **does not promise the same numbers**: it sums in
another order, fuses `a·b + c` into one instruction, flushes subnormals to zero, rounds
operands to bf16 without warning. The result is that "the model runs on
device X" does not say whether it computes **the same thing** — and the difference
only shows up in production, in an edge case.

vapor already had a canonical semantics (sums in a fixed tree of 16
lanes, correctly rounded division; the same bits on every
substrate) and a **proved error envelope** (dyadic Wilkinson/Higham)
for the substrates that do not reproduce it. What was missing was the way in.

## 2. The substrate airlock (`Vapor.Substrate`)

A substrate computes only after it has been **measured**. Admission runs probes with
known answers — each one isolates one behaviour:

| probe | what it reveals |
|---|---|
| `contraction` | `1 + 2⁻¹²` at a tie: FMA changes the last bit |
| `subnormal_out`, `subnormal_in` | FTZ (output) and DAZ (input) |
| `signed_zero`, `nan_select` | IEEE or not |
| `reduction` | the summation order (canonical tree or another) |
| `input_precision` | rows `1 + 2⁻ᵏ`: how many mantissa bits the operand really has |
| `division`, `functions` | canonical division and transcendentals |
| `linear_f32/bf16`, `qgemv_sb4`, `gemm_i8`, `attention`, `sample` | the real kernels, in real shapes |

The oracle's exact answer decides the verdict:

- **`:canonical`** — all bits equal;
- **`:envelope`** — different, but within the proved envelope, with the
  **numerical fingerprint** (`contraction`, `flush_to_zero`,
  `denormals_are_zero`, `reduction_order`, `significand_bits`, …)
  recorded; `:canonical` programs are not dispatched to it,
  `:fast` programs may be;
- **`:refused`** — outside the envelope, integers that do not wrap around, or
  operands with fewer than 24 mantissa bits (measured: "8 significand
  bits" on a bf16 engine), or a device that hangs, answers outside the
  protocol or leaves an output out; and a difference **where no
  bound exists** (division, functions, attention) only passes if the probes
  measured a cause for it (contraction, flushed subnormals, another summation
  order) — without a cause, it is a device computing something else.

A probe that did not run (admission with `only:`, a kit without it) appears in the
fingerprint as `:unmeasured`, never as "canonical".

**Finding (0.10, independent review):** the first version looked at the
envelope only for the analytic probes — a device wrong only in division
came out `:envelope` — and crashed (instead of refusing) on a response missing one
of the outputs. Fixed, with the cases in the test suite.

The admission is a canonical CBOR record **signed (Ed25519)**; an
altered byte is caught. The dispatcher consults the admission; an accelerator that
arrives is admitted on arrival.

**Finding (0.10):** the portable kit (§4) run on CPU XLA showed
DAZ — *input* subnormals read as zero — which the envelope did not
cover: `min(subnormal, 0.25)` came out beyond the bound. The envelope now covers
DAZ (`max(e, |v|)` per suspect operand), and stays tight: the control
with twice the exact value is refused (`substrate_test.exs`).

Measured: native worker, AVX-512, emulated RVV, lavapipe (Vulkan) and oracle —
**canonical**; "contracting" Metal shim — envelope with FMA; shim with FTZ —
envelope with FTZ and DAZ; simulated bf16 engine — refused; int8 GEMM that
saturates — refused; CPU XLA (via the kit) — envelope (FMA, FTZ/DAZ, another
reduction order).

## 3. Metal (Apple)

- **MSL translator** (`Vapor.Emit.MSL`): the symbolic library of SPIR-V
  kernels (the same one that generates x86, NEON, RVV and SPIR-V) translated to Metal
  Shading Language, with the control flow **reconstructed** (MSL has no
  `goto`: structured loops and selections).
- **`vapor-metal` daemon** (Zig, macOS): the fabric protocol
  (HELLO/RUN/OPEN/STEP/CLOSE), the Objective-C runtime opened at
  run time (`dlopen` + `objc_msgSend`), `MTLMathModeSafe`, shared
  *buffers* (unified memory, `newBufferWithBytesNoCopy`).
  It compiles from any host (`make metal`), without an SDK.
- **CPU worker for macOS**: `MAP_JIT` + `pthread_jit_write_protect_np`
  + instruction cache invalidation; `__ulock_wait/wake` in place of
  futex.
- **How it is tested without a Mac:** the same MSL text compiles with `clang` through a
  header *shim*; the daemon runs on Linux on top of that *shim*
  (`vapor-metal-sim`). Three "devices" by compilation *flag*:
  conforming (`-ffp-contract=off`) — canonical, bit for bit with the oracle on
  canonical programs, SSM with iterations, GEMM, attention, a
  Llama session and the engine; contracting (`-ffp-contract=fast -mfma`) — envelope; FTZ
  (MXCSR) — envelope.

**Limit:** the real daemon (`mtl.zig`) is **compiled and type-checked,
never executed** here. Admitted on arrival by the same process, it will tell
what it is.

## 4. Tenstorrent (and any PJRT)

The honest path is not to fake a *backend* without the hardware: it is to **export**
to the format that the Tenstorrent compiler (tt-xla / tt-mlir) accepts and
**judge the result with the same airlock**.

- **StableHLO export** (`Vapor.Export.StableHLO`, `mix vapor.export`):
  elementwise, reductions, `linear` (and masked, grouped), `gemm_i8`,
  `gather_row`, RoPE, KV cache writes, attention (with window and
  head grouping), `transpose`, `reshape`; index arithmetic
  always signed; dense constants in hexadecimal. What has no
  exact equivalent is **refused by name** (sampling, 4-bit
  kernels, paged operations, exact `gelu`).
  Checked: tiny Llama/Qwen2/Mistral exported and run on CPU
  XLA via JAX — *logits* within ~10⁻⁶, same argmax.
- **Portable admission kit** (`mix vapor.substrate kit DIR`): the probes
  as StableHLO, a `run_kit.py` (`--platform tt`, `tpu`, `cpu`, …) that
  runs on the device through PJRT, and `mix vapor.substrate judge DIR [--sign KEY]`
  that brings the answers back and issues the admission signed with the operator's key (the one from `mix vapor.audit keygen`: the signature says *who* admitted it). The
  device never needs the BEAM.

**Limit:** no Tenstorrent board here; the kit was run on CPU XLA.

## 5. Cluster orchestration (`Vapor.Cluster`)

Determinism changes what an orchestrator can do:

- **Content-addressed cache across the whole cluster**: a job's key is the
  digest of the program and the inputs; repeating it is a cache hit, and the
  result is the same on any node.
- **Auditing by redundant execution**: a sample of the jobs runs on
  two nodes and the bits are compared; a divergence is decided by the
  oracle, the node that was wrong goes into **quarantine** with the evidence and only
  comes back **by measurement** (the airlock on the node itself). The sample is a keyed
  *hash* (`H(salt ‖ key) < rate·2²⁵⁶`): whoever has the salt recomputes which
  jobs were checked — the sample cannot be chosen afterwards.
- **Failure and *hedging* without changing a bit**: a lost node has its jobs
  redone on another; a straggler is overtaken by a copy.
- **Distributed training with the bits of one machine** (§ [TRAINING.md](TRAINING.md)).

Tested with real `:peer` nodes (`cluster_test.exs`): a node that flips a bit
is caught and put in quarantine — and, without the auditing (the control), the
wrong answers pass; readmission refused while it lies; lost node;
*hedging* (and the control without *hedging* waits for the straggler).

## 6. FreeBSD

The CPU worker compiles for `x86_64-freebsd` and `aarch64-freebsd`
(`make freebsd`; zig ships the FreeBSD libc headers): libc, futex
via `_umtx_op`, W^X code pages via `mprotect`, and isolation via
**Capsicum** — after `cap_enter` the process has no global
namespace (no `open` by path, sockets, `execve`, new processes);
it keeps only what it already has: stdio, memory and **one directory descriptor
limited to lookup, read, `seek`, `fstat` and read-only mapping**, through
which the weights are opened (`openat`). The same policy as the Linux seccomp
filter, by capabilities instead of a list of calls.

**Limit:** compiled and checked (the binary is FreeBSD, imports
`cap_enter`, `cap_rights_limit`, `_umtx_op`, `openat`, `mprotect`, and does not
import anything from Apple's JIT — `freebsd_test.exs`; Linux's seccomp is done with raw calls, invisible in the import table: it is kept out by conditional compilation, not by checking); **not
executed** — there is no FreeBSD on this machine. The Vulkan fabric has not been
ported (it uses raw Linux calls).
