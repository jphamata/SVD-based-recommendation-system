# Open items — state as of 2026-10-08 (0.17.0)

Only what is **open**. What was closed is in the [CHANGELOG](../CHANGELOG.md),
with the test that proves it. Each item says why it matters and what closes it;
◐ = partly done (what is missing is written down). The items from the attachments of
rounds 0.6, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16 and 0.17 that were left out are here with the reason
([DIRECTIVE.md §9, §11–§20](DIRECTIVE.md)).

## Round 0.17 — what was left open

- ☐ **Palingenesis on production checkpoints** (Qwen 2.5 7B/14B, a quantised DeepSeek-V3 expert): the
  gates, the RCU publication and the lineage are measured on a tiny model only. Also: growing a
  mixture (adding an expert changes the router's contract and is refused today), and a worker-level
  swap inside a running session (today the next session sees the new generation).
- ☐ **Qālib**: sequential equivalence (k-induction on the miter, IC3/PDR), cell functions read from the
  liberty files instead of the table, and the area and timing those files hold.
- ☐ **Integer programs**: cutting planes (Gomory, each with its Chvátal–Gomory derivation in the
  certificate) and export in the VIPR format, so other checkers can read vapor's certificates.
- ☐ **Causes**: the two witness models built from a hedge (today the hedge is checked structurally, and
  the bow's two models are built by hand in the tests); counterfactuals (ID*) and transportability.
- ☐ **Ternary weights `:sb2`** with a model trained ternary (BitNet b1.58): a GEMV of additions and
  subtractions in every backend, and the accumulator's overflow bound in Lean. Without such a model
  the format measures nothing.
- ☐ **Agents and the web, as the 0.17 request designed it**: an `observe` tool that fetches through the
  hermetic seal, strips scripts, stores the snapshot in the Khazāna by hash under a byte budget, and is
  never re-fetched on replay; an `act` plan executed only with a person's Ed25519 signature.
- ☐ **Sketch**: a parametric constraint solver (coincidence, tangency, distance) with a degrees-of-freedom
  report; Gröbner bases to prove a constraint set inconsistent.
- ☐ **Language server**: rename, code actions ("add the missing `(box …)`"), semantic tokens.
- ☐ **Recommendations**: implicit feedback (weighted ALS) and a worker kernel for tables beyond ~10⁵ ratings.

## Round 0.16 — what was left open

- ☐ **Conversations: *streaming***. Today the answer arrives whole; the server already streams on
  `/v1/chat/completions` — what is missing is connecting `Majlis.reply` to a *stream* (the node is only written at the end, so
  atomicity does not change) and the page showing it.
- ☐ **Conversations: attachments** (images and documents in the message, stored by hash in the Khazāna, read
  by vapor's readers with provenance) and **memory across conversations** (pinned facts that hold
  for all of them, with the same context bar showing the cost).
- ☐ **Conversations: agent with visible steps** on the page (today the journal is kept and verifiable through
  `GET /v1/vapor/journal/:id`, but the page shows only the answer).
- ◐ **Almizan**: the Lean exports close in core Lean ✅ 0.17 (`grind`/`decide`, checked by the `:lean`
  tier); missing are bounded quantifiers over integers (Presburger) as another decider, and `import`
  across files with a pinned hash (the embryo of Khazāna P2P).
- ☐ **Khazāna P2P** (the manifesto): content-addressed dependencies exchanged between peers — it first needs
  a trust model (who signs what), otherwise it is a supply-chain vector.
- ☐ **The Khazāna root proved in Lean** (the two-slot protocol), as ASAS §11 also owes.
- ☐ **Weak memory** of the `/dev/shm` protocol of the *worker* (litmus against an RVWMO/TSO model) — in the
  ledger as owed.
- ☐ **Editors**: actually open the extension in VS Code, the *plugin* in Neovim and the mode in Emacs on a
  machine that has them (here only the server and the formats are tested); publish the extension.

## Round 0.15 — what was left open

- ☐ **Amalgam in the worker**: the exact sum as a native program (a Kulisch accumulator of ~4,300 bits for f32 in registers, or the sum in two passes by exponent) — today ≈ 6 M additions/s on the BEAM, which is enough for gradients across micro-batches and nodes, not for inside a *kernel*. Also: the exact *all-reduce* across nodes of `Vapor.Cluster` (the cells already cross the network through `to_wire/1`), and the split **by rows** in `Vapor.Shard` (refused today) redone with `partial_dot/3` — the piece exists and is tested; what is missing is connecting it to the partitioner.
- ☐ **Cupel in the training path**: the same identity in the *backward* (`∂x = ∂y·W` checks with `r`) and in the optimizer update; the sentinel connected to `Vapor.Cluster` (quarantine of a whole node, not only of a worker); measure the overhead on the native worker.
- ☐ **Rebis**: sequential circuits (k-induction over the multi-step *miter*, with IC3/PDR as the horizon); read structural Verilog/BLIF ✅ 0.17 ([QALIB.md](QALIB.md)); `PCLMULQDQ`/`vclmul` as a compiler operation (carry-less multiplication in the emitters) for GHASH and for binary towers; algebraic rewriting with rules for parallel-prefix adders (today `:unknown` above 50,000 terms).
- ☐ **Aludel**: also split by degree (degree elevation when the enclosure is wide), more variables with adaptive per-axis subdivision, and barriers with rational terms; export the witness to a verifier in Lean when `lake` is present.
- ☐ **Tabula**: deadlines (bounded linear temporal logic over the facts), quantification over parties, and drafting of clauses from text by a model (`Mind`), with back-translation, as Alembic has.
- ☐ **Mechanisms** (VCG, Gale–Shapley) with stability and truthfulness checked — small, deferred for lack of a case (§18).
- ☐ **Diverse double-compiling** (Wheeler) of the worker with two independent Zigs — the right answer to *trusting trust* (§18).
- ☐ **JBIG2**: Huffman with refinement (SDREFAGG/SBREFINE under SDHUFF/SBHUFF) and arithmetic contexts retained across segments; JPX.

## Round 0.14 — what was left open

- ☐ **Distributed Athanor**: the evaluations are pure and the journal is canonical — split the budget across BEAM nodes (`Vapor.Cluster` already exists) and join the journals by Merkle root. Today: one node, sequential evaluations per strategy.
- ☐ **Compiled Alembic**: the closure compiler is ~20–50× slower than native Elixir; the numerical objectives could go down to the tensor algebra (the path of `Vapor.Expr.compile`) with the same fuel counted per block.
- ☐ **Proofs beyond enumeration**: export the claim and the space to SAT/SMT (the logic desk already has DRUP) when the space does not fit in the enumeration — "proved" would then hold for spaces that today only have evidence.
- ☐ **Crucible**: p orbitals (6-31G bases) and UHF to break bonds; PDEs in 2D with the same evidence of order; conservation laws for systems with symbolic parameters.
- ☐ **Assay**: evaluation of generative models with IRT (difficulty per item), sequential tests (stopping early with error control), and *dedup* at corpus scale (LSH on disk).
- ☐ **Mind**: an optional autonomous loop (the model proposes, the furnace measures, the model reads the certificate and proposes again) with a call budget and recording in the journal; today the loop is guided by the person.
- ◐ **Free scenes**: text operations, per-frame expressions and direction by a model exist; what is missing is simple physics between entities (collision, springs) written in Alembic, and the editable timeline in the console.

## Round 0.13 — what was left open

- ☐ **Low-latency order engine**: matching as a worker program (Zig, no allocation, book in per-level arrays) with the same journal and the same judge; latency measured with PMU on *bare-metal*. *Kernel bypass* and FPGA stay out. Today: ~8.5 µs per event on the BEAM, with the hash.
- ☐ **Real market data**: a sample day of Nasdaq ITCH and ANBIMA's curves and indicative prices (they need network access): rebuild the book and check it against the published snapshots; reprice government bonds with the day's rates.
- ☐ **Integer generator in the compiler**: Philox/Threefry require 32/64-bit integer multiplication in the algebra (the same gap as for the NTT kernel); they would replace the worker's Wichmann–Hill. Quasi-Monte Carlo (Sobol, with *scrambling*) and Greeks by *pathwise* automatic differentiation (the autodiff over terms already exists).
- ☐ **Rate and volatility models**: Hull–White and LMM; Dupire local volatility and calibration of Heston to a whole surface (SVI per slice → SSVI); XVA and credit.
- ☐ **Backtests**: intraday data with the book (the engine already exists), cost with Almgren–Chriss impact inside the backtest, *walk-forward*; CSCV costs O(N·12,870) — sample the halves when N > 100.
- ☐ **Arbitrage over the whole surface**: calendar × strike in a single LP (today: one maturity at a time, plus the SVI calendar separately).
- ☐ **Judge in another language**: the naive engine also in Python, so that the independence is one of language and not only of algorithm; session-level FIX (logon, *heartbeat*, *resend*), not only application messages.
- ☐ **Pre-trade**: credit limits per counterparty and rate in wall-clock time (today: event time).

## Out of reach of this machine (they need hardware)

- ☐ RVV 1.0 in silicon (BPI-F3 / Milk-V) and `cooperative_matrix` on a real GPU. *Here: QEMU, our own RVV interpreter, lavapipe.*
- ☐ PMU and RAPL (J/token) on bare-metal. *The code already reads both; this VM has neither.*
- ☐ Measure the resident sessions (0.8) on a real GPU, discrete (*staging* path) and integrated.
- ◐ **Apple Silicon** (0.10): CPU worker (`MAP_JIT`, `__ulock`) and Metal daemon **compiled**; the MSL executed by a *shim* with clang and admitted by the airlock. What is missing is **executing on a Mac** (the `mtl.zig` daemon has never run) and the isolation of the worker on macOS (no seccomp; a process without rights + restricted `posix_spawn`); and Windows. *It also decides Tauri ([INTERFACES.md](INTERFACES.md)).*
- ☐ **Real Tenstorrent** (0.10): the StableHLO + PJRT kit was judged on CPU XLA; what is missing is running it on a board (tt-xla) and admitting what it is.
- ☐ **FreeBSD executed** (0.10): the worker compiles and the binary is checked; what is missing is running the suite on a FreeBSD (Capsicum, `_umtx_op`) and porting the Vulkan fabric (`fabric.zig` uses raw Linux calls) to `sys.zig`.

## Round 0.12 — what was left open

- ☐ **Proteins with real alignments**: read an MSA (Pfam's Stockholm/A3M), sequence weights (80 % identity), pseudocounts and DCA by pseudo-likelihood (plmDCA), which is much more accurate than the mean-field one on real alignments; secondary structure predicted from the sequence (not taken from the native); side chains. Measure on the set of Jones et al. (PSICOV) against what is published. *Structure prediction with trained networks remains out: weights and databases.*
- ☐ **Render**: multiple importance sampling (BSDF × light) to close the noisy caustics; BVH and triangle meshes (the sketch's GLB → scene), image textures, microfacet materials (GGX) and subsurface; a denoiser checked against the many-sample reference.
- ◐ **Workbench**: affine units (°C, °F) as readings ✅ 0.13; missing are time-dependent 2-D PDEs and coupled systems (multi-species reaction–diffusion), unstructured meshes; DAEs (index 1) and delay ODEs; global optimization (multistart with an interval certificate).
- ◐ **Engineering**: transistors (Ebers–Moll, level-1 MOSFET) ✅ 0.13, equal to ngspice; missing are `.subckt`, device capacitances in the transient; short circuit and reactive limits in power flow; buckling (geometric eigenvalue) and geometric nonlinearity in frames; pumps and valves in networks; non-ideal liquid–vapour equilibrium (NRTL/UNIQUAC).
- ◐ **Logic**: linear arithmetic (rational simplex with a Farkas certificate) ✅ 0.13 ([LOGIC.md §5](LOGIC.md)); missing are the DRUP checker in the native worker (R(3, 4) in seconds); LRAT (linear checking); exporting Gröbner and Knuth–Bendix certificates to Lean (now that Lean runs here); integer programming ✅ 0.17 ([LOGIC.md §6](LOGIC.md); cutting planes are in round 0.17's list).
- ☐ **Boards**: evaluation by a network trained by generic self-play (the loop already exists) for small chess/shogi (5×5 minishogi, 6×6 Los Alamos); NNUE as a compiler program; 9×9 Go with a network; hold'em with card abstraction.
- ☐ **Generic self-play** loses 21 % of the optimal lines with 8 simulations at tic-tac-toe, against 13 % for the specialized one — it still has to match it (residual network, *temperature schedule*, more games) before going to larger games.
- ☐ **Scene**: exact-frame MP4 (the exact-frame GIF exists); direction by clause grammar does not yet parse coordination ("Ana e Bento dançam") nor subordinate clauses.

## Round 0.11 — what was left open

- ☐ **Learned depth for the living scene**: the ground-plane heuristic is the fallback; a monocular depth model through the airlock (the layers and the depths are already data) would open up photos without ground (portraits, aerial views) and wider navigation. With segmentation, the people in the photo itself become inhabitants.
- ☐ **Direction by a language model**: the scene's operation schema under the JSON Schema constrained decoding that vapor already has, when a useful model is loaded; the vocabulary remains the fallback that reports what it does not understand.
- ◐ **Exact frames offline**: ✅ 0.12 — exact-frame GIF by vapor's encoder (up to 240 frames, fixed step). Missing: a loop that closes (the last frame = the first) and MP4.
- ☐ **Sketch → photorealistic** with a user checkpoint (img2img already exists) and the measurement that says whether the generated image respects the sketch (the vectorized lines of the output against those of the sketch).
- ☐ **Geometry**: Wu's triangulation (points by two quadratic conditions), inequalities, readable proofs; export certificates to Lean (when `lake` is present).
- ☐ **Discovery**: 3×3 (rank 23), the *flip graph* of Kauers & Moosbauer; minimum depth of the networks; synthesis with symbolic proof at 32/64 bits (bit-vectors) instead of a sample.
- ☐ **Science as vapor programs**: the *split-step* (DFT as `linear`), Boris and Lennard-Jones as programs — the same bits on every substrate; DFT of solids, electrolytes, rigid bodies.
- ✅ **Signed archives** (0.13): Ed25519 over the manifest with the operator's key; `verify(zip, trusted: …)`; `mix vapor.archive`. Missing: signing by KMS/HSM (the same item as for `Keys`).
- ☐ **Stronger controls in science**: the coherent state and tunnelling are compared with a reference, but the "control" of the first is a conservation and that of the second a computed prediction; E×B and HeH⁺ have none. A wrong scheme, run (Lie instead of Strang with a large step; a potential with its sign flipped), would be the real control.
- ☐ **Larger self-play**: a game whose state does not fit in memory (Connect-Four 6×7) and the network as a compiler program; environments generated by an adversary (PAIRED) beyond randomization.

## Round 0.10 — what was left open

- ☐ **Endless context in the paged engine**: `Vapor.Streaming` uses the dense session; what is missing is pinning the anchors' pages in the ring of `Vapor.Engine` and re-rotating per *slot* (many sequences), and measuring the effect of the anchors on a large model (weights this machine does not download) and with *needle-in-a-haystack*.
- ☐ **Real handwriting**: no handwriting reader is shipped (handwritten fonts: 64 % CER on never-seen hands). The path is to train `test/python/train_ocr.py` on IAM (Latin), KHATT (Arabic), CASIA-HWDB (Chinese) and a Cyrillic set, on a machine with network access — the reader's contract does not change.
- ☐ **Small and serif axis labels**: the digitizer refuses 12 of 30 charts of the hard set (it never reads wrong). A digit reader by shape models (like the CJK one) and segmentation of touching digits would close a good part of it.
- ☐ **Language models from another domain** for CJK (the current one comes from Faker's word lists: in Chinese, on a new seed, it does not help) and one for Arabic (in visual order).
- ☐ **Rigid-body physics** (the 0.11 science covers quantum, relativity, tokamak, chemistry and biology, not this): rotation and inertia, angular joints, contact between bodies and friction (XPBD treats them all the same way); the RL policy inside the program (one episode = one run).
- ☐ **Large networks**: sparse operators in the compiler (PageRank and SIR on millions of edges); personalized PageRank in the library (`Vapor.Docs.Library`) as a RAG ranking.
- ☐ **Formulas beyond the grammar**: matrices, accents, `\left…\right`, multiple lines; and handwritten formulas.

## Performance

- ◐ GPU: resident sessions in the fabric ✅ (0.8: `OPEN/STEP/CLOSE`, direct memory or *staging*, reused recordings, the engine serves on the GPU); missing are attention with *workgroup* memory (the end of `@max_dh = 512`) and the whole diffusion loop in one session (the protocol serves; it is not measured).
- ☐ FlashAttention as a declared `:fast` policy (the online softmax in blocks is another canonical order: it cannot be the canonical one).
- ☐ Mamba *prefill* in a single `RUN` frame with iterations (today one `STEP` per token); serving recurrent models in `Vapor.Engine` (state per *slot*).
- ☐ Tree speculation without recomputing the partial page (copy-on-write of the page) and integrated into `Vapor.Engine`; general trees (today: root-to-leaf branches).
- ☐ `sb4` GEMV with vectorized α/β and `k` not a multiple of 256; AVX-512 VNNI/AMX. (Predicated GEMV for `sb4`: ✅ 0.8.)
- ◐ Merging: 20× faster ✅; disk→disk *streaming* ✅ (0.7, element-wise methods, same bytes); missing is RegMean as a program in the worker (Cholesky and the calibration matrices; today `O(d³)` on the BEAM and with the models in memory) — requested in the 0.8 attachment, deferred with no measured pain.
- ◐ Tensor parallelism: across BEAM nodes ✅ (0.8: resident shards with SHA-256, *failover* without drift, replicas compared bit for bit); missing are attention sharded by heads and serving a whole model through `Vapor.Engine` on a cluster.

## Models and operators

- ◐ SSMs: Mamba ✅, **Mamba-2 ✅** (0.8); missing are Jamba/Zamba/Bamba (hybrids: KV cache **and** per-sequence state in the engine) and Falcon-Mamba (RMS of B, C, Δ inside the mixer); attention *soft-capping* (Gemma 2); `relu²`; dynamic NTK and LongRoPE (refused by name).
- ☐ 2D/3D RoPE (Qwen2-VL's M-RoPE, video DiTs) — when an admitted model needs it, checked against `transformers`.
- ☐ Unigram and WordPiece tokenizers (T5/BERT families).
- ◐ Image search by meaning: text in images via OCR ✅, CLIP vision **and text** towers checked against `transformers` ✅; missing is connecting the two towers to the library (`Vapor.Docs.Library`) — it only makes sense with trained CLIP weights, which are not shipped.
- ◐ Whisper: encoder + decoder checked ✅; missing are the log-mel front-end identical to that of `transformers`, the task/language/timestamp tokens of `generate` and a WER measurement with real weights.
- ☐ LLaVA (encoder + projector + `inject`) as an adapter checked against `transformers`.
- ◐ Diffusion: diffusers U-Net ✅, VAE encoder ✅, DDIM/Euler/DPM++ 2M ✅, txt2img/img2img/inpainting pipelines = diffusers ✅ (0.9); missing are **ControlNet** (a copy of the encoder + zero convolutions), diffusion LoRA, **SDXL** (two text towers, `add_embeds`), the 9-channel inpainting U-Net, Karras sigmas, DiTs with cross-attention to text (PixArt, SD3, Flux: refused with a near-miss) and **measuring a real SD** (quality and time per step) on a machine that downloads weights.
- ☐ Grouped/*depthwise* and transposed convolutions in `Vapor.Spatial`.

## Studio (0.9)

- ☐ Video: H.264/MP4/WebM. Our own encoder is a whole project, and ffmpeg through a *shell* violates the product's rule; the path is a **confined extractor** (below) that speaks the frame protocol.
- ☐ ComfyUI nodes beyond the subset: `LoraLoader`, `ControlNetApply`, `KSamplerAdvanced`, `UpscaleModelLoader`/`ImageUpscaleWithModel` (translatable to `image.upscale` when the model is ours), `SetLatentNoiseMask`. Each one translated from the documented semantics, with a test.
- ☐ The checkpoint's key in the cache: today it is the path; it should come to include the digest of the weights without re-reading them on every run (for example, the safetensors index and the `mtime`).
- ☐ Cheaper camera Lanczos: today the weights go through the correctly rounded sine, ~0.25 s per 320×192 frame on one core. The identity sin(π(d+k)) = (−1)ᵏ sin(πd) would reduce it to one sine per output pixel, but it changes the bits; it has to come in with a recalibration of the parity tests.
- ☐ Distillation robust to adversarial perturbations (the defensive sense, offered in DIRECTIVE §12): the student's loss under bounded perturbation, against the ordinary student.
- ☐ A measurement of a model's **over-refusal** (refusals on benign requests), as a gate calibrated with controls.
- ☐ RL: PPO and DQN as programs; LeRobot's dataset format for the demonstrations; a game environment.
- ☐ 3D: NeRF or *Gaussian splatting* checked against a reference; image → 3D only with trained weights.
- ☐ Signed context manifests for the MCP server (what an agent received, with a root) and SFT/DPO in `Vapor.Train`.

## Reading (documents and vision)

- ☐ **Confined pluggable extractors**: a launcher that applies seccomp to a foreign binary and speaks the worker's frame protocol — only then do external formats (CAD, camera RAW…) come in without running hostile code in the BEAM.
- ◐ OCR: horizontal printed text ✅; reading order in columns ✅ and CTC beam with a language model ✅ (0.7); **tables** with rules or ruling lines ✅ (0.8: exact structure on 12/12, per-cell CER 5.5 %); missing are tables without rules, tables across pages, short header tokens ("T1"), geometric fonts (URW Gothic: 7.5 %), handwriting, a corpus from the user's domain.
- ◐ Other writing systems: Arabic with RTL ✅, CJK ✅, Cyrillic ✅, formulas → LaTeX ✅ (0.10, [OCR.md §3g–§3k](OCR.md)); missing are a language model for Arabic and Cyrillic, Nastaliq, vertical CJK, multi-line formulas and matrices, and **real handwriting** (above, round 0.10).
- ◐ Speech: spoken digits ✅; missing are more voices, data augmentation and vocabulary beyond digits (or Whisper with real weights).
- ◐ Images from scans in PDF: CCITT Group 3/4 ✅, LZW and RunLength ✅ (0.7, = libtiff); **arithmetic JBIG2** ✅ (0.8, = jbig2dec); **JBIG2 Huffman and halftone** ✅ (0.15, checked by an independent encoder and by jbig2dec); missing are Huffman with refinement and JPX.
- ☐ Incremental index of the library and its persistence in the server.

## Numerics and verification

- ☐ **NTT kernel** in the worker: it requires high integer multiplication (64-bit `mulhi`) in the five encoders (today the exact NTT runs on the BEAM).
- ☐ Analytic envelope for `log`, `softplus` and `div` (today `:na` — the ladder measures, it does not bound).
- ◐ Lean: rewrite rules ✅ (0.8, `Binary32.lean`, finite values; NaN out by principle); missing are constant folding (today: the oracle's semantics, checked by a test) and the formal correspondence of the RVV emulator with the ISA (it would require the RVV 1.0 specification in Lean).
- ☐ DO-178C / IEC 62304 kits.
- ☐ Autodiff of attention, RoPE, gather and RMSNorm; multi-core/GPU training; `Vapor.Train` as an airlock *callback*.

## Agents, ecosystem, ZK

- ◐ Attestations anchored in a transparency log ✅ (`Vapor.Tlog`, witness, verifier in the browser); missing are C2SP *tiles* (`tlog-tiles`) for logs of hundreds of millions of entries and third-party witnesses actually running.
- ☐ *Lease* on the `intent`; KMS/HSM for `Keys`; journal with O(1) append; `fsync` of the directory in `Store.File`.
- ☐ Ecto/Postgres, Oban, LiveView tested with real services (official Ecto *store*, `UNIQUE(run_id, seq)`); zero-copy *streaming* in the Plug/Bandit transport; a Nerves image and AOT for microcontrollers (a new MVE/Helium *backend*). *Without those services or devices in this environment.*
- ◐ Audit dossiers ✅ (0.8); missing are mapping profiles for other standards (NIST AI RMF, ISO/IEC 23894) and signing by KMS/HSM.
- ◐ JSON Schema `pattern`/`format` ✅ (0.7: ECMA-262 → byte automaton; `date`, `time`, `date-time`, `uuid`, `ipv4`, `email`, `hostname`); missing are `ipv6`/`uri`, the intersection of `pattern` with `minLength`/`maxLength` (refused today) and `\b`/lookahead (not regular over bytes, or expensive).
- ☐ `s32 → s8` requantization; PLONK/STARK without per-circuit *setup*; propagated BFV noise bounds. *Refused for now: our own CKKS/BFV, a zkVM without a case, "70 B zkFHE".*

## Minor (each one, hours)

- ☐ Log-softmax of the checkpoint judge (`Vapor.Quality.Model`) on the substrate — now possible with the canonical `log`.
- ◐ Claim HF directories by the tensor index before reading the weights: done for *streaming* merging (`Lock.select/1` over the safetensors catalog); missing is using it in `Lock.open/2` to refuse before reading gigabytes.
- ☐ Export to HF the RoPE factors that arrived as a GGUF tensor.
- ☐ Image gate recalibrated with held-out photos (today: synthetic scenes).
- ☐ Seal the tool names in the journals when the policy requires it.
- ☐ Nx: `view` term, reductions over other axes, batched `dot`; zero-copy bridge with Python.
- ☐ Test `send_body/5` and the console token in the Plug transport (no Plug in this environment).

## Proposed priority

| | item | why |
|---|---|---|
| P0 | real GPU (sessions measured, attention with *workgroup*) | the resident session exists; the number that matters is missing |
| P1 | real silicon; Apple Silicon | removes the emulation asterisk; the largest fleet of developer machines |
| P1 | Engine with recurrent and hybrid models (Jamba); integrated tree; attention by heads across nodes | serving SSMs and hybrids in the engine; models larger than one node |
| P2 | ControlNet, SDXL, a real SD measured; DiTs with text; CLIP connected to the library | the studio with conditioned generation; semantic search of photos (with weights) |
| P2 | confined extractor for video (H.264/MP4) | the studio reading and writing the video that people have |
| P2 | confined extractors; tables without rules; JPX | the rest of the office scans, external formats without risk |
| P2 | order engine in the worker; real market data; integer generator (Philox) in the compiler | the trading desk with machine latency and data from the world; Monte Carlo with a modern generator |
| P3 | NTT kernel; Unigram/WordPiece; training; certification | larger scope, later return |
