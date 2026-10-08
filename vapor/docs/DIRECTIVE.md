# This round's directive, and its scrutiny

> Request (2026-10-02), translated: "Provide a refined zip at the end → this
> directive itself is subject to refinement and scrutiny → the result must be
> an artifact that solves the real pains of industry and academia with real
> innovation, lateral thinking and first principles + any to any, merging,
> solve the whole + other pertinent questions + no dependencies on prior
> knowledge of models in the core (do it with a model airlock, so that the
> specifics of each model and its topologies are abstracted in the airlock and
> it is easy to add support in the airlock for diverse and innovative new
> models) + benchmark test + quality test of the outputs of text and any to
> any models to ensure that it is not just generating noise".

## 1. Scrutiny

Each clause was read as a testable requirement. Where it was not, it was
reformulated — and the reformulation is here, to be contested.

| clause | literal reading | problem | refinement adopted |
|---|---|---|---|
| "solve the whole" | implement everything | unfalsifiable; leads to a façade (which the project itself forbids) | solve **the verifiable pains** and declare, with the reason, what stays out (§4) |
| "real innovation" | novelty | novelty is not a criterion; the criterion is a property that did not exist before and is now tested | each delivery has a test that would fail without it |
| "any to any" | every pair of modalities | `N²` converters is the anti-pattern; and "image" without a trained model is noise | **hub with a pivot** (`N` codecs), **no new operator** in the algebra, **measured** on held-out data with a control |
| "merging" | ambiguous | three senses: merging weights (*model merging*), merging modalities, fusing *kernels* | the first two (*kernel* fusion the compiler already does: *cut sweep*) |
| "no prior knowledge of models in the core" | intention | "core" and "knowledge" need an operational definition | core = 27 listed modules; knowledge = a reference to a family module; **test** over the BEAM atom table |
| "easy to add models" | intention | "easy" without a measure | three cost levels: data (JSON), *blueprint* (≈ 40 lines), topology; the cheapest one that serves |
| "quality test … not just noise" | a test | a test with a hand-picked threshold passes noise without anyone knowing | gates **calibrated against controls**, which refuse to exist if they do not separate; **planted** models with closed-form ground truth |
| "no dependencies" (implicit, project axiom) | — | — | kept: `deps: []`; PNG, PPM, WAV, FFT, k-means, Cholesky, ridge, CBOR — all here |

## 2. Real pains attacked

**Industry**

1. *Family coupling in the server* → airlock with contracts; the engine
   serves any `:causal_lm` that declares `:paged`, `:sample`, `:last`.
2. *Marginal cost of each new architecture* → alias by data (Phi-3 in 20
   lines of JSON), *blueprint* (Granite).
3. *Silent failure* → typed refusal with near-misses and repair; the
   GGUF exporter that wrote wrongly in silence now refuses (a real bug
   found in this round).
4. *Regressions that leave the output "plausible"* → quality gate in CI
   (`mix vapor.quality` exits 1), `--model` for real checkpoints.
5. *Merging without provenance* → signable and co-signable receipt, bits
   independent of the host.
6. *One runtime per modality* → the same certified compiler carries
   image and audio.

**Academia**

1. *Reproducibility of evaluations* → every number in the report is
   deterministic (native substrate = oracle, equal bits) and regenerable.
2. *Metrics without a control* → each check carries a declared control and
   threshold.
3. *Generalisation vs. memorisation* → held-out pairs, variants; the first
   version of the world measured recall (PSNR 155 dB) and was hardened before
   publishing any number.
4. *Method assumptions* → TIES/DARE measured where the sparsity assumption
   fails.

## 3. Lateral thinking, concretely

- **Bidirectional = causal with the horizon at the end.** No new mask.
- **Convolution with a *stride* = *kernel* = `linear` over rows.** No `Conv2d`.
- **VQ = greedy `sample` over `2x·c − ‖c‖²`.** The LLM's sampling operator
  becomes the image tokenizer.
- **Modality injection = `sel`.** The same operator that makes MoE
  immune to NaN from an unchosen expert.
- **Training = counting.** A bigram embedded exactly in a transformer is
  ground truth for the whole stack.
- **The test tests itself.** A gate that does not separate its controls does not exist.
- **Contract instead of family.** The program is already self-describing; the core reads
  sorts, not names.

## 4. What was left out, and why

| item | reason |
|---|---|
| image generation by diffusion, video | requires trained U-Nets/DiTs; the blocks can be expressed, nothing was built or verified — doing it now would be a façade |
| ComfyUI, *upscaling*, game rendering | product interface / dense convolution with no use case / rasterisation: see [ANY_TO_ANY.md §5](ANY_TO_ANY.md) |
| pre-trained Whisper, CLIP, LLaVA | path designed and pieces tested (encoder, spectrum, projector, injection); adapters not embedded without verification against `transformers` |
| parity of the new levels with `transformers` | no PyTorch in this environment; done against independent NumPy references; the `:torch` level closes it |
| fast merging of billion-parameter models | BEAM ≈ 0.6 M parameters/s; merging as a program on the worker is the path |
| "better" TIES/DARE | measured worse for dense models; they were not "tuned" to look good |

## 5. Response to the attached document ("Today: NO")

The document that accompanied the request described vapor as strictly
text→tensors→text. Item by item, after this round:

| claim in the attachment | now |
|---|---|
| "vapor has no audio or vision models" | **partial → yes for the infrastructure**: bidirectional encoder (HF ViT admitted and verified against NumPy), certified spectrum, synthesis, VQ codecs, projectors, injection; test world measured on 10 routes. Pre-trained audio models: not yet. |
| "`Conv2d`/`Conv3d` are missing" | **not needed** for *patch embedding* (= `linear`); overlapping convolutions are sums of shifted `gather_row` — expressible, not packaged |
| "cross-attention would be missing" | expressible without a new operator (attention over the K/V of another stream with the horizon at the end); the route used here is injection (LLaVA), tested |
| "patchification blocks and VAEs are missing" | patchification: `Vapor.Modal.Image.patches/3`, exact; VAE: no (the VQ codec takes the place of a discrete tokenizer) |
| "Model merging — NO, roadmap P3" | **yes**: six methods, deterministic, with a receipt, CLI, *streaming* |
| "ComfyUI / diffusion / upscaling / video — NO" | **still no**, for the reasons in §4 |
| "Games: rendering — NO; NPCs — YES" | unchanged |
| "GUI: Livebook, LiveView, Open WebUI, CLI" | **its own web console** at `/` (conversation with evidence, documents, contract, noise meter), plus `mix vapor.lock`, `vapor.merge`, `vapor.quality`, `vapor.rag` and the PNG/WAV gallery; Open WebUI and the like keep working through the API |

## 6. Second request: "UI/UX and RAG over zip, PDF, images etc."

| clause | reading | refinement |
|---|---|---|
| UI/UX | "an interface" | the interface missing from the existing ones is not another chat: it is the **evidence next to the answer** (sources with a path down to the page, checked citations, receipts) and a **noise meter** one click away; offline, no CDN, served by the server itself, usable without a model |
| RAG over zip | unzip and index | no *zip bomb* (ratio and ceiling checked before inflating, lying header refused), recursive, and with the path inside the archive preserved all the way to the receipt |
| RAG over PDF | extract text | correct where extractors fail (composite fonts via `ToUnicode`, *object streams*), explicit refusal of encrypted files, a warning on pages without text; checked against `pdftotext` |
| RAG over images | "understand images" | with no OCR or CLIP here, the honest thing is: embedded text (PNG/EXIF) in the text index and **visual similarity** in an index of its own — said in those words in the interface |
| "etc." | every format | Office, OpenDocument, EPUB, HTML, CSV/JSON, Markdown; and what was not done (OCR, JPEG, reading order of columns) is in the TODO, not hidden |

Details: [DOCUMENTS.md](DOCUMENTS.md), [CONSOLE.md](CONSOLE.md).

## 7. How to contest this document

Every claim above points to a test or to a regenerable number:

```sh
mix test test/vapor/lock_test.exs test/vapor/modal_test.exs test/vapor/merge_test.exs \
         test/vapor/quality_test.exs test/vapor/docs_test.exs test/vapor/console_test.exs
mix vapor.quality                       # docs/bench/QUALITY.md, quality.json, modal/
mix vapor.quality --model ./SeuModelo --text retido.txt --reference corpus.txt
```

## 8. Round 0.5.0: "attack the limitations"

> Request (2026-10-02, afternoon), translated: "Provide a refined zip at the
> end → this directive itself is subject to refinement and scrutiny → the
> result must be an artifact that solves the real pains of industry and
> academia with real innovation, lateral thinking and first principles
> (attack the limitations, the TODO, and complete whatever is pertinent … +
> quality tests to ensure the answers are not just noise + original and
> elegant UI/UX)", followed by the six limitations that 0.4.0 declared;
> then: "no loose ends + minimise the TODO + in the UI/UX English as default
> and Portuguese as alternative (besides light and dark) + create an SVG logo
> + a favicon + consider or not Tauri + GUI/TUI + make sure it is not
> generating output noise".

### Scrutiny

| clause | literal reading | problem | refinement adopted |
|---|---|---|---|
| "attack the limitations" | make the six sentences disappear | deleting a limitation sentence is easy; closing the limitation requires a test that would have failed before | each limitation becomes a new test **or** a measured refusal with a reason (table below) |
| "checked against NumPy that I wrote myself" | swap NumPy for PyTorch | the lesson is epistemological: an oracle written from the same reading shares the error | the oracle is now the **executed reference code** (`transformers` recording and computing); it found a real bug that NumPy could not find |
| "synthetic world" | use real data | real data without a control only changes the kind of noise | **held-out** real data (fonts, voices, digits never seen) **with** controls (chance, fluent wrong text, memorisation) |
| "there is no diffusion" | implement diffusion | diffusion without verification is a generator of pretty images | the sampler is tested against the closed-form optimal denoiser before any network; the network is measured by an independent judge and against copying |
| "there is no OCR" | add OCR | OCR through a dependency (Tesseract) contradicts `deps: []` and the airlock | OCR is a **model admitted** by the airlock; Tesseract stays as a comparison oracle in the tests |
| "search by colour, not by meaning" | semantic search | without CLIP weights in the environment, promising "semantic" would be a façade | the meaning that **exists** in document images is the text: OCR → index; CLIP vision tower checked; text tower declared in the TODO |
| "merging at 0.6 M/s" | faster | speed that changes bits destroys receipts | 20×, and **bit-for-bit equal** to 0.4.0 (tested against the old module) |
| "TIES and DARE got worse" | make them improve | tuning the benchmark until the method wins is the classic academic pain | measure where the assumption holds (fine-tunes) and where it does not (no common ancestor), and give the user the **diagnosis and selection by measurement** instead of a recipe |
| "minimise the TODO" | delete items | items deleted without delivery are hidden loose ends | close what fit, move what was closed to the CHANGELOG, keep what is open with the reason |
| "English default, Portuguese alternative" | translate | translating the page does not translate the server; prose generated on the server in a single language leaks | all page text in a dictionary; the diagnoses are **assembled on the client from the numbers**, in both languages; server messages stay in English (like the API) |
| "consider or not Tauri" | decide | — | decided **no**, by criteria (toolchains, `deps: []`, BEAM sidecar, Linux-only worker), with the dependency-free alternative adopted: the console as an installable app; revisit when there is a worker outside Linux |
| "GUI/TUI" | two interfaces | two interfaces with different logic diverge | the TUI is a pure interpreter over the same modules the GUI calls (`Vapor.TUI.eval/2`, tested without a terminal) |
| "make sure it does not generate output noise" | no noise | two readings: noise in the *content* and noise in the *channel* | content: every answer carries its measure (confidence, certainty, read-back, distance to training) and the suite fails outputs without signal; channel: no colour outside a terminal, `NO_COLOR` respected, CLIs without debug logs |

### The six limitations, after this round

| limitation (0.4.0) | now | evidence |
|---|---|---|
| Phi-3, Granite, ViT checked against our own NumPy | checked against `transformers` 5.18 executing; + partial rotary (bug fixed), ViT pooler, CLIP-vision | `lock_hf_test.exs` |
| synthetic any-to-any world | routes on held-out real data: handwriting (both directions), speech from a voice never heard, voice → drawing chain | `mix vapor.quality` §4c |
| no diffusion, OCR, JPEG | diffusion verified in closed form; OCR by an admitted model; JPEG = libjpeg bit for bit | `diffusion_test.exs`, `vision_test.exs`, `jpeg_test.exs`, §4c |
| image search only visual | text in images indexed by OCR; CLIP-vision checked; text tower open | `docs_test.exs`, `lock_hf_test.exs`, TODO |
| merging at 0.6 M/s | ≈ 11 M/s end to end, same bits | `merge_test.exs`, MERGING.md |
| TIES/DARE got worse | explained and measured in both regimes with trained models; diagnosis and selection by measurement | §4b, `merge_test.exs` |

## 9. Round 0.6: the two attachments — towards 1.0 and "Sora / Midjourney / world models"

> Request (2026-10-03), translated: the same as §8 ("refined zip … subject to
> refinement and scrutiny … real pains … quality tests so that the answers
> are not just noise … original and elegant UI/UX"), with an attachment of
> **fifteen items for 1.0** (sparse MoE by deterministic token permutation;
> MLA latent cache by query absorption; worker for Apple Silicon; pluggable
> document airlocks with extractors in a confined subprocess; resident
> sessions on Vulkan; circular KV cache; verified NTT kernel; correctly
> rounded canonical division; Higham's lemma in Lean; CLIP text tower + BPE
> with `</w>`; Whisper; Mamba/SSM; zero-copy Plug/Bandit *streaming*;
> official Ecto *store*; attestations anchored in transparency logs). Then:
> "continue from where you stopped, without loose ends and without abandoning
> the previous directives + weigh the new directives (subject to refinement
> and scrutiny, tested to avoid noise as an answer and benchmark)", with a
> second attachment on what would be missing to reach Sora, Midjourney and
> world models: `Conv2d`/`Conv3d`, continuous VAE, DiT, FlashAttention on
> Vulkan/CUDA, latent multimodal fusion by cross-attention (without a text
> pivot), spatial conditioning (ControlNet), long-term KV for video, tensor
> parallelism across nodes, speculative decoding trees.

### Scrutiny of the round

| clause | literal reading | problem | refinement adopted |
|---|---|---|---|
| "sparse MoE via deterministic token permutation" | sort the tokens by expert and dispatch in blocks, as GPU kernels do | on CPU the GEMV is *weight-stationary*: the cost is **reading the weights**, not grouping tokens; the permutation adds gather/scatter and a data-dependent shape, which breaks the certified extents | **row predication** (`linear_masked`): the same kernel skips the unchosen rows and reads only the experts that some token chose; bits **= dense** on every substrate; 2.6× in decode (T = 1) |
| "MLA latent cache by query absorption" | absorb `W_UK` into the query and keep only the latent | absorption changes the contraction order (other bits) and triples the attention FLOPs in exchange for 85× less memory — it is not a free optimisation | the latent form is **another program**, with its own bits, checked against `transformers` (same argmax, 3.5·10⁻⁷); the expanded one is still there (`mla: :expanded`); the choice is the operator's, with the cost measured |
| "worker for Apple Silicon" | port | no macOS here; on Darwin there is no seccomp (`sandbox_init` is deprecated), code generation requires `MAP_JIT` + `pthread_jit_write_protect_np` and waiting uses `os_sync_wait_on_address`; a worker that was never executed is worse than none | **not done, for lack of a machine** — plan in the TODO; the protocol no longer depends on the OS |
| "pluggable document airlocks (confined extractors)" | accept external extractors | an external extractor is **arbitrary code over hostile files** — exactly what the airlock exists to avoid; confining foreign binaries requires the seccomp filter *outside* the binary (a launcher) | **not done** — the readers remain pure Elixir, with no execution; the design (launcher with seccomp + frame protocol) is in the TODO |
| "resident sessions on Vulkan" | `OPEN/STEP/CLOSE` in the fabric | large work and only measurable with a real GPU (here: lavapipe, a CPU pretending to be a GPU) | **not done** (still P0); `STEP` gained the state feedback that the fabric will also need |
| "circular KV cache" | circular buffer | in contiguous memory, the ring breaks the canonical order (the window has to be read in position order) | **page ring**: the block table maps logical page `j` to `mine[j mod R]`, the logical order stays intact; `R = ⌈(w + T − 1)/page⌉` **proved tight** (one fewer changes the bits — control test); 7.9× more sequences at 32 k / window 4 k |
| "verified NTT kernel" | NTT on the worker | the exact NTT already exists (BabyBear/Goldilocks/BN254 on the BEAM); taking it to the worker requires high integer multiplication (64-bit `mulhi`) in the five backends — a new primitive in each encoder | **not done**; reason and path in the TODO |
| "correctly rounded division (0 ULP)" | IEEE | changes the bits of existing programs: requires a new version of the semantics (the certificates record which) | **done**: semantics version 2; Markstein + Dekker's exact residual, without FMA; 0 errors in 400 thousand pairs and in 50 M exhaustive; cost ≈ 2.2× `a·rcp(b)` |
| "Higham's lemma in Lean" | prove | — | **done**, kernel only (`propext`, `Quot.sound`) |
| "CLIP text tower + BPE `</w>`" | — | — | **done**: tokenizer = CLIP's on 20/20 lines; tower 3.7·10⁻⁷ against `transformers`. **Semantic** photo search needs trained CLIP weights (not shipped): it stays in the TODO with that reason |
| "Whisper" | adapter | without trained weights here, parity is not transcription quality; the log-mel front end is another piece | **done**: encoder (convolutions by `Vapor.Spatial`) + decoder with cross-attention, cross K/V computed **once** per audio; 4.8·10⁻⁷ / 4.2·10⁻⁷ and greedy decoding identical to `transformers`; controls (other audio, reversed frames) |
| "Mamba/SSM" | adapter | the parallel (associative) *scan* reorders sums: the bits would depend on the parallelism; and the paged engine does not serve state | **done**: a recurrent step (`t = 1`) is the definition — *prefill* and decode are the same instructions; the state stays on the worker (`STEP` feeds back `s ← s_next`); new canonical `log` and `softplus`; 3.4·10⁻⁷ and greedy identical to `transformers`; constant cost per token (measured) |
| "zero-copy Plug/Bandit *streaming*" | — | no Plug/Bandit in the environment (`deps: []`); Bandit has no HTTP/3 | **not done** (not testable here); our own server already streams SSE |
| "official Ecto *store*" | — | no Ecto/Postgres here; an untested adapter is a disguised loose end | **not done** |
| "attestations anchored in transparency logs" | blockchain? | a blockchain adds paid consensus and does not change the guarantee: what matters is **detecting a fork** (divergent views of the log), and witnesses do that (RFC 9162, C2SP) | **done**: `Vapor.Tlog` (Merkle RFC 9162, inclusion and consistency proofs, C2SP *signed-note* *checkpoints*, witness co-signatures), anchored search receipts, verifier **in the browser** (WebCrypto Ed25519, key pinned on first use); 196 probes from transparency-dev |

| clause of the 2nd attachment | literal reading | problem | refinement adopted |
|---|---|---|---|
| "`Conv2d`/`Conv3d`" | new kernels | five backends × verification × policy per kernel | **no new kernel**: im2col = `gather_row` + `sel` (padding exactly `+0`) + `reshape` + `linear`; same bits on every substrate; ≤ 4·10⁻⁷ against torch |
| "continuous VAE" | adapter | without trained weights, no claim of image quality is possible | `AutoencoderKL` (decoder) from diffusers, 8.7·10⁻⁷ against diffusers |
| "DiT" | adapter | likewise | `DiTTransformer2DModel` (adaLN-Zero), 3.1·10⁻⁷; position table **bit for bit**; *timestep embedding* 8.3·10⁻⁷ |
| "FlashAttention on Vulkan/CUDA" | kernel with online softmax in blocks | online softmax reduces maximum and sum in blocks — **another canonical order**, other bits than the other substrates; CUDA: no NVIDIA GPU here | **not done**; it would only make sense as a declared `:fast` policy; the window and the paging already cut the reading where it matters |
| "latent multimodal fusion by cross-attention" | new operator | — | it is **the existing attention** (the K/V of another stream, the whole horizon on the last row): ≤ 10⁻⁶ against `nn.MultiheadAttention`; it is what Whisper uses |
| "spatial conditioning (ControlNet)" | adapter | ControlNet = a copy of the U-Net encoder + zero convolutions; there is no U-Net adapter yet | **not done**; the blocks (conv, GroupNorm, per-pixel attention, upsampling) exist and are checked |
| "long-term KV for video" | long memory | "long term" without a criterion becomes an infinite cache | three measured answers, each with its trade-off: **window + ring** (`O(w)` memory), **latent** (85×), **fixed state** (SSM). A video world model: out of reach without data and training |
| "tensor parallelism across nodes" | Megatron | row-parallel + all-reduce **changes the bits** (measured: 13,284 of 16,384) | column-parallel + **all-gather**: exact; each shard is a worker (port), on the same machine or on another |
| "speculative decoding trees" | tree of drafts | a tree without a guarantee becomes an approximation | branches in *slots* over the **shared pages** of the context (no copy; the partial page is recomputed by each branch); the output **is** the target's greedy one (tested on five tree shapes); draft by **prompt lookup** with overlapping copy: 6 tokens/step where the output follows the context, 1.2 with random weights — measured |

### What the scrutiny found along the way (without being asked)

- **DeepSeek with `n_group = 1`** (DeepSeek-V2-Lite) broke the construction
  of the program: the group limiting had no terms. Now it is the identity
  (same bits as "all groups kept"), with a test.
- **Options ignored in the engine tests**: the *helper* joined the defaults
  *before* the options (`++`), so `sequences: 1`, `step_tokens: 5` etc.
  never took effect — the invariance tests passed without testing what they said.
  Fixed (`Keyword.merge`) and re-verified.
- **A request larger than the whole pool** waited in the queue forever; now
  it is refused (`{:kv_pages, need, pool}`).
- **`softplus` as a single microprogram did not fit in the registers** of a
  kernel: it became a composition of canonical nodes (same bits as the microprogram).
- **`layer_types` ignored outside Gemma 3**: a hybrid Mistral (global
  layers between the sliding ones) would be read as all sliding — computed
  wrongly and with recycled pages that the global layers would still read. Now
  the key is read or refused, never skipped.
- **The engine read the family's configuration** to decide the ring — the test that
  audits the core's atom table caught it. Now it asks the adapter
  (`Vapor.Lock.ring_window/2`, optional *callback*); the core still does not
  know any family.

### What this document does not claim

- **No image, video or transcription quality.** VAE, DiT, Whisper and
  Mamba are checked **against the reference implementations** with random
  weights; quality requires trained weights, which are not in this environment.
- **Sora and Midjourney were not reached.** What was missing *in the runtime*
  (convolution, VAE, DiT, cross-attention, long memory, exact parallelism)
  now exists and is verifiable; what separates a video model from that is
  data, training and compute — outside the scope of a compiler.
- **The measurements are from this VM** (2 vCPUs, no GPU): [bench/FRONTIER.md](bench/FRONTIER.md).

### How to contest

```sh
mix test test/vapor/sparse_experts_test.exs test/vapor/latent_attention_test.exs \
         test/vapor/sliding_window_test.exs test/vapor/engine_test.exs test/vapor/division_test.exs \
         test/vapor/tlog_test.exs test/vapor/spatial_test.exs test/vapor/mamba_test.exs \
         test/vapor/speculative_tree_test.exs test/vapor/shard_test.exs
mix test --include torch test/vapor/mamba_hf_test.exs test/vapor/whisper_hf_test.exs test/vapor/lock_hf_test.exs
mix vapor.bench --frontier             # docs/bench/FRONTIER.md
mix vapor.quality                      # §5b of docs/bench/QUALITY.md: each feature against a control
```

## 10. Round 0.7: the same request, for the third time — and the office scan

> Request (2026-10-03), translated: "Provide a refined zip at the end → this
> directive itself is subject to refinement and scrutiny → the result must be
> an artifact that solves the real pains of industry and academia with real
> innovation, lateral thinking and first principles (attack the limitations,
> the TODO, and complete whatever is pertinent and subject to refinement +
> quality tests to ensure the answers are not just noise + original and
> elegant UI/UX)".

### Scrutiny

| clause | literal reading | problem | refinement adopted |
|---|---|---|---|
| the request repeats that of 0.5 | redo 0.5 | repeating the same deliveries is noise; what changed were the limitations declared by 0.6 | attack **the limitations declared in the README and in the TODO of 0.6** that this machine can verify; say which ones stayed and why |
| "attack the limitations" | all of them | half require hardware that is not here (real GPU, Apple Silicon, RVV in silicon) or trained weights (Whisper/CLIP/VAE with quality) — "doing" without being able to execute is a façade | chosen by **verifiability here** and by **pain**: the office scanned PDF (CCITT, columns, language-less CTC), structured output with invalid fields, merging that does not fit in memory |
| "real innovation" | something new | showcase novelty is not a criterion | a property that used to fail and is now tested **with a control** (the naive form failed in the same test) |
| "quality tests … not noise" | measure the output | a language model is the classic example of **plausible noise**: it improves the average metric and invents text where there is no language | every improvement of the reader is measured together with its failure mode: random strings (the model has to abstain), codes and values (it cannot get worse), shuffled corpus (the gain has to come from the language) |
| "original and elegant UI/UX" | redesign | the console already has an identity (the airlock: each result at a water level); changing the skin would be work with no pain solved | the interface gains **what the evidence needs to show now**: the reading order (numbered blocks and the reading thread) and *who decided each letter* (frames or language model), with the alternative reading one click away |
| "the TODO, and complete whatever is pertinent" | everything | — | close what fit, with a test; the rest stays in the TODO with the reason (§ below) |

### Pains attacked and what proves them

| pain | before (0.6) | now (0.7) | proof (test / suite §5c) |
|---|---|---|---|
| B&W *scanner* PDF (CCITT) | "an image in CCITTFaxDecode (not decoded here)" | Group 3 1-D/2-D and Group 4 decoded, LZW and RunLength | 42 streams = libtiff bit for bit, LZW/PackBits = libtiff; `ccitt_test.exs` |
| 2–3 column document | lines crossing columns: CER 73 % on a page that, in order, gives 1.3 % | XY-cut with gutter before gap, thresholds from the region itself | 8 scanned pages: CER 1.3 % (without order: 53 %; Tesseract: 1.6 %); `reading_test.exs` |
| "c1áusula", "rão", "agerdar" | greedy CTC | CTC beam + character language model, which **abstains** where there is no language and only chooses among what the frames find plausible | held-out lines CER 6.5 % → 4.4 %; random strings: 0 lines changed (without the guard: 27); shuffled corpus: no gain |
| valid JSON output with an invalid field | `pattern`/`format` refused | ECMA-262 regex and formats → byte automaton | same verdict as Python's `re` and the standard library on 7,240 strings; `regex_test.exs` |
| merging models larger than RAM | three copies in memory | `Merge.stream/3`: one tensor at a time | files byte for byte equal to the in-memory merge; 4 MB × 83 MB peak; `merge_stream_test.exs` |

### What the scrutiny found along the way (without being asked)

- **Stencil masks (`ImageMask`) read inverted** in the PDF reader —
  untested until now. Fixed and tested (Flate and CCITT).
- **Accents discarded** on lines without tall letters: the line finder
  accepted components up to 3 px from the core of the line; the tilde and the cedilla of
  "mesma execução não anunciam a mesma ação" were left out and the reader saw
  "mesmã eeeução rão anuneiãm". Found **by looking at the new interface**. Now the
  reach is relative to the text height; the greedy CER of the held-out lines dropped
  from 6.8 % to 6.5 %, and that of the random strings from 16.0 % to 10.7 %.
- **Three failure modes of the language model**, each became a tested rule:
  rewriting random strings (a guard in bits/character), erasing letters
  read with certainty (only plausible candidates, the *blank* included), and bringing
  in the corpus's domain — Markdown backticks invented on printed pages
  (the model is counted over the text as printed; the cost on the set
  rendered from Markdown is stated in [OCR.md §3b](OCR.md)).
- **The title cut in half** by the first version of the XY-cut (the space between
  words of a large title looked like a gutter): the thresholds now come
  from the region itself.

### What this document does not claim

- **No OCR of tables, handwriting or JBIG2.** The XY-cut reads a table
  column by column; JBIG2 is still refused with a warning.
- **The language model is not "intelligence"**: 5-grams from 27 thousand words.
  It makes mistakes (there is a "retomam" → "retoman" on a test page, visible in the
  interface). What is claimed is the measure, with the controls.
- **Tesseract is still better** on English pages with common fonts;
  vapor is better where the Tesseract here does not have the language (Portuguese) and on the
  photo with uneven lighting.
- The hardware items (GPU, Apple Silicon, RVV in silicon) and the trained-weight
  items remain in the [TODO](TODO.md), with the reason.

### How to contest

```sh
mix test test/vapor/ccitt_test.exs test/vapor/reading_test.exs test/vapor/regex_test.exs test/vapor/merge_stream_test.exs
python3 test/python/scan_pages.py /tmp/scans --tesseract     # regenerates the scanned pages and the Tesseract readings
python3 test/python/ocr_render.py /tmp/sets val 160 101      # the set on which the language model's weights were chosen
mix vapor.quality                                            # §5c of docs/bench/QUALITY.md
```

## 11. Round 0.8: the "Vapor 1.0" roadmap — what went in, what was refused, and why

> Request (2026-10-03), translated: the same text as 0.7 ("Provide a refined
> zip at the end → this directive itself is subject to refinement and
> scrutiny → … attack the limitations, the TODO, and complete whatever is
> pertinent (which interesting features are missing?) … + quality tests to
> ensure the answers are not just noise + original and elegant UI/UX"), with
> an attachment: a roadmap for "a definitive Vapor 1.0" in five areas and
> seventeen items.

### Scrutiny of the request

| clause | literal reading | problem | refinement adopted |
|---|---|---|---|
| the attachment as a specification | do the 17 items | half require what this machine does not have (Apple Silicon, real GPU, Postgres, a Nerves device) or trained weights that do not exist here (Arabic/CJK OCR, formulas, text→video); "doing" without being able to execute or measure is a **façade**, which the project forbids | each item was read as **pain + criterion verifiable here**; what can be checked with a control went in; the rest was refused **by name**, with what would close it (table below) |
| "bulletproof", "100 % of PDFs", "definitive platform" | absolute goals | unfalsifiable; "100 % of scanned PDFs" is false even with JBIG2 (JBIG2 Huffman, halftone, JPX remain) | replaced by measures: streams checked bit for bit against the reference, and the list of what is still refused |
| "which interesting features are missing?" | suggest more | a wish list is noise | only gaps that **the scrutiny found** while executing go in (§ "findings") — each one became a test |
| "quality tests … not noise" | measure | an image decoder, a model adapter, a dossier: each has a way of "passing" while being wrong | each new delivery has the control that the wrong form would produce (§5d of the suite): the generic model declared wrong in JBIG2, the norm on the other side in Mamba-2, the zeroed state on the GPU, the swapped shard in the cluster, the swapped byte in the dossier |
| "original and elegant UI/UX" | redesign | the identity (the airlock: each result at its water level) already exists | the console gains what the new evidence needs to show: the table drawn from its cells and linked to the page; the **weave** of evidence × device, which shows the gap instead of hiding it |

### The seventeen items

| attachment item | decision | what proves it (test / suite §5d / bench) | scrutiny |
|---|---|---|---|
| Resident sessions on the GPU (P0) | **done** | `OPEN/STEP/CLOSE` in the fabric; direct memory or *staging*; recorded *command buffers* reused by the key of the exact bytes; 72.4 → 12.4 ms/token, 1 MB → 8 kB per token; whole engine on the GPU with the tokens from the CPU; driver crash → `:session_lost`, never the BEAM. `gpu_session_test.exs`, bench §1 | the attachment asked for *timeline semaphores*: unnecessary — one *fence* per step and the recording cache suffice; what matters is not moving the KV cache. Measured on lavapipe (CPU): the throughput says nothing about a real GPU |
| Apple Silicon / Metal | **refused** | — | no Mac here; MoltenVK would solve Vulkan but not the *sandbox* (seccomp is Linux). It stays in the TODO with the design (MAP_JIT, `posix_spawn` without rights) |
| Sparse 4-bit GEMV (`qgemv_masked`) | **done** | `gemv_sb4_masked` kernel on x86/AVX-512/NEON/RVV/SPIR-V; bits = dense; 1.6–2.6×; 15.0 M → 5.3 M instructions. `sparse_sb4_test.exs`, bench §2 | "unlock DeepSeek-V3" is a scale claim that a reduced model does not prove; what is claimed is the measured ratio |
| RegMean Cholesky on the worker | **postponed** | — | no measured pain in this round (the test models are small); TODO |
| Universal OCR (Arabic, CJK, cursive, Cyrillic) | **refused** | — | the limit is neither the geometry nor the CTC: it is a reader **trained** on those writing systems, with data that does not exist here. A 64 px strip and 10 thousand classes without training = noise that looks like output — the opposite of what the suite exists to prevent |
| Tables (TSR) | **done** | grids with rules and merged cells, rule-only tables, typed columns and forms; structure F1 1.000 (0.7: 0.343), per-cell CER 5.5 % (free: 11.7 %); Markdown, HTML, CSV, JSON through the API. `table_test.exs`, [OCR.md §3e](OCR.md) | "Ecto schemas" refused (the core does not depend on Ecto); Tesseract reads the cells better even so (2.3 % with perfect boxes) — stated |
| JBIG2 | **done, arithmetic** | MQ, generic, MMR, refinement, symbols, text, PDF with globals; 43 streams = jbig2dec bit for bit (control: 19/43). `jbig2_test.exs`, [OCR.md §3f](OCR.md) | Huffman and halftone refused by name: no encoder here emits them, decoding without checking would be a façade |
| Formula OCR → LaTeX | **refused** | — | the "restricted grammar" part exists (decoding inside a form, §3e, is a CTC automaton); what is missing is a trained reader of mathematical symbols |
| Text → video | **refused** | — | the blocks (conv3d, causal VAE, DiT) are checked; without trained video weights the output is noise, and the directive asks for the opposite |
| Mamba-2 | **done** | adapter through the airlock; = `transformers` (7.8·10⁻⁷), identical greedy, native = oracle, GPU = CPU. `mamba2_hf_test.exs`, `gpu_session_test.exs`, suite §5d | it found two divergences of `transformers` from the training code (below) |
| Hybrids (Jamba, Zamba, Bamba) | **postponed** | — | they require a KV cache **and** per-sequence state in the engine; refused with a near-miss by the airlock |
| 2D/3D RoPE | **postponed** | — | no admitted model needs it yet; when one does (Qwen2-VL M-RoPE), it is checked against `transformers` like the others |
| Lean: rewrite rules | **done** | IEEE-754 model from the bit patterns (Lean kernel, no Mathlib): `x·1`, `1·x`, `x+(−0)`, `(−0)+x`, `x−(+0)` — **the only correctly rounded result is `x`**; `x+(+0)→x` **refuted**; `units` extracted and checked against `Vapor.F32`. `proofs/Vapor/Binary32.lean`, `rewrite_soundness_test.exs` | the attachment asked for "including NaNs": **impossible** bit for bit across substrates (x86 quiets the sNaN, RISC-V returns the canonical NaN) — and the 0.7 documentation claimed it; fixed and tested |
| Lean: RVV emulator | **refused** | — | it would require the formal specification of RVV 1.0 in Lean (the Sail model is not in Lean); the emulator remains checked against QEMU |
| Lean: Wilkinson with a window | **already covered** | `withinEnvelope_mono` (Lean, 0.5): the envelope of a sum of `w` terms is dominated by that of `n ≥ w`; the exact window is a shorter sum in the same canonical order (`sliding_window_test.exs`) | there is no new theorem to prove — saying otherwise would be inflating |
| Ecto/Postgres for `Agent.Store` | **refused here** | — | no Postgres in this environment; an adapter not tested against the database is exactly what the project does not publish |
| Tensor parallelism across BEAM nodes | **done** | `Vapor.Shard.Cluster` with real `:peer` nodes: bits of one worker on 1, 2, 3 nodes; lost node → shards relocated, same bits; replicas compared bit for bit catch a node that corrupts a bit; shard with a wrong SHA-256 refused. `shard_cluster_test.exs`, suite §5d | attention sharded by heads across nodes stays in the TODO |
| EU AI Act / ISO 42001 kit | **done** | `mix vapor.audit export/verify/demo`: signed and co-signed dossier, Merkle root, anchor in the log, each item checked by its own rules; PDF with the dossier attached; HTML that checks itself offline in the browser; *Dossier* panel with the weave. `audit_dossier_test.exs`, [AUDIT.md](AUDIT.md) | the command is `mix vapor.audit export` (not `vapor.audit.export`); **it is not a conformity assessment** — the warning goes in every dossier |
| AOT for Nerves | **refused** | — | no device; Cortex-M does not have the vector ISAs that the emitters cover (it would be a new *backend*, MVE/Helium), and without checking on the target there is no certificate |

### What the scrutiny found along the way (without being asked)

- **`transformers` and the Mamba-2 training code disagree on the gated
  norm** when there is more than one group (per group × whole width): the default
  here is the training one, the other is an option, and each fails against the reference
  of the other (error 0.74/0.66) — the comparison discriminates.
- **The cached step of `transformers` (Mamba-2) skips the `time_step_limit`**
  that its *scan* applies: its `generate()` mixes two semantics.
- **"x · 1 → x exact, including NaN"** was written in `Rewrite` since
  0.5: false for sNaN on x86 and for any NaN with a payload on RISC-V. The
  Lean model covers the finite values and says why NaN stays out; a test
  prevents the sentence from coming back.
- **The quality suite called `System.cmd("epmd")`** — the source audit
  test (no *shell* in the product) caught it. Removed: without epmd, the
  cluster check does not run and the report says so.
- **Old `transformers` configurations write `Infinity`** (invalid
  JSON) in `time_step_limit`: the `config.json` reader now accepts Python's
  non-finite values **only there** (`nonfinite: true`); strict JSON
  remains strict.
- **jbig2dec and pdf.js disagree** on the TPGRON contexts (template 1), and
  jbig2dec uses the whole page as the reference of a refinement without
  offset: we follow what can be checked (jbig2dec), documented.
- **Looking at the new interface**: the header "T1" of a table read as "TP1"
  without low confidence — the typed steps only apply in the body; stated in
  [OCR.md §3e](OCR.md) instead of hidden.

### What this document does not claim

- **Real GPU**: everything that is GPU here runs on lavapipe; the equality of the bits
  and the protocol are proved, the speed on a real GPU is not.
- **Regulatory conformity**: a dossier authenticates evidence; the
  legal sufficiency is not its business.
- **OCR better than Tesseract** on the cells: it is not (5.5 % × 2.3 %); it is the
  structure, which Tesseract does not give, that is exact on the 12 tables.
- **Mamba-2 with real weights**: checked with random weights around
  the `transformers` initialisation, like the other adapters; no perplexity
  number from a published checkpoint was measured here.

### How to contest

```sh
mix test test/vapor/gpu_session_test.exs test/vapor/sparse_sb4_test.exs test/vapor/table_test.exs \
         test/vapor/jbig2_test.exs test/vapor/mamba2_hf_test.exs test/vapor/rewrite_soundness_test.exs \
         test/vapor/shard_cluster_test.exs test/vapor/audit_dossier_test.exs
cd proofs && lake build                                       # Binary32.lean among the proofs, with no sorry or axiom
python3 test/python/hf_mamba2.py /tmp/m2 mamba2-g2 11         # the Mamba-2 reference, with and without the per-group norm
python3 test/python/table_render.py /tmp/val --seed 2028      # the validation set for the table thresholds
mix vapor.audit demo && mix vapor.audit verify _build/audit-demo/dossier.vdossier
mix vapor.bench --round08                                     # docs/bench/ROUND08.md
mix vapor.quality                                             # §5d of docs/bench/QUALITY.md
```

## 12. Round 0.9: "something similar to ComfyUI, any-to-any, and everything from the Hugging Face courses"

> Request 1 (2026-10-03, in the middle of round 0.8), translated: "weigh
> support for adversarial distillation and removal of watermarks and model
> censorship".
>
> Request 2 (2026-10-04), translated: "my intention was completeness and a
> philosophy of technology neutrality; in that sense, let's go after making
> possible something similar to Comfy with any-to-any, audio, video and
> images and editing of these + pertinent innovative features that solve
> pains and utilities in the agentic part and that cover everything from the
> Hugging Face courses (games, 3D, deep RL etc.) + perhaps AI upscaling from
> any source with innovation and quality" — with the links to twelve courses.

### The record of the decision on request 1

Three things were refused, by name:

- a tool to **remove watermarks** from model outputs;
- a tool to **remove the refusals** ("censorship") of models;
- **cloning closed models through their APIs** (adversarial distillation
  in the sense of extracting a third party's model).

The reason is not the subject. vapor trains, distils, merges and runs the user's
weights without a content filter of its own, and that stays so. The reason is the
function: the main use of these three tools is to undo the control that a
third party placed on what is theirs, be it the provenance of a piece of content, the policy
of a model or the terms of an API. The whole project exists for the
opposite. Receipts, the transparency log, audit dossiers and now the
Merkle root of each studio execution are **provenance** machines.
A tool for erasing provenance inside them would contradict what they
claim.

"Technology neutrality" was taken seriously as a design principle and
read like this:

- the **general tools** have no opinion: the studio runs any of the user's weights,
  any prompt, any graph;
- the **single-purpose tools**, designed to defeat someone else's
  safeguard, are not neutral: they are that function.

What was offered instead, and where it is:

| alternative | state |
|---|---|
| distillation from **local weights** | already exists: `Vapor.Train` (LoRA by KL distillation as a recurrent program) |
| distillation **robust to adversarial perturbations** (the defensive sense of the term) | TODO, with the criterion: the student's loss under bounded perturbation, against the ordinary student |
| **measuring the over-refusal** of a model (refusals on benign requests) | TODO: a gate calibrated like those of `Vapor.Quality`, with controls |

### Scrutiny of request 2

| clause | literal reading | problem | refinement adopted |
|---|---|---|---|
| "something similar to Comfy" | clone the interface | ComfyUI's value is the ecosystem: thousands of nodes and models. A copy of the screen without that would be a façade. Reproducing its code is not the path either | a graph of typed nodes with what ComfyUI does **not** have: exact content-addressed cache (Merkle keys), receipts per output, a root per execution, `verify` without cache, whole refusal of badly typed graphs. The **import** of ComfyUI workflows (API format) translates the subset whose semantics are fixed and flags each translation; a node without a translation refuses the whole workflow. The translations were written from the documented semantics of the nodes, without copying code |
| "any-to-any, audio, video and images and editing" | every format, every edit | "every" is unfalsifiable; H.264/MP4 are huge, and ffmpeg through a *shell* violates the product rule, which does not execute foreign processes | 64 nodes in image, sound, video, vision, 3D, RL and diffusion, each family checked against a reference (torch, Pillow, libjpeg, ffmpeg, gymnasium, trimesh, diffusers); own codecs (GIF, Y4M, MJPEG-AVI); H.264/MP4/WebM refused by name |
| "innovative features … pains in the agentic part" | more functions | a wish list is noise | the pains were named one by one and each got an MCP tool with a test: the agent that guesses parameters (typed catalogue), the error discovered late (validate before), the retry that recomputes everything (the cache lives between calls: 2 of 5 nodes), the result that cannot be checked (root + verify), the invented citation (search with a Merkle proof) |
| "cover everything from the courses" | twelve complete courses | unfalsifiable, and part of it requires simulators, hardware or trained weights that this machine does not have | a **coverage map** ([STUDIO.md §6](STUDIO.md)): per course, what exists and is checked and what is missing, with the reason |
| "AI upscaling from any source with innovation and quality" | a super-resolver that improves everything | "quality" without a measure is noise, and a GAN that invents texture passes the eye test and fails on fidelity | an upscaler with a **guarantee**: D(y) = x by construction, that is, downscaling the result returns the input. Trained here and reproducible. Measured on held-out images against the baseline with the same guarantee: +1.5 to +5.8 dB on text and on the phantom, −0.2 to +0.1 dB on photographs, and −1.4 dB on the colour wheel, a smooth gradient on which both exceed 53 dB (stated, not hidden). "Any source" = image or video from any codec read here, colour or grey, ×2 or ×4 |

### Deliveries and what proves them

| delivery | proof (test / suite §5e) | control |
|---|---|---|
| Stable Diffusion: U-Net, VAE encoder, three *schedulers*, txt2img/img2img/inpainting pipelines | = diffusers at ~10⁻⁶ in all modes, on the tiny checkpoint included. `diffusion_pipeline_test.exs` | DPM++ against the DDIM reference: 0.096 |
| The studio | two executions without cache = one root; editing a parameter recomputes 3 of 6 nodes | another seed changes the root; without cache, 6 of 6 |
| The ComfyUI import | the txt2img workflow runs and gives **the same bits** as the pipeline | a node without a translation refuses the workflow, named |
| The consistent upscaler | +2.76 dB (worst case on text) over Lanczos + projection; \|D(y) − x\| = 1.1·10⁻¹⁶ | Lanczos: 0.077 |
| RL | Q-learning = value iteration (74.7 %); CartPole 472.9/500 | always left: 0 %; untrained: 18 |
| 3D | closed sphere, volume within 0.4 % | one face fewer: not closed |
| Sound | 69.6 dB SNR in resampling | misaligned decimation: 17.7 dB |
| MCP server | the SDK's official client; the same execution again is all cache; verify accepts the root | wrong root refused |
| Console: *Studio* | node canvas; six starter templates; previews; seal and verification. `console_test.exs` | — |

### What the scrutiny found along the way (without being asked)

- **Each diffusers *scheduler* spaces the *timesteps* in its own way.** In
  `leading`, DPM-Solver divides by `steps + 1`; in `linspace`, Euler
  keeps fractional *timesteps* and interpolates σ; DDIM uses `steps` points,
  not `steps + 1`. A single implementation gets the latents wrong by up to 0.19.
  Now each one reproduces its own to 6·10⁻⁷.
- **diffusers' img2img and inpainting sample the VAE latent**, a
  hidden randomness that the user's seed does not control. vapor uses
  the mean; for the comparison, diffusers was forced to the mean.
- **SD checkpoints ship only the slow tokenizer files**
  (vocab.json + merges.txt, no tokenizer.json). vapor builds the tokenizer
  that `transformers` would write; the test checks the ids.
- **The digest of a long video filled the *heap* with garbage.** A
  160-step CartPole episode took 36 s, almost all of it in garbage collection. The hash
  now comes out of the external term format, outside the *heap*: 2.8 s. The definition
  of the image digest changed and is documented in `Studio.Value`.
- **A fully cached execution took 19.6 s in the console.** The previews were
  redone and the digests recomputed every time. The previews are now
  memoised by digest, and the cache keeps the digests: 1 s.
- **ComfyUI's KSampler is not the diffusers sampler.** The sigmas are
  different, and the pixels differ. Each translation says so; nothing claims equality
  with ComfyUI.
- **Focusing a node scrolled the graph canvas** (`overflow: hidden` scrolls even
  so) and detached the wires. Scrolling now becomes a canvas offset, and the
  focused node stays in view.
- **The consistency projection alone improves Lanczos** by 0.2–1 dB. It
  is the fair baseline, and it is against it that the upscaler is measured.

### What this document does not claim

- **SD with real weights**: the parity is with random weights. Neither the
  image quality of a published checkpoint nor the speed of a 512×512 SD
  on the CPU were measured here.
- **Equality with ComfyUI**: neither in the pixels nor in node coverage; the
  import is of a subset and says which.
- **The upscaler being better on photographs**: it ties, and loses 1.4 dB on a
  smooth gradient. It is also not a GAN, and does not invent detail.
- **Complete coverage of the courses**: the map says what is missing (PPO/DQN,
  real LeRobot, NeRF/splatting, ControlNet, SDXL, TTS…).

### How to contest

```sh
mix test test/vapor/diffusion_pipeline_test.exs test/vapor/studio_test.exs test/vapor/studio_media_test.exs \
         test/vapor/upscale_test.exs test/vapor/rl_test.exs test/vapor/geom_test.exs \
         test/vapor/mcp_server_test.exs test/vapor/console_test.exs
python3 test/python/diffusers_pipeline.py /tmp/sd        # the tiny checkpoint and the diffusers references
python3 test/python/diffusers_scheduler.py dpmpp_2m leading 10
mix vapor.upscale eval DIR                               # DIR from test/python/upscale_data.py
mix vapor.rl eval
mix vapor.quality                                        # §5e of docs/bench/QUALITY.md
```

## 13. Round 0.10: substrates, training, OCR of other scripts — and, in the middle, physics, networks and endless context

> Request 1 (2026-10-04), translated: "the result must be an artifact that
> solves the real pains of industry and academia with real innovation,
> lateral thinking and first principles (attack the limitations, TODO, and
> complete whatever is pertinent) + quality tests to ensure the answers are
> not just noise + original and elegant UI/UX + consider: Metal (Apple)
> support + Tenstorrent + vapor cluster orchestration + cursive, Arabic, CJK
> and figure OCR + complete training pipeline, with HPC". And: "this
> directive itself is subject to refinement and scrutiny".
>
> Request 2 (in the middle of the round, with an attachment on "inverting
> RoPE" for infinite context), translated: "incorporate the idea + Cyrillic +
> Cyrillic, Arabic and CJK handwriting and LaTeX + FreeBSD support + weigh a
> physics engine in vapor (in the sense of *reinforcement learning* and
> *digital twins*) + also weigh applications of complex networks".

### Scrutiny of the request

| clause | literal reading | problem | refinement adopted |
|---|---|---|---|
| "Metal support" | a Metal *backend* | without a Mac, an unexecuted *backend* would be a façade | MSL translator from the same kernel library; compiled Metal daemon; **the same MSL text executed** through a header *shim* with clang, on three "devices" (conforming, contracting, flushing subnormals to zero); and a gate that admits any device **by measurement** ([SUBSTRATES.md](SUBSTRATES.md)) |
| "Tenstorrent" | a Tenstorrent *backend* | no board; their compiler speaks StableHLO | **StableHLO export** (checked on XLA: Llama/Qwen2/Mistral at 10⁻⁶) and a **portable admission kit** that runs the probes on the device through PJRT and comes back to be judged and signed |
| "cluster orchestration" | a scheduler | schedulers exist; what determinism allows and they do not have is the point | content-addressed cache across the whole cluster, **audit by redundant execution** with a sample that cannot be chosen afterwards, quarantine with evidence and readmission by measurement, *failover* and *hedging* without changing a bit |
| "complete training pipeline, with HPC" | train a large LLM | no GPU and no machine-days | a **real pre-training, small and complete**: gradients checked against PyTorch, data parallelism whose bits do not depend on the number of workers, exact checkpoint/resume, export to Hugging Face, a shipped model with a receipt that beats Witten–Bell on held-out text ([TRAINING.md](TRAINING.md)) |
| "cursive OCR" | read handwriting | no real handwriting dataset fits on this machine | a substitute **stated as such** (handwriting fonts), **measured** — 64 % CER on hands never seen — and, by the suite's rule, **not shipped**: the reader refuses with the measure ([OCR.md §3j–§3k](OCR.md)) |
| "Arabic, CJK" | readers | — | Arabic by CTC with exact inverse bidi; CJK without a trained network, by directional features and segmentation decided by the recognition ([OCR.md §3g, §3j](OCR.md)) |
| "figure OCR" | find images on the page | finding is not enough; the pain is **the numbers trapped in charts** | detection of figures with a caption, and **chart digitisation that refuses** when the labels do not confirm a scale ([OCR.md §3h](OCR.md)) |
| "LaTeX" | formula OCR | — | typeset formulas → LaTeX, symbols by shape and structure by geometry, measured on typefaces never seen ([OCR.md §3i](OCR.md)) |
| "Cyrillic + Cyrillic handwriting" | two readers | handwriting, as above | printed Cyrillic reader (2.8 % CER on fonts never seen); Cyrillic, Arabic and Japanese "handwriting" measured on calligraphic fonts and reported, not promised ([OCR.md §3k](OCR.md)) |
| "FreeBSD" | port | no FreeBSD here | the worker compiles for FreeBSD (x86-64 and AArch64) with **Capsicum** isolation; the test checks the binary; **not executed** |
| "inverting RoPE" (attachment) | infinite context | the attachment overstates what already existed (§2 of [TRAINING.md](TRAINING.md)) | anchors + window with RoPE applied in the cache's frame of reference: constant memory, no distance outside training, **no new kernel** |
| "physics engine" | a simulator | simulators exist; the pain is that they do not reproduce | physics whose step is a vapor program: **the same bits on every substrate**, differentiable, batched; a digital twin with a verifiable ledger ([PHYSICS.md](PHYSICS.md)) |
| "complex networks" | graph functions | libraries exist; the pain is reproducibility and claims without a null | reproducible generators and statistics, **each claim against a null model** (removed in 0.16, §19) |
| "original and elegant UI/UX" | new screens | — | *Substrates*, *Training*, *Physics*, *Networks* panels; *Vision* with the choice of script, RTL lines, figures with the chart's data, LaTeX; in English and Portuguese, light and dark, in the same visual language (the airlock and the water level) |

### Deliveries and what proves them

| delivery | proof | control |
|---|---|---|
| Substrate airlock | verdict and numerical fingerprint by probes; signed registry. `substrate_test.exs` | simulated bf16 engine: refused, "8 significand bits" |
| Metal (via *shim*) | canonical programs, SSM, GEMM, attention, Llama session, the engine: = oracle bit for bit. `metal_test.exs` | contracting *shim*: envelope; FTZ: envelope |
| StableHLO + kit | Llama/Qwen2/Mistral on XLA at 10⁻⁶; CPU XLA judged: envelope. `stablehlo_test.exs` | operations without an exact equivalent: refused by name |
| Cluster | a node that flips a bit is caught and put in quarantine. `cluster_test.exs` | without audit, the wrong answers pass |
| Pre-training | gradients = PyTorch (7.7·10⁻⁷); the same bits with 1 or 2 workers and with a dead worker. `train_lm_test.exs` | another block size: other bits |
| The shipped model | 2.919 bits/byte on held-out text | Witten–Bell order 5: 3.415; shuffled text: much worse |
| Endless context | 3.03 bits/byte 14× beyond the training length, over 64 lines; until it fills: the bits of the causal model. `streaming_test.exs` | growing positions: 5.81 |
| Physics | first-order pendulum period; oracle = native in chaos; twin parameters recovered; cart-pole 200/200; alarm 18 steps after the failure. `physics_test.exs` | one ulp separates the worlds; shuffled measurements recover nothing; null policy ~50; no failure, no alarm |
| Networks | scale-free BA, WS z ≈ 400, Louvain NMI 1, giant = theory, epidemic threshold; = networkx. `graph_test.exs` | ER is not scale-free and has z ≈ 0; the partition null |
| Figures | 27/30 charts within tolerance; 10/10 figures with a caption (after the fix below). `figure_test.exs` | permuted labels: 12/12 refused; no chart read grossly wrong |
| Formulas | 4.9 % error per *token* in Computer Modern and STIX. `math_test.exs` | flat reading: > 3× |
| CJK | 8.9 % / 2.6 % / 11.3 % CER (zh / ja / ko) on fonts never seen | random characters: the language model changes nothing |
| Arabic | 19.2 % CER on fonts never seen; inverse bidi = python-bidi on 600/600 | Latin reader: 91 % |
| Cyrillic | 2.8 % CER on fonts never seen, half of the vocabulary never seen. `scripts_test.exs` | Latin reader: 98 % |
| FreeBSD | the binary is FreeBSD and imports Capsicum and `_umtx_op`. `freebsd_test.exs` | nothing of Apple's JIT (seccomp, through raw calls, is not visible in the binary) |

### What the scrutiny found along the way (without being asked)

- **The envelope did not cover DAZ** (input subnormals read as zero),
  found by the kit on CPU XLA. Now it covers it, and stays tight.
- **The term tree of a transformer grows exponentially** without
  *let-bindings*: differentiation was redone over them (10 min → 0.8 s).
- **The CJK language model does not help Chinese on a new seed** (on
  the development set it seemed to take 7.8 % to 2.5 %). The published
  numbers are those of the new seed.
- **Velocity as `(p − p_before)/h` loses three digits in f32**: the
  pendulum period got worse with more substeps. Fixed; it now converges
  at first order.
- **The biased double edge swap** made a random graph look
  structured against its own null (z = 20). Fixed.
- **The dots of Arabic letters formed lines of their own** (one line in
  four came out split). The line locator joins low marks to their
  neighbours; Latin does not change.
- **One digit misread in all the labels of an axis** gives a consistent
  and wrong scale (1, 21, 41…). The rule "*ticks* are multiples of the
  step" catches it; without it, the digitiser would lie with confidence.
- **Captions lost in the test set** (one above the figure with
  text right below; another after an axis title outside the box): the
  first complete run of the suite caught it. Fixed — and stated that, for
  the captions, the set stopped being blind.
- **The console attributed every reading to the Latin reader** and showed the Latin language
  model next to Arabic and Cyrillic lines. The footer now says
  which reader read and whether there was a language model.
- **Two sessions on the same worker**: opening the second brings down the first. The
  console's labs use one worker each.

### What this document does not claim

- **Metal, Tenstorrent and FreeBSD on hardware**: none was executed on the
  real hardware; each one says what was checked in its place.
- **Handwriting**: what was measured were handwriting fonts; the cursive
  reader is not shipped.
- **A useful LLM**: the trained model is small; what it proves is the
  *pipeline* (gradients, determinism, receipt, export), not the
  quality of a large model.
- **The effect of the anchors on large models**: on this small model, almost
  none; what was proved is the mechanism (constant memory, distances in
  range, the bits of the causal model until the cache fills).
- **Rigid-body physics**: particles and rods, without rotation, contact
  between bodies or friction.

### How to contest

```sh
mix test test/vapor/substrate_test.exs test/vapor/metal_test.exs test/vapor/stablehlo_test.exs \
         test/vapor/cluster_test.exs test/vapor/train_lm_test.exs test/vapor/streaming_test.exs \
         test/vapor/physics_test.exs test/vapor/graph_test.exs test/vapor/figure_test.exs \
         test/vapor/math_test.exs test/vapor/freebsd_test.exs test/vapor/scripts_test.exs
mix vapor.substrate kit /tmp/kit && python3 /tmp/kit/run_kit.py /tmp/kit --platform cpu && mix vapor.substrate judge /tmp/kit
mix vapor.train --steps 1000                     # rebuilds priv/lm: the receipt's digests
python3 test/python/chart_render.py /tmp/f charts 30 104      # the test charts
python3 test/python/math_render.py /tmp/m test 60 5           # the test formulas
mix vapor.quality                                 # §5f of docs/bench/QUALITY.md
```

## 14. Round 0.11: discovery, mathematics, science, self-play — and bringing an image to life

> Request 3 (2026-10-04, evening), translated: "this directive is subject to
> refinement and scrutiny: weigh something similar to AlphaProof for
> mathematics (geometry, topology, and beyond), something similar for the
> synthesis and discovery of algorithms with or without formal methods and
> complexity analysis, physical simulation (classical, relativistic,
> quantum, tokamak, chemistry — batteries, molecules, materials —, biology —
> mutations, genomics, something similar to AlphaFold) + reinforcement
> learning (manipulation of environments, something similar to AlphaZero) +
> a request to push the limits: from a sketch, models to draw with AI
> (photorealistic, architecture, engineering); offline animations or
> infinite loops with entropy from drawings; from a complex image, an
> interactive infinite loop with NPCs, 3D, depth, free navigation, skeletons
> and actions for entities, effects (light, gravity and beyond) — bringing
> an image to life as in a demo scene, with direction and adjustments by
> prompt, and saving and export of the results (this applies to
> everything!)".

### Scrutiny of the request

The common thread of AlphaProof, AlphaGeometry, AlphaDev, AlphaTensor,
AlphaFold and AlphaZero is not "a big AI": it is **a search that proposes and
a verifier that decides**. The trained models of those works do not fit
on this machine (no GPU, no network for weights); the verifiers, and honest
searches guided by them, do. That is the refinement adopted in almost
every clause.

| clause | literal reading | problem | refinement adopted |
|---|---|---|---|
| "AlphaProof for mathematics" | RL over Lean with a language model | no weights, no GPU; `lake` (Lean) is not on this machine in this round | the **verifier** and a complete search for a large class: geometry by the algebraic method (numerator ≡ 0, with the non-degeneracy conditions as a certificate) and checking in another arithmetic (exact rationals); **conjectures found and proved without being asked for**; topology by exact (torsion) and persistent homology ([MATHEMATICS.md](MATHEMATICS.md)) |
| "synthesis and discovery of algorithms" | AlphaDev/AlphaTensor | — | three searches, three certificates that do not trust them: sorting networks (0-1 principle), matrix multiplication (exact tensor over the integers), bit tricks (minimality by sound exhaustion over a finite domain); complexity fitted to the counts (DESCOBERTA.md (removed in 0.16, §19)) |
| "physics on its various fronts" | a universal engine | it does not exist; each front has its own method | one experiment per front, each against a closed form or a published value, with a control: quantum, relativity, plasma, **tokamak** (Solov'ev), chemistry (Hartree–Fock against Szabo & Ostlund, and the known failure of RHF shown), materials (Lennard-Jones) ([SCIENCE.md](SCIENCE.md)) |
| "biology, something similar to AlphaFold" | predict protein structures | requires the model and the databases | **refused by name**; instead, what is checked: fixation of mutants against the exact chain, phylogeny by neighbour joining (RF 0), folding in the HP model up to the published optimum |
| "batteries, materials" | DFT, electrochemistry | beyond what a round measures honestly | the first rung checked (Hartree–Fock of molecules, an LJ liquid); the rest in the TODO |
| "AlphaZero and beyond" | self-play | — | the whole of AlphaZero at tic-tac-toe scale (policy and value network, PUCT, training by self-play only), judged against the **perfect player**, with the untrained search as a control; **domain randomisation** as "manipulation of environments" (JOGOS.md (removed in 0.16, §19)) |
| "sketch → photorealistic" | generation by diffusion | no weights here | **not measured**: vapor's img2img accepts the sketch with a user checkpoint. Instead, what is measured: **sketch → technical drawing** (lines, circles, constraints, SVG/DXF) and **floor plan → 3D** (rooms, doors, GLB) ([SCENE.md §5](SCENE.md)) |
| "bringing an image to life: NPCs, 3D, depth, free navigation, skeletons, light, gravity" | a world generator | monocular depth, segmentation and pose are trained networks | a **2.5D scene** from first principles: layers by SLIC and the ground plane (depth as a heuristic, stated and editable), the background reconstructed by push-pull, inhabitants by A* with their heads on the horizon, weather, light, wind, particles, drawings animated by skeleton and skinning ([SCENE.md](SCENE.md)) |
| "free navigation" | walk inside the scene | an image does not have the geometry behind it | parallax in a small camera window, stated; "free" waits until there is learned depth |
| "infinite loop with entropy" | — | — | a seeded generator and a fixed step: the loop is **reproducible** (the same seed, the same operations, the same loop); entropy is a control |
| "direction by prompt" | an LLM interprets | no useful LLM shipped | a PT/EN vocabulary → operations; **every word not understood is reported**; the operation schema is the plug-in point for a loaded model (TODO) |
| "saving and export — applies to everything" | download buttons | a download says nothing about what it contains | **`Vapor.Archive`**: zip with a manifest and hashes; the identity is the hash of the manifest; the deterministic types **recompute themselves**; the scene comes out as standalone HTML and video; the sketch as SVG/DXF/GLB |

### Deliveries and what proves them

| delivery | proof | control |
|---|---|---|
| Sorting networks | known optima for n = 3…8, certified by the 0-1 principle. `discover_test.exs` | random and pruned: larger; removing one comparator breaks the network |
| Matrix multiplication | 7 products, exact over the integers; the recursion's counts fit n^2.807 | rank 6 never found; the earlier rounded attempts, wrong |
| Bit tricks | ⌊(x+y)/2⌋ in 4 operations, nothing in 3; checked on 8/16/32 bits | `(x+y)>>1` fails |
| Geometry | 12 theorems (Euler, nine points, Pappus, Simson…): 11 proved symbolically and checked; Simson checked in exact rationals. `prove_test.exs` | 5 false twins refuted by both routes |
| Conjectures | Euler line and nine-point circle found among 1,001 candidates, each survivor proved symbolically | no trivial triple reported |
| Topology | Betti over GF(2) and ℚ; the persistence of a loop | GF(2) alone confuses torus and Klein; the blob has no long bar |
| Science (11) | each experiment against the reference. `science_test.exs` | where there is a good one: wrong implementations run (Euler, Cartesian Laplacian, shuffled sites); two without a control, and two weaker — stated in [SCIENCE.md §1](SCIENCE.md) |
| AlphaZero | against **all** the optimal lines of perfect play: none lost with 128 simulations; with 8, 17 of 129. `games_test.exs` (the 60-game sample of the first version said "0 with 64" — the exhaustive count finds 4 of 131; fixed) | the same search without training: 169 of 175 with 8 |
| Domain randomisation | 484 steps on average on unseen cart-poles | trained on one: 275 |
| Living scene | sky, horizon, ground, light; skeleton of a drawing; prompts → operations. `scene_test.exs` | a meaningless word: reported |
| Sketch and floor plan | rectangle straightened; rooms 12/20 m²; doors 0.9/1.0 m; GLB in trimesh. `sketch_test.exs` | without constraints, the rectangle stays crooked |
| Archives | verification, recomputation `{:ok, :same}`. `archive_test.exs` | one byte swapped and re-zipped: `{:tampered, ["result.json"]}`; a coherent lie: caught by the recomputation |

### What the scrutiny found along the way (without being asked)

- **Operator precedence** in Elixir (`-(x)**2` and `a + b |> min(2)`)
  produced two wrong physical results — the quantum wave packet and the
  mutations. The references caught both on the first run: the rule
  "every measurement has its reference" did its job.
- **A barrier misaligned with the grid** changed the transmission by 18 %.
- **The size of the inhabitants** was first a constant (0.42 of the image's
  height at the bottom line) and came out disproportionate; the
  first-principles rule — in a photo at eye level, **heads stay
  on the horizon** — solved it.
- **The scene analysis did not find the whole sky**: a sky gradient becomes
  several regions; the sky now grows downwards through smooth regions of neighbouring
  colour.
- **Closed loops** (a circle, a rectangle) had no nodes in the skeleton
  graph and vanished from the vectorisation.
- **An existing module was overwritten** during the round (the new
  `Vapor.Bundle` on top of the deployment `Vapor.Bundle`); restored from the
  original zip, and the new one became `Vapor.Archive`.

**The independent review of 0.11** (a reviewer who did not write the code,
reading it against these documents) found, and it was fixed:

- **Archives as an attack vector**: a recipe is data coming from outside, and
  none had a limit — a 32-wire sorting network (2³² vectors),
  `beam: 0` (endless loop), 10⁶ self-play games; and a zip was
  decompressed without a limit (bomb). Now every parameter is bounded before
  running and decompression is counted while it happens (tested).
- **0/0 "proved"**: the symbolic prover did not check for identically
  zero denominators; a construction degenerate for every value (the foot of
  a perpendicular onto the "line" of a single point) came out `:proved`. Now
  it is `{:degenerate, :construction}` (tested).
- **"Independent checking"** was independent of the polynomial algebra,
  not of the formulas of the constructions — rewritten; and the
  conjectures were "proved" only by sampling in rationals: now each
  survivor is proved symbolically.
- **"Never loses"** came from 60 random games; the exhaustive count
  of all optimal lines finds 4 lost of 131 with 64 simulations. The
  claim became the one that holds (none with 128) — JOGOS.md (removed in 0.16, §19).
- Weak controls stated as such (classical tunnelling is computed; the
  norm is a conservation); the tokamak's h² rate comes from the axis
  locator, not from the scheme (exact on those polynomials); the n^2.807 class is
  in the counts by construction; an "unseen" cart-pole is at the corner
  of the training range; an undecoded image brought down the lab.

### What this document does not claim

- No claim of **generative quality** (photorealism, generated
  video): none of that runs here.
- **AlphaFold, battery DFT, general relativity**: refused by name.
- **"Real" depth**: that of the living scene is heuristic.
- **Readable proofs**: the geometric certificates are algebraic.
- The science numbers come from **textbook models** (STO-3G, HP,
  Solov'ev): correct against the references, not predictive for the world.

### How to contest

```sh
mix test test/vapor/discover_test.exs test/vapor/prove_test.exs test/vapor/science_test.exs \
         test/vapor/games_test.exs test/vapor/scene_test.exs test/vapor/sketch_test.exs \
         test/vapor/archive_test.exs test/vapor/console_test.exs
mix run -e 'IO.inspect Vapor.Games.train(games: 400, sims: 32, lr: 0.02, seed: 1).net |> Vapor.Games.digest()'   # = the digest of priv/games
mix vapor.quality                                 # §5g of docs/bench/QUALITY.md
```

## 15. Round 0.12: solving arbitrary problems — workbench, engineering, logic, boards, proteins, render

> Request 4 (2026-10-05), translated: "Provide a refined zip at the end →
> this directive itself is subject to refinement and scrutiny → the result
> must be an artifact that solves the real pains of industry and academia
> with real innovation, lateral thinking and first principles (attack the
> limitations, TODO and complete whatever is pertinent and subject to
> refinement + quality tests to ensure the answers are not just noise +
> original and elegant UI/UX). Primary task: a matter of finish, feature
> completeness, malleability and flexibility of operation, solving real
> pains of electrical, mechanical, chemical and civil engineering, as well as
> physics, chemistry, mathematics, CS and biology, maximum HPC, removal of
> proprietary names like alpha fold etc., and beyond that expanding
> capabilities and skills as similar or superior (in capabilities and
> features and flexibility) to alpha fold and alpha zero (with comparisons
> vs alpha fold itself or its open source equivalent) + interactive
> direction of the 'animated' scene + fine control of NPCs + breadth of
> features (… increase flexibility, real use cases, real pains and fine
> control like alpha proof or equivalent for CS, mathematics, physics
> (Fields, Turing, Nobel and frontier level, using AI with mastery with or
> without a human in the loop) etc. for arbitrary problems and not just
> pre-defined and limited categories, and the scene and creation and editing
> in the studio much more flexible and customisable and aiming at the
> possibility of photorealism) + more finish in UI/UX and a professional and
> stunning interface → focus on flexibility and fine and arbitrary control
> by the user instead of limitation to pre-determined options + perhaps
> special care for chess, shogi and go and card games, and for everything:
> professional interface, alphabetical order etc."

### Scrutiny of the request (and of this directive)

The request has a thesis worth more than any item: **"arbitrary
problems and not just pre-defined categories"**. Rounds 0.10 and
0.11 delivered labs — each one a well-checked demonstration of
*one* problem chosen by us. That is exactly the "limitation to
pre-determined options" that the request rejects. The refinement adopted:

1. **The input is domain text, not a form.** The equation as on
   paper, the netlist as in SPICE, the bus list, the nodes and members of
   a frame, the reactions as on the blackboard, DIMACS, FEN. The examples are
   starting points; the user writes their own problem.
2. **Each answer carries what allows it to be judged** — and that "what" is
   computed **outside the solver**: Kirchhoff re-evaluated, equilibrium of
   loads and reactions, mismatch recomputed from the admittances, invariants
   from the stoichiometry, KKT, the observed order by manufactured solution, the
   DRUP refutation checked by another program, the mate tree redone.
   It is the rule of 0.11 ("a search proposes, a verifier decides") extended
   to everything — and it is what answers "ensure the answers are not
   just noise".
3. **"Similar or superior to" is decomposed into what is verifiable.** A
   state-of-the-art structure predictor and a state-of-the-art self-play agent are,
   each one, a chain — metrics, signal, search, judge. Here each link
   exists, with a test and a control, and the comparison says where it is equal (the
   metrics, the verifiability) and where it is far inferior (prediction from
   a real sequence; playing strength on a real board).
4. **"AI with or without a human in the loop" is implemented as a protocol, not
   as a promise**: the logic desk accepts the *proposal* of anyone —
   a person, a search or a language model through MCP — and only the verifier
   decides (`Vapor.Logic.check/2`, `logic_check`).
5. **Proprietary names** leave the product surface (identifiers,
   panels, API, file types; the 0.11 names are still accepted so as
   not to break saved files) and stay only where the request wants them: in the
   **comparisons**, as a reference.
6. **About this directive itself**: section 14 said "AlphaFold: refused
   by name". The refusal was honest but lazy — it treated the request as
   all-or-nothing. 0.12 refines it: prediction from the sequence
   remains out of reach (and that is said), but everything around it that can be
   checked — metrics equal to TM-align, folding by contacts,
   contacts from coevolution, the pipeline with a control — was done.

| request | literal reading | what prevents it | what was delivered (and measured) |
|---|---|---|---|
| "arbitrary problems" | a universal solver | it does not exist | **Workbench**: ODEs (stiff included), PDEs (parabolic, hyperbolic, Poisson), systems, fits, optimisation, worksheets with **units checked before running** — [WORKBENCH.md](WORKBENCH.md) |
| "real pains of electrical, mechanical, chemical, civil engineering" | commercial packages | scale, licences | eight tools with an **independent certificate**: SPICE (MNA, .op/.dc/.ac/.tran), power flow (Newton), frames and trusses (modes), plane FEM (QM6), pipe networks, kinetics with invariants, flash, distillation — [ENGINEERING.md](ENGINEERING.md) |
| "maximum HPC" | — | no GPU on this machine | the uncertainty ensemble **compiled for the native worker**: 4096 oscillators × 1000 steps in 236 ms, **53×** the BEAM, bit-for-bit parity with the oracle; the browser's GPU for light (render) |
| "similar or superior to AlphaProof … Fields/Turing level" | RL over Lean | no weights, no GPU, no Lean | four decidable logics with **proposer/verifier**: CDCL + DRUP (S(3) = 13, W(3; 2) = 9, R(3, 3) = 6 certified), Knuth–Bendix, Gröbner; **external proposals checked** through MCP — [LOGIC.md](LOGIC.md). Fields level is not claimed. |
| "similar or superior to AlphaFold (with comparisons)" | predict structures | weights and databases | metrics **equal to TM-align**, folding by contacts (TM > 0.75 from true contacts), DCA (precision 0.96), pipeline (TM 0.69; control 0.20), comparison table with AF2/OpenFold/ESMFold — [PROTEINS.md](PROTEINS.md) |
| "similar or superior to AlphaZero … chess, shogi, go, cards" | a state-of-the-art engine | training scale | rules **pinned by perft** and by python-chess/python-shogi, mate proofs checked, Go with superko, k-in-a-row solved, poker by CFR+ with exact exploitability, **generic** self-play judged against perfect play — [BOARDS.md](BOARDS.md) |
| "photorealism" in the studio | an image generator | no weights | physically based **path tracing** on the browser's GPU, with a reference on the server; furnaces and N^−½ as checks — [RENDER.md](RENDER.md) |
| "interactive direction of the scene; fine control of NPCs" | — | — | inhabitants with a name, speech, actions, behaviours, routes by click, timeline; direction by sentences with time and pronouns; GIF of exact frames — [SCENE.md §6.1](SCENE.md) |
| "professional, stunning interface; alphabetical order" | — | — | navigation regrouped and **alphabetical per language**, command palette (Ctrl/⌘ K), six new panels in the same design, own diagrams (single-line, frame, FEM, network, McCabe–Thiele, Bode, boards, 3-D viewer) — [CONSOLE.md](CONSOLE.md) |
| "quality tests" | — | — | §5h: 27 checks with a control; tests of each tool against a closed form, a published value or an external oracle (SciPy, python-chess, python-shogi, TM-align, Biopython); **every example of every panel** run in Chromium |

### What this round's review found (and fixed)

- **Optimisation**: without bounds, the optimal can went down to r < 0 and the
  result returned −6·10⁶² as "optimal". Now: box bounds by
  projection (projected BFGS), divergence detected and stated, and an explicit KKT
  verdict in every result.
- **Render**: the scene reader accepted `r=oops` and broke later;
  now it refuses with the line. The first N^−½ check used glass under the
  sun and failed — not because of a defect, but because those caustics have
  heavy-tailed variance without MIS; the check moved to a diffuse
  scene and the limit was written down ([RENDER.md §3](RENDER.md)).
- **Scene direction**: in English, "a knight named Arthur walks to the
  door, then at 3s he says…" became three nameless people and four
  unknown words; time on its own in a sentence ("depois de 2
  segundos, …") got lost. Now: creating and directing in the same sentence,
  pronouns, nameless roles, time carried to the following sentence — with a test and
  a control.
- **Console**: a poker CSS class shrank the black chess
  pieces; the page referenced external scripts (now embedded, to stay
  a single document); the browser's history was shadowed by a
  variable. Found by Chromium screenshots, not by reading.

### What this document does not claim

- **Superiority** over state-of-the-art structure predictors or game
  engines: the comparisons say the opposite, forcefully.
- **Fields/Turing/Nobel** level: the logics are decidable and the numbers
  proved are classical; what is new is the verifiable protocol, not the
  theorem.
- **Generated photorealism**: there is correct light transport for scenes of
  primitives; there is no generation of images, meshes, textures nor MIS.
- **Design engineering**: the tools are linear/stationary
  (except the circuits and the kinetics) and do not replace design codes.

### How to contest

```sh
mix test test/vapor/workbench_test.exs test/vapor/engineering_test.exs test/vapor/logic_test.exs \
         test/vapor/boards_test.exs test/vapor/proteins_test.exs test/vapor/render_test.exs \
         test/vapor/scene_test.exs test/vapor/console_desks_test.exs test/vapor/mcp_server_test.exs \
         test/vapor/rodada12_test.exs
mix vapor.quality                                 # §5h of docs/bench/QUALITY.md
node test/js/console_desks.mjs http://127.0.0.1:8000/ /tmp/capturas   # with `mix vapor.serve --docs .`
```

## 16. Round 0.13: finance and HFT — and the directive itself as an object of defence

> Request 5 (2026-10-05), translated: "Provide a refined zip at the end →
> this directive itself is subject to refinement and scrutiny → the result
> must be an artifact that solves the real pains of industry and academia
> with real innovation, lateral thinking and first principles (attack the
> limitations, TODO and complete whatever is pertinent and subject to
> refinement + quality tests to ensure the answers are not just noise +
> original and elegant UI/UX) + consider: support for finance, HFT, attack
> the current limitations and TODO + update of the slides + a LaTeX document
> in thesis format in Portuguese (with figures and didactics from CS
> bachelor to CS PhD, and with emphasis on the architecture and
> innovations: what, for whom, for what, why, how, but in a sober and
> academic format) about vapor + script for the oral defence of that thesis
> (md format)."

### Scrutiny of the request (and of this directive)

**The thesis.** The pain of finance is not speed, it is **verifiability**. A
price that changes when the computation changes machine, a backtest that looked at
tomorrow, the best of fifty strategies presented as the only one, an
order book that no one from outside audits — these are failures of proof, not of
performance. vapor has had, since 0.1, the principled answer (the same
bits on every substrate; a search proposes, a verifier decides); the round
takes it to the market instead of inventing a parallel product.

**"HFT", read literally**, is an exchange engine in nanoseconds (FPGA,
*kernel bypass*). The BEAM is soft real time; claiming that would be lying.
The refinement: deliver what of HFT can be checked — a price-time engine
whose journal is a verifiable object, the real protocols (ITCH 5.0, FIX
4.4), the pre-trade risk required by regulation, the microstructure with the
measure that says whether the model fits, and an exchange session in which **the
backtest is the exchange's code** — and measure the latency honestly
(microseconds per event, with the hash).

**"Real innovation, lateral thinking, first principles"** was
decomposed into five moves, each with a test and a control:

1. **Look-ahead as a prefix property.** Instead of auditing
   operators (which would let through a normalisation over the whole
   sample, causal step by step and non-causal as a whole), the signal is
   recomputed over truncated histories and has to be equal bit for bit — a
   black-box test of the whole pipeline, with the day of the peek.
2. **The fundamental theorem as Farkas' lemma.** Arbitrage is not
   estimated: the rational simplex decides, and the answer is **either** the portfolio
   **or** the state prices — objects that anyone checks
   by multiplying. And the desk accepts the *proposal* of anyone (a person,
   a search, a language model via MCP), like the logic of 0.12.
3. **Monte Carlo as a canonical program.** The path step — the
   generator included, Wichmann–Hill chosen because it fits **exactly** in
   binary32 — is vapor algebra: bits equal to the oracle and across thread
   counts, checked in the answer itself.
4. **The book judged by a naive engine.** A second engine, written
   separately with lists and sorting, redoes the journal and requires the same
   reports; the invariants are checked without executing anything. The
   differential *fuzzing* found a bug that both engines shared
   (below).
5. **The simulation is production.** Market makers, Hawkes aggressors and an
   informed trader go through the same risk gate and the same engine; the
   session returns its own audit and repeats itself at the same hash head.

**"Attack the limitations and the TODO"** was refined instead of spread thin: the
open items closed are those that the new domain **made cheap or
necessary** — the simplex with a Farkas certificate (an open item of the logic
in 0.12) is the arbitrage engine; **signed** archives (an open item of
0.11) are what an exchange session or a backtest require to count as a
record; the affine scales (°C/°F, an open item of the workbench) and the transistors
(level-1 MOSFET and Ebers–Moll, an open item of the engineering, now equal to
ngspice) came along because they cost hours and removed excuses.

**About this directive itself.** The request repeats, round after round,
with "solve the real pains… real innovation… tests… UI/UX…". The risk of
a formula request is the formula answer: one more panel, one more table of
"request × delivery". This round's refinement is twofold: (i) **one domain
per round, deep** — the whole of finance, from exact money to the order
book — instead of ten shallow ones; (ii) the directive now asks for the
artifact to be **defensible in writing and orally**: the thesis and the
defence script force the project to explain *what, for whom, for
what, why and how* in a sober register, and any claim that does not
survive that explanation goes. (Two went: "HFT" became "verifiability
of desk and exchange"; "superior to" does not appear.)

| request | literal reading | what prevents it | what was delivered (and measured) |
|---|---|---|---|
| "support for finance" | a complete quantitative library | scale (QuantLib is 20 years old) | exact money; ANBIMA/NYSE/TARGET calendars **equal to QuantLib day by day (1990–2078)**; DI1/LTN/NTN-F/swap curves repriced; BSM/Heston/trees **= QuantLib** (10⁻⁹–10⁻¹²); SVI with arbitrage; VaR with Kupiec/Christoffersen/Basel; portfolios with KKT — [FINANCE.md](FINANCE.md) |
| "HFT" | exchange engine in nanoseconds | the BEAM; no FPGA | price-time engine with **SHA-256 + Merkle journal**, independent naive judge (differential fuzzing), ITCH 5.0, FIX 4.4 (= simplefix), pre-trade gate, Hawkes, Avellaneda–Stoikov (the paper reproduced), Almgren–Chriss, audited exchange session; latency measured (~8.5 µs/event with hash) |
| "HPC" (inherited) | — | no GPU | Monte Carlo **on the worker** with the generator inside the program: 6–23× the BEAM, bits = oracle = 2 threads |
| "answers not just noise" | — | — | **four noise gates** on backtests; §5i: **23 checks with a control**; the size and power of the Kupiec test measured |
| "innovation, lateral thinking" | — | — | prefix invariance; arbitrage by Farkas with proposals from anyone; risk with canonical bits; the book judged by a naive engine; backtest = the exchange's code |
| "TODO" | everything | hardware, weights | rational LP with Farkas; signed archives; °C/°F; MOSFET/BJT = ngspice |
| "original and elegant UI/UX" | — | — | *Markets* group: seal, gates, Basel traffic light, book ladder, journal chain; 87 examples run in Chromium in EN and PT — [CONSOLE.md](CONSOLE.md) |
| "slides" | — | — | `slides/vapor.tex` updated (round 0.13), PDF recompiled |
| "LaTeX thesis, PT, bachelor → PhD" | — | — | `monografia/` (abnTeX2): what, for whom, for what, why, how; architecture and innovations; TikZ/pgfplots figures and screenshots; PDF compiled |
| "oral defence script" | — | — | `monografia/DEFESA.md`: lines per slide, timings, likely questions from the examining committee and answers |

### What the scrutiny found along the way (without being asked)

- **Market FOK became IOC** in both engines — the fast one and the naive one,
  because the same wrong reading of the specification was written twice.
  Only the invariant "FOK is all-or-nothing", which executes nothing, saw it. Lesson
  recorded in the document: two engines are not enough; the invariants are the
  third judge.
- **Calendars**: the first comparison with QuantLib diverged on a
  single NYSE day in 89 years (27/04/1994, the mourning for Nixon) and in the
  TARGET years before 2000. They became data and rules; the test requires
  equality day by day.
- **Monte Carlo**: the first version took 95 s — 30 s in the compilation of
  16 unrolled steps for four ISAs (the allocation verifier
  extracted from Lean is quadratic) and 105 s in the exact oracle over 8,192
  lanes. Now: one step per call, only the machine's ISA, and the parity
  with the oracle done in a 64-lane program compared to the first 64
  of the worker (the lanes are independent — which the comparison itself
  confirms). And the first generator came from the BEAM: the "native" one was not
  faster than the BEAM. With Wichmann–Hill inside the program, 6–23×.
- **The exchange session** "failed" its own Hawkes: the simulator
  spread the arrivals uniformly within each step, and the time-rescaling
  test — correctly — refused the model. The arrivals now
  come from an exact continuous-time Hawkes. And the market makers lost a lot
  to the informed trader because they quoted around the mid of their own book (which
  only followed the fundamental through the aggressions); they now quote
  around the public price of the previous step. It was the simulator, not the
  engine.
- **Avellaneda–Stoikov**: the average spread here (1.49) differs from that of the
  paper's table (1.29): the paper seems to report only the term
  (2/γ)ln(1 + γ/k). Stated in the document; the dispersions, which are the result
  of the paper, match.
- **Kupiec**: the control (normal VaR on t(3)) rejects with p = 0.038 —
  close to the edge. Recorded as it is, without changing the seed.
- **The OTP 25 compiler** failed with an internal error (`beam_ssa_type`)
  on a comprehension with complex patterns in the options text; the function was
  split into functions per task — the code ended up better than it was.

### What this document does not claim

- Exchange latency, nor that the BEAM serves nanosecond HFT.
- Real market data: the examples are illustrative; the oracles are
  libraries (QuantLib, simplefix, SciPy, ngspice), not the market.
- That passing the gates makes a strategy profitable.
- Coverage of XVA, credit, multi-factor rates, local
  or stochastic volatility calibrated to a surface.
- That Wichmann–Hill is a modern generator (it is not; it serves pricing
  with the `rng: :host` path as an alternative).

### How to contest

```sh
mix test test/vapor/finance_test.exs test/vapor/console_markets_test.exs test/vapor/engineering_test.exs \
         test/vapor/workbench_test.exs test/vapor/archive_test.exs test/vapor/mcp_server_test.exs
mix vapor.quality --only round13                      # §5i, in ~25 s
node test/js/console_markets.mjs http://127.0.0.1:8000/ /tmp/capturas
cd monografia && latexmk -pdf monografia.tex          # the thesis
```

## 17. Round 0.14: from showcase to tool — the open workbench

> Request 6 (2026-10-05), translated: "many of the features are thrown
> together and without freedom (above all in the science module, which seem
> to be just already-solved problems being merely presented to the user) →
> total change of philosophy and operation of vapor: instead of being
> something expository it must be a real tool that solves real pains and real
> problems with flexible and open (but sanitised) input from the user,
> without being limited to pre-defined categories […] human in the loop with
> AI models […] return to vapor's roots with a more complete suite in terms
> of AI research activities […] unix philosophy with everything being
> possible to run via terminal […] scenes […] free for creation editing
> research". And, in the middle: "rename the internal language to something
> along the lines of alchemy […] in English + do not use established names
> […] focus on science, cs, mathematics, AI and finance + improve the look
> of the interface […] more striking and innovative ui/ux".

### The request, scrutinised

- **"A collection of" famous search systems** — taken literally, it would be
  N narrow tools with N input formats: exactly the lack of
  freedom criticised. The first principle common to all of them is *propose,
  evaluate with a verifier, keep the certificate*. So: **one** language
  to write the problem (Alembic), **one** furnace that searches anything
  written in it (Athanor) and **one** touchstone that checks without
  trusting (Touchstone). The famous cases become starting examples, not
  products. No third-party name is used.
- **"Intensive use of AI in everything"** — with a risk: a model that decides is
  a model that hallucinates with authority. The rule adopted: the model
  **proposes and drafts** (formalises words in Alembic, with back-translation for
  the person to check; suggests candidates), and the machine **decides** by the same
  verifier that judges the person and chance. Without a model, everything works.
- **"Open but sanitised input"** — the right sanitisation is not a
  list of forbidden words, it is a language without effects and with costs:
  no I/O, fuel at every step, size ceilings, an isolated process with a
  memory ceiling, identifiers that never become atoms, data read by a
  reader that refuses code, and in the browser an interpreted tree (never
  `eval`). Each limit has a test that tries to break it.
- **"Science that only presents solved problems"** — the criticism is fair.
  Crucible inverts it: the person brings the system, and the evidence is the kind that **does not
  need an answer key** (observed order, theorems for any input,
  two methods that agree, controls that a wrong method would fail).
  The fixed experiments become **Calibration**, which is what they were.
- **"vapor's roots in AI research"** — the most common pain is not training,
  it is **knowing whether a difference is real**. Assay answers that: paired
  comparison with power, a *leaderboard* with ties, calibration against its
  floor, agreement, judge bias, contamination, verified deduplication,
  scaling laws that must predict the largest runs without having seen them.
- **"A more striking interface"** — the risk is decorating. The identity (soot,
  parchment, brass, verdigris, cinnabar; old-style serif in the titles) carries
  a metaphor that is also the function: the **furnace** is the live chart of the
  search (sparks, the best-so-far line, the dashed line of the control, the
  allocation of the portfolio) and the **touchstone** is the verdict (a gold
  streak per check that passed, a lead one per check that failed). The ornament
  is where the information is.

### What was done

[ALEMBIC.md](ALEMBIC.md), [ATHANOR.md](ATHANOR.md), [CRUCIBLE.md](CRUCIBLE.md),
[ASSAY.md](ASSAY.md), [CLI.md](CLI.md); console: Workspace, Crucible,
Assay; free scenes (operations in text, per-frame expressions, `direct`
by model); MCP with 20 tools; TUI with the same verbs.

### What was found along the way

- **The scaling law overflowed `exp`** with shuffled losses (the round 14
  control found it): log parameters without structure grew without bound. The
  fit is now finite and the *holdout* says that the law predicts nothing.
- **The Kepler control** in the first design (dt = 0.05, 30 orbits) did not
  separate anything: RK4 also conserved energy to 10⁻⁶. A control that
  cannot fail is not a control; the check now uses dt = 0.1 and
  ~250 orbits, where RK4 drifts 40× more than the bounded error of the symplectic one.
- **`Vapor.Expr.compile`** created a module (and an atom) per new expression:
  with open input, that is an unbounded leak. Now there is a ceiling, and the
  excess is interpreted without creating any atom.
- **The furnace chose random too often** with plain UCB; without the random
  arm, it lost on the rugged problems. Discounted UCB with the arm
  kept solves both.
- **The holdout of moving-average rules**: on a random walk the winner
  in-sample has negative ρ out of sample — selection bias, measured.
  To ensure the test does not accuse everything, a planted AR(1) momentum
  (φ = 0.5) sustains ρ ≈ 0.73; φ = 0.2 no longer does (ρ ≈ 0.21), and that is also
  information.

### What this document does not claim

- That "proved" means more than **complete enumeration of a finite
  space**, with the size stated in the certificate.
- That the furnace is competitive with specialised solvers (SAT, MIP)
  in their own domains; the exact logic and LP remain on their
  desks.
- That the model's draft is right: the back-translation exists for the
  person to check.
- Chemistry beyond closed-shell STO-3G with H and He.

### How to contest

```sh
mix test test/vapor/alembic_test.exs test/vapor/athanor_test.exs test/vapor/crucible_test.exs \
         test/vapor/assay_test.exs test/vapor/mind_test.exs test/vapor/scene_ops_test.exs \
         test/vapor/workspace_test.exs test/vapor/mcp_server_test.exs
node test/js/scene_noise.mjs
mix vapor.quality --only round14                      # §5j, in ~60 s
bin/vapor alembic --card                              # and then write your problem
```

## 18. Round 0.15: the Opus — deciding, not displaying

> Request 7 (2026-10-07), translated: "complete and innovative re-engineering of the material provided in the attachments,
> operating under first principles and lateral thinking", in six phases (critical audit,
> relentless cleanup, rebuilding of the core, authorial UI/UX, rigorous tests, delivery as a
> `.zip`). "The base to be refined is the vapor system, the other attachments must be absorbed +
> total freedom to rename systems/components while keeping the alchemy theme." Attachments:
> vapor 0.14.0 (code, quality, models) and an archive with three projects — **PALADIN** (with
> JESTER and WIZARD), **HYDRA-Z** and **GHOST**. And two long lists of ideas: 19 fronts (from
> circuits over GF(2) to string theory) and 13 "frontiers" (from post-quantum cryptography to
> deontic law).

### The instruction, scrutinised

- **"Complete re-engineering"** — taken literally, rewriting 88 thousand lines of a tested
  system, with signed canonical semantics and proofs in Lean, destroys what it has that is most
  valuable: the continuity of the evidence. A rewrite is not more trustworthy than the original until it
  goes through the same ladder, and the ladder is vapor. The reading adopted: **re-engineering where the
  first principle changes the answer**, and only there — the canonical reduction (which fixed the shape instead
  of eliminating the order dependence), the verification of results (which replicated instead of
  checking), the refused paths of the documents, merging without alignment.
- **"Rename while keeping the alchemy theme"** — vapor is already alchemical (Alembic, Athanor,
  Touchstone, Crucible, Assay). Renaming what exists would only break references, receipts and
  documents; the new names go to what is new: **Amalgam**, **Cupel**, **Rebis**,
  **Aludel**, **Tabula**, and the console group that gathers them, **Opus** (the work).
- **"Zero technical debt, no TODO"** — the base has no `TODO`/`FIXME` in the code (the
  `docs/TODO.md` is a roadmap, not hidden debt). The real debt was somewhere else and was
  searched for with instruments: a compile-time call tracer found **15
  public functions without a caller** in the base (removed; two others, documented but unused API,
  gained tests); the audit test failed in
  0.14 itself (a `System.cmd` outside the sanctioned ports); a figure test failed because of a
  race in the cleanup of `/dev/shm` between suites; the `dot16` oracle truncated silently. The
  four are fixed, with the test that caught them.
- **"Absorb the attachments"** — one project in Racket, another in Clojure, another in Zig with Lean, three
  in Guix: absorbing the code would mean bringing three languages and three environments into a
  system whose principle is to have **one** control language and **one** executor. What is absorbed
  is the idea that solves a pain, rewritten over what vapor already has:
  - **PALADIN** ("polynomial positivity on a rational box, decided by a procedure in
    exact integer arithmetic, with an exhausted budget as a named verdict") → **Aludel**, almost
    whole: Bernstein, de Casteljau, three verdicts, reproducible witness, barriers.
  - **WIZARD** ("the reduction order is declared, so the results are bit-for-bit identical across
    block sizes and executors") is the canonical policy vapor already had — and the first
    principle behind it goes further: **Amalgam** eliminates the order dependence instead of
    fixing it.
  - **GHOST** (quorum `n ≥ 3f + 1`, replicas that vote) → **Cupel**: to detect a
    defective core there is no need to vote among replicas; it is enough to **check** the product with an
    adjoint identity, cheaper than the product, with a proved tolerance. A replica
    only comes in when the check fails.
  - **JESTER** (deterministic epoch root, persistent ledger) is already vapor's Merkle
    journal; nothing to absorb beyond the confirmation.
  - **HYDRA-Z** (NCD of all pairs, with a partition proved in Lean): the formal piece is beautiful, but
    the pain it solves — `O(N²)` compressions — vapor already avoids by another path (MinHash + LSH
    with exact Jaccard on the candidates, sub-quadratic). And NCD itself with real compressors
    **is not a metric**: `C(xx) ≠ C(x)`, and beyond the compressor's window (32 KiB in deflate)
    the distance stops measuring similarity. Absorbed as criticism, not as code.
- **"Six phases in sequence"** — the suggested order (clean, then innovate, then test) is the
  wrong order for tests: each new piece came in **with** its test and its control, and phase 4
  is a quality round that measures each decision against what a wrong method would say (§5k,
  `--only round15`).
- **"Deterministic under failure conditions"** — that was already vapor's thesis (the same answer with
  any worker, any crash). The round extends it to where it still depended on
  shape: training with `reduce: :exact` gives a single *digest* with 1, 2 or 3 workers, any
  assignment and a worker killed in the middle, for **any** number of micro-batches.

### The lists of ideas, one by one

Four destinations: **done** (in this round, with a test), **already existed** (and where), **postponed** (with what
closes it), **refused** (with the reason).

| idea | destination |
|---|---|
| Circuits over GF(2), Gröbner bases, formal hardware equivalence | **done**: Rebis — with the correction that the ANF *is* the normal form (Möbius, no Buchberger) and that Gröbner over ℤ is the tool for word arithmetic ([REBIS.md](REBIS.md)) |
| Carry-less multiplication GF(2¹²⁸), AES, GHASH | **done**: `Rebis.Field`, AES-GCM = OpenSSL; the *kernel* with `PCLMULQDQ` on the worker is postponed (high integer multiplication in the emitters, the same gap as the NTT) |
| Silent corruption: adjoint identity ⟨Wx, v⟩ = ⟨x, Wᵀv⟩ | **done**: Cupel + sentinel, with Higham's tolerance proved and the bound on what is not seen ([CUPEL.md](CUPEL.md)) |
| Bit-for-bit deterministic *all-reduce* | **done**: Amalgam, and training with `reduce: :exact` ([AMALGAM.md](AMALGAM.md)) |
| Merkle journal of training, Ed25519 certificate | **already existed** (journals, receipts, `Keys`). **Claim corrected**: a Merkle root proves *what was declared* as input, not the *absence* of unauthorised data — a negative about the origin of data cannot be proved by hash |
| Stabilisers over GF(2) | **done**: CHP tableau with phase by masks, 400 qubits |
| PQC, lattices, NTT | **already existed**: negacyclic `Vapor.NTT` (the ML-KEM ring) ([ZK_FHE.md](ZK_FHE.md)); *kernel* postponed |
| Exact FHE over gates | **refused as before** (ZK_FHE §2.1): a scheme without audit and without the standard's parameters is a dangerous toy |
| STARK VM, binary towers (Binius), "physics-based economy" / currency | GF(2ⁿ) **done** (the base of a binary tower); the STARK VM **postponed** (no use case in vapor that needs a succinct proof of execution); the currency **refused** — there is no named pain it solves here |
| Tokamak, MHD stability | Grad–Shafranov equilibrium **already existed** (SCIENCE); what this round adds is the means to **prove** properties of a reduced polynomial model (Aludel, barriers). Controlling a real reactor: out of reach, and stated |
| Drift-free astrodynamics | **already existed**: symplectic integrators (Yoshida 4) in Crucible, with RK4 as the control |
| Axiomatic robotics, avionics "beyond DO-178C" | the **done** core is the barrier certificate (Aludel). DO-178C certification is an engineering process with traceability and independence, not a theorem; "transcending level A" by proof is a claim that no proof sustains. Evidence kits: **postponed** (TODO) |
| Computable deontic law | **done**: Tabula ([TABULA.md](TABULA.md)) — antinomies with a scenario, consistency with a DRUP proof, precedences, silences, Hohfeld |
| Mechanisms (VCG, Gale–Shapley) | **postponed**: small and useful (stability and truthfulness are checkable by enumeration), but with no case in vapor that asks for them now |
| Autoformalisation without hallucination | **already existed** in part (`Mind.formalize` with back-translation and the verifier deciding). Lean is not on this machine; exporting certificates to Lean remains in the TODO |
| *Trusting trust* | **postponed** with the right technique named: diverse double-compiling (Wheeler, 2009) — the worker compiled by two independent Zigs and compared; "500 lines of audited Assembly" is a seed, not a proof |
| Model alignment for merging | **done**: Git Re-Basin with exact Hungarian (MERGING §8) — closed the open item that `diagnose` pointed to |
| Clean emulation of old chips as networks over GF(2⁸) | **postponed** (Rebis checks a *netlist*; an emulator is another project). **Claim refused** of the "non-infringement certificate in Lean 4": a mathematical proof does not establish a legal fact, and re-synthesising from the original binary is not *clean room* (which requires separation of people and of access, not of algorithms) |
| Audio: "isomorphic permutation" to keep n-gram collision **below the forensic plagiarism thresholds** "while preserving the perceptual identity" | **refused**. It is, by construction, a tool for copying a work and escaping detection — the intended use is evasion, not analysis. Also refused is the premise of "overcoming the phonogram": reconstructing a recording from air pressure does not dissolve the rights to the composition. The defensive sense is available: measuring melodic similarity to **detect** copying |
| Super-resolution (video, spectrograms), vocoders | **already existed** in part (studio: upscaling, audio); trained vocoders require weights: postponed |
| BCI, low-power implants, artificial life | **postponed**: no signals, no hardware, no pain that vapor solves today; what is transferable (verified fixed-point decoding) is the ladder that already exists |
| "Digital Socrates": P2P crawlers, entropy filter | **postponed/refused**: crawling the internet does not fit on this machine without a network, and an "entropy" filter for text "without epistemic value" measures compressibility, not truth (the 0.7 criticism holds) |
| Smart sand, self-replicating von Neumann probes | **refused as engineering** (there is neither hardware nor a case); the formal part — replication conditioned on a verification — is what the airlock already does with programs |
| "Post-NCD" information theory | criticism **done** (above, HYDRA-Z); a measurable substitute (exact Jaccard, MinHash) already existed |
| Information geometry in compilers | **already existed** where it measures something (Fisher in Crucible and in Assay); "compiler guided by the Fisher metric": without a case, postponed |
| Categorical semantic web | **postponed**: no named pain |
| Eliminating the operating system (bare-metal) | **postponed** (TODO: PMU and RAPL on *bare-metal*); the worker already talks to the kernel through a handful of calls |
| Physiological digital twin | twins of physical systems **already existed** (PHYSICS); multiscale physiology: no data nor validation possible here |
| Solving the 10⁵⁰⁰ landscapes of string theory | **claim refused**. The topological data (Hodge numbers) are integers, and a filter such as `|χ|/2 = 3` generations is real — but it is a necessary condition, not a sufficient one, and the number 10⁵⁰⁰ counts fluxes, not enumerable manifolds. Search with a control over a given list (Kreuzer–Skarke) fits in Athanor like any finite space; "solving the problem" does not |
| Exact quantum chemistry, DMRG, dendrite-free batteries | RHF/STO-3G **already existed**; DMRG postponed (Crucible's TODO). **Refused** the "proof that dendrites are topologically impossible": a barrier proves something about a model, and dendrite nucleation is not a polynomial model of few variables with validation |
| Reversible compilation, Landauer limit | **postponed**: Toffoli/Fredkin over GF(2) are circuits that Rebis already reads; the energy gain requires adiabatic hardware that does not exist here |
| Integrated photonics | **postponed**: no case; the ideal unitary mesh is linear algebra that vapor already has |
| Parametrisation-free geophysical modelling | **postponed**: time-dependent 2-D PDEs are in the workbench's TODO |
| Nanosecond HFT | 0.13 dealt with it (verifiable matching, independent judge); the engine on the worker remains in the TODO |

### What was done

[AMALGAM.md](AMALGAM.md), [CUPEL.md](CUPEL.md), [REBIS.md](REBIS.md),
[ALUDEL.md](ALUDEL.md), [TABULA.md](TABULA.md); merging with alignment ([MERGING.md §8](MERGING.md));
JBIG2 Huffman and halftone ([OCR.md §3f](OCR.md)); console (Opus group), terminal
(`vapor rebis|aludel|tabula|cupel|amalgam`), MCP (25 tools); the §5k quality round.

### What was found along the way

- **The `dot16` oracle truncated the tail** of contractions with `k` not a multiple of 16 —
  silently. Found while writing Cupel, whose tolerance did not close; now it is an error.
- **The audit test already failed in 0.14**: a `System.cmd` in Athanor's `--measure`, outside
  the sanctioned ports. The external command moved to `Vapor.Main.Measure`: its own process
  group (`setsid`), a deadline that kills the whole group (the first design held the *pipe* and
  left orphans), an output ceiling.
- **A race between suites in `/dev/shm`**: the cleanup of one removed the file that another
  had just written. The cleanup gained a grace period and the writer touches the file.
- **jbig2dec 0.20 gets `HDEFPIXEL = 1` wrong** in halftone regions: it fills with the byte `0x01`
  (one black pixel in eight) instead of black. The fixture for that case is judged by T.88 6.6.5.2, and the
  difference is recorded in the manifest.
- **Table B.2 has negative widths and B.11 has no `DT = 0`** — the test encoder
  had to respect that; the decoder already did.
- **A miter with repeated literals hung the solver**; **out-of-order AIGER** was refused;
  **the phase of the destabilisers** in CHP can be odd. All three fixed with a test.
- **The commutativity of a multiplier** is exponential for resolution (2,963 conflicts at 5 bits;
  6 bits does not finish), and polynomial for the algebra over ℤ — and the opposite for parallel-prefix
  adders. The round measures both and says which to use.

### What this document does not claim

- That Cupel sees everything: below the rounding envelope, a bit flip is
  indistinguishable from rounding, and the per-bit profile says where the line is.
- That Amalgam is suitable for the inner loop of a *kernel*: it is a control-plane reduction.
- That Aludel proves anything about the world: it proves things about the polynomial it receives.
- That Tabula reads contracts: it decides on clauses written in its form.
- That Rebis's AES-GCM is suitable for encrypting data: it is a checker, without constant time.

### How to contest

```sh
mix test test/vapor/amalgam_test.exs test/vapor/train_exact_test.exs test/vapor/cupel_test.exs \
         test/vapor/rebis_test.exs test/vapor/aludel_test.exs test/vapor/tabula_test.exs \
         test/vapor/merge_align_test.exs test/vapor/jbig2_test.exs test/vapor/opus_test.exs
python3 test/python/jbig2_streams.py /tmp/jb2          # the JBIG2 fixtures again, each judged by jbig2dec
mix vapor.quality --only round15                        # §5k, in ~2 s
bin/vapor rebis equiv a.net b.net                       # and then bring your own circuits
```

## 19. Round 0.16: purifying — the Majlis, the Dīwān, the Khazāna and Almizan

> Request 8 (2026-10-07), translated: "continue + closing of loose ends, TODO + purge of merely
> illustrative features + refinement and polish + chat/agent features for the models: editing
> conversations, context, cloning sessions (etc, whatever else is pertinent to match and surpass a
> claude or chat gpt of the world even considering the agentic part) + terminal utilities (terminal
> gui and tui equivalents as well as api) + absorbing some of the ideas attached in the initial
> prompt that have value (like information geometry) + absorbing the ideas of the attached pdf [ASAS,
> *Atomic Stream Application Substrate*] + weigh: [the **Almizan** manifesto] + development
> ecosystem (à la VS Code, Neovim and Emacs) + total freedom to rename […] via the three-letter
> roots (e.g.: mizan → mzn) + total freedom for refinement and scrutiny of this directive
> + […] agentic capabilities of sandbox and code execution […] + would you keep Alembic and vapor in
> universal syntax (ASCII/Latin)? + would you implement Almizan as a 'secret layer'? +
> purify vapor to the extreme […] + weigh [three criticisms: Riemann × discrete hardware,
> dependence on an ITP, syntax × semantics]."

### The instruction, scrutinised

- **"Purge of the merely illustrative"** needs a criterion, otherwise it becomes taste. The one adopted is that of
  ASAS (§2, the lexicographic order over a non-negotiable floor), translated to vapor: the **floor** is
  evidence — nothing stays that claims without checking; above it, **usefulness before lightness before
  reach**. A feature is useful when it answers a question that someone brings *with their own
  data*; it is illustrative when it only re-enacts a fixed demonstration. By that yardstick:

  | removed | why | what replaces it |
  |---|---|---|
  | `Vapor.Graph` (complex networks) | generators and statistics over synthetic graphs; nobody brought their own graph | — (networkx does better what it did) |
  | `Vapor.Discover` (sorting networks, Strassen, synthesis) | *known* optima found again | Crucible (your system, with evidence) |
  | `Vapor.Games` (AlphaZero at tic-tac-toe) | a solved game, relearned | `Vapor.Play` (chess, shogi, Go, generic MNK), which serves any game |
  | Physics, Networks, Algorithms, Mathematics, Science, Games, Draw, Listen, Training, Merging panels; the demonstration dossier button | each one ran a fixed example | the modules that serve other doors stay (Physics and Science serve Crucible and the studio; the proof serves the replayable archive) |
  | TUI: draw, listen, merge | likewise | the TUI is now the Dīwān: **all** the verbs |

  The balance is **2,949 fewer lines** in what existed, 3 documents removed (REDES, DESCOBERTA,
  JOGOS) and no ledger claim lost. Sections §13–§14 of this directive stay as a
  historical record; their contestation commands that cite removed tests point to the
  git history.
- **"Match and surpass Claude and ChatGPT, including in the agentic part."** Surpassing *the model* is not an
  honest promise for a system that serves whatever model you have. What can be surpassed is **the
  conversation as an object**: what the two products treat as a mutable list, vapor treats as
  a content-addressed tree ([MAJLIS.md](MAJLIS.md)):

  | | ChatGPT / Claude (products, Oct. 2026) | vapor 0.16 |
  |---|---|---|
  | editing a message | creates a branch; ‹ › navigation | likewise, and the old branch is **immutable** (the hash of each message covers the parent) |
  | another answer | likewise | likewise, with the *backend*, the tokens and the time of each one |
  | forking | "branch into a new conversation" (copy) | **O(1)**: 0 messages copied (measured: §5l) |
  | context | opaque; invisible automatic compaction | **exactly** what the model will read, per message, with those left out; **pinned** messages that never leave; explicit compaction that names the hash it summarises, undoable |
  | search | by title/content | BM25 over all messages, of all branches |
  | import | — | ChatGPT and Claude exports (tree preserved) |
  | export | JSON/Markdown | **verifiable** JSON: a swapped character is refused on import |
  | sharing | link; deleting revokes | link = MAC over (conversation, generation); **revoking kills all** links at once (ASAS §6.2) |
  | agent | the product's tools | vapor's tools by allow-list, with a **verifiable journal** of each execution |
  | where it lives | in the vendor's cloud | in a file of yours, *crash-atomic* (the Khazāna) |

  What is **not** claimed: token-by-token *streaming* (the answer arrives whole), image attachments in the
  conversation, memory across conversations. They stay in the [TODO](TODO.md).
- **"Equivalent terminal GUI, TUI and API"** — equivalence is only true if it is *built*,
  not maintained by hand. There is **one** interpreter, the Dīwān ([DIWAN.md](DIWAN.md)): pipes, redirection,
  files, `help`; the command line (`bin/vapor`), the TUI and the console's terminal are three
  presentations of it, and the HTTP API is the same call. A new verb appears at all four doors without
  one more line.
- **"Sandbox and code execution"** — the console's terminal is a door for strangers, so it is a
  **cage**: files only from the session (128 files, 8 MB each, 64 MB in total), no *shell*,
  `--measure` refused, a heap ceiling and a deadline per command, one process per command. The cage is
  measured against the control (the local session reads the file; the cage refuses — §5l). Alembic already
  executed generated code only in isolated *workers*; that is now a ledger claim,
  checked by audit at every build.
- **"Rename via three-letter roots (mizan → mzn)"** — scrutinised, and corrected: the root of
  *mīzān* is **و-ز-ن** (*w-z-n*, "to weigh"); *m-z-n* is not a root, it is the instrument pattern *mif'āl*
  applied to it. And `.mzn` is already the extension of MiniZinc, a constraint language in use — the
  conflict would be real in every editor. The extension is **`.wzn`**. The new names follow the theme of the
  request (Arabic, like *alchemy*, *alembic*, *elixir*), each for what it does: **Majlis** (the
  council where one converses), **Dīwān** (the register, and the hall where business is dispatched), **Khazāna** (the
  treasury, the depository), **Almizan** (the balance). Nothing old was renamed: the old names are in
  receipts, signed archives and in the thesis, and renaming would break the evidence for aesthetics.
- **"Purify to the extreme; product finish; frontier academic enabler"** — the three
  things pull in different directions, and the lexicographic order decides: the floor (evidence) first;
  the product is what remains after the purge; the academic frontier comes in only where it decides something
  (Almizan, information geometry with a control, the Khazāna with fault injection at every byte).

### ASAS, absorbed

ASAS is an operating system for RISC-V resolved at build time. Almost nothing of it fits in a
system that runs *on top of* an ordinary OS — but five ideas are independent of the *kernel*, and all five
came in:

| idea (ASAS) | in vapor 0.16 | measured |
|---|---|---|
| §8.3–8.4 immutable content + atomic pointer swap; root in **two slots** with sequence and tag | `Vapor.Khazana`: two pack files and two 128-byte root slots (`KHZ1` · seq · generation · size · root · tag) | a crash at **every byte** of a *commit* (344 points): always the old root or the new one; the single file overwritten in place reads as v2 with a torn value — worse than lost |
| §6.1 **one** keyed hash | HMAC-SHA256 under the depository's key (SHA-256, not BLAKE3: it is the hash of vapor's whole history — receipts, Merkle journal, archives; *one* hash is worth more than *the best* hash) | — |
| §6.2 **computed** capabilities, mass revocation by generation | shared links = MAC(conversation, generation); revoking increments the generation | a forged or revoked link: 403 |
| §6.3 entropy at a **single boundary** | `Vapor.Entropy`: the OS generator only there; everything else seeded and reproducible | source-code audit at every build; an injected random draw is caught |
| §11 the assurance **ledger** | `Vapor.Assurance` → [ASSURANCE.md](ASSURANCE.md): *proved*, *checked*, *tested*, *argued*, *owed* | a test fails if the cited evidence disappears or the document diverges from the data |

Refused, with reason: PMP instead of MMU, *kernel bypass*, flat binaries, rasterisation on RVV,
device tree at build time — vapor is not an OS and does not replace the user's. The Zig
(mechanism) × Rust (policy) split already exists in vapor as BEAM (policy, supervision) × Zig *worker*
(mechanism), separated by a process — the "seam" of ASAS §3.3. The weak-memory guarantee of the
lock-free rings of ASAS §4.4 has an analogue in the *worker*'s `/dev/shm` protocol, and is in the
ledger as **owed**, not as done.

### The Almizan manifesto, weighed

The manifesto proposes a language with triliteral roots and *awzān* as the type system, an
S-expression syntax in Arabic from right to left, proofs in Lean, addressing by *abjad*, the
`ikseer` chain (*distill*, *assay*, *transmute*), content-addressed P2P dependencies (Al-Khazāna) and the
Alembic (execution) × Almizan (proof) split. Weighed item by item:

| proposal | verdict | what was done |
|---|---|---|
| roots = domains, *awzān* = regimes | **kept** — it is the best idea of the manifesto: the morphology carries the type | the root says *what kind of claim* (ح-س-ب arithmetic, ح-ف-ظ conservation, ن-ق-ل transition, ك-ت-ب record); the *wazn* says *how it may be used* (فاعل transient, مفعول persistent, برهان proved). Meaningless pairs are refused: a conservation law without a proof does not compile |
| RTL Arabic syntax | **kept as a projection**, not as the language | a neutral tree; two bijective printings — Latin (Buckwalter, ASCII) and Arabic (Arabic keywords, Arabic-Indic digits). `read(print(t)) = t` on 300 random programs in both scripts; the identity is the hash of the tree, equal in both |
| addressing by *abjad* | **refuted by measurement** | of the 21,952 three-letter roots, 21,950 share their value with another (99.99 %); the largest class has 82. *Abjad* is shown (`vapor wzn abjad`), never used as an address: the identity is SHA-256 (0 collisions on the same roots) |
| proofs in Lean | **replaced by decision procedures**, Lean as a second opinion | see criticism 2 below |
| `ikseer distill/assay/transmute` | **kept as verbs** | `vapor wzn check` (distils: decides the obligations), `vapor wzn assay` (the fāʿil in f32 against the exact value in ℚ, ULP by ULP), `vapor wzn transmute --to vapor\|aiger\|lean` (lowers to vapor's compiler — x86-64, AVX-512, AArch64, RISC-V, SPIR-V —, to AIGER via Rebis, or to Lean 4 theorems) |
| P2P Al-Khazāna | **postponed**; the local content-addressed, *crash-atomic* depository is done | P2P without a trust model is a supply-chain vector; when it comes, it will come over the same two-slot root |
| VHDL | postponed | AIGER covers verification; synthesis is not vapor's role |

**Would I keep Alembic and vapor in universal syntax (ASCII/Latin)? Yes.** Alembic is a
*kernel* language read by performance engineers all over the world, in diffs, in code
reviews, in terminals without an Arabic font; changing its script would cost readers and would buy
no guarantee. **Would I implement Almizan as a "secret layer"? Not as secret — as a
declared formal dialect.** A secret layer is a layer without reviewers; what gives value to a
proof dialect is precisely being read and contested. The form adopted is that of the final request
("a neutral representation besides the Arabic"): the tree is neutral, the Arabic projection is a first-class
citizen (the LSP, the syntax highlighting of VS Code, Neovim and Emacs and `wzn show
--arabic` treat both equally), and whoever wants to write and read only in Arabic can.

### The three criticisms, answered

1. **"Riemannian geometry × discrete hardware."** The criticism is right about the *implementation* and
   wrong about the *use*. Nobody runs a manifold on silicon: one runs the **closed formula** that
   the geometry delivers, and that is where it pays. In 0.16 (`Vapor.InfoGeom`, [GEOMETRY.md](GEOMETRY.md)):
   the Fisher–Rao distance on the simplex is `2·arccos(Σ√(pᵢqᵢ))` — a sum and an arc cosine; the
   geodesic is a *slerp* on the sphere of square roots; between normals, the closed formula of the hyperbolic
   metric. And each use has a control that shows why the geometry matters: KL violates the
   triangle inequality in 217 of 1,000 triples (Fisher–Rao: 0), so it cannot order "closer";
   the natural gradient gives the **same** predictions with a variable rescaled ×1000
   (difference 7·10⁻¹²), the plain gradient does not (0.54). What is discrete is the arithmetic, and for
   that vapor already has Higham's envelope proved. In Assay, the `geometry` tool compares
   models by their output distributions with the right metric.
2. **"Dependence on an ITP (Lean) → needs SMT/SAT automation."** Correct, and accepted. An
   interactive proof assistant needs a person who knows how to use it for each obligation — the
   manifesto's bottleneck. In Almizan, **each obligation that the language can state has a
   decider that vapor already has**, with a certificate checked by separate code: polynomial
   identities and conservation laws by exact normal form over ℚ (dH/dt = ∇H·f ≡ 0); positivity
   and bounds on a box by Aludel (Bernstein, replayable witness); invariants of Boolean transition
   systems by SAT with a DRUP proof. The verdict is *proved*, *refuted with a counterexample*
   or *unknown* — and *unknown* **does not compile**. The language is cut to fit the
   deciders, not the other way round. Lean stays as an independent second opinion
   (`transmute --to lean` generates the theorems; closing them with `ring`/`decide` is in the ledger as
   **owed**, because Lean is not on this machine). The §5l control shows the difference: sampling
   dH/dt at 1,000 points accepts the law with 10⁻⁹ damping; the decider refutes it
   and gives the point.
3. **"Syntax × semantics (Arabic × neutral)."** The semantics live in the tree; the script is a
   projection. That is not a demotion of Arabic: it is what allows the root and the *wazn* — which are
   *semantics*, not spelling — to survive any script, and two people, one reading
   `(claim energy (root H-f-Z) (wazn burhan) …)` and the other `(دعوى energy (جذر ح-ف-ظ) (وزن برهان) …)`,
   to discuss **the same hash**. The bijection is tested; lossy transliteration (folding the forms of
   the hamza) merges 7 names into 2 — hence Buckwalter, and not a "pretty" romanisation.

### The development ecosystem

A single language server (`vapor lsp`, LSP 3.17 over stdio: diagnostics, *hover*, completion,
go to definition, symbols, formatting and the script switch, `vapor.almizan.toArabic`/`toLatin`), and three
thin clients that reimplement nothing ([EDITORS.md](EDITORS.md)): a VS Code extension, a
Neovim *plugin* (Lua, `vim.lsp`), an Emacs mode (`eglot`). The TextMate and Vim grammars
highlight both scripts of Almizan and Alembic. The test talks to the real server, with
`Content-Length` *framing* and UTF-16 columns, from a Node client.

### What was found along the way

- **The naive depository control was more serious than expected**: overwriting half of a
  CBOR root of the same size does not make it unreadable — it decodes as v2 with a torn value. The
  §5l check was fixed to call this by its name (before, it expected "lost").
- **The entropy-boundary check itself matched itself** (the regular expression
  was written in the file it scans); the rule is now assembled in pieces, and the build
  audit would catch it.
- **An Arabic name with an ASCII digit** (`سالم-1`) is refused, by design: a name does not mix
  scripts. The generation of test names uses Arabic-Indic digits.
- **The anchor root**: top-level messages of different conversations were "siblings" of each other; each conversation
  gained an anchor node and each message its owner (`o`) inside the hash — without that, the garbage collection of one
  conversation deleted messages of another.
- **The Dīwān accepted a leading `|`** (an empty command); splitting by `|` preserves the empty
  segments in order to refuse them.
- **The audit caught round 0.15** using the OS generator for test inputs — now a seeded
  stream.
- **A regenerated answer did not become the current one**: the pointer only followed an answer requested *at the*
  pointer, so "another answer" created the sibling and the page stayed on the old one. The test was
  called "the pointer follows it" and did not check the pointer. Found by the browser test; now
  regeneration moves the pointer when the question is on the visible path, and a late answer to
  a question that the person has already moved past still does not steal it — both cases tested.
- **The conversations API test depended on the load order** of the test files (it used a
  *backend* defined in another file); the *backend* moved to `test/support`.

### What this document does not claim

- That vapor converses better than a frontier model: the conversation is as good as the model
  served; what is better is what can be done with it.
- That the cage withstands an attacker with access to the BEAM: it rests on the BEAM's process
  isolation and on every verb reading through the `Vapor.Main.read_input` door (the ledger says so).
- That the Khazāna survives a disk that lies about `fsync`: that is the platform's half of the contract
  (as in ASAS §8.4).
- That Almizan proves everything: it proves what its deciders decide, and refuses the rest.

### How to contest

```sh
mix test test/vapor/khazana_test.exs test/vapor/majlis_test.exs test/vapor/hall_test.exs \
         test/vapor/diwan_test.exs test/vapor/almizan_test.exs test/vapor/lsp_test.exs \
         test/vapor/editors_test.exs test/vapor/info_geom_test.exs test/vapor/assurance_test.exs \
         test/vapor/console_majlis_test.exs test/vapor/audit_test.exs --include playwright
mix vapor.quality --only round16                         # §5l, in ~2 s
mix vapor.assurance --check                              # the ledger against its document
bin/vapor wzn check priv/almizan/oscillator.wzn          # proved; refuted, with the point
bin/vapor wzn show priv/almizan/oscillator.wzn --arabic  # the same tree, the other script
T=$(bin/vapor chat new --title teste) && bin/vapor chat say $T "olá" && bin/vapor chat context $T   # with VAPOR_MIND
```

## 20. Round 0.17: the alchemical re-engineering — the build, English, Palingenesis, the mould, causes

> Request 9 (2026-10-08), translated and condensed: "Act as principal software engineer, systems
> researcher and product designer: a complete, first-principles re-engineering of the attachments.
> Phase 0, a critical audit, including of this instruction. Phase 1, ruthless cleanup: no dead
> code, no TODO, FIXME or stub. Phase 2, the core rebuilt for resilience, low latency and safe
> concurrency, with non-trivial innovation. Phase 3, an original, minimal, ergonomic UI. Phase 4,
> rigorous tests: unit, edge and stress; deterministic under failure. Phase 5, the complete source,
> packaged as a zip. The target is vapor; the other bases are to be absorbed; total freedom to rename
> under the theme of alchemy. English the default, not Portuguese." Attached: two long analyses
> (a "Ship of Theseus" for AI over the Merkle store, Fisher–Rao and Re-Basin; a frontier programme of
> ternary quantisation, an "Arché suite", 3D Gaussian splatting, infinite context, robotics and
> artificial life); a `Vapor.Silicon` that emits GDSII; notes on LoRA, distillation and natural
> gradient; a design for agents behind a frozen, content-addressed web and human-signed action
> plans; a "ghost network" with traffic mimicry and its incentives; notes on metaprogramming in
> Almizan and Alembic; `.nbq` for Alembic, and perhaps "Almizan" for Mīzān. And a build log:
> `nix develop` fails on `poppler_utils`, then `mix vapor.serve` fails to compile on Elixir 1.18.5.

### The instruction, scrutinised

- **"Rebuild the core."** The core's value is its evidence: bits identical across five substrates,
  a register-allocation checker proved in Lean, certificates on every answer, a ledger that fails
  the build when it overstates. Rewriting 96,000 tested lines in one round would discard exactly
  that. First principles here meant finding what is actually wrong and fixing it where it lives,
  then adding what is missing. The defects found are listed below, each with the test that pins it.
- **"No TODO, FIXME or stub."** The code had none (a search of `lib/`, `native/` and `proofs/`
  finds no marker, no `sorry`, no `axiom`). The open work lives in [TODO.md](TODO.md), where each
  item says why it matters. Deleting that file would hide the work, not clean it. This round closes
  items from it instead (below).
- **"English the default."** Done for everything a reader meets first: the README, the CHANGELOG,
  every document under `docs/`, the generated reports, the ledger, the CLI, and the console (already English by
  default, with Portuguese as its second language). Two things stay in Portuguese on purpose: the
  thesis (`monografia/`, abnTeX2) and its defence slides, which are documents of a Brazilian
  institution. The Portuguese keywords the parsers accept (`deve`, `maximizar`, `se … então`) stay
  too: bilingual input is a feature.
- **"Rename freely, alchemically."** Renames break receipts, signed archives and the thesis's
  citations, so only three were made, each for a reason beyond taste. **Almizan**, because the
  language was spelled four ways (Mīzān, Mizān, Al-Mizān, Al-Mīzān) and English fuses the Arabic
  article everywhere else (alembic, alchemy, algebra); the hashed format tag stays `mizan`, so
  every 0.16 module keeps its identity. **`.nbq`** for Alembic, its triliteral root (ن-ب-ق), as
  `.wzn` is Almizan's (و-ز-ن). **English file names** for documents and tests. The new modules took
  alchemical names: the **hermetic seal**, **Palingenesis** (rebirth from the ashes), **Qālib** (the
  mould).
- **"An original UI."** The console already had an identity (the canal lock, water level as
  measurement). The round adds to the surfaces people already use: four verbs that appear in the
  CLI, the TUI and the console's terminal through the one interpreter; two MCP tools; and inlay hints,
  so a claim's verdict sits beside it in every editor. A new page would have been decoration.
- **"Package as a zip."** Done, in the same three parts as the upload (code, quality data,
  admitted models), each one a subset of the repository at the commit.

### The build, first

| failure | cause | fix |
|---|---|---|
| `nix develop`: `'poppler_utils' has been renamed` | nixpkgs renamed it (and deprecated top-level `elixir`/`erlang`, `nixpkgs-fmt`, `texlive.combine`) | `poppler-utils`, `beamPackages.*`, `nixfmt`, `texliveMedium.withPackages`; every flake output evaluates on the user's pinned nixpkgs (151fa4e8) with no error or deprecation warning |
| `cannot inject attribute @modal … cannot escape #Reference` | on OTP 28 a compiled regex is a reference, and three modules kept regexes in module attributes | regexes live in functions |
| 7 warnings, then 9 from the type checker | ranges without steps, `0.0` patterns, `Tuple.append`, dead clauses; **one real bug**: `mix vapor.lock` with no argument raised `MatchError` (a match inside `cond`) | all fixed; the build has no warnings |
| Lean | the proofs pinned Lean 4.23, and this machine had none | ported to 4.34.1 (deprecated lemma names, one `omega` that exceeds the recursion depth); extraction byte-identical except the sources' digest; `lake build` warning-free |
| a test pinned 941 primary composites | OTP 28 ships Unicode 16.0, which has 961 (Python 3.14's `unicodedata` agrees) | the test knows both versions |

### The attachments, weighed claim by claim

| claim | verdict | what was done |
|---|---|---|
| A Fisher–Rao brake "ends catastrophic forgetting" | **overstated**: it bounds drift on the anchor set only | built as Palingenesis's brake, with that scope written into the module, the doc and the ledger |
| Fisher–Rao is "the only Riemannian metric invariant to reparametrisation" | half right: Čencov's theorem makes it the unique metric (up to scale) invariant under sufficient statistics on the simplex; computed on output distributions, parametrisation-independence is immediate | used where it is right: the brake |
| Hungarian Re-Basin gives a "zero-tolerance fit" of a new plank | **wrong for whole-block replacement**: permuting a block's hidden units changes nothing (measured: drift 4.9·10⁻⁷); alignment matters only when blending, and joint alignment across layers is a heuristic, not exact | `mode: {:blend, t}` aligns first; without it the blend drifts more than 3× |
| an atomic pointer swap on `/dev/shm` (RCU) | on the BEAM, immutability gives RCU for free; a worker's shared-memory session takes a new generation at its next session | built at the BEAM level, stated |
| validate on Qwen 7B/14B and DeepSeek-V3 | not on this machine (no weights, no memory) | owed, in the ledger |
| ternary `:sb2` ("90 % less energy", "no multiplications") | the saving is memory bandwidth; ternarising a model trained in floating point destroys it (it needs a model trained ternary, such as BitNet b1.58); every backend needs a kernel | deferred, in the TODO |
| K-FAC "in the SIMD registers, without DRAM" | false: the Kronecker factors are d × d (16 M entries at d = 4096) | not built; the natural gradient with its control exists (0.16) |
| "one natural-gradient step = 10–20 AdamW steps", "3–5× fewer epochs", "half the LoRA rank" | numbers without a measurement | not claimed |
| Fisher–Rao geodesics over Lean's proof trees ("Logos") | a category error: tactic trees carry no statistical manifold | refused |
| 3D Gaussian splatting sorted by sorting networks at 60 FPS | sorting networks are O(n log² n) and suit small fixed n; millions of splats want a radix sort | refused |
| infinite context in O(1) memory | true of the SSM state, but the state is a lossy summary and recall degrades | exists (0.10), with its measurement |
| a "zero sim-to-real gap" from exact rationals | false: the gap is model mismatch (friction, compliance, sensors), not rounding | refused |
| `Vapor.Silicon` to GDSII, "the end of proprietary EDA" | the sky130 flow is open source (Yosys, OpenROAD, Magic, KLayout), and a layout that has not passed DRC and LVS is not a chip | built **Qālib**: read, map and prove, i.e. translation validation of the open flow ([QALIB.md](QALIB.md)) |
| interpretability by Gröbner bases of an LLM's Boolean polynomials | exponential beyond tiny binarised networks | refused |
| do-calculus in Almizan; "an agent may act only with a proof that X causes Y" | identifiability given a diagram is decidable (the ID algorithm is complete); causation itself is not provable from data | built: the root **s-b-b** and the logic desk's `causal`; the second half refused, and the doc says why |
| mixed-integer programming certified over ℚ | sound | built: branch and bound with a checkable tree ([LOGIC.md §6](LOGIC.md)); cutting planes owed |
| moving the PDF and JBIG2 parsers into the Zig worker under seccomp | it trades a memory-safe language for an unsafe one, for isolation the BEAM already gives each process; the real exposures are memory, time and crashes | built **the hermetic seal**, which also closed a real hole (below) |
| a frozen, content-addressed web; human-signed action plans | sound, and already half true: the agent journal never re-executes `observe` | a sanitising, budgeted fetcher is in the TODO |
| a "ghost network" (mimicry, onion routing, incentives) | a security system for specialists, and against vapor's axiom that agents have no network | refused; Tor, I2P or Yggdrasil if a transport is needed |
| metaprogramming without `eval`, hygienic, fuelled | already true of Alembic (fuel, no atoms, no I/O); Almizan has no macros, and its terms are closed | nothing to change |
| "rewrite the slides and thesis in a sovereign posture, deleting 'expectation vs reality'" | refused: the limit sections are the evidence; deleting them is marketing | kept |
| "`sketch.ex` is broken" | false: it is a working vectoriser with constraint beautification; parametric solving is a real gap | in the TODO |
| RNA folding (Nussinov/Zuker), Lenia, MCTS, docking | possible; no user brought data | not built |

### What was built

- **The hermetic seal** (`Vapor.Hermetic`, [HERMETIC.md](HERMETIC.md)): one containment primitive
  in place of four copies, now around document ingestion too.
- **Integer programs** (`Vapor.Logic.MIP`): 40/40 random programs equal brute force; forgeries
  refused.
- **Causes** (`Vapor.Logic.Causal`, the Almizan root s-b-b): estimands exact against the true
  intervention on random models; hedges checked; Lean is not asked, since core Lean has no causal
  calculus.
- **Palingenesis** ([PALINGENESIS.md](PALINGENESIS.md)): planks, the brake, the target test, RCU,
  signed lineage.
- **Qālib** ([QALIB.md](QALIB.md)): sky130 Verilog and BLIF read, circuits mapped, every step proved;
  a 20-bit trojan trigger found by SAT that 4,096 random patterns miss.
- **Recommend** ([RECOMMEND.md](RECOMMEND.md)), absorbing this repository's `SVD_Recommendation_System.py`
  and its six defects; with `Dense.svd`, one-sided Jacobi with its certificate.
- **A second kernel for Almizan**: the Lean export now targets core Lean (`Rat`, `grind`), and the
  `:lean` tier checks that Lean proves what vapor proved and rejects what vapor refuted. The ledger
  item moves from owed to tested.
- **The editor question** ([EDITORS.md](EDITORS.md)): no editor of vapor's own. The evidence goes to
  the editors people have (inlay hints), and the notebook-like surface stays the console.

Every item has its controls in §5m of the quality suite (`mix vapor.quality --only round17`:
12/12).

### What was found on the way

- **The four sandboxes capped the heap but not the binaries.** Every binary over 64 bytes lives off
  the process heap, so under a 64 MB cap a job could hold 512 MB, in the console's jailed terminal
  among other places. The seal counts both.
- **Significance without size.** The first recommender called an unstructured table "signal": a
  heavily regularised model beat the biases by 10⁻¹⁰ on every rating, which a paired test calls
  significant. The control caught it, and a minimum gain is now part of the verdict.
- **The Lean export needed Mathlib** (`ℚ`, `ring`) on a project whose proofs are core-only, and
  exported positivity claims as a vacuous `theorem … : True`. Both are fixed: core `Rat` and `grind`,
  and positivity exported as the statement, with the Bernstein witness named.
- **A rename that a pattern missed**: the Neovim client mapped the extension `alb` (no dot). The
  editors' grammars and clients were re-read by hand afterwards.
- **Duplicated lists**: the console's completion kept its own copy of the verbs (now derived from
  `Vapor.Main.verbs/0`); `bin/vapor` ignored `MIX_BUILD_PATH`.
- **Six contradictions in the documents**, found by the translation: transistors called absent after
  they were added (ENGINEERING), °C/°F refused after they were admitted (WORKBENCH), `sb4` given as
  4.75 bits/weight (that is `sb4x`; `sb4` is 4.6875), and three wrong cross-references.
- **The repository's own script** (`SVD_Recommendation_System.py`): unseeded; 60 % of the ratings
  overwritten with 50.0 and trained on as data (measured here: over 1.5× the honest error); a
  table row two values short; a grid search that saw the test set; a declared scale the data does not
  use; an RMSE with no baseline.

### What this document does not claim

- That Palingenesis has renewed a production model. It has renewed a tiny one, under measurement.
- That Qālib makes chips. It proves that the open flow's netlists compute the specification.
- That a causal verdict says anything without its diagram.
- That the seal resists code with access to the BEAM itself.

### How to contest

```sh
mix test test/vapor/hermetic_test.exs test/vapor/mip_test.exs test/vapor/causal_test.exs \
         test/vapor/almizan_test.exs test/vapor/palingenesis_test.exs test/vapor/qalib_test.exs \
         test/vapor/recommend_test.exs test/vapor/forge_cli_test.exs test/vapor/lsp_test.exs \
         test/vapor/unicode_test.exs test/vapor/audit_test.exs --include lean
mix vapor.quality --only round17                        # §5m, about a minute
mix vapor.assurance --check
bin/vapor wzn check priv/almizan/causes.wzn             # three proved, two refuted
bin/vapor recommend test/fixtures/recommend/genres_films.csv
nix flake check                                         # on the pinned nixpkgs
```

## 21. Round 0.17, continued: Kimi K3 through the airlock, the siphon, names, and fifteen proposals

> Requests 10–13 (2026-10-08), translated and condensed. With the Kimi K3 technical report: "tie loose
> ends, close TODOs, weigh the attached ideas, study the attachments, above all K3." The attached ideas:
> an editor of vapor's own (Al-Qalam, in Zig and Vulkan), a REPL with a "triple return" (Al-Mukhbār), a
> terminal mode, human–AI and human–human collaboration, a suckless rule, a canonical formatter
> (`vapor fmt`), an Emacs-like trinity of configuration, automation and plugins, what blockchains could
> give and take, self-renewal when Zig, the BEAM, Lean or Unix change, a literate format whose
> documentation cannot lie (Kitāb), faster tests ("the suites are slow and hold back evolution: is there
> a way?"), a release artifact with customisation, a guild of agents, a private cluster in a garage. "And
> the scene system (the characters, the fireflies…), far too much of a toy: aim at fine control,
> abstraction, photorealism or stylisation and beyond, or remove it. Support for models at the level of
> Fable, Astra, video models such as Flux, K3, DeepSeek 4, remembering the airlock strategy." Then: a
> proposal to stream weights from the network into a transient file system, and "a way to couple the
> internet airlock-style (a Python script with the official libraries, or something custom for my own
> cluster, a ghost network, S3 or beyond) — and **never connect the agent to the internet on its own:
> always leave the connection to the user, even with minimal effort, for safety**." Then four
> attachments for a clean-room extraction (the Perception Encoder paper, DFlash, V-JEPA 2's code, SAM 3).
> Then: "are files like llama or kimi_k3 inevitable and correct, or should models, open or proprietary,
> named by family or not, live only in the airlocks, with a pure mathematical core?"

### The instruction, scrutinised

- **"Support K3, DeepSeek 4, Fable, Astra, Flux…"** These are four different things. K3 has a report, so
  it can be admitted the vapor way: equations, an independent reference, an adapter, controls (below).
  DeepSeek 4 came with no report, so there is nothing to check an adapter against. Fable, Sol and Astra
  are closed: there are no weights to admit, only an API, which vapor reaches as a proposer through
  the agent backends, never as a source of truth. Flux-class generators and video models are owed, in
  the TODO, with what they need first (a 2D/3D RoPE checked against a reference).
- **"Never connect the agent on its own."** Taken as a rule, not a feature: **the siphon**
  ([SIPHON.md](SIPHON.md)). The person declares fetchers in `$VAPOR_HOME/siphons.json` (any program: the
  official Python libraries, `aws s3 cp`, `rsync` to their own cluster). Only the person runs one, at
  their own terminal. An agent has one tool, `siphon_propose`, which queues a request with its reason and
  fetches nothing. The console does not offer the verb at all. A fetch runs as a port with a reduced
  environment, a deadline and a byte cap; what lands goes through the format airlocks, may be pinned by
  SHA-256, and leaves a receipt with the exact argv.
- **"Stream the weights."** Refused for chat by arithmetic: decoding reads every active weight once per
  token, so a 15 GB model over a 20 MB/s link costs about twelve minutes per token, and K3 about two
  hours. What survives is header-only admission (`vapor siphon headers`, `preflight`): the airlock's
  verdict on a checkpoint from its `config.json` and two ranged reads per shard, before a byte of data
  is fetched. A layer-streaming executor for read-once work (one prefill pass) is in the TODO.
- **"Names only in the airlocks?"** Yes, once *function* and *spelling* are told apart
  ([AIRLOCK.md §11](AIRLOCK.md)). The code that computes a family's function is inevitable, and it is
  mathematics, so it is named for what it computes. A family's spelling (`model_type`, keys, tensor
  names) is data. `Vapor.Model.Llama` became `Vapor.Model.Decoder` (eight families read it), the Whisper
  and Granite adapters became `encoder_decoder` and `multipliers`, and the K3 adapter is the
  **delta-rule hybrid**, which `kimi_k3` reaches as an alias. `lock_test.exs` now fails the build if a
  module name carries a product or family name. The agent backends keep their vendors' protocol names,
  the one allowance.
- **"The scene: fine control, or remove it."** Removed (below).
- **"The tests are slow."** Measured first: the full suite is about 100 minutes on this machine (2
  vCPUs). The slow files lower and run models on the native worker (the K3 file takes about three
  minutes, a third of it one lowering); the Python oracles that run here start numpy, not torch, so
  start-up is not where the time goes. The answer is a cache keyed on what a test file can reach
  (below), plus one compile-time fix (a full build is 32 s).

### Kimi K3, checked ([KIMI.md](KIMI.md))

| claim or piece | verdict | evidence |
|---|---|---|
| KDA, Gated MLA (NoPE), Block Attention Residuals, Stable LatentMoE, SiTU-GLU | admitted as one step program (`Vapor.Lock.Adapters.DeltaHybrid`) | logits within 4.4·10⁻⁶ (relative) of an independent float64 reference written from the report alone; 1.35·10⁻⁵ with the soft caps biting, a low-rank query and MXFP4 experts; greedy decoding identical; native worker = oracle, bit for bit; forgetting the KDA state or the MLA cache fails |
| chunkwise KDA (Eq. 4) equals the recurrence (Eq. 1) | holds | 3·10⁻¹⁶; the UT transform the report defers to Kimi Linear was derived and checked |
| the bounded decay (Eq. 5) keeps the chunk's reciprocal decay finite | holds | float32, a 16-token tile: finite with `g ∈ (−5, 0)`, overflow with Kimi Linear's form |
| MXFP4 experts | exact | every MXFP4 value inside binary32's range is a binary32 value; overflow and NaN scales refused by name (`Vapor.Quant.MXFP4`) |
| Quantile Balancing: the relaxation is integral | holds | a bipartite b-matching (totally unimodular); the rational simplex returns 0/1 with a checked certificate |
| Quantile Balancing, Algorithm 1, recovers the balanced assignment | **does not, as written** | thresholds set *at* the (k+1)-th entry put margins at zero and create the ties the appendix calls measure-zero; balanced in 5 of 60 batches. Midpoint thresholds reach the certified optimum in ≤ 10 rounds on every batch (`Vapor.Train.Balance`). The training recipe is unaffected |
| K3's own `config.json`, tensor names and MXFP4 layout | **not verified** | huggingface.co is refused by this machine's network; the spelling is vapor's, and a real checkpoint spelled otherwise is refused with the field named, reconciled by an alias file |

### The attachments for a clean-room extraction

Each was read and written up as a specification in prose; the code was written from the
specification, not from the source.

| attachment | what was extracted | built |
|---|---|---|
| Perception Encoder (Bolya et al., 2025) | the finding that the best features are inside the network, not at its output, as a protocol any encoder can be put through | Assay `layers`: a probe per layer, the layer and λ chosen on validation, test read once, McNemar against the output, a shuffled-label control |
| SAM 3 (Meta, 2025) | **cgF1** = 100 · pmF1 · IL_MCC, which separates "is it there" from "where" and forces calibration | Assay `detect`: the IoU matching solved exactly by the rational simplex (a totally unimodular program, so the optimum is certified), presence over images with and without the object, a bootstrap interval; a greedy matching loses a true positive the optimal one keeps |
| DFlash (2602.06036) | block drafting is a chain, a degenerate tree; losslessness needs the verify pass's argmax to equal step-by-step decoding, a condition the paper does not state | nothing new needed: vapor's verify pass is bit-identical to single steps by batch invariance (`speculative_test.exs`); a drafter bound to its target's hash is in the TODO |
| V-JEPA 2 (code) | its RoPE pairs adjacent elements but tiles the angles, so the two elements of a pair get different angles: not a rotation, a bug the code keeps for checkpoint compatibility (2.1 fixes it). Also LayerNorm ε = 10⁻⁶ and the CEM planner's settings | not built: no video encoder is admitted yet; the quirk is recorded for whoever writes that adapter |

### The proposals, weighed

| proposal | verdict | what was done |
|---|---|---|
| **Al-Qalam**, an editor in Zig and Vulkan (MSDF glyphs, "opens in 4 ms, < 45 MB") | the numbers are unmeasured, and a GPU editor is years of work the evidence does not need. Then the person asked for an editor anyway, "even if only for me": a request is a reason, and the size must follow from the use | built suckless: `vapor qalam` (`Vapor.Qalam`, ~650 lines, no dependency), a vi subset in the terminal with the **balance in the gutter**, **scrubbable numbers** (walk a damping coefficient to zero and watch `✗` turn into `✓`), `%` and top-level motions, `:fmt`/`:ar`/`:la`, and a **Merkle undo tree**. Not built: Vulkan, MSDF, 144 FPS, viewports, the "Composer", plugins ([EDITORS.md](EDITORS.md), "Al-Qalam") |
| **Al-Mukhbār**, a REPL with a "triple return" (exact value, proved envelope, silicon cost) | the value is already exact (`vapor alembic -e`, `vapor wzn run`); a cost in nanoseconds is a measurement, not part of an answer that must be the same everywhere; instructions and spills are deterministic and could be reported | not built |
| "the editor does nothing, it only looks" (suckless) | agreed: that is what a language server is | the server's formatting no longer deletes comments (below) |
| the centaur: agents propose, the person decides | already the rule of the Touchstone, and now of the network (the siphon) | — |
| human–human collaboration by CRDT over a Merkle AST, P2P over Nebula | merge by union holds for disjoint edits only; a transport is a security system of its own (§20 refused the ghost network) | not built |
| "hermit mode": no network, ever, unless asked | already true; the siphon is the only door, and only the person opens it | the siphon |
| **`vapor fmt`**: canonical form, glosses for comments, fractions in lowest terms, zero configuration | sound, except **reordering declarations**: the order is the author's argument, and identity already ignores layout and script | built (`Vapor.Almizan.Format`, `vapor wzn fmt --check/--write`): the printer's form with comments kept as glosses; the language server uses it |
| configuration as a Tabula contract, automation in Alembic with fuel, plugins in Elixir | the second and third exist; the first needs an editor with keymaps to configure, and Tabula already decides any such file (`vapor tabula`) | — |
| from blockchains: UTXO as linear types for registers; AIR traces | the register allocator already has a checker proved in Lean; ZK lives in [ZK_FHE.md](ZK_FHE.md) | — |
| to blockchains: Tabula for contracts, exact arbitrage, deterministic oracles | Tabula is propositional and deontic, not a contract language; exact arbitrage exists (the finance desk's LP certificates); bit-identical inference across substrates is exactly a deterministic oracle | nothing to build; the claim that holds is stated |
| self-renewal: canary workers, hot code loading, Lean extraction, unikernels, proof-gated upgrades over the ghost network | the first three exist (the substrate airlock admits a worker by probes; the BEAM; extraction checked byte for byte); the Linux worker needs no libc, the BEAM does; the last rests on the network refused in §20 | — |
| **Kitāb**, documentation that cannot lie | sound, and needs no new file format | built as a test over the docs as they are: every `Vapor.…` module and function they name exists, every repository path they name exists, the Almizan pair in two scripts has one hash. It found a citation of the retired Alembic sandbox, a misspelled JBIG2 module and `priv/games` |
| teaching models the languages by constrained decoding | exists (the JSON Schema grammar, the repair loop of `Vapor.Mind`) | — |
| faster tests: a Merkle test cache, frozen oracles, binary tables for `ccitt.ex`, tiers | the cache is right; the oracles that matter are already frozen as fixtures, and the tiers that call Python are the independent checks, run when their tooling is present; the `ccitt.ex` diagnosis was right and its cure is simpler | built: `mix vapor.test` (below); `ccitt.ex` compiles in 0.3 s instead of 5 |
| a release artifact "of about 4 MB" via `mix vapor.archive` | `vapor.archive` signs and verifies *result* archives, not releases; a release carries the BEAM runtime | not built this round |
| a guild of agents with locks on AST nodes and branches in `/dev/shm` | many agents on one repository is solved by branches and tests; locks on syntax do not stop semantic conflicts | not built |
| a private cluster in the garage | `Vapor.Cluster` exists (content-addressed cache, redundant audit, quarantine, hedging); "a notebook with 8 GB that acts as if it had 128 GB of VRAM" is a remote server, which `mix vapor.serve` is | — |
| the SVD recommender of this repository | absorbed in §20 (`Vapor.Recommend`) | — |

### What was built

- **K3** through the airlock, MXFP4, Quantile Balancing with its flaw and fix, header-only preflight.
- **The siphon** (`Vapor.Siphon`, `vapor siphon`, MCP `siphon_propose`).
- **Names**: the core and the topologies named for what they compute; products only as spellings.
- **Assay `layers` and `detect`**, from the Perception Encoder and SAM 3.
- **`vapor wzn fmt`** and a formatting language server that keeps comments.
- **`mix vapor.test`**, a content-addressed test cache (`Vapor.TestCache`): a test file's key covers the
  file, the support files, the fixtures, `priv/`, the native binaries, the toolchain, the excluded tiers,
  the bytecode of every module it can reach (transitively through the atom tables), and every file under
  the repository trees its text names (`docs/`, `lib/`, `notebooks/`, `bin/`…), because a test that
  reads files at run time depends on them. That last part was found by the full suite: the Livebook
  tour failed on a stale name, and a cache keyed on modules alone would have kept skipping the tests that
  read the docs or the sources. A file is recorded only when every
  test in it passed. Change one module and only the files that can reach it run again; change `priv/` or
  a fixture and everything runs. It is for the edit loop, not a release gate: `mix test` ignores it.
  Measured: the suite's 159 files keyed in about 4 s; a second run of a passing subset went from 19 s
  to 4.3 s.
- **The scene, removed, and the renderer, extended.** The living scene was a 2.5D canvas engine whose
  depth was a heuristic "stated as such", directed by a grammar of clauses. Nothing in it could be checked
  against anything but itself, in a project where every other result carries its evidence. Removed:
  `Vapor.Scene`, `Vapor.Scene.Ops`, the console tab and its HTML export, `vapor scene`, MCP `scene_ops`,
  the mind layer's `direct/3`, the archive kind `scene`, and `Vapor.Alembic.Tree`, which existed only to
  run a scene's motion in the browser. Kept: sketch → drawing and floor plan → 3D, now in
  [SKETCH.md](SKETCH.md), with the raster tools in `Vapor.Raster`. The answer to "fine control,
  abstraction, photorealism or stylised" is the renderer that was already checked, plus **ink**
  (`Vapor.Render.ink/2`): the same scene text as flat bands, hard sun shadows and outlines on silhouettes
  and folds, deterministic, in milliseconds, so a scene can be composed in ink and then rendered with
  physical light. In the console, the CLI (`vapor render --ink`) and MCP (`render_scene`, `style: "ink"`).
  Its quality check has a control: the ball's shadow reads exactly 0.8 × 0.3, and without the ball the
  same pixel is lit.

### What was found on the way

- **Formatting deleted comments.** The language server's formatting and its Arabic/Latin lens printed
  the parsed tree, and the reader drops comments, so formatting a file erased its glosses. The end-to-end
  test now formats a messy document with a comment and checks the comment survives.
- **`bin/vapor` overwrote the person's `VAPOR_HOME`.** It used the name for the repository root, and an
  assignment to an exported variable in `sh` is exported: with `VAPOR_HOME=~/.vapor` set, `vapor chat`
  and `vapor siphon` read the repository instead. A test runs the launcher with the variable set.
- **Quantile Balancing's Algorithm 1** (above).
- **`ccitt.ex`** built its Huffman tables in the module body: four seconds of every compile of that file.
  They are built once per VM now.
- **Stale documents.** The console's guide still described panels removed in 0.16 (Physics, Networks,
  Training, Mathematics, Algorithms, Science, Games), the README listed `graph.ex`, `discover.ex` and
  `games.ex`, and two documents cited `Alembic.sandbox`. All fixed; the new docs test catches the kind.
- **Dead code**: the Assay CLI built a list of documents and discarded it.
- **Two test files with Portuguese names** (`rodada12`, `rodada13`), now `quality_round12/13`.

### Asked later in the round: the editor, technical documents, and super-sampling

> Translated: "for the record, I would very much like Al-Qalam (the editor), even if only for me;
> a new and separate set of slides and monograph in LaTeX, this time with a technical and
> architectural emphasis, without the record of historical scars and evolution (we keep the current
> ones for that), to serve as technical, professional presentations of the system as it is; and weigh
> the brainstorm again for anything else worth taking, remembering suckless." Then a proposal for a
> "sovereign DLSS": super-sampling by an implicit neural representation in ternary weights, on the CPU
> at 4K and 60 FPS, with ghosting forbidden by a formal invariant.

- **Al-Qalam**: built, suckless (the row in the table above).
- **The technical monograph and slides** (`technical/`, `make technical`): 33 pages and 18 slides,
  LuaLaTeX, in English; the system as it is (layers, semantics, compiler, runtime, ladder, airlocks,
  deciders, languages, evidence, interfaces, limits), no history. The academic thesis and its deck
  stay as they were.
- **VaporScale**, claim by claim:

| claim | verdict | what was done |
|---|---|---|
| DLSS, FSR and XeSS are closed | half wrong: AMD's FSR is open source (MIT), and runs on any GPU | — |
| an implicit representation F(x, y, t) makes the cost "constant" whatever the output size | the memory of the representation is constant; the work is one query per output pixel, so it scales with the resolution | — |
| ternary weights make it additions only, "4K, 8K, 16K in the same clock cycle", 60 FPS on a CPU | a 4K frame is 8.3 million pixels; at 60 FPS even a few hundred operations per pixel is about 10¹¹ per second, and ternary needs a model trained ternary (§20) | not claimed |
| ghosting forbidden by a formal invariant on the optical flow | the invariant that holds is consistency, `D(y) = x` for every frame, which the upscaler already guaranteed for one frame; it cannot forbid detail inside a block, and disocclusions have no history to transport | built `Upscale.temporal/4`: history moved by the motion, clamped to the current frame's local range, blended, projected. Every frame stays consistent whatever the history holds; a half-pixel pan gains +0.72 dB, a whole-pixel pan (the control) +0.08 dB; a vanished object's ghost is rejected, where a naive blend keeps it |
| open, any hardware, the same bits everywhere | true of everything compiled by vapor | the temporal step runs on the BEAM today; a compiled kernel is owed |

- **The brainstorm, again, under suckless.** Nothing else passes the bar of being small and useful
  with one's own data: the REPL's deterministic parts already exist (`vapor alembic -e`,
  `vapor wzn run`); "Kitāb" is the docs test; paredit's slurp and barf would double the editor for a
  convenience; a release artifact (`mix release`) is real but needs the Mix-task verbs (`serve`, `tui`,
  `ocr`, …) rewritten as entry points that do not need Mix at run time, so it is in the TODO with that
  reason, not half-built.

### What this document does not claim

- That K3 runs at its real size here, or that its real spelling was read.
- That the siphon makes a fetcher safe: it bounds what a fetch can do and what it brings in, and leaves
  the choice of fetcher to the person.
- That a cached test passed on this commit: it passed on the same bytes of everything it can reach.
- That ink is a style engine: one style, with its thresholds.

### How to contest

```sh
mix test test/vapor/kimi_test.exs test/vapor/balance_test.exs test/vapor/siphon_test.exs \
         test/vapor/assay_detect_test.exs test/vapor/assay_layers_test.exs test/vapor/almizan_format_test.exs \
         test/vapor/test_cache_test.exs test/vapor/docs_references_test.exs test/vapor/render_test.exs \
         test/vapor/lock_test.exs test/vapor/lsp_test.exs
python3 test/python/kimi_k3_reference.py /tmp/k3 k3 1   # the independent reference, from the report alone
mix vapor.test --dry                                    # what would run, and why the rest would not
printf 'camera pos=0,3,5 look=0,0,0 fov=45\nsun dir=0,1,0 color=1,1,1 power=2\nplane y=0 mat=diffuse albedo=0.8,0.8,0.8\nsphere c=0,1.2,0 r=0.6 mat=diffuse albedo=0.2,0.4,0.8\n' \
  | bin/vapor render - --ink --out ink.png              # the quality check's scene, in ink
```
