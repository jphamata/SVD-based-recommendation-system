# Output quality: signal or noise?

> A quality test that has never been seen failing noise tests nothing.

## 1. The pain

In almost every inference repository, "works" means "generated
something". vapor's own demo model (`mix vapor.demo_model`,
random weights) generates `" scen scen authorised authorised scen …"` and all the
tests passed. Bit-for-bit parity proves that two substrates agree — not
that what they agree to produce is signal. In industry, the same
problem shows up as regressions in tokenization, *chat template*,
quantization or *kernel* that leave the output plausible at first sight and
degraded in fact; in academia, as metrics with no baseline and no control.

## 2. The method

### 2.1 Calibrated gates that refuse to exist

`Vapor.Quality.Gate` has no hand-picked threshold. Each gate is **fitted to
controls**:

- **negative** (must fail): white noise; the signal itself **shuffled**
  (same marginal distribution, no structure — the hardest control);
  degenerate repetition;
- **positive** (must pass): real **held-out** signal.

`t_noise` = the worst negative; `t_natural` = the worst positive. If `t_noise ≥
t_natural`, calibration returns `{:error, {:inseparable, …}}` and **there is no
gate** — for example, samples too short to separate text from shuffled
letters. Between the two thresholds lies `:structured`: more structure than
any noise, less than all real signal. "Not noise" = `:structured` or
`:natural`.

| modality | main score | why |
|---|---|---|
| text | fraction of byte trigrams present in a reference corpus | random bytes almost never hit; shuffled letters hit some; real text hits most |
| text (collapse) | zlib compression ratio of the sample itself | degenerate loops compress too much |
| image | neighbour correlation (lag 1, luminance) | natural images ≈ 0.8–0.99; noise and shuffled pixels ≈ 0 |
| audio | spectral flatness (Wiener entropy) | white noise ≈ 1; tones → 0 |

More measures are available (1/f spectral slope, PSNR, SSIM, SNR,
dominant frequency, corpus-conditional compression, entropy, UTF-8).

### 2.2 Planted models: truth in closed form

`Vapor.Quality.Planted.bigram/3` writes — does not train — the weights of a Llama
decoder so that it **is exactly** the counted bigram of a corpus:
one-hot embedding, all attention and MLP branches null (they add +0 to the residual
stream), head = `log P(j|i)/s` with `s` what RMSNorm does to `e_i`. The
checkpoint is a `config.json` + ordinary weights; it goes through the airlock, compiler,
ladder and substrate like a real model, and **must reproduce the analytic
table** — measured: |Δ| ≤ 1.2·10⁻⁶ in the logits, bits/character identical to
the table's down to 10⁻⁹. Any defect in the stack becomes a measurable deviation.

### 2.3 For a real checkpoint

`mix vapor.quality --model PATH` (`Vapor.Quality.Model`) requires **two**
independent tests:

1. **bits per byte** on held-out text (*teacher forcing* in windows), against the
   byte unigram of a reference corpus and against the uniform (8);
2. **generations through the calibrated text gate**.

Measured on the demo model (Qwen2, real vocabulary of 151,936 tokens,
random weights): 5.59 bits/byte against 5.10 for the unigram → **noise**; 2 of 3
generations failed. On a planted byte bigram with the byte BPE
tokenizer: 4.01 bits/byte < 5.05 → **signal**, generations passed. Both cases
are in the tests.

## 3. Honest findings (measured, not presumed)

- **The trigram gate alone can be fooled by large BPE vocabularies.**
  Random tokens from a real vocabulary are already real subwords
  ("actualizar_standard bart…"), and one of the three generations of the random model
  passed as `:structured`. That is why the verdict on checkpoints requires bits
  per byte as well; it was that criterion that failed the model.
- **Strict UTF-8 is too rigid for byte models**: a byte bigram
  cuts a multibyte character now and then. The gate requires ≥ 95% of the
  bytes to be inside valid characters — it still catches broken *detokenization*.
- **Padded vocabulary**: real models have an embedding matrix larger than the
  tokenizer (Qwen2: 151,936 vs. 151,646). IDs beyond the tokenizer count
  against the sample instead of crashing the test.
- **The text gate measures language statistics, not meaning.** The text
  of the planted bigram is `:structured` — and that is what it should be. For meaning
  a task with an exact answer is needed (the colour ↔ note translation, with accuracy 1.0).
- **TIES and DARE make the merging of dense models worse** (see [MERGING.md](MERGING.md)).

## 4. The suite

`mix vapor.quality` (a few minutes on the native worker, most of it on the real data) runs everything and writes
[bench/QUALITY.md](bench/QUALITY.md), `bench/quality.json` and the gallery
`bench/modal/` (PNG ×8, WAV). It exits with status 1 if any check fails
— CI can make the *merge* conditional on "the outputs are signal". The checks
(55 in 0.7.0):

- calibrated gates with a positive margin (text, collapse, image, audio);
- planted model = table; bits below the unigram; generations passed; **the
  same generations with random weights failed** (the control);
- ten any-to-any routes on held-out inputs, each with a control
  ([ANY_TO_ANY.md](ANY_TO_ANY.md));
- merging: linear beats the wrong specialist in each domain, `merge(A, A) =
  A` and `slerp(t = 0) = A` bit for bit, verifiable receipts;
- substrates: every native modal program = oracle bit for bit;
- real data (0.5.0): OCR, speech, handwriting in both directions, merging of
  trained transformers — each with its own control;
- **round 0.6** (`Vapor.Quality.Round06`, §5b of the report): each new
  feature against its truth **and** against a control that shows the
  check discriminates — sparse experts = dense (control: a
  perturbed chosen expert changes the bits); latent MLA = expanded
  (control: permuted projection); window = attention over the moved window
  (control: full attention); `÷` = IEEE (control: the old `a·rcp(b)`
  gets ~30% wrong); convolution = direct binary64 (control: mirrored kernel,
  correlation in place of convolution); transparency log on the
  transparency-dev probes (control: naive verifier); native Mamba = oracle
  (control: stateless); tree speculation = greedy (control: draft
  accepted without checking).
- **round 0.7** (`Vapor.Quality.Round07`, §5c): CCITT = libtiff (control:
  the wrong declared encoding); scanned pages read in order
  (control: without the ordering, 53% CER); language model (control: the same
  corpus shuffled gains nothing); **the model's abstention** (control: without
  the guard it rewrites 27 of 30 random lines); codes and values do not
  get worse; generated formats valid according to OTP's *parsers* (control: the naive
  date pattern generates 145 invalid out of 150); merging from disk with the same
  root and a tenth of the memory (control: another `t` changes the root).
- **round 0.10** (`Vapor.Quality.Round10`, §5f): the substrate airlock
  (control: a simulated bf16 engine, refused with 8 bits measured); the
  envelope with DAZ (control: twice the exact value, out); the model
  trained against Witten–Bell and against its own shuffled text; the endless
  stream against growing positions; the first-order pendulum; chaos
  bit for bit across substrates (control: one ulp); the twin's identification
  (control: measurements shuffled in time); the cart-pole (control:
  null policy); the twin's alarm (control: no fault); the networks against
  their nulls; the charts (control: permuted labels, **all
  refused**); the formulas (control: flat reading); CJK (control:
  random characters — the language model cannot help, and in Japanese and
  Korean it has to help); Arabic and Cyrillic (control: the Latin reader).
- **round 0.11** (`Vapor.Quality.Round11`, §5g, 26 checks): sorting
  networks at the known optima (control: randomly drawn and pruned); the minimal bit
  trick (control: the naive formula overflows); the exact rank 7
  (control: rank 6 never); 15 theorems and false twins by both routes;
  conjectures (control: no trivial triple); homology over ℚ against
  GF(2); the persistence of a loop against a blob; 11 science
  experiments, each against its reference and its control; the self-play
  agent against perfect play (control: search without training); varied
  worlds against one world; the scene, the skeleton, the direction (control: a
  meaningless word is reported), the sketch (control: without constraints,
  it stays crooked), the floor plan; and the archive (control: one byte changed and
  re-zipped).
- **round 0.12** (`Vapor.Quality.Round12`, §5h, 27 checks, ~6 s):
  Dormand–Prince against fixed-step RK4; Robertson at the values of
  Hairer & Wanner; Crank–Nicolson's order 2 against implicit Euler's order 1
  by a manufactured solution; incoherent units refused before
  running; the optimal can with bounds (control: without bounds, "diverged");
  trapezoids against Euler on the RC; Newton against Gauss–Seidel on the
  Stagg system; 10 elements against 1 in the modes; QM6 against the Q4 that locks;
  stoichiometric invariants against a species that is not conserved;
  Fenske against reflux below the minimum; S(3) with a witness and DRUP
  (control: the truncated proof); Knuth–Bendix against the axioms only
  oriented; Thales against the false variant; the right proposal against the
  swapped one; chess and shogi perft, legal Go positions; CFR+ against
  uniform play; the planned alignment against the shuffled one (precision and
  fold); the furnace against the biased estimator; the direction with names against
  the sentence with no one in it. Also in `rodada12_test.exs`.

## 5. Limits

- The planted worlds and models are **instruments**: they prove that the stack
  carries signal, not that a production model is good. For that: the
  `--model` option with held-out text in the model's language, and task evaluations.
- The image gate is calibrated on synthetic 16×16 scenes; real images
  call for recalibration with held-out photos (the API is the same).
- The bits-per-byte evaluation for large vocabularies runs the log-softmax on the
  BEAM (≈ 6 min for 1500 bytes with 151,936 tokens); moving it to the substrate is
  an open item.
