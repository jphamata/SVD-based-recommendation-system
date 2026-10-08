# Changes

## 0.17.0 — 2026-10-08

Scrutiny of the ninth request (translated: "a complete, first-principles re-engineering; ruthless
cleanup; the core rebuilt; an original UI; rigorous tests; the source as a zip; English the
default; rename alchemically") and of its attachments, claim by claim:
[docs/DIRECTIVE.md §20](docs/DIRECTIVE.md). Checks with a control in `mix vapor.quality` (§5m, 12;
`--only round17`: [docs/bench/ROUND17.md](docs/bench/ROUND17.md)).

**The build.** `nix develop` failed on the current nixpkgs (`poppler_utils` renamed; top-level
`elixir`/`erlang`, `nixpkgs-fmt` and `texlive.combine` deprecated): fixed, and every flake output
evaluates on the pinned nixpkgs without error or warning. On Elixir 1.18.5 / OTP 28 the code did
not compile: regexes in module attributes are references now (`Tabula`, `Scene`, `Vision.Figure`).
That and every warning are fixed, among them a real `MatchError` in `mix vapor.lock` without
arguments. The Lean development moved to Lean 4.34.1 (deprecated lemmas, one `omega`), and
extraction is byte-identical. A test that pinned 941 primary composites knows Unicode 16.0's 961.
`bin/vapor` honours `MIX_BUILD_PATH`.

**English.** README, CHANGELOG, every document, the generated reports and the ledger are in
English; documents and tests have English names (`DIRETRIZ.md` → `DIRECTIVE.md`,
`GARANTIAS.md` → `ASSURANCE.md`, `bancada_test.exs` → `workbench_test.exs`, …). The thesis and its
slides stay in Portuguese. Six contradictions in the documents, found while translating, were fixed.

**Names.** Mīzān → **Almizan** (`Vapor.Almizan`, `vapor almizan`, `vapor.almizan.toArabic`, the
editors' modes; the hashed format tag is kept, so 0.16 modules keep their hashes). Alembic files:
`.alb` → **`.nbq`**.

**The hermetic seal** ([HERMETIC.md](docs/HERMETIC.md)) — `Vapor.Hermetic.seal/2` and `cap_self/1`
replace four copies of the same containment (the Alembic sandbox, the Dīwān's jail, Athanor's
session, the interactive furnace). The copies capped the heap only. Off-heap binaries escaped them:
512 MB under a 64 MB cap. The seal counts them, and document ingestion now runs sealed.

**Integer programs** ([LOGIC.md §6](docs/LOGIC.md)) — `Vapor.Logic.MIP`: `int`/`bin` declarations
on the LP text form, branch and bound over the exact simplex, and a certificate (the tree, the
incumbent, a Farkas or dual vector at every leaf) checked by code separate from the search. 40/40
random programs equal brute force; forgeries are refused. `logic_check` takes MIP proposals.

**Causes** ([LOGIC.md §7](docs/LOGIC.md), [ALMIZAN.md](docs/ALMIZAN.md)) — `Vapor.Logic.Causal`:
the ID algorithm (complete) returns the estimand or a hedge, checked; the back-door criterion;
d-separation. Estimands equal the true intervention in exact rationals on random models; the naive
`P(y | x)` is the control. The Almizan root **س-ب-ب s-b-b** puts causal claims beside conservation
laws, in both scripts.

**Palingenesis** ([PALINGENESIS.md](docs/PALINGENESIS.md)) — a model as planks under Merkle roots; a
plank enters through a contract gate, a Fisher–Rao drift brake on anchor sequences, a paired target
test and caller invariants; generations are published by read-copy-update on immutable values; the
lineage is a chain of (optionally signed) records that `verify/2` re-derives. Re-Basin alignment
where it matters (blending), not where it does nothing (whole-block replacement, measured).

**Qālib** ([QALIB.md](docs/QALIB.md)) — structural Verilog and BLIF over sky130_fd_sc_hd cells read
(unknown cells refused), circuits mapped to cells (`:cells`, `:nand`), and every step proved by
Rebis; a 20-bit trojan trigger in a 21-input adder found by SAT, where 4,096 random patterns miss it.

**Recommend** ([RECOMMEND.md](docs/RECOMMEND.md)) — the repository's `SVD_Recommendation_System.py`,
absorbed: biased factorisation by ALS, nested selection, two baselines, a paired test with a
minimum gain, a shuffled control. `Dense.svd/1` (one-sided Jacobi) with its certificate.

**Editors** ([EDITORS.md](docs/EDITORS.md)) — the question "a vapor editor?" answered (no: the
evidence goes to the editors people use); the language server gains inlay hints (each claim's
verdict at the end of its line), Neovim enables them, JetBrains IDEs are listed. Almizan's Lean
export targets core Lean (`Rat`, `grind`), and the `:lean` tier checks that Lean proves what vapor
proved and rejects what it refuted.

**Interfaces** — `vapor logic`, `vapor qalib`, `vapor recommend`, `vapor palingenesis`; MCP
`qalib_check`, `recommend_run` (27 tools); the console's completion derives its verbs from
`Vapor.Main.verbs/0`.

## 0.16.0 — 2026-10-07

Scrutiny of the eighth request (translated: "loose ends; purge of the merely illustrative; conversations and agent up to
the level of the frontier products; equivalent GUI, TUI and API terminal; the ASAS; the Almizan
manifesto; an ecosystem of editors; information geometry; purify") and the three critiques
answered: [docs/DIRECTIVE.md §19](docs/DIRECTIVE.md). The thesis: **purify** — what was only
on display went out; what stayed is a tool with evidence. Checks with a control in
`mix vapor.quality` (§5l, 11; `--only round16`, ~2 s: [docs/bench/ROUND16.md](docs/bench/ROUND16.md)). The version in `mix.exs` had been 0.14.0
since round 0.14; now 0.16.0 (0.15 went in without touching it).

**Purge** — 2,949 fewer lines in what existed. Out went `Vapor.Graph` (complex networks),
`Vapor.Discover` (algorithm discovery) and `Vapor.Games` (AlphaZero on tic-tac-toe; the
generic `Vapor.Play` stays), the console panels Physics, Networks, Algorithms, Mathematics, Science,
Games, Draw, Listen, Training and Merging, the demonstration dossier, the TUI's draw, listen
and merge commands, the corresponding HTTP routes, 84 interface strings that only they used and the
documents REDES, DESCOBERTA and JOGOS. Criterion: a feature stays if it answers a question
someone brings with their own data; it goes if it only re-enacts a fixed example. The furnace test of
Render stays: it is a check with a control, not a demonstration.

**Khazāna** ([KHAZANA.md](docs/KHAZANA.md)) — the content-addressed store with a *crash-atomic* root from
ASAS §8.4 in a POSIX directory: two packs, two 128-byte root slots with sequence and
tag, no file created after `init/1` (OTP does not give `fsync` on a directory).
A crash injected at **every byte** of a *commit*: always the old root or the new one. HMAC-SHA256
capabilities with mass revocation by generation (ASAS §6.2).

**Majlis** ([MAJLIS.md](docs/MAJLIS.md)) — conversations as a content-addressed tree (the hash
of a message covers its parent): edit and another answer as branches, ‹ i/n ›, continue from here,
fork in O(1), pinned messages, the context computed and shown (sent, pinned,
summarized, left out; exact tokens with the served tokenizer), compaction that names the
hash it summarizes and can be undone, BM25 search, verifiable JSON export and Markdown, import from
ChatGPT and from Claude, shared links by capability and revocable, tools by
allow-list with a verifiable journal. `vapor chat …`; `/v1/vapor/threads…`; the *Conversations* panel.

**Dīwān** ([DIWAN.md](docs/DIWAN.md)) — one interpreter for the command line, the TUI, the console's terminal
and the API: verbs, `|`, `>`, `>>`, `<`, `;`, quotes, built-ins; in the console, a
jailed session (its own files, no external process, a heap ceiling and a deadline per command). The TUI was
rebuilt on top of it. The *Terminal* panel: ANSI colours, history, Tab, file editor.

**Almizan** ([ALMIZAN.md](docs/ALMIZAN.md)) — the formal dialect `.wzn`: a neutral tree with two
bijective printings (Latin/Buckwalter and Arabic, Arabic-Indic numerals), the tree's hash
as identity; roots (ح-س-ب, ح-ف-ظ, ن-ق-ل, ك-ت-ب) as domains and *awzān* (فاعل, مفعول,
برهان) as regimes; obligations decided by exact normal form over ℚ, by Aludel and by SAT
with DRUP — *unknown* does not compile; lowers to vapor's compiler, to AIGER and to Lean 4
theorems; `abjad` measured (99.99 % of the roots collide) and shown, never used as an address.
`vapor wzn check|show|hash|run|transmute|assay|abjad`.

**Editors** ([EDITORS.md](docs/EDITORS.md)) — `vapor lsp` (LSP 3.17 over stdio: diagnostics
with the decided obligations, *hover*, completion, definition, symbols, formatting, switching
of script) and thin clients for VS Code, Neovim/Vim and Emacs.

**Information geometry** ([GEOMETRY.md](docs/GEOMETRY.md)) — `Vapor.InfoGeom` (Fisher–Rao
on the simplex, geodesics, Karcher mean, normals in closed form, logistic regression by
natural gradient) and `vapor assay geometry`, each one with the control that shows why the metric
matters.

**ASAS** — beyond the Khazāna: **one** entropy boundary (`Vapor.Entropy`; everything else seeded
and reproducible, checked by an audit on every build) and the **assurance ledger**
(`Vapor.Assurance` → [ASSURANCE.md](docs/ASSURANCE.md), `mix vapor.assurance [--check]`):
proved · checked · tested · argued · owed, with a test that fails if the
cited evidence disappears or the document diverges.

**Fixed along the way** — a regenerated answer did not become the current one (the pointer stayed on the
old one; the test said "the pointer follows it" without checking); top-level messages of different
conversations were siblings of each other (anchor per conversation and owner in the hash); a leading `|` was accepted
in the terminal; round 0.15 used the OS generator in test inputs; the test backend of the
conversations depended on the load order of the test files (now in `test/support`).

## 0.15.0 — 2026-10-07

Scrutiny of the seventh request (translated: "complete re-engineering from first principles, in six phases;
absorb the attachments; rename while keeping the alchemy") and of the attachments (PALADIN with JESTER and WIZARD,
HYDRA-Z, GHOST) and of the two lists of ideas, item by item — done, already existed, deferred or refused,
with the reason: [docs/DIRECTIVE.md §18](docs/DIRECTIVE.md). The thesis: **decide, do not display** — every
new piece returns a verdict that can be checked without trusting it. Checks with a control in
`mix vapor.quality` (§5k, 12; `--only round15`, ~2 s, and `--md` to write the table:
[docs/bench/ROUND15.md](docs/bench/ROUND15.md)).

**Amalgam** ([AMALGAM.md](docs/AMALGAM.md))
- `Vapor.Amalgam`: exact, order-free sum (a Kulisch accumulator in the BEAM's integers) for
  f16/bf16/f32/f64; `merge/2` is a commutative monoid; rounds once, to even, with
  gradual *underflow*; specials per IEEE 754 §6.3 in any order; `dot`/`partial_dot`
  (slices of a contraction joined in any order = the product without slices); correctly
  rounded `mean`; `to_wire`/`from_wire`.
- `reduce: :exact` training: any number of micro-batches; the bits depend only on the set of
  micro-batches (1, 2 or 3 workers, any assignment, a crash, the oracle — one *digest*);
  *checkpoint* and resume preserve the bits.

**Cupel** ([CUPEL.md](docs/CUPEL.md))
- `Vapor.Cupel`: silent corruption via `y·r = x·(Wᵀr)` in exact arithmetic with Higham's
  tolerance proved (+ FTZ/DAZ); `:ok | :corrupt | :unchecked` per row; exact int8; per-bit
  detection profile (`sensitivity/3`).
- `Vapor.Cupel.Sentinel`: a guard over a set of substrates with quarantine on the evidence,
  recomputation on the next healthy one or on the oracle, SHA-256 + Merkle journal; concurrent calls.

**Rebis** ([REBIS.md](docs/REBIS.md))
- Circuit equivalence: truth table in bits (≤ 16 inputs) and ANF by Möbius; beyond that,
  random simulation and *miter* + CDCL with a **checked DRUP** proof; shrunk counterexample;
  our own *netlist* and ASCII AIGER (reading and writing).
- `Rebis.Ideal`: word identities by algebraic rewriting over ℤ (the Gröbner basis of the
  circuit) — 32-bit multiplier in 1.4 s; `:unknown` with a ceiling, never a guess.
- `Rebis.Field` (GF(2ⁿ), carry-less, derived S-box, GHASH two ways), `Rebis.GCM`
  (= OpenSSL), `Rebis.Stabilizer` (CHP tableau with phase by masks, = dense simulator),
  `Rebis.Gen` (*ripple* and Kogge–Stone adders, multiplier, Trojan horse).

**Aludel** ([ALUDEL.md](docs/ALUDEL.md)) — absorbed from PALADIN: polynomial positivity on a
box in exact integers (dense Bernstein, de Casteljau), three verdicts, witness reproduced
by direct conversion per leaf, `enclose`, barrier certificates and synthesis by exact LP with
constraint generation.

**Tabula** ([TABULA.md](docs/TABULA.md)) — contracts as norms over facts (English and
Portuguese): antinomies with a scenario, pairs proved collision-free (DRUP), precedences (*lex
specialis*), silences, Hohfeld positions under a set of facts.

**Paths previously refused, closed**
- JBIG2 **Huffman** (tables B.1–B.15, user tables, SDHUFF dictionaries, SBHUFF text) and
  **halftone** (pattern dictionaries, Gray planes, grid, `HSKIP`), checked by an
  independent Python encoder and by jbig2dec: 21 new fixtures, 64 in total
  ([OCR.md §3f](docs/OCR.md)).
- **Permutation alignment** before merging (`merge align: true`, Git Re-Basin with exact
  Hungarian; [MERGING.md §8](docs/MERGING.md)).

**Interfaces**
- Console: **Opus** group with five desks (the Rebis seal, the cells and the phase portrait of
  Aludel, the Tabula with facts and scenarios, the Cupel bit strip, the Amalgam number line); desk
  state preserved when switching language; headless browser test. `GET /v1/vapor/opus`,
  `POST /v1/vapor/{rebis,aludel,tabula,cupel,amalgam}`.
- Terminal: `vapor rebis|aludel|tabula|cupel|amalgam`, with exit codes that a *script* can
  require (0 proved/equivalent/certified/consistent, 1 otherwise).
- MCP: `rebis_check`, `aludel_decide`, `tabula_analyze`, `cupel_drill`, `amalgam_sum` (25
  tools).

**Fixed along the way**
- The `dot16` oracle silently truncated contractions with `k` not a multiple of 16; it now refuses.
- The audit test failed on 0.14 itself (`System.cmd` in `--measure`):
  `Vapor.Main.Measure` with a process group (`setsid`), a deadline that kills the group and an output ceiling.
- A race between suites in the `/dev/shm` cleanup (the figures test failed on 0.14): a grace
  period and a write that touches the file.
- 15 public functions without a caller removed (found by a compile-time call
  tracer); two of documented API gained tests.
- The defect of jbig2dec 0.20 with `HDEFPIXEL = 1` recorded (the fixture is judged by T.88).
- Console strings: a translation key repeated across scripts overwrote the text of another
  desk; a test now forbids repeated keys with different text.

**Delivery**: `scripts/pack.py` packs the three archives (`-1-codigo`, `-2-qualidade`,
`-3-modelos`) from `git ls-files`, with sorted entries, fixed dates and normalized
permissions — the same tree gives the same bytes, checkable through `SHA256SUMS`.

**Refused**, with the reason in §18: the audio "isomorphic permutation" to stay below the
forensic plagiarism thresholds (it is evasion of copy detection), the "non-infringement certificate
in Lean", the claim that a Merkle root proves the *absence* of unauthorized data,
"solving" the string-theory landscape, the proof of the impossibility of dendrites.

## 0.14.0 — 2026-10-05

Scrutiny of the sixth request (translated: "from expository to a real tool, open and sanitized input, no predefined categories, human and model in the loop, AI research, everything from the terminal, free scenes"; and "alchemy names in English, no established names, focus on science, computing, mathematics, AI and finance, a more striking interface"): [docs/DIRECTIVE.md §17](docs/DIRECTIVE.md). The thesis: **one language, one furnace, one touchstone** — the famous searchers are cases of *propose, evaluate, certify*, so the product is the general case, and the cases become examples. Checks with a control in `mix vapor.quality` (§5j, 20; `--only round14`).

**Alembic** ([ALEMBIC.md](docs/ALEMBIC.md))
- A pure language for problems: arbitrary integers, lists, tuples, maps, comprehensions, lambdas, *pipes*, `let`, ~120 functions; errors with line, column and suggestion.
- Sanitized: fuel, recursion ≤ 5,000, integers ≤ 65,536 bits, lists ≤ 2 M, `sandbox/2` with a memory and time ceiling, identifiers never become atoms, `literal/1` reads only data.
- `Alembic.Tree`: the numerical subset as a JSON tree interpreted in the browser; `noise()` bit for bit equal in Elixir and JS.

**Athanor and Touchstone** ([ATHANOR.md](docs/ATHANOR.md))
- Ten spaces (bits, ints, reals, perm, subset, subsets, seq, graph, partition, program); minimize, maximize or refute a claim; `violation`, `holdout`, `describe`, `neighbor`, `measured`.
- Portfolio under discounted UCB: resumable exhaustive, random, annealing, evolution/MAP-Elites, CMA-ES, Bayesian (GP Matérn-5/2, EI, batches), mind, human.
- Random control with the same budget (rule of three), *holdout* with Spearman and the max-z of the noise, SHA-256 journal; `Touchstone.verify/3` with `full:` and `replay:`.
- Supervised sessions (propose, pin, ban, extend, measure); two-player games (negamax, UCT, learning by self-play, a match with Wilson).
- `Vapor.Mind`: `formalize` (up to three repairs with the compiler's error, and back-translation), `propose`, `ask`; Anthropic, OpenAI-compatible or a recorded script.

**Crucible** ([CRUCIBLE.md](docs/CRUCIBLE.md)) — ten open domains with evidence without an answer key: conservation laws proved over ℚ (with `ln`), quantum (Sturm, split-step), symplectic Hamiltonians, reactions, Wright–Fisher, phylogeny with *bootstrap*, RHF/STO-3G, HP folding, Boris, symbolic regression.

**Assay** ([ASSAY.md](docs/ASSAY.md)) — compare, leaderboard, calibration, agreement, judge, contamination, dedup, scaling, each one with its control.

**Interfaces**
- `bin/vapor` / `mix vapor` ([CLI.md](docs/CLI.md)): JSON in *pipes*, `NO_COLOR`, exit codes 0–4; `--measure` hooks up an external program as the objective.
- Console: an "Open workbench" group (Workspace, Crucible, Assay) as the opening; the live furnace and the touchstone; a new identity (soot, parchment, brass, verdigris, cinnabar), the alembic mark, new icons and manifest; the Science panel renamed Calibration.
- Scenes as documents edited by text operations (`Vapor.Scene.Ops`), with per-frame expressions and direction by a model; `vapor scene new|edit|direct|export`.
- MCP: `alembic_eval`, `athanor_run` (with `proposals`), `athanor_verify`, `game_query`, `crucible_run`, `assay_run`, `scene_ops` (20 tools); TUI with the same verbs.

**Fixed along the way**
- `Assay.Scaling` overflowed `exp` on data without structure (found by the shuffled-losses control).
- `Vapor.Expr.compile/2` created a module and an atom per new expression without limit; now there is a ceiling and the excess is interpreted.
- `mix vapor.serve` no longer requires `--model`/`--docs` (the workbench works without a model).
- `Finance.MonteCarlo` without the native process: the oracle did not carry the generator state (w1, w2, w3) between calls.
- `vapor assay` exits 1 when a check fails (noise, unstable leader, biased judge), like the other commands.

**Documents**: slides (42, with four new ones on the open workbench), thesis (new chapter 6, table of the 20 checks, abstract and conclusion), defence script, DIRECTIVE §17, README, TODO, SCENE §9, CONSOLE, INTERFACES, SCIENCE.

## 0.13.0 — 2026-10-05

Scrutiny of the fifth request (translated: "support for finance, HFT, attack the limitations and the TODO, slides, a thesis in LaTeX and a defence script"): [docs/DIRECTIVE.md §16](docs/DIRECTIVE.md). The thesis of the round: **the pain of finance is verifiability, not speed** — every market number comes out with the object that allows judging it. Checks with a control in `mix vapor.quality` (§5i, 23; `--only round13` runs only the round). Document: [FINANCE.md](docs/FINANCE.md).

**Finance** ([FINANCE.md](docs/FINANCE.md))
- `Vapor.Finance.Money`: exact decimal (integer + scale), seven rounding modes through a single primitive, allocation by largest remainder that sums exactly, the factor (1 + r)^(du/252) by integer root and truncated as ANBIMA does.
- `Vapor.Finance.Calendar`: ANBIMA/B3, NYSE (with special closings as data) and TARGET by rules and by the computus; **equal to QuantLib day by day from 1990 to 2078**; DU/252, ACT/360, ACT/365F, 30/360, 30E/360, ACT/ACT ISDA = QuantLib to 10⁻¹⁴; DI1 adjustments and expiries.
- `Vapor.Finance.Curve`: *bootstrap* of DI1, LTN, NTN-F, deposits, bonds (clean price) and par swaps; flat-forward or linear zero; repricing certificate and negative forwards pointed out; Nelson–Siegel–Svensson. Deposits = QuantLib's `PiecewiseLogLinearDiscount` to 10⁻¹³.
- `Vapor.Finance.Options`: BSM and Greeks (against finite differences), Black-76, Bachelier, implied volatility with the **no-arbitrage bounds checked first**, CRR and Leisen–Reimer (American = QuantLib to 10⁻¹⁰), Heston by Lewis (= QuantLib to 10⁻⁹), SVI with Durrleman's g(k) and calendar, model-free static arbitrage with the portfolio that exploits it.
- `Vapor.Finance.MonteCarlo`: GBM **compiled for the worker** with the generator (Wichmann–Hill, exact in binary32) and Φ⁻¹ (AS241) inside the program; European, Asian (Kemna–Vorst geometric as control variate), barrier; bits = oracle (64-lane program) and = 2 threads; 6–23× the BEAM; the control without the Itô term caught (z = 8.7). Longstaff–Schwartz on the BEAM.
- `Vapor.Finance.Risk`: historical, normal, Cornish–Fisher and EWMA VaR/ES; Kupiec, Christoffersen, Basel; the size of the test measured; Ledoit–Wolf, minimum variance with KKT, risk parity (Spinu), HRP.
- `Vapor.Finance.Backtest`: a causal signal language and **four noise gates** — prefix invariance (the no-look-ahead certificate, with the day), deflated Sharpe, PBO by CSCV, Reality Check with stationary bootstrap.
- `Vapor.Finance.Arbitrage`: the fundamental theorem as Farkas's lemma — the arbitrage portfolio **or** the state prices, in rationals; calls across strikes (complete decision for static portfolios), FX; `check/2` checks anyone's proposal.

**Trading desk**
- `Vapor.Finance.Book`: price–time; limit and market; GTC/IOC/FOK; post-only; modification with and without loss of priority; self-trade prevention; *kill switch*; **SHA-256 journal + Merkle root** and inclusion proofs.
- `Vapor.Finance.Book.Check`: the independent naive judge — rebuilds the chain, re-executes and demands the same reports, checks nine invariants; differential *fuzzing* with 6,000 events per policy.
- `Vapor.Finance.Itch` (ITCH 5.0: ten message types, BinaryFILE, the session feed, rebuilt book = the engine's book) and `Vapor.Finance.Fix` (FIX 4.4 with BodyLength and CheckSum = simplefix; D/F/G → events; reports → ExecutionReports).
- `Vapor.Finance.PreTrade`: quantity, notional, collar, worst-case position, message rate, *kill switch* (15c3-5 / RTS 6); refusals in the journal with the cause; `check/2` redoes everything from the journal.
- `Vapor.Finance.Micro`: Hawkes (Ogata, maximum likelihood, time-rescaling test; Poisson as the control), Avellaneda–Stoikov (§4 of the paper reproduced), Almgren–Chriss (closed form = numerical optimum to 10⁻¹⁶), Roll, Kyle, variance signature, *microprice*.
- `Vapor.Finance.Exchange`: market makers, Hawkes aggressors in continuous time and an informed trader through the same gate and the same engine; the session returns the three certificates and repeats to the same hash head.

**Open items closed** (from the [TODO](docs/TODO.md))
- `Vapor.Logic.LP`: exact rational simplex (two phases, Bland) with certificates of optimality (dual, zero gap), infeasibility (Farkas) and unboundedness (ray), checked by `LP.check/2`; in the logic desk (`maximize …`) and in `logic_check` with a proposal. = SciPy's HiGHS to 10⁻⁹.
- `Vapor.Archive`: **signed archives** (Ed25519 over the manifest, the operator's key from `mix vapor.audit keygen`; the identity does not change); `verify(zip, trusted: …)` refuses the unsigned one, the unknown key and the signature that does not check; the console signs with `VAPOR_ARCHIVE_KEY`; `mix vapor.archive sign|verify|replay`; the `finance.*` kinds are recomputable.
- `Vapor.Units`/`Vapor.Expr`: **affine scales** °C/°F as readings (98.6 °F = 37 °C; pV = nRT at 20 °C), refused inside compound units with the suggestion of `degC`.
- `Vapor.Engineering.Circuit`: **level-1 MOSFET** and **Ebers–Moll bipolar** (two branches; KCL certified without change), GMIN; **equal to ngspice 42** to the 6th digit (operating point and AC gain).

**Console**: a *Markets* group with *Finance* (eight tasks) and *Trading desk* (three), seal, gates, Basel traffic light, the book's ladder, the journal's chain; `test/js/console_markets.mjs` runs the 40 examples in EN and PT (87 checks) and **requires that no English label or phrase be left over in the Portuguese version** — a single translation pass after each visualization covers the labels and the server's phrases (verdicts, gates, refusals), without touching the code, the *hashes* or the FIX/ITCH bytes. Endpoint `POST /v1/vapor/finance`.

**MCP**: `finance_run` and `arbitrage_check` (thirteen tools). **CLI**: `mix vapor.finance KIND ARQUIVO`, `mix vapor.archive`, `mix vapor.quality --only roundNN`.

**Defence**: `slides/vapor.tex` updated (section 7 with eight slides on the round; sizes and lines of the core re-measured — 1.8 MB and 19,085 lines in the core, 8.7 MB with the thirteen rounds —, 82 theorems, label overlaps fixed); **thesis** in abnTeX2 (`monografia/`) and **script of the oral defence** (`monografia/DEFESA.md`).

**Measured**: 108 tests, 0 failures in the eight files touched (the full run was interrupted by a VM restart and not repeated); `mix vapor.quality --only round13` 23/23 in 23 s; console 87 checks in EN and PT.

**Fixes in this round** ([DIRECTIVE.md §16](docs/DIRECTIVE.md)): a market FOK became IOC (found by the invariant, in both engines); NYSE/TARGET closings and rules; Monte Carlo 95 s → 0.1 s (one step per call, the machine's ISA, parity on 64 lanes, generator in the worker); the exchange simulator distributed arrivals per step (Hawkes correctly failed) and quoted around its own mid; an internal error of the OTP 25 compiler worked around.

## 0.12.0 — 2026-10-05

Scrutiny of the fourth request (translated: "arbitrary problems and not just predefined categories", the four engineerings and the sciences, HPC, "similar or superior" with comparisons, human or AI in the loop, NPCs, photorealism, chess/shogi/Go/cards, a professional interface in alphabetical order): [docs/DIRECTIVE.md §15](docs/DIRECTIVE.md). The rule that runs through the round: **the input is the domain's text, and every answer carries a certificate computed outside the solver**. Checks with a control in `mix vapor.quality` (§5h, 27). Documents: [WORKBENCH.md](docs/WORKBENCH.md), [ENGINEERING.md](docs/ENGINEERING.md), [LOGIC.md](docs/LOGIC.md), [BOARDS.md](docs/BOARDS.md), [PROTEINS.md](docs/PROTEINS.md), [RENDER.md](docs/RENDER.md), [SCENE.md §6.1](docs/SCENE.md).

**Workbench** ([WORKBENCH.md](docs/WORKBENCH.md))
- `Vapor.Units`, `Vapor.Expr`: quantities with units (7 SI dimensions, named units, prefixes, `in [unidade]`), parser with the column of the error, symbolic derivative, simplification, LaTeX, **compilation to BEAM modules** (cache by sha-256); line-by-line worksheets.
- `Vapor.Solve`: ODEs by Dormand–Prince 5(4) with dense output, automatic switch to Rosenbrock (symbolic Jacobian) under stiffness, events by bisection, derived outputs, units checked before integrating; parabolic PDEs (Crank–Nicolson + AB2), hyperbolic ones (leapfrog, CFL refused) and 2-D Poisson (CG), with **verification by manufactured solution** (observed order); systems with all the roots in a box; fits (Levenberg–Marquardt, standard errors, AIC); optimization (augmented Lagrangian + **projected** BFGS: exact box bounds, divergence reported, KKT verdict).
- `Vapor.Solve.Ensemble`: uncertainty (`k ~ normal(…)`) compiled for the native worker — 4096 members × 1000 steps in 236 ms, 53× the BEAM, bit-for-bit parity with the oracle.
- `Vapor.Dense`: partial pivoting, Jacobi, Cholesky, generalized eigenvalues, CG, Householder, reverse Cuthill–McKee + banded Cholesky. `workbench_test.exs` (20; SciPy as the oracle).

**Engineering** ([ENGINEERING.md](docs/ENGINEERING.md))
- `Vapor.Engineering.Circuit` (MNA; .op/.dc/.ac/.tran; diode, controlled sources, ideal op-amp; Kirchhoff and power certificate; nodes without a DC path named), `Power` (polar Newton; Gauss–Seidel as control; Stagg & El-Abiad), `Structure` (frames and trusses, consistent mass, modes, mechanism refused), `FEM` (Q4 and QM6, patch test), `Pipes` (global gradient, Colebrook, loops), `Process` (kinetics with invariants by rational null space, CSTR, Rachford–Rice, McCabe–Thiele/Fenske/Underwood/Gilliland). `engineering_test.exs` (21).

**Logic** ([LOGIC.md](docs/LOGIC.md))
- `Vapor.Logic.SAT` (CDCL) and `Vapor.Logic.DRUP` (independent checker), Schur/van der Waerden/Ramsey/pigeonhole/queens problems with witness and refutation, formulas by Tseitin, `Rewrite` (Knuth–Bendix with LPO), `Groebner` (Buchberger, Rabinowitsch). **`Vapor.Logic.check/2`**: anyone's proposal (model, DRUP proof, colouring, counterexample) checked, never trusted. `logic_test.exs` (8).

**Boards and cards** ([BOARDS.md](docs/BOARDS.md))
- `Vapor.Play`: generic MCTS (UCT/PUCT, noise at the root), exact negamax with a table; `Chess` (published perft, python-chess, engine, **checked mate proofs**), `Shogi` (drops, nifu, uchifuzume; perft 30/900/25,470; python-shogi), `Go` (Tromp–Taylor, superko; 1/57/12,675 legal positions), `MNK`, `Poker` (CFR+, exact exploitability in Kuhn and Leduc), `SelfPlay` (generic self-play judged against perfect play). `boards_test.exs` (16).

**Proteins** ([PROTEINS.md](docs/PROTEINS.md))
- `Vapor.Bio.Structure` (PDB, Horn, TM-score = TM-align, GDT, lDDT, secondary structure, **folding by distance geometry** with chirality from the helices), `Coevolution` (Potts, MI/APC, DCA), `Align` (Gotoh, BLOSUM62 = Biopython). Pipeline on 1A8O: TM 0.69 (control 0.20). `proteins_test.exs` (7).

**Render** ([RENDER.md](docs/RENDER.md))
- `Vapor.Render`: path tracing (diffuse, metal, glass, emissive, sky, sun by next-event estimation, Russian roulette), deterministic, parallel; white furnace, gradient furnace (the biased estimator caught), N^−½. `priv/console/gpu_tracer.js`: the same on the GPU (WebGL2), progressive, checked in Chromium against the reference. `render_test.exs` (6).

**Living scene** ([SCENE.md §6.1](docs/SCENE.md))
- Inhabitants with id, name, colour, behaviour (wander, stay, patrol, follow, flee, go), actions (wave, dance, sit, jump, run), speech in balloons, routes by click, timeline (`at`), **exact-frame GIF**; direction by clauses with time, pronouns and roles (PT/EN).

**Console** ([CONSOLE.md](docs/CONSOLE.md))
- Navigation regrouped (new *Solve*) and **in alphabetical order in the language shown**; **command palette** (Ctrl/⌘ K); six new panels — Workbench, Engineering, Logic, Boards and cards, Proteins, Render — with their own diagrams; inhabitants inspector; the scripts embedded in the page (a single document). `console_desks_test.exs` + `test/js/console_desks.mjs`: every example of every panel through Chromium.

**MCP**: `workbench_solve`, `engineering_run`, `logic_check` (with a proposal), `board_query`, `render_scene` (eleven tools).

**Names**: identifiers, panels and API without proprietary names (`games.selfplay`, task `selfplay`; the 0.11 names are still accepted for saved archives); the comparisons stay in the documents.

**Measured**: `mix test` 702 tests, 0 failures (115 excluded for a missing tool); `mix vapor.quality` 147/147 (native).

**Fixes in this round** ([DIRECTIVE.md §15](docs/DIRECTIVE.md)): unbounded optimization returned −6·10⁶² as the optimum (now: projected bounds, divergence reported); the scene reader accepted non-numeric values; direction in English lost names, pronouns and times; a CSS collision shrank chess pieces; a page with external scripts.

## 0.11.0 — 2026-10-04

Scrutiny of the third request of the day (translated: AlphaProof/AlphaDev/AlphaFold/AlphaZero "and beyond", physics on several fronts, bringing a picture to life, saving and exporting everything): [docs/DIRECTIVE.md §14](docs/DIRECTIVE.md). The rule that runs through the round: **a search proposes, a verifier decides**. Every delivery has a check with a control in `mix vapor.quality` (§5g, 26 checks). Documents: [SCENE.md](docs/SCENE.md), [DESCOBERTA.md](docs/DESCOBERTA.md), [MATHEMATICS.md](docs/MATHEMATICS.md), [SCIENCE.md](docs/SCIENCE.md), [JOGOS.md](docs/JOGOS.md).

**Bringing a picture to life** ([docs/SCENE.md](docs/SCENE.md))
- `Vapor.Scene`: SLIC + region graph, grown sky, depth from the ground plane (heuristic, editable), layers with the background reconstructed by push-pull, walkable ground, light; skeleton of drawings (Zhang–Suen, graph by *crossing number*, loops), mesh with skinning; PT/EN direction → operations, unknown words reported; standalone HTML page. `scene_test.exs`.
- The `SceneEngine` engine (in the console and in the exported HTML): layers as planes in perspective, ground in bands, inhabitants by A* with the head on the horizon, rain/snow/fog/storm, torches, candles, embers, smoke, fireflies, birds, butterflies, leaves, wind in the vegetation, time of day and cycle, animated drawings (wave, walk, dance, breathe), entropy, seed and fixed step (the loop is reproducible), video recording.
- `Vapor.Sketch`: sketch → technical drawing with constraints (SVG, DXF R12); floor plan → rooms, doors and 3D model (GLB). `sketch_test.exs`.

**Discovering** ([DESCOBERTA.md](docs/DESCOBERTA.md), [MATHEMATICS.md](docs/MATHEMATICS.md))
- `Vapor.Discover`: optimal sorting networks for n ≤ 8 (0-1 principle), 2×2 multiplication with 7 products (exact over the integers; rank 6 never), minimal bit tricks by exhaustion (⌊(x+y)/2⌋ = `(x&y)+((x^y)>>1)`), complexity class from counts. `discover_test.exs`.
- `Vapor.Prove`: geometry by the algebraic method (numerator ≡ 0 + non-degeneracy + a check in exact rationals; false ones refuted), conjectures found and proved (Euler line, nine-point circle), homology over GF(2) and ℚ (Klein and RP² torsion), persistent homology. `prove_test.exs`.

**Science and games** ([SCIENCE.md](docs/SCIENCE.md), [JOGOS.md](docs/JOGOS.md))
- `Vapor.Science`: Schrödinger by split-step (coherent state, exact tunnelling), relativistic Boris (γ, E×B), Grad–Shafranov against Solov'ev, Hartree–Fock STO-3G (H₂ −1.1167; HeH⁺ −2.860662; the failure of RHF shown), Lennard-Jones, Wright–Fisher against the exact chain, phylogeny (RF 0), HP folding down to −9. `science_test.exs`.
- `Vapor.Games`: tic-tac-toe AlphaZero (policy + value + PUCT + self-play; `priv/games/tictactoe.json`) — against **all** the optimal lines of perfect play, none lost with 128 simulations and 13 % with 8 (the untrained search: 97 %); domain randomization on the cart-pole (484 against 275 steps in unseen worlds). `games_test.exs`.

**Save and export, for everything**
- `Vapor.Archive`: zip with a manifest (recipe, hashes), identity by the hash of the manifest, byte-for-byte check, **recomputation** of the deterministic kinds; an archive names a kind, never a function. `archive_test.exs`. Every new console panel has *Save*; *Trust → Archives* checks and recomputes.

**Console**: *Living scene*, *Sketch*, *Science*, *Games*, *Mathematics*, *Algorithms*, *Archives* panels (English and Portuguese, light and dark). Endpoints `/v1/vapor/{scene/*,sketch,prove,discover,science,games,games/move,archive,archive/check}`.

**Fixes from the independent review of 0.11** ([DIRECTIVE.md §14](docs/DIRECTIVE.md))
- `Vapor.Archive`: decompression counted as it happens (at most 512 entries, 256 MB; a 1 MB bomb that expands to 300 MB is refused without being expanded), a manifest that is malformed, lacks `result.json` or has duplicate entries is refused (before, it crashed); every recipe parameter bounded before running (`n` 2–10, `beam` 1–256, self-play ≤ 400 games and 64 simulations, theorems, complexes and experiments by name only); the beam search that dies is reported, not repeated forever.
- `Vapor.Prove`: a construction degenerate for every value is `{:degenerate, :construction}`, never "proved" by 0/0; the surviving conjectures are proved symbolically (before, only checked in rationals); the "independent check" is called what it is.
- `Vapor.Games.versus_every_optimal_line/2`: all the optimal lines of perfect play, from both sides. It found 4 lines lost out of 131 with 64 simulations that the 60-game sample did not see; the claim, the test, the console and §5g moved to this measurement.
- The lab does not crash on an image that is read but not decoded (more than 4 megapixels, GIF); the game board is validated.
- Documents: weak controls called such (SCIENCE §1), the tokamak's h² rate attributed to the axis locator, the n^2.807 class as a check of the classifier, the corner cart-pole, the cart mass ×0.5–2, the §5g of sorting networks requires every comparator to be necessary.

**Fixes from the independent review of 0.10**
- The substrate airlock: a difference without a bound and without a measured cause is refused; an answer outside the protocol or missing an output is refused (before, it crashed); probes not run appear as `:unmeasured`; `judge --sign` uses the operator's key.
- `/v1/vapor/ocr` no longer gives a 500 on a generic `{:error, _}`; the cluster drops from the cache the answers of a node put in quarantine; the console labs serialize the first run (one session per worker) and close the twin's session.
- Documents: z ≈ 400 (not 300) for Watts–Strogatz; the kit command; references to non-existent tests; the twin's log does not protect measurements without an external anchor (stated); the absence of seccomp on FreeBSD is not visible in the binary (stated); the Witten–Bell baseline of §5f now on the same bytes; CJK requires that the language model help in Japanese and Korean.

## 0.10.0 — 2026-10-04

Scrutiny of the two requests of this round (substrates, training, OCR of other scripts; and, in between, endless context, physics, networks, Cyrillic, LaTeX, FreeBSD): [docs/DIRECTIVE.md §13](docs/DIRECTIVE.md). Every delivery has a check with a control in `mix vapor.quality` (§5f). Documents of the round: [SUBSTRATES.md](docs/SUBSTRATES.md), [TRAINING.md](docs/TRAINING.md), [PHYSICS.md](docs/PHYSICS.md), [REDES.md](docs/REDES.md), [OCR.md §3g–§3k](docs/OCR.md).

**Substrates** ([docs/SUBSTRATES.md](docs/SUBSTRATES.md))
- **Substrate airlock** (`Vapor.Substrate`, `mix vapor.substrate list|kit|judge`): probes with known answers (FMA, FTZ, DAZ, signed zero, NaN, reduction order, real mantissa bits, division, functions, the real kernels) → verdict `:canonical | :envelope | :refused` with the **numerical fingerprint**; signed CBOR record (Ed25519); the dispatcher only sends canonical programs to canonical substrates; admission on arrival. `substrate_test.exs`.
- **Envelope with DAZ**: found by the kit on CPU XLA (subnormal inputs read as zero); covered, and still tight.
- **Metal**: MSL translator (`Vapor.Emit.MSL`, structured control flow), `vapor-metal` daemon (Objective-C runtime opened at run time, `MTLMathModeSafe`, unified memory), CPU worker for macOS (`MAP_JIT`, `__ulock`); `make metal` from any host. The same MSL executed by a *shim* with clang (`vapor-metal-sim`): = oracle bit for bit on canonical programs, SSM, GEMM, attention, a Llama session and the engine; a contracting *shim* and an FTZ *shim* admitted within the envelope. The real daemon is compiled, not executed. `metal_test.exs`.
- **Tenstorrent and any PJRT**: **StableHLO** export (`Vapor.Export.StableHLO`, `mix vapor.export`) — Llama/Qwen2/Mistral on XLA to ~10⁻⁶ —, and a **portable admission kit** (StableHLO + `run_kit.py --platform tt|tpu|cpu`) judged and signed back. `stablehlo_test.exs`.
- **Cluster** (`Vapor.Cluster`): content-addressed cache, auditing by redundant execution with a keyed *hash* sample, quarantine and readmission by measurement, *failover* and *hedging* without changing a bit, training across nodes with the bits of one machine. Real `:peer` nodes. `cluster_test.exs`.
- **FreeBSD**: the worker compiles for x86-64 and AArch64 (`make freebsd`), libc, `_umtx_op`, W^X by `mprotect`, **Capsicum** isolation (a directory descriptor with read and map rights; `openat`). Not executed. `freebsd_test.exs`.

**Training and context** ([docs/TRAINING.md](docs/TRAINING.md))
- **Pre-training** (`Vapor.Train.LM`, `mix vapor.train`): byte-level Llama, gradients by `Autodiff.grad_lets/5` (= PyTorch to 7.7·10⁻⁷), deterministic data parallelism (a fixed tree over blocks accumulated in the resident session: bits independent of the number of workers and of a worker dying), AdamW with *clipping*, exact checkpoint/resume, Hugging Face export. **`priv/lm`**: 492,160 parameters, 2.919 bits/byte on held-out text (Witten–Bell order 5: 3.415), with a receipt. `train_lm_test.exs`.
- **Endless context** (`Vapor.Streaming`, `kv: {:stream, âncoras, janela}`): unrotated keys in a ring with fixed anchors, rotated at every step in the cache's frame of reference — constant memory, no distance outside training, the bits of the causal model until the cache fills. 3.03 bits/byte 14× beyond the training length; growing positions: 5.81. `streaming_test.exs`.
- **Autodiff**: rules for `max`, `min`, `relu`, `sel` (piecewise), `tanh`, `fma`.

**Physics** ([docs/PHYSICS.md](docs/PHYSICS.md))
- `Vapor.Physics`: XPBD with substeps, as a vapor program — batched, **bit for bit on every substrate**, differentiable. First-order pendulum against the exact elliptic period; chaos (double pendulum) oracle = native while one ulp separates the worlds; `sysid/3` recovers rod and damping from noisy measurements by the gradient of the trajectory (shuffled in time: nothing); cart-pole by random search 200/200; **digital twin** with CUSUM and a log in a *hash* chain that is rebuilt from the model and the actions. `physics_test.exs`.

**Networks** ([docs/REDES.md](docs/REDES.md))
- `Vapor.Graph`: reproducible generators (ER, BA, WS, planted), Clauset–Shalizi–Newman power law with *bootstrap* and likelihood ratio, configuration null and z-scores, deterministic Louvain, SIR and mean-field threshold, percolation and robustness, PageRank on the host and **as a vapor program**; = networkx. `graph_test.exs`.

**Vision** ([docs/OCR.md §3g–§3k](docs/OCR.md))
- **CJK** (`Vapor.Vision.CJK`, `priv/ocr-cjk-{zh,ja,ko}`): directional element features, segmentation decided by recognition, a language model that abstains. CER on never-seen fonts (new seed): zh 8.9 %, ja 2.6 %, ko 11.3 %; random characters: the language model changes nothing.
- **Arabic** (`OCR.default(:arabic)`, `priv/ocr-arabic`): reading in visual order, returned to logical order (`Vapor.Vision.Bidi` = python-bidi on 600/600); 19.2 % CER on never-seen fonts (Latin reader: 91 %); RTL columns; the line finder no longer splits Arabic lines at the letters' dots.
- **Cyrillic** (`OCR.default(:cyrillic)`, `priv/ocr-cyrillic`): 2.8 % CER on four never-seen fonts, with half of the vocabulary unseen by training (Latin reader: 98 %). `scripts_test.exs`.
- **Captions**: the four nearest lines on both sides of the figure (before: two below), lettered numbering ("Figure A:"); found by the suite on the test set, stated in [OCR.md §3h](docs/OCR.md).
- **Figures** (`Vapor.Vision.Figure`, and in `OCR.read`): detection with caption; **chart digitization** that refuses when the labels do not confirm a scale (permuted labels: 12/12 refused; default style: 27/30 within tolerance, none grossly wrong). `figure_test.exs`.
- **Formulas → LaTeX** (`Vapor.Vision.Math`, `priv/math`): 4.9 % error per *token* on Computer Modern and STIX, never seen; the flat reading errs more than 3× as much. `math_test.exs`.
- **Handwriting**: measured on handwritten fonts and **not shipped** (Latin cursive: 64 % CER on never-seen hands; Nastaliq 57 %; Cyrillic 78 %; Japanese in brush fonts, 6.4 % with the language model — still fonts, not hands); `OCR.default(:cursive)` refuses with the measurement and the path ([OCR.md §3k](docs/OCR.md)).

**Console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- New panels: **Substrates** (numerical fingerprints and verdicts), **Training** (curve, baselines, the stream beyond the training length), **Physics** (the chaos run twice, animated; the twin with CUSUM and log), **Networks** (force layout coloured by community, power law, robustness). *Vision* with the choice of script (Latin, Arabic, Cyrillic, cursive, 中文, 日本語, 한국어, formula), RTL lines, figures with the chart's data (lines, points, bars in the series' colours) and LaTeX; the footer says which reader read it; cursive shows the refusal and the path. Endpoints `/v1/vapor/{substrates,physics,graph,lm}`; `/v1/vapor/ocr` with `script`.

**Minor**
- `Vapor.Runtime.Substrates` recognizes FreeBSD's ISA names (`amd64`, `arm64`).
- `mix.exs` 0.10.0.
- The test of the whole quality suite has a 60 min limit (round 0.10 adds ~11 min to the ~12 from before).
- The comment of the MSL translator no longer names a compiler (the source-code audit forbids the name in the product; it is the tests that compile the *shim*).

## 0.9.0 — 2026-10-04

Scrutiny of the two requests of this round (translated: "an any-to-any ComfyUI … and everything from the Hugging Face courses"; and, before that, watermark removal and refusal removal, adversarial distillation — the record of the decision): [docs/DIRECTIVE.md §12](docs/DIRECTIVE.md). Every delivery has a check with a control in `mix vapor.quality` (§5e). The document of the round: [docs/STUDIO.md](docs/STUDIO.md).

**The studio** (`Vapor.Studio`, [docs/STUDIO.md](docs/STUDIO.md))
- A graph of typed nodes (`image`, `mask`, `audio`, `video`, `mesh`, `text`, `number`, `tensor`, `latent`, `json`). **Exact content-addressed cache**: the key of a node is the SHA-256 of the type, version, parameters and the keys of whatever feeds it (a Merkle DAG). A receipt per output, **Merkle root per run**, `verify/3` re-executes without cache. Ill-typed graphs refused before running, each problem with the node and the repair; subgraphs (`studio.input`/`studio.output`, `video.map` per frame). `studio_test.exs`.
- **64 nodes**: image (resizing = `torch` and Pillow to 2.2·10⁻⁶; native program = the BEAM's sparse evaluator, bit for bit), masks and compositing, sound (resampling with 69.6 dB SNR, spectrogram), video (camera, per-frame map, concatenation, *crossfade*), 3D, RL, vision, diffusion; `image.scene`, deterministic scenes to start without a file. Our own codecs: **GIF** (LZW = Go/Pillow), **Y4M**, **MJPEG-AVI** reading (= libjpeg frame by frame). `studio_media_test.exs`.
- **ComfyUI import** (API format): `LoadImage`, `SaveImage`/`PreviewImage`, `EmptyImage`, `ImageScale(By)`, `ImageInvert`, `ImageCrop`, `ImageBlur` (σ in units of the radius, converted), `ImageCompositeMasked`, `CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`, `VAEDecode`, `VAEEncode`. Every translation is declared, and a node without a translation refuses the whole workflow. Written from the documented semantics of the nodes, without reproducing code.
- `Vapor.Studio.Templates`: six starter graphs (image → animated plane, sound and spectrogram, 3D relief, trained policy × control, editing with a mask, text → image).

**Stable Diffusion** (`Vapor.Diffusion`, [docs/STUDIO.md §2](docs/STUDIO.md))
- **U-Net** (`Vapor.Lock.Adapters.UNet`, `UNet2DConditionModel`): blocks with cross-attention, heads padded to 16 with the original 1/√dₕ scale, GEGLU, exact channel concatenation, correctly rounded *timestep* *features*. = diffusers: SD 1.x 1.17·10⁻⁶, SD 2.x 1.03·10⁻⁶.
- **VAE encoder** (the `encoder: :map` part of the adapter, with diffusers' asymmetric *pads*): 1.14·10⁻⁶. `Vapor.Spatial.conv2d` gains `pads: {topo, base, esq, dir}`.
- **Schedulers** (`Vapor.Diffusion.Scheduler`): DDIM, Euler, DPM-Solver++ 2M, with leading/linspace/trailing spacing; diffusers' *timesteps* and final latents to ≤ 6·10⁻⁷ in the 18 combinations. **Finding**: each diffusers *scheduler* spaces the steps its own way, and a single implementation is off by up to 0.19.
- **Pipeline** (`Vapor.Diffusion.Pipeline`): text → image, image → image and inpainting over a diffusers directory, with a CLIP tokenizer from `tokenizer.json` **or** from the slow files (vocab.json + merges.txt, = `transformers`). = `StableDiffusion{,Img2Img,Inpaint}Pipeline` to ~10⁻⁶ on an included tiny checkpoint (`priv/quality/sd_tiny`); control (another sampler): 0.096. The VAE latent is the mean (diffusers samples, hidden). `diffusion.*` nodes in the studio; the ComfyUI txt2img workflow gives the same bits as the pipeline. `diffusion_pipeline_test.exs`.

**Consistent upscaling** (`Vapor.Vision.Upscale`, `mix vapor.upscale`)
- ×2/×4 with **D(y) = x by construction**: downscaling the result gives back the input (10⁻¹⁶; Lanczos 0.05–0.17). A *patch* MLP over Lanczos and an exact projection with redistribution at saturation; luma by the network and chroma by Lanczos. Trained here, reproducible bit for bit, with a receipt. On held-out images, against Lanczos with the same projection: **+1.5 to +5.8 dB on text and line charts**, a tie on photographs, −1.4 dB on a smooth gradient (both > 53 dB). The CER of downstream OCR halves. `upscale_test.exs`.

**Reinforcement learning and 3D** (`Vapor.RL`, `Vapor.Geom`, `Vapor.Learn`)
- `Vapor.Learn`: MLPs trained as a recurrent program (AdamW, cosine *schedule* through the CR functions), oracle = native bit for bit.
- CartPole (= gymnasium to 10⁻¹²), FrozenLake (= gymnasium's table), two-joint arm. Value iteration, Q-learning (74.7 % = the optimum; always-left 0 %), **REINFORCE** as a program via `Vapor.Autodiff` (472.9/500; untrained 18), behaviour cloning (77 %; random 1 %). Replay = seed + actions. `mix vapor.rl`. `rl_test.exs`.
- Meshes from SDF (marching tetrahedra, closed, volume to 0.4 %), relief from images, OBJ/PLY/GLB (read by trimesh), deterministic rasterizer, spinning video. `geom_test.exs`.

**Agents** (`Vapor.MCP.Server`, `mix vapor.mcp`)
- **MCP server** (stdio): `studio_catalogue`, `studio_validate`, `studio_run` (the cache lives between calls: the agent that edits a node recomputes only what depends on it), `studio_verify`, `comfy_import`, `context_search` (BM25 with a Merkle proof per passage). Failures become `isError` results with the repair; an exception in a node does not bring the server down; paths outside the directory are refused. Tested with the **official client of the MCP Python SDK**. `mcp_server_test.exs`.

**Console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- *Studio* (new): a node canvas with a palette by category and search, wires by dragging **or by keyboard** (the sources of each input in the inspector), drag/zoom/frame, previews (image, GIF, sound, rendered mesh), the water level of each node (empty, cached, computed), the **seal** with the Merkle root and *Verify*, ComfyUI import, JSON export. Endpoints `/v1/vapor/studio/{nodes,run,verify,comfy}`; cache and previews live as long as the server lives (an entirely cached run: ~1 s). `console_test.exs`.

**Minor**
- The image digest in `Studio.Value` now uses the external term format (the same binary64 bits, tagged), off the *heap*: a 160-step episode went from 36 s to 2.8 s.
- The studio cache keeps the digests together with the values: a cached node is not hashed again.
- `video.camera` computes the frames in parallel, with the same bits.
- `mix.exs` 0.9.0.

## 0.8.0 — 2026-10-03

Scrutiny of the request and the attachment of this round (a "Vapor 1.0" roadmap of seventeen items: what went in, what was refused and why): [docs/DIRECTIVE.md §11](docs/DIRECTIVE.md). Every delivery has a check with a control in `mix vapor.quality` (§5d). Measurements: [docs/bench/ROUND08.md](docs/bench/ROUND08.md).

**GPU and sparsity** ([docs/FRONTIER.md §1, §7](docs/FRONTIER.md))
- **Resident sessions in Vulkan** (`OPEN/STEP/CLOSE` in `vapor-fabric`): *pipelines* once, *buffers* in direct memory or by *staging* (discrete GPU; chosen automatically if direct memory runs short), state fed back inside the GPU, recorded *command buffers* reused keyed by the exact bytes of the step (31/32). 72.4 → **12.4 ms/token** (10.5 by *staging*) and 1 MB → **8 kB per token** on a reduced Llama; the same bits as the CPU. **`Vapor.Engine` serves on the GPU** (`mix vapor.serve --gpu`, `mix vapor.generate --gpu`); driver crash → `:session_lost` and reopening, never the BEAM; hostile frames refused. `gpu_session_test.exs`.
- **Sparse 4-bit MoE** (`Term.qgemv_masked/3`, kernel `gemv_sb4_masked` on x86/AVX-512/NEON/RVV/SPIR-V): rows not chosen read neither nibbles nor scales; bits = dense; 1.6× (T = 1) and 2.6× (T = 8), 15.0 M → 5.3 M instructions retired. NaN scales in an expert not chosen do not change a bit. `sparse_sb4_test.exs`.

**Documents and vision** ([docs/OCR.md §3e–§3f](docs/OCR.md))
- **Tables** (`Vapor.Vision.Table`): ruled grids (virtual borders, cells merged by rule coverage) and tables with ruling lines only (columns by the gutters); cells read with typed columns (numeric ones by the constrained reading, space convention, **shapes** decoded by CTC Viterbi over the shape's automaton — `Vapor.Vision.Template`), thresholds chosen on a separate validation. Structure F1 **1.000** (0.7: 0.343), merges 1.000 (without detection: 0.905), per-cell CER **5.5 %** (free: 11.7 %; Tesseract with perfect boxes: 2.3 %). Markdown, HTML, CSV; in the OCR text in reading order; one passage per table in the library; `mix vapor.ocr --table-format`, `mix vapor.ocr tables DIR`. `table_test.exs`.
- **Arithmetic JBIG2** (`Vapor.Docs.JBIG2`): MQ, generic (templates 0–3, AT, TPGDON), MMR, refinement, symbol dictionaries (with aggregation), text regions (8 corners, strips, refinement), pages, `/JBIG2Decode` with `/JBIG2Globals`. **43 streams = jbig2dec bit for bit** (from jbig2enc and from our own MQ encoder checked against H.2 of the standard); control with the wrong declared template: 19/43. Huffman and halftone: a warning by name. `jbig2_test.exs`.

**Models** ([docs/FRONTIER.md §4](docs/FRONTIER.md))
- **Mamba-2** (`Vapor.Lock.Adapters.Mamba2`, `Mamba2ForCausalLM`): scalar A per head, B/C per group, Δ with `time_step_limit`, gated RMSNorm; exact head → channel expansion (*one-hot* product, `gather_row`). = `transformers` (7.8·10⁻⁷, identical greedy output), native = oracle, GPU = CPU. **Finding**: `transformers` normalizes the whole width where `mamba_ssm` normalizes per group — the training one is the default, `gated_norm: :whole` for the other, each fails against the other's reference; and its cached step skips `time_step_limit`. `mamba2_hf_test.exs`.

**Distribution** ([docs/FRONTIER.md §6](docs/FRONTIER.md))
- **Tensor parallelism across BEAM nodes** (`Vapor.Shard.Cluster`, `Vapor.Shard.Host`): resident shards per node with SHA-256 checked on arrival; the bits of one worker on 1–3 nodes; **lost node → shards relocated, same bits**; `replicas: 2` compares replicas bit for bit and catches a node that corrupts a bit. Tested with real `:peer` nodes. `shard_cluster_test.exs`.

**Conformity** ([docs/AUDIT.md](docs/AUDIT.md))
- **Audit dossiers** (`Vapor.Audit`, `mix vapor.audit keygen|export|verify|demo`): certificates, attested journals, log receipts, quality report, model contracts and documents in a canonical CBOR file with a Merkle root, Ed25519 signatures with quorum and an anchor in the transparency log; each item checked by its own rules; mapping to the AI Act (Art. 11, 12, 13, 15, 19; Annex IV) and ISO/IEC 42001 (A.6.2.3/4/7/8) as data. **PDF report** with the dossier attached (`verify` accepts the PDF) and an **HTML page that checks itself offline** in the browser (WebCrypto). One altered byte: refused, saying where. It is not a conformity assessment (the notice goes inside). `audit_dossier_test.exs`.

**Verification** (`proofs/Vapor/Binary32.lean`)
- **The rewrite rules proved** in an IEEE-754 model from the bit patterns (Lean kernel): for finite `x`, the only correctly rounded result of `x·1`, `1·x`, `x+(−0)`, `(−0)+x`, `x−(+0)` is `x`; `x+(+0) → x` refuted (`(−0)+(+0) = +0`); `neg(neg x) = x`. `units` extracted and checked against `Vapor.F32` at every exponent. **Correction**: the `Rewrite` doc had said "exact including NaN" since 0.5 — false (x86 quiets sNaN; RISC-V returns the canonical NaN); corrected, with a test. `rewrite_soundness_test.exs`.

**Console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- *Vision*: tables framed on the page and drawn from the cells (merged, header, numbers on the right, confidence per cell, cell ↔ box link), with Markdown/CSV/HTML one click away; the table's block marked in the reading list.
- *Dossier* (new): check a `.vdossier` or its PDF, or assemble the demonstration one; verdict at the water level; the evidence × provision **weave**, with the gaps in view; download the dossier and the page that verifies itself.
- The identity shows the substrate (CPU or `GPU · dispositivo`).

**Minor**
- `Vapor.JSON.decode(…, nonfinite: true)` for the `Infinity`/`NaN` that Python writes (used only for `config.json`); the default remains strict.
- The suite's cluster check no longer calls `epmd` through a *shell* (the source audit test caught it); without epmd, it does not run and the report says so.
- `Session.info/1` states the device; `/v1/vapor/info` states the substrate.

## 0.7.0 — 2026-10-03

Scrutiny of the request of this round (the same as 0.5's, for the third time): [docs/DIRECTIVE.md §10](docs/DIRECTIVE.md). Theme: **the office scanned document**, structured output with invalid fields, and the merge that does not fit in memory. Every delivery has a check with a control in `mix vapor.quality` (§5c).

**Office scans** ([docs/OCR.md §3b–§3d](docs/OCR.md))
- **CCITT Group 3 (1-D and 2-D) and Group 4** (`Vapor.Docs.CCITT`), with no dependency: 42 streams equal **bit for bit** to libtiff's encoder (noise, text, runs up to 2,600, *fill bits*, aligned MH); truncated data or garbage never hang. **LZW** (`EarlyChange` 0 and 1) and **RunLength** in PDF, checked against libtiff and an independent encoder; TIFF predictor 2. `ccitt_test.exs`.
- **Reading order** (`Segment.blocks/2`): recursive XY-cut in which a column gutter cuts before a horizontal gap, with thresholds from the region itself. Eight scanned pages (1–3 columns, title and footer spanning, never-seen fonts, CCITT in PDF): **CER 1.3 %**; without the order, 53 %; Tesseract 1.6 %. `reading_test.exs`, `priv/quality/scans`, `test/python/scan_pages.py`.
- **CTC beam + character language model** (`Vapor.Vision.CharLM`, Witten–Bell, counted from the training corpus *as printed*): held-out lines CER 6.5 % → **4.4 %**, lines with values and codes 8.9 % → 7.4 %; the model **abstains** on lines without language (0 of 30 random strings changed; without the guard, 27), only chooses among what the frames find plausible, and a model from a shuffled corpus gains nothing (control). Weights chosen on a separate validation (`priv/ocr/lm.json`).
- **Real bugs**: stencil masks (`ImageMask`) were read inverted; accents of lines without tall letters were discarded by the line finder ("não" read as "rão") — found by looking at the new interface; a page's title was cut in half by the first version of the XY-cut.

**Structured output** ([docs/AGENTS.md](docs/AGENTS.md))
- JSON Schema **`pattern`** (ECMA-262: classes, escapes, groups, quantifiers, anchors at the ends; what is not regular over bytes is refused by name) and **`format`** (`date`, `time`, `date-time` with the real calendar, `uuid`, `ipv4`; `email` and `hostname` as subsets that never emit an invalid value) become UTF-8 byte automata with JSON's escapes (`Vapor.Grammar.Regex`). The same verdict as Python's `re` and the standard library's *parsers* on 7,240 strings. `regex_test.exs`.

**Merging** ([docs/MERGING.md](docs/MERGING.md))
- **Disk to disk** (`Merge.stream/3`, `mix vapor.merge --stream`): one tensor at a time, the same kernels in the same order — files equal **byte for byte** to the in-memory merge, the same roots in the receipt; memory peak 4 MB against 83 MB (the suite caught a peak of 0.9× when called from a process with a large *heap* — garbage not collected between tensors; now 0.08×). `Safetensors.catalog/1`, `read_entry/3`, `header/2` (writing one tensor at a time). `merge_stream_test.exs`.

**Console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- *Vision* panel: the scanned page appears even when it comes from a PDF; **blocks numbered in reading order** and the **reading thread**; text grouped by block; **the characters chosen by the language model** highlighted, with the reading from the frames alone one click away — including where the model errs. English and Portuguese, light and dark.

## 0.6.0 — 2026-10-03

Scrutiny of the two attachments of this round (fifteen items for 1.0; what would be missing for Sora, Midjourney and world models): [docs/DIRECTIVE.md §9](docs/DIRECTIVE.md). Measurements: [docs/bench/FRONTIER.md](docs/bench/FRONTIER.md). Every new feature has a check with a control in `mix vapor.quality` (§5b, 48/48).

**Serving frontier models** ([docs/FRONTIER.md](docs/FRONTIER.md))
- **Sparse MoE by row predication** (`Term.linear_masked/3`): the GEMV skips the rows that did not choose the expert and reads only the chosen weights; bits = dense on AVX2/AVX-512/RVV/Vulkan/QEMU; 2.6× in the decode of a reduced Mixtral. The attachment's token permutation was refused with a reason (the cost on CPU is reading weights). `sparse_experts_test.exs`.
- **MLA latent cache** (`mla: :latent`, the default for MLA models): a `[c | k_rope]` cache shared by the heads, query and value absorbed (`Term.linear_grouped/3`, block-diagonal); 85× less memory at DeepSeek-V3's shapes; checked against `transformers`. `latent_attention_test.exs`.
- **Sliding window executed exactly** (it used to be refused when it switched on): only the band changes, in the canonical order; Mistral and Gemma 3 against `transformers` (control: without the window it fails). And the **circular cache**: when the window is on in every layer, each sequence keeps `⌈(w + T − 1)/página⌉` pages in a ring of the block table (7.9× more sequences at 32 k / window 4 k), bits = full cache; the bound is tight (one fewer changes the bits — test). `sliding_window_test.exs`, `engine_test.exs`.
- **Mamba** (`Vapor.Lock.Adapters.Mamba`, `MambaForCausalLM`): recurrent step — prefill and decode are the same instructions —, state kept in the worker (the `STEP` frame now feeds back state that is not updated in place), `Vapor.Recurrent` to generate; 3.4·10⁻⁷ and greedy output identical to `transformers`; constant cost per token. `mamba_test.exs`, `mamba_hf_test.exs`.
- **Tree speculation** (`Vapor.Speculative.Tree`): branches in *slots* over the context's shared pages (the winning branch inherits its pages — no copy), output = the target's greedy output for every tree shape; drafting by **prompt lookup** with overlapping copy (6 tokens/step where the output follows the context). `speculative_tree_test.exs`.
- **Exact tensor parallelism** (`Vapor.Shard`): column-parallel + all-gather, bits = one worker; the row-parallel form is measured (it changes 81 % of the elements). `shard_test.exs`.
- **Real bug**: DeepSeek with `n_group = 1` (V2-Lite) did not build; now group limiting is the identity, with a test.

**Numerics and verification**
- **Correctly rounded division** (semantics version 2, recorded in the certificates): Markstein + Dekker's exact residual without FMA; `+ − × ÷` = IEEE (DAZ/FTZ) on every substrate, GPUs included; 0 errors in 50 M exhaustive quotients. `division_test.exs`.
- Canonical **`log`** (≤ 1 ulp from the correctly rounded logarithm, IEEE specials) and **`softplus`** (a composition of canonical nodes, ≤ 4 ulps), on every substrate = oracle.
- **Higham's lemma in Lean** (`proofs/Vapor/Higham.lean`, kernel only); extraction regenerated.

**Spatial, latent and audio** ([docs/SPATIAL.md](docs/SPATIAL.md))
- `Vapor.Spatial`: 2-D and 3-D convolution (any *stride*, *padding*, dilation) **without a convolution kernel** (gather + sel + reshape + GEMV), GroupNorm by exact selector contractions, *upsampling*, attention over pixels; ≤ 4·10⁻⁷ against torch. `spatial_test.exs`.
- **VAE** adapters (`AutoencoderKL`, decoder; 8.7·10⁻⁷ against diffusers) and **DiT** (`DiTTransformer2DModel`, adaLN-Zero; positions bit for bit).
- **Cross-attention** = the existing attention (horizon at the last row of the other stream), ≤ 10⁻⁶ against `nn.MultiheadAttention`. `cross_attention_test.exs`.
- **Whisper** (`Vapor.Lock.Adapters.Whisper`): encoder and decoder as two programs (the contract gained declared **parts**), cross K/V once per audio; 4.8·10⁻⁷ / 4.2·10⁻⁷ and greedy output identical to `transformers`. `whisper_hf_test.exs`.
- **CLIP text tower** + BPE tokenizer with `</w>` (= CLIP's on 20/20 lines). `clip_tokenizer_test.exs`, `lock_hf_test.exs`.

**Transparency** ([docs/TRANSPARENCY.md](docs/TRANSPARENCY.md))
- `Vapor.Tlog`: RFC 9162 Merkle log in an append-only file with `fsync`, inclusion and consistency proofs, C2SP *signed-note* *checkpoints*, witness co-signatures (`Vapor.Tlog.Witness`: refuses rollback and forks); 196 probes from transparency-dev. `tlog_test.exs`.
- Server: `GET/POST /v1/vapor/tlog`, proofs over HTTP, anchored search receipts; `mix vapor.serve --tlog`.
- Console: **Registro/Ledger** tab — the browser verifies on its own (WebCrypto Ed25519, key pinned on first use, consistency with the last *checkpoint* seen) and draws the inclusion proof. `test/js/tlog_verify.mjs`.

**Tests and measurements**
- `mix vapor.bench --frontier` → `docs/bench/FRONTIER.md`; `mix vapor.quality` gained the 0.6 section (eight checks with a control).
- Fixed: options ignored in the engine tests' *helper*; a request larger than the whole pool waited forever (now `{:kv_pages, …}`); `layer_types` was ignored outside Gemma 3 (a hybrid Mistral would be read as all sliding) — now it is read or refused; the engine read the family's configuration for the ring (the architecture test caught it) — now it asks the adapter (`Vapor.Lock.ring_window/2`, an optional *callback*).

## 0.5.0 — 2026-10-02

Scrutiny of the request of this round: [docs/DIRECTIVE.md §8](docs/DIRECTIVE.md). The six limitations that 0.4.0 declared, one by one:

**1. Parity with `transformers` itself** (no longer against NumPy written from it)
- `test/vapor/lock_hf_test.exs` + `test/python/hf_lock.py`: checkpoints **written by `transformers` 5.18** (Phi-3, Phi-3 with partial rotary, Granite, ViT, ViT with pooler, CLIP-vision) admitted by the airlock with no forgotten tensor and compared to its forward: ≤ 4.5·10⁻⁷ relative, identical greedy decoding.
- **Real bug found:** `transformers` ≥ 5 stores `partial_rotary_factor` *inside* `rope_parameters`, where 0.4.0 did not look — a Phi-4-mini-style checkpoint was admitted and silently computed wrong (relative error 0.5). Now partial rotary is **built** (a permutation folded into the q/k weights, a table with pass-through pairs) and **every unknown key** of `rope_parameters` is refused by name.
- Encoder: `ViTModel` pooler (`tanh(dense)`), **CLIP** vision tower (`pre_layrnorm`, final norm only on the pooled row, `quick_gelu`, `visual_projection`).

**2. Any-to-any on real data** ([docs/ANY_TO_ANY.md §6](docs/ANY_TO_ANY.md))
- Real handwriting (UCI/scikit-learn, 1,797 digits, 497 held out): image → digit (`vapor_mlp`), digit → image by **diffusion** (`Vapor.Modal.Diffusion`, deterministic DDIM on the substrate), read back by the classifier and compared with the nearest training image.
- Real speech (Free Spoken Digit Dataset): certified front-end (`Vapor.Modal.Speech`: Hann spectrum + mel bank as a program) and a `vapor_encoder` trained on 5 voices, measured on a 6th never heard.
- The real voice → text → drawing → reading chain, measured.
- New topology adapter `vapor_mlp`; the encoder gained `head: "rows"` (classification per row) and `rows:` (shorter programs).

**3. Diffusion, OCR and JPEG** ([docs/OCR.md](docs/OCR.md))
- `Vapor.Docs.JPEG`: baseline, extended sequential and progressive; **the same pixels as Pillow (libjpeg-turbo) on 51 of 51 files** (integer IDCT, fancy *upsampling* and libjpeg's colour tables reproduced exactly).
- `Vapor.Vision.OCR`: OCR as a **model admitted by the airlock** — geometry from first principles (Sauvola, components, lines by the vertical core) and a `vapor_encoder` that reads the line's columns with **CTC** (without segmenting characters). Scanned PDF pages (`DCTDecode`, Flate with PNG predictors, 1 bit) and images enter the index with their confidence. `mix vapor.ocr`. `test/vapor/ocr_test.exs`: the shipped reader measured on fonts outside training and on a real photo, and the per-column logits equal to those of an independent PyTorch implementation.
- Verifiable diffusion: the optimal denoiser of a Gaussian mixture in closed form (`analytic/2`) — and, for a finite set, **attention** over the training points — tests the sampler; a consistency bug of DDIM with *clamp* was found this way (17 % → 99 % of the generated digits read correctly).

**4. Image search by meaning**
- Images with text (screenshots, scans, slides) are found by what is written in them (OCR → index). The CLIP vision tower checked; the text tower stays in the TODO.

**5. Fast merging: ≈ 0.6 → ≈ 11 M parameters/s** ([docs/MERGING.md](docs/MERGING.md))
- Kernels by binary pattern matching, blocks in all the schedulers, DARE with a counter-based generator (any block starts at its own offset): **output bit for bit equal to 0.4.0** in every method, receipts included.

**6. "TIES and DARE got worse" → diagnosis, measurement and selection**
- `Merge.diagnose/2` (`--diagnose`): regime from the weights — size of the deltas, energy concentration, sign conflict, and **whether there is a common ancestor** (cosine of the weights).
- `Merge.select/4` (`--try … --eval`): each candidate measured on held-out text, receipt `vapor.merge.select/1`.
- `:regmean` (least squares over the activations; Grams on the substrate; `Vapor.Lock.taps/1` says what to measure) — measured, ties with linear; recorded as such.
- Benchmark with **genuinely trained transformers** (`priv/quality/merge`): the linear merge of two fine-tunes beats the base in both languages and, on average, the two specialists; that of networks without a common ancestor is worse than both — as the diagnosis warns beforehand.

**Interfaces** ([docs/INTERFACES.md](docs/INTERFACES.md))
- Console redesigned (airlock identity: each result in a tank whose level is its measurement), **English by default, and Portuguese**, light and dark, *Vision*, *Listen* (recording from the microphone), *Draw*, *Merge* panels; SVG logo, favicon, installable app manifest; a **token** required to expose it beyond 127.0.0.1.
- `mix vapor.tui`: the console in the terminal, with no dependency.
- Tauri considered and not adopted (reasons, and when to revisit, in the document).

**Quality** — new checks with a control in `mix vapor.quality`: merging of trained models (5), real data (OCR on fonts outside training and on a real photo, speech from a never-heard voice, handwriting in both directions, novelty against memorization, chain), JPEG.

## 0.4.0 — 2026-10-02

Scrutiny of the request and of what was left out: [docs/DIRECTIVE.md](docs/DIRECTIVE.md).

**Model airlock** ([docs/AIRLOCK.md](docs/AIRLOCK.md))
- `Vapor.Lock`: the only place where a model family is known; contracts (`:causal_lm`, `:encoder`, `:codec`, `:map`) checked at the boundary; refusal with near-misses and repair; `mix vapor.lock`.
- Three adapter levels: alias by data/JSON (Phi-3/Phi-4 built in), *blueprint* (Granite 3.x), topology (HF encoder/ViT, VQ codec, linear projector).
- Engine, embedder, server and RAG read only the contract; an architecture test over the BEAM's atom table.

**Any-to-any without a new operator** ([docs/ANY_TO_ANY.md](docs/ANY_TO_ANY.md))
- Exact GELU as a canonical microprogram (closes TODO 4.1).
- Image (PPM/PNG, exact patches), audio (WAV, certified Hann spectrum, additive synthesis), VQ by `sample`, *soft token* injection (`inject: true`), hub with a pivot; ten routes measured on held-out data.

**Model merging** ([docs/MERGING.md](docs/MERGING.md))
- linear, *task arithmetic*, SLERP, TIES, DARE; deterministic, *streaming*, co-signable receipt; `mix vapor.merge` (closes TODO 6.3).

**Output quality** ([docs/QUALITY.md](docs/QUALITY.md), [docs/bench/QUALITY.md](docs/bench/QUALITY.md))
- Gates calibrated against controls that refuse to exist without separation; planted models; `mix vapor.quality` (27 checks, exits 1 on failure) and `--model` for real checkpoints.

**Documents and RAG over files** ([docs/DOCUMENTS.md](docs/DOCUMENTS.md))
- `Vapor.Docs`: recursive zip proof against *zip bombs*, PDF (object streams, ToUnicode, Type 1/TrueType; encrypted refused), Office/OpenDocument/EPUB, HTML, full PNG, JPEG metadata; checked against `pdftotext` and Pillow.
- `Vapor.Docs.Library`: a root over text and files, provenance down to the page, search by similar image; `mix vapor.rag`.

**Web console** ([docs/CONSOLE.md](docs/CONSOLE.md))
- A page at `/` of `Vapor.Serve`, offline: conversation with evidence and checked citations, documents, the model's contract, noise meter; `mix vapor.serve --docs`, works without a model.

**Repairs**
- GGUF export silently wrote Qwen3/Gemma 3/Mixtral/DeepSeek as `llama`: it now refuses.
- Phi-3 exported to HF would lose the sliding window: written with Mistral's spelling.
- `Vapor.RAG` read a decoder field (`cfg.hidden`): now the contract.
- CLIs without a UTF-8 locale received arguments as mojibake: repaired.
