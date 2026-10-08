# The console (`/` in `Vapor.Serve`)

```sh
mix vapor.serve --model ./Qwen2-0.5B --docs ./folder      # API at /v1, console at /
mix vapor.serve --docs ./folder                           # no text model: documents, vision, studio, quality, the desks
mix vapor.serve --data ~/.vapor                           # where the conversations live (default: VAPOR_HOME or ~/.vapor)
mix vapor.serve --ip 0.0.0.0 --token SECRET --docs ./p    # exposed: the token is mandatory
mix vapor.tui                                             # the same, in the terminal
```

A page served by the server itself (`priv/console/index.html`): no CDN,
no external font, works without internet. **English by default, Portuguese one
click away** (remembered in the browser); light and dark from the system or the button;
keyboard-navigable; readable at 390 px. Installable as an app (manifest and
icons), with its own logo and favicon. Interface decisions — and why not
Tauri — in [INTERFACES.md](INTERFACES.md).

## 0.16 — conversing, and a terminal

A new group, **Converse**:

- **Conversations** ([MAJLIS.md](MAJLIS.md)) — on the left, the conversations (new, import an export
  from vapor, ChatGPT or Claude, search across all of them); in the centre, the conversation, in which each message
  has *edit*, *another answer*, *pin*, *continue from here*, *fork from here* and *copy*, and the
  branches appear as **‹ i/n ›**; on the right, the **context** as a bar (instructions, each
  message sent, the pinned ones in gold, the budget line) with the ones left out counted, the
  compaction (which names what it summarises, and can be undone), the **tree** of branches (the path in gold;
  a click goes there), the settings (title, instructions, model, tools, budget,
  temperature) and share · revoke · export (verifiable JSON, Markdown).
- **Ask the library** — the old *Chat*: a question answered with its sources.
- **Terminal** ([DIWAN.md](DIWAN.md)) — the same commands as `bin/vapor`, in a jailed session,
  with ANSI colours, history, Tab, and the session's files in an editor alongside.

Removed, because they were fixed demonstrations ([DIRECTIVE §19](DIRECTIVE.md)): Physics, Networks, Algorithms,
Mathematics, Science, Games, Draw, Listen, Training, Merging and the demonstration dossier. The sections
below that describe them remain as a record of rounds 0.10–0.11.

A browser test (`test/js/console_majlis.mjs`) converses, edits, moves between branches, asks for
another answer, pins, compacts, forks, searches, shares and revokes (reading the link without the
console), and in the terminal decides the Almizan oscillator, uses a *pipe*, completes with Tab, edits a
file and tries to read outside the jail; then it switches to Portuguese and to dark. Screenshots:
`docs/img/conversas-*.png`, `docs/img/terminal.png`.

## 0.14 — the open workbench

The console opens on the **Open workbench** group: *Workspace* (write or describe any
problem; the type is detected — search, claim, game, system, scene, Alembic), *Crucible* (your
system, with evidence) and *Assay* (model evaluations: signal or noise). The search
appears in the **furnace** — sparks per evaluation, the best-so-far line, the dashed line of the
random control, the portfolio's allocation — and the verdict on the **touchstone**: a gold
streak for each check that passed, a lead one for each that failed. The person proposes candidates, pins and
bans finalists, measures external objectives and asks the model for a draft or proposals. Calls:
`GET /v1/vapor/workspace`, `POST /v1/vapor/{detect,alembic,athanor,athanor/verify,game,crucible,assay,formalize,scene/ops,scene/mind}`,
`GET|POST /v1/vapor/athanor/:id` (`?since=N` for the new sparks). Identity: soot
`#14110D`, parchment, brass, verdigris and cinnabar; headings in an old-style serif from the
system fonts (no font downloaded). Screenshots: `docs/img/bancada-*.png`.

## 0.15 — the Opus

A new group, **Opus**, with five desks — each one decides something and shows the decision in the
form in which it is checked:

- **Rebis** — two vessels side by side (A and B) and a **seal** between them that closes (same
  function), breaks (different) or stays open (unknown). The counterexample appears
  grouped into words (`a = 0xDEADBEEF`), the outputs that differ named; the ANF,
  word identity (`m[16] = a[8] * b[8]`), stabiliser (a strip of qubits, the
  random ones hatched) and AIGER modes.
- **Aludel** — the polynomial, the box interval by interval and the sense (`≥ 0` or `> 0`); the
  result draws the **subdivision cells** over the box (the certified leaves, the
  refuting point marked, the cell where the budget ran out highlighted). In barrier mode, the **phase portrait** (field
  arrows), the initial and unsafe sets and the curve `B = 0` (by *marching squares*), with the three
  conditions on the touchstone; "find one" asks the LP for the synthesis.
- **Tabula** — the clauses as a tabula (the overridden ones struck through, the ones in force marked), the
  facts as buttons that recompute the positions, and each finding with **"set these facts"**, which
  puts the tabula into the scenario that triggers the collision.
- **Cupel** — the exercise: choose the bit to flip in a correct product; a strip with the 32
  bits of a float32 (sign, exponent, mantissa) shows what the cupel catches and what stays below
  the envelope.
- **Amalgam** — numbers on a line: the exact sum (marked once) and the naive sums in several
  orders, scattered around it.

Each desk has a **shelf** of examples (titles in English and Portuguese). The desk's state —
the mode, the values typed, the facts switched on — lives outside the DOM: switching language redraws the
desk in the same mode, returns each value to its field and recomputes the result that was in view.
Calls: `GET /v1/vapor/opus` (the shelves), `POST /v1/vapor/{rebis,aludel,tabula,cupel,amalgam}`.
A headless browser test (`test/js/console_opus.mjs`, `@tag :playwright`) goes through all
the examples, switches on a fact, sets a scenario, switches to Portuguese and to the dark theme, and fails on
any page error or HTTP response ≥ 400.

## The idea

What is missing from local-model interfaces is not another chat: it is the
**evidence beside the output**. The visual identity is that of a canal lock: each
result sits in a tank whose **water level is its measure** — the confidence
of a line read by the OCR, the certainty of the speech reader, the probability the
classifier gives the drawn digit, the relative score of a passage — with
the calibrated line marked. The rest is silent.

## The panels

**Ask — Conversation.** With "answer with the documents", the question is
searched in the library; the passages come in as numbered sources and the answer
arrives by *streaming*. The *Evidence* column shows the path to the page
(`bundle.zip ▸ pasta/simple.pdf page 1`) and the receipts; citations
`<quote src="N">…</quote>` are checked on the server.

**Read — Documents.** Zip (nested), PDF (scanned too: the OCR reads it),
Office, EPUB, HTML, PNG, JPEG: what does not become text is stated. Search shows
provenance, score, Merkle proof and the file's hash, and recomputes the receipt.
Images with text are found by what is written in them.

**Read — Vision.** A photo, a screenshot or a scanned PDF (CCITT, JPEG,
Flate…): the page with the lines marked, **the blocks numbered in the order in
which they are read** and the **reading thread** — a thin line from the end of each line
to the start of the next, which shows at a glance whether the machine read the whole left
column before the right one. Alongside, the text grouped by block, each
character tinted by its confidence (dotted below 80 %, wavy below
50 %) and **the characters the language model chose** highlighted, with
a button that shows the frames-only reading (what the model changed appears
struck through in a warning colour). The evidence of the reading is not only the confidence: it is
also *who* decided each letter.

![Vision](img/console-visao.png)

A **table** on the page (0.8) appears framed in the image and, below,
drawn from its cells — header, merged cells, numbers
right-aligned, each cell with its confidence (the same dotted and
wavy underlines) and linked to its box on the page: hovering over one lights up the
other. In the reading list, the table's block says where it is. A selector
swaps the drawing for Markdown, CSV or HTML, and *Copy* copies the format
shown.

![Vision with a table](img/console-tabelas.png)

**Read — Listen.** Record 1.5 s through the microphone (the browser encodes a WAV; the
server resamples to 8 kHz) or drop a WAV: the digit, all the
probabilities and the mel spectrum the reader heard; a button draws the digit
heard — voice → text → image.

**Create — Draw.** A digit by diffusion, the denoising trajectory
step by step, what the real-data classifier reads in the generated image and the
distance to the nearest training image. The same seed gives the same image
on any machine.

![Draw, dark, Portuguese](img/console-desenhar-escuro.png)

**Measure — Merging.** The laboratory over the trained decoders in
`priv/quality/merge`: the regime diagnosed from the weights (in words assembled
from the numbers, in both languages), each method measured on validation and test, the
one chosen.

![Merging](img/console-fusao.png)

**Measure — Quality.** Paste a text or measure the last answer: where it falls
between the calibrated gates of noise and of real text.

**Measure — Airlock.** The contract of the served model and the registered adapters.

**Trust — Ledger.** The server's transparency log (`--tlog`):
size, root and signed *checkpoint*, the latest entries (notes and anchored search
receipts) and a field to anchor a note. The *verify* button
does not ask the server whether all is well: **the browser checks on its own**
— the *checkpoint*'s Ed25519 signature (WebCrypto), the inclusion proof of
each entry and the consistency with the last *checkpoint* this browser
saw; the log's key is pinned on first use and a change is flagged. The
chosen proof is drawn: the leaf, the siblings going up, the root.
Details: [TRANSPARENCY.md](TRANSPARENCY.md).

![Ledger, dark](img/console-ledger.png)

**Trust — Dossier** (0.8). Drop an audit dossier (`.vdossier`, or the
PDF that carries it) or assemble the demonstration one: the verdict in the water level
(the fraction of items verified; red if something failed), the Merkle root
recomputed, and the **weave** — provisions of the EU AI Act and of ISO/IEC
42001 in the rows, the evidence in the columns, on the right how many verified
items support each provision, and "none" where there is no evidence.
Below, each item with its hash and the check by its own rules, the
signatures and the anchors in the log. The demonstration dossier can be downloaded,
with the page that verifies itself. Details: [AUDIT.md](AUDIT.md).

![Dossier, dark](img/console-dossie-escuro.png)

**Create — Studio** (0.9). A node canvas for image, sound, video, 3D,
diffusion and RL ([STUDIO.md](STUDIO.md)).

- **Building the graph**: the palette on the left, by category and with search,
  adds nodes by click or by drag. Wires go from an output point
  to an input point, and during the drag only the inputs of a compatible type
  light up. **From the keyboard**, the inspector lists the possible sources of each
  input. Clicking a wire removes it; `Delete` removes the selected node; `Alt`
  + arrows moves it.
- **Navigating**: dragging the background pans the canvas, the wheel zooms, and a starter
  template or an imported workflow arrives framed.
- **Running**: after *Run*, each node shows its preview (image,
  GIF, playable sound, rendered mesh, value) and whether it was **computed** or came
  from the **cache**, with the time and the digest. It is the same canal-lock metaphor: a
  node's tank is empty before running, fills with flowing water when it is
  computed and shows still water when it comes from the cache.
- **Checking**: the run ends in the **seal**, the Merkle root, and
  *Verify* re-runs without the cache and says whether the root matches.
- **Importing and exporting**: *Import ComfyUI* accepts the API-format JSON
  and shows the translation notes; *Export JSON* downloads the graph.

There are six starter templates, and all of them run with no file and no download, except
text → image, which needs a diffusers checkpoint. The graph being edited
is kept in the browser. The cache and the previews live as long as the server
lives: changing a parameter recomputes only what depends on it, and a run
entirely in cache comes back in about 1 s.

![Studio](img/console-estudio.png)

![Studio, dark, in Portuguese: the same graph, already computed, comes back entirely from the cache (0 computed, 5 from the cache)](img/console-estudio-escuro.png)

### Round 0.10: substrates, training, physics, networks, other scripts

**Measure — Substrates.** Each substrate present runs the airlock's probes;
the table shows the verdict (canonical, within the envelope, refused) and the
**numerical fingerprint** — FMA, FTZ, DAZ, reduction order, mantissa
bits, signed zero, NaN, division, functions —, with each field that
departs from the exact oracle marked.

**Measure — Training.** The receipt of the model vapor trained (`priv/lm`): the
curve on held-out text against the baselines it has to beat
(byte frequency, Witten–Bell 3 and 5), the digests, the sample; and the
**stream** beyond the training length, window by window, against the
control of growing positions.

**Simulate — Physics.** *Chaos, run twice*: the double pendulum and
its copy one ulp away, animated; the oracle checks the native worker bit for bit
over the first steps; the distance between the worlds on a log scale. *A
digital twin*: residuals and CUSUM with the threshold, the fault and the alarm marked, and the
log rebuilt from the model and the actions.

**Simulate — Networks.** A model (Barabási–Albert, Erdős–Rényi,
Watts–Strogatz, planted communities), the force-directed layout coloured by
Louvain's communities, the clustering against the configuration null model, the
power-law verdict with the water level at the *bootstrap* p, the
robustness to failures and to attacks, and the top of the PageRank.

**Read — Vision** gained a choice of **script** (Latin, Arabic, Cyrillic,
cursive, 中文, 日本語, 한국어, formula), with a note that says what each
reader promises — cursive answers with the measured refusal and the path. Arabic
lines appear right to left (the visual order of the frames on
*hover*); figures are marked on the page with their caption, and the data of a
chart that was read appear redrawn alongside — lines, points or bars, in the
series' colours, with the categories read (or the reason for refusal); a
formula comes back as LaTeX. The footer says **which reader** read it (and whether there was a
language model): Arabic and Cyrillic have their own, without a language model.

![Physics: the digital twin](img/console-fisica.png)

![Networks, dark, in Portuguese](img/console-redes-escuro.png)

![Training](img/console-treino.png)

![Substrates](img/console-substratos.png)

![Vision: Arabic, right to left](img/console-visao-arabe.png)

![Vision: Cyrillic](img/console-visao-cirilico.png)

![Vision: a figure with a caption, and the chart read back](img/console-figura.png)

![Vision: formula → LaTeX, dark, in Portuguese](img/console-formula-escuro.png)

![Vision: cursive refused, with the measurement and the path](img/console-cursiva-recusa.png)

The identity line at the top also says **where the model runs**: `substrate CPU` or
`substrate GPU · <device>` (with `mix vapor.serve --gpu`).

### Round 0.11: bringing images to life, sketches, discovering, science, games, archives

**Make — Living scene**: drop a photo, a painting, a generated image or
choose an example; within seconds it moves — a perspective camera over
the layers, inhabitants walking on the ground, weather, light, wind. The direction
bar accepts sentences ("a stormy night, three villagers walking to
the door, fireflies; orbit slowly") and the chips apply one operation
each; the script appears alongside, the depths adjust layer by
layer, "show depth" and "show the walkable ground" reveal the
analysis. A loose drawing gets a skeleton and waves, walks, dances. Outputs:
an 8 s video, **an HTML file that plays offline**, and the verifiable
archive.

**Make — Sketch**: technical drawing (the sketch beside what it meant
to say, the constraints listed, SVG and DXF; "show without the constraints" is the
control) or floor plan → 3D (rooms with area, doors with width, a
3D viewer that rotates and zooms, GLB).

**Discover — Mathematics**: choose a theorem (or a false one, marked ✗) and
prove it — the figure with the claim drawn, the certificate and the independent
check; *conjecture and prove* draws the lines and circles found;
the Betti table highlights the torsion; persistence shows the cloud and the
bars. **Algorithms**: the sorting network diagram, the 7 products and
the counts chart, the minimal program with the 8/16/32-bit
checks.

**Simulate — Science**: eleven cards, each with the verdict, value,
reference and control, and a chart when there is one (the orbit of the
coherent state, the liquid's g(r), the 20-mer's fold). **Games**: play against
the self-play agent while seeing where the search looked; the curve of losses per simulation;
the domain-randomisation bars.

**Trust — Archives**: drop a zip saved from any panel: intact or
altered, and — if deterministic — recomputed and compared.

![Living scene: a guild hall on a stormy night, with torches, embers and three villagers going to the door](img/console-cena-guilda.png)

![Living scene: the landscape at dusk, with birds and butterflies](img/console-cena-paisagem.png)

![A hand-drawn stick figure, with a skeleton, dancing in the landscape](img/console-cena-desenho.png)

![The exported scene, standalone, offline](img/console-cena-exportada.png)

![Sketch → technical drawing: lines, circle and arc with the constraints found](img/console-esboco-tecnico.png)

![Sketch → 3D floor plan](img/console-esboco-planta.png)

![Mathematics](img/console-matematica.png)

![Algorithms](img/console-algoritmos.png)

![Science](img/console-ciencia.png)

![Games](img/console-jogos.png)

### Round 0.12: workbench, engineering, logic, boards, proteins, render

Navigation was regrouped by what one does — *Ask*, **Solve**
(Workbench, Engineering, Logic), *Discover*, *Simulate* (with Boards and
cards, Proteins), *Make* (with Render), *Read*, *Measure*, *Trust* — and
each group is **in alphabetical order in the language shown** (the
`Intl.Collator` reorders when the language changes; the arrows follow the visible
order). **Ctrl/⌘ K** opens the **command palette**: every panel and every
example, in alphabetical order, accent-insensitive search, Enter opens. The address
keeps the panel (`/#bench`).

The new panels (`priv/console/workbench.js`, served **inlined** in the
page — it remains a single document that works offline) follow the
same design: a bar (examples in alphabetical order, the action, the
status), the **text** on the left (Tab indents, Ctrl+Enter runs) and a
**summary** on the right (what was recognised, the certificate with ✓/✗ and the
numbers that justify it), and the results below:

- **Workbench**: 16 examples (oscillator with units, Lorenz, Robertson,
  projectile with drag and event, SIR, verified heat, Fisher–KPP,
  Terzaghi consolidation, plucked string, verified Poisson, beam
  spreadsheet, roots, fit, Rosenbrock, optimal can, 4096 native
  oscillators); selectable series, phase portrait, profile with a time
  control and u(x, t) heat map, verification table with the order, roots,
  data and fit with residuals, KKT, percentile bands.
- **Engineering**: eight tools in tabs (alphabetical order), 20
  examples; tables of voltages and currents, Bode (magnitude and phase),
  transient; **single-line diagram** with flows and colour by voltage;
  **deformed frame**, supports, axial/shear/moment diagrams per
  member and **animated modes**; **FEM mesh coloured by von Mises**,
  deformed; **pipe network** with flow arrows; species, invariants
  and stoichiometric matrix; flash; **McCabe–Thiele** drawn.
- **Logic**: 12 examples; verdict, certificates (witness, DRUP),
  the witness drawn (Schur/van der Waerden coloured strip, Ramsey's
  K₅, the queens' board), Knuth–Bendix rules, normal forms and
  derivations, Gröbner bases.
- **Boards and cards**: chess (click to move, promotion, an engine that
  replies, undo, flip, FEN, analysis, mate proof with the tree,
  perft with *divide*), shogi (pieces in kanji, the opponent's rotated,
  a clickable **hand** for drops, optional promotion asked), Go
  (5–13, komi, MCTS that replies, pass, area scoring), k-in-a-row
  (m, n, k, gravity; the best moves highlighted by the exact solver),
  poker (Kuhn/Leduc, exploitability curve, strategy per information
  set).
- **Proteins**: samples (1A8O, 1LCD and their NMR models) or an opened
  PDB; a **3-D viewer** of the Cα trace coloured by secondary structure
  (drag rotates, wheel zooms), contact map, sequence; the
  **pipeline** with model × native superposition, truth/prediction map,
  precisions and the model's PDB; compare; align.
- **Render**: the **GPU** tracer converging live, a scene editable
  as text with recompilation on every keystroke, **dragging orbits the camera and
  rewrites the `camera` line**, five examples, resolution, PNG, the
  server's reference with the **agreement** of the mean radiances, the
  furnace test with the control.
- **Living scene**: inhabitant inspector, route by clicks, time
  line, GIF of exact frames ([SCENE.md §6.1](SCENE.md)).

Checked by `test/js/console_desks.mjs` in headless Chromium (with
WebGL2 over SwiftShader): **every example of every panel** run through the
page, chess and Go played, the palette used, the alphabetical order checked
in both languages, no page error — 65 checks
(`console_desks_test.exs`).

![Workbench: the Lorenz attractor, series and phase portrait](img/console-bancada.png)

![Engineering: deformed frame, supports and diagrams](img/console-engenharia-portico.png)

![Engineering: Stagg & El-Abiad power flow, single-line diagram](img/console-engenharia-potencia.png)

![Engineering: cantilever plate (QM6) coloured by von Mises](img/console-engenharia-mef.png)

![Logic: R(3, 3) = 6 with the witness on K₅ and the DRUP refutation](img/console-logica.png)

![Boards: chess with the engine replying and the knight's legal moves](img/console-xadrez.png)

![Boards: shogi](img/console-shogi.png)

![Proteins: the pipeline on 1A8O — superposition and contact map](img/console-proteinas.png)

![Render: the GPU tracer, scene as text](img/console-render.png)

![The command palette (Ctrl/⌘ K)](img/console-paleta.png)

### Markets (0.13): Finance and Trading desk

A new group in the navigation, in alphabetical order like the others. The two
panels follow the design of the *Solve* panels: the domain text on the
left, on the right a **seal** with the verdict and the numbers that
support it, and below the charts and the tables. What is new in the design:

- **the seal** — a frame that changes colour (lock, ember, sand) and says in
  one line whether the result holds and why ("every instrument
  repriced to 4.5·10⁻¹⁶", "butterfly arbitrage at k ∈ [0.645,
  1.255]", "identical bits — x86_64 · 2 threads");
- **the gates** — a list of checks with ✓/✗ and the detail of each
  one (the backtest's four noise gates; the exchange
  session's three certificates);
- **the Basel traffic light** — green, yellow or red for the last
  250 VaR forecasts;
- **the book ladder** — prices in the centre, bid depth on the
  left and ask depth on the right, the spread shaded;
- **the chain** — each journal entry as a block with the start of its
  hash, linked to the previous one; a click shows the event and the reports;
  trades in lock green, rejections in ember; below, the Merkle proof of the
  first trade, the ITCH feed in hexadecimal and the FIX reports.

Finance has eight tasks (Arbitrage, Backtest, Calendar and money,
Curve, Native Monte Carlo, Options, Portfolio, Risk) and the Trading desk three (Order
book, Exchange session, Microstructure), each with examples —
including the ones that **must** go wrong (a price below the no-arbitrage
bound, a peek at tomorrow, a quote that leaves a negative
forward). *Save* writes a vapor archive that recomputes itself
(`finance.*` is replayable; Monte Carlo is not, because it needs the worker).
`test/js/console_markets.mjs` runs **every example of every task** through the
page in Chromium, in English and then in Portuguese, clicks on the chain and uses
the palette (87 checks, no errors).

**In Portuguese, everything in Portuguese.** The visualisations are written once,
with the English the server speaks; in Portuguese, a single pass over the
already drawn result translates every label and every server sentence
(verdicts, gates, refusals, the first comment line of the
examples) through a table of phrases and rules with numeric slots. It runs
**after** the visualisation, so the logic that reads the English (a verdict by
regular expression) does not change; code, *hashes* and the FIX and ITCH bytes
are never translated. The browser test lists, example by example, every
English word left over in the Portuguese version — a gap in the table
shows up as a failure, not as a half-translated panel.

![Finance: DI curve (DI1 + LTN + NTN-F), every instrument repriced](img/console-financas-curva.png)

![Finance: Vogt's SVI slice — the fit recovers the parameters, the density goes negative and g(k) < 0 is flagged](img/console-financas-sorriso.png)

![Finance: Monte Carlo on the worker — bits from the oracle and from two threads, Asian option with a control variate](img/console-financas-montecarlo.png)

![Finance: the best of 30 moving-average crossovers on noise — rejected by the deflated Sharpe](img/console-financas-backtest.png)

![Finance: normal VaR on fat tails — Kupiec rejects, yellow light](img/console-financas-var.png)

![Finance: a butterfly that pays, found and checked in rationals](img/console-financas-arbitragem.png)

![Trading desk: the order book, the journal chain, the Merkle proof, ITCH and FIX](img/console-mesa-livro.png)

![Trading desk: an audited exchange session — naive engine, limits, feed, Hawkes](img/console-mesa-sessao.png)

![Trading desk: Hawkes planted and fitted, with the time-rescaling test](img/console-mesa-hawkes.png)

## The calls, for any client

| | |
|---|---|
| `GET /v1/vapor/info` | model contract, server context, adapters, library state |
| `GET /v1/vapor/library` · `POST /v1/vapor/library` `{name, data}` | files and warnings · ingest (base64, up to ~12 MB per call) |
| `POST /v1/vapor/search` `{query, k}` | passages with provenance, proofs, root and receipt |
| `POST /v1/vapor/verify` | recompute a search result |
| `POST /v1/vapor/citations` `{answer, query, k}` | check literal citations |
| `POST /v1/vapor/search_image` `{name, data}` | similar images |
| `POST /v1/vapor/quality` `{text}` | text gate verdict, thresholds and measures |
| `POST /v1/vapor/ocr` `{name, data, script?}` | text from an image or from the scanned pages of a PDF: lines, boxes, per-character confidences; tables with structure, cells and Markdown/HTML/CSV; figures with caption and the charts' data; `script`: `latin`, `arabic`, `cyrillic`, `cursive` (refused), `zh`, `ja`, `ko`, `math` |
| `GET /v1/vapor/substrates` | each substrate present, admitted by measurement: verdict, numerical fingerprint, probes |
| `POST /v1/vapor/scene/analyze` `{name, data}` (or `name: "sample:outdoor"`) | the scene: layers (PNG with alpha) and depths, horizon, walkable ground, light, palette |
| `POST /v1/vapor/scene/rig` `{name, data}` | a drawing's skeleton: bones, mesh and weights, the image with alpha |
| `POST /v1/vapor/scene/direct` `{prompt}` | the prompt as operations and the words not understood |
| `POST /v1/vapor/scene/export` `{scene, title?}` | **a standalone HTML page** that plays the scene offline |
| `POST /v1/vapor/sketch` `{name, data, mode, snap?, longest?}` | `vector`: lines, circles, arcs, constraints, SVG, DXF; `plan`: walls, doors, rooms with area, mesh and GLB |
| `POST /v1/vapor/archive` `{kind, recipe, result}` · `POST /v1/vapor/archive/check` `{data}` | the archive (zip, base64; the deterministic kinds computed by the server from the recipe); the check and the recomputation |
| `POST /v1/vapor/audit` `{data, log_key?}` | check a dossier (or its PDF): items, clauses, signatures, anchors, root |
| `GET /v1/vapor/opus` | the example shelves of the five Opus desks (0.15) |
| `POST /v1/vapor/rebis` `{op: equivalent \| anf \| identity \| stabilizer \| aiger, a, b?, spec?, n?, seed?}` | the same function, with the DRUP proof or the shrunk counterexample; the ANF; the word identity; the measures; the AIGER |
| `POST /v1/vapor/aludel` `{op: decide \| enclose \| barrier, vars, poly \| field, box \| domain/init/unsafe, sense?, barrier?, synthesize?}` | certified (witness reproduced, leaves), refuted (exact point) or exhausted; the interval; the three barrier conditions |
| `POST /v1/vapor/tabula` `{text, facts?}` | antinomies with a scenario, proved pairs (DRUP), resolved ones, silences; the positions under the facts |
| `POST /v1/vapor/cupel` `{n, k, seed, trials, bit}` | the per-bit detection profile, the example of a flipped bit, the exact int8 |
| `POST /v1/vapor/amalgam` `{numbers, format}` | the exact sum rounded once, and the naive ones in several orders |
| `GET /v1/vapor/studio/nodes` | the studio's node catalogue (typed ports, parameters with range and default) and the starter templates |
| `POST /v1/vapor/studio/run` `{graph}` | run a graph: per node, computed or cached, time, digest and preview of each output; the Merkle root |
| `POST /v1/vapor/studio/verify` `{graph, root}` | re-run without the cache and compare the root |
| `POST /v1/vapor/studio/comfy` `{workflow}` | translate a ComfyUI workflow (API format): the graph and the notes, or the refusal with the untranslated nodes |
| `POST /v1/vapor/solve` `{text, ensemble?}` | the workbench: recognised type, solution and evidence (steps, observed order, KKT, bands) |
| `POST /v1/vapor/engineering` `{kind, text, method?}` | circuit, power, structure, fem, pipes, reactions, flash, distill — result and certificate |
| `POST /v1/vapor/logic` `{text}` | verdict with certificate (model, DRUP, witness, rules, basis) |
| `POST /v1/vapor/chess` `{fen, action, move?, depth?, n?}` · `/shogi` `{sfen, …}` | state and legal moves, move, engine, analysis, mate proof, perft |
| `POST /v1/vapor/go` `{size, komi, moves, action?, sims?}` · `/mnk` `{m, n, k, gravity, moves}` · `/poker` `{game, iterations}` | Go by move list; k-in-a-row solved; CFR+ with curve and strategy |
| `POST /v1/vapor/protein` `{action: analyse \| compare \| pipeline \| align, …}` | structure, metrics, pipeline, alignment |
| `POST /v1/vapor/render` `{text, width, height, spp}` · `GET /v1/vapor/render/furnace` | reference PNG and mean radiance; the furnaces and the control |
| `POST /v1/vapor/scene/gif` `{frames, fps}` | exact PNG frames → GIF |
| `POST /v1/vapor/finance` `{kind, text}` | the finance desk (0.13): `arbitrage`, `backtest`, `book`, `calendar`, `curve`, `exchange`, `mc`, `micro`, `options`, `portfolio`, `risk` — result and certificate ([FINANCE.md](FINANCE.md)) |
| `GET /v1/vapor/thumb?doc=…` | PNG thumbnail of an indexed image |
| `GET /favicon.ico`, `/logo.svg`, `/manifest.webmanifest`, `/icon-192.png`, `/icon-512.png` | identity and installation |

With `--token` (or `VAPOR_TOKEN`), every call except `/health` requires
`Authorization: Bearer …` or the *HttpOnly* cookie that `/?token=…` sets once;
the server refuses to listen outside 127.0.0.1 without a token.

Verified by `test/vapor/console_test.exs` (real HTTP, no text
model) and in a headless browser (Chromium) in light, dark, English, Portuguese
and at 390 px width, with no console error.
| `GET /v1/vapor/threads` · `POST /v1/vapor/threads` `{title, system, model, tools, budget}` | the conversations, the models, the tools · a new one (0.16) |
| `GET /v1/vapor/threads/:id` · `/tree` · `/context` · `/export?format=json\|markdown` | settings and path · all the branches · exactly what the model will read · the export |
| `POST /v1/vapor/threads/:id/:op` | `say` `{text}`, `reply`, `edit` `{node, text}`, `regenerate` `{node}`, `switch`/`rewind` `{node}`, `fork` `{node, title}`, `pin` `{node, on}`, `compact` `{upto, text}`, `uncompact`, `settings`, `share`, `revoke`, `delete` |
| `POST /v1/vapor/threads/import` `{data}` · `POST /v1/vapor/chat/search` `{query}` | an export from vapor, ChatGPT or Claude · search across all conversations |
| `GET /v1/vapor/journal/:id` | the journal of an agent run (verifiable) |
| `GET /shared/:id?cap=…` · `GET /v1/vapor/shared/:id?cap=…` | a shared conversation, read-only (page without *script* · JSON) — **without the console token: the capability is the authority** |
| `POST /v1/vapor/diwan` `{session, line}` · `GET\|POST /v1/vapor/diwan/file` · `POST /v1/vapor/diwan/complete` | the terminal: one line in a jailed session · the session's files · completion |

## Limits

- One library per server, in memory (persistence: `mix vapor.rag`).
- Upload through the page is limited to ~12 MB per file; for more, `--docs` or `mix vapor.rag`.
- The token protects access; confidentiality on the network requires TLS in front (a proxy).
- Messages coming from the server (refusals, errors) stay in English, like the API; so do the names and descriptions of the studio's nodes.
- The studio reads files (`image.load` by path, checkpoints) only inside `VAPOR_STUDIO_DIR` (default: the directory where the server was started) or `VAPOR_MODELS`; through the page, files come in as data.
