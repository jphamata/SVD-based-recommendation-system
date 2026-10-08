# Vision: OCR by a model admitted through the airlock, bit-exact JPEG, and the office scan

0.4.0 declared: *there is no OCR and no JPEG decoder; image search is
by visual similarity, not by meaning.* 0.6.0 still declared: *no
column order, greedy CTC without a language model, CCITT/JBIG2 PDFs refused.*
This page says what was done, how it was measured and what is still left out;
§3b–§3d are from 0.7.0.

## 1. JPEG: libjpeg's decoder, bit for bit, with no dependency

`Vapor.Docs.JPEG` decodes *baseline*, extended sequential and
**progressive** JPEG (with successive approximation), any subsampling (4:4:4, 4:2:2,
4:2:0, 4:4:0, 4:1:1), *restart* intervals, greyscale, YCbCr and Adobe
RGB. It does not approximate libjpeg: it **is** the arithmetic libjpeg specifies —

- the exact integer IDCT (`jidctint.c`, `JDCT_ISLOW`: 13-bit constants, two
  passes, the range-limit table with the 10-bit fold);
- the "fancy" *upsampling* (`jdsample.c`: the triangular filter for h2v1, h1v2 and h2v2
  with libjpeg's rounding biases and edge rules, context rows
  replicated at the top and bottom);
- YCbCr → RGB through the 16-bit fixed-point tables (`jdcolor.c`).

Result: **the same pixels as Pillow (libjpeg-turbo) in 51 of 51
files** — the two real scikit-learn photos as shipped and re-encoded, a
synthetic image with hard edges, all subsamplings, progressive,
*restart*, optimised tables, quality 30 to 100, odd sizes
(`test/vapor/jpeg_test.exs`, `test/python/jpeg_fixtures.py`). A
640×427 photo decodes in ~0.5 s on the BEAM. Refused with a reason: arithmetic
coding, lossless and hierarchical JPEG, 12 bits, CMYK/YCCK.

Consequences: JPEG images enter the library's visual index, get a
thumbnail in the console, and **scanned PDF pages** (which are almost always
JPEG, `DCTDecode`) become readable by the OCR.

## 2. OCR: a printed line is a sequence of columns

Classical OCR cuts the line into characters and classifies each one. That fails
exactly where real text is hard: letters that touch (serifs, *kerning*,
low resolution, blur), ligatures. We measured this here: with connected-component
segmentation, **54 %** of the lines in the test fonts did not have the right
number of characters — "ex", "ti", "rn" became a single blob.

Lateral thinking: **reading a line is the same problem as hearing speech.** The
line becomes a sequence of frames (windows of 8 columns every 2), a bidirectional
encoder classifies each frame, and **CTC** (connectionist temporal
classification) collapses the frames into characters. There is no character
segmentation: touching letters and ligatures are the model's problem, not the
geometry's.

```
image ─ Segment: ink (Sauvola), components, lines ─▶ line bitmap (32 px, baseline on row 22)
      ─ 32×8 frames every 2 columns ─▶ vapor_encoder (head: rows) on the substrate ─▶ greedy CTC ─▶ text
```

**No new operator.** The reader is a `vapor_encoder` — the same topology that
reads image patches — with a per-row head (`head: "rows"`, added
in this round: a `linear` over all rows instead of row 0). Overlapping
windows are just a copy of rows (a convolution with a stride smaller than the
kernel is a *gather* of windows). The checkpoint (`priv/ocr`: `config.json` with the
alphabet in `labels`, `model.safetensors`) comes in through the airlock like any
model; another reader with the same contract — another alphabet, another language — comes in
the same way.

**The geometry** (`Vapor.Vision.Segment`, no model):

1. **Ink** by Sauvola's adaptive threshold over integral images — a
   page photographed under uneven light binarises like a clean *scan*.
2. 8-connected **components** (labels in `:atomics`).
3. **Lines** by the projection of each component's vertical **core** (the middle
   half): ascenders and descenders of neighbouring lines do not join them.
4. **Line bitmap** with only that line's ink (the descender of the line above stays
   out), scaled so that the median glyph height is 12 px and
   the baseline sits on row 22 of 32.

**Training** (`test/python/ocr_render.py`, `test/python/train_ocr.py`): 31,434 lines
rendered by FreeType in 18 fonts (DejaVu, Free, Liberation, Nimbus,
Inter, Caladea), 14–44 px, degraded like a *scan* or a photo (blur, sensor
noise, light gradient, JPEG, slight rotation), text from the reference corpora
and random strings over the whole alphabet (printable ASCII and the Portuguese
accents). The training bitmaps are **the ones vapor computes**
(`Vapor.Vision.OCR.dataset/2`): training and inference see the same thing.

## 3. Measured

Fonts **never seen in training** (C059, P052, Carlito, URW Gothic, URW Bookman)
and text from the **held-out** corpora; a **real photo** of a printed page
(`skimage.data.page()`, uneven light); Tesseract 5.3.4 (LSTM, English) on the same
images, for reference — readings frozen in `priv/quality/ocr/tesseract.json`
by `test/python/ocr_tesseract.py`: the product does not run an external tool
(`audit_test.exs`), and the comparison reproduces without Tesseract installed. Numbers regenerable by `mix vapor.quality`
([bench/QUALITY.md §4c](bench/QUALITY.md)) and by `mix vapor.ocr eval`.

| set | vapor CER | vapor WER | Tesseract CER | Tesseract WER |
|---|---|---|---|---|
| 5 fonts outside training, 40 lines | **6.8 %** | 25.6 % | 5.2 % | 20.6 % |
| — C059 | 2.3 % | 11.9 % | 2.3 % | 7.1 % |
| — P052 | 1.9 % | 8.3 % | 3.8 % | 16.7 % |
| — Carlito | 1.7 % | 9.1 % | 9.9 % | 33.3 % |
| — URW Bookman | 6.2 % | 31.1 % | 4.1 % | 20.0 % |
| — URW Gothic | 17.8 % | 58.8 % | 7.4 % | 27.5 % |
| real page photo (6 lines) | **11.7 %** | 32.6 % | 36.4 % | 39.5 % |

**Since 0.7.0** (the same reader; beam with a language model, §3b, and the reach
of the accents fixed, §3c): **4.4 %** CER on the 40 lines (Tesseract 5.2 %) —
C059 1.5 %, Carlito 1.2 %, P052 1.9 %, URW Bookman 2.4 %, URW Gothic 12.5 % —
and **9.5 %** on the real photo (Tesseract 36.4 %). The table above is the one from 0.5.0, for
comparison.

Read honestly: a 2.6 MB reader, trained in 24 minutes on a
2-core CPU, comes **close to Tesseract** on fonts it has never seen (better on three
of the five, worse on the geometric URW Gothic, whose single-storey `a` and uncurved
`t` it does not know) and **reads the unevenly lit photo better**, where Tesseract
loses the start of the lines in the shadow. WER is high for both because a single
wrong character brings down the whole word; a beam with a language model
(TODO) is the next gain. Validation during training (592 other lines in the same five fonts,
only monitored — the checkpoint is the one from the last step, not the best on validation):
CER 7.2 %.

The "fluent wrong text" control (each line read as the ground truth of the previous
line) shows that the measure separates *looking like text* from *being the right text*.

## 3b. Language model: choosing among what the page can say, never against it (0.7.0)

Greedy CTC decides each frame alone: "c1áusula", "rão" for "não",
"agerdar". A **character language model** (`Vapor.Vision.CharLM`:
5-grams with interpolated Witten–Bell smoothing — no zero
probability, no discount to tune) enters a **CTC prefix beam
search** (`OCR.ctc_beam/3`; Graves 2006, Hannun et al. 2014): extending a
prefix by `c` adds `log P_frames + 0.8·log P_LM(c | prefix) + 3.0`.

The model *is* its corpus: counted at load time from the
**reference** corpora (the text the reader was trained on), never from the held-out text
it is measured on; `priv/ocr/lm.json` says which ones, the order and the weights.

**Scrutiny found three ways for the language model to turn into noise — each
one became a tested rule:**

1. **Rewriting what is not language.** Without a guard, random strings got worse
   (CER 16.0 % → 22.5 %): the model pushed the reading towards what is common.
   Rule: **the model abstains** when the line's greedy reading costs more than
   8 bits/character under it (held-out prose ≈ 2.6; random strings ≈ 12) — the
   threshold was chosen on a separate random **validation** set.
   Permanent control in the suite: with the guard, 0 of 30 random lines
   change; without it, 27.
2. **Deleting a letter read with certainty.** A synthetic test caught it: "á" read
   at 0.97 was deleted because "clá" is rare in the corpus. Rule: **at each frame
   the model is offered only what the frames find plausible** (probability ≥
   10⁻³, the *blank* included) — a certain letter cannot be deleted, a
   certain space does not become a letter.
3. **Bringing in the corpus's domain.** The corpora are Markdown; a model counted
   with backticks and asterisks **wrote backticks on scanned pages that had
   none**. Rule: the model is counted over the text **as printed**
   (`strip` in `lm.json`). Cost, stated: on the held-out line set — which
   was rendered from the Markdown and contains backticks — the gain drops from
   −51 % to −32 % CER; on the scanned pages, which have no markup, the
   reading improves and no backtick is invented.

Weights (0.8 and 3.0), beam (8) and order (5) were chosen on a
**validation** set (160 lines, another seed, same test fonts), never on the
test set. Measured ([bench/QUALITY.md §5c](bench/QUALITY.md)):

| set | greedy | beam + LM | control |
|---|---|---|---|
| 40 held-out lines (fonts outside training), CER | 6.5 % | **4.4 %** | LM from a shuffled corpus: 7.0 % (no gain: the gain comes from the language) |
| 30 lines with amounts, dates, codes (R$ 85.691,34, AZ-69511, #28129), CER | 8.9 % | **7.4 %** | — (the model cannot make it worse) |
| 30 lines of random strings: lines changed | — | **0** | without the guard: 27 |

In the interface, each character the model chose is highlighted, and a button
shows the frames-only reading — including when the model is wrong (there is a
"retomam" → "retoman" on one of the test pages: it is there to be seen).

## 3c. Reading order: columns (0.7.0)

Before, lines were horizontal strips across the whole page: on a
two-column page each "line" joined the left line with the right one —
**CER 73 %** on a page that, read in order, gives 1.3 %. `Segment.blocks/2` cuts
the page by a **recursive XY-cut** over the components, with a
first-principles rule: **a column gutter (an empty vertical strip that
crosses the whole region) cuts before a horizontal gap**. So a
heading or a footer that spans the columns is separated first (it
blocks the gutter), and then the columns of each strip are read one after
another. The thresholds are relative to the text height **of the region itself** (a
larger heading has larger spaces between words — the interface's first cut
split the heading in half, and that is how we found this), and a
horizontal gap only cuts if it is larger than 1.5× the region's typical gap (double-spaced
text does not become one block per line).

**Scanned pages** (`priv/quality/scans`, `test/python/scan_pages.py`): 8
pages of 1, 2 and 3 columns, one with a heading and footer spanning the columns,
in fonts never seen by the reader, at 200 dpi, binarised like a B&W *scan*,
with dust, **compressed as CCITT inside a PDF** (Group 4, Group 3 2-D,
stencil mask, `BlackIs1`); held-out text in Portuguese and English and, outside
the corpora's domain, legal prose (the Apache-2.0 and MPL-2.0 licences).

| | vapor | without reading order (control) | Tesseract 5.3.4 (eng, psm 3) |
|---|---|---|---|
| mean CER of the 8 pages, PDF → CCITT → blocks → reader → LM | **1.3 %** | 53 % | 1.6 % |

Read honestly: Tesseract is better on the English pages with
"common" fonts (0.0–0.3 %), vapor on the Portuguese pages (the Tesseract here only
has the `eng` model), and vapor's worst page is still the URW Gothic one
(7.5 %), the geometric font the reader does not know.

**A real bug found by looking at the interface**: a line with no ascenders or
descenders ("mesma execução não anunciam a mesma ação") was read as "mesmã
eeeução rão anuneiãm ã mesnã ãção": the line finder accepted a
component at most 3 px from the line's core, and the tilde and the cedilla sit
farther away than that when there are no tall letters to widen the core — the
accents were **discarded**, and the reader saw another word. Now the reach is
0.6× the text height; the line reads correctly (test), and the greedy CER of the held-out
lines fell from 6.8 % to 6.5 % and that of the random strings from 16.0 % to
10.7 % from this alone.

## 3d. CCITT: the format of office scans (0.7.0)

Almost every black-and-white *scanner* PDF stores the page as
`/CCITTFaxDecode`. `Vapor.Docs.CCITT` decodes, with no dependency, the three
encodings (Group 3 1-D, Group 3 2-D, Group 4; T.4/T.6), with `EndOfLine`,
`EncodedByteAlign`, `EndOfBlock`, `BlackIs1`, EOL resynchronisation after
damage — the same state machine as Xpdf/pdf.js. **Checked bit for bit against
libtiff's encoder** (through Pillow): 42 streams — noise at three densities,
a page of text, runs of every length up to 2,600 (extended *make-up*
codes), the three encodings with and without *fill bits*, TIFF's byte-aligned
MH — and, in the same test file, LZW (`EarlyChange` 0 and 1) and
PackBits (`RunLengthDecode`). A 1,700 × 1,000 page decodes in ~1 s
on the BEAM. Garbage and truncated data: never hang, they return the lines that were there.

And another real bug: **stencil masks (`ImageMask`) were read
inverted** (sample 0 is ink; the code inverted it). There was no test; now
there is, for Flate and CCITT.

JBIG2 (the other scan format, with symbol dictionaries and
arithmetic coding): decoded since 0.8.0 (§3f).

## 3e. Tables: structure, cells, column types (0.8.0)

Up to 0.7 a table was read by the XY-cut as columns of text: "Código
Descrição T1 T2 DQ-20430 lado…", with no rows or cells. `Vapor.Vision.Table`
reads the **structure** before the text, and each cell afterwards:

1. **Rule segments** come from the components shaped like rules (long and
   thin, or pieces where the rule broke). A **ruled grid** joins the
   horizontal and vertical segments into rows and columns (virtual borders where
   the outer rule is missing) and decides each **merged cell** by the coverage of the
   rules: a missing divider between two cells merges them (the
   "Trimestre" over T1 and T2, a label that spans two rows).
2. **Rule-only tables** (*booktabs*: a rule at the top, under the header and at the
   end, no verticals) have their columns found by the **gutters** — vertical
   gaps across all the body rows, wider than a space between
   words; digitisation noise does not open a gutter.
3. **The cells are read** by the same reader, with three typed steps per
   column: (a) a column is numeric when ≥ 70 % of its cells can be read
   with only digits and the symbols the column itself shares (`R`, `$`,
   `.`, `,`, `%`…) at a cost of at most 2 nats per character over the best
   free path — and then a "5" read as "õ" goes back to being the best digit; (b) the
   space between digits is either the column's convention ("1 234,56") or noise, and
   the hypothesis whose cells agree most on a shape decides; (c) when the
   majority of the cells have a **shape** (`AA-99999`, `99/99/9999`,
   `R$ 99.999,99` — `Vapor.Vision.Template`), each cell is decoded
   again *within* the shape (CTC Viterbi over the shape's automaton) and the
   result stands if it costs at most 1.5 nats more. The thresholds were chosen
   on a **separate validation set** (seed 2028) and measured on the test set
   (seed 2027).

The output is structure (`rows`, `cols`, `header_rows`, cells with `row`, `col`,
`rowspan`, `colspan`, box, text and confidence) and three renderings: Markdown
(GFM; a multi-row header flattened as "Group / Sub"), HTML (with `rowspan`,
`colspan`, `<thead>`) and CSV. In the OCR text the table takes its place in the
reading order, as Markdown; in the document library each table is a
passage of its own (`#table1`, with HTML and CSV in `meta.table`); in the console, the
table appears drawn from its cells — merged ones, header,
right-aligned numbers, each cell with its confidence and linked to its
box on the page.

**Measured** (`mix vapor.quality` §5d; 12 scanned tables in 4 styles —
grid, inner grid, *booktabs*, horizontal rules —, fonts outside training,
*scanner* noise; `test/python/table_render.py`):

| measure | vapor 0.8 | control |
|---|---:|---|
| structure (ICDAR 2013 adjacency F1) | **1.000** | 0.343 (the 0.7 reading: lines in order) |
| grids with merged cells | **1.000** | 0.905 (merge detection turned off) |
| per-cell CER | **5.5 %** | 11.7 % (free reading of the same cells) |
| per-cell CER, validation | 6.0 % | 12.5 % (free) |
| Tesseract, cells **cropped by hand** (perfect box) | 2.3 % | — |

Tesseract, even when given the perfect boxes that no real reader has,
reads the cells better: vapor's reader is small and trained on lines of
prose. The structure — which Tesseract does not provide — is exact on all 12. **A
visible error**: short isolated header tokens ("T1") sometimes come out wrong
("TP1") and not always with low confidence — the typed steps only apply in the
body.

Out of scope: tables with no rule at all (alignment only), tables that span
pages, cells with several lines of text in rule-only tables.

## 3f. JBIG2: the last scan format (0.8.0; Huffman and halftone in 0.15)

`Vapor.Docs.JBIG2` decodes JBIG2 (ITU-T T.88) with **arithmetic
coding**, with no dependency: the MQ decoder, the IAx/IAID contexts,
generic regions (templates 0–3, adaptive pixels, TPGDON), MMR (through the 0.7
CCITT), refinement (TPGRON), symbol dictionaries (including refinement
aggregates), text regions (the 8 corners/transpositions, strips,
per-instance refinement), striped and unstriped pages, the standalone file and the
PDF embedded mode (`/JBIG2Decode` with `/JBIG2Globals`).

**Checked bit for bit against jbig2dec** on 43 streams: those from jbig2enc
(generic, symbols, PDF with globals) and those from our own Python encoder
(`test/python/jbig2_streams.py`: an MQ encoder ported from jbig2enc and
checked against the standard's example H.2; each stream is validated by jbig2dec
before becoming a fixture). Control: the same streams with the generic template
declared wrong — 19 of 43 still "match" (the symbol/MMR ones do not use the
template), the other 24 do not; a decoder that ignored the template would pass
the naive test. A full 1,700 × 870 page in ~0.5–1 s on the BEAM. A
table scanned into a JBIG2 PDF is read with the same structure and the same text
as the same page in PNG.

Two divergences found between reference implementations, documented
in the code: jbig2dec uses the whole page as the reference for a refinement
with no offset (T.88 7.4.7.4 says the region), and the TPGRON SLTP contexts
of template 1 differ between jbig2dec and pdf.js (we follow jbig2dec, which is
what is checked against here).

**Huffman and halftone (0.15).** What 0.8 refused, because there was no open encoder
that emitted them to check against, was closed from the other side: an
**independent** Python encoder (`test/python/jbig2_streams.py`, sharing no code
with the decoder) emits the streams, and each one is judged by jbig2dec before
becoming a fixture. Now decoded:

- **Huffman coding** (`Vapor.Docs.JBIG2Huffman`): the fifteen standard tables
  B.1–B.15 (with the lower/upper range lines and OOB), user
  tables (segment type 53, built by algorithm B.3), SDHUFF symbol
  dictionaries (heights, widths, aggregation sizes, and each height class's
  collective bitmap, raw or MMR) and SBHUFF text regions (the symbol ID
  table by *run-length* code lengths, strips and
  coordinates, with the user tables in the order FS, DS, DT, …);
- **pattern dictionaries and halftone regions** (types 16, 20, 22, 23): the
  grey-scale planes in Gray code (generic or MMR), the rotated grid,
  `HSKIP` (cells outside the region skipped), the four combinations and
  `HDEFPIXEL`.

21 new fixtures (13 Huffman, 8 halftone), **64 in total**, all bit for bit; the
39 old ones regenerate byte for byte. A defect in **jbig2dec** turned up along the way:
with `HDEFPIXEL = 1` it fills the region with the byte `0x01` (one black pixel in
eight, vertical stripes) instead of black. That fixture is judged by T.88
6.6.5.2 step 1, and the difference (in pixels) is recorded in the manifest. Two other
traps in the specification, respected: table B.2 has negative widths and
B.11 has no `DT = 0`.

**Still refused with a warning**, by name: Huffman **with refinement** (SDREFAGG
or SBREFINE under SDHUFF/SBHUFF), arithmetic contexts **retained** between
segments, and a halftone without its pattern dictionary.


## 3g. Chinese, Japanese and Korean: thousands of classes without a trained network (0.10)

`Vapor.Vision.CJK`. Three facts about the scripts carry the design:

1. **The script is a grid.** Every hanzi, kanji, kana or hangul syllable
   occupies the same square cell, as tall as the line. A character of
   several parts (好 = 女 + 子) is not segmented by its components,
   but by **cells**: the candidate cuts lie in the gaps in the ink, and a
   segment is judged in a line-height cell centred on it (only its own
   columns: a narrow digit does not see its neighbour).
2. **A font is the training set itself.** The class templates are
   rendered from the national standards (GB 2312 level 1: 3,755 hanzi; JIS X
   0208 level 1 + kana; KS X 1001: 2,350 syllables) in several fonts and
   summarised by **directional element features** — the gradient of the
   stroke edges in 8 directions on an 8×8 grid (512 values), the
   representation that printed CJK OCR has relied on since the 1990s,
   because the direction of a stroke survives a change of font much better
   than its pixels. Reading is the nearest template by cosine: a
   `linear` on the worker against the whole class matrix.
3. **Recognition decides segmentation.** Each segmentation of the line into
   cells is scored by how well the cells are recognised, minus a
   fixed price per cell; the best path wins (dynamic programming
   over the gaps; touching runs are cut at the line's pitch).
   A per-character language model (Witten–Bell) reranks only among
   visually plausible candidates, and **abstains** when the greedy
   reading is already improbable for it (> 10 bits/character).

Bundled packages (`priv/ocr-cjk-{zh,ja,ko}`, 8-bit templates: 1.2–1.9
MB each; `CJK.default(:zh | :ja | :ko)`). Measured on lines **rendered
with a new seed, in fonts that never went into the templates**, with the
same noise as the rest of the OCR (blur, noise, lighting, JPEG, rotation):

| language (test fonts) | greedy CER | with the language model | random characters (control) |
|---|---|---|---|
| Chinese (Noto Serif SC, AR PL UKai) | **8.9 %** (Noto Serif 0.0 %; UKai 16.9 %) | 9.2 % | 9.6 % → 9.6 % |
| Japanese (Noto Serif JP, Sawarabi Mincho) | **5.0 %** | **2.6 %** | 0.5 % → 0.5 % |
| Korean (Noto Serif KR, NanumBarunGothic) | **16.0 %** | **11.3 %** | 9.9 % → 9.9 % |

**What scrutiny found:** on the development set (the seed
with which the choices were made), the language model took Chinese from
7.8 % to 2.5 %; on a new seed, **it does not help** (8.9 % → 9.2 %). The earlier
improvement was partly fitting to the set — that is why the numbers above are
the new seed's. The language model's corpus comes from the same Faker word
lists as the test lines (another seed): the gain in
Japanese and Korean is an **in-domain** number. The control
(random characters) proves the other half of the promise: where there is nothing
to exploit, the language model changes nothing.

Kai (brush) is the hard font: without a Kai among the templates, UKai stood
at 46.6 %; with AR PL KaitiM (same lineage as UKai — stated here), 16.9 %.

## 3h. Figures: where they are, what the captions say, what the numbers are (0.10)

`Vapor.Vision.Figure`, and in `OCR.read/2` (a figure's marks no longer
pollute the text; the caption is text).

- **Detection.** Text is made of marks at most ~1.5 text heights tall;
  a chart's frame, a curve, a bar or a photograph's
  blotches are not. A mark 6 heights in size (in both directions) seeds a
  figure; the figure grows over what touches it (axis labels, titles,
  caption), never over a **line of the page's text** that runs past its
  sides; a photograph is extended to where the paper begins again. A
  box that contains lines of text is a framed paragraph, not a figure.
- **Caption**: the nearest line below (or above) that **says** it is one
  ("Figure 3", "Fig. 2.", "Figura 1 —", "Gráfico", "图 2", "図", "그림",
  "شكل"), read from an enlarged crop (captions come in a smaller type size).
- **Chart digitisation**: the frame (two spines at a corner), the
  *tick* marks, the labels read — and **the scale is accepted only if the
  labels agree with one**: a linear or logarithmic pixel-to-value map
  fitted to *all* the labels; more than one in disagreement and the axis is
  **refused**. Another first-principles rule: linear *ticks* fall
  on multiples of the step (0, 20, 40 — never 1, 21, 41); a reading that
  fits a straight line but not this rule is a digit misread in every
  label, and it is refused. A log axis with only two labels (10¹, 10²) is
  checked by the minor *ticks*, which must fall at log₁₀(2…9).
  Small labels are read from enlarged crops, each axis also as
  **one phrase** (all the labels side by side: a lone "4" is a line
  the reader has never seen), and by **glyph consensus**: an
  axis's labels use a single font, so the same digit is the same drawing — the
  glyphs are grouped by shape and each group gets the character the
  reader most often gives it; decimal point and minus sign are decided by shape and
  by position. The series are separated by **the direction of the colour from
  white** (a line's anti-aliased edge is its colour mixed with the
  paper, `W − p = (1 − t)(W − c)`: the direction of `W − p` is that of the colour), and
  classified by the shape of the marks: rectangles with a common base are
  bars (gaps in the grid closed), compact blobs are points (overlapping
  blobs split by k-means), the rest is a line read column by
  column through the ink-weighted centre.

Measured on matplotlib charts that were never used to tune the
digitiser:

| set | within tolerance | refused | read grossly wrong |
|---|---|---|---|
| default style (30) | **27** (median per-column error ≤ 1 % of the axis range; bars ≤ 2 %; points: recall and precision ≥ 0.9) | 2 | 0 |
| hard: serif, grid, log axis, JPEG (30) | 18 | 12 | **0** |
| **control: permuted labels** (12) | — | **12 of 12** | 0 |
| pages (10): figure found, IoU ≥ 0.9; caption | 10/10, right type (chart × photo) in all; captions 10/10, CER < 10 % | | |

**Scrutiny of the captions:** the first full run of the suite found
2 captions missed on the test pages — one **above** the figure with
body text just below it (only the two nearest lines *below* were
looked at), another after an axis title that had been left outside the box; and an
"8" read as "B". Now the four nearest lines on both
sides are looked at, within a band of 8 text heights, and a caption can be numbered
by a letter ("Figure A:", from an appendix; a capital followed by punctuation). The
10/10 above come **after** this fix: for the captions, this set
is no longer blind (stated here).

The hard set shows the limit — 8 px serif digits that the
reader cannot separate —, and it shows the property that matters: **what is not
read is refused, never invented**.

## 3i. Typeset formulas → LaTeX (0.10)

`Vapor.Vision.Math`. What each mark is, and where it is, kept separate:
symbols by the nearest template (edge features in 4 directions and
coverage in a square cell, plus the log of the aspect ratio compared
separately, so that `-`, `=`, `)` and `∫` are not confused); **structure by
geometry** — the widest bar first (numerator above it, within its
span; denominator below), radicals (the hollow mark that encloses others), the
limits of a `∑` (the line above and the one below, which may run past its
sides), and the rest from left to right, each mark on the line of the previous
base or raised (exponent), lowered (subscript) or both, by its
centre relative to the base's **body** (without the descender of a y, μ, β,
nor the ascender of a d, λ or digit). Canonical spelling (`x^{2}`,
`a_{i}^{n}`, `\frac{a}{b}`, `\sqrt{x}`, `\sum_{i=1}^{n}`).

Templates from 6 typeface families (DejaVu Sans/Serif, STIX Sans and sets
with Liberation Serif, FreeSerif, Noto Serif); **measured on Computer Modern
and STIX, which never went in** (60 formulas from a new seed):
**4.9 % error per *token*, 36/60 identical**; the control — the same
symbols read without the geometry — errs more than 3×. Remaining confusions:
Computer Modern's n/π and a/α, 5/6. Outside the grammar (matrices,
accents, `\left…\right`, multiple lines): outside the reader.

## 3j. Arabic, Cyrillic — and what "handwriting" can and cannot mean here (0.10)

**Arabic** (`OCR.default(:arabic)`, `priv/ocr-arabic`): the same CTC reader,
trained on 59 text fonts (all of the system's, minus the four
test ones, the Nastaliq ones and the decorative ones), labels in **visual order** — that of the
frames a column reader sees — and returned to logical order by
`Vapor.Vision.Bidi.logical/1` (reverses the line, re-reverses the
left-to-right runs — numbers with their separators, Latin
words — and undoes the mirroring of parentheses): identical to `python-bidi`
on **all 600 test lines**. The columns of an RTL page are read
from right to left. The letters' dots (ب ت ث ن ي) formed
"lines" of their own and one in four Arabic lines came out split; the
line finder now joins low marks lying over the columns of a
tall neighbour (Latin does not change: the 0.7 suite still passes).

| | CER |
|---|---|
| 4 never-seen fonts (Scheherazade 7.0 %, KacstNaskh 9.2 %, ae_Cortoba 16.1 %, ae_Granada 39.4 %) | **19.2 %** |
| first version, 17 training fonts | 25.7 % |
| control: the Latin reader on the same lines | 91 % |
| Nastaliq (Persian/Urdu calligraphy, never in training) | 57 % |

The Arabeyes test fonts share a lineage with the training ones (stated
here). No language model for Arabic yet.

**Cyrillic** (`priv/ocr-cyrillic`): see §3k, below.

**Handwriting.** No real handwriting dataset is available on this machine
(no network for IAM, KHATT, CASIA-HWDB, the Cyrillic sets); the
substitute was **handwriting fonts**. Trained on 17 "hands", the Latin
cursive reader read the 4 test hands at **64 % CER** (the
print reader: 76 %). By the suite's rule, that is noise — and it **is not
shipped**: `OCR.default(:cursive)` refuses, stating the measurement and the
path (`test/python/train_ocr.py` accepts any set of real
lines; the reader's contract does not change). The same goes for Arabic
"handwriting" (Nastaliq: 57 %), Cyrillic (78 %) and Japanese (§3k): measured and
reported, not promised.

## 3k. Cyrillic, and the "handwriting" measured (0.10)

**Cyrillic** (`OCR.default(:cyrillic)`, `priv/ocr-cyrillic`): the same
CTC reader (91 classes: the Russian alphabet with Ё, digits, punctuation with « »,
— and №), trained 5,000 steps on 11,989 lines in 30 fonts, with the
words of **one half** of the vocabulary (Faker `ru_RU`); the test uses the
other half and four fonts that never went in (Lora, Carlito, URW
Bookman Light, Noto Sans Display):

| | CER |
|---|---|
| 4 never-seen fonts, 60 lines (Noto Sans Display 1.5 %, URW Bookman 2.2 %, Carlito 3.6 %, Lora 4.1 %) | **2.8 %** |
| control: the Latin reader on the same lines | 98 % |

Cyrillic is the easiest of the new scripts: alphabetic, left to
right, no contextual forms — the same problem as Latin with other classes. The
5,000-step training took ~2 h on this 2-core machine (shared with
the suite). On the console page, the first letter of a line pressed against the
edge of the crop sometimes disappears (the test crop has a margin): stated here.

**The "handwriting" requested, measured with the only substitute possible here**
(calligraphic fonts; no real handwriting is available on this machine) — and
none of it promised:

| script | substitute (never in training) | CER | what it says |
|---|---|---|---|
| Latin cursive | 4 handwriting fonts, reader trained on 17 others | 64 % (print: 76 %) | noise → `OCR.default(:cursive)` refuses |
| Arabic | Nastaliq (IranNastaliq, Noto Nastaliq Urdu) | 57 % | a different calligraphy: the baseline descends diagonally |
| Cyrillic | SteveHand 60 %; Klee One and SetoFont (Japanese fonts with Cyrillic) 94 % | 78 % | noise: no hand reader |
| Japanese | Klee One, SetoFont (brush/pen) | **9.8 % → 6.4 %** with the language model | a square of strokes survives the hand; it is still a font, not a hand |

The "handwritten" Japanese is the only useful number — and it is still from **fonts**:
real handwriting (ETL, CASIA-HWDB, KHATT, IAM) varies from person to
person, which no font reproduces. The path is the same for all of them: a
set of real lines, `test/python/train_ocr.py`, and the resulting reader
comes in through the same airlock and is measured by the same suite.

**LaTeX** (requested alongside): §3i — single-line typeset formulas.

## 4. Where the OCR comes in

- **Documents**: PDF pages with no text layer are read (`DCTDecode`
  images, `FlateDecode` with PNG predictors, 1 bit); the passage says
  `meta.ocr` with the confidence. Standalone images (PNG, JPEG) get an
  `#ocr` passage when the reader is confident — **this is how search finds an
  image by what is written in it**: search by meaning for the large
  class of images that carry text (scans, screenshots, slides, photos of a
  whiteboard).
- **Console**, *Vision* panel: the image (or the scanned PDF page) with the
  lines marked, the **blocks numbered in reading order** and the **reading
  thread** that runs from the end of each line to the start of the next; the text
  alongside, grouped by block, with each character tinted by its confidence
  (dotted below 80 %, wavy below 50 %) and the characters the
  **language model chose** highlighted — with a button to see the
  frames-only reading.
- **CLI**: `mix vapor.ocr read file.{png,jpg,pdf}`, `mix vapor.ocr eval DIR
  --tesseract`, `mix vapor.ocr page image.png truth.json`.

## 5. Image search by meaning: what exists and what is missing

- Text inside the image → OCR → text index: **done**.
- The **CLIP** vision tower (`clip_vision_model`, with `visual_projection`) through the
  airlock, **checked against `transformers`** (random weights, ≤ 3·10⁻⁷).
  With a user's CLIP checkpoint, vapor computes the certified
  image embedding.
- CLIP's **text** tower and the BPE tokeniser with `</w>`: **not done** — without
  them there is no text query against image embeddings. It is the next step
  declared in the TODO. With no pretrained weights in this round's environment, no
  semantic-search number for photos without text was measured.

## 6. Limits

- Horizontal printed text. Handwriting and text in perspective or curved
  (`skimage.data.text()`): out. Columns, heading and footer (§3c) and ruled or
  rule-only tables (§3e): done; tables with no rule at all, not.
- Latin alphabet with the Portuguese accents; other alphabets require another
  reader (the contract is the same) — and another corpus for the language model.
- The language model is small (≈ 27,000 words of technical documentation):
  it also helps on out-of-domain legal prose (measured), but a corpus from the
  user's domain would help more; `lm.json` accepts other corpora.
- The reader runs on the native worker (one line per run, ~50–150 ms); on the
  oracle it is exact but slow.
- Scanned PDF: CCITT, LZW, RunLength (§3d) and arithmetic, Huffman and
  halftone JBIG2 (§3f) decoded; JBIG2 Huffman with refinement and JPX: warning.
- **Arabic, CJK, cursive, formulas** (requested in round 0.8): refused — the
  pipeline (geometry, CTC, beam with a language model) is not the limit; the
  limit is a reader trained on those writing systems, with data this
  environment does not have. Changing the strip height to 64 px or the head to
  10,000 classes without training would only produce noise that looks like output
  ([DIRECTIVE.md §11](DIRECTIVE.md)).
