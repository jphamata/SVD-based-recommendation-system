"""Scanned tables for vapor's table-structure tests (docs/OCR.md §3e).

usage: table_render.py OUT_DIR [--pdf] [--seed N]

The default seed (2027) makes the test set (priv/quality/tables); seed 2028
makes the validation set (priv/quality/tables_val) on which the decoding
gates were chosen — the test set was never used to choose anything.

Every page holds a short paragraph, a table, and another paragraph, typeset
here (Pillow/FreeType) in a font the OCR reader never saw in training (C059,
P052, Carlito, URW Bookman, URW Gothic), 200 dpi, a 10-point body (28 px),
thresholded to 1 bit like a scanner's B&W mode, with dust. The four table
styles of real documents:

  grid      every cell boxed (invoices, forms); a two-row header with a cell
            spanning two columns and one spanning both header rows
  inner     lines between cells but no outer frame (statements, reports)
  booktabs  three horizontal rules — top, under the header, bottom — and no
            vertical line (scientific papers); a group header spans columns
  hrules    a horizontal rule under every row, no vertical line (bank
            statements, price lists)

Cell contents are what tables hold: words of the held-out corpora, money
(R$ 1.234,56 / $1,234.56), dates, IDs, percentages, integers — numbers
right-aligned, text left-aligned.

Writes OUT_DIR/<name>.png (1-bit) and OUT_DIR/truth.json: {name: {style,
font, size, before: [line], after: [line], table: {box, rows, cols,
header_rows, cells: [{row, col, rowspan, colspan, text, box}]}}}. With
--pdf, also OUT_DIR/<name>.pdf: the page as a CCITT Group 4 stream inside a
one-page PDF (what a scanner's "scan to PDF" produces). With --tesseract,
also OUT_DIR/tesseract.json: Tesseract's reading of every truth cell, each
cropped from the page by its true box (inset past the rules) and read as
one line (psm 7) — Tesseract given a *perfect* table structure, a reference
for the cells' text; frozen so the comparison runs without Tesseract.
"""
import io, json, os, random, re, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, TiffImagePlugin

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 2027
rng = random.Random(SEED)
here = os.path.dirname(os.path.abspath(__file__))
priv = os.path.join(here, "..", "..", "priv", "quality")
TiffImagePlugin.WRITE_LIBTIFF = True

TT = "/usr/share/fonts/truetype/"
OT = "/usr/share/fonts/opentype/urw-base35/"
FONTS = {"C059": OT + "C059-Roman.otf", "P052": OT + "P052-Roman.otf", "Carlito": TT + "crosextra/Carlito-Regular.ttf",
         "URWBookman": OT + "URWBookman-Light.otf", "URWGothic": OT + "URWGothic-Book.otf"}
CHARSET = set(chr(c) for c in range(33, 127) if chr(c) != '"') | set("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÇ") | {" "}


def words_of(text):
    text = re.sub(r"[`*#|_\[\]<>{}()]", " ", text)
    return ["".join(ch for ch in w if ch in CHARSET).strip(".,;:") for w in text.split() if 2 < len(w) < 13]


WORDS = {"pt": [w for w in words_of(open(os.path.join(priv, "pt_holdout.txt"), encoding="utf-8").read()) if w.isalpha()],
         "en": [w for w in words_of(open(os.path.join(priv, "en_holdout.txt"), encoding="utf-8").read()) if w.isalpha()]}


def money(lang):
    v = rng.randrange(100, 9_999_999) / 100
    if lang == "pt":
        s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return "R$ " + s
    return f"${v:,.2f}"


def cell_value(kind, lang):
    if kind == "text":
        return " ".join(rng.choice(WORDS[lang]) for _ in range(rng.choice([1, 1, 2])))
    if kind == "money":
        return money(lang)
    if kind == "date":
        d, m, y = rng.randrange(1, 29), rng.randrange(1, 13), rng.randrange(2019, 2027)
        return f"{d:02d}/{m:02d}/{y}" if lang == "pt" else f"{y}-{m:02d}-{d:02d}"
    if kind == "id":
        return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2)) + "-" + str(rng.randrange(10000, 99999))
    if kind == "pct":
        v = rng.randrange(1, 9999) / 10
        return (f"{v:.1f}".replace(".", ",") if lang == "pt" else f"{v:.1f}") + "%"
    return str(rng.randrange(1, 99999))


HEAD = {"pt": {"text": "Descrição", "money": "Valor", "date": "Data", "id": "Código", "pct": "Taxa", "int": "Qtd"},
        "en": {"text": "Item", "money": "Amount", "date": "Date", "id": "Code", "pct": "Rate", "int": "Count"}}

# (name, style, font, lang, column kinds, body rows)
TABLES = [("t01_grid", "grid", "C059", "pt", ["id", "text", "money", "money"], 5),
          ("t02_grid", "grid", "Carlito", "en", ["date", "text", "int", "int", "money"], 6),
          ("t03_inner", "inner", "P052", "pt", ["text", "date", "pct"], 6),
          ("t04_inner", "inner", "URWBookman", "en", ["id", "text", "money", "pct"], 5),
          ("t05_booktabs", "booktabs", "C059", "en", ["text", "int", "int", "pct"], 6),
          ("t06_booktabs", "booktabs", "Carlito", "pt", ["text", "money", "money"], 7),
          ("t07_hrules", "hrules", "P052", "pt", ["date", "text", "money"], 6),
          ("t08_hrules", "hrules", "URWGothic", "en", ["date", "id", "text", "money"], 5),
          ("t09_grid", "grid", "URWGothic", "pt", ["text", "int", "money"], 5),
          ("t10_booktabs", "booktabs", "URWBookman", "pt", ["id", "text", "date", "money"], 5),
          ("t11_inner", "inner", "Carlito", "en", ["text", "money", "money", "money"], 4),
          ("t12_hrules", "hrules", "C059", "en", ["id", "text", "int", "money"], 7)]

W, MARGIN, SIZE, LEAD, PADX, PADY = 1700, 110, 28, 42, 22, 14


def prose(lang, font, n):
    lines, cur = [], []
    while len(lines) < n:
        w = rng.choice(WORDS[lang])
        if font.getlength(" ".join(cur + [w])) > W - 2 * MARGIN and cur:
            lines.append(" ".join(cur)); cur = []
        cur.append(w)
    return lines


truth = {}
for name, style, fname, lang, kinds, nbody in TABLES:
    font = ImageFont.truetype(FONTS[fname], SIZE)
    ncol = len(kinds)
    # header: one row; grid and booktabs get a second header row with a
    # group cell spanning the two last columns (and, in grid, a first
    # header cell spanning both rows)
    two = style in ("grid", "booktabs")
    head = [HEAD[lang][k] for k in kinds]
    if two:
        group = "Trimestre" if lang == "pt" else "Quarter"
        sub = ["T1", "T2"] if lang == "pt" else ["Q1", "Q2"]
    body = [[cell_value(k, lang) for k in kinds] for _ in range(nbody)]
    # logical cells: (row, col, rowspan, colspan, text)
    cells = []
    if two:
        for c in range(ncol - 2):
            if style == "grid" and c == 0:
                cells.append([0, 0, 2, 1, head[0]])
            elif style == "grid":
                cells.append([0, c, 2, 1, head[c]])
            else:
                cells.append([1, c, 1, 1, head[c]])
        cells.append([0, ncol - 2, 1, 2, group])
        cells.append([1, ncol - 2, 1, 1, sub[0]])
        cells.append([1, ncol - 1, 1, 1, sub[1]])
        hrows = 2
    else:
        for c in range(ncol):
            cells.append([0, c, 1, 1, head[c]])
        hrows = 1
    for r, row in enumerate(body):
        for c, v in enumerate(row):
            cells.append([hrows + r, c, 1, 1, v])
    nrows = hrows + nbody

    # column widths from content, row heights from the lead
    colw = [0] * ncol
    for r0, c0, rs, cs, t in cells:
        if cs == 1:
            colw[c0] = max(colw[c0], int(font.getlength(t)) + 2 * PADX)
    gap = 0 if style in ("grid", "inner") else 2 * PADX
    tw = sum(colw) + gap * (ncol - 1)
    rowh = SIZE + 2 * PADY
    before = prose(lang, font, 2)
    after = prose(lang, font, 2)
    x0 = MARGIN + rng.randrange(0, max(1, (W - 2 * MARGIN - tw) // 2))
    y_table = 110 + len(before) * LEAD + 40
    th = nrows * rowh
    H = y_table + th + 50 + len(after) * LEAD + 110
    im = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(im)
    for i, l in enumerate(before):
        dr.text((MARGIN, 110 + i * LEAD), l, font=font, fill=0)
    xs = [x0]
    for c in range(ncol):
        xs.append(xs[-1] + colw[c] + (gap if c < ncol - 1 else 0))
    ys = [y_table + r * rowh for r in range(nrows + 1)]
    lw = 3
    # rules
    def hline(y, xa, xb):
        dr.rectangle([xa, y - lw // 2, xb, y - lw // 2 + lw - 1], fill=0)
    def vline(x, ya, yb):
        dr.rectangle([x - lw // 2, ya, x - lw // 2 + lw - 1, yb], fill=0)
    occupied = {}
    for r0, c0, rs, cs, t in cells:
        for r in range(r0, r0 + rs):
            for c in range(c0, c0 + cs):
                occupied[(r, c)] = (r0, c0, rs, cs)
    if style in ("grid", "inner"):
        for r in range(nrows + 1):
            if style == "inner" and r in (0, nrows):
                continue
            for c in range(ncol):
                above = occupied.get((r - 1, c)); below = occupied.get((r, c))
                if r not in (0, nrows) and above == below:
                    continue  # inside a cell spanning rows
                hline(ys[r], xs[c], xs[c + 1])
        for c in range(ncol + 1):
            if style == "inner" and c in (0, ncol):
                continue
            for r in range(nrows):
                left = occupied.get((r, c - 1)); right = occupied.get((r, c))
                if c not in (0, ncol) and left == right:
                    continue
                vline(xs[c], ys[r], ys[r + 1])
    elif style == "booktabs":
        hline(ys[0], xs[0], xs[-1]); hline(ys[hrows], xs[0], xs[-1]); hline(ys[-1], xs[0], xs[-1])
        # the group header's own short rule (\cmidrule)
        hline(ys[1] - 4, xs[ncol - 2] + 6, xs[ncol] - 6)
    elif style == "hrules":
        for r in range(nrows + 1):
            hline(ys[r], xs[0], xs[-1])
    # text: numbers right-aligned, text left; spanning cells centred
    out_cells = []
    num = re.compile(r"^[R$\d][\d$.,% -]*$")
    for r0, c0, rs, cs, t in cells:
        cx0, cx1 = xs[c0], xs[c0 + cs] - (gap if c0 + cs < ncol else 0)
        cy0, cy1 = ys[r0], ys[r0 + rs]
        tl = font.getlength(t)
        if cs > 1:
            tx = (cx0 + cx1 - tl) / 2
        elif num.match(t) and r0 >= hrows:
            tx = cx1 - PADX - tl
        else:
            tx = cx0 + PADX
        ty = (cy0 + cy1) / 2 - SIZE * 0.62
        dr.text((tx, ty), t, font=font, fill=0)
        out_cells.append({"row": r0, "col": c0, "rowspan": rs, "colspan": cs, "text": t, "box": [cx0, cy0, cx1, cy1]})
    ya = y_table + th + 50
    for i, l in enumerate(after):
        dr.text((MARGIN, ya + i * LEAD), l, font=font, fill=0)

    a = np.asarray(im) < 140
    nr = np.random.default_rng(rng.randrange(1 << 30))
    for _ in range(25):
        y, x = nr.integers(0, H - 2), nr.integers(0, W - 2)
        a[y:y + nr.integers(1, 3), x:x + nr.integers(1, 3)] = True
    page = Image.fromarray(np.where(a, 0, 255).astype(np.uint8)).convert("1")
    page.save(os.path.join(out, name + ".png"), optimize=True)
    truth[name] = {"style": style, "font": fname, "lang": lang, "size": [W, H], "before": before, "after": after,
                   "table": {"box": [xs[0], ys[0], xs[-1], ys[-1]], "rows": nrows, "cols": ncol, "header_rows": hrows,
                             "cells": sorted(out_cells, key=lambda c: (c["row"], c["col"]))}}

    if "--pdf" in sys.argv and name in ("t01_grid", "t05_booktabs"):
        tif = Image.fromarray(np.where(a, 255, 0).astype(np.uint8)).convert("1")
        b = io.BytesIO()
        tif.save(b, "TIFF", compression="group4", tiffinfo={278: H})
        t = Image.open(io.BytesIO(b.getvalue()))
        stream = b.getvalue()[t.tag_v2[273][0]:t.tag_v2[273][0] + t.tag_v2[279][0]]
        parms = f"<< /K -1 /Columns {W} /Rows {H} >>"
        objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W * 72 // 200} {H * 72 // 200}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode(),
                f"<< /Type /XObject /Subtype /Image /Width {W} /Height {H} /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode /DecodeParms {parms} /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
                None]
        content = f"q {W * 72 // 200} 0 0 {H * 72 // 200} 0 0 cm /Im0 Do Q".encode()
        objs[4] = f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        pdf, offs = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), []
        for i, o in enumerate(objs, 1):
            offs.append(len(pdf))
            pdf += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
        xref = len(pdf)
        pdf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
        pdf += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        open(os.path.join(out, name + ".pdf"), "wb").write(bytes(pdf))

json.dump(truth, open(os.path.join(out, "truth.json"), "w"), ensure_ascii=False, indent=1)
if "--tesseract" in sys.argv:
    readings = {}
    for name, t in truth.items():
        page = Image.open(os.path.join(out, name + ".png")).convert("L")
        cells = []
        for c in t["table"]["cells"]:
            x0, y0, x1, y1 = c["box"]
            crop = page.crop((int(x0) + 5, int(y0) + 5, int(x1) - 4, int(y1) - 4))
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                crop.save(f.name)
                r = subprocess.run(["tesseract", f.name, "-", "--psm", "7", "-l", "eng"], capture_output=True, text=True)
            cells.append(r.stdout.strip())
        readings[name] = cells
    version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0].strip()
    json.dump({"tesseract": version, "cells": readings}, open(os.path.join(out, "tesseract.json"), "w"), ensure_ascii=False, indent=1)
print(len(truth), "tables, seed", SEED)
