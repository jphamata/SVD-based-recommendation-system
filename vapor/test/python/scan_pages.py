"""Scanned office pages for vapor's reading tests: multi-column layouts,
black and white, CCITT-compressed inside a PDF — what a scanner's
"scan to PDF" produces.

usage: scan_pages.py OUT_DIR [--tesseract]

Every page is typeset here (Pillow/FreeType) in a font vapor's OCR reader
never saw in training (C059, P052, Carlito, URW Bookman, URW Gothic), at
200 dpi with an 11-point body (30 px), thresholded to 1 bit like a
scanner's B&W mode, with a few specks of dust. Layouts: one column (the
control: nothing to reorder), two and three columns, two columns between a
full-width title and a full-width footer. Text: the held-out corpora
(priv/quality/*_holdout.txt — never used to train the reader or the
language model) and, **out of the corpora's domain**, the English legal
prose of the Apache-2.0 and MPL-2.0 licence texts.

The image of each page is put in a one-page PDF as a CCITT stream encoded
by libtiff — Group 4, Group 3 2-D, as a stencil mask, with BlackIs1 — so
the whole path (PDF → CCITT → layout → reader) is what gets measured.

Writes OUT_DIR/<name>.pdf and OUT_DIR/truth.json: {name: {layout, font,
source, coding, blocks: [[line, …], …]}} — blocks in reading order, lines in
order. With --tesseract, also OUT_DIR/tesseract.json: Tesseract's reading of
the same page images (psm 3: automatic page segmentation, which handles
columns), frozen so the comparison runs without Tesseract.
"""
import io, json, os, random, re, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, TiffImagePlugin

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = random.Random(2026)
here = os.path.dirname(os.path.abspath(__file__))
priv = os.path.join(here, "..", "..", "priv", "quality")
TiffImagePlugin.WRITE_LIBTIFF = True

TT = "/usr/share/fonts/truetype/"
OT = "/usr/share/fonts/opentype/urw-base35/"
FONTS = {"C059": OT + "C059-Roman.otf", "P052": OT + "P052-Roman.otf", "Carlito": TT + "crosextra/Carlito-Regular.ttf",
         "URWBookman": OT + "URWBookman-Light.otf", "URWGothic": OT + "URWGothic-Book.otf"}
CHARSET = set(chr(c) for c in range(33, 127) if chr(c) != '"') | set("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÇ") | {" "}


def words_of(text):
    text = re.sub(r"[`*#|_\[\]<>{}]", " ", text)
    return ["".join(ch for ch in w if ch in CHARSET) for w in text.split() if len(w) < 24]


def corpus(name):
    return [w for w in words_of(open(os.path.join(priv, name), encoding="utf-8").read()) if w]


def licence(name, skip):
    t = open("/usr/share/common-licenses/" + name, encoding="utf-8").read()
    return [w for w in words_of(t) if w and not re.fullmatch(r"[-=.]+", w)][skip:]


SOURCES = {"pt_holdout": corpus("pt_holdout.txt"), "en_holdout": corpus("en_holdout.txt"),
           "apache": licence("Apache-2.0", 40), "mpl": licence("MPL-2.0", 30)}


def wrap(words, font, width, nlines, start):
    lines, cur, i = [], [], start
    while len(lines) < nlines and i < len(words):
        cand = " ".join(cur + [words[i]])
        if font.getlength(cand) <= width or not cur:
            cur.append(words[i]); i += 1
        else:
            lines.append(" ".join(cur)); cur = []
    if cur and len(lines) < nlines:
        lines.append(" ".join(cur))
    return lines, i


# (name, layout, font, source, coding)
PAGES = [("p1_one_col", "1col", "C059", "pt_holdout", "g4"),
         ("p2_two_col", "2col", "P052", "pt_holdout", "g4"),
         ("p3_three_col", "3col", "Carlito", "en_holdout", "g3_2d"),
         ("p4_two_col_title", "2col_title_footer", "URWBookman", "pt_holdout", "mask"),
         ("p5_two_col", "2col", "C059", "en_holdout", "black_is_1"),
         ("p6_legal_two_col", "2col", "P052", "apache", "g4"),
         ("p7_legal_one_col", "1col", "Carlito", "mpl", "g4"),
         ("p8_three_col", "3col", "URWGothic", "pt_holdout", "g4")]

W, MARGIN, GUTTER, SIZE, LEAD = 1700, 110, 70, 30, 44
truth, cursor = {}, {k: rng.randrange(0, 200) for k in SOURCES}

for name, layout, fname, source, coding in PAGES:
    font = ImageFont.truetype(FONTS[fname], SIZE)
    big = ImageFont.truetype(FONTS[fname], 42)
    words = SOURCES[source]
    ncols = {"1col": 1, "2col": 2, "3col": 3, "2col_title_footer": 2}[layout]
    nlines = {"1col": 14, "2col": 13, "3col": 12, "2col_title_footer": 10}[layout]
    colw = (W - 2 * MARGIN - (ncols - 1) * GUTTER) // ncols
    blocks, y0 = [], 110
    title = None
    if layout == "2col_title_footer":
        title, cursor[source] = wrap(words, big, W - 2 * MARGIN, 1, cursor[source])
        blocks.append(title)
        y0 = 110 + 90
    cols = []
    for c in range(ncols):
        lines, cursor[source] = wrap(words, font, colw, nlines, cursor[source])
        cols.append(lines)
        blocks.append(lines)
    footer = None
    if layout == "2col_title_footer":
        footer, cursor[source] = wrap(words, font, W - 2 * MARGIN, 2, cursor[source])
        blocks.append(footer)
    H = y0 + nlines * LEAD + (150 if footer else 60) + 60
    im = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(im)
    if title:
        dr.text((MARGIN, 110), title[0], font=big, fill=0)
    for c, lines in enumerate(cols):
        x = MARGIN + c * (colw + GUTTER)
        for i, l in enumerate(lines):
            dr.text((x, y0 + i * LEAD), l, font=font, fill=0)
    if footer:
        fy = y0 + nlines * LEAD + 60
        for i, l in enumerate(footer):
            dr.text((MARGIN, fy + i * LEAD), l, font=font, fill=0)
    a = np.asarray(im) < 140
    # dust: a few isolated specks of 1–2 px
    nr = np.random.default_rng(rng.randrange(1 << 30))
    for _ in range(25):
        y, x = nr.integers(0, H - 2), nr.integers(0, W - 2)
        a[y:y + nr.integers(1, 3), x:x + nr.integers(1, 3)] = True
    # the CCITT stream (ink coded as black runs, as a scanner's)
    tif = Image.fromarray(np.where(a, 255, 0).astype(np.uint8)).convert("1")
    comp, t4 = {"g4": ("group4", None), "g3_2d": ("group3", 1), "mask": ("group4", None), "black_is_1": ("group4", None)}[coding]
    info = {278: H}
    if t4 is not None:
        info[292] = t4
    b = io.BytesIO()
    tif.save(b, "TIFF", compression=comp, tiffinfo=info)
    t = Image.open(io.BytesIO(b.getvalue()))
    assert (np.asarray(t.convert("L")) >= 128).tolist() == a.tolist()
    stream = b.getvalue()[t.tag_v2[273][0]:t.tag_v2[273][0] + t.tag_v2[279][0]]
    k = 1 if coding == "g3_2d" else -1
    if coding == "black_is_1":
        # the same runs read with BlackIs1: 1 = black; Decode [1 0] turns it back
        parms = f"<< /K {k} /Columns {W} /Rows {H} /BlackIs1 true >>"
        img = f"/ColorSpace /DeviceGray /BitsPerComponent 1 /Decode [1 0]"
    elif coding == "mask":
        parms = f"<< /K {k} /Columns {W} /Rows {H} >>"
        img = "/ImageMask true"
    else:
        parms = f"<< /K {k} /Columns {W} /Rows {H} >>"
        img = "/ColorSpace /DeviceGray /BitsPerComponent 1"
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W * 72 // 200} {H * 72 // 200}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode(),
            f"<< /Type /XObject /Subtype /Image /Width {W} /Height {H} {img} /Filter /CCITTFaxDecode /DecodeParms {parms} /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
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
    truth[name] = {"layout": layout, "font": fname, "source": source, "coding": coding, "size": [W, H], "blocks": blocks}
    if "--tesseract" in sys.argv:
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            Image.fromarray(np.where(a, 0, 255).astype(np.uint8)).save(f.name)
            r = subprocess.run(["tesseract", f.name, "-", "--psm", "3", "-l", "eng"], capture_output=True, text=True)
            truth[name]["_tess"] = "\n".join(l.strip() for l in r.stdout.split("\n") if l.strip())

tess = {n: t.pop("_tess") for n, t in truth.items() if "_tess" in t}
json.dump(truth, open(os.path.join(out, "truth.json"), "w"), ensure_ascii=False, indent=1)
if tess:
    version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0].strip()
    json.dump({"tesseract": version, "readings": tess}, open(os.path.join(out, "tesseract.json"), "w"), ensure_ascii=False, indent=1)
print(len(truth), "pages")
