"""CCITT fax fixtures for vapor's decoder, encoded by libtiff (through Pillow).

usage: ccitt_fixtures.py OUT_DIR

For every image × coding, writes NAME.bin (the TIFF strip: the raw CCITT
stream), NAME.raw (the expected bitmap, rows packed MSB-first, 1 = black)
and an entry in index.json: {name: {columns, rows, k, byte_align}}.
Codings: Group 4 (K < 0); Group 3 1-D (K = 0) and 2-D (K > 0), each with and
without fill bits (T4Options bit 2: EOLs end on a byte boundary); TIFF's
CCITT RLE (Modified Huffman, every row byte-aligned, no EOL). Images:
random noise at three densities, a rendered text page, stripes whose runs
cover every length 0–2600 (make-up and extended make-up codes), all white,
all black, odd widths.
"""
import io, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, TiffImagePlugin

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
TiffImagePlugin.WRITE_LIBTIFF = True
rng = np.random.default_rng(7)

images = {}
for d, (h, w) in zip([0.05, 0.3, 0.6], [(37, 101), (64, 64), (23, 333)]):
    images[f"noise{int(d*100)}"] = rng.random((h, w)) < d
# runs of every length: row i alternates white and black runs of lengths i, i+1, …
w = 2700
rows = []
for start in range(0, 2601, 37):
    r = np.zeros(w, bool); x = 0; c = False; L = start
    while x < w:
        r[x:x + L] = c; x += max(L, 1) if L else 1; c = not c; L = (L * 7 + 13) % 2601
    rows.append(r)
images["runs"] = np.array(rows)
images["white"] = np.zeros((9, 1728), bool)
images["black"] = np.ones((9, 1728), bool)
# a printed page at 200 dpi: what an office scanner produces
page = Image.new("L", (1240, 420), 255)
dr = ImageDraw.Draw(page)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 26)
txt = ["CONTRATO DE PRESTAÇÃO DE SERVIÇOS — cláusula 7ª", "The parties agree that the fee of R$ 12.500,00 is due",
       "within thirty (30) days of the invoice, under penalty of", "a fine of 2% and interest of 1% per month."]
for i, t in enumerate(txt):
    dr.text((60, 40 + i * 80), t, font=font, fill=0)
images["page"] = np.asarray(page) < 128

codings = {"g4": ("group4", None, -1, False), "g3": ("group3", 0, 0, False), "g3fill": ("group3", 4, 0, True),
           "g3_2d": ("group3", 1, 1, False), "g3_2dfill": ("group3", 5, 1, True),
           # TIFF's CCITT RLE: Modified Huffman rows, each byte-aligned, no EOLs
           "mh_aligned": ("tiff_ccitt", None, 0, True)}
index = {}
for iname, a in images.items():
    h, w = a.shape
    # libtiff's fax coder codes 1-bits as black runs; Pillow's mode "1" sets
    # 1 for white (MinIsBlack): ink is given as 1-bits so that the stream
    # codes ink as black, as a scanner's does
    im = Image.fromarray(np.where(a, 255, 0).astype(np.uint8)).convert("1")
    for cname, (comp, t4, k, fill) in codings.items():
        info = {278: h}
        if t4 is not None:
            info[292] = t4
        b = io.BytesIO()
        im.save(b, "TIFF", compression=comp, tiffinfo=info)
        t = Image.open(io.BytesIO(b.getvalue()))
        offs, counts = t.tag_v2[273], t.tag_v2[279]
        assert len(offs) == 1, (iname, cname, offs)
        # libtiff's own reading must give the image back
        assert (np.asarray(t.convert("L")) >= 128).tolist() == a.tolist(), (iname, cname)
        raw = b.getvalue()[offs[0]:offs[0] + counts[0]]
        name = f"{iname}_{cname}"
        open(os.path.join(out, name + ".bin"), "wb").write(raw)
        open(os.path.join(out, name + ".raw"), "wb").write(np.packbits(a, axis=1).tobytes())
        index[name] = {"columns": w, "rows": h, "k": k, "byte_align": comp == "tiff_ccitt"}

# ---- LZW (TIFF's = PDF's with EarlyChange 1) and PackBits (= RunLengthDecode),
# on 8-bit images; and LZW with EarlyChange 0 from an independent encoder here
def lzw_encode(data, early):
    out, acc, nbits = bytearray(), 0, 0
    def put(code, width):
        nonlocal acc, nbits
        acc = (acc << width) | code; nbits += width
        while nbits >= 8:
            nbits -= 8; out.append((acc >> nbits) & 255)
    table, nxt, width, w = {bytes([i]): i for i in range(256)}, 258, 9, b""
    put(256, width)
    for c in data:
        wc = w + bytes([c])
        if wc in table:
            w = wc
            continue
        put(table[w], width)
        if nxt < 4096:
            table[wc] = nxt; nxt += 1
            if nxt + early > (1 << width) and width < 12:
                width += 1
        else:
            put(256, width); table = {bytes([i]): i for i in range(256)}; nxt, width = 258, 9
        w = bytes([c])
    if w:
        put(table[w], width)
    put(257, width)
    if nbits:
        out.append((acc << (8 - nbits)) & 255)
    return bytes(out)

grads = {"gradient": (np.add.outer(np.arange(90), np.arange(130)) % 256).astype(np.uint8),
         "noise8": rng.integers(0, 256, (40, 77)).astype(np.uint8),
         "flat": np.full((50, 300), 200, np.uint8)}
for iname, a in grads.items():
    for comp in ["tiff_lzw", "packbits"]:
        b = io.BytesIO()
        Image.fromarray(a).save(b, "TIFF", compression=comp, tiffinfo={278: a.shape[0]})
        t = Image.open(io.BytesIO(b.getvalue()))
        o, n = t.tag_v2[273][0], t.tag_v2[279][0]
        name = f"{iname}_{'lzw' if comp == 'tiff_lzw' else 'rle'}"
        open(os.path.join(out, name + ".bin"), "wb").write(b.getvalue()[o:o + n])
        open(os.path.join(out, name + ".raw"), "wb").write(a.tobytes())
        index[name] = {"filter": "lzw" if comp == "tiff_lzw" else "rle", "early": 1}
    name = f"{iname}_lzw0"
    big = np.tile(a, (3, 3)).tobytes()  # long enough to fill the table and clear it
    open(os.path.join(out, name + ".bin"), "wb").write(lzw_encode(big, 0))
    open(os.path.join(out, name + ".raw"), "wb").write(big)
    index[name] = {"filter": "lzw", "early": 0}
json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
print(len(index))
