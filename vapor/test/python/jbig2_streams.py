"""JBIG2 streams for vapor's decoder tests (test/vapor/jbig2_test.exs).

usage: jbig2_streams.py OUT_DIR [--jbig2enc PATH]

Two kinds of streams, both judged by the reference decoder, jbig2dec:

1. **Written here**, for every part of T.88 no encoder in reach exercises
   (jbig2enc only writes generic template 0 and plain text regions; its
   refinement is disabled upstream): generic templates 0–3 with nominal
   and moved adaptive pixels, with and without typical prediction; MMR
   generic regions (Group 4 by libtiff); refinement regions (templates 0
   and 1, TPGRON); text regions in all eight reference-corner/transposed
   combinations, with strips, DSOFFSET, XOR composition, a black default
   pixel and refined instances; symbol dictionaries with refinement and
   aggregation; a striped page of unknown height; and **Huffman coding**
   (Annex B, written here from the standard): symbol dictionaries with
   SDHUFF (B.2–B.5 and custom tables, uncompressed and MMR collective
   bitmaps), text regions with SBHUFF (B.6–B.13 and custom tables, the
   run-coded symbol ID table with codes 32–34, values on the lower and
   upper range lines), custom code table segments. The MQ coder is
   jbig2enc's, transcribed (`jbig2arith.cc`), and checked on the T.88 H.2
   test sequence.

2. **Written by jbig2enc** (when --jbig2enc is given): a scanned page in
   generic mode, with TPGD, in symbol mode, and in PDF mode (globals +
   page) — what real PDFs carry.

For every stream, OUT_DIR/<name>.jb2 (or .sym + .0000 for PDF mode, and
enc_pdf.pdf: the page in a PDF with JBIG2Decode and JBIG2Globals). A
stream is kept only if **jbig2dec decodes it to exactly the bitmap this
script composed** — so the fixtures do not rest on this script's reading
of the standard alone. Writes OUT_DIR/manifest.json: {name: {files, w, h,
sha256 of the bitmap as a binary PBM}}.
"""
import hashlib, io, json, os, random, subprocess, sys
import numpy as np
from PIL import Image, TiffImagePlugin

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = random.Random(88)
TiffImagePlugin.WRITE_LIBTIFF = True

# ------------------------------------------------------------- MQ encoder --
# (Qe, NMPS, NLPS, SWITCH), Table E.1
QE = [(0x5601, 1, 1, 1), (0x3401, 2, 6, 0), (0x1801, 3, 9, 0), (0x0AC1, 4, 12, 0), (0x0521, 5, 29, 0), (0x0221, 38, 33, 0),
      (0x5601, 7, 6, 1), (0x5401, 8, 14, 0), (0x4801, 9, 14, 0), (0x3801, 10, 14, 0), (0x3001, 11, 17, 0), (0x2401, 12, 18, 0),
      (0x1C01, 13, 20, 0), (0x1601, 29, 21, 0), (0x5601, 15, 14, 1), (0x5401, 16, 14, 0), (0x5101, 17, 15, 0), (0x4801, 18, 16, 0),
      (0x3801, 19, 17, 0), (0x3401, 20, 18, 0), (0x3001, 21, 19, 0), (0x2801, 22, 19, 0), (0x2401, 23, 20, 0), (0x2201, 24, 21, 0),
      (0x1C01, 25, 22, 0), (0x1801, 26, 23, 0), (0x1601, 27, 24, 0), (0x1401, 28, 25, 0), (0x1201, 29, 26, 0), (0x1101, 30, 27, 0),
      (0x0AC1, 31, 28, 0), (0x09C1, 32, 29, 0), (0x08A1, 33, 30, 0), (0x0521, 34, 31, 0), (0x0441, 35, 32, 0), (0x02A1, 36, 33, 0),
      (0x0221, 37, 34, 0), (0x0141, 38, 35, 0), (0x0111, 39, 36, 0), (0x0085, 40, 37, 0), (0x0049, 41, 38, 0), (0x0025, 42, 39, 0),
      (0x0015, 43, 40, 0), (0x0009, 44, 41, 0), (0x0005, 45, 42, 0), (0x0001, 45, 43, 0), (0x5601, 46, 46, 0)]


class MQ:
    def __init__(self):
        self.a, self.c, self.ct, self.bp, self.b = 0x8000, 0, 12, -1, 0
        self.out = bytearray()

    def byteout(self):
        if self.b == 0xFF:
            return self._r()
        if self.c < 0x8000000:
            return self._l()
        self.b += 1
        if self.b != 0xFF:
            return self._l()
        self.c &= 0x7FFFFFF
        self._r()

    def _r(self):
        if self.bp >= 0:
            self.out.append(self.b)
        self.b = (self.c >> 20) & 0xFF  # u8, as in jbig2enc: the carry was propagated into the byte before
        self.bp += 1
        self.c &= 0xFFFFF
        self.ct = 7

    def _l(self):
        if self.bp >= 0:
            self.out.append(self.b)
        self.b = (self.c >> 19) & 0xFF
        self.bp += 1
        self.c &= 0x7FFFF
        self.ct = 8

    def bit(self, ctx, cx, d):
        i, mps = ctx.get(cx, (0, 0))
        qe, nmps, nlps, sw = QE[i]
        if d == mps:
            self.a -= qe
            if self.a & 0x8000:
                self.c += qe
                return
            if self.a < qe:
                self.a = qe
            else:
                self.c += qe
            ctx[cx] = (nmps, mps)
        else:
            self.a -= qe
            if self.a < qe:
                self.c += qe
            else:
                self.a = qe
            ctx[cx] = (nlps, 1 - mps if sw else mps)
        while True:
            self.a = (self.a << 1) & 0xFFFF
            self.c = (self.c << 1) & 0xFFFFFFFF
            self.ct -= 1
            if self.ct == 0:
                self.byteout()
            if self.a & 0x8000:
                break

    def final(self):
        tempc = self.c + self.a
        self.c |= 0xFFFF
        if self.c >= tempc:
            self.c -= 0x8000
        self.c = (self.c << self.ct) & 0xFFFFFFFF
        self.byteout()
        self.c = (self.c << self.ct) & 0xFFFFFFFF
        self.byteout()
        self.out.append(self.b)
        if self.b != 0xFF:
            self.out.append(0xFF)
        self.out.append(0xAC)
        return bytes(self.out)

    # IAx (A.2), as jbig2enc writes it
    RANGES = [(0, 3, 0, 2, 0, 2), (-1, -1, 9, 4, 0, 0), (-3, -2, 5, 3, 2, 1), (4, 19, 2, 3, 4, 4), (-19, -4, 3, 3, 4, 4),
              (20, 83, 6, 4, 20, 6), (-83, -20, 7, 4, 20, 6), (84, 339, 14, 5, 84, 8), (-339, -84, 15, 5, 84, 8),
              (340, 4435, 30, 6, 340, 12), (-4435, -340, 31, 6, 340, 12), (4436, 2000000000, 62, 6, 4436, 32),
              (-2000000000, -4436, 63, 6, 4436, 32)]

    def _ibit(self, ctx, prev, v):
        self.bit(ctx, prev, v)
        return ((((prev << 1) | v) & 0x1FF) | 0x100) if prev & 0x100 else ((prev << 1) | v)

    def int(self, ctx, value):
        prev = 1
        bot, top, data, bits, delta, intbits = next(r for r in self.RANGES if r[0] <= value <= r[1])
        value = abs(value) - delta
        for _ in range(bits):
            prev = self._ibit(ctx, prev, data & 1)
            data >>= 1
        for j in range(intbits - 1, -1, -1):
            prev = self._ibit(ctx, prev, (value >> j) & 1)

    def oob(self, ctx):
        for cx, v in ((1, 1), (3, 0), (6, 0), (12, 0)):
            self.bit(ctx, cx, v)

    def iaid(self, ctx, length, value):
        prev = 1
        for j in range(length - 1, -1, -1):
            v = (value >> j) & 1
            self.bit(ctx, prev, v)
            prev = (prev << 1) | v


def h2_check():
    data = bytes([0, 2, 0, 0x51, 0, 0, 0, 0xC0, 0x03, 0x52, 0x87, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA, 0x82, 0xC0, 0x20, 0, 0xFC, 0xD7, 0x9E,
                  0xF6, 0xBF, 0x7F, 0xED, 0x90, 0x4F, 0x46, 0xA3, 0xBF])
    m, ctx = MQ(), {}
    for byte in data:
        for k in range(7, -1, -1):
            m.bit(ctx, 0, (byte >> k) & 1)
    got = m.final()
    want = bytes.fromhex("84C73BFCE1A14304022000004 10DBB86F4317FFF88FF37471ADB6ADFFFAC".replace(" ", ""))
    assert got == want, got.hex()


# -------------------------------------------------------------- bitmaps --

def get(bm, x, y):
    h, w = bm.shape
    return int(bm[y, x]) if 0 <= x < w and 0 <= y < h else 0


NOMINAL = {0: [(3, -1), (-3, -1), (2, -2), (-2, -2)], 1: [(3, -1)], 2: [(2, -1)], 3: [(2, -1)]}
SLTP = {0: 0x9B25, 1: 0x0795, 2: 0x00E5, 3: 0x0195}


def gctx(bm, x, y, t, at):
    g = lambda dx, dy: get(bm, x + dx, y + dy)
    if t == 0:
        c = g(-1, 0) | g(-2, 0) << 1 | g(-3, 0) << 2 | g(-4, 0) << 3 | g(*at[0]) << 4
        c |= g(2, -1) << 5 | g(1, -1) << 6 | g(0, -1) << 7 | g(-1, -1) << 8 | g(-2, -1) << 9
        c |= g(*at[1]) << 10 | g(*at[2]) << 11 | g(1, -2) << 12 | g(0, -2) << 13 | g(-1, -2) << 14 | g(*at[3]) << 15
    elif t == 1:
        c = g(-1, 0) | g(-2, 0) << 1 | g(-3, 0) << 2 | g(*at[0]) << 3 | g(2, -1) << 4 | g(1, -1) << 5 | g(0, -1) << 6
        c |= g(-1, -1) << 7 | g(-2, -1) << 8 | g(2, -2) << 9 | g(1, -2) << 10 | g(0, -2) << 11 | g(-1, -2) << 12
    elif t == 2:
        c = g(-1, 0) | g(-2, 0) << 1 | g(*at[0]) << 2 | g(1, -1) << 3 | g(0, -1) << 4 | g(-1, -1) << 5 | g(-2, -1) << 6
        c |= g(1, -2) << 7 | g(0, -2) << 8 | g(-1, -2) << 9
    else:
        c = g(-1, 0) | g(-2, 0) << 1 | g(-3, 0) << 2 | g(-4, 0) << 3 | g(*at[0]) << 4 | g(1, -1) << 5 | g(0, -1) << 6
        c |= g(-1, -1) << 7 | g(-2, -1) << 8 | g(-3, -1) << 9
    return c


def enc_generic(m, ctx, bm, t, at, tpgdon):
    h, w = bm.shape
    ltp = 0
    for y in range(h):
        if tpgdon:
            same = y > 0 and (bm[y] == bm[y - 1]).all() or (y == 0 and not bm[0].any())
            m.bit(ctx, SLTP[t], int(same) ^ ltp)
            ltp = int(same)
            if same:
                continue
        for x in range(w):
            m.bit(ctx, gctx(bm, x, y, t, at), int(bm[y, x]))


def rctx(bm, ref, x, y, t, dx, dy, grat):
    g = lambda i, j: get(bm, x + i, y + j)
    r = lambda i, j: get(ref, x - dx + i, y - dy + j)
    if t == 0:
        return (g(-1, 0) | g(1, -1) << 1 | g(0, -1) << 2 | g(grat[0], grat[1]) << 3 | r(1, 1) << 4 | r(0, 1) << 5 | r(-1, 1) << 6 |
                r(1, 0) << 7 | r(0, 0) << 8 | r(-1, 0) << 9 | r(1, -1) << 10 | r(0, -1) << 11 | r(grat[2], grat[3]) << 12)
    return (g(-1, 0) | g(1, -1) << 1 | g(0, -1) << 2 | g(-1, -1) << 3 | r(1, 1) << 4 | r(0, 1) << 5 | r(1, 0) << 6 | r(0, 0) << 7 |
            r(-1, 0) << 8 | r(0, -1) << 9)


def enc_refine(m, ctx, bm, ref, t, dx, dy, grat, tpgron):
    h, w = bm.shape
    ltp = 0
    for y in range(h):
        if tpgron:
            # typical: every pixel whose 3×3 reference neighbourhood is uniform equals it
            typ = True
            for x in range(w):
                nb = [get(ref, x - dx + i, y - dy + j) for j in (-1, 0, 1) for i in (-1, 0, 1)]
                if len(set(nb)) == 1 and nb[0] != bm[y, x]:
                    typ = False
                    break
            m.bit(ctx, 0x100 if t == 0 else 0x40, int(typ) ^ ltp)
            ltp = int(typ)
        for x in range(w):
            if ltp:
                nb = [get(ref, x - dx + i, y - dy + j) for j in (-1, 0, 1) for i in (-1, 0, 1)]
                if len(set(nb)) == 1:
                    continue
            m.bit(ctx, rctx(bm, ref, x, y, t, dx, dy, grat), int(bm[y, x]))


def compose(dst, src, x, y, op):
    h, w = src.shape
    H, W = dst.shape
    for j in range(h):
        for i in range(w):
            X, Y = x + i, y + j
            if 0 <= X < W and 0 <= Y < H:
                s, d = int(src[j, i]), int(dst[Y, X])
                dst[Y, X] = [s | d, s & d, s ^ d, 1 - (s ^ d), s][op]


# ------------------------------------------------------------- segments --

def region_info(w, h, x, y, op):
    return w.to_bytes(4, "big") + h.to_bytes(4, "big") + x.to_bytes(4, "big") + y.to_bytes(4, "big") + bytes([op])


def segment(num, typ, data, refs=(), page=1):
    assert len(refs) < 5
    flags = typ
    rts = bytes([len(refs) << 5])
    size = 1 if num <= 256 else (2 if num <= 65536 else 4)
    rb = b"".join(r.to_bytes(size, "big") for r in refs)
    return num.to_bytes(4, "big") + bytes([flags]) + rts + rb + bytes([page]) + len(data).to_bytes(4, "big") + data


def page_info(w, h, default=0, striped=0, op=0):
    return (w.to_bytes(4, "big") + h.to_bytes(4, "big") + (0).to_bytes(4, "big") * 2 + bytes([(default << 2) | (op << 3)])
            + striped.to_bytes(2, "big"))


def jb2_file(segs):
    return b"\x97JB2\r\n\x1a\n" + bytes([0x01]) + (1).to_bytes(4, "big") + b"".join(segs) + segment(999, 51, b"", page=0)


def at_bytes(at):
    return bytes([(v & 0xFF) for p in at for v in p])


def generic_seg(num, bm, x, y, op, t, at, tpgdon, typ=38):
    m, ctx = MQ(), {}
    enc_generic(m, ctx, bm, t, at, tpgdon)
    h, w = bm.shape
    flags = (t << 1) | (int(tpgdon) << 3)
    return segment(num, typ, region_info(w, h, x, y, op) + bytes([flags]) + at_bytes(at) + m.final())


def mmr_seg(num, bm, x, y, op):
    h, w = bm.shape
    tif = Image.fromarray(np.where(bm == 1, 255, 0).astype(np.uint8)).convert("1")
    b = io.BytesIO()
    tif.save(b, "TIFF", compression="group4", tiffinfo={278: h})
    t = Image.open(io.BytesIO(b.getvalue()))
    stream = b.getvalue()[t.tag_v2[273][0]:t.tag_v2[273][0] + t.tag_v2[279][0]]
    return segment(num, 38, region_info(w, h, x, y, op) + bytes([1]) + stream)


# ------------------------------------------------------ symbols and text --

def text_stream(m, cx, syms, insts, p, codelen):
    """Encode instances [(id, x, y, refined_or_None)] into the arithmetic
    stream `m` with integer contexts `cx` (dict of name → ctx); returns the
    region bitmap they compose to."""
    strips, rc, tr, dso = p["strips"], p["refcorner"], p["transposed"], p["dsoffset"]
    region = np.full((p["h"], p["w"]), p["defpixel"], dtype=np.uint8)
    placed = []
    for sid, x, y, ref in insts:
        ib = syms[sid] if ref is None else ref["bitmap"]
        h, w = ib.shape
        if not tr:
            s = x + (w - 1 if rc in (2, 3) else 0)
            t = y + (h - 1 if rc in (0, 2) else 0)
        else:
            t = x + (w - 1 if rc in (2, 3) else 0)
            s = y + (h - 1 if rc in (0, 2) else 0)
        placed.append((t // strips * strips, s, t, sid, ib, ref, x, y))
    placed.sort(key=lambda q: (q[0], q[1]))
    # initial STRIPT = −DT₀·SBSTRIPS: 0 for the arithmetic streams; the Huffman
    # DT tables (B.11–B.13) start at 1, so those streams pass dt0 = 1
    dt0 = p.get("dt0", 0)
    m.int(cx["dt"], dt0)
    stript, firsts = -dt0 * strips, 0
    i = 0
    while i < len(placed):
        st = placed[i][0]
        m.int(cx["dt"], (st - stript) // strips)
        stript = st
        first, curs = True, None
        while i < len(placed) and placed[i][0] == st:
            _, s, t, sid, ib, ref, x, y = placed[i]
            h, w = ib.shape
            pre = (w - 1) if (not tr and rc > 1) else ((h - 1) if (tr and not (rc & 1)) else 0)
            post = (w - 1) if (not tr and rc < 2) else ((h - 1) if (tr and (rc & 1)) else 0)
            if first:
                m.int(cx["fs"], (s - pre) - firsts)
                firsts = s - pre
                first = False
            else:
                m.int(cx["ds"], (s - pre) - curs - dso)
            curs = s + post
            if strips > 1:
                m.int(cx["it"], t - st)
            m.iaid(cx["id"], codelen, sid)
            if p["refine"]:
                if ref is None:
                    m.int(cx["ri"], 0)
                else:
                    m.int(cx["ri"], 1)
                    sym = syms[sid]
                    rdw, rdh, rdx, rdy = ref["rdw"], ref["rdh"], ref["rdx"], ref["rdy"]
                    for name, v in (("rdw", rdw), ("rdh", rdh), ("rdx", rdx), ("rdy", rdy)):
                        m.int(cx[name], v)
                    enc_refine(m, cx["gr"], ib, sym, p["rtemplate"], (rdw >> 1) + rdx, (rdh >> 1) + rdy, p.get("rat", [0, 0, 0, 0]), False)
            compose(region, ib, x, y, p["combop"])
            i += 1
        m.oob(cx["ds"])
    return region


def int_ctxs():
    return {k: {} for k in ("dt", "fs", "ds", "it", "id", "ri", "rdw", "rdh", "rdx", "rdy", "gr")}


def text_seg(num, refs, syms, insts, p):
    m, cx = MQ(), int_ctxs()
    codelen = max(0, (len(syms) - 1).bit_length())
    region = text_stream(m, cx, syms, insts, p, codelen)
    flags = (int(p["refine"]) << 1) | ({1: 0, 2: 1, 4: 2, 8: 3}[p["strips"]] << 2) | (p["refcorner"] << 4) | (int(p["transposed"]) << 6)
    flags |= (p["combop"] << 7) | (p["defpixel"] << 9) | ((p["dsoffset"] & 31) << 10) | (p.get("rtemplate", 0) << 15)
    rat = at_bytes([(p["rat"][0], p["rat"][1]), (p["rat"][2], p["rat"][3])]) if p["refine"] and p.get("rtemplate", 0) == 0 else b""
    data = region_info(p["w"], p["h"], p["x"], p["y"], p["op"]) + flags.to_bytes(2, "big") + rat + len(insts).to_bytes(4, "big") + m.final()
    return segment(num, 6, data, refs), region


def dict_seg(num, refs, insyms, entries, sdtemplate=0, at=None, refagg=False, rtemplate=0, rat=(-1, -1, -1, -1)):
    """entries: list of ('generic', bitmap) | ('refine', id, rdx, rdy, bitmap) | ('agg', [(id, x, y)], h, w) —
    grouped by height (the order given must be by height class). Exports every new symbol."""
    at = at or NOMINAL[sdtemplate]
    m = MQ()
    gb, gr = {}, {}
    ic = {k: {} for k in ("dh", "dw", "ex", "ai", "rdx", "rdy", "id")}
    tc = int_ctxs()
    tc["id"], tc["rdx"], tc["rdy"], tc["gr"] = ic["id"], ic["rdx"], ic["rdy"], gr
    nin, nnew = len(insyms), len(entries)
    codelen = max(0, (nin + nnew - 1).bit_length())
    news = []
    hc, i = 0, 0
    while i < len(entries):
        h = shape_of(entries[i])[0]
        m.int(ic["dh"], h - hc)
        hc, symw = h, 0
        while i < len(entries) and shape_of(entries[i])[0] == h:
            e = entries[i]
            w = shape_of(e)[1]
            m.int(ic["dw"], w - symw)
            symw = w
            if not refagg:
                enc_generic(m, gb, e[1], sdtemplate, at, False)
                news.append(e[1])
            elif e[0] == "refine":
                _, rid, rdx, rdy, bm = e
                m.int(ic["ai"], 1)
                m.iaid(ic["id"], codelen, rid)
                m.int(ic["rdx"], rdx)
                m.int(ic["rdy"], rdy)
                enc_refine(m, gr, bm, (insyms + news)[rid], rtemplate, rdx, rdy, list(rat), False)
                news.append(bm)
            elif e[0] == "agg":
                _, parts, ah, aw = e
                m.int(ic["ai"], len(parts))
                p = {"strips": 1, "refcorner": 1, "transposed": False, "dsoffset": 0, "defpixel": 0, "combop": 0, "w": aw, "h": ah,
                     "refine": True, "rtemplate": rtemplate, "rat": list(rat)}
                bm = text_stream(m, tc, insyms + news, [(pid, x, y, None) for pid, x, y in parts], p, codelen)
                news.append(bm)
            else:
                # a generic symbol inside a refinement/aggregate dictionary: an aggregate of one refined instance is
                # not allowed (REFAGGNINST = 1 means refinement), so use a refinement of an input symbol
                raise ValueError(e[0])
            i += 1
        m.oob(ic["dw"])
    # export: none of the inputs, every new symbol
    m.int(ic["ex"], nin)
    m.int(ic["ex"], nnew)
    flags = int(refagg) << 1 | (sdtemplate << 10) | (rtemplate << 12)
    hdr = flags.to_bytes(2, "big") + at_bytes(at)
    if refagg and rtemplate == 0:
        hdr += at_bytes([(rat[0], rat[1]), (rat[2], rat[3])])
    hdr += nnew.to_bytes(4, "big") + nnew.to_bytes(4, "big")
    return segment(num, 0, hdr + m.final(), refs), news


def shape_of(e):
    if e[0] == "generic":
        return e[1].shape
    if e[0] == "refine":
        return e[4].shape
    return (e[2], e[3])


# ------------------------------------------------- Huffman coding (Annex B) --
# An encoder written from T.88 Annex B, independent of vapor's decoder; its
# streams are judged by jbig2dec like every other stream here.
# Lines: (PREFLEN, RANGELEN, RANGELOW), then lower, upper, [OOB] as in B.5.

STD = {
    1: ([(1, 4, 0), (2, 8, 16), (3, 16, 272)], (0, -1), (3, 65808), None),
    2: ([(1, 0, 0), (2, 0, 1), (3, 0, 2), (4, 3, 3), (5, 6, 11)], (0, -1), (6, 75), 6),
    3: ([(8, 8, -256), (1, 0, 0), (2, 0, 1), (3, 0, 2), (4, 3, 3), (5, 6, 11)], (8, -257), (7, 75), 6),
    4: ([(1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 3, 4), (5, 6, 12)], (0, -1), (5, 76), None),
    5: ([(7, 8, -255), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 3, 4), (5, 6, 12)], (7, -256), (6, 76), None),
    6: ([(5, 10, -2048), (4, 9, -1024), (4, 8, -512), (4, 7, -256), (5, 6, -128), (5, 5, -64), (4, 5, -32), (2, 7, 0),
         (3, 7, 128), (3, 8, 256), (4, 9, 512), (4, 10, 1024)], (6, -2049), (6, 2048), None),
    7: ([(4, 9, -1024), (3, 8, -512), (4, 7, -256), (5, 6, -128), (5, 5, -64), (4, 5, -32), (4, 5, 0), (5, 5, 32),
         (5, 6, 64), (4, 7, 128), (3, 8, 256), (3, 9, 512), (3, 10, 1024)], (5, -1025), (5, 2048), None),
    8: ([(8, 3, -15), (9, 1, -7), (8, 1, -5), (9, 0, -3), (7, 0, -2), (4, 0, -1), (2, 1, 0), (5, 0, 2), (6, 0, 3),
         (3, 4, 4), (6, 1, 20), (4, 4, 22), (4, 5, 38), (5, 6, 70), (5, 7, 134), (6, 7, 262), (7, 8, 390), (6, 10, 646)],
        (9, -16), (9, 1670), 2),
    9: ([(8, 4, -31), (9, 2, -15), (8, 2, -11), (9, 1, -7), (7, 1, -5), (4, 1, -3), (3, 1, -1), (3, 1, 1), (5, 1, 3),
         (6, 1, 5), (3, 5, 7), (6, 2, 39), (4, 5, 43), (4, 6, 75), (5, 7, 139), (5, 8, 267), (6, 8, 523), (7, 9, 779),
         (6, 11, 1291)], (9, -32), (9, 3339), 2),
    10: ([(7, 4, -21), (8, 0, -5), (7, 0, -4), (5, 0, -3), (2, 2, -2), (5, 0, 2), (6, 0, 3), (7, 0, 4), (8, 0, 5),
          (2, 6, 6), (5, 5, 70), (6, 5, 102), (6, 6, 134), (6, 7, 198), (6, 8, 326), (6, 9, 582), (6, 10, 1094),
          (7, 11, 2118)], (8, -22), (8, 4166), 2),
    11: ([(1, 0, 1), (2, 1, 2), (4, 0, 4), (4, 1, 5), (5, 1, 7), (5, 2, 9), (6, 2, 13), (7, 2, 17), (7, 3, 21),
          (7, 4, 29), (7, 5, 45), (7, 6, 77)], (0, 0), (7, 141), None),
    12: ([(1, 0, 1), (2, 0, 2), (3, 1, 3), (5, 0, 5), (5, 1, 6), (6, 1, 8), (7, 0, 10), (7, 1, 11), (7, 2, 13),
          (7, 3, 17), (7, 4, 25), (8, 5, 41)], (0, 0), (8, 73), None),
    13: ([(1, 0, 1), (3, 0, 2), (4, 0, 3), (5, 0, 4), (4, 1, 5), (3, 3, 7), (6, 1, 15), (6, 2, 17), (6, 3, 21),
          (6, 4, 29), (6, 5, 45), (7, 6, 77)], (0, 0), (7, 141), None),
}


class HTable:
    """Prefix codes assigned by B.3 to lines [(kind, preflen, rangelen, rangelow)]."""

    def __init__(self, lines):
        self.lines = lines
        maxlen = max([l[1] for l in lines] + [0])
        count = {}
        for l in lines:
            count[l[1]] = count.get(l[1], 0) + 1
        count[0] = 0
        self.codes = {}
        first = 0
        for length in range(1, maxlen + 1):
            first = (first + count.get(length - 1, 0)) << 1
            code = first
            for i, l in enumerate(lines):
                if l[1] == length:
                    self.codes[i] = (code, length)
                    code += 1

    @staticmethod
    def standard(n):
        normal, (lp, llow), (up, ulow), oob = STD[n]
        lines = [("n", p, r, low) for p, r, low in normal] + [("lo", lp, 32, llow), ("hi", up, 32, ulow)]
        if oob is not None:
            lines.append(("oob", oob, 0, 0))
        return HTable(lines)

    def write(self, w, v):
        """Write value v (or None for OOB) into the bit writer w."""
        for i, (kind, p, r, low) in enumerate(self.lines):
            if p == 0:
                continue
            if v is None and kind == "oob" or (v is not None and (
                    (kind == "n" and low <= v < low + (1 << r)) or (kind == "lo" and v <= low) or (kind == "hi" and v >= low))):
                code, length = self.codes[i]
                w.bits(code, length)
                if kind == "n":
                    w.bits(v - low, r)
                elif kind == "lo":
                    w.bits(low - v, 32)
                elif kind == "hi":
                    w.bits(v - low, 32)
                return
        raise ValueError(f"value {v} has no line in the table")


class BitWriter:
    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def bits(self, v, n):
        for k in range(n - 1, -1, -1):
            self.acc = (self.acc << 1) | ((v >> k) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                self.acc, self.n = 0, 0

    def align(self):
        if self.n:
            self.bits(0, 8 - self.n)

    def raw(self, data):
        assert self.n == 0
        self.out += data

    def final(self):
        self.align()
        return bytes(self.out)


def custom_table(lines, low, htoob, lower_p, upper_p, oob_p=0):
    """A code table segment body (B.2) for normal lines [(preflen, rangelen)] from `low` up, and its HTable."""
    high = low + sum(1 << r for _, r in lines)
    htps = max([p for p, _ in lines] + [lower_p, upper_p, oob_p]).bit_length()
    htrs = max([r for _, r in lines] + [1]).bit_length()
    w = BitWriter()
    tl, cur = [], low
    for p, r in lines:
        w.bits(p, htps)
        w.bits(r, htrs)
        tl.append(("n", p, r, cur))
        cur += 1 << r
    assert cur == high, (cur, high)
    w.bits(lower_p, htps)
    w.bits(upper_p, htps)
    tl += [("lo", lower_p, 32, low - 1), ("hi", upper_p, 32, high)]
    if htoob:
        w.bits(oob_p, htps)
        tl.append(("oob", oob_p, 0, 0))
    flags = htoob | ((htps - 1) << 1) | ((htrs - 1) << 4)
    body = bytes([flags]) + low.to_bytes(4, "big", signed=True) + high.to_bytes(4, "big", signed=True) + w.final()
    return body, HTable(tl)


def huff_dict_seg(num, refs, classes, dh=4, dw=2, mmr=False, custom=None):
    """classes: [[bitmap, …] of one height, …] in order. dh ∈ {4, 5, 'c'}, dw ∈ {2, 3, 'c'}; custom: {'dh': HTable, 'dw': HTable}."""
    custom = custom or {}
    tdh = custom["dh"] if dh == "c" else HTable.standard(dh)
    tdw = custom["dw"] if dw == "c" else HTable.standard(dw)
    tbm = HTable.standard(1)
    w = BitWriter()
    hc, news = 0, []
    for cls in classes:
        h = cls[0].shape[0]
        tdh.write(w, h - hc)
        hc, symw = h, 0
        for bm in cls:
            tdw.write(w, bm.shape[1] - symw)
            symw = bm.shape[1]
            news.append(bm)
        tdw.write(w, None)
        coll = np.concatenate(cls, axis=1)
        if mmr:
            stream = g4(coll)
            tbm.write(w, len(stream))
            w.align()
            w.raw(stream)
        else:
            tbm.write(w, 0)
            w.align()
            w.raw(np.packbits(coll, axis=1).tobytes())
    # export: no input symbol (run 0), every new one
    tbm.write(w, 0)
    tbm.write(w, len(news))
    sel_dh = {4: 0, 5: 1, "c": 3}[dh]
    sel_dw = {2: 0, 3: 1, "c": 3}[dw]
    flags = 1 | (sel_dh << 2) | (sel_dw << 4)
    hdr = flags.to_bytes(2, "big") + len(news).to_bytes(4, "big") + len(news).to_bytes(4, "big")
    return segment(num, 0, hdr + w.final(), refs), news


def g4(bm):
    h, w = bm.shape
    tif = Image.fromarray(np.where(bm == 1, 255, 0).astype(np.uint8)).convert("1")
    b = io.BytesIO()
    tif.save(b, "TIFF", compression="group4", tiffinfo={278: h})
    t = Image.open(io.BytesIO(b.getvalue()))
    return b.getvalue()[t.tag_v2[273][0]:t.tag_v2[273][0] + t.tag_v2[279][0]]


class HuffCoder:
    """text_stream's coder interface over Huffman tables: int(name, v), oob(name), iaid(name, len, id)."""

    def __init__(self, w, tables, idcodes, logstrips):
        self.w, self.t, self.idcodes, self.logstrips = w, tables, idcodes, logstrips

    def int(self, ctx, v):
        if ctx == "it":
            self.w.bits(v, self.logstrips)
        else:
            self.t[ctx].write(self.w, v)

    def oob(self, ctx):
        self.t[ctx].write(self.w, None)

    def iaid(self, ctx, length, sid):
        code, n = self.idcodes[sid]
        self.w.bits(code, n)


def huffman_lengths(freqs):
    """Code lengths of a Huffman code for the positive frequencies; 0 for the others."""
    import heapq
    items = [(f, i, [i]) for i, f in enumerate(freqs) if f > 0]
    lens = [0] * len(freqs)
    if len(items) == 1:
        lens[items[0][1]] = 1
        return lens
    heapq.heapify(items)
    k = len(freqs)
    while len(items) > 1:
        f1, _, a = heapq.heappop(items)
        f2, _, b = heapq.heappop(items)
        for i in a + b:
            lens[i] += 1
        k += 1
        heapq.heappush(items, (f1 + f2, k, a + b))
    return lens


def symbol_id_table(w, lens):
    """Write the run-coded symbol ID table (7.4.3.1.7) for code lengths `lens`; returns {id: (code, length)}."""
    run = HTable([("n", 6, 0, i) for i in range(35)])  # every run code 6 bits long
    for _ in range(35):
        w.bits(6, 4)
    i = 0
    while i < len(lens):
        L = lens[i]
        j = i
        while j < len(lens) and lens[j] == L:
            j += 1
        n = j - i
        if L == 0 and n >= 11:
            k = min(n, 138)
            run.write(w, 34)
            w.bits(k - 11, 7)
            i += k
        elif L == 0 and n >= 3:
            k = min(n, 10)
            run.write(w, 33)
            w.bits(k - 3, 3)
            i += k
        elif i > 0 and lens[i - 1] == L and n >= 3:
            k = min(n, 6)
            run.write(w, 32)
            w.bits(k - 3, 2)
            i += k
        else:
            run.write(w, L)
            i += 1
    w.align()
    t = HTable([("n", L, 0, k) for k, L in enumerate(lens)])
    return {k: t.codes[k] for k in range(len(lens)) if lens[k] > 0}


def huff_text_seg(num, refs, syms, insts, p, fs=6, ds=8, dt=11, custom=None, freqs=None):
    """A text region with SBHUFF = 1 (no refinement). fs ∈ {6, 7, 'c'}, ds ∈ {8, 9, 10, 'c'}, dt ∈ {11, 12, 13, 'c'}."""
    custom = custom or {}
    pick = lambda v, name: custom[name] if v == "c" else HTable.standard(v)
    w = BitWriter()
    freqs = freqs or [1] * len(syms)
    lens = huffman_lengths(freqs)
    idcodes = symbol_id_table(w, lens)
    coder = HuffCoder(w, {"fs": pick(fs, "fs"), "ds": pick(ds, "ds"), "dt": pick(dt, "dt")}, idcodes,
                      {1: 0, 2: 1, 4: 2, 8: 3}[p["strips"]])
    cx = {k: k for k in ("dt", "fs", "ds", "it", "id")}
    region = text_stream(coder, cx, syms, insts, dict(p, refine=False, dt0=1), 0)
    flags = 1 | ({1: 0, 2: 1, 4: 2, 8: 3}[p["strips"]] << 2) | (p["refcorner"] << 4) | (int(p["transposed"]) << 6)
    flags |= (p["combop"] << 7) | (p["defpixel"] << 9) | ((p["dsoffset"] & 31) << 10)
    hflags = {6: 0, 7: 1, "c": 3}[fs] | ({8: 0, 9: 1, 10: 2, "c": 3}[ds] << 2) | ({11: 0, 12: 1, 13: 2, "c": 3}[dt] << 4)
    data = region_info(p["w"], p["h"], p["x"], p["y"], p["op"]) + flags.to_bytes(2, "big") + hflags.to_bytes(2, "big")
    data += len(insts).to_bytes(4, "big") + w.final()
    return segment(num, 6, data, refs), region


# ------------------------------------------------------------- material --

def blobs(w, h, n, seed):
    r = np.random.default_rng(seed)
    bm = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n):
        x, y = r.integers(0, w - 6), r.integers(0, h - 6)
        bm[y:y + r.integers(2, 9), x:x + r.integers(2, 12)] = 1
    # a few lone pixels and a run of repeated rows (for TPGD)
    for _ in range(n // 3):
        bm[r.integers(0, h), r.integers(0, w)] = 1
    bm[h // 2:h // 2 + 6, :] = bm[h // 2]
    return bm


def glyph(w, h, seed):
    r = np.random.default_rng(seed)
    g = (r.random((h, w)) < 0.45).astype(np.uint8)
    g[0, :] = 1
    g[:, 0] = 1
    return g


def pbm(bm):
    h, w = bm.shape
    rows = np.packbits(bm, axis=1)
    return f"P4\n{w} {h}\n".encode() + rows.tobytes()


manifest = {}
jbig2dec = "jbig2dec"


def keep(name, files, bm):
    """Write the stream(s) and the expected bitmap; keep only if jbig2dec agrees."""
    for fname, data in files.items():
        open(os.path.join(out, fname), "wb").write(data)
    exp = pbm(bm)
    open(os.path.join(out, name + ".pbm"), "wb").write(exp)
    args = [jbig2dec, "-t", "pbm", "-o", os.path.join(out, "_check.pbm")]
    if len(files) == 2:
        args += ["-e"] + [os.path.join(out, f) for f in files]
    else:
        args += [os.path.join(out, f) for f in files]
    r = subprocess.run(args, capture_output=True, text=True)
    got = open(os.path.join(out, "_check.pbm"), "rb").read() if os.path.exists(os.path.join(out, "_check.pbm")) else b""
    os.remove(os.path.join(out, "_check.pbm")) if got else None
    assert got == exp, f"{name}: jbig2dec disagrees ({r.stderr.strip()[:200]})"
    h, w = bm.shape
    manifest[name] = {"files": list(files), "w": w, "h": h, "sha256": hashlib.sha256(exp).hexdigest()}


def main():
    h2_check()

    # generic regions: every template, nominal and moved adaptive pixels, with and without TPGD
    moved = {0: [(4, -1), (-4, -2), (1, -3), (-2, 0)], 1: [(-3, -2)], 2: [(-2, 0)], 3: [(-4, -1)]}
    for t in range(4):
        for at_name, at in (("nominal", NOMINAL[t]), ("moved", moved[t])):
            for tp in (False, True):
                bm = blobs(150, 60, 70, 10 * t + len(at_name) + tp)
                page = np.zeros((70, 170), dtype=np.uint8)
                compose(page, bm, 9, 5, 0)
                segs = [segment(1, 48, page_info(170, 70)), generic_seg(2, bm, 9, 5, 0, t, at, tp), segment(3, 49, b"")]
                keep(f"generic_t{t}_{at_name}{'_tpgd' if tp else ''}", {f"generic_t{t}_{at_name}{'_tpgd' if tp else ''}.jb2": jb2_file(segs)}, page)

    # MMR generic region, XOR-composed over an arithmetic one
    a, b = blobs(120, 50, 50, 1), blobs(120, 50, 40, 2)
    page = np.zeros((60, 140), dtype=np.uint8)
    compose(page, a, 5, 4, 0)
    compose(page, b, 12, 7, 2)
    segs = [segment(1, 48, page_info(140, 60)), generic_seg(2, a, 5, 4, 0, 0, NOMINAL[0], False), mmr_seg(3, b, 12, 7, 2), segment(4, 49, b"")]
    keep("mmr_xor", {"mmr_xor.jb2": jb2_file(segs)}, page)

    # refinement regions over the page (templates 0, 1; TPGRON). At the page's
    # origin: jbig2dec takes the whole page as the reference with no offset
    # (its "TODO: subset the image"), where T.88 7.4.7.4 takes the page under
    # the region — the two agree only at (0, 0), so that is what is judged
    for t in (0, 1):
        for tp in (False, True):
            base = blobs(100, 40, 40, 30 + t)
            target = base.copy()
            r = np.random.default_rng(5 + t)
            for _ in range(25):
                target[r.integers(0, 40), r.integers(0, 100)] ^= 1
            page = np.zeros((50, 120), dtype=np.uint8)
            compose(page, base, 0, 0, 0)
            m, ctx = MQ(), {}
            ref = page[0:40, 0:100].copy()
            enc_refine(m, ctx, target, ref, t, 0, 0, [-1, -1, -1, -1], tp)
            data = region_info(100, 40, 0, 0, 4) + bytes([t | (int(tp) << 1)]) + (at_bytes([(-1, -1), (-1, -1)]) if t == 0 else b"") + m.final()
            final = page.copy()
            compose(final, target, 0, 0, 4)
            segs = [segment(1, 48, page_info(120, 50)), generic_seg(2, base, 0, 0, 0, 0, NOMINAL[0], False), segment(3, 42, data), segment(4, 49, b"")]
            keep(f"refine_t{t}{'_tpgron' if tp else ''}", {f"refine_t{t}{'_tpgron' if tp else ''}.jb2": jb2_file(segs)}, final)

    # text regions: 4 corners × transposed, strips, DSOFFSET, combination, default pixel, refined instances
    syms = [glyph(7 + k % 4, 9 + k % 3, 100 + k) for k in range(6)]
    entries = sorted([("generic", s) for s in syms], key=lambda e: e[1].shape[0])
    ordered = [e[1] for e in entries]
    for rc in range(4):
        for tr in (False, True):
            for strips, dso, combop, defpix, refine in ((1, 0, 0, 0, False), (4, 3, 2, 1, True)):
                W, H = 140, 70
                insts = []
                r = np.random.default_rng(rc * 10 + tr * 5 + strips)
                for k in range(14):
                    sid = int(r.integers(0, 6))
                    x, y = int(r.integers(0, W - 14)), int(r.integers(0, H - 14))
                    ref = None
                    if refine and k % 3 == 0:
                        sym = ordered[sid]
                        rdw, rdh = int(r.integers(-1, 3)), int(r.integers(-1, 2))
                        nb = np.zeros((sym.shape[0] + rdh, sym.shape[1] + rdw), dtype=np.uint8)
                        hh, ww = min(nb.shape[0], sym.shape[0]), min(nb.shape[1], sym.shape[1])
                        nb[:hh, :ww] = sym[:hh, :ww]
                        nb[r.integers(0, nb.shape[0]), r.integers(0, nb.shape[1])] ^= 1
                        ref = {"bitmap": nb, "rdw": rdw, "rdh": rdh, "rdx": int(r.integers(-1, 2)), "rdy": 0}
                    insts.append((sid, x, y, ref))
                p = {"strips": strips, "refcorner": rc, "transposed": tr, "dsoffset": dso, "combop": combop, "defpixel": defpix,
                     "refine": refine, "rtemplate": rc % 2, "rat": [-1, -1, -1, -1], "w": W, "h": H, "x": 6, "y": 4, "op": 0}
                dseg, news = dict_seg(2, [], [], entries)
                tseg, region = text_seg(3, [2], news, insts, p)
                page = np.zeros((80, 150), dtype=np.uint8)
                compose(page, region, 6, 4, 0)
                name = f"text_rc{rc}{'_tr' if tr else ''}{'_strips_ref' if refine else ''}"
                keep(name, {name + ".jb2": jb2_file([segment(1, 48, page_info(150, 80)), dseg, tseg, segment(4, 49, b"")])}, page)

    # symbol dictionaries: template 1 with a moved AT pixel; a second dictionary refining and aggregating the first's symbols
    base = [glyph(8, 10, 300 + k) for k in range(4)]
    d1, s1 = dict_seg(2, [], [], [("generic", b) for b in base], sdtemplate=1, at=[(-2, -1)])
    r1 = s1[2].copy()
    r1[3, 4] ^= 1
    r1[6, 1] ^= 1
    agg_h, agg_w = 10, 20
    entries2 = [("refine", 2, 0, 0, r1), ("agg", [(0, 0, 0), (1, 10, 0)], agg_h, agg_w)]
    d2, s2 = dict_seg(3, [2], s1, entries2, refagg=True, rtemplate=0, rat=(-1, -1, -1, -1))
    allsyms = s1 + s2  # what a text region referring to both sees (each dictionary exports its new symbols)
    insts = [(4, 5, 5, None), (5, 30, 5, None), (0, 60, 8, None), (3, 80, 12, None)]
    p = {"strips": 1, "refcorner": 1, "transposed": False, "dsoffset": 0, "combop": 0, "defpixel": 0, "refine": False, "w": 110,
         "h": 30, "x": 0, "y": 0, "op": 0}
    tseg, region = text_seg(4, [2, 3], allsyms, insts, p)
    keep("dict_refagg", {"dict_refagg.jb2": jb2_file([segment(1, 48, page_info(110, 30)), d1, d2, tseg, segment(5, 49, b"")])}, region)

    # a striped page of unknown height: two stripes, each ended by an end-of-stripe segment
    a, b = blobs(90, 30, 30, 7), blobs(90, 25, 25, 8)
    page = np.zeros((55, 90), dtype=np.uint8)
    compose(page, a, 0, 0, 0)
    compose(page, b, 0, 30, 0)
    segs = [segment(1, 48, page_info(90, 0xFFFFFFFF, striped=0x8000 | 32)), generic_seg(2, a, 0, 0, 0, 0, NOMINAL[0], True),
            segment(3, 50, (29).to_bytes(4, "big")), generic_seg(4, b, 0, 30, 0, 0, NOMINAL[0], True), segment(5, 50, (54).to_bytes(4, "big")),
            segment(6, 49, b"")]
    keep("striped", {"striped.jb2": jb2_file(segs)}, page)

    # Huffman coding (Annex B): standard tables, custom tables, MMR collective bitmaps, values on the range lines
    def huff_case(name, classes, page_w, page_h, insts, p, dict_kw=None, text_kw=None, pre=None, refs_extra=()):
        segs = [segment(1, 48, page_info(page_w, page_h))] + (pre or [])
        dseg, news = huff_dict_seg(2, list(refs_extra), classes, **(dict_kw or {}))
        tseg, region = huff_text_seg(3, [2] + list(refs_extra), news, insts, p, **(text_kw or {}))
        page = np.zeros((page_h, page_w), dtype=np.uint8)
        compose(page, region, p["x"], p["y"], 0)
        keep(name, {name + ".jb2": jb2_file(segs + [dseg, tseg, segment(4, 49, b"")])}, page)

    rr = np.random.default_rng(1234)
    def classes_of(heights, per, seed, wmin=3, wmax=14):
        r = np.random.default_rng(seed)
        # widths non-decreasing within a class: B.2 has no line for a negative width delta
        return [sorted([glyph(int(r.integers(wmin, wmax)), h, seed * 100 + h * 10 + k) for k in range(per)], key=lambda g: g.shape[1])
                for h in heights]

    def scatter(n, nsyms, W, H, seed, margin=16):
        r = np.random.default_rng(seed)
        return [(int(r.integers(0, nsyms)), int(r.integers(0, W - margin)), int(r.integers(0, H - margin)), None) for _ in range(n)]

    base_p = {"strips": 1, "refcorner": 1, "transposed": False, "dsoffset": 0, "combop": 0, "defpixel": 0, "x": 4, "y": 3, "op": 0}
    cl = classes_of([6, 9, 13], 4, 1)
    huff_case("huff_std", cl, 160, 70, scatter(30, 12, 150, 64, 2), dict(base_p, w=150, h=64))
    # heights that fall (B.5), widths that shrink (B.3), strips, transposition, other corners, DSOFFSET, B.7/B.9/B.12
    cl2 = [[glyph(12 - k, 15, 500 + k) for k in range(4)], [glyph(9 - k, 8, 600 + k) for k in range(3)], [glyph(5, 4, 700)]]
    for rc in range(4):
        for tr in (False, True):
            p = dict(base_p, strips=4, refcorner=rc, transposed=tr, dsoffset=-3, combop=2, w=140, h=90)
            huff_case(f"huff_alt_rc{rc}{'_tr' if tr else ''}", cl2, 150, 100, scatter(28, 8, 120, 80, 40 + rc + 4 * tr), p,
                      dict_kw={"dh": 5, "dw": 3}, text_kw={"fs": 7, "ds": 9, "dt": 12})
    # MMR collective bitmaps, B.10 and B.13, a black default pixel with XOR
    huff_case("huff_mmr", classes_of([7, 11, 16, 20], 5, 3, 6, 20), 220, 110, scatter(40, 20, 200, 90, 5),
              dict(base_p, strips=2, defpixel=1, combop=3, w=210, h=100), dict_kw={"mmr": True}, text_kw={"ds": 10, "dt": 13})
    # large values: a wide page with sparse instances (S deltas on B.8's upper and lower range lines), a symbol 90 px wide
    wide = [[glyph(3, 6, 802), glyph(90, 6, 801)], [glyph(4, 10, 803)]]  # 3 then +87: B.2's upper range line
    far = [(0, 10, 2, None), (1, 2400, 4, None), (2, 3000, 30, None), (1, 15, 33, None), (0, 2900, 50, None), (2, 40, 52, None)]
    huff_case("huff_far", wide, 3200, 80, far, dict(base_p, w=3150, h=72))
    # custom code tables (type 53): widths, first-S and S deltas with their own lines and OOB
    t_dw, H_dw = custom_table([(2, 2), (2, 3), (3, 4), (3, 6)], -2, 1, 4, 4, 3)
    t_fs, H_fs = custom_table([(3, 6), (2, 8), (2, 9), (3, 10)], -64, 0, 4, 4)
    t_ds, H_ds = custom_table([(3, 3), (2, 2), (2, 5), (3, 8)], -8, 1, 4, 4, 3)
    pre = [segment(10, 53, t_dw, page=0), segment(11, 53, t_fs, page=0), segment(12, 53, t_ds, page=0)]
    def custom_case():
        segs = [segment(1, 48, page_info(170, 80))] + pre
        dseg, news = huff_dict_seg(2, [10], classes_of([5, 8, 12], 4, 9), dw="c", custom={"dw": H_dw})
        insts = scatter(30, 12, 150, 64, 11)
        tseg, region = huff_text_seg(3, [2, 11, 12], news, insts, dict(base_p, w=160, h=70), fs="c", ds="c", custom={"fs": H_fs, "ds": H_ds})
        page = np.zeros((80, 170), dtype=np.uint8)
        compose(page, region, 4, 3, 0)
        keep("huff_custom", {"huff_custom.jb2": jb2_file(segs + [dseg, tseg, segment(4, 49, b"")])}, page)
    custom_case()
    # many symbols with a skewed ID code: repeats and long zero runs in the run-coded table (codes 32, 33, 34)
    many = classes_of([5, 7, 9, 11, 13, 15], 12, 21, 3, 9)
    nsym = sum(len(c) for c in many)
    freqs = [0] * nsym
    for k in range(nsym):
        freqs[k] = 0 if (20 <= k < 40 or 50 <= k < 56) else (64 if k < 4 else 1)
    used = [k for k in range(nsym) if freqs[k] > 0]
    r = np.random.default_rng(31)
    insts = [(int(used[int(r.integers(0, len(used)))]), int(r.integers(0, 180)), int(r.integers(0, 90)), None) for _ in range(60)]
    def ids_case():
        segs = [segment(1, 48, page_info(210, 115))]
        dseg, news = huff_dict_seg(2, [], many)
        tseg, region = huff_text_seg(3, [2], news, insts, dict(base_p, w=200, h=108), freqs=freqs)
        page = np.zeros((115, 210), dtype=np.uint8)
        compose(page, region, 4, 3, 0)
        keep("huff_ids", {"huff_ids.jb2": jb2_file(segs + [dseg, tseg, segment(4, 49, b"")])}, page)
    ids_case()

    # jbig2enc on a real scanned page
    if "--jbig2enc" in sys.argv:
        enc = sys.argv[sys.argv.index("--jbig2enc") + 1]
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(here, "..", "..", "priv", "quality", "tables", "t03_inner.png")
        img = Image.open(src).convert("L")
        tmp = os.path.join(out, "_scan.png")
        img.save(tmp)
        page = (np.asarray(img) < 128).astype(np.uint8)
        for name, flags in (("enc_generic", []), ("enc_tpgd", ["-d"])):
            data = subprocess.run([enc] + flags + [tmp], capture_output=True).stdout
            keep(name, {name + ".jb2": data}, page)
        # symbol mode is lossy (symbols are classified): its bitmap is jbig2dec's
        for name, flags in (("enc_symbol", ["-s"]),):
            data = subprocess.run([enc] + flags + [tmp], capture_output=True).stdout
            open(os.path.join(out, name + ".jb2"), "wb").write(data)
            subprocess.run([jbig2dec, "-t", "pbm", "-o", os.path.join(out, name + ".pbm"), os.path.join(out, name + ".jb2")], check=True)
            raw = open(os.path.join(out, name + ".pbm"), "rb").read()
            manifest[name] = {"files": [name + ".jb2"], "w": img.width, "h": img.height, "sha256": hashlib.sha256(raw).hexdigest(), "lossy": True}
        cwd = os.getcwd()
        os.chdir(out)
        subprocess.run([enc, "-s", "-p", "-b", "enc_pdf", "_scan.png"], check=True, capture_output=True)
        os.chdir(cwd)
        subprocess.run([jbig2dec, "-t", "pbm", "-o", os.path.join(out, "enc_pdf.pbm"), "-e", os.path.join(out, "enc_pdf.sym"),
                        os.path.join(out, "enc_pdf.0000")], check=True)
        raw = open(os.path.join(out, "enc_pdf.pbm"), "rb").read()
        manifest["enc_pdf"] = {"files": ["enc_pdf.sym", "enc_pdf.0000"], "w": img.width, "h": img.height, "sha256": hashlib.sha256(raw).hexdigest(), "lossy": True}
        # the same page as a scanned PDF: an XObject in JBIG2Decode with its JBIG2Globals
        sym = open(os.path.join(out, "enc_pdf.sym"), "rb").read()
        pg = open(os.path.join(out, "enc_pdf.0000"), "rb").read()
        W, H = img.width, img.height
        objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W * 72 // 200} {H * 72 // 200}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode(),
                f"<< /Type /XObject /Subtype /Image /Width {W} /Height {H} /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /JBIG2Decode /DecodeParms << /JBIG2Globals 6 0 R >> /Length {len(pg)} >>\nstream\n".encode() + pg + b"\nendstream",
                None,
                f"<< /Length {len(sym)} >>\nstream\n".encode() + sym + b"\nendstream"]
        content = f"q {W * 72 // 200} 0 0 {H * 72 // 200} 0 0 cm /Im0 Do Q".encode()
        objs[4] = f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        pdf, offs = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), []
        for i, o in enumerate(objs, 1):
            offs.append(len(pdf))
            pdf += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
        xref = len(pdf)
        pdf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
        pdf += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        open(os.path.join(out, "enc_pdf.pdf"), "wb").write(bytes(pdf))
        os.remove(tmp)

    # the bitmaps are pinned by their SHA-256 in the manifest; the PBM files go
    for name in manifest:
        os.remove(os.path.join(out, name + ".pbm"))

    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1, sort_keys=True)
    print(len(manifest), "streams, every one decoded by jbig2dec to its bitmap")


main()
