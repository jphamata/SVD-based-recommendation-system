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
   aggregation; a striped page of unknown height. The MQ coder is
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
    m.int(cx["dt"], 0)  # initial STRIPT = 0
    stript, firsts = 0, 0
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
