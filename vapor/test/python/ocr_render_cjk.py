"""Render Chinese, Japanese or Korean for vapor's CJK reader.

usage: ocr_render_cjk.py LANG OUT_DIR SPLIT [COUNT] [SEED]

LANG is zh (simplified), ja or ko. SPLIT is
  templates — every character of the class set, in every *training* font,
              clean, 24 to a line, with each character's advance cell
              (x0, x1): what `Vapor.Vision.CJK.build/3` turns into the
              class templates;
  test, val — lines of text (Faker's word lists for the language: real
              words, no grammar) in fonts *never* used for templates,
              degraded like a scan or a photo (blur, noise, illumination,
              JPEG, slight rotation);
  hand      — (ja) the lines of `test` in handwriting-style faces (Klee One,
              SetoFont): a proxy for handwriting, reported as such;
  random    — the same fonts and degradations, random characters of the
              class set: the control where a language model has nothing
              to exploit;
  lm        — OUT_DIR/lm_corpus.txt, the character language model's
              corpus: 120 000 characters of Faker's text for the language
              (seed 4242). It comes from the *same word lists* as the test
              lines (other seeds), so the language model's gain on `test`
              is an in-domain figure — docs/OCR.md §3g says so; `random`
              is the control where it must not help.

Class sets come from the national standards, enumerated through Python's
codecs: GB 2312 level 1 (3755 hanzi), JIS X 0208 level 1 (2965 kanji) plus
the kana, KS X 1001 (2350 hangul syllables); plus the full-width
punctuation of each language and the ASCII digits.
"""
import io, json, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from fontTools.ttLib import TTFont

lang, out, split = sys.argv[1], sys.argv[2], sys.argv[3]
count = int(sys.argv[4]) if len(sys.argv) > 4 else 0
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
rng = random.Random(seed)
OT, TT = "/usr/share/fonts/opentype/", "/usr/share/fonts/truetype/"


def enum(codec, rows, cols=range(0xA1, 0xFF)):
    out = []
    for r in rows:
        for c in cols:
            try:
                ch = bytes([r, c]).decode(codec)
            except UnicodeDecodeError:
                continue
            if len(ch) == 1:
                out.append(ch)
    return out


PUNCT = {"zh": list("，。、；：？！（）《》“”‘’—…·"), "ja": list("、。・「」『』（）！？ー〜"), "ko": list(".,?!()·~")}
DIGITS = list("0123456789")

if lang == "zh":
    CLASSES = enum("gb2312", range(0xB0, 0xD8))[:3755]
elif lang == "ja":
    kana = [chr(c) for c in range(0x3041, 0x3097)] + [chr(c) for c in range(0x30A1, 0x30FB)]
    CLASSES = kana + enum("euc_jp", range(0xB0, 0xD0))[:2965]
else:
    CLASSES = enum("euc_kr", range(0xB0, 0xC9))[:2350]
CLASSES = CLASSES + PUNCT[lang] + DIGITS

FONTS = {
    # one Kai among the templates (KaitiM GB); the held-out Kai (UKai) is
    # of the same lineage — said where the numbers are reported
    "zh": {"train": [(OT + "noto/NotoSansCJK-Regular.ttc", 2), (OT + "noto/NotoSansCJK-Bold.ttc", 2), (TT + "arphic/uming.ttc", 0),
                     (TT + "wqy/wqy-zenhei.ttc", 0), (TT + "arphic-gkai00mp/gkai00mp.ttf", 0)],
           "test": [(OT + "noto/NotoSerifCJK-Regular.ttc", 2), (TT + "arphic/ukai.ttc", 0)]},
    "ja": {"train": [(OT + "ipafont-gothic/ipag.ttf", 0), (OT + "ipafont-mincho/ipam.ttf", 0), (OT + "noto/NotoSansCJK-Regular.ttc", 0),
                     (OT + "noto/NotoSansCJK-Bold.ttc", 0)],
           "test": [(TT + "sawarabi-mincho/sawarabi-mincho-medium.ttf", 0), (OT + "noto/NotoSerifCJK-Regular.ttc", 0)],
           # handwriting-style faces (pen, hand-lettered): a proxy for handwriting, never in the templates
           "hand": [(TT + "klee/KleeOne-Regular.ttf", 0), (TT + "seto/setofont.ttf", 0)]},
    "ko": {"train": [(TT + "nanum/NanumGothic.ttf", 0), (TT + "nanum/NanumMyeongjo.ttf", 0), (OT + "noto/NotoSansCJK-Regular.ttc", 1),
                     (TT + "nanum/NanumSquareR.ttf", 0)],
           "test": [(OT + "noto/NotoSerifCJK-Regular.ttc", 1), (TT + "nanum/NanumBarunGothic.ttf", 0)]},
}

cmaps = {}


def has(font, ch):
    if font not in cmaps:
        cmaps[font] = set(TTFont(font[0], fontNumber=font[1], lazy=True).getBestCmap().keys())
    return ord(ch) in cmaps[font]


def degrade(im, paper):
    w, h = im.size
    if rng.random() < 0.5:
        im = im.rotate(rng.uniform(-0.5, 0.5), resample=Image.BICUBIC, expand=False, fillcolor=paper)
    a = np.asarray(im, dtype=np.float64)
    if rng.random() < 0.5:
        a = a + np.linspace(0, 1, w)[None, :] * rng.uniform(-50, 50) + np.linspace(0, 1, h)[:, None] * rng.uniform(-30, 30)
    if rng.random() < 0.6:
        a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 10), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7)))
    if rng.random() < 0.5:
        b = io.BytesIO()
        im.save(b, "JPEG", quality=rng.randint(45, 90))
        im = Image.open(io.BytesIO(b.getvalue())).convert("L")
    return im


def draw(text, font, size, clean):
    f = ImageFont.truetype(font[0], size, index=font[1])
    pad = size // 2 + 6
    width = int(f.getlength(text)) + 2 * pad
    asc, desc = f.getmetrics()
    height = asc + desc + 2 * pad
    ink = 0 if clean else rng.randint(0, 60)
    paper = 255 if clean else rng.randint(180, 255)
    im = Image.new("L", (width, height), paper)
    ImageDraw.Draw(im).text((pad, pad), text, font=f, fill=ink)
    cells = []
    for i in range(len(text)):
        x0 = pad + f.getlength(text[:i])
        x1 = pad + f.getlength(text[: i + 1])
        cells.append([round(x0, 2), round(x1, 2)])
    return (im if clean else degrade(im, paper)), cells


if split == "lm":
    from faker import Faker
    f = Faker({"zh": "zh_CN", "ja": "ja_JP", "ko": "ko_KR"}[lang])
    f.seed_instance(4242)
    parts = []
    while sum(map(len, parts)) < 120000:
        parts.append(f.text(400) if lang != "ko" else " ".join([f.address(), f.catch_phrase(), f.company(), f.name()]))
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "lm_corpus.txt"), "w").write("\n".join(parts))
    print(lang, "lm", sum(map(len, parts)))
    sys.exit(0)

d = os.path.join(out, split)
os.makedirs(d, exist_ok=True)
lines = {}

if split == "templates":
    k = 0
    for font in FONTS[lang]["train"]:
        chars = [c for c in CLASSES if has(font, c)]
        for i in range(0, len(chars), 24):
            text = "".join(chars[i:i + 24])
            im, cells = draw(text, font, 48, True)
            name = f"{k:05d}.png"
            im.save(os.path.join(d, name))
            lines[name] = {"text": text, "cells": cells, "font": os.path.basename(font[0]), "index": font[1]}
            k += 1
else:
    from faker import Faker
    fk = Faker({"zh": "zh_CN", "ja": "ja_JP", "ko": "ko_KR"}[lang])
    fk.seed_instance(seed)
    cls = set(CLASSES)
    pool = []
    while len(pool) < 20000:
        t = fk.text(200) if lang != "ko" else " ".join([fk.address(), fk.catch_phrase(), fk.company(), fk.name()])
        pool += [c for c in t if c in cls]
    fonts = FONTS[lang]["hand" if split == "hand" else "test"]
    for i in range(count):
        n = rng.randint(6, 18)
        if split == "random":
            text = "".join(rng.choice(CLASSES) for _ in range(n))
        else:
            st = rng.randrange(len(pool) - n)
            text = "".join(pool[st:st + n])
        font = rng.choice(fonts)
        if not all(has(font, c) for c in text):
            continue
        im, cells = draw(text, font, rng.randint(26, 52), False)
        name = f"{i:05d}.png"
        im.save(os.path.join(d, name))
        lines[name] = {"text": text, "font": os.path.basename(font[0]), "index": font[1]}

json.dump({"lang": lang, "classes": CLASSES, "lines": lines}, open(os.path.join(d, "labels.json"), "w"), ensure_ascii=False)
print(lang, split, len(lines))
