"""Render text lines for vapor's OCR: training data and held-out tests.

usage: ocr_render.py OUT_DIR SPLIT COUNT SEED

SPLIT is `train` (training fonts, text from priv/quality/*_reference.txt
plus random strings over the whole charset), `test` (fonts never used in
training, text from *_holdout.txt), `val` (as `test`, another seed: the
set the decoder's language-model weights are calibrated on) or `random`
(test fonts, random strings only: the control a language model must not
make worse — there is no language in it to exploit) or `codes` (test
fonts, held-out words mixed with what an office page carries that no
language model predicts: amounts, dates, invoice and ID codes). Each line is rendered by Pillow
(FreeType, anti-aliased) at 14–44 px, then degraded like a scan or a photo:
blur, sensor noise, an illumination gradient, JPEG compression, a slight
rotation. Writes OUT_DIR/SPLIT/<i>.png and OUT_DIR/SPLIT/labels.json
({file: text}). The degradations are the same for both splits; only fonts
and text differ — what the test measures is generalisation to unseen
typefaces and unseen words.
"""
import io, json, math, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

out, split, count, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
here = os.path.dirname(os.path.abspath(__file__))
priv = os.path.join(here, "..", "..", "priv", "quality")

TT = "/usr/share/fonts/truetype/"
OT = "/usr/share/fonts/opentype/urw-base35/"
TRAIN = [TT + "dejavu/DejaVuSans.ttf", TT + "dejavu/DejaVuSerif.ttf", TT + "dejavu/DejaVuSansMono.ttf", TT + "dejavu/DejaVuSans-Bold.ttf",
         TT + "freefont/FreeSans.ttf", TT + "freefont/FreeSerif.ttf", TT + "freefont/FreeMono.ttf", TT + "freefont/FreeSansBold.ttf",
         TT + "liberation/LiberationSans-Regular.ttf", TT + "liberation/LiberationSerif-Regular.ttf", TT + "liberation/LiberationMono-Regular.ttf",
         TT + "liberation/LiberationSans-Bold.ttf", TT + "crosextra/Caladea-Regular.ttf",
         OT + "NimbusSans-Regular.otf", OT + "NimbusRoman-Regular.otf", OT + "NimbusMonoPS-Regular.otf", OT + "NimbusSans-Bold.otf",
         "/usr/share/fonts/opentype/inter/Inter-Regular.otf"]
TEST = [OT + "C059-Roman.otf", OT + "P052-Roman.otf", TT + "crosextra/Carlito-Regular.ttf", OT + "URWGothic-Book.otf",
        OT + "URWBookman-Light.otf"]

CHARSET = [chr(c) for c in range(33, 127) if chr(c) != '"'] + list("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÇ")
allowed = set(CHARSET) | {" "}


def text_source(names):
    words = []
    for n in names:
        t = open(os.path.join(priv, n), encoding="utf-8").read()
        t = "".join(ch if ch in allowed else " " for ch in t.replace("\n", " "))
        words += t.split()
    return words


words = text_source(["pt_reference.txt", "en_reference.txt"] if split == "train" else ["pt_holdout.txt", "en_holdout.txt"])


def code():
    k = rng.randrange(5)
    if k == 0:
        return f"R$ {rng.randint(1, 99)}.{rng.randint(0, 999):03d},{rng.randint(0, 99):02d}"
    if k == 1:
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2030)}"
    if k == 2:
        return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2)) + "-" + str(rng.randint(1000, 99999))
    if k == 3:
        return "".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(6, 10)))
    return f"#{rng.randint(10, 99999)}"


def line_text():
    if split == "codes":
        n = rng.randint(4, 7)
        i = rng.randrange(0, max(len(words) - n, 1))
        ws = words[i:i + n]
        for _ in range(rng.randint(1, 2)):
            ws.insert(rng.randrange(len(ws) + 1), code())
        return " ".join(ws)[:60].strip()
    if split == "random" or (split == "train" and rng.random() < 0.25):
        # random strings over the whole charset: every class seen, in every neighbourhood
        n = rng.randint(3, 8)
        return " ".join("".join(rng.choice(CHARSET) for _ in range(rng.randint(1, 7))) for _ in range(n))
    n = rng.randint(3, 9)
    i = rng.randrange(0, max(len(words) - n, 1))
    s = " ".join(words[i:i + n])
    return s[:60].strip() or "vapor"


def render(text, font_path, size):
    font = ImageFont.truetype(font_path, size)
    l, t, r, b = font.getbbox(text)
    pad = size // 2 + 4
    w, h = r - l + 2 * pad, b - t + 2 * pad
    ink = rng.randint(0, 70)
    paper = rng.randint(175, 255)
    im = Image.new("L", (w, h), paper)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=font, fill=ink)
    if rng.random() < 0.5:
        im = im.rotate(rng.uniform(-0.6, 0.6), resample=Image.BICUBIC, expand=False, fillcolor=paper)
    a = np.asarray(im, dtype=np.float64)
    if rng.random() < 0.5:     # illumination gradient (a photographed page)
        gx = np.linspace(0, 1, w)[None, :] * rng.uniform(-60, 60)
        gy = np.linspace(0, 1, h)[:, None] * rng.uniform(-40, 40)
        a = a + gx + gy
    if rng.random() < 0.6:
        a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 14), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    if rng.random() < 0.5:
        b = io.BytesIO()
        im.save(b, "JPEG", quality=rng.randint(35, 90))
        im = Image.open(io.BytesIO(b.getvalue())).convert("L")
    return im


fonts = TRAIN if split == "train" else TEST
d = os.path.join(out, split)
os.makedirs(d, exist_ok=True)
labels = {}
for i in range(count):
    text = line_text()
    f = rng.choice(fonts)
    size = rng.randint(14, 44)
    name = f"{i:05d}.png"
    render(text, f, size).save(os.path.join(d, name))
    labels[name] = {"text": text, "font": os.path.basename(f), "size": size}
json.dump({"charset": CHARSET, "lines": labels}, open(os.path.join(d, "labels.json"), "w"), ensure_ascii=False)
print(split, count)
