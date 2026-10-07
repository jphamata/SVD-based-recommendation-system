"""Render text lines for vapor's readers of other scripts.

usage: ocr_render_script.py SCRIPT OUT_DIR SPLIT COUNT SEED

SCRIPT is `arabic` (connected, right to left: shaped by HarfBuzz through
Pillow's raqm layout), `cyrillic` (Russian) or `cursive` (Latin in
handwriting and connected script typefaces). SPLIT is `train` (training
fonts), `test` / `val` (fonts never used in training — for handwriting,
the hands of other writers) or `hand` (Arabic, Cyrillic: handwriting-style
or calligraphic faces never used in training — a proxy for handwriting,
reported as such). Degradations are those of ocr_render.py (blur, noise,
illumination, JPEG, slight rotation).

Labels are in *visual* order, left to right, the order of the frames a
CTC reader sees: for Arabic that is the logical text run through the
Unicode bidirectional algorithm (python-bidi, without shaping), so digits
inside Arabic stay left to right. vapor turns a visual reading back into
logical order (`Vapor.Vision.Bidi`); `labels.json` keeps both.

Text: Arabic from Faker's ar_AA word lists (real words, sentences without
grammar — what is measured is reading, not understanding), with Eastern
Arabic and Western digits and Arabic punctuation; handwriting from vapor's
Portuguese/English reference corpora (held-out corpora for test), as the
Latin reader. A line is only drawn in a font that has every one of its
characters (fontTools cmap), so no .notdef box is ever labelled as a letter.
"""
import io, json, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features
from fontTools.ttLib import TTFont

script, out, split, count, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rng = random.Random(seed)
here = os.path.dirname(os.path.abspath(__file__))
priv = os.path.join(here, "..", "..", "priv", "quality")
TT, OT = "/usr/share/fonts/truetype/", "/usr/share/fonts/opentype/"

FONTS = {
    "arabic": {
        # 0.10: 17 typefaces left a 28 % error on unseen ones (the reader
        # learnt the fonts, not the script) — every text face of the
        # system's Arabic collections is in now, except the four held out,
        # their bold/other weights, Nastaliq (another style) and the
        # decorative faces (shadowed, outlined, pixel)
        "train": [OT + "fonts-hosny-amiri/Amiri-Regular.ttf", OT + "fonts-hosny-amiri/Amiri-Bold.ttf", OT + "fonts-hosny-amiri/Amiri-Italic.ttf",
                  TT + "noto/NotoNaskhArabic-Regular.ttf", TT + "noto/NotoNaskhArabic-Bold.ttf", TT + "noto/NotoSansArabic-Regular.ttf",
                  TT + "noto/NotoSansArabic-Bold.ttf", TT + "noto/NotoKufiArabic-Regular.ttf", TT + "noto/NotoKufiArabic-Bold.ttf",
                  TT + "kacst-one/KacstOne.ttf", TT + "kacst-one/KacstOne-Bold.ttf", TT + "kacst/KacstBook.ttf", TT + "kacst/KacstOffice.ttf",
                  TT + "kacst/KacstFarsi.ttf", TT + "kacst/KacstLetter.ttf", TT + "kacst/KacstPen.ttf", TT + "kacst/KacstScreen.ttf",
                  TT + "kacst/KacstTitle.ttf", TT + "kacst/KacstQurn.ttf", TT + "kacst/KacstPoster.ttf",
                  OT + "lateef/Lateef-Regular.ttf", OT + "lateef/Lateef-Bold.ttf", OT + "lateef/Lateef-Light.ttf",
                  TT + "harmattan/Harmattan-Regular.ttf", TT + "harmattan/Harmattan-Bold.ttf",
                  TT + "vazirmatn/Vazirmatn-Regular.ttf", TT + "vazirmatn/Vazirmatn-Bold.ttf", TT + "vazirmatn/Vazirmatn-Light.ttf", TT + "vazirmatn/Vazirmatn-Black.ttf",
                  OT + "lemonada/Lemonada-Regular.otf", OT + "lemonada/Lemonada-Bold.otf", TT + "freefont/FreeSerif.ttf", TT + "freefont/FreeSerifBold.ttf"] +
                 [TT + "fonts-arabeyes/ae_" + n + ".ttf" for n in
                  ["AlArabiya", "AlBattar", "AlHor", "AlManzomah", "AlYarmook", "Arab", "Dimnah", "Furat", "Hani", "Hor", "Kayrawan", "Khalid",
                   "Mashq", "Mashq-Bold", "Nada", "Nagham", "Ostorah", "Ouhod-Bold", "Petra", "Rasheeq-Bold", "Rehan", "Salem", "Sharjah",
                   "Sindbad", "Tarablus", "Tholoth"]],
        # unseen designs: another Naskh, a Kacst Naskh, two Arabeyes faces
        # (the Arabeyes faces share a lineage with the training ones — said
        # where the numbers are)
        "test": [TT + "scheherazade/Scheherazade-Regular.ttf", TT + "kacst/KacstNaskh.ttf",
                 TT + "fonts-arabeyes/ae_Cortoba.ttf", TT + "fonts-arabeyes/ae_Granada.ttf"],
        # calligraphic Nastaliq (a proxy for a hand: the slanted, stacked style of Persian and Urdu), never in training
        "hand": [TT + "irannastaliq/IranNastaliq.ttf", TT + "noto/NotoNastaliqUrdu-Regular.ttf"],
    },
    "cursive": {
        "train": [OT + "dancingscript/DancingScript-Regular.otf", OT + "dancingscript/DancingScript-Bold.otf",
                  TT + "kristi/Kristi.ttf", OT + "kaushanscript/KaushanScript-Regular.otf", OT + "urw-base35/Z003-MediumItalic.otf",
                  OT + "bwht/BecauseWeBuild-Regular.otf", OT + "bwht/BecauseWeConnect-Regular.otf", OT + "bwht/BecauseWeCreate-Regular.otf",
                  OT + "bwht/BecauseWeLearn-Regular.otf", TT + "fifthhorseman/dkg.ttf", TT + "fifthhorseman/dkgIt.ttf",
                  TT + "femkeklaver/femkeklaver.ttf", TT + "sjfonts/Delphine.ttf", TT + "breip/Breip.ttf",
                  TT + "humor-sans/Humor-Sans.ttf", TT + "rufscript/Rufscript010.ttf", TT + "klee/KleeOne-Regular.ttf"],
        # other writers' hands, and a connected school cursive never seen
        "test": [TT + "sjfonts/SteveHand.ttf", OT + "bwht/BecauseWeMentor-Regular.otf", OT + "bwht/BecauseWeOrganize-Regular.otf",
                 TT + "ecolier-court/Ecolier-court.ttf"],
    },
    "cyrillic": {
        "train": [TT + "dejavu/DejaVuSans.ttf", TT + "dejavu/DejaVuSans-Bold.ttf", TT + "dejavu/DejaVuSerif.ttf", TT + "dejavu/DejaVuSerif-Bold.ttf",
                  TT + "dejavu/DejaVuSansCondensed.ttf", TT + "dejavu/DejaVuSansMono.ttf", TT + "liberation/LiberationSans-Regular.ttf",
                  TT + "liberation/LiberationSans-Bold.ttf", TT + "liberation/LiberationSerif-Regular.ttf", TT + "liberation/LiberationMono-Regular.ttf",
                  TT + "freefont/FreeSans.ttf", TT + "freefont/FreeSerif.ttf", TT + "freefont/FreeMono.ttf", TT + "freefont/FreeSerifBold.ttf",
                  TT + "noto/NotoSans-Regular.ttf", TT + "noto/NotoSerif-Regular.ttf", TT + "noto/NotoMono-Regular.ttf",
                  OT + "inter/Inter-Regular.otf", OT + "inter/Inter-Bold.otf", OT + "inter/Inter-Light.otf",
                  OT + "urw-base35/C059-Roman.otf", OT + "urw-base35/C059-Bold.otf", OT + "urw-base35/C059-Italic.otf",
                  OT + "urw-base35/NimbusRoman-Regular.otf", OT + "urw-base35/NimbusSans-Regular.otf", OT + "urw-base35/NimbusSans-Bold.otf",
                  OT + "urw-base35/NimbusMonoPS-Regular.otf", OT + "urw-base35/NimbusSansNarrow-Regular.otf", OT + "urw-base35/P052-Roman.otf",
                  OT + "urw-base35/URWGothic-Book.otf"],
        # unseen designs: a book serif, a humanist sans, Bookman, a display sans
        "test": [TT + "google-fonts/Lora-Variable.ttf", TT + "crosextra/Carlito-Regular.ttf", OT + "urw-base35/URWBookman-Light.otf",
                 TT + "noto/NotoSansDisplay-Regular.ttf"],
        # handwriting-style faces (a proxy for handwriting, said where measured):
        # never in training
        "hand": [TT + "sjfonts/SteveHand.ttf", TT + "klee/KleeOne-Regular.ttf", TT + "seto/setofont.ttf"],
    },
}

AR_LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
AR_PUNCT = "،؛؟.:!()-"
LATIN = [chr(c) for c in range(33, 127) if chr(c) != '"'] + list("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÇ")

CY_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

if script == "arabic":
    assert features.check("raqm"), "Arabic needs Pillow with raqm (HarfBuzz + FriBiDi)"
    from bidi.algorithm import get_display
    from faker import Faker
    CHARSET = list(AR_LETTERS) + list(AR_DIGITS) + list("0123456789") + list(AR_PUNCT)
    fk = Faker("ar_AA")
    fk.seed_instance(seed)
    words = sorted({w.strip(".،") for _ in range(400) for w in fk.text(400).split() if w.strip(".،")})
    rng.shuffle(words)
    # training and held-out lines draw from disjoint halves of the vocabulary
    words = words[: len(words) // 2] if split == "train" else words[len(words) // 2:]
elif script == "cyrillic":
    from faker import Faker
    CHARSET = list(CY_LETTERS) + list("0123456789") + list(".,:;!?()-«»—№%")
    fk = Faker("ru_RU")
    fk.seed_instance(seed)
    words = sorted({w.strip(".,") for _ in range(400) for w in fk.text(400).split() if w.strip(".,")})
    rng.shuffle(words)
    # training and held-out lines draw from disjoint halves of the vocabulary
    words = words[: len(words) // 2] if split == "train" else words[len(words) // 2:]
else:
    CHARSET = LATIN
    names = ["pt_reference.txt", "en_reference.txt"] if split == "train" else ["pt_holdout.txt", "en_holdout.txt"]
    allowed = set(CHARSET) | {" "}
    words = []
    for n in names:
        t = open(os.path.join(priv, n), encoding="utf-8").read()
        words += "".join(ch if ch in allowed else " " for ch in t.replace("\n", " ")).split()

charset_set = set(CHARSET) | {" "}


def ar_number():
    k = rng.randrange(3)
    if k == 0:
        return "".join(rng.choice(AR_DIGITS) for _ in range(rng.randint(1, 4)))
    if k == 1:
        return str(rng.randint(1, 9999))
    return f"{rng.randint(1, 28)}/{rng.randint(1, 12)}/{rng.randint(1990, 2030)}"


def line_text():
    if script == "arabic":
        ws = [rng.choice(words) for _ in range(rng.randint(3, 7))]
        if rng.random() < 0.4:
            ws.insert(rng.randrange(len(ws) + 1), ar_number())
        s = " ".join(ws)
        if rng.random() < 0.5:
            s += rng.choice("،.؟:")
        return s[:48].strip()
    if script == "cyrillic":
        ws = [rng.choice(words) for _ in range(rng.randint(3, 7))]
        if rng.random() < 0.3:
            ws.insert(rng.randrange(len(ws) + 1), str(rng.randint(1, 2030)))
        s = " ".join(ws)
        if rng.random() < 0.5:
            s += rng.choice(".,:!?")
        return s[:46].strip()
    n = rng.randint(3, 7)
    i = rng.randrange(0, max(len(words) - n, 1))
    return (" ".join(words[i:i + n]))[:44].strip() or "vapor"


cmaps = {}


def covers(path, text):
    if path not in cmaps:
        f = TTFont(path, fontNumber=0, lazy=True)
        cmaps[path] = set(f.getBestCmap().keys())
    return all(ord(ch) in cmaps[path] for ch in text if ch != " ")


def render(text, font_path, size):
    kw = {"direction": "rtl", "language": "ar"} if script == "arabic" else {}
    font = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
    l, t, r, b = font.getbbox(text, **kw)
    pad = size // 2 + 4
    w, h = r - l + 2 * pad, b - t + 2 * pad
    ink = rng.randint(0, 70)
    paper = rng.randint(175, 255)
    im = Image.new("L", (w, h), paper)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=font, fill=ink, **kw)
    if rng.random() < 0.5:
        im = im.rotate(rng.uniform(-0.6, 0.6), resample=Image.BICUBIC, expand=False, fillcolor=paper)
    a = np.asarray(im, dtype=np.float64)
    if rng.random() < 0.5:
        a = a + np.linspace(0, 1, w)[None, :] * rng.uniform(-60, 60) + np.linspace(0, 1, h)[:, None] * rng.uniform(-40, 40)
    if rng.random() < 0.6:
        a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 12), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    if rng.random() < 0.5:
        bio = io.BytesIO()
        im.save(bio, "JPEG", quality=rng.randint(40, 90))
        im = Image.open(io.BytesIO(bio.getvalue())).convert("L")
    return im


fonts = FONTS[script]["train" if split == "train" else ("hand" if split == "hand" else "test")]
d = os.path.join(out, split)
os.makedirs(d, exist_ok=True)
labels = {}
i = 0
while i < count:
    text = line_text()
    f = rng.choice(fonts)
    if not text or not covers(f, text) or not set(text) <= charset_set:
        continue
    size = rng.randint(22, 44) if script == "arabic" else rng.randint(18, 44)
    if not covers(f, text):
        continue
    name = f"{i:05d}.png"
    render(text, f, size).save(os.path.join(d, name))
    visual = get_display(text) if script == "arabic" else text
    labels[name] = {"text": visual, "logical": text, "font": os.path.basename(f), "size": size}
    i += 1
json.dump({"charset": CHARSET, "script": script, "order": "visual", "lines": labels}, open(os.path.join(d, "labels.json"), "w"), ensure_ascii=False)
print(script, split, count)
