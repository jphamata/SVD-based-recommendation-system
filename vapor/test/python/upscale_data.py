"""High-resolution greyscale images for the consistent upscaler (Vapor.Vision.Upscale).

usage: upscale_data.py OUT_DIR

Writes OUT_DIR/{train,val,test}/*.pgm (8-bit luma, BT.601, even sides).
Photographs and scientific images from scikit-image's bundled data, split
by image (no image in two splits), plus rendered text pages — the fonts of
the test pages never appear in training. The split is the experiment:

  train  astronaut coffee rocket coins moon horse clock hubble immunohistochemistry
         brick gravel cell retina + text in DejaVu Sans/Serif, Liberation, FreeSans
  val    grass page colorwheel + text in C059 (Century Schoolbook)
  test   camera chelsea text shepp_logan + text in Inter, P052 (Palatino), Courier
"""
import os, sys, warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import skimage.data as data

warnings.filterwarnings("ignore")
out = sys.argv[1]

def luma(x):
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[..., :3].astype(np.float64)
        x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    x = x.astype(np.float64)
    if x.max() <= 1.0:
        x = x * 255.0
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

def even(x, max_side=512):
    h, w = x.shape
    if max(h, w) > max_side:
        # a central crop, not a resize: resizing would change the statistics we learn
        y0, x0 = max(0, (h - max_side) // 2), max(0, (w - max_side) // 2)
        x = x[y0:y0 + max_side, x0:x0 + max_side]
        h, w = x.shape
    return x[: h - h % 2, : w - w % 2]

def save(split, name, x):
    d = os.path.join(out, split)
    os.makedirs(d, exist_ok=True)
    Image.fromarray(even(x), "L").save(os.path.join(d, name + ".pgm"))

def font(names, size):
    for n in names:
        for root in ["/usr/share/fonts", "/usr/local/share/fonts"]:
            for dp, _, fs in os.walk(root):
                for f in fs:
                    if f.lower().startswith(n.lower()) and f.lower().endswith((".ttf", ".otf", ".t1", ".pfb")):
                        try:
                            return ImageFont.truetype(os.path.join(dp, f), size)
                        except Exception:
                            pass
    raise SystemExit("font not found: %s" % names)

WORDS = ("the quick brown fox jumps over a lazy dog contract clause penalty invoice total amount due date "
         "parágrafo cláusula multa vencimento recibo número tabela página relatório 2026 R$ 1.234,56 "
         "AZ-69511 #28129 voltage current sample error ratio model signal noise").split()

def page(fonts, seed, w=480, h=360):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    y = 8
    while y < h - 40:
        size = int(rng.integers(14, 34))
        f = font(fonts, size)
        line = " ".join(rng.choice(WORDS, 9))
        d.text((8, y), line, fill=int(rng.integers(0, 60)), font=f)
        y += int(size * 1.35)
    return np.asarray(img)

splits = {
    "train": ["astronaut", "coffee", "rocket", "coins", "moon", "horse", "clock", "hubble_deep_field",
              "immunohistochemistry", "brick", "gravel", "cell", "retina"],
    "val": ["grass", "page", "colorwheel"],
    "test": ["camera", "chelsea", "text", "shepp_logan_phantom"],
}
for split, names in splits.items():
    for n in names:
        save(split, n, luma(getattr(data, n)()))

for i, fonts in enumerate([["DejaVuSans."], ["DejaVuSerif."], ["LiberationSerif-Regular"], ["FreeSans."], ["LiberationSans-Regular"]]):
    save("train", "text_%d" % i, page(fonts, 100 + i))
save("val", "text_c059", page(["C059-Roman"], 200))
for i, fonts in enumerate([["Inter-Regular", "Inter."], ["P052-Roman"], ["NimbusMonoPS-Regular", "Courier"]]):
    save("test", "text_%d" % i, page(fonts, 300 + i))
print("ok")
