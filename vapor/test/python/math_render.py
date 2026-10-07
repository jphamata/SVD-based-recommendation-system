"""Render typeset formulas for vapor's formula reader (`Vapor.Vision.Math`).

usage: math_render.py OUT_DIR SPLIT COUNT SEED

SPLIT is
  templates — every symbol class, alone, in the *template* font sets
              (matplotlib's mathtext: dejavusans, dejavuserif, stixsans, and
              custom sets of Liberation Serif, FreeSerif and Noto Serif) at
              three sizes, with its class name: the reader's templates;
  dev       — random formulas in dejavusans (the development set);
  test      — random formulas in the font sets never used for templates:
              Computer Modern (`cm`) and STIX (`stix`).

Formulas come from a small grammar — sums of terms; a term is an atom
with an optional sub- and/or superscript, a fraction, a square root, a
parenthesised sum, or a sum/integral with limits — so the truth is a
LaTeX string in one canonical spelling (the reader's output spelling):
`x^{2}`, `\\frac{a}{b}`, `\\sqrt{x}`, `\\sum_{i=1}^{n}`, `\\alpha`.
"""
import io, json, os, random, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

out, split, count, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
d = os.path.join(out, split)
os.makedirs(d, exist_ok=True)

LETTERS = list("abcdxyznkt")
DIGITS = list("0123456789")
GREEK = ["alpha", "beta", "gamma", "theta", "lambda", "mu", "pi", "sigma", "omega"]
OPS = ["+", "-", "="]

# symbol classes: name -> LaTeX
CLASSES = {c: c for c in LETTERS + DIGITS}
CLASSES.update({g: "\\" + g for g in GREEK})
CLASSES.update({"+": "+", "-": "-", "=": "=", "(": "(", ")": ")", "sum": "\\sum", "int": "\\int"})


def atom(depth):
    r = rng.random()
    if r < 0.45:
        return rng.choice(LETTERS)
    if r < 0.75:
        return "".join(rng.choice(DIGITS) for _ in range(rng.choice([1, 1, 1, 2])))
    return "\\" + rng.choice(GREEK)


def small(depth):
    # a script: an atom, or a short sum
    if depth < 2 and rng.random() < 0.25:
        return atom(depth + 1) + rng.choice(["+", "-"]) + atom(depth + 1)
    return atom(depth + 1)


def term(depth):
    r = rng.random()
    if depth < 2 and r < 0.15:
        return "\\frac{" + expr(depth + 1, 2) + "}{" + expr(depth + 1, 2) + "}"
    if depth < 2 and r < 0.25:
        return "\\sqrt{" + expr(depth + 1, 2) + "}"
    if depth < 1 and r < 0.32:
        return "(" + expr(depth + 1, 2) + ")"
    if depth < 1 and r < 0.38:
        v = rng.choice(["i", "k", "n"])
        return "\\sum_{" + v + "=1}^{" + rng.choice(["n", "k", "9"]) + "}" + atom(depth + 1)
    if depth < 1 and r < 0.42:
        return "\\int_{0}^{1}" + atom(depth + 1)
    a = atom(depth)
    s = rng.random()
    if s < 0.25:
        return a + "^{" + small(depth) + "}"
    if s < 0.4:
        return a + "_{" + small(depth) + "}"
    if s < 0.45:
        return a + "_{" + atom(depth + 1) + "}^{" + atom(depth + 1) + "}"
    return a


def expr(depth, maxterms):
    n = rng.randint(1, maxterms)
    s = term(depth)
    for _ in range(n - 1):
        s += rng.choice(OPS[:2] if depth else OPS) + term(depth)
    return s


def render(tex, fontset, size):
    if isinstance(fontset, tuple):
        # a custom set: letters, digits and Greek from a text family; the
        # rest (big operators) from STIX Sans, itself a template family
        matplotlib.rcParams["mathtext.fontset"] = "custom"
        matplotlib.rcParams["mathtext.rm"] = fontset[1]
        matplotlib.rcParams["mathtext.it"] = fontset[1] + ":italic"
        matplotlib.rcParams["mathtext.bf"] = fontset[1] + ":bold"
        matplotlib.rcParams["mathtext.fallback"] = "stixsans"
    else:
        matplotlib.rcParams["mathtext.fontset"] = fontset
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, "$" + tex + "$", fontsize=size)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


items = {}
if split == "templates":
    k = 0
    for fs in ["dejavusans", "dejavuserif", "stixsans", ("custom", "Liberation Serif"), ("custom", "FreeSerif"), ("custom", "Noto Serif")]:
        for size in [20, 28, 36]:
            for name, tex in CLASSES.items():
                im = render(tex, fs, size)
                fn = f"{k:05d}.png"
                im.save(os.path.join(d, fn))
                items[fn] = {"class": name, "font": fs if isinstance(fs, str) else fs[1], "size": size}
                k += 1
else:
    fonts = ["dejavusans"] if split == "dev" else ["cm", "stix"]
    for i in range(count):
        tex = expr(0, 4)
        fs = rng.choice(fonts)
        size = rng.choice([20, 24, 28, 32])
        im = render(tex, fs, size)
        fn = f"{i:04d}.png"
        im.save(os.path.join(d, fn))
        items[fn] = {"latex": tex, "font": fs, "size": size}

json.dump({"classes": CLASSES, "items": items}, open(os.path.join(d, "truth.json"), "w"))
print(split, len(items))
