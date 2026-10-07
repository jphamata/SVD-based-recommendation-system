"""Render charts and pages with figures, with their ground truth, for
vapor's figure reader (`Vapor.Vision.Figure`).

usage: chart_render.py OUT_DIR SPLIT COUNT SEED

SPLIT is
  charts    — line, bar and scatter charts drawn by matplotlib, with the
              data behind them (`truth.json`: kind, axis limits and scale,
              every series' colour and points);
  test      — the same kinds in another typeface, other sizes and DPIs,
              some with grid lines, log axes and JPEG compression: the
              digitizer was never tuned on these;
  shuffled  — the control: real charts whose *tick labels are permuted*
              (a FixedFormatter over the true ticks), so no linear or
              logarithmic scale explains them — the digitizer must refuse,
              not report numbers;
  pages     — pages of prose (vapor's frozen held-out corpora) with one
              figure (a chart, or a photograph-like picture) and its
              caption ("Figure 3: …", "Figura 2 — …", "Fig. 5. …") below
              or above it: the figure's box and the caption's box and text.

vapor never runs this script; it reads the PNGs and `truth.json`.
"""
import io, json, math, os, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.formatter.useoffset"] = False   # no "+1.99e3" offset text: tick labels are the values
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from PIL import Image, ImageDraw, ImageFont

out, split, count, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
nrng = np.random.default_rng(seed)
here = os.path.dirname(os.path.abspath(__file__))
priv = os.path.join(here, "..", "..", "priv", "quality")
d = os.path.join(out, split)
os.makedirs(d, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
FONT = {"charts": "DejaVu Sans", "shuffled": "DejaVu Sans", "pages": "DejaVu Sans", "test": "Liberation Serif"}[split]


def nice_range():
    k = rng.randint(-1, 4)
    span = rng.uniform(1, 9) * 10 ** k
    lo = rng.choice([0.0, 0.0, -span * rng.uniform(0.2, 1.0), span * rng.uniform(0.1, 2.0)])
    return lo, lo + span


def series_line(n_series, log):
    x0 = rng.choice([0.0, rng.uniform(-50, 50), float(rng.randint(1990, 2010))])
    xs = np.linspace(x0, x0 + rng.uniform(5, 100), rng.randint(30, 120))
    lo, hi = (10 ** rng.uniform(-1, 1), 10 ** rng.uniform(2, 5)) if log else nice_range()
    out = []
    for _ in range(n_series):
        t = (xs - xs[0]) / (xs[-1] - xs[0])
        kind = rng.randrange(3)
        if kind == 0:
            f = np.sin(2 * math.pi * rng.uniform(0.3, 2.0) * t + rng.uniform(0, 6)) * 0.5 + 0.5
        elif kind == 1:
            f = np.cumsum(nrng.normal(0, 1, len(t)))
            f = (f - f.min()) / max(f.max() - f.min(), 1e-9)
        else:
            f = t ** rng.uniform(0.3, 3.0)
        f = 0.1 + 0.8 * f
        ys = np.exp(np.log(lo) + f * (np.log(hi) - np.log(lo))) if log else lo + f * (hi - lo)
        out.append((xs, ys))
    return out


def chart(i, kind, shuffled=False):
    plt.rcParams["font.family"] = FONT
    w, h = rng.uniform(4.0, 7.0), rng.uniform(3.0, 5.0)
    dpi = rng.choice([80, 100, 120]) if split != "test" else rng.choice([72, 90, 110, 140])
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    grid = split == "test" and rng.random() < 0.4
    log = kind == "line" and split == "test" and rng.random() < 0.25
    truth = {"kind": kind, "series": [], "log_y": log}
    if kind == "line":
        ss = series_line(rng.randint(1, 3), log)
        for k, (xs, ys) in enumerate(ss):
            ax.plot(xs, ys, color=COLORS[k], linewidth=rng.choice([1.0, 1.5, 2.0]))
            truth["series"].append({"color": COLORS[k], "x": xs.tolist(), "y": ys.tolist()})
        if log:
            ax.set_yscale("log")
    elif kind == "bar":
        n = rng.randint(3, 10)
        lo, hi = nice_range()
        lo = 0.0
        ys = [rng.uniform(0.1, 1.0) * hi for _ in range(n)]
        labels = [chr(65 + j) for j in range(n)]
        c = rng.choice(COLORS)
        ax.bar(range(n), ys, color=c, width=rng.uniform(0.5, 0.85))
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels)
        truth["series"].append({"color": c, "x": list(range(n)), "y": ys})
    else:
        lo, hi = nice_range()          # one scale for every series of the chart
        for k in range(rng.randint(1, 2)):
            m = rng.randint(10, 40)
            xs = nrng.uniform(0, rng.uniform(1, 100), m)
            ys = lo + nrng.uniform(0.1, 0.9, m) * (hi - lo)
            ax.scatter(xs, ys, color=COLORS[k], s=rng.choice([12, 20, 30]))
            truth["series"].append({"color": COLORS[k], "x": xs.tolist(), "y": ys.tolist()})
    if grid:
        ax.grid(True, color="#cccccc", linewidth=0.6)
    if rng.random() < 0.6:
        ax.set_xlabel(rng.choice(["time (s)", "year", "dose", "x", "epoch"]))
        ax.set_ylabel(rng.choice(["value", "loss", "count", "y", "rate"]))
    if rng.random() < 0.4:
        ax.set_title(rng.choice(["Results", "Measured response", "Growth", "Comparison"]))
    fig.tight_layout()
    fig.canvas.draw()
    if shuffled:
        ticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        labels = [l.get_text() for l in ax.get_yticklabels() if ax.get_ylim()[0] <= l.get_position()[1] <= ax.get_ylim()[1]]
        perm = labels[:]
        while len(perm) > 2 and perm == labels:
            rng.shuffle(perm)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(perm))
        truth["shuffled_labels"] = perm
    truth["xlim"], truth["ylim"] = list(ax.get_xlim()), list(ax.get_ylim())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if split == "test" and rng.random() < 0.3:
        b = io.BytesIO()
        im.save(b, "JPEG", quality=rng.randint(70, 92))
        im = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    return im, truth


def photo(w, h):
    # a photograph-like picture: smooth colour fields with texture and a few shapes
    yy, xx = np.mgrid[0:h, 0:w] / max(w, h)
    img = np.zeros((h, w, 3))
    for c in range(3):
        a = sum(rng.uniform(-1, 1) * np.sin(rng.uniform(1, 8) * xx + rng.uniform(1, 8) * yy + rng.uniform(0, 6)) for _ in range(4))
        img[..., c] = 128 + 50 * a
    img += nrng.normal(0, 12, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for _ in range(rng.randint(2, 6)):
        x0, y0 = rng.randrange(w), rng.randrange(h)
        dr.ellipse([x0, y0, x0 + rng.randint(10, w // 3), y0 + rng.randint(10, h // 3)], fill=tuple(rng.randrange(256) for _ in range(3)))
    return im


def words():
    t = open(os.path.join(priv, "pt_holdout.txt"), encoding="utf-8").read() + " " + open(os.path.join(priv, "en_holdout.txt"), encoding="utf-8").read()
    return [w for w in t.split() if w.isprintable()]


def para(draw, font, ws, x, y, width, lines):
    lh = int(font.size * 1.45)
    for _ in range(lines):
        line = []
        while True:
            w = ws[rng.randrange(len(ws))]
            if font.getlength(" ".join(line + [w])) > width:
                break
            line.append(w)
        draw.text((x, y), " ".join(line), font=font, fill=0)
        y += lh
    return y


def page(i, ws):
    W, H = 900, 1200
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    fp = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    font = ImageFont.truetype(fp, rng.choice([15, 16, 17]))
    m = 70
    y = para(dr, font, ws, m, 60, W - 2 * m, rng.randint(4, 9)) + int(font.size * 1.6)
    if rng.random() < 0.65:
        fig, t = chart(i, rng.choice(["line", "bar", "scatter"]))
        kind = "chart"
    else:
        fig, t = photo(rng.randint(380, 640), rng.randint(220, 380)), {}
        kind = "photo"
    fw = min(fig.width, W - 2 * m)
    fig = fig.resize((fw, int(fig.height * fw / fig.width)))
    # the figure's truth box is its ink (a chart's white margins are page)
    ink = np.asarray(fig.convert("L")) < 235
    iy, ix = np.nonzero(ink)
    ink_box = [int(ix.min()), int(iy.min()), int(ix.max()), int(iy.max())]
    n = rng.randint(1, 9)
    cap = rng.choice([f"Figure {n}: ", f"Figura {n} — ", f"Fig. {n}. "]) + " ".join(ws[rng.randrange(len(ws))] for _ in range(rng.randint(4, 9)))
    capfont = ImageFont.truetype(fp, max(font.size - 2, 12))
    above = rng.random() < 0.2
    fx = (W - fw) // 2
    if above:
        cap_box = [m, y, m + int(capfont.getlength(cap)), y + capfont.size + 4]
        dr.text((m, y), cap, font=capfont, fill=0)
        y += int(capfont.size * 2.0)
        im.paste(fig, (fx, y))
        fig_box = [fx + ink_box[0], y + ink_box[1], fx + ink_box[2], y + ink_box[3]]
        y += fig.height + int(font.size * 1.6)
    else:
        im.paste(fig, (fx, y))
        fig_box = [fx + ink_box[0], y + ink_box[1], fx + ink_box[2], y + ink_box[3]]
        y += fig.height + int(capfont.size * 1.0)
        cap_box = [m, y, m + int(capfont.getlength(cap)), y + capfont.size + 4]
        dr.text((m, y), cap, font=capfont, fill=0)
        y += int(font.size * 2.6)
    para(dr, font, ws, m, y, W - 2 * m, max(0, (H - 60 - y) // int(font.size * 1.45)))
    return im, {"kind": kind, "figure": fig_box, "caption": {"box": cap_box, "text": cap, "above": above}, "chart": t}


truth = {}
if split == "pages":
    ws = words()
    for i in range(count):
        im, t = page(i, ws)
        name = f"{i:04d}.png"
        im.save(os.path.join(d, name))
        truth[name] = t
else:
    kinds = ["line", "bar", "scatter"]
    for i in range(count):
        kind = "line" if split == "shuffled" else kinds[i % 3]
        im, t = chart(i, kind, shuffled=split == "shuffled")
        name = f"{i:04d}.png"
        im.save(os.path.join(d, name))
        truth[name] = t
json.dump(truth, open(os.path.join(d, "truth.json"), "w"))
print(split, count)
