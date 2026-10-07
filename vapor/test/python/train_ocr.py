"""Train vapor's OCR reader: a vapor_encoder over line frames, with CTC.

usage: train_ocr.py OUT_DIR TRAIN_PREFIX[,TRAIN_PREFIX…] VALID_PREFIX [STEPS]

With OCR_DIRECTION=rtl the checkpoint is marked right to left
(`"direction": "rtl"`): its labels are in visual order and vapor turns
each reading back into logical order (`Vapor.Vision.Bidi`).

Inputs are the line bitmaps vapor itself computed (`Vapor.Vision.OCR.dataset/2`:
PREFIX.bin + PREFIX.json), so training sees exactly what inference sees.
The model is written in vapor's own topology — `model_type:
"vapor_encoder"`, `head: "rows"`: frames (32×8 windows every 2 columns) →
linear embedding + learned positions → pre-norm layers (LayerNorm, exact
GELU, bidirectional attention over the real frames) → final norm → a
classifier on every frame; CTC with blank = 0 and `labels` = the charset
plus the space. The PyTorch module below is that topology written out, so
the checkpoint means the same thing in both places (tested: vapor's
outputs = this module's, `test/vapor/ocr_test.exs`).
"""
import json, math, os, random, struct, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

out_dir, train_prefixes, valid_prefix = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(2)

H, WIN, STRIDE = 32, 8, 2
T_MAX = 512                    # frames (lines up to 1030 px at the normalised scale)
D, LAYERS, HEADS, FF = 128, 4, 4, 256


def load(prefix):
    meta = json.load(open(prefix + ".json", encoding="utf-8"))
    data = open(prefix + ".bin", "rb").read()
    out, off = [], 0
    for text in meta["texts"]:
        w = struct.unpack_from("<I", data, off)[0]
        off += 4
        img = np.frombuffer(data, np.uint8, 32 * w, off).reshape(32, w)
        off += 32 * w
        out.append((img, text))
    return meta["charset"], out


charset, train = None, []
for p in train_prefixes:
    charset, part = load(p)
    train += part
_, valid = load(valid_prefix)
labels = charset + [" "]
index = {c: i + 1 for i, c in enumerate(labels)}        # 0 = CTC blank
C = len(labels) + 1


def frames(img):
    w = img.shape[1]
    if w < WIN:
        img = np.pad(img, ((0, 0), (0, WIN - w)))
    t = torch.from_numpy(img.astype(np.float32) / 255.0)          # [32, W]
    return t.unfold(1, WIN, STRIDE).permute(1, 0, 2).reshape(-1, H * WIN)   # [T, 256], row-major (y, x)


def encode(text):
    return [index[c] for c in text if c in index]


def n_frames(img):
    return (max(img.shape[1], WIN) - WIN) // STRIDE + 1


# bitmaps stay uint8 (frames are 32× larger: built per batch)
train = [(img, encode(t)) for img, t in train]
train = [(img, y) for img, y in train if 0 < len(y) and n_frames(img) <= T_MAX and n_frames(img) >= 2 * len(y)]
valid = [(img, t) for img, t in valid if n_frames(img) <= T_MAX]
print(f"train {len(train)} lines, valid {len(valid)}, classes {C}", flush=True)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(D, eps=1e-6), nn.LayerNorm(D, eps=1e-6)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(D, FF), nn.Linear(FF, D)

    def forward(self, x, mask):
        b, t, _ = x.shape
        h = self.norm1(x)
        sh = lambda z: z.view(b, t, HEADS, D // HEADS).transpose(1, 2)
        q, k, v = sh(self.q(h)), sh(self.k(h)), sh(self.v(h))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // HEADS)
        att = att.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        a = (att @ v).transpose(1, 2).reshape(b, t, D)
        x = x + self.o(a)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(H * WIN, D)
        self.pos = nn.Parameter(torch.randn(T_MAX, D) * 0.02)
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D, eps=1e-6)
        self.head = nn.Linear(D, C)

    def forward(self, x, mask):
        t = x.shape[1]
        x = self.embed(x) + self.pos[:t]
        for l in self.layers:
            x = l(x, mask)
        return self.head(self.norm(x))


model = Reader()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)


def batch(items):
    items = [(frames(img), y) for img, y in items]
    t = max(f.shape[0] for f, _ in items)
    x = torch.zeros(len(items), t, H * WIN)
    mask = torch.zeros(len(items), t, dtype=torch.bool)
    for i, (f, _) in enumerate(items):
        x[i, :f.shape[0]] = f
        mask[i, :f.shape[0]] = True
    return x, mask


def augment(img):
    # light jitter of the bitmap's intensity (it is already scale-normalised)
    return np.clip(img.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)


def greedy(logits):
    best = logits.argmax(-1).tolist()
    out, prev = [], 0
    for k in best:
        if k != prev and k != 0:
            out.append(labels[k - 1])
        prev = k
    return "".join(out)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def evaluate(n=300):
    model.eval()
    errs, total = 0, 0
    with torch.no_grad():
        for img, text in valid[:n]:
            f = frames(img)
            hyp = greedy(model(f[None], torch.ones(1, f.shape[0], dtype=torch.bool))[0])
            errs += lev(hyp, text)
            total += len(text)
    model.train()
    return errs / max(total, 1)


def export(done, cer):
    """Write the checkpoint in vapor's names (also every 1000 steps: a stopped run keeps its progress)."""
    sd = model.state_dict()
    ws = {"embed.weight": sd["embed.weight"], "embed.bias": sd["embed.bias"], "pos": sd["pos"],
          "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"], "head.weight": sd["head.weight"], "head.bias": sd["head.bias"]}
    for l in range(LAYERS):
        for n in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "q.weight", "q.bias", "k.weight", "k.bias",
                  "v.weight", "v.bias", "o.weight", "o.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
            ws[f"layers.{l}.{n}"] = sd[f"layers.{l}.{n}"]
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.contiguous().float() for k, v in ws.items()}, os.path.join(out_dir, "model.safetensors"))
    cfg = {"model_type": "vapor_encoder", "rows": T_MAX, "row_width": H * WIN, "hidden_size": D, "num_hidden_layers": LAYERS,
           "num_attention_heads": HEADS, "intermediate_size": FF, "norm": "layer", "eps": 1e-6, "hidden_act": "gelu",
           "cls": False, "num_labels": C, "head": "rows", "modality": "text_image", "labels": labels,
           "frames": {"height": H, "window": WIN, "stride": STRIDE},
           **({"direction": os.environ["OCR_DIRECTION"]} if os.environ.get("OCR_DIRECTION") else {}),
           "training": {"script": "test/python/train_ocr.py", "steps": done, "lines": len(train),
                        "valid_cer": cer}}
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), ensure_ascii=False, indent=1)
    print("checkpoint at step", done, "valid CER", cer, flush=True)


# length-bucketed batches: similar widths together
order = sorted(range(len(train)), key=lambda i: train[i][0].shape[1])
buckets = [order[i:i + 24] for i in range(0, len(order), 24)]
t0 = time.time()
for step in range(1, steps + 1):
    items = [train[i] for i in random.choice(buckets)]
    x, mask = batch([(augment(f), y) for f, y in items])
    logp = model(x, mask).log_softmax(-1).transpose(0, 1)       # [T, B, C]
    ys = torch.tensor([c for _, y in items for c in y])
    loss = ctc(logp, ys, mask.sum(1), torch.tensor([len(y) for _, y in items]))
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 500 == 0 or step == steps:
        print(f"step {step} loss {loss.item():.3f} valid CER {evaluate():.4f} ({time.time() - t0:.0f} s)", flush=True)
    if step % 1000 == 0 and step < steps:
        export(step, evaluate(150))

export(steps, evaluate(len(valid)))

