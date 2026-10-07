"""Train the handwritten-digit models of vapor's real-data any-to-any routes.

usage: train_digits.py OUT_DIR

Data: scikit-learn's `load_digits` — 1 797 real 8×8 handwritten digits
(UCI "Optical Recognition of Handwritten Digits", 4-bit grey levels),
split once with a fixed seed: 1 300 for training, 497 held out. Writes:

  OUT_DIR/data.safetensors     images u8[1797, 64] (0–16), labels, split mask (1 = train)
  OUT_DIR/classifier/          vapor_mlp 64 → 128 → 128 → 10 (GELU): image → digit
  OUT_DIR/denoiser/            vapor_mlp 96 → 256 → 256 → 256 → 64 (SiLU): ε(x_t, t, class)

The denoiser is a DDPM noise predictor over x ∈ [−1, 1]⁶⁴ with a cosine
schedule of 200 steps, conditioned on the class (one-hot, dropped 15 % of
the time for classifier-free guidance) and on t (8 sine/cosine pairs);
its input row is [x_t (64) | time (16) | class one-hot padded to 16].
`Vapor.Modal.Diffusion` samples it with DDIM on vapor's substrate.
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.datasets import load_digits

out = sys.argv[1]
torch.manual_seed(0)
torch.set_num_threads(2)
d = load_digits()
imgs = d.images.reshape(-1, 64).astype(np.uint8)       # 0..16
labels = d.target.astype(np.int64)
rng = np.random.default_rng(0)
perm = rng.permutation(len(imgs))
train_mask = np.zeros(len(imgs), np.uint8)
train_mask[perm[:1300]] = 1
os.makedirs(out, exist_ok=True)
save_file({"images": torch.from_numpy(imgs), "labels": torch.from_numpy(labels.astype(np.int32)),
           "train": torch.from_numpy(train_mask)}, os.path.join(out, "data.safetensors"))
tr, te = train_mask == 1, train_mask == 0
X = torch.from_numpy(imgs).float()
Y = torch.from_numpy(labels)


def mlp(widths, act):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def export(model, widths, act, path, extra):
    os.makedirs(path, exist_ok=True)
    lin = [m for m in model if isinstance(m, nn.Linear)]
    ws = {}
    for i, m in enumerate(lin):
        ws[f"layers.{i}.weight"] = m.weight.detach().contiguous().float()
        ws[f"layers.{i}.bias"] = m.bias.detach().contiguous().float()
    save_file(ws, os.path.join(path, "model.safetensors"))
    cfg = {"model_type": "vapor_mlp", "in_width": widths[0], "hidden_sizes": widths[1:-1], "out_width": widths[-1],
           "hidden_act": act, **extra}
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"), indent=1)


# ------------------------------------------------------------ classifier --
cw = [64, 128, 128, 10]
clf = mlp(cw, nn.GELU)
opt = torch.optim.AdamW(clf.parameters(), lr=2e-3, weight_decay=1e-3)
xtr, ytr = X[tr] / 16, Y[tr]
for ep in range(300):
    p = torch.randperm(len(xtr))
    for i in range(0, len(p), 64):
        idx = p[i:i + 64]
        # a pixel of jitter: shift by one with probability 1/2
        xb = xtr[idx].view(-1, 8, 8)
        if torch.rand(1).item() < 0.5:
            xb = torch.roll(xb, shifts=(int(torch.randint(-1, 2, (1,))), int(torch.randint(-1, 2, (1,)))), dims=(1, 2))
        loss = F.cross_entropy(clf(xb.reshape(-1, 64)), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
clf.eval()
with torch.no_grad():
    acc = (clf(X[te] / 16).argmax(-1) == Y[te]).float().mean().item()
print("classifier held-out accuracy", acc)
export(clf, cw, "gelu", os.path.join(out, "classifier"),
       {"labels": [str(i) for i in range(10)], "from": "image", "to": "text_digit",
        "input": "8×8 grey levels / 16, row-major", "training": {"data": "sklearn load_digits (UCI)", "train": 1300, "held_out": 497,
                                                                 "held_out_accuracy": acc}})

# ------------------------------------------------------------- denoiser --
T = 200
s = 0.008
steps = torch.arange(T + 1, dtype=torch.float64)
f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
abar = (f / f[0]).clamp(1e-5, 1.0)[1:]                # ᾱ_t, t = 1..T
abar = torch.minimum(abar, torch.tensor(0.9999, dtype=torch.float64))


def time_feats(t):
    # t in 1..T → 8 sine/cosine pairs (frequencies 1, 2, 4, …, 128 over [0, 1])
    u = (t.double() / T)[:, None] * (2 ** torch.arange(8, dtype=torch.float64))[None, :] * math.pi
    return torch.cat([torch.sin(u), torch.cos(u)], 1).float()


dw = [96, 256, 256, 256, 64]
den = mlp(dw, nn.SiLU)
opt = torch.optim.AdamW(den.parameters(), lr=1e-3, weight_decay=1e-4)
x0 = X[tr] / 8 - 1
yc = Y[tr]
n_steps = 30000
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=n_steps)
for step in range(n_steps):
    idx = torch.randint(0, len(x0), (128,))
    t = torch.randint(1, T + 1, (128,))
    a = abar[t - 1].float()[:, None]
    eps = torch.randn(128, 64)
    xt = a.sqrt() * x0[idx] + (1 - a).sqrt() * eps
    oh = F.one_hot(yc[idx], 16).float()
    oh[torch.rand(128) < 0.15] = 0.0                 # the unconditional model, for guidance
    pred = den(torch.cat([xt, time_feats(t), oh], 1))
    loss = F.mse_loss(pred, eps)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 5000 == 0:
        print("denoiser step", step, loss.item(), flush=True)
export(den, dw, "silu", os.path.join(out, "denoiser"),
       {"from": "text_digit", "to": "image", "schedule": {"kind": "cosine", "T": T, "s": s, "abar": abar.tolist()},
        "input": "[x_t (64, in [-1, 1]) | 8 sin + 8 cos of t/T·π·2^k | one-hot class padded to 16 (zeros = unconditional)]",
        "predicts": "epsilon", "training": {"data": "sklearn load_digits (UCI), 1300 training images", "steps": n_steps}})
print("done")
