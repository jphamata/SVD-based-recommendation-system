"""Train vapor's spoken-digit reader on features vapor computed.

usage: train_speech.py FEATURES.safetensors OUT_DIR TEST_SPEAKER [EPOCHS] [--loso]

FEATURES comes from `Vapor.Modal.Speech.dataset/3` over the Free Spoken
Digit Dataset (6 speakers × 10 digits × 50 takes, 8 kHz): certified log-mel
rows f32[64, 32] per clip. The model is a `vapor_encoder` written out in
PyTorch — linear frame embedding, a [CLS] row, learned positions,
pre-norm layers (LayerNorm, exact GELU, bidirectional attention over the
real frames), final norm, a classifier on the [CLS] row — trained on five
speakers and tested on the sixth, never heard (speaker-independent).
With --loso every speaker is held out in turn and the mean accuracy
printed (the shipped model is the TEST_SPEAKER fold).
"""
import json, math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

feats, out_dir, test_name = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 60
loso = "--loso" in sys.argv
torch.set_num_threads(2)
SPEAKERS = ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"]
D, LAYERS, HEADS, FF, BANDS, FRAMES = 64, 3, 4, 128, 32, 64
ROWS = FRAMES + 1

data = load_file(feats)
X, L, Y, S = data["x"], data["len"].long(), data["digit"].long(), data["speaker"].long()


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
        x = x + self.o((att @ v).transpose(1, 2).reshape(b, t, D))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(BANDS, D)
        self.cls = nn.Parameter(torch.randn(1, D) * 0.02)
        self.pos = nn.Parameter(torch.randn(ROWS, D) * 0.02)
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D, eps=1e-6)
        self.head = nn.Linear(D, 10)

    def forward(self, x, lens):
        b = x.shape[0]
        # vapor's encoder: row 0 is the [CLS] row (its input row is zero), frames follow
        e = self.embed(torch.cat([torch.zeros(b, 1, BANDS), x], 1))
        e = torch.cat([self.cls.expand(b, 1, D), e[:, 1:]], 1) + self.pos
        mask = torch.arange(ROWS)[None, :] < (lens + 1)[:, None]
        for l in self.layers:
            e = l(e, mask)
        return self.head(self.norm(e)[:, 0])


def augment(x, lens):
    # SpecAugment-style: a random band block and a time block masked, a gain shift
    x = x.clone()
    for i in range(x.shape[0]):
        f0 = torch.randint(0, BANDS - 4, (1,)).item(); x[i, :, f0:f0 + torch.randint(0, 5, (1,)).item()] = 0
        n = int(lens[i]); t0 = torch.randint(0, max(n - 6, 1), (1,)).item(); x[i, t0:t0 + torch.randint(0, 7, (1,)).item(), :] = 0
    return x


def run_fold(test_id, seed=0):
    torch.manual_seed(seed)
    tr, te = (S != test_id).nonzero().squeeze(1), (S == test_id).nonzero().squeeze(1)
    m = Reader()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.05)
    steps = epochs * (len(tr) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm) - 63, 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(m(augment(X[idx], L[idx]), L[idx]), Y[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        acc = (m(X[te], L[te]).argmax(-1) == Y[te]).float().mean().item()
    return m, acc


if loso:
    accs = {}
    for sid, name in enumerate(SPEAKERS):
        _, accs[name] = run_fold(sid)
        print(name, round(accs[name], 4), flush=True)
    print("leave-one-speaker-out mean accuracy:", round(sum(accs.values()) / len(accs), 4))
    json.dump(accs, open(os.path.join(out_dir, "loso.json"), "w"), indent=1) if os.path.isdir(out_dir) else None

test_id = SPEAKERS.index(test_name)
model, acc = run_fold(test_id)
print(f"held-out speaker {test_name}: accuracy {acc:.4f}")
sd = model.state_dict()
ws = {"embed.weight": sd["embed.weight"], "embed.bias": sd["embed.bias"], "cls": sd["cls"], "pos": sd["pos"],
      "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"], "head.weight": sd["head.weight"], "head.bias": sd["head.bias"]}
for l in range(LAYERS):
    for n in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "q.weight", "q.bias", "k.weight", "k.bias",
              "v.weight", "v.bias", "o.weight", "o.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
        ws[f"layers.{l}.{n}"] = sd[f"layers.{l}.{n}"]
os.makedirs(out_dir, exist_ok=True)
save_file({k: v.contiguous().float() for k, v in ws.items()}, os.path.join(out_dir, "model.safetensors"))
cfg = {"model_type": "vapor_encoder", "rows": ROWS, "row_width": BANDS, "hidden_size": D, "num_hidden_layers": LAYERS,
       "num_attention_heads": HEADS, "intermediate_size": FF, "norm": "layer", "eps": 1e-6, "hidden_act": "gelu",
       "cls": True, "num_labels": 10, "head": "pooled", "modality": "audio",
       "labels": [str(d) for d in range(10)],
       "front_end": "Vapor.Modal.Speech (8 kHz, 256/128 Hann, 32 mel, log, per-clip mean removed)",
       "training": {"script": "test/python/train_speech.py", "data": "Free Spoken Digit Dataset (CC BY-SA 4.0)",
                    "train_speakers": [s for s in SPEAKERS if s != test_name], "test_speaker": test_name,
                    "test_accuracy": acc, "epochs": epochs}}
json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
