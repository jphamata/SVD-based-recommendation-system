"""PyTorch reference of a vapor_encoder with `head: "rows"` (the OCR reader).

usage: ocr_reference.py MODEL_DIR < {"frames": [[256 floats], ...]}  →  {"logits": [[C floats], ...]}

The topology of test/python/train_ocr.py, built from config.json alone and
loaded from model.safetensors: frames → linear + learned positions →
pre-norm layers (LayerNorm, bidirectional attention, exact GELU) → final
norm → a classifier on every frame. Used by test/vapor/ocr_test.exs.
"""
import json, math, os, sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
w = load_file(os.path.join(d, "model.safetensors"))
D, L, Hh, eps = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["eps"]

x = torch.tensor(json.load(sys.stdin)["frames"], dtype=torch.float32)
t = x.shape[0]
lin = lambda z, n: z @ w[n + ".weight"].T + w[n + ".bias"]
norm = lambda z, n: F.layer_norm(z, (D,), w[n + ".weight"], w[n + ".bias"], eps)

with torch.no_grad():
    h = lin(x, "embed") + w["pos"][:t]
    for l in range(L):
        p = f"layers.{l}."
        a = norm(h, p + "norm1")
        sh = lambda z: z.view(t, Hh, D // Hh).transpose(0, 1)
        q, k, v = sh(lin(a, p + "q")), sh(lin(a, p + "k")), sh(lin(a, p + "v"))
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(D // Hh)).softmax(-1)
        h = h + lin((att @ v).transpose(0, 1).reshape(t, D), p + "o")
        h = h + lin(F.gelu(lin(norm(h, p + "norm2"), p + "fc1")), p + "fc2")
    out = lin(norm(h, "norm"), "head")

print(json.dumps({"logits": out.tolist()}))
