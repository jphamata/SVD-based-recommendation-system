"""Independent reference for Vapor.Train.LM: the same Llama-style byte model
written plainly in PyTorch (binary64), its mean cross-entropy and the
gradient of every parameter by torch.autograd.

usage: lm_reference.py DIR   (DIR/config.json, DIR/params.safetensors,
DIR/batch.json with "x" and "y" token ids) → DIR/grads.safetensors (f64),
DIR/loss.json
"""
import json, math, os, sys
import torch
from safetensors.torch import load_file, save_file

d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
P = {k: v.double().requires_grad_(True) for k, v in load_file(os.path.join(d, "params.safetensors")).items()}
batch = json.load(open(os.path.join(d, "batch.json")))
V, D, L, H, S, NS, eps, theta = (cfg[k] for k in ["vocab", "d", "layers", "heads", "seq", "seqs", "eps", "theta"])
dh = D // H
R = NS * S
x = torch.tensor(batch["x"]).view(NS, S)
y = torch.tensor(batch["y"]).view(NS, S)

def rms(t, g):
    return g * (t * torch.rsqrt((t * t).mean(-1, keepdim=True) + eps))

inv = theta ** (-torch.arange(0, dh, 2, dtype=torch.float64) / dh)
pos = torch.arange(S, dtype=torch.float64)
ang = torch.outer(pos, inv)
cos = torch.cat([ang.cos(), ang.cos()], -1)
sin = torch.cat([ang.sin(), ang.sin()], -1)

def rope(t):  # t: [NS, S, dh]
    t1, t2 = t[..., : dh // 2], t[..., dh // 2:]
    return t * cos + torch.cat([-t2, t1], -1) * sin

h = P["emb"][x]  # [NS, S, D]
causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
for l in range(L):
    a = rms(h, P[f"l{l}.ln1"][0])
    att = 0
    for hh in range(H):
        q = rope(a @ P[f"l{l}.q{hh}"].T)
        k = rope(a @ P[f"l{l}.k{hh}"].T)
        v = a @ P[f"l{l}.v{hh}"].T
        s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        s = s.masked_fill(~causal, float("-inf"))
        att = att + (torch.softmax(s, -1) @ v) @ P[f"l{l}.o{hh}"].T
    h = h + att
    b = rms(h, P[f"l{l}.ln2"][0])
    h = h + (torch.nn.functional.silu(b @ P[f"l{l}.gate"].T) * (b @ P[f"l{l}.up"].T)) @ P[f"l{l}.down"].T
z = rms(h, P["norm"][0]) @ P["head"].T
loss = torch.nn.functional.cross_entropy(z.reshape(R, V), y.reshape(R))
loss.backward()
save_file({k: v.grad.contiguous() for k, v in P.items()}, os.path.join(d, "grads.safetensors"))
json.dump({"loss": loss.item(), "logits": z.reshape(R, V).detach().flatten().tolist()}, open(os.path.join(d, "loss.json"), "w"))
