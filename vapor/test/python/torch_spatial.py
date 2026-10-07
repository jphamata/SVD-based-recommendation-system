"""PyTorch references for vapor's spatial operators (Vapor.Spatial).

usage: torch_spatial.py OUT.safetensors SEED
Writes, for each case, its input, its parameters and torch's output
(float32): conv2d with padding / stride / dilation / 1×1 / non-square
kernels, conv3d, GroupNorm, nearest upsampling and pixel self-attention
(a VAE mid-block's), all NCHW / NCDHW.
"""
import sys, torch
import torch.nn.functional as F
from safetensors.torch import save_file

out, seed = sys.argv[1], int(sys.argv[2])
torch.manual_seed(seed)
T = {}

def put(prefix, **kv):
    for k, v in kv.items():
        T[f"{prefix}.{k}"] = v.contiguous().float()

conv2d = {
    "c3_p1": dict(cin=5, cout=7, k=(3, 3), stride=1, padding=1, dilation=1, hw=(8, 8)),
    "c3_s2": dict(cin=16, cout=20, k=(3, 3), stride=2, padding=1, dilation=1, hw=(8, 6)),
    "c3_d2": dict(cin=3, cout=4, k=(3, 3), stride=1, padding=2, dilation=2, hw=(7, 9)),
    "c1": dict(cin=24, cout=8, k=(1, 1), stride=1, padding=0, dilation=1, hw=(4, 4)),
    "c25": dict(cin=4, cout=6, k=(2, 5), stride=(1, 2), padding=(0, 2), dilation=1, hw=(6, 10)),
}
for name, c in conv2d.items():
    x = torch.randn(1, c["cin"], *c["hw"])
    w = torch.randn(c["cout"], c["cin"], *c["k"]) * 0.3
    b = torch.randn(c["cout"]) * 0.1
    y = F.conv2d(x, w, b, stride=c["stride"], padding=c["padding"], dilation=c["dilation"])
    put(name, x=x[0], w=w, b=b, y=y[0])

x = torch.randn(1, 3, 4, 6, 5); w = torch.randn(5, 3, 3, 3, 3) * 0.3; b = torch.randn(5) * 0.1
put("c3d", x=x[0], w=w, b=b, y=F.conv3d(x, w, b, stride=(1, 2, 1), padding=1)[0])

x = torch.randn(1, 24, 4, 8) * 2 + 0.5
gn = torch.nn.GroupNorm(6, 24, eps=1e-6)
with torch.no_grad():
    gn.weight.copy_(1 + 0.2 * torch.randn(24)); gn.bias.copy_(0.1 * torch.randn(24))
    put("gn", x=x[0], w=gn.weight, b=gn.bias, y=gn(x)[0])

x = torch.randn(1, 5, 3, 4)
put("up", x=x[0], y=F.interpolate(x, scale_factor=2, mode="nearest")[0])

c, hw = 20, (4, 4)
x = torch.randn(1, c, *hw)
lin = [torch.nn.Linear(c, c) for _ in range(4)]
with torch.no_grad():
    seq = x.flatten(2).transpose(1, 2)                      # [1, n, c]
    q, k, v = (l(seq) for l in lin[:3])
    a = torch.softmax(q @ k.transpose(1, 2) / c ** 0.5, dim=-1) @ v
    y = lin[3](a).transpose(1, 2).reshape(1, c, *hw)
    put("attn", x=x[0], y=y[0], **{f"w{i}": l.weight for i, l in enumerate(lin)}, **{f"b{i}": l.bias for i, l in enumerate(lin)})

save_file(T, out)
