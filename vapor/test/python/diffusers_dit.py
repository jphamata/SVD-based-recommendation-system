"""A diffusers DiTTransformer2DModel (random weights, float32) and its
noise prediction — the reference for vapor's DiT adapter.

usage: diffusers_dit.py OUT_DIR SEED
OUT_DIR: the checkpoint and reference.safetensors with z : [C, H, W], t, label,
temb : [1, 256] (diffusers' own sinusoidal features), out : [C_out, H, W],
pos : [N, d] (the 2-D sin-cos table, to check vapor's own).
"""
import math, sys, torch
from diffusers import DiTTransformer2DModel
from safetensors.torch import save_file

out, seed = sys.argv[1], int(sys.argv[2])
torch.manual_seed(seed)
m = DiTTransformer2DModel(num_attention_heads=2, attention_head_dim=16, in_channels=4, out_channels=8, num_layers=2,
                          sample_size=8, patch_size=2, num_embeds_ada_norm=10)
m.eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith("bias"):
            p.copy_(0.1 * torch.randn_like(p))
        elif "embedding_table" in n:
            p.copy_(torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) / math.sqrt(p[0].numel()))
    z = torch.randn(1, 4, 8, 8)
    t = torch.tensor([637])
    label = torch.tensor([3])
    y = m(z, timestep=t, class_labels=label).sample
    temb = m.transformer_blocks[0].norm1.emb.time_proj(t)
m.save_pretrained(out)
save_file({"z": z[0].contiguous(), "t": t.to(torch.int32), "label": label.to(torch.int32), "temb": temb.contiguous().float(),
           "out": y[0].contiguous(), "pos": m.pos_embed.pos_embed[0].contiguous().float()}, f"{out}/reference.safetensors")
