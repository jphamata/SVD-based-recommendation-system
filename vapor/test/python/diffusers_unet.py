"""A diffusers UNet2DConditionModel (random weights, float32) and one noise
prediction — the reference for vapor's U-Net adapter.

usage: diffusers_unet.py OUT_DIR SEED VARIANT
  sd1  two blocks (32, 64), one layer, 8 heads of 4 (padded to 16 here), 1×1 projections, latent 8×8
  sd2  three blocks (32, 64, 64), two layers, heads (2, 4, 4) of 16, linear projections, latent 16×16
OUT_DIR gets the checkpoint and reference.safetensors: z [4, h, w], t [1], ctx [S, D], eps [4, h, w].
"""
import math, sys, torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

out, seed, variant = sys.argv[1], int(sys.argv[2]), sys.argv[3]
torch.manual_seed(seed)
if variant == "sd1":
    m = UNet2DConditionModel(block_out_channels=(32, 64), layers_per_block=1, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                             up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, attention_head_dim=8,
                             norm_num_groups=8, sample_size=8)
    h = w = 8
else:
    m = UNet2DConditionModel(block_out_channels=(32, 64, 64), layers_per_block=2,
                             down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                             up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=48,
                             attention_head_dim=(2, 4, 4), use_linear_projection=True, norm_num_groups=8, sample_size=16)
    h = w = 16
m.eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n and n.endswith("weight"):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif n.endswith("bias"):
            p.copy_(0.1 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) / math.sqrt(p[0].numel()))
    z = torch.randn(1, 4, h, w)
    ctx = torch.randn(1, 7, m.config.cross_attention_dim)
    t = torch.tensor([417.0])
    eps = m(z, t, encoder_hidden_states=ctx).sample
m.save_pretrained(out)
save_file({"z": z[0].contiguous(), "t": t, "ctx": ctx[0].contiguous(), "eps": eps[0].contiguous()}, f"{out}/reference.safetensors")
print("ok")
