"""A diffusers AutoencoderKL (random weights, float32) and its decoder's
output on random latents — the reference for vapor's VAE adapter.

usage: diffusers_vae.py OUT_DIR SEED [VARIANT]
VARIANT: sd (4 latent channels, two blocks) or wide (16 latent channels as
Flux/SD3, three blocks, no post_quant_conv).
OUT_DIR gets the checkpoint (config.json, diffusion_pytorch_model.safetensors)
and reference.safetensors: z : [C, h, w], image : [3, H, W].
"""
import math, sys, torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

out, seed = sys.argv[1], int(sys.argv[2])
variant = sys.argv[3] if len(sys.argv) > 3 else "sd"
torch.manual_seed(seed)
if variant == "sd":
    m = AutoencoderKL(block_out_channels=(32, 64), layers_per_block=1, latent_channels=4, norm_num_groups=8,
                      down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2, sample_size=8)
    h = w = 4
else:
    m = AutoencoderKL(block_out_channels=(16, 32, 32), layers_per_block=1, latent_channels=16, norm_num_groups=8,
                      down_block_types=("DownEncoderBlock2D",) * 3, up_block_types=("UpDecoderBlock2D",) * 3, sample_size=16,
                      use_post_quant_conv=False, use_quant_conv=False)
    h, w = 4, 8
m.eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n and n.endswith("weight"):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif n.endswith("bias"):
            p.copy_(0.1 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) / math.sqrt(p[0].numel()))
    z = torch.randn(1, m.config.latent_channels, h, w)
    img = m.decode(z).sample
m.save_pretrained(out)
save_file({"z": z[0].contiguous(), "image": img[0].contiguous()}, f"{out}/reference.safetensors")
