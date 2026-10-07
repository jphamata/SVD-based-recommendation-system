"""A tiny Stable Diffusion (random weights, float32) and three runs of
diffusers' own pipelines — the reference for `Vapor.Diffusion.Pipeline`.

usage: diffusers_pipeline.py OUT_DIR
OUT_DIR gets a diffusers layout (unet/, vae/, text_encoder/, scheduler/,
model_index.json) and reference.safetensors with:
  ids, neg       token ids [16] (prompt and negative prompt, already padded)
tokenizer/ holds a small CLIP BPE (vocab.json + merges.txt) for prompts as text.
  latents        the starting noise [4, 8, 8]
  noise          the img2img / inpainting noise [4, 8, 8]
  init           the source image [16, 16, 3] in [0, 1]
  mask           the inpainting mask [16, 16] (1 = repaint)
  txt2img_ddim, txt2img_euler, txt2img_dpmpp_2m   [16, 16, 3] (6 steps, guidance 3)
  img2img        [16, 16, 3] (DDIM, 10 steps, strength 0.6, guidance 3)
  inpaint        [16, 16, 3] (DDIM, 10 steps, strength 1.0, guidance 3; a 4-channel U-Net: latent blending)
The tokenizer is not exercised here (vapor's CLIP BPE has its own test):
prompt_embeds are the text encoder's output for the given ids. The VAE
encoder's latent is its distribution's mean, as vapor's (diffusers samples
it: the sampler is patched to the mean, and its noise to the given noise).
"""
import json, math, os, sys, torch
import numpy as np
from diffusers import (AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline)
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img as i2i
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint as inp
from transformers import CLIPTextConfig, CLIPTextModel
from safetensors.torch import save_file
from PIL import Image

out = sys.argv[1]
torch.manual_seed(11)


def init_(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n and n.endswith("weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            elif n.endswith("bias"):
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / math.sqrt(p[0].numel()))
    return m.eval()


unet = init_(UNet2DConditionModel(block_out_channels=(32, 64), layers_per_block=1, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                  up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, attention_head_dim=8,
                                  norm_num_groups=8, sample_size=8))
vae = init_(AutoencoderKL(block_out_channels=(32, 64), layers_per_block=1, latent_channels=4, norm_num_groups=8,
                          down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2, sample_size=16))
torch.manual_seed(12)
te = CLIPTextModel(CLIPTextConfig(vocab_size=1000, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                  max_position_embeddings=16, bos_token_id=998, eos_token_id=999, pad_token_id=999)).eval()
cfg = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", steps_offset=1,
           timestep_spacing="leading")
ddim = DDIMScheduler(**cfg, set_alpha_to_one=False, clip_sample=False)
for name, m in [("unet", unet), ("vae", vae), ("text_encoder", te), ("scheduler", ddim)]:
    m.save_pretrained(os.path.join(out, name))
json.dump({"_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"], "vae": ["diffusers", "AutoencoderKL"],
           "text_encoder": ["transformers", "CLIPTextModel"], "scheduler": ["diffusers", "DDIMScheduler"]},
          open(os.path.join(out, "model_index.json"), "w"))

# a small CLIP BPE (the slow files SD checkpoints ship), specials at the text encoder's 998/999
tdir = os.path.join(out, "tokenizer")
os.makedirs(tdir, exist_ok=True)
chars = list("abcdefghijklmnopqrstuvwxyz0123456789,.?!'")
merges = ["a s", "t r", "o n", "a n", "e r", "i n", "c a", "ca t</w>", "d o", "do g</w>"]
toks = list(dict.fromkeys([t for c in chars for t in (c, c + "</w>")] + [m.replace(" ", "") for m in merges]))
toks += [f"<unused{i}>" for i in range(998 - len(toks))] + ["<|startoftext|>", "<|endoftext|>"]
json.dump({t: i for i, t in enumerate(toks)}, open(os.path.join(tdir, "vocab.json"), "w"))
open(os.path.join(tdir, "merges.txt"), "w").write("#version: 0.2\n" + "\n".join(merges) + "\n")
json.dump({"model_max_length": 16, "pad_token": "<|endoftext|>", "bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>",
           "tokenizer_class": "CLIPTokenizer"}, open(os.path.join(tdir, "tokenizer_config.json"), "w"))

ids = torch.tensor([[998, 17, 300, 42, 999] + [999] * 11])
neg = torch.tensor([[998, 999] + [999] * 14])
with torch.no_grad():
    pe, ne = te(ids)[0], te(neg)[0]
latents = torch.randn(1, 4, 8, 8)
noise = torch.randn(1, 4, 8, 8)
init = torch.rand(16, 16, 3)
mask = torch.zeros(16, 16)
mask[4:12, 6:14] = 1.0

ref = {"ids": ids[0].to(torch.float32), "neg": neg[0].to(torch.float32), "latents": latents[0], "noise": noise[0],
       "init": init, "mask": mask}
common = dict(prompt_embeds=pe, negative_prompt_embeds=ne, guidance_scale=3.0, output_type="np")
for kind, s in [("ddim", ddim), ("euler", EulerDiscreteScheduler(**cfg)),
                ("dpmpp_2m", DPMSolverMultistepScheduler(**cfg, algorithm_type="dpmsolver++", solver_order=2))]:
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=None, unet=unet, scheduler=s, safety_checker=None,
                                   feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    ref[f"txt2img_{kind}"] = torch.from_numpy(pipe(num_inference_steps=6, latents=latents.clone(), height=16, width=16, **common).images[0]).float()

# the encoder's mean, not a sample; the given noise, not the generator's
mean_ = lambda enc, generator=None, sample_mode="sample": enc.latent_dist.mode()
i2i.retrieve_latents = mean_
inp.retrieve_latents = mean_
i2i.randn_tensor = lambda shape, **kw: noise.clone()
inp.randn_tensor = lambda shape, **kw: noise.clone()
pil = Image.fromarray((init.numpy() * 255).round().astype(np.uint8))
ref["init"] = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0)  # what the pipelines see
p2 = StableDiffusionImg2ImgPipeline(vae=vae, text_encoder=te, tokenizer=None, unet=unet, scheduler=ddim, safety_checker=None,
                                    feature_extractor=None, requires_safety_checker=False)
p2.set_progress_bar_config(disable=True)
ref["img2img"] = torch.from_numpy(p2(image=pil, strength=0.6, num_inference_steps=10, **common).images[0]).float()
p3 = StableDiffusionInpaintPipeline(vae=vae, text_encoder=te, tokenizer=None, unet=unet, scheduler=ddim, safety_checker=None,
                                    feature_extractor=None, requires_safety_checker=False)
p3.set_progress_bar_config(disable=True)
mpil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
ref["inpaint"] = torch.from_numpy(p3(image=pil, mask_image=mpil, strength=1.0, num_inference_steps=10, height=16, width=16,
                                     latents=latents.clone(), **common).images[0]).float()
save_file({k: v.contiguous() for k, v in ref.items()}, os.path.join(out, "reference.safetensors"))
print("ok")
