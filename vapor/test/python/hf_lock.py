"""Reference checkpoints for the model airlock's adapters (torch tier).

usage: hf_lock.py OUT_DIR VARIANT SEED

The checkpoints the airlock admits through an alias, a blueprint or a
topology — built and saved by Hugging Face transformers itself (float32,
random weights), so vapor reads exactly the names, fused layouts and
config keys transformers writes, and is compared against transformers'
own forward pass rather than against a re-reading of it:

  decoders (OUT_DIR/reference.safetensors)
    prompt  : int32[T]        logits : float32[T, V]     greedy : int32[T + N]
  encoders
    pixels  : float32[C, H, W] the preprocessed image
    hidden  : float32[T, d]   last_hidden_state (after the final norm)
    logits  : float32[labels] classifier logits (when the variant has a head)
    pooled  : float32[d]      pooler output (when the variant has one)
    embeds  : float32[p]      projected embedding (CLIP with projection)

Variants: phi3, phi3-partial (Phi-4-mini's partial rotary factor),
granite, vit (ViTForImageClassification), vit-pooler (ViTModel with its
pooler), clip-vision (CLIPVisionModelWithProjection, quick_gelu),
clip-text (CLIPTextModelWithProjection: ids, hidden, embeds) and
clip-model (a whole CLIPModel, read through its text tower).
"""
import math, sys
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

out, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(seed)


def randomise(model, norm_center=1.0):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(norm_center + 0.2 * torch.randn_like(p))
            elif name.endswith("bias") or "norm" in name:
                p.copy_(0.1 * torch.randn_like(p))
            elif "embed" in name or "cls_token" in name or "class_embedding" in name:
                p.copy_(torch.randn_like(p) * (0.5 if p.dim() > 1 and "patch" not in name else 1.0))
            elif p.dim() >= 2:
                fan_in = p[0].numel()
                p.copy_(torch.randn_like(p) / math.sqrt(fan_in))
            else:
                p.copy_(0.1 * torch.randn_like(p))


decoders = {
    "phi3": ("phi3", {}),
    "phi3-partial": ("phi3", dict(head_dim=32, num_attention_heads=2, num_key_value_heads=1,
                                  rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                   "partial_rotary_factor": 0.5})),
    "granite": ("granite", dict(embedding_multiplier=3.0, attention_multiplier=0.125, residual_multiplier=0.22,
                                logits_scaling=4.0, attention_bias=True)),
}

if variant in decoders:
    common = dict(vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                  rms_norm_eps=1e-5, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    mt, extra = decoders[variant]
    cfg = AutoConfig.for_model(mt, **{**common, **extra})
    cfg._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    randomise(model)
    model.save_pretrained(out)
    g = torch.Generator().manual_seed(seed + 1)
    prompt = torch.randint(3, common["vocab_size"], (1, 9), generator=g)
    with torch.no_grad():
        logits = model(prompt).logits[0]
        model.generation_config.eos_token_id = None
        greedy = model.generate(prompt, max_new_tokens=16, do_sample=False, min_new_tokens=16)[0]
    save_file({"prompt": prompt[0].to(torch.int32), "logits": logits.contiguous().float(),
               "greedy": greedy.to(torch.int32)}, f"{out}/reference.safetensors")
    sys.exit(0)

g = torch.Generator().manual_seed(seed + 1)
if variant in ("vit", "vit-pooler"):
    from transformers import ViTConfig, ViTForImageClassification, ViTModel
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=16,
                    patch_size=4, num_channels=3, num_labels=5, layer_norm_eps=1e-12)
    cfg._attn_implementation = "eager"
    model = ViTForImageClassification(cfg) if variant == "vit" else ViTModel(cfg, add_pooling_layer=True)
    model.eval()
    randomise(model)
    pixels = torch.rand(1, 3, 16, 16, generator=g) * 2 - 1
    with torch.no_grad():
        if variant == "vit":
            o = model(pixel_values=pixels, output_hidden_states=True)
            hidden = model.vit.layernorm(o.hidden_states[-1])[0]
            refs = {"hidden": hidden, "logits": o.logits[0]}
        else:
            o = model(pixel_values=pixels)
            refs = {"hidden": o.last_hidden_state[0], "pooled": o.pooler_output[0]}
elif variant == "clip-vision":
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=16, patch_size=4, num_channels=3, projection_dim=24, hidden_act="quick_gelu",
                           layer_norm_eps=1e-5)
    cfg._attn_implementation = "eager"
    model = CLIPVisionModelWithProjection(cfg)
    model.eval()
    randomise(model)
    pixels = torch.rand(1, 3, 16, 16, generator=g) * 2 - 1
    with torch.no_grad():
        o = model(pixel_values=pixels, output_hidden_states=True)
        pooled = model.vision_model.post_layernorm(o.last_hidden_state[:, 0, :])[0]
        refs = {"hidden": o.last_hidden_state[0], "pooled": pooled, "embeds": o.image_embeds[0]}
elif variant in ("clip-text", "clip-model"):
    # CLIP's text tower: as CLIPTextModelWithProjection, or the text half of a
    # whole CLIPModel (both towers in one checkpoint, `text_model.` prefix)
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPConfig, CLIPModel
    tcfg = dict(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                max_position_embeddings=24, projection_dim=24, hidden_act="quick_gelu", layer_norm_eps=1e-5,
                bos_token_id=510, eos_token_id=511, pad_token_id=1)
    if variant == "clip-text":
        cfg = CLIPTextConfig(**tcfg)
        cfg._attn_implementation = "eager"
        model = CLIPTextModelWithProjection(cfg)
    else:
        cfg = CLIPConfig(text_config=tcfg, vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, image_size=16, patch_size=4), projection_dim=24)
        cfg._attn_implementation = "eager"
        model = CLIPModel(cfg)
    model.eval()
    randomise(model)
    ids = torch.cat([torch.tensor([510]), torch.randint(2, 500, (9,), generator=g), torch.tensor([511])]).unsqueeze(0)
    with torch.no_grad():
        if variant == "clip-text":
            o = model(input_ids=ids)
            refs = {"hidden": o.last_hidden_state[0], "embeds": o.text_embeds[0]}
        else:
            o = model.text_model(input_ids=ids)
            refs = {"hidden": o.last_hidden_state[0], "embeds": model.text_projection(o.pooler_output)[0]}
    model.save_pretrained(out)
    save_file({"ids": ids[0].to(torch.int32), **{k: v.contiguous().float() for k, v in refs.items()}},
              f"{out}/reference.safetensors")
    sys.exit(0)
else:
    raise SystemExit(f"unknown variant {variant}")

model.save_pretrained(out)
save_file({"pixels": pixels[0].contiguous(), **{k: v.contiguous().float() for k, v in refs.items()}},
          f"{out}/reference.safetensors")
