"""Reference models for vapor's differential tier (phase P3).

usage: hf_reference.py OUT_DIR VARIANT SEED

Builds a small random model of the given variant with Hugging Face
transformers, saves it as a checkpoint directory (config.json +
model.safetensors) and writes OUT_DIR/reference.safetensors with
  prompt  : int32[T]         the prompt
  logits  : float32[T, V]    transformers' logits for the prompt (prefill)
  greedy  : int32[T + N]     transformers' greedy continuation
and, for the variants ending in "-q" (hidden width 256, so 4-bit
superblocks fit), the prefill logits with every projection matrix
fake-quantized by the two common 4-bit baselines of 32-weight groups:
  logits_q4_0 : symmetric, scale = absmax / 7        (llama.cpp Q4_0-like)
  logits_q4_1 : affine, min + scale · q, scale = (max − min) / 15  (Q4_1-like)
Weights are re-drawn at unit activation scale (the default initialiser
makes every logit ~0, which would test nothing).
"""
import math, sys
import torch
from safetensors.torch import save_file
from transformers import (LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM)

out, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(seed)
quantized = variant.endswith("-q")
variant = variant.removesuffix("-q")
common = dict(vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
              rms_norm_eps=1e-5, pad_token_id=0)
if quantized:
    common.update(hidden_size=256, intermediate_size=512, vocab_size=128)

if variant == "llama":
    model = LlamaForCausalLM(LlamaConfig(**common, attention_bias=True, tie_word_embeddings=False))
elif variant == "llama3-rope":
    scaling = {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
               "high_freq_factor": 2.0, "original_max_position_embeddings": 32}
    model = LlamaForCausalLM(LlamaConfig(**common, rope_parameters=scaling, tie_word_embeddings=False))
elif variant == "linear-rope":
    scaling = {"rope_type": "linear", "rope_theta": 10000.0, "factor": 4.0}
    model = LlamaForCausalLM(LlamaConfig(**{**common, "num_key_value_heads": 4}, rope_parameters=scaling))
elif variant == "mistral":
    model = MistralForCausalLM(MistralConfig(**common, sliding_window=None))
elif variant == "qwen2":
    model = Qwen2ForCausalLM(Qwen2Config(**{**common, "num_key_value_heads": 1}, tie_word_embeddings=True))
else:
    raise SystemExit(f"unknown variant {variant}")

model.eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif name.endswith("bias"):
            p.copy_(0.1 * torch.randn_like(p))
        elif "embed" in name:
            p.copy_(torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) / math.sqrt(p.shape[1]))

model.save_pretrained(out)
g = torch.Generator().manual_seed(seed + 1)
prompt = torch.randint(0, common["vocab_size"], (1, 24 if quantized else 9), generator=g)
with torch.no_grad():
    logits = model(prompt).logits[0]
    model.generation_config.eos_token_id = None
    greedy = model.generate(prompt, max_new_tokens=16, do_sample=False, min_new_tokens=16)[0]

ref = {"prompt": prompt[0].to(torch.int32), "logits": logits.contiguous().float(), "greedy": greedy.to(torch.int32)}

def q4_0(w):
    g = w.reshape(-1, 32)
    s = g.abs().amax(1, keepdim=True) / 7
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.clamp(torch.round(g / s), -8, 7) * s).reshape(w.shape)

def q4_1(w):
    g = w.reshape(-1, 32)
    lo, hi = g.amin(1, keepdim=True), g.amax(1, keepdim=True)
    s = (hi - lo) / 15
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (lo + torch.clamp(torch.round((g - lo) / s), 0, 15) * s).reshape(w.shape)

if quantized:
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    for name, fq in [("q4_0", q4_0), ("q4_1", q4_1)]:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.dim() == 2 and "embed" not in n:
                    p.copy_(fq(base[n]))
            ref[f"logits_{name}"] = model(prompt).logits[0].contiguous().float()
            for n, p in model.named_parameters():
                p.copy_(base[n])

save_file(ref, f"{out}/reference.safetensors")
