"""Reference checkpoints for the Mamba adapter (torch tier).

usage: hf_mamba.py OUT_DIR VARIANT SEED

A MambaForCausalLM built and saved by Hugging Face transformers (float32,
random weights around its own initialisation), and its forward pass —
transformers' sequential selective scan, the path it takes without the
CUDA kernels:

    prompt : int32[T]   logits : float32[T, V]   greedy : int32[T + N]

Variants: mamba (state 16, Δ rank 16: Mamba-130m's proportions, reduced)
and mamba-odd (state 8, Δ rank 4, conv kernel 3, biases in the
projections — extents that must be padded to the 16-lane contraction).
"""
import sys
import torch
from safetensors.torch import save_file
from transformers import MambaConfig, MambaForCausalLM

out, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(seed)

variants = {
    "mamba": dict(state_size=16, time_step_rank=16, conv_kernel=4, use_bias=False, use_conv_bias=True),
    "mamba-odd": dict(state_size=8, time_step_rank=4, conv_kernel=3, use_bias=True, use_conv_bias=True),
}

cfg = MambaConfig(vocab_size=96, hidden_size=64, num_hidden_layers=2, expand=2, layer_norm_epsilon=1e-5,
                  pad_token_id=0, bos_token_id=1, eos_token_id=2, use_mambapy=False, **variants[variant])
model = MambaForCausalLM(cfg)
model.eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("A_log"):
            continue                      # S4D-real init: log(1..N), as trained models start
        if "norm" in name:
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif name.endswith("dt_proj.bias"):
            p.add_(0.1 * torch.randn_like(p))
        elif name.endswith("bias") or name.endswith(".D"):
            p.copy_(0.1 * torch.randn_like(p) + (1.0 if name.endswith(".D") else 0.0))
        elif "embeddings" in name:
            p.copy_(torch.randn_like(p) * 0.5)
        elif p.dim() >= 2:
            fan_in = p[0].numel()
            p.copy_(torch.randn_like(p) / fan_in ** 0.5)
model.save_pretrained(out)

g = torch.Generator().manual_seed(seed + 1)
prompt = torch.randint(3, cfg.vocab_size, (1, 9), generator=g)
with torch.no_grad():
    logits = model(prompt).logits[0]
    model.generation_config.eos_token_id = None
    greedy = model.generate(prompt, max_new_tokens=16, do_sample=False, min_new_tokens=16)[0]
save_file({"prompt": prompt[0].to(torch.int32), "logits": logits.contiguous().float(), "greedy": greedy.to(torch.int32)},
          f"{out}/reference.safetensors")
