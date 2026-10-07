"""Reference checkpoints for the Mamba-2 adapter (torch tier).

usage: hf_mamba2.py OUT_DIR VARIANT SEED

A Mamba2ForCausalLM built and saved by Hugging Face transformers (float32,
random weights around its own initialisation), and its forward pass —
transformers' chunked SSD scan, the path it takes without the CUDA
kernels (`torch_forward`, no cache):

    prompt : int32[T]   logits : float32[T, V]   greedy : int32[T + N]

`greedy` is decoded by re-running that full forward pass at every step
(no cache), on purpose: transformers' cached decode step skips the
`time_step_limit` clamp its chunked scan applies, so `generate()` mixes
two semantics; the full pass is the one the model was trained with.

Variants:
  mamba2     one group, state 16, head_dim 16 (mamba2-130m's proportions, reduced)
  mamba2-g2  two groups, state 8 (padded to the 16-lane contraction), head_dim 8,
             conv kernel 3, biases, time_step_limit (0, 0.6) — the clamp bites.
             Also `logits_grouped`/`greedy_grouped`: the same weights with the
             gated RMSNorm taken per group, as mamba_ssm (the training code)
             does it — transformers normalises over the whole width.
"""
import sys
import torch
from safetensors.torch import save_file
from transformers import Mamba2Config, Mamba2ForCausalLM
from transformers.models.mamba2 import modeling_mamba2 as mm

out, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(seed)

variants = {
    "mamba2": dict(num_heads=8, head_dim=16, n_groups=1, state_size=16, conv_kernel=4, use_bias=False),
    "mamba2-g2": dict(num_heads=16, head_dim=8, n_groups=2, state_size=8, conv_kernel=3, use_bias=True,
                      time_step_limit=(0.0, 0.6)),
}

cfg = Mamba2Config(vocab_size=96, hidden_size=64, num_hidden_layers=2, expand=2, layer_norm_epsilon=1e-5,
                   pad_token_id=0, bos_token_id=1, eos_token_id=2, chunk_size=4, tie_word_embeddings=False,
                   **variants[variant])
model = Mamba2ForCausalLM(cfg)
model.eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("A_log"):
            continue                      # log(1..num_heads), as transformers initialises it
        if "norm" in name:
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif name.endswith("dt_bias"):
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


def run():
    with torch.no_grad():
        logits = model(prompt).logits[0]
        ids = prompt
        for _ in range(16):
            nxt = model(ids).logits[0, -1].argmax()
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return logits.contiguous().float(), ids[0].to(torch.int32)


logits, greedy = run()
tensors = {"prompt": prompt[0].to(torch.int32), "logits": logits, "greedy": greedy}

if cfg.n_groups > 1:
    group = (cfg.expand * cfg.hidden_size) // cfg.n_groups

    def grouped(self, hidden_states, gate=None):
        # mamba_ssm's RMSNormGated(group_size = d_ssm / ngroups, norm_before_gate=False)
        h = hidden_states.to(torch.float32)
        if gate is not None:
            h = h * torch.nn.functional.silu(gate.to(torch.float32))
        shape = h.shape
        h = h.view(*shape[:-1], -1, group)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.view(shape)

    mm.MambaRMSNormGated.forward = grouped
    lg, gg = run()
    tensors |= {"logits_grouped": lg, "greedy_grouped": gg}

save_file(tensors, f"{out}/reference.safetensors")
