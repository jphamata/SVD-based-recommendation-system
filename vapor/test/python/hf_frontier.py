"""Reference checkpoints for vapor's frontier families (differential tier).

usage: hf_frontier.py OUT_DIR VARIANT SEED

Like hf_reference.py: a small random model built by Hugging Face
transformers (float32), saved as a checkpoint directory, plus
OUT_DIR/reference.safetensors with
  prompt  : int32[T]       the prompt
  logits  : float32[T, V]  transformers' prefill logits
  greedy  : int32[T + N]   transformers' greedy continuation
Variants exercise each new operator: per-head q/k norms (qwen3), YaRN
(qwen3-yarn, deepseek-yarn), softmax top-k mixtures with and without
renormalisation and with dense layers mixed in (mixtral, qwen3-moe),
Gemma 3's (1 + w) sandwich norms, GELU-tanh, √d embedding scale, local and
global RoPE per layer type, a query scale other than 1/√dh and logit
soft-capping (gemma3), sliding windows that bind (mistral-window,
gemma3-window: attention over the last w positions), and DeepSeek-V3's
latent attention with
interleaved RoPE, a value head narrower than the key head, sigmoid
group-limited routing with a correction bias, shared experts and a
routed scale (deepseek, deepseek-yarn).
"""
import math, sys
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

out, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(seed)
common = dict(vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
              rms_norm_eps=1e-5, pad_token_id=0, bos_token_id=1, eos_token_id=2)

yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16,
        "beta_fast": 32.0, "beta_slow": 1.0}

V = {
  "qwen3": ("qwen3", dict(head_dim=16)),
  "qwen3-yarn": ("qwen3", dict(head_dim=32, rope_parameters={**yarn, "rope_theta": 1000000.0})),
  "qwen3-moe": ("qwen3_moe", dict(head_dim=16, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
                                  norm_topk_prob=False, decoder_sparse_step=1, mlp_only_layers=[0])),
  "qwen3-moe-norm": ("qwen3_moe", dict(head_dim=16, num_experts=6, num_experts_per_tok=3, moe_intermediate_size=48,
                                       norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[])),
  "mixtral": ("mixtral", dict(num_local_experts=4, num_experts_per_tok=2, sliding_window=None)),
  "gemma3": ("gemma3_text", dict(head_dim=16, sliding_window=64, query_pre_attn_scalar=24,
                                 layer_types=["sliding_attention", "full_attention"], final_logit_softcapping=30.0,
                                 rope_parameters={"full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0},
                                                  "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}})),
  "deepseek": ("deepseek_v3", dict(q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=24,
                                   n_routed_experts=8, num_experts_per_tok=3, moe_intermediate_size=32, n_shared_experts=1,
                                   first_k_dense_replace=1, n_group=4, topk_group=2, routed_scaling_factor=2.5,
                                   num_key_value_heads=4)),
  # sliding windows that bind (4 < the 25 positions of prefill + greedy):
  # every layer (Mistral), the local layers only (Gemma 3)
  "mistral-window": ("mistral", dict(sliding_window=4)),
  "gemma3-window": ("gemma3_text", dict(head_dim=16, sliding_window=4, query_pre_attn_scalar=16,
                                        layer_types=["sliding_attention", "full_attention"])),
  "deepseek-yarn": ("deepseek_v3", dict(q_lora_rank=None, kv_lora_rank=32, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=16,
                                        n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=2,
                                        first_k_dense_replace=0, n_group=2, topk_group=1, routed_scaling_factor=1.5,
                                        num_key_value_heads=4, rope_interleave=False,
                                        rope_parameters={**yarn, "rope_theta": 10000.0, "mscale": 1.0, "mscale_all_dim": 0.707})),
}

mt, extra = V[variant]
cfg = AutoConfig.for_model(mt, **{**common, **extra})
cfg._attn_implementation = "eager"
model = AutoModelForCausalLM.from_config(cfg)
model.eval()
gemma = mt.startswith("gemma")
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_((0.0 if gemma else 1.0) + 0.2 * torch.randn_like(p))
        elif name.endswith("bias"):
            p.copy_(0.1 * torch.randn_like(p))
        elif "embed" in name:
            p.copy_(torch.randn_like(p))
        elif p.dim() == 3:   # fused experts [E, out, in]
            p.copy_(torch.randn_like(p) / math.sqrt(p.shape[2]))
        else:
            p.copy_(torch.randn_like(p) / math.sqrt(p.shape[1]))
    for name, b in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.copy_(0.1 * torch.randn_like(b))

model.save_pretrained(out)
g = torch.Generator().manual_seed(seed + 1)
prompt = torch.randint(3, common["vocab_size"], (1, 9), generator=g)
with torch.no_grad():
    logits = model(prompt).logits[0]
    model.generation_config.eos_token_id = None
    greedy = model.generate(prompt, max_new_tokens=16, do_sample=False, min_new_tokens=16)[0]

save_file({"prompt": prompt[0].to(torch.int32), "logits": logits.contiguous().float(), "greedy": greedy.to(torch.int32)},
          f"{out}/reference.safetensors")
