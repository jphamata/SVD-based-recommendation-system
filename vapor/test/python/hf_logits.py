"""Logits of a checkpoint directory under Hugging Face transformers.

usage: hf_logits.py DIR TOKENS_JSON OUT.safetensors
Loads DIR with AutoModelForCausalLM (float32, eager attention), runs the
prompt and writes float32[T, V] `logits`.
"""
import json, sys
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

d, toks, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32, attn_implementation="eager").eval()
with torch.no_grad():
    z = m(torch.tensor([toks])).logits[0].float().contiguous()
save_file({"logits": z}, out)
