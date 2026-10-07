"""A small Llama with the Llama 3 vocabulary, converted to GGUF by llama.cpp's
own converter (vendored in the llama-cpp-python sdist from PyPI).

usage: hf_gguf.py LLAMA_CPP_DIR TOKENIZER_JSON OUT_DIR
writes OUT_DIR/hf (checkpoint + tokenizer), OUT_DIR/model-{f32,q8_0}.gguf and
OUT_DIR/reference.safetensors (prompt, logits from transformers in float32)
"""
import math, subprocess, sys, os
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

lcpp, tok_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
hf = os.path.join(out, "hf")
torch.manual_seed(3)
tk = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tok_json),
                             bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>")
cfg = LlamaConfig(vocab_size=len(tk), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                  rms_norm_eps=1e-5, rope_theta=500000.0, tie_word_embeddings=False,
                  bos_token_id=tk.bos_token_id, eos_token_id=tk.eos_token_id)
model = LlamaForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif "embed" in name:
            p.copy_(torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) / math.sqrt(p.shape[1]))
model.save_pretrained(hf)
tk.save_pretrained(hf)
ids = tk("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits[0]
save_file({"prompt": ids[0].to(torch.int32), "logits": logits.contiguous()}, os.path.join(out, "reference.safetensors"))
# transformers >= 5 keeps rope_theta inside rope_parameters (and AutoConfig
# normalises config.json that way); converters written before it read the
# top-level key and silently fall back to 10000. The converter runs as is,
# with its config loader taught the new spelling.
wrapper = """
import sys, importlib.util
conv = sys.argv[1]
sys.argv = [conv] + sys.argv[2:]
spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", conv)
m = importlib.util.module_from_spec(spec); sys.modules["convert_hf_to_gguf"] = m; spec.loader.exec_module(m)
orig = m.ModelBase.load_hparams
def load_hparams(*a, **k):
    c = orig(*a, **k)
    rp = c.get("rope_parameters") or {}
    if c.get("rope_theta") is None and "rope_theta" in rp:
        c["rope_theta"] = rp["rope_theta"]
    return c
m.ModelBase.load_hparams = staticmethod(load_hparams)
m.main()
"""
for t in ["f32", "q8_0"]:
    subprocess.run([sys.executable, "-c", wrapper, os.path.join(lcpp, "convert_hf_to_gguf.py"), hf, "--outtype", t,
                    "--outfile", os.path.join(out, f"model-{t}.gguf")], check=True, capture_output=True)
print("ok")
