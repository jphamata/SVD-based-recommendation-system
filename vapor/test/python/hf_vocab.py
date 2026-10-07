"""Hugging Face tokenizer.json files for the vocabularies in test/fixtures/vocab.

usage: hf_vocab.py FIXTURE_DIR OUT_DIR

transformers converts each llama.cpp vocabulary GGUF into a `tokenizers`
object; its tokenizer.json is written as OUT_DIR/<name>.json. The
conversion does not restore every option of the original release, so the
originals' configuration is also written:
  llama-bpe-orig : Llama 3 sets `ignore_merges`
  qwen2-orig     : Qwen2 normalizes to NFC
  llama-spm-orig : Llama 2's tokenizer.json (Prepend + Replace normalizer,
                   no pre-tokenizer) instead of the Metaspace form
"""
import copy, json, sys
from transformers import AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
for name in ["llama-bpe", "qwen2", "gpt-2", "llama-spm"]:
    t = AutoTokenizer.from_pretrained(src, gguf_file=f"ggml-vocab-{name}.gguf")
    j = json.loads(t.backend_tokenizer.to_str())
    json.dump(j, open(f"{out}/{name}.json", "w"))
    o = copy.deepcopy(j)
    if name == "llama-bpe":
        o["model"]["ignore_merges"] = True
    elif name == "qwen2":
        o["normalizer"] = {"type": "NFC"}
    elif name == "llama-spm":
        o["normalizer"] = {"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "▁"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]}
        o["pre_tokenizer"] = None
    else:
        continue
    json.dump(o, open(f"{out}/{name}-orig.json", "w"))
