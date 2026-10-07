"""CLIP's tokenizer, rebuilt from OpenAI's BPE merges exactly as
`clip/simple_tokenizer.py` builds its vocabulary, saved by transformers as
a `tokenizer.json`, and the ids Hugging Face `tokenizers` (Rust) gives for
every line of stdin.

usage: clip_tokenizer.py BPE_GZ OUT_DIR < lines   → OUT_DIR/tokenizer.json, ids as JSON lines on stdout
"""
import gzip, json, sys

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

bpe, out = sys.argv[1], sys.argv[2]
merges = gzip.open(bpe).read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
vocab = list(bytes_to_unicode().values())
vocab = vocab + [v + "</w>" for v in vocab] + ["".join(m.split()) for m in merges] + ["<|startoftext|>", "<|endoftext|>"]
json.dump({t: i for i, t in enumerate(vocab)}, open(out + "/vocab.json", "w"), ensure_ascii=False)
open(out + "/merges.txt", "w").write("#version: 0.2\n" + "\n".join(merges) + "\n")

from transformers import CLIPTokenizer
CLIPTokenizer(out + "/vocab.json", out + "/merges.txt").save_pretrained(out)
from tokenizers import Tokenizer
tk = Tokenizer.from_file(out + "/tokenizer.json")
for line in sys.stdin.read().split("\n"):
    print(json.dumps(tk.encode(line).ids))
