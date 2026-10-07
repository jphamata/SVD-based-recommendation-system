"""Hugging Face `tokenizers` as the reference for vapor's tokenizer tier.

usage: hf_tokenize.py TOKENIZER_JSON < texts   (texts separated by NUL)
prints, per text, the ids of encode(text, add_special_tokens=False) and
then the decoded string's UTF-8 bytes in hex, tab-separated.
"""
import sys
from tokenizers import Tokenizer

tk = Tokenizer.from_file(sys.argv[1])
texts = sys.stdin.buffer.read().decode("utf-8").split("\x00")[:-1]
out = []
for t in texts:
    ids = tk.encode(t, add_special_tokens=False).ids
    dec = tk.decode(ids, skip_special_tokens=False)
    out.append(" ".join(map(str, ids)) + "\t" + dec.encode("utf-8").hex())
sys.stdout.write("\n".join(out) + "\n")
