"""Train the small, *real* transformers of the fusion benchmark (not planted).

usage: train_merge_models.py PRIV_QUALITY_DIR OUT_DIR

Character-level Llama-topology decoders (vocab 48 = vapor's planted
alphabet, width 64, 2 layers, 4 heads) trained with AdamW by PyTorch on the
frozen corpora of priv/quality:

  base       mixed Portuguese + English, the "pretrained" model
  ft_pt      base fine-tuned on Portuguese     (small, dense deltas from base)
  ft_en      base fine-tuned on English
  solo_pt    trained from scratch on Portuguese, another seed
  solo_en    trained from scratch on English,   another seed  (no common base)

Each OUT_DIR/<name>/ is a transformers checkpoint (config.json +
model.safetensors) that vapor admits through the airlock. The checkpoints
are frozen in the repository (their SHA-256 in MANIFEST); this script is
how they were made, so anyone can retrain and compare.
"""
import os, sys, json, hashlib
import torch
from transformers import LlamaConfig, LlamaForCausalLM

src, out = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)

chars = [chr(c) for c in range(ord('a'), ord('z') + 1)] + [str(d) for d in range(10)] + list(" .,;:!?-'\n")
index = {c: i for i, c in enumerate(chars)}
OTHER = len(chars)          # 46: the alphabet is 47 ids, padded to a vocabulary of 48


def encode(text):
    # the planted alphabet's encode: lowercase bytes, everything else one id
    return [index.get(chr(b), OTHER) if b < 128 else OTHER for b in text.lower().encode("utf-8")]


def corpus(name):
    with open(os.path.join(src, name), "rb") as f:
        return torch.tensor(encode(f.read().decode("utf-8")), dtype=torch.long)


pt, en = corpus("pt_reference.txt"), corpus("en_reference.txt")
mixed = torch.cat([pt, en])

cfg = LlamaConfig(vocab_size=48, hidden_size=64, intermediate_size=176, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=4, head_dim=16, max_position_embeddings=256, rms_norm_eps=1e-5,
                  rope_theta=10000.0, tie_word_embeddings=False, bos_token_id=None, eos_token_id=None, pad_token_id=None)
cfg._attn_implementation = "eager"


def batches(data, seed, n, bs=32, seq=128):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        idx = torch.randint(0, len(data) - seq - 1, (bs,), generator=g)
        yield torch.stack([data[i:i + seq + 1] for i in idx])


def train(model, data, steps, lr, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    model.train()
    for i, b in enumerate(batches(data, seed, steps)):
        # labels = inputs: transformers shifts them, predicting token t+1 from tokens ≤ t
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def holdout_bits(model):
    out = []
    for f in ("pt_holdout.txt", "en_holdout.txt"):
        ids = corpus(f)[:1024]
        tot, n = 0.0, 0
        for i in range(0, 1024, 256):
            ch = ids[i:i + 256][None]
            with torch.no_grad():
                tot += model(input_ids=ch, labels=ch).loss.item() * (ch.shape[1] - 1)
            n += ch.shape[1] - 1
        out.append(round(tot / n / 0.6931471805599453, 4))
    return out


def save(model, name):
    print(name, "holdout bits/char (pt, en):", holdout_bits(model), flush=True)
    d = os.path.join(out, name)
    model.save_pretrained(d)
    with open(os.path.join(d, "model.safetensors"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


BASE_STEPS, FT_STEPS, FT_LR = int(os.environ.get("BASE_STEPS", 500)), int(os.environ.get("FT_STEPS", 150)), float(os.environ.get("FT_LR", 1e-4))
manifest = {}
torch.manual_seed(1)
base = train(LlamaForCausalLM(cfg), mixed, BASE_STEPS, 3e-3, 10)
manifest["base"] = save(base, "base")

for name, data, seed in [("ft_pt", pt, 20), ("ft_en", en, 30)]:
    m = LlamaForCausalLM(cfg)
    m.load_state_dict(base.state_dict())
    m = train(m, data, FT_STEPS, FT_LR, seed)
    manifest[name] = save(m, name)

for name, data, seed in [("solo_pt", pt, 40), ("solo_en", en, 50)]:
    torch.manual_seed(seed)
    m = train(LlamaForCausalLM(cfg), data, BASE_STEPS + FT_STEPS, 3e-3, seed + 1)
    manifest[name] = save(m, name)

with open(os.path.join(out, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=1, sort_keys=True)
print(json.dumps(manifest, indent=1))
