"""Independent NumPy (float64) references for the model airlock's adapters.

Written from the Hugging Face modelling code's *semantics*, without any of
vapor's code paths: fused projections are split here by NumPy slicing,
convolutions are computed as convolutions, attention by explicit heads.

usage: np_reference.py MODE weights.safetensors config.json  < input.json  > output.json
  MODE decoder  input {"tokens": [...]}           output {"logits": [[...], ...]}
  MODE vit      input {"pixels": [[[...]]] (C,H,W)} output {"hidden": [[...]], "logits": [...]}
"""
import json, math, struct, sys
import numpy as np


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, h in header.items():
        if name == "__metadata__":
            continue
        assert h["dtype"] == "F32", h["dtype"]
        a, b = h["data_offsets"]
        out[name] = np.frombuffer(data[a:b], dtype="<f4").astype(np.float64).reshape(h["shape"])
    return out


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def layernorm(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def silu(x):
    return x / (1 + np.exp(-x))


gelu = np.vectorize(lambda v: 0.5 * v * (1 + math.erf(v / math.sqrt(2))))


def rope(x, theta, dh):
    t = x.shape[0]
    inv = 1.0 / theta ** (np.arange(0, dh, 2) / dh)
    ang = np.arange(t)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang)] * 2, -1)[:, None, :]
    sin = np.concatenate([np.sin(ang)] * 2, -1)[:, None, :]
    half = dh // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos + rot * sin


def decoder(w, c, tokens):
    mt = c["model_type"]
    d, L, h = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
    hkv = c.get("num_key_value_heads", h)
    dh = c.get("head_dim") or d // h
    eps = c.get("rms_norm_eps", 1e-6)
    theta = c.get("rope_theta", 10000.0)
    em = c.get("embedding_multiplier", 1.0) if mt == "granite" else 1.0
    rm = c.get("residual_multiplier", 1.0) if mt == "granite" else 1.0
    scale = c.get("attention_multiplier") if mt == "granite" else None
    scale = scale if scale is not None else 1.0 / math.sqrt(dh)
    ls = c.get("logits_scaling", 1.0) if mt == "granite" else 1.0
    bias = c.get("attention_bias", False)
    T = len(tokens)
    E = w["model.embed_tokens.weight"]
    x = E[tokens] * em
    for l in range(L):
        p = f"model.layers.{l}."
        hh = rmsnorm(x, w[p + "input_layernorm.weight"], eps)
        if mt == "phi3":
            qkv = hh @ w[p + "self_attn.qkv_proj.weight"].T
            q, k, v = qkv[:, : h * dh], qkv[:, h * dh : (h + hkv) * dh], qkv[:, (h + hkv) * dh :]
        else:
            q = hh @ w[p + "self_attn.q_proj.weight"].T
            k = hh @ w[p + "self_attn.k_proj.weight"].T
            v = hh @ w[p + "self_attn.v_proj.weight"].T
            if bias:
                q, k, v = q + w[p + "self_attn.q_proj.bias"], k + w[p + "self_attn.k_proj.bias"], v + w[p + "self_attn.v_proj.bias"]
        q = rope(q.reshape(T, h, dh), theta, dh)
        k = rope(k.reshape(T, hkv, dh), theta, dh)
        v = v.reshape(T, hkv, dh)
        rep = h // hkv
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        s = np.einsum("thd,shd->hts", q, k) * scale
        s = s + np.triu(np.full((T, T), -np.inf), 1)[None]
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s / s.sum(-1, keepdims=True)
        a = np.einsum("hts,shd->thd", s, v).reshape(T, h * dh)
        o = a @ w[p + "self_attn.o_proj.weight"].T
        if bias and mt != "phi3":
            o = o + w[p + "self_attn.o_proj.bias"]
        x = x + o * rm
        h2 = rmsnorm(x, w[p + "post_attention_layernorm.weight"], eps)
        if mt == "phi3":
            gu = h2 @ w[p + "mlp.gate_up_proj.weight"].T
            gate, up = np.split(gu, 2, axis=-1)
            f = (up * silu(gate)) @ w[p + "mlp.down_proj.weight"].T
        else:
            f = (silu(h2 @ w[p + "mlp.gate_proj.weight"].T) * (h2 @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
        x = x + f * rm
    xf = rmsnorm(x, w["model.norm.weight"], eps)
    head = w.get("lm_head.weight", E)
    return {"logits": ((xf @ head.T) / ls).tolist()}


def vit(w, c, pixels):
    px = np.array(pixels, dtype=np.float64)  # (C, H, W)
    pre = "vit." if "vit.embeddings.cls_token" in w else ""
    d, L, h = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
    p = c["patch_size"]
    eps = c.get("layer_norm_eps", 1e-12)
    W = w[pre + "embeddings.patch_embeddings.projection.weight"]  # (d, C, p, p)
    b = w[pre + "embeddings.patch_embeddings.projection.bias"]
    C, H, Wd = px.shape
    emb = []
    for py in range(H // p):
        for pxx in range(Wd // p):
            patch = px[:, py * p : (py + 1) * p, pxx * p : (pxx + 1) * p]
            emb.append(np.einsum("dcij,cij->d", W, patch) + b)
    x = np.concatenate([w[pre + "embeddings.cls_token"].reshape(1, d), np.array(emb)], 0)
    x = x + w[pre + "embeddings.position_embeddings"].reshape(-1, d)
    T = x.shape[0]
    dh = d // h
    for l in range(L):
        q_ = pre + f"encoder.layer.{l}."
        hh = layernorm(x, w[q_ + "layernorm_before.weight"], w[q_ + "layernorm_before.bias"], eps)
        q = (hh @ w[q_ + "attention.attention.query.weight"].T + w[q_ + "attention.attention.query.bias"]).reshape(T, h, dh)
        k = (hh @ w[q_ + "attention.attention.key.weight"].T + w[q_ + "attention.attention.key.bias"]).reshape(T, h, dh)
        v = (hh @ w[q_ + "attention.attention.value.weight"].T + w[q_ + "attention.attention.value.bias"]).reshape(T, h, dh)
        s = np.einsum("thd,shd->hts", q, k) / math.sqrt(dh)
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s / s.sum(-1, keepdims=True)
        a = np.einsum("hts,shd->thd", s, v).reshape(T, d)
        x = x + a @ w[q_ + "attention.output.dense.weight"].T + w[q_ + "attention.output.dense.bias"]
        h2 = layernorm(x, w[q_ + "layernorm_after.weight"], w[q_ + "layernorm_after.bias"], eps)
        f = gelu(h2 @ w[q_ + "intermediate.dense.weight"].T + w[q_ + "intermediate.dense.bias"])
        x = x + f @ w[q_ + "output.dense.weight"].T + w[q_ + "output.dense.bias"]
    xf = layernorm(x, w[pre + "layernorm.weight"], w[pre + "layernorm.bias"], eps)
    out = {"hidden": xf.tolist()}
    if "classifier.weight" in w:
        out["logits"] = (xf[0] @ w["classifier.weight"].T + w["classifier.bias"]).tolist()
    return out


if __name__ == "__main__":
    mode, wpath, cpath = sys.argv[1:4]
    w = read_safetensors(wpath)
    c = json.load(open(cpath))
    inp = json.load(sys.stdin)
    out = decoder(w, c, inp["tokens"]) if mode == "decoder" else vit(w, c, inp["pixels"])
    json.dump(out, sys.stdout)
