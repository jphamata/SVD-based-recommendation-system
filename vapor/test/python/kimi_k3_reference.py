"""An independent reference for the Kimi K3 adapter (numpy, float64).

usage: kimi_k3_reference.py OUT_DIR VARIANT SEED

Written from the equations of the Kimi K3 technical report (Kimi Team, 2026)
and nothing else: it shares no code with vapor, and its algorithms are not
vapor's. vapor runs one token per step with caches; this runs the whole
sequence at every step with none:

  * Kimi Delta Attention by the recurrence of Eq. 1 (and, as a check of the
    report, by the chunkwise form of Eq. 4 with the UT transform derived
    below; the two must agree);
  * Gated MLA with NoPE by decompressed keys and values (vapor absorbs the
    key map into the query and caches the latent);
  * Block Attention Residuals (Eq. 8-10), Stable LatentMoE (Eq. 11, 13),
    SiTU-GLU (Eq. 12).

It writes a checkpoint (config.json, model.safetensors) and
reference.safetensors:

    prompt : int32[T]   logits : float32[T, V]   greedy : int32[T + N]
    chunk_gap : float32[1]      max |recurrent - chunkwise| over every KDA layer
    overflow : float32[2]       finite reciprocal decays over a 16-token tile in
                                float32: [bounded (K3), softplus (Kimi Linear)]

Variants:
  k3        the report's SiTU-GLU caps (4, 25), a direct query projection
  k3-tight  caps (0.5, 0.8) so the soft caps bite, a low-rank query
            (q_lora_rank), routed_scaling_factor 2.5, and the routed experts
            stored as MXFP4 (latent and expert widths 32; uint8 blocks of 32 E2M1 codes, low nibble first,
            and one E8M0 scale per block)
"""
import json
import struct
import sys

import numpy as np

out, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = np.random.default_rng(seed)

V, D, L = 80, 32, 6
cfg = {
    "model_type": "kimi_k3", "vocab_size": V, "hidden_size": D, "num_hidden_layers": L,
    "rms_norm_eps": 1e-6, "l2norm_eps": 1e-6, "tie_word_embeddings": False,
    "bos_token_id": 1, "eos_token_id": 2,
    "kda_num_heads": 2, "kda_head_dim": 16, "short_conv_kernel_size": 4, "kda_alpha_rank": 16, "kda_g_min": -5.0,
    "num_attention_heads": 2, "q_lora_rank": None, "kv_lora_rank": 16, "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0, "v_head_dim": 16,
    "first_k_dense_replace": 1, "intermediate_size": 48,
    "n_routed_experts": 16, "num_experts_per_tok": 3, "moe_intermediate_size": 16, "moe_latent_size": 16,
    "n_shared_experts": 2, "shared_expert_intermediate_size": 16, "routed_scaling_factor": 1.0,
    "hidden_act": "situ_glu", "situ_beta": [4.0, 25.0], "attn_res_block_size": 2,
}
mxfp4 = False
if variant == "k3-tight":
    cfg.update({"situ_beta": [0.5, 0.8], "q_lora_rank": 16, "routed_scaling_factor": 2.5,
                "moe_intermediate_size": 32, "moe_latent_size": 32})
    mxfp4 = True
elif variant != "k3":
    sys.exit("unknown variant " + variant)

# the K3 pattern: three KDA layers then one Gated MLA, and a final Gated MLA
types = ["mla" if (l % 4 == 3 or l == L - 1) else "kda" for l in range(L)]
H, DK, DV, K = cfg["kda_num_heads"], cfg["kda_head_dim"], cfg["kda_head_dim"], cfg["short_conv_kernel_size"]
RA, GMIN = cfg["kda_alpha_rank"], cfg["kda_g_min"]
MH, RQ, RKV, DQK, MDV = cfg["num_attention_heads"], cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["qk_nope_head_dim"], cfg["v_head_dim"]
E, TOPK, IE, LAT = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"], cfg["moe_latent_size"]
IS = cfg["n_shared_experts"] * cfg["shared_expert_intermediate_size"]
B1, B2 = cfg["situ_beta"]
EPS, L2EPS, SCALE, BLOCK = cfg["rms_norm_eps"], cfg["l2norm_eps"], cfg["routed_scaling_factor"], cfg["attn_res_block_size"]

W = {}


def mat(name, shape, s=None):
    s = s if s is not None else 1.0 / np.sqrt(shape[-1])
    W[name] = (rng.standard_normal(shape) * s).astype(np.float32)


def vec(name, n, mean=0.0, s=0.1):
    W[name] = (mean + rng.standard_normal(n) * s).astype(np.float32)


# E2M1 code points and an E8M0 scale per 32 values: an MXFP4 matrix is
# exactly representable in binary32, so its float32 copy is its value
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
packed = {}


def mx_mat(name, shape):
    rows, k = shape
    codes = rng.integers(0, 16, size=(rows, k)).astype(np.uint8)
    scales = rng.integers(122, 126, size=(rows, k // 32)).astype(np.uint8)  # 2^-5 .. 2^-2
    vals = E2M1[codes] * np.repeat(np.exp2(scales.astype(np.float64) - 127), 32, axis=1)
    W[name] = vals.astype(np.float32)
    blocks = (codes[:, 0::2] | (codes[:, 1::2] << 4)).reshape(rows, k // 32, 16)
    base = name[: -len(".weight")]
    packed[base + ".weight_blocks"] = blocks
    packed[base + ".weight_scales"] = scales


mat("model.embed_tokens.weight", [V, D], 1.0)
vec("model.norm.weight", D, 1.0)
mat("lm_head.weight", [V, D])
vec("model.attn_res.final_query", D, 0.0, 0.5)

for l, t in enumerate(types):
    p = f"model.layers.{l}."
    vec(p + "input_layernorm.weight", D, 1.0)
    vec(p + "post_attention_layernorm.weight", D, 1.0)
    vec(p + "attn_res.attn_query", D, 0.0, 0.5)
    vec(p + "attn_res.ffn_query", D, 0.0, 0.5)
    a = p + "self_attn."
    if t == "kda":
        mat(a + "q_proj.weight", [H * DK, D]); mat(a + "k_proj.weight", [H * DK, D]); mat(a + "v_proj.weight", [H * DV, D])
        for x, w in (("q", DK), ("k", DK), ("v", DV)):
            W[a + f"{x}_conv1d.weight"] = (rng.standard_normal([H * w, 1, K]) * 0.5).astype(np.float32)
        mat(a + "b_proj.weight", [H, D]); mat(a + "f_a_proj.weight", [RA, D]); mat(a + "f_b_proj.weight", [H * DK, RA])
        vec(a + "dt_bias", H * DK, 0.0, 1.0); vec(a + "A_log", H, 0.0, 0.5)
        mat(a + "g_proj.weight", [H * DV, D]); vec(a + "o_norm.weight", DV, 1.0); mat(a + "o_proj.weight", [D, H * DV])
    else:
        if RQ:
            mat(a + "q_a_proj.weight", [RQ, D]); vec(a + "q_a_layernorm.weight", RQ, 1.0); mat(a + "q_b_proj.weight", [MH * DQK, RQ])
        else:
            mat(a + "q_proj.weight", [MH * DQK, D])
        mat(a + "kv_a_proj_with_mqa.weight", [RKV, D]); vec(a + "kv_a_layernorm.weight", RKV, 1.0)
        mat(a + "kv_b_proj.weight", [MH * (DQK + MDV), RKV])
        mat(a + "g_proj.weight", [MH * MDV, D]); mat(a + "o_proj.weight", [D, MH * MDV])
    m = p + "mlp."
    if l < cfg["first_k_dense_replace"]:
        mat(m + "gate_proj.weight", [cfg["intermediate_size"], D], 0.6); mat(m + "up_proj.weight", [cfg["intermediate_size"], D], 0.6)
        mat(m + "down_proj.weight", [D, cfg["intermediate_size"]])
    else:
        mat(m + "gate.weight", [E, D]); vec(m + "gate.e_score_correction_bias", E, 0.0, 0.05)
        mat(m + "latent_down.weight", [LAT, D]); vec(m + "latent_norm.weight", LAT, 1.0); mat(m + "latent_up.weight", [D, LAT])
        for e in range(E):
            for x, shp in (("gate_proj", [IE, LAT]), ("up_proj", [IE, LAT]), ("down_proj", [LAT, IE])):
                n = m + f"experts.{e}.{x}.weight"
                mx_mat(n, shp) if mxfp4 else mat(n, shp, 0.6 if x != "down_proj" else None)
        mat(m + "shared_experts.gate_proj.weight", [IS, D], 0.6); mat(m + "shared_experts.up_proj.weight", [IS, D], 0.6)
        mat(m + "shared_experts.down_proj.weight", [D, IS])


def save(path, tensors):
    dt = {np.dtype(np.float32): "F32", np.dtype(np.int32): "I32", np.dtype(np.uint8): "U8"}
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        b = a.tobytes()
        header[name] = {"dtype": dt[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


stored = {k: v for k, v in W.items() if not (mxfp4 and ".experts." in k)}
stored.update(packed)
save(f"{out}/model.safetensors", stored)
with open(f"{out}/config.json", "w") as f:
    json.dump(cfg, f, indent=1)

# ------------------------------------------------------------- the model --
w = {k: v.astype(np.float64) for k, v in W.items()}
sig = lambda x: 1.0 / (1.0 + np.exp(-x))
silu = lambda x: x * sig(x)


def rms(x, g=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return y if g is None else y * g


def l2(x):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + L2EPS)


def situ(g, u):
    return B1 * np.tanh(g / B1) * sig(g) * B2 * np.tanh(u / B2)


def conv(x, wt):
    # depthwise causal: y_t = sum_j w[:, j] * x_{t-(K-1)+j}
    T = x.shape[0]
    xp = np.concatenate([np.zeros((K - 1, x.shape[1])), x])
    return sum(wt[:, 0, j] * xp[j:j + T] for j in range(K))


def kda_recurrent(q, k, v, beta, alpha):
    # Eq. 1: S_t = (I - b k k^T) Diag(a) S_{t-1} + b k v^T,  o_t = S_t^T q_t   (per head)
    T = q.shape[0]
    o = np.zeros((T, H, DV))
    for h in range(H):
        S = np.zeros((DK, DV))
        for t in range(T):
            kt = k[t, h]
            S = alpha[t, h][:, None] * S
            S = S - beta[t, h] * np.outer(kt, kt @ S) + beta[t, h] * np.outer(kt, v[t, h])
            o[t, h] = S.T @ q[t, h]
    return o


def kda_chunkwise(q, k, v, beta, alpha, C=4):
    # Eq. 4 with the UT transform (derived here, as the report defers to Kimi Linear):
    # within a chunk, U~ = T diag(b) V - T diag(b) (K*Gam) S with T = (I + diag(b) Ls)^-1,
    # Ls_ri = (k_r*gam_r).(k_i/gam_i) for i < r; then O = (Gam*Q) S + Tril((Q*Gam)(K/Gam)^T) U~
    # and S' = Gam_C (S + (K/Gam)^T U~)
    T = q.shape[0]
    o = np.zeros((T, H, DV))
    for h in range(H):
        S = np.zeros((DK, DV))
        for s0 in range(0, T, C):
            idx = range(s0, min(s0 + C, T))
            Q, Kc, Vc = q[idx, h], k[idx, h], v[idx, h]
            b, a = beta[idx, h], alpha[idx, h]
            gam = np.cumprod(a, axis=0)
            n = len(idx)
            Ls = np.tril((Kc * gam) @ (Kc / gam).T, -1)
            Tm = np.linalg.inv(np.eye(n) + b[:, None] * Ls)
            U = Tm @ (b[:, None] * Vc) - Tm @ (b[:, None] * (Kc * gam)) @ S
            o[list(idx), h] = (gam * Q) @ S + np.tril((Q * gam) @ (Kc / gam).T) @ U
            S = gam[-1][:, None] * (S + (Kc / gam).T @ U)
    return o


chunk_gap = 0.0


def kda(l, x):
    global chunk_gap
    a = f"model.layers.{l}.self_attn."
    T = x.shape[0]
    q = l2(silu(conv(x @ w[a + "q_proj.weight"].T, w[a + "q_conv1d.weight"])).reshape(T, H, DK))
    k = l2(silu(conv(x @ w[a + "k_proj.weight"].T, w[a + "k_conv1d.weight"])).reshape(T, H, DK))
    v = silu(conv(x @ w[a + "v_proj.weight"].T, w[a + "v_conv1d.weight"])).reshape(T, H, DV)
    beta = sig(x @ w[a + "b_proj.weight"].T)
    z = (x @ w[a + "f_a_proj.weight"].T) @ w[a + "f_b_proj.weight"].T + w[a + "dt_bias"]
    z = z.reshape(T, H, DK)
    g = GMIN * sig(np.exp(w[a + "A_log"])[None, :, None] * z)  # Eq. 5, lower-bounded log-decay
    alpha = np.exp(g)
    o = kda_recurrent(q, k, v, beta, alpha)
    chunk_gap = max(chunk_gap, float(np.max(np.abs(o - kda_chunkwise(q, k, v, beta, alpha)))))
    o = rms(o, w[a + "o_norm.weight"]).reshape(T, H * DV)
    return (sig(x @ w[a + "g_proj.weight"].T) * o) @ w[a + "o_proj.weight"].T  # Eq. 6


def mla(l, x):
    a = f"model.layers.{l}.self_attn."
    T = x.shape[0]
    if RQ:
        q = rms(x @ w[a + "q_a_proj.weight"].T, w[a + "q_a_layernorm.weight"]) @ w[a + "q_b_proj.weight"].T
    else:
        q = x @ w[a + "q_proj.weight"].T
    q = q.reshape(T, MH, DQK)
    c = rms(x @ w[a + "kv_a_proj_with_mqa.weight"].T, w[a + "kv_a_layernorm.weight"])
    kv = (c @ w[a + "kv_b_proj.weight"].T).reshape(T, MH, DQK + MDV)
    kk, vv = kv[:, :, :DQK], kv[:, :, DQK:]
    o = np.zeros((T, MH, MDV))
    for h in range(MH):
        s = (q[:, h] @ kk[:, h].T) / np.sqrt(DQK)  # NoPE: no rotation of q or k
        s = np.where(np.tril(np.ones((T, T))) > 0, s, -np.inf)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        o[:, h] = (p / p.sum(axis=1, keepdims=True)) @ vv[:, h]
    return (sig(x @ w[a + "g_proj.weight"].T) * o.reshape(T, MH * MDV)) @ w[a + "o_proj.weight"].T  # Eq. 7


def ffn(l, x):
    m = f"model.layers.{l}.mlp."
    if l < cfg["first_k_dense_replace"]:
        return situ(x @ w[m + "gate_proj.weight"].T, x @ w[m + "up_proj.weight"].T) @ w[m + "down_proj.weight"].T
    shared = situ(x @ w[m + "shared_experts.gate_proj.weight"].T, x @ w[m + "shared_experts.up_proj.weight"].T) @ w[m + "shared_experts.down_proj.weight"].T
    s = sig(x @ w[m + "gate.weight"].T)
    choice = s + w[m + "gate.e_score_correction_bias"]
    z = x @ w[m + "latent_down.weight"].T
    u = np.zeros((x.shape[0], LAT))
    for t in range(x.shape[0]):
        top = sorted(range(E), key=lambda j: (-choice[t, j], j))[:TOPK]  # Eq. 13, ties to the lower index
        den = sum(s[t, j] for j in top)
        for j in top:
            e = f"{m}experts.{j}."
            y = situ(z[t] @ w[e + "gate_proj.weight"].T, z[t] @ w[e + "up_proj.weight"].T) @ w[e + "down_proj.weight"].T
            u[t] += (s[t, j] / den * SCALE) * y
    return shared + rms(u, w[m + "latent_norm.weight"]) @ w[m + "latent_up.weight"].T  # Eq. 11


def attn_res(sources, query):
    # Eq. 9: softmax over sources of q . RMSNorm(v), then the weighted sum
    if len(sources) == 1:
        return sources[0]
    sc = np.stack([rms(v) @ query for v in sources], axis=0)
    p = np.exp(sc - sc.max(axis=0))
    p = p / p.sum(axis=0)
    return sum(p[i][:, None] * v for i, v in enumerate(sources))


def forward(toks):
    x = w["model.embed_tokens.weight"][np.array(toks)]
    done, partial = [x], None
    for l, t in enumerate(types):
        p = f"model.layers.{l}."
        for kind in ("attn", "ffn"):
            src = done + ([partial] if partial is not None else [])
            h = attn_res(src, w[p + f"attn_res.{kind}_query"])
            u = rms(h, w[p + ("input_layernorm.weight" if kind == "attn" else "post_attention_layernorm.weight")])
            y = (kda if t == "kda" else mla)(l, u) if kind == "attn" else ffn(l, u)
            partial = y if partial is None else partial + y
        if (l + 1) % BLOCK == 0 or l == L - 1:
            done, partial = done + [partial], None
    h = attn_res(done, w["model.attn_res.final_query"])
    return rms(h, w["model.norm.weight"]) @ w["lm_head.weight"].T


prompt = [int(t) for t in rng.integers(3, V, size=12)]
logits = forward(prompt)
greedy = list(prompt)
for _ in range(10):
    greedy.append(int(np.argmax(forward(greedy)[-1])))

# the report's reason for the bounded decay: over a 16-token tile the chunkwise
# form divides by the cumulative decay; with g in (-5, 0) the reciprocal stays
# below e^80, finite in float32 (and bf16); with Kimi Linear's -e^A softplus(z),
# large decay logits make it overflow
zt = np.full(16, 30.0)
with np.errstate(over="ignore"):
    bounded = np.exp(-np.cumsum(GMIN * sig(zt))).astype(np.float32).max()
    softplus = np.exp(-np.cumsum(-np.log1p(np.exp(zt)))).astype(np.float32).max()
overflow = np.array([np.isfinite(bounded), np.isfinite(softplus)], dtype=np.float32)

save(f"{out}/reference.safetensors", {
    "prompt": np.array(prompt, dtype=np.int32), "logits": logits.astype(np.float32),
    "greedy": np.array(greedy, dtype=np.int32), "chunk_gap": np.array([chunk_gap], dtype=np.float32),
    "overflow": overflow,
})
print(json.dumps({"chunk_gap": chunk_gap, "types": types}))
