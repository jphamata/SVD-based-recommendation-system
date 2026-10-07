"""Reference dequantisation by gguf-py (llama.cpp's Python package).

usage: ggml_dequant.py [GGUF_PY_DIR]
prints, per type, "<type> <raw hex> <float32 hex>" for 37 random blocks.
"""
import sys, numpy as np
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as Q, GGML_QUANT_SIZES
rng = np.random.default_rng(0)
out = []
for name in ["Q8_0","Q4_0","Q4_1","Q5_0","Q5_1","Q4_K","Q5_K","Q6_K","F16","BF16"]:
    qt = Q[name]
    bs, ts = GGML_QUANT_SIZES[qt]
    nb = 37
    raw = rng.integers(0, 256, size=nb*ts, dtype=np.uint8)
    # sane f16 scales: overwrite fp16 fields with moderate values
    if name in ["Q8_0","Q4_0","Q5_0"]:
        r = raw.reshape(nb, ts); r[:, 0:2] = np.frombuffer(rng.uniform(-1,1,nb).astype(np.float16).tobytes(), np.uint8).reshape(nb,2)
    if name in ["Q4_1","Q5_1","Q4_K","Q5_K"]:
        r = raw.reshape(nb, ts); r[:, 0:4] = np.frombuffer(rng.uniform(-1,1,2*nb).astype(np.float16).tobytes(), np.uint8).reshape(nb,4)
    if name == "Q6_K":
        r = raw.reshape(nb, ts); r[:, ts-2:ts] = np.frombuffer(rng.uniform(-1,1,nb).astype(np.float16).tobytes(), np.uint8).reshape(nb,2)
    if name == "F16":
        raw = rng.uniform(-100,100,nb*ts//2).astype(np.float16).view(np.uint8)
    if name == "BF16":
        raw = (rng.uniform(-100,100,nb*ts//2).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    deq = dequantize(raw, qt).astype(np.float32)
    out.append(f"{name.lower()} {raw.tobytes().hex()} {deq.tobytes().hex()}")
print("\n".join(out))
