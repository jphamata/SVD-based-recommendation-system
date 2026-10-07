"""Run a StableHLO module exported by vapor (Vapor.Export.StableHLO) on a
PJRT device through JAX — XLA's CPU here; Tenstorrent's tt-xla, a TPU or a
GPU wherever their PJRT plugin is installed (`--platform tt`, `tpu`, `cuda`).

usage: stablehlo_run.py DIR [--platform cpu]
DIR holds module.mlir and manifest.json:
  {"inputs": [{"name", "dtype", "shape", "file"}], "outputs": [{"name", "dtype", "shape"}]}
Outputs are written as DIR/out_<name>.bin (raw little-endian), and
DIR/device.json names the device that ran them.
"""
import json, os, sys
import numpy as np

import jax
from jax._src import xla_bridge
from jax._src.lib import xla_client

d = sys.argv[1]
platform = "cpu"
if "--platform" in sys.argv:
    platform = sys.argv[sys.argv.index("--platform") + 1]

DT = {"f32": np.float32, "bf16": None, "s32": np.int32, "s8": np.int8, "u8": np.uint8, "f16": np.float16}

man = json.load(open(os.path.join(d, "manifest.json")))
mod = open(os.path.join(d, "module.mlir")).read()
backend = xla_bridge.get_backend(platform)
dev = backend.devices()[0]
exe = backend.compile_and_load(mod, xla_client.DeviceList((dev,)), xla_client.CompileOptions())

args = []
for i in man["inputs"]:
    raw = open(os.path.join(d, i["file"]), "rb").read()
    if i["dtype"] == "bf16":
        a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        a = jax.numpy.asarray(a.reshape(i["shape"]), dtype=jax.numpy.bfloat16)
    else:
        a = np.frombuffer(raw, DT[i["dtype"]]).reshape(i["shape"])
    args.append(jax.device_put(a, dev))

res = exe.execute_sharded(args).disassemble_into_single_device_arrays()
for o, r in zip(man["outputs"], res):
    a = np.asarray(r[0])
    open(os.path.join(d, "out_%s.bin" % o["name"]), "wb").write(np.ascontiguousarray(a).tobytes())

json.dump({"platform": platform, "device": str(dev), "kind": getattr(dev, "device_kind", "?"),
           "jax": jax.__version__}, open(os.path.join(d, "device.json"), "w"))
