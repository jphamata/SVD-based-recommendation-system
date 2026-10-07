"""JPEG fixtures and Pillow's decoding of them (the oracle of Vapor.Docs.JPEG).

usage: jpeg_fixtures.py OUT_DIR

Writes OUT_DIR/<name>.jpg for every layout the decoder claims — 4:4:4,
4:2:2, 4:2:0, 4:4:0 (h1v2, via cjpeg when present), 4:1:1, grayscale,
progressive with successive approximation, restart intervals, optimised
Huffman tables, Adobe RGB, odd sizes, high and low quality — from two real
photographs (scikit-learn's china.jpg and flower.jpg, re-encoded and also
as they are) and a synthetic pattern with hard edges; plus OUT_DIR/<name>.raw,
Pillow's decoded pixels (8-bit, interleaved), and OUT_DIR/index.json
{name: [width, height, mode]}. Refusals: arithmetic coding, 12-bit.
"""
import io, json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
index = {}

import sklearn
imgdir = os.path.join(os.path.dirname(sklearn.__file__), "datasets", "images")
china = Image.open(os.path.join(imgdir, "china.jpg"))
flower = Image.open(os.path.join(imgdir, "flower.jpg"))

# a synthetic image with hard edges, gradients and saturated colours (stresses clamping)
h, w = 61, 83
yy, xx = np.mgrid[0:h, 0:w]
synth = np.zeros((h, w, 3), np.uint8)
synth[..., 0] = (xx * 3) % 256
synth[..., 1] = ((yy // 7 + xx // 9) % 2) * 255
synth[..., 2] = np.where((xx - 40) ** 2 + (yy - 30) ** 2 < 300, 250, 10)
synth = Image.fromarray(synth, "RGB")


def put(name, data):
    path = os.path.join(out, name + ".jpg")
    with open(path, "wb") as f:
        f.write(data)
    im = Image.open(path)
    im.load()
    with open(os.path.join(out, name + ".raw"), "wb") as f:
        f.write(im.tobytes())
    index[name] = [im.width, im.height, im.mode]


def enc(im, **kw):
    b = io.BytesIO()
    im.save(b, "JPEG", **kw)
    return b.getvalue()


# the real photographs as shipped
for nm in ("china", "flower"):
    with open(os.path.join(imgdir, nm + ".jpg"), "rb") as f:
        put(nm + "_original", f.read())

crop = china.crop((100, 50, 100 + 237, 50 + 151))      # odd sizes
small = flower.resize((97, 65))
for src_name, src in [("china", crop), ("flower", small), ("synth", synth)]:
    for ss, tag in [(0, "444"), (1, "422"), (2, "420")]:
        put(f"{src_name}_{tag}", enc(src, quality=90, subsampling=ss))
    put(f"{src_name}_420_q30", enc(src, quality=30, subsampling=2))
    put(f"{src_name}_420_q100", enc(src, quality=100, subsampling=2))
    put(f"{src_name}_prog", enc(src, quality=85, subsampling=2, progressive=True))
    put(f"{src_name}_prog444", enc(src, quality=95, subsampling=0, progressive=True))
    put(f"{src_name}_opt", enc(src, quality=80, subsampling=1, optimize=True))
    put(f"{src_name}_gray", enc(src.convert("L"), quality=88))
    put(f"{src_name}_gray_prog", enc(src.convert("L"), quality=75, progressive=True))
    put(f"{src_name}_restart", enc(src, quality=85, subsampling=2, restart_marker_blocks=3))
    put(f"{src_name}_prog_restart", enc(src, quality=85, subsampling=2, progressive=True, restart_marker_rows=1))
    put(f"{src_name}_rgb", enc(src, quality=90, subsampling=0, keep_rgb=True))

# layouts Pillow cannot write: cjpeg (libjpeg-turbo's own encoder)
if shutil.which("cjpeg"):
    for src_name, src in [("china", crop), ("synth", synth)]:
        ppm = io.BytesIO()
        src.save(ppm, "PPM")
        for sample, tag in [("1x2", "440"), ("4x1", "411"), ("2x2,1x1,1x1", "420c"), ("1x1,1x1,1x1", "444c")]:
            data = subprocess.run(["cjpeg", "-quality", "85", "-sample", sample], input=ppm.getvalue(), capture_output=True, check=True).stdout
            put(f"{src_name}_{tag}", data)
        data = subprocess.run(["cjpeg", "-quality", "85", "-progressive", "-sample", "1x2"], input=ppm.getvalue(), capture_output=True, check=True).stdout
        put(f"{src_name}_440_prog", data)
        # refusals: arithmetic coding
        arith = subprocess.run(["cjpeg", "-arithmetic"], input=ppm.getvalue(), capture_output=True).stdout
        if arith:
            with open(os.path.join(out, f"refuse_{src_name}_arith.jpg"), "wb") as f:
                f.write(arith)

with open(os.path.join(out, "index.json"), "w") as f:
    json.dump(index, f, indent=0, sort_keys=True)
print(len(index), "fixtures")
