"""Freeze Tesseract's readings of an OCR evaluation set, for comparison.

usage: ocr_tesseract.py DIR

Writes DIR/tesseract.json: {"tesseract": version, "readings": {file: text}}
— every line image read with --psm 7 (one line), every page*.png with
--psm 3 (a page, one line per line). vapor itself never runs Tesseract (no
external tool is invoked by the product, test/vapor/audit_test.exs):
`mix vapor.quality` and `mix vapor.ocr eval|page --tesseract` read this file.
"""
import json, os, subprocess, sys

d = sys.argv[1]
version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0].strip()
files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
out = {}
for f in files:
    psm = "3" if f.startswith("page") else "7"
    r = subprocess.run(["tesseract", os.path.join(d, f), "-", "--psm", psm, "-l", "eng"], capture_output=True, text=True)
    lines = [l.strip() for l in r.stdout.split("\n") if l.strip()]
    out[f] = (" " if psm == "7" else "\n").join(lines)
json.dump({"tesseract": version, "readings": out}, open(os.path.join(d, "tesseract.json"), "w"), ensure_ascii=False, indent=1)
print(f"{len(out)} readings by {version}")
