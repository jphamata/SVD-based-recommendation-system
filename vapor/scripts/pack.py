#!/usr/bin/env python3
"""Pack vapor into the three archives it ships as, reproducibly.

    python3 scripts/pack.py [OUT_DIR]          # default: ../dist

    vapor_<version>-1-codigo.zip      code, tests, fixtures, docs, proofs, console
    vapor_<version>-2-qualidade.zip   priv/quality: the retained data of the quality suite
    vapor_<version>-3-modelos.zip     the small trained readers and policies (priv/ocr*, priv/lm)
    SHA256SUMS

Every archive holds paths under `vapor/`, so unzipping the three in one place
gives the tree back. The file list is `git ls-files` when run in a checkout
(untracked build output never ships), else a walk that skips `_build`, `deps`
and `.git`. Entries are sorted, timestamps fixed at 2026-01-01 and permissions
normalised (0755 for executables, 0644 otherwise), so the same tree packs to
the same bytes — the archive digests in SHA256SUMS can be compared across
machines.
"""
import hashlib
import os
import re
import stat
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("priv/lm/", "priv/ocr/", "priv/ocr-arabic/", "priv/ocr-cyrillic/",
          "priv/ocr-cjk-ja/", "priv/ocr-cjk-ko/", "priv/ocr-cjk-zh/")
QUALITY = ("priv/quality/",)
SKIP_DIRS = {"_build", "deps", ".git", ".elixir_ls", "node_modules", "__pycache__"}
STAMP = (2026, 1, 1, 0, 0, 0)


def version():
    with open(os.path.join(ROOT, "mix.exs"), encoding="utf-8") as f:
        m = re.search(r'version:\s*"([^"]+)"', f.read())
    if not m:
        sys.exit("pack: no version in mix.exs")
    return m.group(1)


def files():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
        names = [n for n in out.decode("utf-8").split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        names = []
        for d, dirs, fs in os.walk(ROOT):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            names += [os.path.relpath(os.path.join(d, f), ROOT) for f in fs]
    # a tracked file deleted in the working tree is not shipped
    return sorted(n.replace(os.sep, "/") for n in names if os.path.isfile(os.path.join(ROOT, n)))


def part(name):
    if name.startswith(QUALITY):
        return 2
    if name.startswith(MODELS):
        return 3
    return 1


def write(path, names):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in names:
            src = os.path.join(ROOT, n)
            info = zipfile.ZipInfo("vapor/" + n, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if os.stat(src).st_mode & stat.S_IXUSR else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            with open(src, "rb") as f:
                z.writestr(info, f.read())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "dist"))
    os.makedirs(out, exist_ok=True)
    v = version()
    names = files()
    groups = {1: [], 2: [], 3: []}
    for n in names:
        groups[part(n)].append(n)
    labels = {1: "codigo", 2: "qualidade", 3: "modelos"}
    sums = []
    for k in (1, 2, 3):
        if not groups[k]:
            sys.exit(f"pack: part {k} ({labels[k]}) is empty — is this the vapor tree?")
        zp = os.path.join(out, f"vapor_{v}-{k}-{labels[k]}.zip")
        write(zp, groups[k])
        sums.append(f"{sha256(zp)}  {os.path.basename(zp)}")
        print(f"{os.path.basename(zp)}: {len(groups[k])} files, {os.path.getsize(zp) / 1e6:.1f} MB")
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8") as f:
        f.write("\n".join(sums) + "\n")
    print(f"{out}/SHA256SUMS")


if __name__ == "__main__":
    main()
