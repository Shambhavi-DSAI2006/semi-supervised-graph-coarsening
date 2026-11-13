"""Fetch Stage 2 dependencies (cross-platform; run from the stage2/ folder):

    python prepare.py

- The authors' SGBGC code (mxjun17/GBNC, commit a60c882) into ../reproduction/GBNC, if missing.
- Loukas's graph-coarsening library (commit 1d7aad0) into third_party/, with the
  compatibility fixes listed in PATCHES (removed NumPy/SciPy aliases only; no
  algorithmic change).
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GBNC_DIR = os.path.join(HERE, "..", "reproduction", "GBNC")
GBNC = ("https://github.com/mxjun17/GBNC", "a60c8827baa45ecd3a0b639b3cad0719685013e7")
LOUKAS_DIR = os.path.join(HERE, "third_party", "graph-coarsening")
LOUKAS = ("https://github.com/loukasa/graph-coarsening", "1d7aad02ef294a37e79e009d52f215b70d3ef1e8")

PATCHES = [  # (regex, replacement): aliases removed in recent NumPy / SciPy
    (r"sp\.newaxis", "np.newaxis"),
    (r"np\.int\b", "int"),
    (r"np\.bool\b", "bool"),
    (r"np\.float\b", "float"),
]


def git(*args):
    subprocess.run(["git", *args], check=True)


def fetch(url_commit, path):
    url, commit = url_commit
    if not os.path.isdir(path):
        git("clone", "-q", url, path)
    git("-C", path, "checkout", "-q", commit)
    git("-C", path, "checkout", "-q", "--", ".")          # undo earlier patches before re-applying


def patch_loukas():
    pkg = os.path.join(LOUKAS_DIR, "graph_coarsening")
    for name in os.listdir(pkg):
        if name.endswith(".py"):
            p = os.path.join(pkg, name)
            src = open(p, encoding="utf-8").read()
            new = src
            for pat, rep in PATCHES:
                new = re.sub(pat, rep, new)
            if new != src:
                open(p, "w", encoding="utf-8").write(new)
                print(f"  patched {name}")


if __name__ == "__main__":
    fetch(GBNC, GBNC_DIR)
    print("GBNC ready at", os.path.normpath(GBNC_DIR))
    os.makedirs(os.path.dirname(LOUKAS_DIR), exist_ok=True)
    fetch(LOUKAS, LOUKAS_DIR)
    patch_loukas()
    print("Loukas library ready at", os.path.normpath(LOUKAS_DIR))
    sys.exit(0)
