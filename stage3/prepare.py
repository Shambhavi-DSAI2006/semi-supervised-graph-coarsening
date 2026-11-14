"""Fetch Stage 3 dependencies (cross-platform; run from the stage3/ folder):

    python prepare.py

- Everything Stage 2 needs (SGBGC code, patched Loukas library), via ../stage2/prepare.py
- Roman-empire and Amazon-ratings (Platonov et al., ICLR 2023) from
  yandex-research/heterophilous-graphs at commit a431395, placed where PyTorch
  Geometric's HeterophilousGraphDataset expects them (dataset/<name>/raw/).
"""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = ("https://github.com/yandex-research/heterophilous-graphs", "a431395582e929d88271309716bea4fe24ce6318")
NAMES = ["roman_empire", "amazon_ratings"]


def git(*args, cwd=None):
    subprocess.run(["git", *args], check=True, cwd=cwd)


if __name__ == "__main__":
    subprocess.run([sys.executable, os.path.join(HERE, "..", "stage2", "prepare.py")], check=True)
    missing = [n for n in NAMES if not os.path.isfile(os.path.join(HERE, "dataset", n, "raw", f"{n}.npz"))]
    if missing:
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "hg")
        git("clone", "-q", "--filter=blob:none", "--sparse", REPO[0], src)
        git("-C", src, "checkout", "-q", REPO[1])
        git("-C", src, "sparse-checkout", "set", "data")
        for n in missing:
            dst = os.path.join(HERE, "dataset", n, "raw")
            os.makedirs(dst, exist_ok=True)
            shutil.copy(os.path.join(src, "data", f"{n}.npz"), dst)
            print("placed", n)
        shutil.rmtree(tmp, ignore_errors=True)
    print("Stage 3 data ready.")
