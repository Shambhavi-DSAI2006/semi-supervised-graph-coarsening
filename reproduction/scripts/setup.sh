#!/usr/bin/env bash
# Fetch the authors' SGBGC code at the exact commit used for this reproduction,
# and pre-download the datasets. Run from the reproduction/ folder:  bash scripts/setup.sh
set -euo pipefail

GBNC_URL="https://github.com/mxjun17/GBNC"
GBNC_COMMIT="a60c8827baa45ecd3a0b639b3cad0719685013e7"   # 2024-12-13, latest at time of reproduction

if [ ! -d GBNC ]; then
  git clone "$GBNC_URL" GBNC
fi
git -C GBNC checkout -q "$GBNC_COMMIT"
echo "GBNC at $(git -C GBNC rev-parse --short HEAD)"

# Cora and Citeseer download through PyTorch Geometric on first use.
# Co-CS: PyG downloads it from GitHub too; if that download is blocked on your
# network, fetch the file through git instead.
CS_RAW="dataset/cs/CS/raw/ms_academic_cs.npz"
if [ ! -f "$CS_RAW" ]; then
  python -c "from torch_geometric.datasets import Coauthor; Coauthor(root='dataset/cs', name='cs')" 2>/dev/null || {
    echo "PyG download failed; fetching Co-CS via git sparse checkout"
    tmp=$(mktemp -d)
    git clone -q --depth 1 --filter=blob:none --sparse https://github.com/shchur/gnn-benchmark "$tmp/gnn-benchmark"
    git -C "$tmp/gnn-benchmark" sparse-checkout set data/npz
    mkdir -p "$(dirname "$CS_RAW")"
    cp "$tmp/gnn-benchmark/data/npz/ms_academic_cs.npz" "$CS_RAW"
    rm -rf "$tmp"
  }
fi
echo "Setup complete."
