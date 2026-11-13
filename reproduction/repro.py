"""
Reproduction of SGBGC (Xia et al., AAAI-25) using the authors' released code
(github.com/mxjun17/GBNC, commit a60c882 of 2024-12-13; fetched by scripts/setup.sh).

The coarsening is the authors' gb_division() called unchanged. The training loop
is copied from the authors' main.py (same GCN class, same early-stopping rule,
same 80/20 random train/val split of super-nodes, same test protocol), with only
the epoch count exposed and per-run seeds added for reproducibility.

Experiments
  table1   : SGBGC at fixed coarsening ratios        (paper Table 1)
  adaptive : SGBGC with ratio set by purity alone     (paper Table 2)
  full     : GCN on the uncoarsened graph             (paper Table 2 "FULL")
  noise    : label noise on training nodes            (paper Fig. 4)

Usage (from the reproduction/ folder, so datasets land in ./dataset):
  python repro.py <experiment> <dataset> [ratio|noise_rate ...] [--epochs N] [--runs N]
  e.g. python repro.py table1 cora 0.5 0.3 0.1 --epochs 60
Writes one JSON line per configuration to results/results.jsonl
"""
import sys, os, time, json, copy, random, argparse, io, contextlib
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("GBNC_DIR", os.path.join(HERE, "GBNC"))
if not os.path.isdir(REPO):
    sys.exit(f"Authors' code not found at {REPO}. Run: bash scripts/setup.sh")
sys.path.insert(0, REPO)
import matplotlib
matplotlib.use = lambda *a, **k: None            # repo forces TkAgg; we run headless
import gb_division as GBD
from split_test import get_planetoid_dataset, get_coauthor_dataset
from tools.split_indices import split_indices
from main import GCN

torch.set_num_threads(max(1, os.cpu_count()))


def load(name):
    if name in ("cs", "physics"):
        return get_coauthor_dataset(name, split="normal")      # 60/20/20 by node index
    return get_planetoid_dataset(name, split="normal")


def add_noise(data, rate, seed):
    """Same rule as the repo's tools/add_noise.py (flip training labels to a random
    other class), but with the rate as a parameter (repo hard-codes 10%) and seeded."""
    rng = np.random.default_rng(seed)
    k = int(data.y.max()) + 1
    y = data.y.clone()
    for i in torch.where(data.train_mask)[0].tolist():
        if rng.random() < rate:
            y[i] = int(rng.choice([c for c in range(k) if c != int(y[i])]))
    out = copy.copy(data); out.y = y
    return out


def coarsen(data, ratio, adaptive=False):
    """Authors' gb_division. Adaptive mode: no ball budget (ratio 1.0) and the
    size-filling step split_ball_further disabled, so splitting stops when every
    ball is pure — the behaviour the paper describes for Table 2."""
    args = argparse.Namespace(noise=0, ball_r=1.0 if adaptive else ratio)
    saved = GBD.split_ball_further
    if adaptive:
        GBD.split_ball_further = lambda graph, id_dict, C, *a, **k: C
    try:
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            nd = GBD.gb_division(copy.deepcopy(data), args)
        secs = time.time() - t
    finally:
        GBD.split_ball_further = saved
    return nd, secs


def make_args(data, n_feat, epochs):
    return argparse.Namespace(num_features=n_feat, hidden=64,
                              num_classes=len(set(np.array(data.y))),
                              epochs=epochs, early_stopping=10, lr=0.01, weight_decay=0.0005)


def train_like_main(x, ei, y, train_idx, val_idx, eval_data, args, seed):
    """Loop copied from main.py lines 108-134."""
    torch.manual_seed(seed)
    model = GCN(args)
    model.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state, hist = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        F.nll_loss(out[train_idx], y[train_idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.nll_loss(model(x, ei)[val_idx], y[val_idx]).item()
        if vl < best_val and epoch > args.epochs // 2:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
        hist.append(vl)
        if args.early_stopping > 0 and epoch > args.epochs // 2:
            if vl > torch.tensor(hist[-(args.early_stopping + 1):-1]).mean().item():
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(eval_data.x, eval_data.edge_index).argmax(1)
    m = eval_data.test_mask
    return (pred[m] == eval_data.y[m]).float().mean().item()


def run_sgbgc(clean, train_data, ratio, adaptive, epochs, runs):
    nd, secs = coarsen(train_data, ratio, adaptive)
    x = torch.tensor(nd["gb_features"], dtype=torch.float)
    ei = torch.tensor(nd["adj"], dtype=torch.long)
    y = torch.tensor(nd["gb_labels"], dtype=torch.long)
    B = len(y)
    args = make_args(clean, x.shape[1], epochs)
    accs = []
    for r in range(runs):
        random.seed(r)
        tr, va = split_indices(list(range(B)), 20, ways="random")
        accs.append(train_like_main(x, ei, y, torch.tensor(tr), torch.tensor(va), clean, args, seed=r))
    return dict(balls=B, ratio_achieved=B / clean.num_nodes, coarsen_seconds=secs, accs=accs)


def run_full(clean, train_data, epochs, runs):
    args = make_args(clean, clean.x.shape[1], epochs)
    tr = torch.where(train_data.train_mask)[0]; va = torch.where(train_data.val_mask)[0]
    accs = [train_like_main(train_data.x, train_data.edge_index, train_data.y, tr, va, clean, args, seed=r)
            for r in range(runs)]
    return dict(accs=accs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["table1", "adaptive", "full", "noise"])
    p.add_argument("dataset")
    p.add_argument("values", nargs="*", type=float)
    p.add_argument("--epochs", type=int, default=200)       # paper: 200 (code default: 60)
    p.add_argument("--runs", type=int, default=20)          # paper: 20
    p.add_argument("--noise_ratio", type=float, default=0.3)
    a = p.parse_args()

    clean = load(a.dataset)
    os.makedirs("results", exist_ok=True)

    def emit(rec):
        rec.update(dataset=a.dataset, epochs=a.epochs, runs=a.runs,
                   mean=100 * float(np.mean(rec["accs"])), std=100 * float(np.std(rec["accs"])))
        with open("results/results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        extra = f" balls={rec['balls']} ratio={rec['ratio_achieved']:.3f} coarsen={rec['coarsen_seconds']:.1f}s" if "balls" in rec else ""
        print(f"{rec['experiment']:9s} {a.dataset:9s} {rec.get('setting','')}: "
              f"{rec['mean']:.2f} ± {rec['std']:.2f}{extra}", flush=True)

    if a.experiment == "table1":
        for r in a.values:
            emit(dict(experiment="table1", setting=f"r={r}", **run_sgbgc(clean, clean, r, False, a.epochs, a.runs)))
    elif a.experiment == "adaptive":
        emit(dict(experiment="adaptive", setting="purity-only", **run_sgbgc(clean, clean, None, True, a.epochs, a.runs)))
    elif a.experiment == "full":
        emit(dict(experiment="full", setting="uncoarsened", **run_full(clean, clean, a.epochs, a.runs)))
    elif a.experiment == "noise":
        for rate in a.values:
            noisy = add_noise(clean, rate, seed=0)
            emit(dict(experiment="noise", setting=f"noise={rate} SGBGC r={a.noise_ratio}",
                      **run_sgbgc(clean, noisy, a.noise_ratio, False, a.epochs, a.runs)))
            emit(dict(experiment="noise", setting=f"noise={rate} FULL",
                      **run_full(clean, noisy, a.epochs, a.runs)))
