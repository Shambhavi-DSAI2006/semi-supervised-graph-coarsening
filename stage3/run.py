"""Run the Stage 3 headroom test (PROTOCOL.md). From the stage3/ folder:

    python run.py                                   # full protocol
    python run.py --datasets cora --seeds 0         # a subset
    python run.py --smoke                           # seed 99, test accuracy never computed or stored

Reuses Stage 2's coarsening methods and training code unchanged.
Appends one JSON line per training run to results/results.jsonl (or --out).
"""
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
import data as D          # noqa: E402  (Stage 2 modules)
import coarsen as Cz      # noqa: E402
import train as T         # noqa: E402

PLANETOID = ["cora", "citeseer"]
HETERO = ["roman_empire", "amazon_ratings"]
METHODS = ["full", "loukas", "gb_struct", "sgbgc_keep", "oracle", "oracle_sep"]
DETERMINISTIC = {"loukas", "gb_struct", "oracle"}       # partition does not depend on the seed


def load(name):
    if name in PLANETOID:
        return D.load(name)
    from torch_geometric.datasets import HeterophilousGraphDataset
    d = HeterophilousGraphDataset(root="dataset", name=name)[0]
    d.train_mask = d.train_mask[:, 0].clone()              # split 0 (PROTOCOL.md)
    d.val_mask = d.val_mask[:, 0].clone()
    d.test_mask = d.test_mask[:, 0].clone()
    return d


def test_accuracy_hidden(best, smoke):
    return {} if smoke else {"test": best["test"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=PLANETOID + HETERO)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.05, 0.02])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--budget", default="20pc")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    if a.smoke:
        a.seeds = [99]
    os.makedirs("results", exist_ok=True)
    out = a.out or os.path.join("results", "smoke.jsonl" if a.smoke else "results.jsonl")
    cache = {}

    for ds in a.datasets:
        data = load(ds)
        pool = D.label_pool(data)
        for seed in a.seeds:
            few = D.sample_train(data, a.budget, seed)
            jobs = ([("full", None)] if "full" in a.methods else []) + \
                   [(m, r) for r in a.ratios for m in a.methods if m != "full"]
            for method, ratio in jobs:
                t0 = time.time()
                if method == "full":
                    x, ei, ew, y = T.full_graph(data, few)
                    stats, info, secs = dict(super_nodes=data.num_nodes, ratio=1.0, nodes_kept=data.num_nodes,
                                             labelled_super_nodes=int(few.sum())), {}, 0.0
                else:
                    key = (ds, method, ratio)
                    if method in DETERMINISTIC and key in cache:
                        part, secs = cache[key]
                    else:
                        if method == "oracle":       # coarsen with every label in the pool
                            part = Cz.run_method("sgbgc_keep", data, pool, ratio, seed)
                        elif method == "oracle_sep":  # separate training labels first, then pool labels
                            part = Cz.run_method("sgbgc_keep_sep", data, few, ratio, seed, second_mask=pool)
                        else:
                            part = Cz.run_method(method, data, few, ratio, seed)
                        secs = time.time() - t0
                        if method in DETERMINISTIC:
                            cache[key] = (part, secs)
                    x, ei, ew, y, stats = T.coarse_graph(part, data, few)   # training labels: few only
                    info = part.info
                best = T.fit_and_evaluate(x, ei, ew, y, data, seed)
                rec = dict(dataset=ds, seed=seed, budget=a.budget, n_train=int(few.sum()), method=method,
                           ratio_target=ratio, coarsen_seconds=round(secs, 3), val=best["val"],
                           best_epoch=best["epoch"], **stats, **info, **test_accuracy_hidden(best, a.smoke))
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{ds:14s} s{seed} {method:10s} r={ratio} balls={stats['super_nodes']:6d} "
                      f"kept={stats['nodes_kept']:6d} lab={stats['labelled_super_nodes']:5d} val={best['val']:.3f}"
                      + ("" if a.smoke else f" test={best['test']:.3f}"), flush=True)


if __name__ == "__main__":
    main()
