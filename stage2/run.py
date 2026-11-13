"""Run the Stage 2 label-scarcity test (PROTOCOL.md). From the stage2/ folder:

    python run.py                         # full protocol: 2 datasets x 4 budgets x 10 seeds
    python run.py --datasets cora --seeds 0 --budgets 5pc   # a subset

Appends one JSON line per training run to results/results.jsonl.
Use --smoke for development: seed 99 (outside the protocol's seeds), results
written to results/smoke.jsonl, and test accuracy is NOT printed or stored.
"""
import argparse, json, os, time

import numpy as np
import torch

import data as D
import coarsen as Cz
import train as T

torch.set_num_threads(max(1, os.cpu_count() or 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--budgets", nargs="+", default=D.BUDGETS)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.3, 0.1])
    ap.add_argument("--methods", nargs="+", default=["full"] + Cz.COARSE_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds = [99]

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "smoke.jsonl" if a.smoke else "results.jsonl")
    label_free_cache = {}                      # deterministic label-free partitions

    for ds in a.datasets:
        data = D.load(ds)
        for seed in a.seeds:
            for budget in a.budgets:
                tm = D.sample_train(data, budget, seed)
                jobs = []
                if "full" in a.methods:
                    jobs.append(("full", None))
                jobs += [(m, r) for r in a.ratios for m in a.methods if m != "full"]
                for method, ratio in jobs:
                    t0 = time.time()
                    if method == "full":
                        x, ei, ew, y = T.full_graph(data, tm)
                        stats, info, secs = dict(super_nodes=data.num_nodes, ratio=1.0,
                                                 nodes_kept=data.num_nodes,
                                                 labelled_super_nodes=int(tm.sum())), {}, 0.0
                    else:
                        key = (ds, method, ratio)
                        if method in ("loukas", "gb_struct") and key in label_free_cache:
                            part, secs = label_free_cache[key]
                        else:
                            part = Cz.run_method(method, data, tm, ratio, seed)
                            secs = time.time() - t0
                            if method in ("loukas", "gb_struct"):
                                label_free_cache[key] = (part, secs)
                        x, ei, ew, y, stats = T.coarse_graph(part, data, tm)
                        info = part.info
                    best = T.fit_and_evaluate(x, ei, ew, y, data, seed)
                    rec = dict(dataset=ds, seed=seed, budget=budget, n_train=int(tm.sum()),
                               method=method, ratio_target=ratio, coarsen_seconds=round(secs, 3),
                               val=best["val"], best_epoch=best["epoch"], **stats, **info)
                    if not a.smoke:
                        rec["test"] = best["test"]
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{ds:8s} s{seed} {budget:5s} {method:12s} r={ratio} "
                          f"balls={stats['super_nodes']:5d} kept={stats['nodes_kept']:5d} "
                          f"lab={stats['labelled_super_nodes']:4d} val={best['val']:.3f}"
                          + ("" if a.smoke else f" test={best['test']:.3f}"), flush=True)


if __name__ == "__main__":
    main()
