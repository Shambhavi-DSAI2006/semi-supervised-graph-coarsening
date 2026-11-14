"""Run the Stage 6 label-guided k-means test (PROTOCOL.md). From the stage6/ folder:

    python run.py                                  # full protocol (seeds 30-39, random splits)
    python run.py --datasets cora --seeds 30       # a subset
    python run.py --smoke                          # seed 99, test accuracy never computed or stored

Reuses Stage 2's training code and Stage 4's split and clash helpers unchanged.
"""
import argparse, importlib.util, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, HERE)
import data as D          # noqa: E402  (Stage 2)
import coarsen as Cz      # noqa: E402  (Stage 2)
import train as T         # noqa: E402  (Stage 2)
import guided as G        # noqa: E402  (this stage)

_spec = importlib.util.spec_from_file_location("stage4_run", os.path.join(HERE, "..", "stage4", "run.py"))
S4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S4)                         # random_split, clash_stats

DATASETS = ["cora", "citeseer", "pubmed"]
VARIANTS = ["K0", "K1", "K2", "K3"]
METHODS = ["full"] + VARIANTS


def coarsen(variant, data, few_t, ratio, seed):
    n_classes = int(data.y.max()) + 1

    def fn(ei, m, x, yk, r):
        groups, info = G.partition(variant, Cz.adjacency(ei, m), x.numpy(), yk, n_classes, r, seed)
        return Cz.Partition(groups, "count", info=info)
    return Cz._on_large_components(fn, data, few_t, ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.02, 0.01, 0.005])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(30, 40)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    if a.smoke:
        a.seeds = [99]
    os.makedirs("results", exist_ok=True)
    out = a.out or os.path.join("results", "smoke.jsonl" if a.smoke else "results.jsonl")

    for ds in a.datasets:
        data = D.load(ds)
        y = data.y.numpy()
        for seed in a.seeds:
            data = S4.random_split(data, seed)
            few_t = D.sample_train(data, "20pc", seed)
            few = few_t.numpy()
            jobs = ([("full", None)] if "full" in a.methods else []) + \
                   [(m, r) for r in a.ratios for m in a.methods if m != "full"]
            for method, ratio in jobs:
                t0 = time.time()
                info = {}
                if method == "full":
                    x, ei, ew, yy = T.full_graph(data, few_t)
                    stats = dict(super_nodes=data.num_nodes, ratio=1.0, nodes_kept=data.num_nodes,
                                 labelled_super_nodes=int(few.sum()))
                else:
                    part = coarsen(method, data, few_t, ratio, seed)
                    c, l = S4.clash_stats(part, few, y)
                    info = dict(part.info, clashing_super_nodes=c, labels_lost=l)
                    x, ei, ew, yy, stats = T.coarse_graph(part, data, few_t)
                secs = time.time() - t0
                best = T.fit_and_evaluate(x, ei, ew, yy, data, seed)
                rec = dict(dataset=ds, seed=seed, n_train=int(few.sum()), method=method, ratio_target=ratio,
                           coarsen_seconds=round(secs, 3), val=best["val"], best_epoch=best["epoch"], **stats, **info)
                if not a.smoke:
                    rec["test"] = best["test"]
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{ds:9s} s{seed} {method:4s} r={ratio} balls={stats['super_nodes']:6d} "
                      f"kept={stats['nodes_kept']:6d} lost={info.get('labels_lost', '-')} val={best['val']:.3f}"
                      + ("" if a.smoke else f" test={best['test']:.3f}"), flush=True)


if __name__ == "__main__":
    main()
