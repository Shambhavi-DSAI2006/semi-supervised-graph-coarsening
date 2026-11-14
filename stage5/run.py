"""Run the Stage 5 test of the general label-protection wrapper (PROTOCOL.md). From stage5/:

    python run.py                                  # full protocol (seeds 20-29, random splits)
    python run.py --datasets cora --seeds 20       # a subset
    python run.py --smoke                          # seed 99, test accuracy never computed or stored

Reuses Stage 2's methods and training code and Stage 4's split/separation helpers unchanged.
"""
import argparse, importlib.util, json, os, sys, time

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, HERE)
import data as D          # noqa: E402  (Stage 2)
import coarsen as Cz      # noqa: E402  (Stage 2)
import train as T         # noqa: E402  (Stage 2)
import protect as P       # noqa: E402  (this stage)

_spec = importlib.util.spec_from_file_location("stage4_run", os.path.join(HERE, "..", "stage4", "run.py"))
S4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S4)                         # random_split, clash_stats, loukas_sep

DATASETS = ["cora", "citeseer", "pubmed"]
BASES = ["loukas", "gb_struct", "kmeans"]
METHODS = ["full"] + [m for b in BASES for m in (b, b + "_protect")] + ["loukas_sep"]
EDGE_MODE = {"loukas": "count", "gb_struct": "binary", "kmeans": "count"}   # Loukas weights == crossing counts


def kmeans_partition(data, few_t, ratio, seed):
    """Feature k-means on large components: Ŝ²X -> PCA(64) -> k-means with k = target super-nodes."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    def fn(ei, m, x, yk, r):
        A = Cz.adjacency(ei, m)
        At = A + sp.eye(m)
        dinv = sp.diags(1.0 / np.sqrt(np.asarray(At.sum(1)).ravel()))
        S = dinv @ At @ dinv
        Z = PCA(n_components=min(64, x.shape[1], m - 1), random_state=seed).fit_transform(
            S @ (S @ x.numpy().astype(np.float64)))
        lab = KMeans(n_clusters=max(1, int(r * m)), n_init=1, random_state=seed).fit_predict(Z)
        return Cz.Partition([np.where(lab == b)[0] for b in np.unique(lab)], "count")
    return Cz._on_large_components(fn, data, few_t, ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.05, 0.02, 0.01])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(20, 30)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    if a.smoke:
        a.seeds = [99]
    os.makedirs("results", exist_ok=True)
    out = a.out or os.path.join("results", "smoke.jsonl" if a.smoke else "results.jsonl")
    cache = {}                                          # label-independent base partitions

    for ds in a.datasets:
        data = D.load(ds)
        A = Cz.adjacency(data.edge_index, data.num_nodes)
        y = data.y.numpy()
        for seed in a.seeds:
            data = S4.random_split(data, seed)
            few_t = D.sample_train(data, "20pc", seed)
            few = few_t.numpy()
            jobs = ([("full", None)] if "full" in a.methods else []) + \
                   [(m, r) for r in a.ratios for m in a.methods if m != "full"]
            base_parts = {}
            for method, ratio in jobs:
                t0 = time.time()
                info = {}
                if method == "full":
                    x, ei, ew, yy = T.full_graph(data, few_t)
                    stats = dict(super_nodes=data.num_nodes, ratio=1.0, nodes_kept=data.num_nodes,
                                 labelled_super_nodes=int(few.sum()))
                else:
                    base = method.replace("_protect", "") if method != "loukas_sep" else "loukas"
                    bkey = (base, ratio)
                    if bkey not in base_parts:
                        if base == "loukas":
                            ck = (ds, ratio, None)
                            if ck not in cache:
                                cache[ck] = Cz.run_method("loukas", data, few_t, ratio, 0)
                            base_parts[bkey] = cache[ck]
                        elif base == "gb_struct":
                            ck = (ds, ratio, "gb")
                            if ck not in cache:
                                cache[ck] = Cz.run_method("gb_struct", data, few_t, ratio, 0)
                            base_parts[bkey] = cache[ck]
                        else:
                            base_parts[bkey] = kmeans_partition(data, few_t, ratio, seed)
                    bp = base_parts[bkey]
                    if method == "loukas_sep":
                        part = S4.loukas_sep(data, few, y, A, len(bp.groups), cache, (ds, ratio))
                    elif method.endswith("_protect"):
                        groups, pinfo = P.protect_labels(bp.groups, A, few, y)
                        part = Cz.Partition(groups, EDGE_MODE[base], info=pinfo)
                    else:
                        part = Cz.Partition(bp.groups, EDGE_MODE[base], info=dict(bp.info))
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
                print(f"{ds:9s} s{seed} {method:17s} r={ratio} balls={stats['super_nodes']:6d} "
                      f"kept={stats['nodes_kept']:6d} lost={info.get('labels_lost', '-')} val={best['val']:.3f}"
                      + ("" if a.smoke else f" test={best['test']:.3f}"), flush=True)


if __name__ == "__main__":
    main()
