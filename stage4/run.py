"""Run the Stage 4 label-separation test (PROTOCOL.md). From the stage4/ folder:

    python run.py                                  # full protocol (seeds 10-19, random splits)
    python run.py --datasets cora --seeds 10       # a subset
    python run.py --smoke                          # seed 99, test accuracy never computed or stored

Reuses Stage 2's coarsening methods and training code unchanged.
"""
import argparse, json, os, sys, time
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
import data as D          # noqa: E402  (Stage 2 modules)
import coarsen as Cz      # noqa: E402
import train as T         # noqa: E402

DATASETS = ["cora", "citeseer", "pubmed"]
METHODS = ["full", "loukas", "loukas_sep", "sgbgc_keep"]


def random_split(data, seed, n_test=1000, n_val=500):
    """New random validation/test split per seed (PROTOCOL.md)."""
    perm = np.random.default_rng(seed).permutation(data.num_nodes)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[torch.as_tensor(perm[:n_test])] = True
    data.val_mask[torch.as_tensor(perm[n_test:n_test + n_val])] = True
    return data


def clash_stats(part, few, y):
    """Super-nodes holding training nodes of > 1 class, and training labels lost to majority voting."""
    clash = lost = 0
    for g in part.groups:
        labs = y[g[few[g]]]
        if len(labs) and len(set(labs.tolist())) > 1:
            clash += 1
            lost += len(labs) - Counter(labs.tolist()).most_common(1)[0][1]
    return clash, lost


def separate(part, A, few, y):
    """Split every super-node with training nodes of > 1 class into one piece per class:
    each member joins the class of its nearest training node (hops inside the super-node;
    ties -> smallest class); members reaching no training node join the class with the most
    training nodes in that super-node (ties -> smallest class)."""
    groups, extra = [], 0
    for g in part.groups:
        tr = g[few[g]]
        classes = sorted(set(y[tr].tolist()))
        if len(classes) <= 1:
            groups.append(g)
            continue
        sub = A[g][:, g]
        pos = {v: i for i, v in enumerate(g)}
        dist = np.vstack([dijkstra(sub, directed=False, unweighted=True, min_only=True,
                                   indices=[pos[v] for v in tr if y[v] == c]) for c in classes])
        choice = np.argmin(dist, axis=0)                       # first minimum = smallest class
        unreachable = np.isinf(dist.min(axis=0))
        if unreachable.any():
            cnt = Counter(y[tr].tolist())
            top = max(cnt.values())
            choice[unreachable] = classes.index(min(c for c, k in cnt.items() if k == top))
        pieces = [g[choice == i] for i in range(len(classes)) if (choice == i).any()]
        groups.extend(pieces)
        extra += len(pieces) - 1
    return Cz.Partition(groups, "count", info=dict(separated_extra=extra))


def loukas_sep(data, few, y, A, plain_count, cache, key):
    """Loukas at a reduced target, then separation, so the final count matches plain Loukas (±2%)."""
    base = cache[key + (None,)]                                 # plain Loukas partition
    large, small = base.info["large_nodes"], base.info["small_components"]
    extra = separate(base, A, few, y).info["separated_extra"]
    best = None
    for attempt in range(3):
        want = plain_count - extra
        ratio = max(1e-6, (want - small) / large)
        ck = key + (round(ratio, 8),)
        if ck not in cache:
            cache[ck] = Cz.run_method("loukas", data, torch.as_tensor(few), ratio, 0)
        part = separate(cache[ck], A, few, y)
        n = len(part.groups)
        if best is None or abs(n - plain_count) < abs(len(best.groups) - plain_count):
            best = part
            best.info.update(attempts=attempt + 1, loukas_ratio=ratio)
        if abs(n - plain_count) <= 0.02 * plain_count:
            break
        extra += n - plain_count
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.05, 0.02, 0.01])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10, 20)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    if a.smoke:
        a.seeds = [99]
    os.makedirs("results", exist_ok=True)
    out = a.out or os.path.join("results", "smoke.jsonl" if a.smoke else "results.jsonl")
    cache = {}                                                   # Loukas partitions (label-independent)

    for ds in a.datasets:
        data = D.load(ds)
        A = Cz.adjacency(data.edge_index, data.num_nodes)
        y = data.y.numpy()
        for seed in a.seeds:
            data = random_split(data, seed)
            few_t = D.sample_train(data, "20pc", seed)
            few = few_t.numpy()
            jobs = ([("full", None)] if "full" in a.methods else []) + \
                   [(m, r) for r in a.ratios for m in a.methods if m != "full"]
            plain_counts = {}
            for method, ratio in jobs:
                t0 = time.time()
                info = {}
                if method == "full":
                    x, ei, ew, yy = T.full_graph(data, few_t)
                    stats = dict(super_nodes=data.num_nodes, ratio=1.0, nodes_kept=data.num_nodes,
                                 labelled_super_nodes=int(few.sum()))
                else:
                    if method in ("loukas", "loukas_sep"):
                        key = (ds, ratio)
                        if key + (None,) not in cache:
                            cache[key + (None,)] = Cz.run_method("loukas", data, few_t, ratio, 0)
                        if method == "loukas":
                            part = cache[key + (None,)]
                            plain_counts[ratio] = len(part.groups)
                        else:
                            pc = plain_counts.get(ratio, len(cache[key + (None,)].groups))
                            part = loukas_sep(data, few, y, A, pc, cache, key)
                    else:
                        part = Cz.run_method(method, data, few_t, ratio, seed)
                    c, l = clash_stats(part, few, y)
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
                print(f"{ds:9s} s{seed} {method:10s} r={ratio} balls={stats['super_nodes']:6d} "
                      f"kept={stats['nodes_kept']:6d} lab={stats['labelled_super_nodes']:4d} "
                      f"lost={info.get('labels_lost', '-')} val={best['val']:.3f}"
                      + ("" if a.smoke else f" test={best['test']:.3f}"), flush=True)


if __name__ == "__main__":
    main()
