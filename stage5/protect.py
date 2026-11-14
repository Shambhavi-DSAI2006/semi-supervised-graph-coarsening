"""Label protection: a general wrapper around ANY coarsening method.

Given a partition of the nodes (from any coarsening algorithm), the few training
labels, and the graph, `protect_labels` returns a partition with

  * no super-node containing training nodes of more than one class, and
  * exactly the same number of super-nodes as the input partition,

using only the partition, the adjacency and the labels - nothing about how the
partition was produced. Two steps:

1. Separate: every super-node holding training nodes of k > 1 classes is split into
   k pieces; each member joins the class of its nearest training node (fewest hops
   inside the super-node; ties -> smallest class index); members that reach no
   training node join the class with most training nodes there (ties -> smallest).
2. Rebalance: to undo the extra pieces without coarsening the rest of the graph
   harder, repeatedly merge the pair of *adjacent*, *label-compatible* super-nodes
   (together they hold training nodes of at most one class) with the highest
   connection score  w_ij / (n_i + n_j)  (w_ij = original edges between them,
   n = sizes), until the original count is restored. If no adjacent compatible pair
   exists, the compatible pair with the smallest combined size is merged instead.
"""
from collections import Counter

import numpy as np
from scipy.sparse.csgraph import dijkstra


def separate(groups, A, few, y):
    out, extra = [], 0
    for g in groups:
        tr = g[few[g]]
        classes = sorted(set(y[tr].tolist()))
        if len(classes) <= 1:
            out.append(g)
            continue
        sub = A[g][:, g]
        pos = {v: i for i, v in enumerate(g)}
        dist = np.vstack([dijkstra(sub, directed=False, unweighted=True, min_only=True,
                                   indices=[pos[v] for v in tr if y[v] == c]) for c in classes])
        choice = np.argmin(dist, axis=0)
        unreachable = np.isinf(dist.min(axis=0))
        if unreachable.any():
            cnt = Counter(y[tr].tolist())
            top = max(cnt.values())
            choice[unreachable] = classes.index(min(c for c, k in cnt.items() if k == top))
        pieces = [g[choice == i] for i in range(len(classes)) if (choice == i).any()]
        out.extend(pieces)
        extra += len(pieces) - 1
    return out, extra


def rebalance(groups, A, few, y, target):
    """Merge adjacent label-compatible super-nodes until len(groups) == target."""
    groups = [np.asarray(g) for g in groups]
    n = A.shape[0]
    owner = np.full(n, -1, dtype=np.int64)
    for b, g in enumerate(groups):
        owner[g] = b
    size = {b: len(g) for b, g in enumerate(groups)}
    cls = {b: set(y[g[few[g]]].tolist()) for b, g in enumerate(groups)}
    members = {b: list(g) for b, g in enumerate(groups)}
    # crossing-edge counts between super-nodes (dict of dicts, symmetric)
    W = {b: Counter() for b in members}
    r, c = A.nonzero()
    for u, v in zip(owner[r], owner[c]):
        if u >= 0 and v >= 0 and u != v:
            W[u][v] += 1
    fallbacks = 0
    while len(members) > target:
        best, best_s = None, -1.0
        for i in members:
            for j, w in W[i].items():
                if i < j and len(cls[i] | cls[j]) <= 1:
                    s = w / (size[i] + size[j])
                    if s > best_s:
                        best, best_s = (i, j), s
        if best is None:                                  # no adjacent compatible pair
            fallbacks += 1
            keys = sorted(members, key=lambda b: size[b])
            best = next(((i, j) for a, i in enumerate(keys) for j in keys[a + 1:]
                         if len(cls[i] | cls[j]) <= 1), None)
            if best is None:
                break
        i, j = best                                       # merge j into i
        members[i].extend(members.pop(j))
        size[i] += size.pop(j)
        cls[i] |= cls.pop(j)
        for k, w in W.pop(j).items():
            if k == i:
                continue
            W[k].pop(j, None)
            W[i][k] += w
            W[k][i] += w
        W[i].pop(j, None)
    return [np.array(sorted(m), dtype=np.int64) for m in members.values()], fallbacks


def protect_labels(groups, A, few, y):
    """Return (new_groups, info): label-clash-free partition with the same number of super-nodes."""
    target = len(groups)
    sep, extra = separate(groups, A, few, y)
    out, fallbacks = rebalance(sep, A, few, y, target)
    return out, dict(separated_extra=extra, rebalance_fallbacks=fallbacks)
