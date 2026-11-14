"""Coarsening methods M2-M6 (PROTOCOL.md). Each returns a Partition.

A Partition lists, for every super-node, the original node ids it contains.
Nodes in no super-node were discarded by the method. Methods may also return
an explicit weighted coarse adjacency (Loukas); otherwise the harness builds
super-edges from the partition.
"""
import os, sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
GBNC_DIR = os.environ.get("GBNC_DIR", os.path.join(HERE, "..", "reproduction", "GBNC"))
LOUKAS_DIR = os.path.join(HERE, "third_party", "graph-coarsening")


@dataclass
class Partition:
    groups: list                         # list of np.ndarray of original node ids
    edge_mode: str = "binary"            # "binary" | "count" | "explicit"
    adj: Optional[sp.spmatrix] = None    # coarse adjacency when edge_mode == "explicit"
    info: dict = field(default_factory=dict)


def adjacency(edge_index, n):
    ei = edge_index.numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A.setdiag(0); A.eliminate_zeros()
    A.data[:] = 1.0
    return A


# --------------------------------------------------------------------------- SGBGC (M2, M3, M5)

def _import_gbnc():
    if not os.path.isdir(GBNC_DIR):
        sys.exit(f"SGBGC code not found at {GBNC_DIR}. Run: python prepare.py")
    if GBNC_DIR not in sys.path:
        sys.path.insert(0, GBNC_DIR)
    import matplotlib
    matplotlib.use("Agg")
    from tools import get_dataset, del_outlier
    from tools.add_id import add_id
    from tools.corse_split import initial_splite
    from tools.find_major import find_major
    from tools.find_major_num import find_major_num
    from tools.find_max_degree import find_max_degree
    import tools.split_ball_purity as sbp
    return dict(get_dataset=get_dataset.get_dataset, del_outlier=del_outlier.del_outlier, add_id=add_id,
                initial_splite=initial_splite, find_major=find_major, find_major_num=find_major_num,
                find_max_degree=find_max_degree, sbp=sbp)


def _make_split(g, keep):
    """The authors' split_2_co, same assignment rule and row order, with a dict lookup
    instead of their O(n^2) row matching. keep=True keeps a child that has no known
    label (purity 1.0, nothing to disagree) instead of discarding it."""
    import networkx as nx

    def split_2_co(graph, id_dict, Ci, index, total_degree_dict):
        data, degree_dict = Ci[0], Ci[1]
        sub = graph.subgraph([row[0] for row in data])
        d1 = nx.single_source_shortest_path_length(sub, id_dict[index[0]])
        d2 = nx.single_source_shortest_path_length(sub, id_dict[index[1]])
        inf = float("inf")
        b1, b2 = [], []
        for nid in degree_dict:
            (b1 if d1.get(id_dict[nid], inf) <= d2.get(id_dict[nid], inf) else b2).append(int(nid))
        row_of = {int(row[-1]): row for row in data}
        out = []
        for ball in (b1, b2):
            rows = [row_of[i] for i in ball]
            deg = {i: total_degree_dict[i] for i in ball}
            maj = g["find_major"](rows)
            n_maj, n_lab = g["find_major_num"](rows, maj)
            if n_lab == 0:
                out.append([rows, deg, 1.0] if (keep and rows) else [])
            else:
                out.append([rows, deg, float(n_maj / n_lab)])
        return out[0], out[1]
    return split_2_co


def _split_further_guarded(g, graph, id_dict, C, tdd, total):
    """The authors' split_ball_further plus one guard: stop when no ball has two nodes
    left to split (the original raises IndexError there)."""
    while len(C) < total:
        C.sort(key=lambda x: len(x[0]), reverse=True)
        if len(C[0][0]) < 2:
            break
        GB = C.pop(0)
        _, idx = g["find_max_degree"](GB[1])
        for child in g["sbp"].split_2_co(graph, id_dict, GB, idx, tdd):
            if len(child) != 0:
                C.append(child)
    return C


def sgbgc(edge_index, n, y_known, ratio, keep=False, y_second=None):
    """SGBGC partition (authors' gb_division up to purification).

    y_known: int array, class for known training labels, -1 otherwise.
    keep=False -> M2 (as released + crash guard); keep=True -> M3 (SGBGC-keep).
    With all labels -1 and keep=True -> M5 (GB-struct).
    y_second (Stage 3, O-sep): after purity splitting on y_known, relabel every node
    with y_second and continue purity splitting until the budget is reached.
    keep=True also reattaches any node the authors' pruning step deleted to the ball
    of its nearest covered node (Stage 3 Amendment 1; never triggered in Stage 2).
    Node features are not used by SGBGC's partitioning, so a 1-column placeholder
    is passed to keep the per-row arrays small.
    """
    import networkx as nx
    g = _import_gbnc()
    g["sbp"].split_2_co = _make_split(g, keep)       # used by split_ball_purity / further

    feats = np.zeros((n, 1), dtype=np.float32)
    ei = edge_index.numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n), dtype=np.float32)
    graph = g["get_dataset"](feats, A, np.asarray(y_known), 0)
    isolated = [v for v in graph.nodes() if graph.degree(v) == 0]
    graph = g["del_outlier"](graph)

    # --- as in gb_division.py lines 44-73
    attributes = np.array(list(nx.get_node_attributes(graph, "attributes").values()))
    labels = np.array(list(nx.get_node_attributes(graph, "label").values())).reshape(-1, 1)
    deg_old = dict(graph.degree())
    tdd = {i: v for i, v in enumerate(deg_old.values())}
    id_dict = {new: old for new, old in enumerate(deg_old)}
    id_o2n = {old: new for new, old in enumerate(deg_old)}
    indices = np.array(list(deg_old)).reshape(-1, 1)
    data = g["add_id"](np.concatenate((indices, attributes, labels), axis=1))
    C = [[data, tdd]]

    total = int(ratio * n)
    new_C = g["initial_splite"](C, graph, id_dict, id_o2n, labels, tdd)
    if keep:  # components with no labelled node get no centre; keep each as one ball
        covered = {int(r[0]) for b in new_C for r in b[0]}
        for comp in nx.connected_components(graph):
            missing = [v for v in comp if v not in covered]
            if missing:
                new_C.append([[data[id_o2n[v]] for v in missing], {id_o2n[v]: tdd[id_o2n[v]] for v in missing}])

    pruned = False
    target = 1                                     # gb_division.py lines 80-89
    while len(new_C) > total:
        pruned = True
        cut = 0
        new_C.sort(key=lambda x: len(x[0]))
        for i in range(len(new_C)):
            if len(new_C[i][0]) == target + 1:
                cut = i
                break
        target += 1
        new_C = new_C[cut:]

    for ball in new_C:                             # add_purity, with purity 1.0 for label-free balls
        maj = g["find_major"](ball[0])
        n_maj, n_lab = g["find_major_num"](ball[0], maj)
        ball.append(float(n_maj / n_lab) if n_lab else 1.0)

    new_C = g["sbp"].split_ball_purity(graph, id_dict, new_C, tdd, total)
    if y_second is not None:                       # O-sep second phase: continue on y_second
        data[:, -2] = np.asarray(y_second)[data[:, 0].astype(np.int64)]   # ball rows are views of data
        for ball in new_C:
            maj = g["find_major"](ball[0])
            n_maj, n_lab = g["find_major_num"](ball[0], maj)
            ball[-1] = float(n_maj / n_lab) if n_lab else 1.0
        new_C = g["sbp"].split_ball_purity(graph, id_dict, new_C, tdd, total)
    if len(new_C) < total:
        new_C = _split_further_guarded(g, graph, id_dict, new_C, tdd, total)

    groups = [np.array([int(r[0]) for r in b[0]], dtype=np.int64) for b in new_C]
    reattached = 0
    if keep:                                       # reattach nodes deleted by pruning
        owner = {int(v): b for b, gr in enumerate(groups) for v in gr}
        missing = [v for v in graph.nodes() if v not in owner]
        if missing:
            reattached = len(missing)
            frontier = list(owner)
            seen = set(owner)
            while frontier:
                nxt = []
                for u in frontier:
                    for w in graph.neighbors(u):
                        if w not in seen:
                            seen.add(w); owner[w] = owner[u]; nxt.append(w)
                frontier = nxt
            # a component whose balls were all pruned has no covered node to grow from:
            # it becomes one ball, like any other label-free component (Stage 3 Amendment 2)
            next_b = len(groups)
            for comp in nx.connected_components(graph):
                rest = [v for v in comp if v not in owner]
                if rest:
                    for v in rest:
                        owner[v] = next_b
                    next_b += 1
            members = {}
            for v, b in owner.items():
                members.setdefault(b, []).append(v)
            groups = [np.array(sorted(members[b]), dtype=np.int64) for b in sorted(members)]
        groups += [np.array([v], dtype=np.int64) for v in isolated]
    impure = sum(1 for b in new_C if b[2] < 1.0)
    return Partition(groups, "binary", info=dict(pruned=pruned, impure_balls=impure, reattached=reattached))


# --------------------------------------------------------------------------- Loukas (M4)

def loukas(edge_index, n, ratio, K=10, method="variation_neighborhoods"):
    """SCAL-style: coarsen each connected component with > 10 nodes using Loukas's
    variation-neighbourhood method at reduction r = 1 - ratio; smaller components are
    kept uncoarsened. Returns the Loukas coarse adjacency (weighted)."""
    if not os.path.isdir(LOUKAS_DIR):
        sys.exit(f"Loukas library not found at {LOUKAS_DIR}. Run: python prepare.py")
    if LOUKAS_DIR not in sys.path:
        sys.path.insert(0, LOUKAS_DIR)
    import pygsp as gsp
    from graph_coarsening.coarsening_utils import coarsen

    A = adjacency(edge_index, n)
    ncomp, comp = connected_components(A, directed=False)
    groups, blocks = [], []
    for c in range(ncomp):
        idx = np.where(comp == c)[0]
        if len(idx) > 10:
            Cm, Gc, _, _ = coarsen(gsp.graphs.Graph(W=A[idx][:, idx]), K=K, r=1 - ratio, method=method)
            Cm = sp.csc_matrix(Cm)
            owner = np.asarray(Cm.argmax(axis=0)).ravel()   # one non-zero per column
            for s in range(Cm.shape[0]):
                groups.append(idx[owner == s])
            blocks.append(sp.csr_matrix(Gc.W))
        else:
            for v in idx:
                groups.append(np.array([v], dtype=np.int64))
            blocks.append(A[idx][:, idx])
    adj = sp.block_diag(blocks, format="csr")
    adj.setdiag(0); adj.eliminate_zeros()
    return Partition(groups, "explicit", adj=adj)


# --------------------------------------------------------------------------- Feature clustering (M6)

def feature_clusters(edge_index, x, ratio, seed, dims=64):
    """Smooth features twice over the normalised adjacency (with self-loops), reduce to
    `dims` with PCA, run k-means, split each cluster into its connected components.
    k is chosen by binary search so the final number of super-nodes is as close as
    possible to (and not above) the target."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n = x.shape[0]
    A = adjacency(edge_index, n)
    At = A + sp.eye(n)
    dinv = sp.diags(1.0 / np.sqrt(np.asarray(At.sum(1)).ravel()))
    S = dinv @ At @ dinv
    X = x.numpy().astype(np.float64)
    Xs = S @ (S @ X)
    Z = PCA(n_components=min(dims, Xs.shape[1], n - 1), random_state=seed).fit_transform(Xs)

    target = int(ratio * n)
    cache = {}

    def groups_for(k):
        if k not in cache:
            lab = KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(Z)
            r, c = A.nonzero()
            same = lab[r] == lab[c]
            Aw = sp.coo_matrix((np.ones(same.sum()), (r[same], c[same])), shape=(n, n))
            _, comp = connected_components(Aw, directed=False)
            cache[k] = comp
        return cache[k]

    def count(k):
        return int(groups_for(k).max() + 1)

    # The count rises with k but not monotonically (clusters split into components),
    # so search for the crossing and keep the k whose count is closest to the target.
    best_k = 1
    def consider(k):
        nonlocal best_k
        if (abs(count(k) - target), count(k) > target) < (abs(count(best_k) - target), count(best_k) > target):
            best_k = k
    lo, hi = 1, max(1, target)
    while lo <= hi:
        mid = (lo + hi) // 2
        consider(mid)
        if count(mid) < target:
            lo = mid + 1
        elif count(mid) > target:
            hi = mid - 1
        else:
            break
    for k in range(max(1, best_k - 5), min(target, best_k + 5) + 1):   # local scan
        consider(k)
    comp = groups_for(best_k)
    groups = [np.where(comp == b)[0] for b in range(comp.max() + 1)]
    return Partition(groups, "count", info=dict(k=best_k))


# --------------------------------------------------------------------------- registry

LABEL_FREE = ["loukas", "gb_struct", "feat_cluster"]
COARSE_METHODS = ["sgbgc", "sgbgc_keep"] + LABEL_FREE


SMALL = 10   # components with <= SMALL nodes (incl. isolated nodes) become one super-node each


def _on_large_components(fn, data, train_mask, ratio, second_mask=None):
    """Amendment 1 (PROTOCOL.md): every small component becomes one super-node, identically
    for M3-M6; the method runs on the subgraph of large components with the ratio applied
    to that subgraph's nodes."""
    from torch_geometric.utils import subgraph
    n = data.num_nodes
    A = adjacency(data.edge_index, n)
    _, comp = connected_components(A, directed=False)
    sizes = np.bincount(comp)
    is_large = sizes[comp] > SMALL
    large = np.where(is_large)[0]
    small_groups = [np.where(comp == c)[0] for c in np.where(sizes <= SMALL)[0]]

    sub_ei, _ = subgraph(torch.as_tensor(large), data.edge_index, relabel_nodes=True, num_nodes=n)
    y_known = np.where(train_mask.numpy(), data.y.numpy(), -1)[large]
    if second_mask is None:
        part = fn(sub_ei, len(large), data.x[large], y_known, ratio)
    else:
        y_second = np.where(second_mask.numpy(), data.y.numpy(), -1)[large]
        part = fn(sub_ei, len(large), data.x[large], y_known, ratio, y_second)

    part.groups = [large[g] for g in part.groups] + small_groups
    if part.edge_mode == "explicit":                 # small components: no edges to anything
        part.adj = sp.block_diag([part.adj, sp.csr_matrix((len(small_groups), len(small_groups)))], format="csr")
    part.info.update(small_components=len(small_groups), large_nodes=int(len(large)))
    return part


def run_method(name, data, train_mask, ratio, seed, second_mask=None):
    n = data.num_nodes
    if name == "sgbgc":                              # M2: as released, whole graph
        y_known = np.where(train_mask.numpy(), data.y.numpy(), -1)
        return sgbgc(data.edge_index, n, y_known, ratio, keep=False)
    fns = {
        "sgbgc_keep": lambda ei, m, x, yk, r: sgbgc(ei, m, yk, r, keep=True),
        "gb_struct": lambda ei, m, x, yk, r: sgbgc(ei, m, np.full(m, -1), r, keep=True),
        "loukas": lambda ei, m, x, yk, r: loukas(ei, m, r),
        "feat_cluster": lambda ei, m, x, yk, r: feature_clusters(ei, x, r, seed),
        # Stage 3 O-sep: purity on train labels first, then on second_mask's labels
        "sgbgc_keep_sep": lambda ei, m, x, yk, r, y2: sgbgc(ei, m, yk, r, keep=True, y_second=y2),
    }
    if name not in fns:
        raise ValueError(name)
    return _on_large_components(fns[name], data, train_mask, ratio, second_mask)
