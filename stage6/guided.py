"""Label-guided k-means coarsening (Stage 6).

Everything works on the subgraph of large components (small components become one
super-node each, as in Stages 2-5) and uses only training labels.

- embed(): node embedding Z = PCA_64( Ŝ² X ), Ŝ = normalised adjacency with self-loops.
- propagate(): label propagation (Zhou et al., 2004): F <- a·Ŝ·F + (1-a)·Y0, 10 steps,
  a = 0.9, Y0 = one-hot training labels; rows normalised to sum 1 (rows that received
  nothing become uniform).
- K0  k-means on Z.
- K1  k-means on Z, then protection in feature space (protect_features).
- K2  k-means on [Z, P] with both blocks scaled to mean row norm 1 (label-propagation features).
- K3  K2, then protection in feature space on [Z, P].

protect_features(): (1) separate - in a cluster holding training nodes of > 1 class,
each member joins the class of its nearest training node in feature space (Euclidean;
ties -> smallest class); (2) rebalance - repeatedly merge the label-compatible pair of
clusters with the smallest Ward cost  n_i n_j / (n_i + n_j) * ||c_i - c_j||^2  until the
original number of clusters is restored.
"""
import numpy as np
import scipy.sparse as sp

ALPHA, STEPS, DIMS = 0.9, 10, 64


def norm_adj(A):
    n = A.shape[0]
    At = A + sp.eye(n)
    d = sp.diags(1.0 / np.sqrt(np.asarray(At.sum(1)).ravel()))
    return d @ At @ d


def embed(A, x, seed):
    from sklearn.decomposition import PCA
    S = norm_adj(A)
    X = x.astype(np.float64)
    return PCA(n_components=min(DIMS, X.shape[1], A.shape[0] - 1), random_state=seed).fit_transform(S @ (S @ X))


def propagate(A, yk, n_classes):
    S = norm_adj(A)
    Y0 = np.zeros((A.shape[0], n_classes))
    idx = np.where(yk >= 0)[0]
    Y0[idx, yk[idx]] = 1.0
    F = Y0.copy()
    for _ in range(STEPS):
        F = ALPHA * (S @ F) + (1 - ALPHA) * Y0
    s = F.sum(1, keepdims=True)
    return np.where(s > 0, F / np.where(s > 0, s, 1), 1.0 / n_classes)


def unit_rows(M):
    return M / max(np.linalg.norm(M, axis=1).mean(), 1e-12)


def kmeans(Z, k, seed):
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=max(1, k), n_init=1, random_state=seed).fit_predict(Z)
    return [np.where(lab == b)[0] for b in np.unique(lab)]


def protect_features(groups, Z, yk):
    target = len(groups)
    # 1) separate in feature space
    sep = []
    for g in groups:
        tr = g[yk[g] >= 0]
        classes = sorted(set(yk[tr].tolist()))
        if len(classes) <= 1:
            sep.append(g)
            continue
        d = np.stack([np.min(np.linalg.norm(Z[g][:, None, :] - Z[tr[yk[tr] == c]][None, :, :], axis=2), axis=1)
                      for c in classes])
        choice = np.argmin(d, axis=0)
        sep.extend(g[choice == i] for i in range(len(classes)) if (choice == i).any())
    extra = len(sep) - target
    # 2) rebalance with Ward cost among label-compatible pairs
    members = {i: list(g) for i, g in enumerate(sep)}
    size = {i: len(g) for i, g in enumerate(sep)}
    cen = {i: Z[g].mean(0) for i, g in enumerate(sep)}
    cls = {i: set(yk[g][yk[g] >= 0].tolist()) for i, g in enumerate(sep)}
    while len(members) > target:
        keys = list(members)
        C = np.stack([cen[i] for i in keys]); nz = np.array([size[i] for i in keys], dtype=float)
        D = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1) * (nz[:, None] * nz[None, :]) / (nz[:, None] + nz[None, :])
        np.fill_diagonal(D, np.inf)
        for a, i in enumerate(keys):                     # forbid incompatible pairs
            for b, j in enumerate(keys):
                if b > a and len(cls[i] | cls[j]) > 1:
                    D[a, b] = D[b, a] = np.inf
        a, b = np.unravel_index(np.argmin(D), D.shape)
        if not np.isfinite(D[a, b]):
            break
        i, j = keys[a], keys[b]
        cen[i] = (cen[i] * size[i] + cen[j] * size[j]) / (size[i] + size[j])
        members[i].extend(members.pop(j)); size[i] += size.pop(j); cls[i] |= cls.pop(j); cen.pop(j)
    return [np.array(sorted(m), dtype=np.int64) for m in members.values()], max(extra, 0)


def partition(variant, A, x, yk, n_classes, ratio, seed):
    """Groups (local ids of the large-component subgraph) for K0-K3, and extra info."""
    k = max(1, int(ratio * A.shape[0]))
    Z = embed(A, x, seed)
    if variant in ("K2", "K3"):
        Z = np.hstack([unit_rows(Z), unit_rows(propagate(A, yk, n_classes))])
    groups = kmeans(Z, k, seed)
    info = {}
    if variant in ("K1", "K3"):
        groups, extra = protect_features(groups, Z, yk)
        info["separated_extra"] = extra
    return groups, info
