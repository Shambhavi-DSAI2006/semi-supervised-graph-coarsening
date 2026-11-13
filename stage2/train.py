"""Coarse-graph construction and the shared GCN training/evaluation (PROTOCOL.md)."""
import copy
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

HP = dict(hidden=64, dropout=0.5, lr=0.01, weight_decay=5e-4, epochs=200)


class GCN(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden=HP["hidden"], dropout=HP["dropout"]):
        super().__init__()
        self.c1, self.c2, self.p = GCNConv(n_in, hidden), GCNConv(hidden, n_out), dropout

    def forward(self, x, ei, ew=None):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.c1(x, ei, ew))
        x = F.dropout(x, self.p, self.training)
        return self.c2(x, ei, ew)


def coarse_graph(part, data, train_mask):
    """Features (member mean), super-edges, training labels and label mask for a Partition."""
    n = data.num_nodes
    B = len(part.groups)
    assign = np.full(n, -1, dtype=np.int64)
    for b, g in enumerate(part.groups):
        assign[g] = b
    kept = assign >= 0

    # features: mean of members
    P = sp.coo_matrix((np.ones(kept.sum()), (assign[kept], np.where(kept)[0])), shape=(B, n)).tocsr()
    P = sp.diags(1.0 / np.asarray(P.sum(1)).ravel()) @ P
    x = torch.tensor(P @ data.x.numpy(), dtype=torch.float)

    # super-edges
    if part.edge_mode == "explicit":
        M = sp.coo_matrix(part.adj)
    else:
        ei = data.edge_index.numpy()
        u, v = assign[ei[0]], assign[ei[1]]
        ok = (u >= 0) & (v >= 0) & (u != v)
        M = sp.coo_matrix((np.ones(ok.sum()), (u[ok], v[ok])), shape=(B, B)).tocsr()   # counts
        if part.edge_mode == "binary":
            M.data[:] = 1.0
        M = M.tocoo()
    keep = M.row != M.col
    edge_index = torch.tensor(np.vstack([M.row[keep], M.col[keep]]), dtype=torch.long)
    edge_weight = torch.tensor(M.data[keep], dtype=torch.float)

    # training labels: majority among members with a training label (ties -> smallest class)
    y = torch.full((B,), -1, dtype=torch.long)
    tm, yy = train_mask.numpy(), data.y.numpy()
    for b, g in enumerate(part.groups):
        labs = yy[g[tm[g]]]
        if len(labs):
            cnt = Counter(labs.tolist())
            top = max(cnt.values())
            y[b] = min(c for c, k in cnt.items() if k == top)
    stats = dict(super_nodes=B, ratio=B / n, nodes_kept=int(kept.sum()),
                 labelled_super_nodes=int((y >= 0).sum()))
    return x, edge_index, edge_weight, y, stats


def fit_and_evaluate(x, edge_index, edge_weight, y, data, seed):
    """Train on (x, edge_index, y); after each epoch, evaluate on the original graph.
    Keep the epoch with the best validation accuracy (earliest on ties)."""
    torch.manual_seed(seed)
    n_cls = int(data.y.max()) + 1
    model = GCN(x.shape[1], n_cls)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    train_idx = torch.where(y >= 0)[0]
    best = dict(val=-1.0, test=None, epoch=None)
    for epoch in range(HP["epochs"]):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x, edge_index, edge_weight)[train_idx], y[train_idx]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
        val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if val > best["val"]:
            test = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            best = dict(val=val, test=test, epoch=epoch)
    return best


def full_graph(data, train_mask):
    """M1: inputs for training directly on the original graph."""
    y = torch.where(train_mask, data.y, torch.full_like(data.y, -1))
    return data.x, data.edge_index, None, y
