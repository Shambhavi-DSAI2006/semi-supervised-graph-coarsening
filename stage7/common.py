"""Stage 7 shared pieces: datasets, the two trainers, GNNs and peak-memory measurement."""
import os, threading, time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

ROOT = os.environ.get("DATA_ROOT", "dataset")

DATASETS = ["cora", "citeseer", "pubmed", "cs", "photo", "computers", "roman_empire", "amazon_ratings"]
NAME = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "PubMed", "cs": "Coauthor-CS",
        "physics": "Coauthor-Physics", "photo": "Amazon-Photo", "computers": "Amazon-Computers",
        "roman_empire": "Roman-empire", "amazon_ratings": "Amazon-ratings", "arxiv": "ogbn-arxiv"}


def load(name):
    """Graph with x, y, edge_index (undirected); splits are drawn per seed by random_split."""
    from torch_geometric.datasets import Planetoid, Coauthor, Amazon, HeterophilousGraphDataset
    if name in ("cora", "citeseer", "pubmed"):
        d = Planetoid(os.path.join(ROOT), name)[0]
    elif name == "cs":
        d = Coauthor(os.path.join(ROOT, "cs"), "CS")[0]
    elif name == "physics":
        d = Coauthor(os.path.join(ROOT, "physics"), "Physics")[0]
    elif name in ("photo", "computers"):
        d = Amazon(os.path.join(ROOT, name), name.capitalize())[0]
    elif name in ("roman_empire", "amazon_ratings"):
        d = HeterophilousGraphDataset(ROOT, name)[0]
    elif name == "arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.utils import to_undirected
        d = PygNodePropPredDataset("ogbn-arxiv", ROOT)[0]
        d.edge_index = to_undirected(d.edge_index, num_nodes=d.num_nodes)
        d.y = d.y.view(-1)
    else:
        raise ValueError(name)
    return type(d)(x=d.x.float(), y=d.y.long(), edge_index=d.edge_index)


def random_split(data, seed, n_test=1000, n_val=500):
    """Same rule as Stages 4-6: uniform random test/validation nodes per seed."""
    perm = np.random.default_rng(seed).permutation(data.num_nodes)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[torch.as_tensor(perm[:n_test])] = True
    data.val_mask[torch.as_tensor(perm[n_test:n_test + n_val])] = True
    return data


def sample_train(data, k, seed):
    """k training labels per class from nodes outside validation/test (fewer if a class is short)."""
    rng = np.random.default_rng(seed)
    pool = torch.where(~(data.val_mask | data.test_mask))[0].numpy()
    y = data.y.numpy()
    chosen = []
    for c in range(int(y.max()) + 1):
        idx = pool[y[pool] == c]
        chosen.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[torch.as_tensor(np.asarray(chosen, dtype=np.int64))] = True
    return mask


# ---------------------------------------------------------------- models and trainers
class GNN(torch.nn.Module):
    def __init__(self, kind, n_in, n_out, hidden, dropout):
        super().__init__()
        conv = {"gcn": GCNConv, "sage": SAGEConv}[kind]
        self.kind, self.dropout = kind, dropout
        self.c1, self.c2 = conv(n_in, hidden), conv(hidden, n_out)

    def forward(self, x, ei, ew=None):
        kw = {"edge_weight": ew} if self.kind == "gcn" else {}
        x = F.dropout(F.relu(self.c1(x, ei, **kw)), self.dropout, self.training)
        return self.c2(x, ei, **kw)


# S = the trainer of Stages 2-6; C = CGC's own trainer (scr/utils.py model_training), with
# validation checked every 10 epochs instead of every epoch to keep full-graph evaluation affordable.
TRAINERS = {
    "S": dict(hidden=64, dropout=0.5, lr=0.01, wd=5e-4, epochs=200, eval_every=1, lr_drop=False),
    "C": dict(hidden=256, dropout=0.8, lr=0.01, wd=5e-4, epochs=1000, eval_every=10, lr_drop=True),
}


def fit(kind, trainer, x, ei, ew, y, data, seed):
    """Train on the (coarse) graph; select the epoch by validation accuracy on the original graph."""
    h = TRAINERS[trainer]
    torch.manual_seed(seed)
    model = GNN(kind, x.shape[1], int(data.y.max()) + 1, h["hidden"], h["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=h["lr"], weight_decay=h["wd"])
    idx = torch.where(y >= 0)[0]
    best = dict(val=-1.0, test=None, epoch=None)
    for epoch in range(1, h["epochs"] + 1):
        if h["lr_drop"] and epoch == h["epochs"] // 2:
            opt = torch.optim.Adam(model.parameters(), lr=h["lr"] * 0.1, weight_decay=h["wd"])
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x, ei, ew)[idx], y[idx]).backward()
        opt.step()
        if epoch % h["eval_every"]:
            continue
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
        val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if val > best["val"]:
            best = dict(val=val, epoch=epoch,
                        test=(pred[data.test_mask] == data.y[data.test_mask]).float().mean().item())
    return best


def train_seconds_per_epoch(kind, x, ei, ew, y, n_cls, seed, epochs=50):
    """Pure training cost on the (coarse) graph, without the original-graph evaluation."""
    torch.manual_seed(seed)
    model = GNN(kind, x.shape[1], n_cls, 64, 0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    idx = torch.where(y >= 0)[0]
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x, ei, ew)[idx], y[idx]).backward()
        opt.step()
    return (time.perf_counter() - t0) / epochs


class PeakRSS:
    """Peak resident memory above the starting level while the block runs (sampled every 5 ms)."""
    def __enter__(self):
        self.p = psutil.Process()
        self.base = self.peak = self.p.memory_info().rss
        self.stop = False
        self.t = threading.Thread(target=self._watch, daemon=True); self.t.start()
        return self

    def _watch(self):
        while not self.stop:
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.005)

    def __exit__(self, *a):
        self.stop = True; self.t.join()
        self.peak = max(self.peak, self.p.memory_info().rss)
        self.mb = (self.peak - self.base) / 2 ** 20
