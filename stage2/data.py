"""Datasets, fixed validation/test splits and training-label budgets (PROTOCOL.md)."""
import numpy as np
import torch
from torch_geometric.datasets import Planetoid

BUDGETS = ["5pc", "20pc", "10pct", "pool"]       # 5 / 20 per class, 10% of nodes, whole label pool
LOW_BUDGETS = ["5pc", "20pc"]


def load(name, root="dataset"):
    """Planetoid dataset with its public validation (500) and test (1,000) masks."""
    return Planetoid(root=root, name=name)[0]


def label_pool(data):
    return ~(data.val_mask | data.test_mask)


def sample_train(data, budget, seed):
    """Boolean mask of training nodes for a budget; drawn from the label pool only."""
    rng = np.random.default_rng(seed)
    pool = torch.where(label_pool(data))[0].numpy()
    y = data.y.numpy()
    if budget.endswith("pc"):
        k = int(budget[:-2])
        chosen = []
        for c in range(int(y.max()) + 1):
            idx = pool[y[pool] == c]
            chosen.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    elif budget == "10pct":
        chosen = rng.choice(pool, size=round(0.10 * data.num_nodes), replace=False)
    elif budget == "pool":
        chosen = pool
    else:
        raise ValueError(budget)
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[np.asarray(chosen, dtype=np.int64)] = True
    assert not (mask & ~label_pool(data)).any()
    return mask
