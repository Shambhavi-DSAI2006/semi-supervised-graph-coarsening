"""Run CGC (Gao et al., WWW 2025; official code, github.com/XYGaoG/CGC at commit
08ff04dc0d23110ff53445438f30bcff887e7c31) on our data objects and splits.

The official pipeline in CGC's main.py is reproduced call for call; only the data
loading and the condensation-rate lookup are replaced, so that CGC sees exactly
the training nodes our methods see. CGC_DIR points at the clone (default
../../ext/CGC relative to this file).
"""
import os, sys, time, types

import numpy as np
import torch

CGC_DIR = os.environ.get("CGC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "..", "ext", "CGC"))
sys.path.insert(0, CGC_DIR)
from scr import utils as CU  # noqa: E402

# CGC's own settings for Cora at its published ratio 0.026 (scr/para.py); the default
# configuration in Stage 7 for every dataset (see PROTOCOL.md for the tuning grid).
DEFAULT = dict(dropout=0.8, adj_T=0.8, alpha=7, tau=1, aug_ratio=0.5)


def condense(data, train_mask, rate, hp=None, seed=0):
    """CGC condensed graph with round(rate x training labels) nodes (CGC's own size rule).

    CGC builds every condensed node from training nodes (plus resampled copies of them),
    so rate cannot usefully exceed 1. Returns x, edge_index, edge_weight, y, seconds.
    """
    hp = dict(DEFAULT, **(hp or {}))
    np.random.seed(seed); torch.manual_seed(seed)
    d = types.SimpleNamespace(x=data.x, y=data.y, edge_index=data.edge_index,
                              train_mask=train_mask, num_nodes=data.num_nodes)
    args = types.SimpleNamespace(dataset_name="stage7", ratio=rate, device="cpu", kernel="gcn",
                                 num_class=int(data.y.max()) + 1, **hp)
    t0 = time.time()
    args, y_syn = CU.generate_labels_syn(args, d)
    H = CU.conv_graph_multi(args, d)
    W = CU.linear_model(args, H, d, None)
    H_aug, y_aug, conf = CU.data_assessment(args, d, W, H)
    M = CU.mask_generation_conf(H_aug, y_aug, args, "spectral", conf)
    h = torch.spmm(M, H_aug)
    a = CU.get_adj(h, args.adj_T)
    x = CU.get_feature(a, h, args.alpha)
    secs = time.time() - t0
    ei = a.nonzero().t()
    ew = a[ei[0], ei[1]]
    return x.float(), ei, ew.float(), y_syn, secs
