# Stage 3 research log

Every experiment, including failed or abandoned ones, newest last. Dates are local (UTC+3).

## 2026-11-14 — Setup

- Motivation: Stage 2 ended in "pivot" (labels gave no advantage at 5–20 per class). Before choosing a new direction, measure whether labels could help coarsening at all (headroom), in settings that favour labels.
- Datasets: Roman-empire and Amazon-ratings fetched via git sparse checkout of yandex-research/heterophilous-graphs (commit a431395); loaded with PyTorch Geometric's `HeterophilousGraphDataset`. Both are single connected components. Edge homophily: Roman-empire 0.047, Amazon-ratings 0.380 (Cora 0.810, Citeseer 0.736).
- Feasibility check (timing only, seed 99, no accuracy computed): on both large graphs at ratios 0.1 and 0.02, Loukas, GB-struct and SGBGC-keep (20 per class and oracle) each finish in 2–23 s, keep every node, and reach the same super-node budget (within +2 super-nodes for Loukas).
- Wrote [PROTOCOL.md](PROTOCOL.md) before producing any Stage 3 result.

## 2026-11-14 — Implementation and development checks

- `stage3/run.py` reuses Stage 2's `data.py`, `coarsen.py` and `train.py`; `stage3/prepare.py` fetches the heterophilous datasets.
- **Development run** (Cora and Roman-empire, seed 99, validation accuracy only, test accuracy never computed): all methods ran (5.5 min). It exposed two problems, both fixed by **Amendment 1** before any test-set result:
  - The oracle had fewer super-nodes carrying a training label than SGBGC-keep (Cora, ratio 0.1: 73 vs 101), because it spends splits on pool-label impurities instead of separating training nodes. Added O-sep (training-label purity first, then pool labels).
  - At ratio 0.02 the oracle deleted 19 Cora nodes via the authors' pruning step. Deleted nodes are now reattached to the nearest ball.
- **Regression check:** after these changes to `stage2/coarsen.py`, SGBGC, SGBGC-keep and GB-struct give partitions identical to the committed Stage 2 code in 18 of 18 cases (Cora and Citeseer; 3 budget/seed/ratio combinations each), so Stage 2 results are unaffected.
- Second development run (Cora and Amazon-ratings, seed 99, validation only): every method keeps all nodes; O-sep has as many training-labelled super-nodes as SGBGC-keep (100 vs 101 on Cora).
- Observation from development runs, to be checked in the real run: on Roman-empire every coarsened method's validation accuracy was far below the full graph (0.11–0.19 vs 0.29).
