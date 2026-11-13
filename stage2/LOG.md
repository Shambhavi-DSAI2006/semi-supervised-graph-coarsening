# Stage 2 research log

Every experiment, including failed or abandoned ones, newest last. Dates are local (UTC+3).

## 2026-11-13 — Setup

- Wrote [PROTOCOL.md](PROTOCOL.md) before producing any Stage 2 result.
- Baseline code check:
  - SCAL (Huang et al., KDD 2021): official code available (MIT). It uses Loukas's coarsening library.
  - Loukas graph-coarsening library (commit 1d7aad0): runs on current Python after replacing removed NumPy/SciPy aliases (`sp.newaxis`, `np.int`, `np.bool`, `np.float`). No algorithmic change. On Cora's largest component it reaches the target ratios exactly (0.300 and 0.100) in about 1.6 s.
  - GBGC (Xia et al., 2025): only the appendix is public (code link is an anonymous mirror). The structure-only granular-ball baseline (M5) is therefore our own.
- Earlier informal probe (before this protocol, not used for any decision): with SGBGC on Cora at ratio 0.3, at ≤ 20% known labels the coarse graph contained only the labelled nodes. That probe used a different split and evaluation, so its numbers are not comparable with this stage.

## 2026-11-13 — Implementation and development checks

- Implemented M1–M6 (`coarsen.py`, `train.py`, `run.py`). SGBGC (M2) calls the authors' functions; the only replacement is a faster `split_2_co` with the same assignment rule and row order.
- **Equivalence check:** the faster split gives partitions identical to the authors' original on 6 cases (Cora and Citeseer; budgets pool / 20 per class / 10%; ratios 0.3 and 0.1).
- **Development run** (Cora, seed 99, validation accuracy only; test accuracy not computed): all methods ran end to end (4.5 min). SGBGC kept only the labelled nodes at 5 and 20 per class (35 and 140 nodes in the coarse graph), as expected from Stage 1.
- The same run showed unequal super-node budgets across methods (Loukas over budget because small components were left uncoarsened; feature clustering under budget because of a non-monotonic k search; Citeseer's 430 tiny components make ratio 0.1 infeasible for connectivity-respecting methods). Fixed by **Amendment 1** in PROTOCOL.md before any test-set result; budgets now match within +4.5%.

## 2026-11-13/14 — Main run (protocol + Amendment 1)

- Added `--threads` and `--out` options to `run.py` so Cora and Citeseer could run as parallel single-thread processes (no effect on results). Started 22:48; Cora finished 23:39; Citeseer was split across two processes for the remaining seeds and finished ~00:40.
- **Integrity:** 880 unique runs, all 88 cells with exactly 10 seeds, no errors. One run (Citeseer, seed 6, 5/class, full graph) was computed by both processes and gave identical results, so runs are deterministic.

### Result of the pre-registered decision rule: **PIVOT**

| Dataset | Best label-free | H1 gap (SGBGC-keep − SGBGC) | H2 gap (SGBGC-keep − best label-free) |
|---|---|---|---|
| Cora | Loukas/SCAL | +7.15 (95% CI ±1.52) | +0.04 (±1.25) |
| Citeseer | Loukas/SCAL | +4.20 (±2.59) | +0.71 (±1.79) |

- **H1 supported:** keeping unlabelled nodes recovers most of SGBGC's loss at low label rates.
- **H2 refuted:** once unlabelled nodes are kept, label-aware coarsening is no better than Loukas/SCAL label-free coarsening followed by the same semi-supervised training.

### Exploratory observations (not part of the rule; hypothesis-generating only)

- At ratio 0.3 and low budgets, coarsened training matches the full graph (SGBGC-keep −0.70 ± 0.88 on Cora, +0.26 ± 1.81 on Citeseer; Loukas similar). There is little accuracy left for labels to recover at this compression on these graphs.
- At ratio 0.1, a gap to the full graph opens on Cora (SGBGC-keep −2.83 ± 1.29, Loukas −3.38 ± 1.66). There SGBGC-keep is ahead of Loukas by +0.55 ± 1.98 (Cora) and +1.53 ± 2.39 (Citeseer): not distinguishable from zero.
- Within the same granular-ball algorithm, labels help: SGBGC-keep − GB-struct = +1.53 ± 0.99 (Cora), +1.16 ± 1.90 (Citeseer). A better label-free method (Loukas) closes that gap.
- Variance at 5 per class is large (std 3–8 points) for every method, including the full graph.

### Candidate pivots (to discuss before any new protocol)

1. **Stronger compression** (ratio ≤ 0.05), where coarsening loses accuracy against the full graph and labels have something to recover.
2. **Heterophilous graphs**, where structure-only coarsening is likely to merge nodes of different classes; both datasets here are homophilous.
3. **Large graphs** (e.g. ogbn-arxiv), where coarsening's purpose (training cost) matters and label-free methods such as Loukas become expensive.
4. **Reverse the direction:** use coarsening to *help* semi-supervised learning (e.g. spreading the few labels through super-nodes), rather than using labels to help coarsening.
