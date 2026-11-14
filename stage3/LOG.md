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

## 2026-11-14 — Main run (protocol + Amendment 1)

- 640 runs, started 09:57, finished ~11:50 (Roman-empire+Cora and Amazon-ratings+Citeseer as two processes; Citeseer moved to a third process when the first finished; the Amazon-ratings process was stopped right after its last Amazon run, so nothing was duplicated). 64 cells × 10 seeds, no errors.
- **Completeness check failed in 12 runs** (Citeseer, ratio 0.02): 24 nodes lost because pruning deleted every ball of some small components. Fixed and the 30 affected runs rerun → **Amendment 2** (made after test results existed; decision unchanged, both versions reported).

### Result of the decision rule: **TARGET FOUND — in one cell only (Cora, 2%)**

| Cell | H = best oracle − best label-free | G = best oracle − SGBGC-keep |
|---|---|---|
| Cora 2% | +3.86 ± 2.14 | +2.01 ± 3.51 |
| all other 11 cells | between −1.01 and +1.36 | between −1.00 and +3.16 |

### How strong is this? (my assessment)

- **Weak.** It is 1 of 12 cells. H clears the bar (CI lower bound +1.7), but G's CI (−1.5 to +5.5) includes zero, so there is no reliable evidence that 20 labels per class fall short of the oracle there. The rule only required G's point estimate to reach 2.
- **The heterophilous graphs could not answer the question (floor effect).** With 20 labels per class the full-graph GCN reaches only 27.5% on Roman-empire (majority class 14.0%) and 32.1% on Amazon-ratings — *below* the 36.8% majority-class rate. Coarsened models on Roman-empire sit at 12–15%, i.e. at the majority-class level. Their "no headroom" means everything fails, not that labels cannot help. This limitation was listed in the protocol; it turned out to be severe.
- **Oracles are not better than 20 labels on Citeseer:** SGBGC-keep beats both oracles there (G negative in all three cells).

### Exploratory observation (hypothesis-generating, not part of the rule)

SGBGC-keep with only 20 labels per class versus the best label-free method (paired 95% CI):

| Dataset | 10% | 5% | 2% |
|---|---|---|---|
| Cora | −1.36 ± 1.36 | −2.75 ± 2.51 | +1.85 ± 3.73 |
| Citeseer | +1.07 ± 1.27 | +1.56 ± 1.50 | **+1.92 ± 1.32** |
| Roman-empire | +0.80 ± 1.24 | +1.30 ± 1.12 | +1.28 ± 1.32 |
| Amazon-ratings | −2.56 ± 3.34 | −2.87 ± 3.77 | −1.67 ± 4.12 |

Together with "SGBGC-keep ≥ oracle on Citeseer", this hints that what helps at strong compression is **keeping the few labelled nodes apart** (not merging training nodes of different classes), rather than having many labels. This is a new hypothesis, to be tested with its own protocol.

### Suggested next steps (to discuss with the supervisor)

1. **Confirm the one target cell before building on it:** rerun Cora at 2% (and 1%) with new seeds and other splits; if the headroom does not replicate, treat it as noise.
2. **Test the label-separation hypothesis directly:** add "never merge training nodes of different classes" as a constraint on top of a strong label-free method (Loukas), and compare at strong compression.
3. **Make heterophilous graphs testable** (more labels, or a heterophily-suited GNN), so the floor effect does not hide any effect.
