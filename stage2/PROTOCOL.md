# Stage 2 protocol — Does label-aware coarsening still help when labels are scarce?

*Written and committed before any Stage 2 result was produced. Changes after this commit are listed under "Amendments" with a reason; the original text stays.*

## Question

SGBGC uses node labels to decide which nodes to merge. Stage 1 showed it is evaluated with labels on 80% of nodes, and that it discards graph regions with no known label. This stage tests, cheaply, whether **label-aware coarsening is worth pursuing in the semi-supervised setting** before any new method is designed.

## Hypotheses

- **H1 — root cause.** SGBGC's accuracy loss at low label rates is mainly caused by *discarding unlabelled nodes*, not by the scarcity of labels itself.
  *Refuted if* a minimal fix that keeps unlabelled nodes (SGBGC-keep) improves accuracy over SGBGC by **≤ 1 point** on average at the low-label budgets.
- **H2 — is the problem real?** At low label rates, label-aware coarsening beats the best *label-free* coarsening followed by the same semi-supervised training.
  *Refuted if* SGBGC-keep is **≤ 1 point** better than the best label-free method on average at the low-label budgets.

## Data and splits

| Dataset | Nodes | Classes | Validation | Test | Label pool (all other nodes) |
|---|---|---|---|---|---|
| Cora | 2,708 | 7 | 500 | 1,000 | 1,208 |
| Citeseer | 3,327 | 6 | 500 | 1,000 | 1,827 |

- Validation and test sets are the standard Planetoid public splits (PyTorch Geometric), fixed for every run.
- Training labels are drawn from the label pool, never from validation or test.
- **The test set is used only to report final numbers.** No design decision, hyperparameter or model selection uses it.
- **Validation labels are used only for model selection** (choosing the training epoch), evaluated on the original graph. They are never passed to any coarsening method.

## Label budgets

| Budget | Cora | Citeseer | Group |
|---|---|---|---|
| 5 per class | 35 | 30 | **low** |
| 20 per class | 140 | 120 | **low** |
| 10% of nodes | 270 | 332 | medium |
| Whole label pool | 1,208 (44.6%) | 1,827 (54.9%) | rich (reference) |

Per-class budgets are stratified; percentage budgets are drawn uniformly from the pool. A new sample is drawn for each seed.

## Coarsening ratios

Keep **30%** and **10%** of the nodes. Methods that cannot hit the ratio exactly report the ratio they reach.

## Methods

All coarsening methods produce a partition of the nodes into super-nodes. Only training labels may be passed to a coarsening method.

| # | Method | Uses labels to coarsen? | Super-edges | Source |
|---|---|---|---|---|
| M1 | **GCN on the full graph** (no coarsening) | — | — | reference |
| M2 | **SGBGC** as released, with one guard so it stops instead of crashing when it cannot reach the ball budget | yes | binary | authors' code, commit a60c882 |
| M3 | **SGBGC-keep**: SGBGC with unlabelled nodes kept — label-free pieces created during splitting are kept as balls instead of discarded; connected components with no labelled node, and isolated nodes, become balls of their own | yes | binary | authors' code + minimal patch (ours) |
| M4 | **Loukas variation-neighbourhood coarsening** (as used by SCAL, KDD 2021); components with ≤ 10 nodes are left uncoarsened, as in SCAL | no | weighted (Loukas) | Loukas library, commit 1d7aad0, patched only for NumPy/SciPy compatibility |
| M5 | **GB-struct**: SGBGC-keep's splitting machinery with **no labels** — every ball is split by its two highest-degree nodes, largest ball first, until the budget is reached | no | binary | ours |
| M6 | **Feature clustering**: features smoothed over the graph (Ŝ²X, Ŝ = normalised adjacency with self-loops), reduced to 64 dimensions with PCA, k-means, then each cluster split into its connected components; k is tuned so the final count matches the target ratio (±1%) | no | weighted (count of crossing edges) | ours |

M3 vs M5 is a clean ablation: the same algorithm with and without labels. M4 and M6 are the strongest label-free baselines we can run.

## Building the coarse graph and training (identical for M2–M6)

- **Super-node features:** mean of member features.
- **Super-node training label:** majority label among members that carry a training label (ties: smallest class index). Super-nodes with no training label are kept in the graph but excluded from the loss.
- **Model:** 2-layer GCN, 64 hidden units, dropout 0.5, Adam, learning rate 0.01, weight decay 5e-4, 200 epochs. Edge weights are used where the method provides them.
- **Model selection:** after every epoch, the model is applied to the **original graph** and accuracy is measured on the validation set; the epoch with the best validation accuracy is kept (earliest on ties).
- **Evaluation:** that model's accuracy on the test set of the original graph.
- M1 uses the same model, training and selection, trained directly on the original graph.

These hyperparameters are fixed in advance and are not tuned for any method.

## Seeds and runs

10 seeds (0–9). Seed *s* sets the training-label sample, k-means initialisation (M6) and GCN initialisation.
Total: 2 datasets × 4 budgets × (1 + 5 methods × 2 ratios) × 10 seeds = **880 training runs**.

## What is recorded per run

Test accuracy, validation accuracy, number of super-nodes, achieved ratio, number of original nodes inside the coarse graph, number of super-nodes with a training label, coarsening time.

## Decision rule

For each dataset, compute each gap as the mean test accuracy over the **low-label budgets (5 and 20 per class) and both ratios** (4 cells × 10 seeds):

- **H1 gap** = SGBGC-keep − SGBGC
- **H2 gap** = SGBGC-keep − best label-free method, where "best" is the label-free method (M4, M5 or M6) with the highest mean on that dataset

| Outcome | Condition |
|---|---|
| **Go** — design a semi-supervised label-aware method | H2 gap ≥ 2 points on **both** datasets |
| **Pivot** — labels add little to coarsening at low rates; rethink the project angle | H2 gap ≤ 1 point on **both** datasets |
| **Inconclusive** — extend to Co-CS before deciding | anything else |

H1 is reported as supported (gap ≥ 2), refuted (gap ≤ 1) or unclear, but does not by itself decide go/pivot.

## Known limitations of this test

- Two small citation graphs only; both are homophilous.
- M5 and M6 are our own implementations (GBGC's official code is not public; its appendix is).
- One fixed set of GCN hyperparameters for all methods.
- Ten seeds give a standard error of roughly 0.3–0.6 points per cell, so differences under 1 point are treated as no difference.

## Amendments

*(none yet)*
