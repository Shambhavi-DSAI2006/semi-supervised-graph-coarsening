# Stage 3 protocol — Is there headroom for labels in coarsening?

*Written and committed before any Stage 3 result was produced. Later changes go under "Amendments" with a reason; the original text stays.*

## Question

Stage 2 found that with few labels (5–20 per class), label-aware coarsening (SGBGC-keep) is no better than label-free coarsening (Loukas/SCAL). That has two explanations:

- **(a)** labels cannot improve coarsening in these settings, however many there are; or
- **(b)** labels could improve it, but a few labels are too little to exploit.

Only (b) leaves a research problem for semi-supervised coarsening. This stage measures the **headroom**: how much accuracy coarsening gains when it may use many labels, in settings chosen to give labels the best chance — strong compression and heterophilous graphs.

## Definitions

- **Oracle coarsening (O):** SGBGC-keep run with the labels of **every node in the label pool** (all nodes outside the validation and test sets). Validation and test labels are never used for coarsening. The GCN is still **trained with only 20 labels per class**, exactly like the other methods. O is a diagnostic upper bound, not a method.
- **Best label-free (LF):** in each (dataset, ratio) cell, whichever of Loukas/SCAL and GB-struct has the higher mean test accuracy.
- **Headroom H** = O − LF (paired per seed, then averaged).
- **Few-label gap G** = O − SGBGC-keep with 20 labels per class.

H says whether labels *can* help this family of coarsening; G says how much of that a few labels currently miss.

## Data and splits

| Dataset | Nodes | Classes | Edge homophily | Validation | Test | Label pool | Split |
|---|---|---|---|---|---|---|---|
| Cora | 2,708 | 7 | 0.81 | 500 | 1,000 | 1,208 | Planetoid public |
| Citeseer | 3,327 | 6 | 0.74 | 500 | 1,000 | 1,827 | Planetoid public |
| Roman-empire | 22,662 | 18 | 0.05 | 5,665 | 5,666 | 11,331 | split 0 of the 10 provided |
| Amazon-ratings | 24,492 | 5 | 0.38 | 6,123 | 6,123 | 12,246 | split 0 of the 10 provided |

Edge homophily = share of edges joining nodes of the same class. Roman-empire and Amazon-ratings are from Platonov et al. (ICLR 2023), via PyTorch Geometric's `HeterophilousGraphDataset`, data from github.com/yandex-research/heterophilous-graphs at commit a431395.

- **The test set is used only to report final numbers.** No decision, setting or model selection uses it.
- **Validation labels are used only for model selection**, on the original graph. They are never passed to any coarsening method.

## Settings

- **Training labels:** 20 per class, stratified, drawn from the label pool; a new sample per seed (Cora 140, Citeseer 120, Roman-empire 360, Amazon-ratings 100).
- **Coarsening ratios:** keep **10%, 5% and 2%** of the nodes.
- **Small components:** as in Stage 2 Amendment 1, every connected component with ≤ 10 nodes becomes one super-node and the ratio applies to the rest (Roman-empire and Amazon-ratings are single components, so this affects only Cora and Citeseer).
- **Seeds:** 10 (0–9). Seed *s* sets the training-label sample and GCN initialisation.

## Methods

| # | Method | Labels used for coarsening | Labels used for training |
|---|---|---|---|
| M1 | GCN on the full graph | — | 20 / class |
| M4 | Loukas/SCAL (as in Stage 2) | none | 20 / class |
| M5 | GB-struct (as in Stage 2) | none | 20 / class |
| M3 | SGBGC-keep (as in Stage 2) | the 20 / class training labels | 20 / class |
| O | SGBGC-keep, oracle | whole label pool | 20 / class |

Feature clustering (Stage 2's M6) is left out: its k search is slow on the large graphs, and it was never the best label-free method in Stage 2.

Coarse-graph construction, the GCN (2 layers, 64 hidden, dropout 0.5, Adam, learning rate 0.01, weight decay 5e-4, 200 epochs), model selection (best validation accuracy on the original graph, earliest on ties) and evaluation are **identical to Stage 2** and reuse its code unchanged.

Total: 4 datasets × 10 seeds × (1 + 4 methods × 3 ratios) = **520 training runs**.

## Decision rule

For each of the 12 (dataset, ratio) cells, compute H and G with 95% confidence intervals from paired per-seed differences.

| Outcome | Condition | Next step |
|---|---|---|
| **Target found** | at least one cell with **H ≥ 3 points** (CI lower bound > 0) **and G ≥ 2 points** | Stage 4 designs a semi-supervised method to close G in those cells |
| **No headroom** | **H ≤ 1 point in every cell** | Label-guided granular-ball coarsening cannot beat label-free coarsening here; move to the reverse direction (coarsening to help semi-supervised learning) |
| **Inconclusive** | anything else | Report, then decide with the supervisor |

## Known limitations

- The oracle measures headroom for **granular-ball-style use of labels** (SGBGC's purity splitting). A "no headroom" result rules out that family, not every conceivable way of using labels to coarsen.
- At a fixed node budget, purity splitting stops when the budget is reached, so even oracle balls may stay mixed, especially on heterophilous graphs.
- One GCN configuration for all datasets; GCN is known to be weak on heterophilous graphs, and heterophily-specific GNNs could change absolute numbers.
- One split per heterophilous dataset (split 0) and 10 seeds.

## Amendments

### Amendment 1 — a second oracle and a node-keeping fix (2026-11-14, before any test-set result)

**Reason.** A development run (seed 99; validation accuracy only; test accuracy never computed or stored) showed two problems:

1. **The oracle as defined is not an upper bound.** With 20 labels per class, SGBGC-keep spends its splits separating training nodes of different classes, so each training label stays a separate training example on the coarse graph. The oracle instead spends its splits on impurities among the ~1,000–12,000 pool labels, so training nodes get merged: on Cora at ratio 0.1 the oracle had 73 super-nodes carrying a training label against 101 for SGBGC-keep. Fewer training examples can lower accuracy for reasons unrelated to how well labels guide coarsening, which could produce "no headroom" for the wrong reason.
2. **The oracle deleted nodes at small budgets.** With many labels, SGBGC starts with more initial balls than a 2% budget allows, and the authors' pruning step deletes the smallest balls (19 Cora nodes at ratio 0.02), breaking the keep-every-node rule.

**Change.**

1. **Add a second oracle, O-sep:** purity splitting first on the 20-per-class training labels, exactly as SGBGC-keep does, then — after relabelling every node with its label-pool label — purity splitting continues on the pool labels until the budget is reached. O-sep has strictly more information than SGBGC-keep and keeps the same separation of training nodes (100 vs 101 training-labelled super-nodes in the development run).
2. **The oracle used in the decision rule is the better of O and O-sep** in each cell (higher mean test accuracy). H = best oracle − best label-free; G = best oracle − SGBGC-keep. Choosing the better oracle per cell slightly favours "target found"; the requirement that the CI lower bound of H be above 0 is kept.
3. **Keep-every-node fix:** in SGBGC-keep (and both oracles), nodes deleted by the authors' pruning step are reattached to the ball of their nearest covered node (multi-source BFS). This step never triggered in Stage 2; a regression check confirmed that the changed code reproduces the Stage 2 partitions exactly (18 of 18 cases).

**Resulting size:** 4 datasets × 10 seeds × (1 + 5 methods × 3 ratios) = **640 training runs**.

### Amendment 2 — bug fix made after the test results existed (2026-11-14)

**This amendment was made after test-set results had been computed; both versions are reported.**

**Reason.** The completeness check after the main run found that 12 of 640 runs — all on Citeseer at ratio 0.02 (all 10 oracle-O runs, 1 SGBGC-keep run, 1 O-sep run) — lost 24 nodes (0.7% of Citeseer). At that budget the authors' pruning step deleted *every* ball of a few small separate components, and Amendment 1's reattachment can only grow from a surviving ball in the same component.

**Change.** A component left with no covered node becomes one ball, as any label-free component already does. The 30 runs of the three affected Citeseer 0.02 cells were rerun with the fix (`results/results_fix.jsonl`); all now keep every node. The original runs remain in `results/results_c.jsonl`; the analysis uses the newest record per run.

**Effect on the decision: none.** Citeseer 0.02: H = +1.36 before and +1.34 after the fix; G = −0.58 in both. The decision ("target found" in Cora at 2% only) is the same with either version.
