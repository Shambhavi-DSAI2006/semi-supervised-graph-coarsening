# Stage 7 protocol — Is there a few-label gap that coarsening fills and condensation cannot?

*Written and committed before any Stage 7 result was produced. Later changes go under "Amendments" with a reason; the original text stays.*

## Why this stage

The literature check after Stage 6 showed that label-guided k-means coarsening (our K2) is not new as a method. The open question is whether there is a *setting* where existing methods fail. The fastest recent competitor is condensation by **CGC** (Gao et al., WWW 2025: training-free, closed-form, faster than gradient-matching condensation). Every condensed node CGC builds comes from training nodes (plus resampled copies of them). We checked this in its code and in a dry run: with k labels per class, CGC cannot produce more than C·k condensed nodes. On Cora with 1 label per class it produces 7 nodes and no edges, and asking for more crashes. Coarsening uses every node, labelled or not.

**H.** With very few labels (1–5 per class), coarsening that uses the unlabelled nodes (K2) trains a more accurate GNN than CGC at the same graph size, across graph types, at comparable cost.

This stage measures the gap. It does not propose the final method: if the gap exists, Stage 8 designs a method around it; if not, we pivot.

## Methods

| Name | Family | What it is | Sizes |
|---|---|---|---|
| **K2** (primary) | coarsening | k-means on [PCA₆₄(Ŝ²X), label propagation from training labels] (Stage 6, constants unchanged), run on the whole graph so it hits any size exactly | M, F |
| K0 | coarsening | k-means on PCA₆₄(Ŝ²X), no labels | M, F |
| Loukas | coarsening | variation-neighbourhoods (as SCAL uses it), small-component rule of Stages 2–6 | F |
| SGBGC-keep | coarsening | SGBGC keeping unlabelled nodes (Stage 2), small-component rule | F |
| **CGC-tuned** (primary comparator) | condensation | official code via `cgc_adapter.py`; configuration and trainer chosen per dataset and label budget on validation (below) | M |
| CGC-default | condensation | official code with its own Cora settings (dropout 0.8, adj_T 0.8, alpha 7, tau 1, aug_ratio 0.5) and its own trainer | M |
| Full graph | reference | GCN on the original graph | — |

Coarsened graphs are built as in Stages 2–6: super-node features = member mean, super-edge weights = crossing-edge counts, label = majority of training labels inside (none if no training node).

**Sizes.**
- **M (matched, primary)** = number of training labels, C·k. This is CGC's largest possible size; every method at M has exactly M nodes.
- **F (free, secondary)** = 2% of the nodes. Only coarsening can reach it when 2% > C·k. Loukas and SGBGC-keep: within ±2% as in earlier stages.

## Adapter check (done before writing this protocol)

`cgc_adapter.py` repeats the call sequence of CGC's `main.py` on our data objects. On Cora with the public split at ratio 0.026 it gives 81.7 / 82.7 / 82.2 (mean 82.2) with CGC's own trainer. The official script gives 82.2 here, and the CGC paper reports 82.3 ± 1.3 (Table 3). With the Stage 2–6 trainer, the same graph gives 80.7, so CGC keeps its own trainer option below.

## Data, splits, labels

- **8 datasets**:
  - homophilous: Cora, Citeseer, PubMed (citation); Coauthor-CS (co-authorship); Amazon-Photo, Amazon-Computers (co-purchase);
  - heterophilous: Roman-empire, Amazon-ratings.
  - Coauthor-Physics was planned but one full-graph evaluation takes 0.4 s on this 2-CPU machine (≈13 h for the stage), so it is replaced by the two Amazon co-purchase graphs.
  - ogbn-arxiv is added later as a scale check if the file can be obtained; it is not part of the decision.
- **Splits:** new uniform random split per seed: 1,000 test and 500 validation nodes, the same rule as Stages 4–6.
- **Labels:** k = **1, 5, 20** per class, drawn from the remaining nodes. Evaluation seeds **40–49** (earlier stages used 0–39). **Tuning seed 20** (its split and labels are used only for the choices below, never reported).
- Only training labels are used by any coarsening or condensation step. Validation labels are used only for model selection and for the tuning below. The test set is used only to report.

## Training and model selection

Two trainers:

| Trainer | Settings |
|---|---|
| S (Stages 2–6) | 2 layers, 64 hidden, dropout 0.5, Adam lr 0.01, weight decay 5e-4, 200 epochs |
| C (CGC's own) | 2 layers, 256 hidden, dropout 0.8, lr 0.01 then 0.001 from epoch 500, weight decay 5e-4, 1,000 epochs |

Every trainer is fitted on the small graph and evaluated on the original graph. The epoch with the best validation accuracy is kept (checked every 5 epochs for S, every 10 for C, to keep full-graph evaluation affordable).

**Tuning (seed 20 only, per dataset and label budget):** CGC-tuned picks the best validation accuracy over 6 configurations × 2 trainers:
- alpha ∈ {1, 7} × adj_T ∈ {0.8, 0.9, 0.95}, with tau 1 and aug_ratio 0.5;
- trainers S and C.

These values are taken from CGC's own per-dataset settings. Coarsening methods and the full graph get no tuning and use trainer S. This favours the baseline: if K2 still wins, the result is conservative.

**Second GNN:** GraphSAGE (mean aggregation, trainer S) for K2, CGC-tuned (same configuration) and the full graph at size M.

## Measurements

- Test accuracy.
- Coarsening/condensation time.
- Training time per epoch on the small graph (excluding original-graph evaluation).
- Peak extra memory (RSS) of coarsening/condensation and of training.
- Size of the small graph (nodes, edges).
- Number of hyperparameters a user must set.

Time and memory are measured on this shared 2-CPU machine. They are reported as ratios between methods measured in the same process, not as absolute claims.

## Hypothesis test and decision rule

Primary comparison: K2 vs CGC-tuned at size M, GCN, at 1 and 5 labels per class. Per dataset, **gain** = K2 − CGC-tuned, paired per seed and averaged over the two budgets, with a 95% CI over 10 seeds.

| Outcome | Condition |
|---|---|
| **Go** (a gap exists) | gain ≥ **1** point with CI lower bound > 0 on **at least 5 of 8** datasets, **no** dataset whose CI lies entirely below −2, and K2's coarsening + training time within **2×** CGC's |
| **Stop** (no gap) | mean gain ≤ 0 on at least 5 of 8 datasets |
| **Inconclusive** | anything else |

Reported but not part of the rule:
- the 20-labels-per-class results;
- the F-size results against CGC at M and against the full graph;
- average rank of every method per budget;
- GraphSAGE results;
- K2 − K0 (do labels help?);
- CGC-tuned − CGC-default (how much tuning matters);
- time, memory and hyperparameter counts.

**Pre-stated predictions:**
1. K2 beats CGC at 1 label per class on the homophilous graphs by a wide margin, with a smaller margin at 5.
2. CGC closes the gap or wins at 20 labels per class.
3. On the heterophilous graphs, both are near the floor and the difference is small.
4. At size F, K2 approaches the full graph on PubMed and Coauthor-CS.

## Compute

About 30 minutes of tuning and about 3 hours for the main runs, over 8 datasets × 10 seeds × 3 budgets × 12 runs = 2,880 training runs. Runs are written one per line and resumed per dataset and seed.

## Known limitations

- One condensation competitor run from official code (CGC). Bonsai (ICLR 2025) also condenses from training-node computation trees, so it has the same limitation; it is added if its official code can be found and runs on CPU. LAGC has no public code.
- 2 CPUs: time and memory are indicative.
- The size-M comparison forces coarsening down to C·k nodes, a disadvantage for K2 at 1 label per class (7 nodes on Cora). Size F shows the other side.
- Tuning on a single seed.

## Amendments
