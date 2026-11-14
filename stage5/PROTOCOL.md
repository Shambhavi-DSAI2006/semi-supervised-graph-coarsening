# Stage 5 protocol — Does a general label-protection wrapper improve any coarsening method?

*Written and committed before any Stage 5 result was produced. Later changes go under "Amendments" with a reason; the original text stays.*

## Question

Stage 4 found that keeping training nodes of different classes in separate super-nodes improved Loukas coarsening on Cora and PubMed, but the way it was done (split, then re-run Loukas at a smaller target) takes budget from the rest of the graph and works only for a method that can be re-run with a different target.

This stage tests a **general** version that works on top of **any** coarsening method, using only its output partition, the graph and the training labels.

## The method under test: label protection (`protect.py`)

Input: any partition into super-nodes, the adjacency matrix, the training labels. Output: a partition with **no super-node holding training nodes of more than one class** and **exactly the same number of super-nodes**.

1. **Separate:** each super-node with training nodes of k > 1 classes is split into k pieces; every member joins the class of its nearest training node (fewest hops inside the super-node; ties → smallest class index); members that reach no training node join the class with the most training nodes there (ties → smallest class).
2. **Rebalance:** repeatedly merge the pair of adjacent, label-compatible super-nodes (together holding training nodes of at most one class) with the highest score w_ij / (n_i + n_j) (w_ij = original edges between them; n = sizes), until the original count is restored. If no adjacent compatible pair exists, merge the compatible pair with the smallest combined size.

The super-edge construction of the protected partition is the same as the base method's (binary for GB-struct; crossing-edge counts for Loukas and k-means — Loukas's own weights equal these counts, Stage 4 log).

## Hypothesis

**H.** Label protection improves accuracy at strong compression **across different coarsening methods**, not just one.

*Refuted if* the gain is ≤ 1 point for every (dataset, base method) pair.

**Pre-stated prediction:** gains are largest where the base method loses most training labels. In a development check (seed 99, no accuracy computed) Loukas and GB-struct lost more labels on Cora (17–60 of 140) than k-means (16–22), and PubMed's bases lost the fewest (1–7 of 60). So: largest gains on Cora, and for Loukas and GB-struct; smallest on PubMed.

## Base methods (three different families)

| Base | Family | Notes |
|---|---|---|
| **Loukas/SCAL** | structure, spectral, merges neighbourhoods | as in Stages 2–4 |
| **GB-struct** | structure, granular balls, splits from hub nodes | as in Stages 2–3 (SGBGC machinery without labels) |
| **k-means** | node features only, ignores the graph | features smoothed twice over the normalised adjacency (with self-loops), PCA to 64 dimensions, k-means with k = target number of super-nodes (`random_state` = seed) |

Feature clustering with connectivity splitting (Stage 2's M6) and heavy-edge matching were considered and dropped before this protocol: in development checks they could not hit the target sizes (M6 gave 1 super-node instead of 197 on PubMed at 1%; heavy-edge matching stalled at 412 instead of 101–201 on Cora).

## Methods compared

For each base B: **B** and **B + protection**. Plus:
- **Full graph** (GCN on the original graph), as reference.
- **Loukas + separation from Stage 4** (split, then re-run Loukas at a smaller target), as reference for the earlier, non-general approach.

## Data, splits and settings

- Cora, Citeseer, PubMed; 20 training labels per class.
- **New random splits for every seed** (1,000 test and 500 validation nodes drawn uniformly; training labels from the rest), **seeds 20–29** (Stages 2–4 used 0–19; development checks used 99).
- Keep **5%, 2% and 1%** of the nodes (applied to large components; components of ≤ 10 nodes become one super-node each, as before).
- Coarse-graph construction, GCN (2 layers, 64 hidden, dropout 0.5, Adam, lr 0.01, weight decay 5e-4, 200 epochs), model selection (best validation accuracy on the original graph) and evaluation: identical to Stages 2–4.
- The test set is used only to report final numbers; validation labels only for model selection; only training labels are given to any coarsening step.
- Recorded per run: test and validation accuracy, super-node count, nodes kept, training labels lost to majority voting, pieces added by separation, rebalancing fallbacks.

Total: 3 datasets × 10 seeds × (1 + 7 coarsened methods × 3 ratios) = **660 training runs**.

## Decision rule

For each of the 9 (dataset, base) pairs: **gain** = (B + protection) − B, paired per seed and averaged over the three ratios, with a 95% CI over the 10 seeds.

| Outcome | Condition |
|---|---|
| **Go — general** | gain ≥ 2 points with CI lower bound > 0 on at least 2 of 3 datasets, **for at least 2 of the 3 base methods**, and **no pair** whose CI lies entirely below −1 point |
| **Stop** | gain ≤ 1 point for **all 9** pairs |
| **Inconclusive** | anything else |

Reported but not part of the rule: per-cell gains; (Loukas + protection) − (Loukas + Stage 4 separation); comparison with the full graph; labels lost.

## Known limitations

- Three homophilous citation graphs, one GCN configuration, 10 seeds.
- The rebalancing score and the nearest-training-node rule are simple fixed choices, not tuned.
- Protection can only act on training nodes; it does nothing for unlabelled nodes of different classes merged together.

## Amendments

*(none yet)*
