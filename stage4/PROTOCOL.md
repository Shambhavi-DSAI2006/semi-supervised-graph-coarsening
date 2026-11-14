# Stage 4 protocol — Does keeping labelled nodes apart improve coarsening?

*Written and committed before any Stage 4 result was produced. Later changes go under "Amendments" with a reason; the original text stays.*

## Question

Coarsening merges nodes into super-nodes, and each super-node gets one training label (the majority among its training nodes). When training nodes of **different classes** land in the same super-node, the minority labels are lost and the super-node's features blend classes. At strong compression this is common: in a development check (seed 99, no accuracy computed), plain Loukas/SCAL coarsening lost 35% of Cora's 140 training labels at 1% of nodes.

Stage 3 hinted (exploratory) that SGBGC-keep — whose splitting keeps training nodes of different classes apart — beats label-free coarsening at strong compression even with few labels. This stage tests that idea directly and on new seeds and splits.

## Hypothesis

**H.** At strong compression, adding one rule to label-free coarsening — **never let a super-node contain training nodes of more than one class** — improves accuracy.

*Refuted if* the rule improves accuracy by **≤ 1 point** on every dataset.

**Pre-stated prediction:** the gain should be largest on Cora, smaller on Citeseer and smallest on PubMed, following how many training labels plain Loukas loses (development check at 1%: Cora 35%, Citeseer 17%, PubMed 10%).

## Methods

| # | Method | Description |
|---|---|---|
| M1 | Full graph | GCN on the original graph |
| M4 | **Loukas/SCAL** | as in Stages 2–3 (variation neighbourhoods; small components become one super-node each) |
| M7 | **Loukas + label separation** (new) | Loukas coarsening, then every super-node containing training nodes of more than one class is split into one piece per class: each member joins the class of its nearest training node (fewest hops inside the super-node; ties broken by smallest class index; members that cannot reach any training node join the piece of the largest class present). Loukas is run at a reduced target so the final number of super-nodes matches M4's for the same cell (within ±2%; up to 3 attempts, closest kept). |
| M3 | SGBGC-keep | as in Stages 2–3; a label-aware reference |

M4 vs M7 is the test: identical except for the separation rule, with the same number of super-nodes.

## Data and splits

| Dataset | Nodes | Classes | Training labels (20 / class) |
|---|---|---|---|
| Cora | 2,708 | 7 | 140 |
| Citeseer | 3,327 | 6 | 120 |
| PubMed | 19,717 | 3 | 60 |

- **New random splits for every seed** (not the public splits used in Stages 2–3): seed *s* draws, uniformly at random, a test set of 1,000 nodes and a validation set of 500 nodes; training labels are 20 per class, stratified, from the remaining nodes.
- **Seeds 10–19** (Stages 2–3 used 0–9 and the development checks used 99).
- **The test set is used only to report final numbers.** Validation labels are used only for model selection, on the original graph, and never for coarsening. Only training labels are given to M7 and M3.

## Settings

- Coarsening ratios: keep **5%, 2% and 1%** of the nodes (applied to large components; components of ≤ 10 nodes become one super-node each, as in Stages 2–3).
- Coarse-graph construction, GCN (2 layers, 64 hidden, dropout 0.5, Adam, lr 0.01, weight decay 5e-4, 200 epochs), model selection (best validation accuracy on the original graph, earliest on ties) and evaluation: **identical to Stages 2–3**, reusing their code.
- Recorded per run, besides accuracies: number of super-nodes, nodes kept, and for every coarse graph the number of super-nodes with training nodes of more than one class ("clashing") and the number of training labels lost to majority voting.

Total: 3 datasets × 10 seeds × (1 + 3 methods × 3 ratios) = **300 training runs**.

## Decision rule

For each dataset, **gain** = M7 − M4 test accuracy, paired per seed and averaged over the three ratios, with a 95% CI over the 10 seeds.

| Outcome | Condition | Next step |
|---|---|---|
| **Go** | gain ≥ 2 points with CI lower bound > 0 on **at least 2 of the 3** datasets | label separation becomes the first component of the semi-supervised coarsening method |
| **Stop** | gain ≤ 1 point on **all 3** datasets | labels do not help coarsening in practice; move to the reverse direction (coarsening to help semi-supervised learning) |
| **Inconclusive** | anything else | report; decide with the supervisor |

Reported but not part of the rule: per-cell gains; M3 − M4 per cell (a replication of Stage 3's exploratory observation on new seeds and splits); M7 − M3; clash statistics.

## Known limitations

- Three homophilous citation graphs; heterophilous graphs are deferred until the base model learns on them (Stage 3 floor effect).
- PubMed has only 3 classes, so few clashes can occur; by the stated prediction it is unlikely to pass the threshold, which means a "Go" effectively needs both Cora and Citeseer.
- The separation rule is applied after Loukas, not inside it; a rule built into the merging itself might behave differently.
- One GCN configuration; 10 seeds.

## Amendments

*(none yet)*
