# Stage 6 protocol — Can a few labels improve feature k-means coarsening?

*Written and committed before any Stage 6 result was produced. Later changes go under "Amendments" with a reason; the original text stays.*

## Question

Stage 5 found (exploratory) that plain k-means on smoothed features was the most accurate of the three base coarsening methods at every compression level, and that a structural label-protection wrapper hurt it. This stage asks whether the few training labels can improve k-means coarsening when they are used in a way consistent with k-means itself (feature space), and in particular through **label propagation**.

## Methods (`guided.py`)

All variants run on the subgraph of large components (components of ≤ 10 nodes become one super-node each, as in Stages 2–5), use only training labels, and produce exactly k = ratio × (large-component nodes) clusters.

| # | Method | Description |
|---|---|---|
| K0 | **k-means** | Z = PCA₆₄(Ŝ²X), Ŝ = normalised adjacency with self-loops; k-means on Z (as in Stage 5) |
| K1 | k-means + feature-space protection | K0, then: separate clusters holding training nodes of > 1 class (each member joins the class of its nearest training node in Z); rebalance by merging the label-compatible pair with the smallest Ward cost until k clusters remain |
| **K2** | **k-means on label-propagation features (primary)** | k-means on [Z, P], both blocks scaled to mean row norm 1. P = label propagation from the training labels (Zhou et al., 2004): F ← 0.9·Ŝ·F + 0.1·Y₀, 10 steps, rows normalised |
| K3 | K2 + feature-space protection | K2, then the K1 protection in the [Z, P] space |

Plus **full graph** (GCN on the original graph) as reference. All constants (64 dimensions, 0.9, 10 steps, equal block weights) are fixed here and not tuned.

**K2 is the pre-designated primary method.** K1 and K3 are reported but do not decide the outcome, so that choosing the best of several variants afterwards cannot inflate the result.

Development check (seed 99, public splits, no accuracy computed): all variants hit the target size exactly in every cell and run in about 1 s; training labels lost to majority voting: K0 2–30, K1 and K3 0, K2 0–1.

## Data, splits and settings

- Cora, Citeseer, PubMed; 20 training labels per class.
- New random splits per seed (1,000 test and 500 validation nodes drawn uniformly; training labels from the rest), **seeds 30–39** (earlier stages used 0–29; development checks used 99).
- Keep **2%, 1% and 0.5%** of the nodes. Stage 5 showed k-means within about 1 point of the full graph at 2–5% on Cora and Citeseer, so the room for improvement lies at strong compression.
- Coarse-graph construction (super-edges = crossing-edge counts), GCN (2 layers, 64 hidden, dropout 0.5, Adam, lr 0.01, weight decay 5e-4, 200 epochs), model selection (best validation accuracy on the original graph) and evaluation: identical to Stages 2–5.
- The test set is used only to report final numbers; validation labels only for model selection; only training labels are used by any coarsening step, including label propagation.

Total: 3 datasets × 10 seeds × (1 + 4 methods × 3 ratios) = **390 training runs**.

## Hypothesis and decision rule

**H.** Label-propagation features (K2) improve k-means coarsening at strong compression.

For each dataset, **gain** = K2 − K0, paired per seed and averaged over the three ratios, 95% CI over 10 seeds.

| Outcome | Condition |
|---|---|
| **Go** | gain ≥ **1.5** points with CI lower bound > 0 on at least 2 of 3 datasets, and no dataset whose CI lies entirely below −1 |
| **Stop** | gain ≤ **0.5** points on all 3 datasets |
| **Inconclusive** | anything else |

The thresholds are lower than in Stages 2–5 (2 / 1 points) because k-means leaves less room: in Stage 5 it was 0.8 points below the full graph on Cora at 2% and level with it on Citeseer. They are fixed here, before any Stage 6 result.

Reported, not part of the rule: per-cell results; K1 − K0, K3 − K0, K3 − K2; each method against the full graph; labels lost.

**Pre-stated prediction:** gains largest where k-means is furthest below the full graph — at 0.5% and 1%, and on Cora and PubMed rather than Citeseer.

## Known limitations

- Three homophilous citation graphs; label propagation relies on neighbours sharing labels, so this may not carry over to heterophilous graphs.
- k-means coarsening produces super-nodes that need not be connected in the graph.
- Fixed constants; one GCN configuration; 10 seeds.

## Amendments

*(none yet)*
