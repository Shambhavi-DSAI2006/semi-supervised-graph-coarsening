# Stage 5 research log

Every experiment, including failed or abandoned ones, newest last. Dates are local (UTC+3).

## 2026-11-14 — Setup

- Requirement from Shambhavi: the method must be **general** — usable with any coarsening algorithm, not tied to one. Stage 4's approach (re-run Loukas at a smaller target) is not: it needs a base method that can be re-run with a different target, and it coarsens the rest of the graph harder.
- Built `protect.py`: separate clashing super-nodes, then rebalance by merging adjacent label-compatible super-nodes back to the original count. It uses only the partition, the adjacency and the training labels.
- **Feasibility checks** (seed 99, public splits, no accuracy computed):
  - On Loukas, GB-struct and k-means over Cora, Citeseer and PubMed at 5%, 2%, 1%: protection always returned exactly the input number of super-nodes, kept every node, and reduced lost training labels to 0 (from up to 60 for Loukas and GB-struct and 22 for k-means on Cora). It takes under 0.1 s per partition, versus re-running Loukas (up to 77 s on PubMed) in Stage 4.
  - **Dropped as base methods:** Stage 2's feature clustering with connectivity splitting could not control its size (PubMed: 1,227 instead of 985 super-nodes at 5%, **1** instead of 197 at 1%); heavy-edge matching (Loukas library) stalled at 412 super-nodes on Cora for every target (101–201) and did not finish on PubMed within 15 minutes. Plain feature k-means hits every target exactly and was chosen as the third, structure-free family.
- Wrote [PROTOCOL.md](PROTOCOL.md) before producing any Stage 5 result.

## 2026-11-14 — Implementation check

- `stage5/run.py` reuses Stage 2's code and Stage 4's split and separation helpers.
- **Development run** (Cora, seed 99, validation accuracy only, test accuracy never computed; 1.5 min): all 7 coarsened methods ran at 5%, 2%, 1%; every protected partition kept all nodes, had exactly the base method's super-node count, and lost 0 training labels (base methods lost 14–64). Stage 4's separation-with-re-run again missed its size at 1% (113 vs 104), as in Stage 4.

## 2026-11-14 — Main run

- 660 runs (Cora+Citeseer and PubMed as two processes), 14:24–15:52. 66 cells × 10 seeds, no duplicates, no errors, every node kept. Every protected partition had exactly the base method's super-node count and lost 0 training labels; rebalancing needed its fallback 14 times in total.

### Result of the decision rule: **INCONCLUSIVE — not general**

| Base | Cora | Citeseer | PubMed |
|---|---|---|---|
| Loukas/SCAL | +1.20 ± 1.96 | −0.57 ± 1.02 | +0.75 ± 1.17 |
| GB-struct | **+3.25 ± 1.34** | **+3.84 ± 1.86** | +0.21 ± 1.11 |
| k-means | **−2.13 ± 0.92** (harmed) | −1.88 ± 1.16 | +0.58 ± 0.45 |

Only GB-struct passes on ≥ 2 datasets (1 of 3 base methods needed 2), and k-means on Cora is clearly harmed. The wrapper is **not** a general improvement.

### What the per-cell results show (my reading)

1. **Protection helps weak partitions and hurts strong ones.** GB-struct is the least accurate base and loses the most labels (up to 43%); it gains +1.4 to +5.8. k-means is the most accurate base; protection costs it up to −6.0 (Citeseer 1%). The wrapper's repair steps are graph-structural (hop distance for splitting, crossing edges for rebalancing); for a feature-based partition they break good groups.
2. **The general rebalancing is worse than Stage 4's re-run approach** for Loukas in all 9 cells (−0.1 to −3.8). Merging neighbouring super-nodes by a fixed score is cruder than letting Loukas recompute the whole partition. Stage 4's "budget cost" explanation was incomplete: *how* the budget is restored matters too.
3. **Prediction partly wrong again:** gains did not follow labels lost across bases (k-means lost the fewest labels yet was harmed, rather than gaining little).

### Exploratory observation: plain feature k-means is a strong coarsener

Without any label use, k-means on smoothed features was the most accurate of the three base methods in all 9 cells (k-means with protection was slightly higher in 4 cells): Cora 2% 79.6 vs Loukas 73.4 (full graph 80.4); Citeseer 1% 66.5 vs 64.8 (full graph 66.5); PubMed 2% 76.4 vs 70.9 (full graph 78.2). Stage 2's feature clustering, which additionally split clusters into connected pieces, was much weaker — the connectivity requirement, not the features, was the problem there. This was not a pre-registered comparison.

### Options for the next step (to decide with the supervisor; the rule says report and decide)

1. **Similarity-consistent protection:** make the repair use the base method's own notion of similarity (feature distance for k-means, graph distance for structural methods). More effective, but less "plug-in".
2. **Build on feature k-means:** it is already near-lossless at strong compression with few labels; test whether the few labels can improve *it* (e.g. label-guided seeding or cannot-link k-means), with protection designed for feature space.
3. **Reverse direction:** use coarsening to help semi-supervised learning, since labels have repeatedly added little to coarsening itself.
