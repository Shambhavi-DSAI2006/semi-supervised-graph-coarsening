# Stage 3 research log

Every experiment, including failed or abandoned ones, newest last. Dates are local (UTC+3).

## 2026-11-14 — Setup

- Motivation: Stage 2 ended in "pivot" (labels gave no advantage at 5–20 per class). Before choosing a new direction, measure whether labels could help coarsening at all (headroom), in settings that favour labels.
- Datasets: Roman-empire and Amazon-ratings fetched via git sparse checkout of yandex-research/heterophilous-graphs (commit a431395); loaded with PyTorch Geometric's `HeterophilousGraphDataset`. Both are single connected components. Edge homophily: Roman-empire 0.047, Amazon-ratings 0.380 (Cora 0.810, Citeseer 0.736).
- Feasibility check (timing only, seed 99, no accuracy computed): on both large graphs at ratios 0.1 and 0.02, Loukas, GB-struct and SGBGC-keep (20 per class and oracle) each finish in 2–23 s, keep every node, and reach the same super-node budget (within +2 super-nodes for Loukas).
- Wrote [PROTOCOL.md](PROTOCOL.md) before producing any Stage 3 result.
