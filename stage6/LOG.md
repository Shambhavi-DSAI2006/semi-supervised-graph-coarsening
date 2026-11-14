# Stage 6 research log

Every experiment, including failed or abandoned ones, newest last. Dates are local (UTC+3).

## 2026-11-14 — Setup

- Chosen with Shambhavi after Stage 5 (option 2): build on feature k-means, the strongest base coarsener found, and test whether few labels improve it.
- Implemented `guided.py`: K0 k-means; K1 + feature-space protection (separate by nearest training node in feature space, rebalance by Ward cost among label-compatible clusters); K2 k-means on label-propagation features; K3 = K2 + protection.
- Development check (seed 99, public splits, no accuracy computed), Cora / Citeseer / PubMed at 2%, 1%, 0.5%: every variant hit the target number of super-nodes exactly; ~1 s each. Training labels lost: K0 2–30; K1 and K3 0; K2 0–1 — label-propagation features keep training nodes of different classes apart almost entirely without any repair step.
- Wrote [PROTOCOL.md](PROTOCOL.md) before producing any Stage 6 result. K2 is pre-designated as the primary method; thresholds lowered to 1.5 / 0.5 points with the reason stated (k-means is already close to the full graph).

## 2026-11-14 — Implementation check

- `stage6/run.py` reuses Stage 2's training code and Stage 4's split and clash helpers.
- Development run (Cora and PubMed, seed 99, validation accuracy only, test accuracy never computed; 3 min): all variants ran at 2%, 1%, 0.5% with exact sizes and every node kept. K3 often equals K2 because K2's clusters are already almost clash-free, so its protection step has little to do.
