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

## 2026-11-14 — Main run (interrupted and resumed)

- Started 16:55 as two processes. Both were stopped by the environment at ~17:00 when the session went idle (Cora seeds 30–33 and part of 34 done; PubMed seed 30 done). Resumed at 17:22 for the remaining seeds (`results_a2.jsonl`, `results_b2.jsonl`, then Citeseer). Runs are deterministic, so the partly-run Cora seed 34 is simply recomputed; the analysis uses the newest record per run.
- Completed 18:13. 390 unique runs (5 Cora seed-34 runs recomputed, identical to the interrupted ones), 39 cells × 10 seeds, every node kept, no errors.

### Result of the decision rule: **INCONCLUSIVE**

| Dataset | Gain K2 − K0 (mean of 2%, 1%, 0.5%) | Passes? |
|---|---|---|
| Cora | +0.79 ± 2.30 | no |
| Citeseer | +0.36 ± 1.04 | no |
| PubMed | **+1.96 ± 1.78** | yes |

### My reading

- **Label-propagation features help where k-means has room and labels propagate well:** on PubMed K2 matches the full graph at 2% and 1% (77.9 and 77.6 vs 77.7) and is 0.9 below it at 0.5%, whereas plain k-means is 1.2–3.5 below.
- **On Cora and Citeseer k-means already does almost everything:** at 2% it is within 1–1.5 points of the full graph, and Citeseer at 1% equals it. At 0.5% on Cora every variant collapses (63–66 vs 80), with very high variance (CI ±5), so no method choice is resolved there.
- **Feature-space protection (K1) is mixed:** slightly worse on Cora at 2–1%, slightly better on PubMed; it fixed the structural wrapper's harm to k-means (Stage 5) but adds little.
- **Prediction partly wrong:** the largest gains were on PubMed, not at the cells where k-means is furthest below the full graph (Cora 0.5%).

### Literature check done during this stage (affects positioning)

- **LAGC** (Kumar et al., UAI 2024): published semi-supervised, label-aware graph coarsening (optimisation-based) — a required baseline. Not the supervisor's work.
- **SCAL** (Huang et al., KDD 2021) already targets the semi-supervised setting and already noted that super-nodes with mixed labels hurt (it discards them).
- **CGC** (Gao et al., WWW 2025): training-free condensation as class partition via clustering of propagated features — close to K0/K2; condenses ogbn-products in ~30 s. Uses only the labelled training nodes.
- **Bonsai** (ICLR 2025): linear-time, gradient-free condensation.
- **GB-CGNN** (SGBGC group, 2026): large-scale granular-ball coarsening; found label-purity splitting inefficient.
- Consequence: k-means/label-propagation coarsening is not novel; any claim must be against these baselines. Open gap worth testing: condensation (CGC, Bonsai) summarises the labelled training nodes, while coarsening keeps all nodes — coarsening may win at very few labels (1–5 per class) on large graphs.
