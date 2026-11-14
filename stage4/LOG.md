# Stage 4 research log

Every experiment, including failed or abandoned ones, newest last. Dates are local (UTC+3).

## 2026-11-14 — Setup

- Motivation: Stage 3's exploratory hint that keeping the few labelled nodes apart matters more than having many labels.
- Feasibility check (seed 99, public splits, no accuracy computed): plain Loukas with 20 labels per class.

| Dataset | Lost at 5% | Lost at 2% | Lost at 1% | Loukas super-nodes vs target at 1% |
|---|---|---|---|---|
| Cora (140 labels) | 20 (14%) | 48 (34%) | 49 (35%) | 106 vs 101 |
| Citeseer (120) | 9 (8%) | 18 (15%) | 20 (17%) | 475 vs 452 |
| PubMed (60) | 1 (2%) | 4 (7%) | 6 (10%) | 198 vs 197 |

  "Lost" = training labels outvoted inside a super-node. Loukas reaches 1%; it overshoots the target slightly, so M7's count is matched to M4's achieved count, not to the nominal target. Loukas on PubMed takes 20–77 s per run.
- Wrote [PROTOCOL.md](PROTOCOL.md) before producing any Stage 4 result.

## 2026-11-14 — Implementation and development checks

- `stage4/run.py` reuses Stage 2's code; adds random splits per seed, Loukas + label separation (M7) and clash statistics.
- **Equivalence check:** Loukas's coarse edge weights equal the number of original edges between two super-nodes in every case (Cora, ratio 0.05), so M7 rebuilds its edges by counting crossing edges and stays identical to M4 apart from the separation rule.
- **Development run** (Cora, Citeseer, PubMed at 1% only; seed 99; validation accuracy only, test accuracy never computed): every method keeps all nodes; M7 loses **0** training labels in every cell (plain Loukas loses up to 62 on Cora) and matches M4's super-node count within 2% (largest gap: +3 on Citeseer at 1%). SGBGC-keep also loses labels at small budgets (e.g. Cora 1%: 48), because its budget runs out before all training nodes are separated. One seed takes ~4 min for Cora+Citeseer and ~2 min per ratio on PubMed.
