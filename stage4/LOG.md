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
