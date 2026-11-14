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

## 2026-11-14 — Main run

- 300 runs (Cora+Citeseer and PubMed as two processes), 12:34–13:20. 30 cells × 10 seeds, no duplicates, no errors, every node kept; M7 lost 0 training labels in every run.
- **Deviation from the stated size matching:** 6 of 90 M7 runs (all Cora) ended outside ±2% of M4's super-node count after the 3 permitted attempts (closest kept, as the protocol's procedure says). In 4 of the 6, M7 had *more* super-nodes (+5 to +7), which favours M7. Sensitivity below.

### Result of the decision rule: **GO**

| Dataset | Gain M7 − M4 (mean of 3 ratios) | Passes? |
|---|---|---|
| Cora | +2.98 ± 1.44 | yes |
| Citeseer | −0.29 ± 0.78 | no |
| PubMed | +2.02 ± 2.01 | yes (CI lower bound +0.01) |

### How strong is this? (my assessment)

- **Fragile overall.** PubMed passes by a hair (CI lower bound +0.01). Excluding the 6 size-mismatched runs, Cora's average gain is +1.90 ± 2.75, which would *not* pass. By the letter of the rule the outcome is Go; the evidence behind it is modest.
- **One effect is large and robust:** Cora at 2% of nodes, +5.97 ± 2.62 (+6.00 ± 3.10 without the mismatched runs). There plain Loukas loses 46% of the training labels.
- **It can also hurt:** Citeseer at 1%, −2.02 ± 1.22.
- **The pre-stated prediction was partly wrong.** Predicted order of gains: Cora > Citeseer > PubMed. Observed: Cora (+3.0) > PubMed (+2.0) > Citeseer (−0.3). The share of labels lost alone does not explain the pattern.

### Exploratory explanation (not part of the rule)

Separating a clashing super-node creates extra pieces, and to keep the same total count Loukas must merge more elsewhere. The cost, as a share of the super-nodes available in the large components:

| Cell | Extra pieces / large-part budget | Labels lost by M4 | Gain |
|---|---|---|---|
| PubMed 5 / 2 / 1% | 0.1% / 1.3% / 3.6% | 2% / 10% / 15% | +1.7 / +1.7 / +2.6 |
| Citeseer 5 / 2 / 1% | 9% / 24% / 34% | 10% / 15% / 20% | +2.1 / −0.9 / −2.0 |
| Cora 5 / 2 / 1% | 14% / 41% / 53% | 14% / 46% / 46% | +2.5 / +6.0 / +0.5 |

The pattern fits **gain ≈ benefit (labels saved) − cost (coarser elsewhere)**: separation helps when it is cheap (PubMed) or when it saves very many labels (Cora 2%), and hurts when it is expensive relative to what it saves (Citeseer 1–2%, Cora 1%). This suggests the next design: protect training labels **without** taking budget from the rest of the graph.

### Secondary (replication of Stage 3's observation)

SGBGC-keep − Loukas on new seeds and splits: Cora 2% +5.82 ± 2.83, PubMed 1% +4.96 ± 3.22, Citeseer 5% +2.93 ± 2.23, but Cora 1% −10.06 ± 5.78 (SGBGC-keep's budget runs out before training nodes are separated). Mixed: the label-aware advantage at strong compression replicates in some cells, not all.
