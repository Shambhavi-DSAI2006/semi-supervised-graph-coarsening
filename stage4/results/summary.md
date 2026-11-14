# Stage 4 results — label-separation test

300 training runs. Test accuracy (%); ± = 95% CI over seeds of paired per-seed differences.

## Decision (pre-registered rule)

| Dataset | Gain: M7 − M4 (mean of 3 ratios) | Passes (≥ 2, CI > 0)? |
|---|---|---|
| Cora | +2.98 ± 1.44 | **yes** |
| Citeseer | -0.29 ± 0.78 | no |
| PubMed | +2.02 ± 2.01 | **yes** |

- Decision: **GO** — label separation gains ≥ 2 points on at least 2 of 3 datasets.

## Per cell

| Dataset | Keep | Full graph | M4 Loukas | M7 Loukas + sep | M3 SGBGC-keep | M7 − M4 | M3 − M4 | Labels lost by M4 |
|---|---|---|---|---|---|---|---|---|
| Cora | 5% | 81.0 | 76.7 ± 2.3 | 79.2 ± 1.6 | 75.5 ± 1.9 | +2.46 ± 1.69 | -1.23 ± 1.62 | 20 / 140 (14%) |
| Cora | 2% | 81.0 | 68.6 ± 3.2 | 74.6 ± 2.7 | 74.4 ± 2.7 | +5.97 ± 2.62 | +5.82 ± 2.83 | 65 / 140 (46%) |
| Cora | 1% | 81.0 | 65.4 ± 3.4 | 65.9 ± 6.3 | 55.4 ± 8.7 | +0.52 ± 4.23 | -10.06 ± 5.78 | 65 / 140 (46%) |
| Citeseer | 5% | 66.8 | 64.2 ± 3.0 | 66.3 ± 1.1 | 67.2 ± 2.1 | +2.07 ± 1.66 | +2.93 ± 2.23 | 12 / 120 (10%) |
| Citeseer | 2% | 66.8 | 64.9 ± 2.0 | 64.0 ± 2.5 | 65.6 ± 2.2 | -0.91 ± 1.17 | +0.69 ± 1.21 | 18 / 120 (15%) |
| Citeseer | 1% | 66.8 | 65.3 ± 2.7 | 63.3 ± 3.0 | 65.5 ± 2.3 | -2.02 ± 1.22 | +0.18 ± 1.40 | 24 / 120 (20%) |
| PubMed | 5% | 76.8 | 73.1 ± 2.9 | 74.9 ± 1.7 | 73.1 ± 3.0 | +1.73 ± 2.04 | -0.07 ± 1.93 | 2 / 60 (2%) |
| PubMed | 2% | 76.8 | 71.7 ± 2.5 | 73.3 ± 3.8 | 73.2 ± 2.1 | +1.68 ± 3.16 | +1.56 ± 2.18 | 6 / 60 (10%) |
| PubMed | 1% | 76.8 | 68.9 ± 5.7 | 71.5 ± 2.9 | 73.9 ± 2.1 | +2.64 ± 4.50 | +4.96 ± 3.22 | 9 / 60 (15%) |

Super-nodes (mean): Cora 5%: M4 204, M7 205, M3 201; Cora 2%: M4 129, M7 130, M3 126; Cora 1%: M4 104, M7 105, M3 102; Citeseer 5%: M4 560, M7 561, M3 540; Citeseer 2%: M4 498, M7 498, M3 474; Citeseer 1%: M4 478, M7 480, M3 459; PubMed 5%: M4 986, M7 986, M3 985; PubMed 2%: M4 395, M7 396, M3 394; PubMed 1%: M4 198, M7 199, M3 197

