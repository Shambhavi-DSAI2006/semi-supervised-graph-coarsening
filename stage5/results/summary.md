# Stage 5 results — general label-protection wrapper

660 training runs. Test accuracy (%); ± = 95% CI over seeds of paired per-seed differences.

## Decision (pre-registered rule)

Gain = (base + protection) − base, averaged over keeping 5%, 2% and 1% of nodes.

| Base method | Cora | Citeseer | PubMed | Datasets passing |
|---|---|---|---|---|
| Loukas/SCAL | +1.20 ± 1.96 | -0.57 ± 1.02 | +0.75 ± 1.17 | 0 / 3 |
| GB-struct | +3.25 ± 1.34 ✓ | +3.84 ± 1.86 ✓ | +0.21 ± 1.11 | 2 / 3 |
| k-means | -2.13 ± 0.92 | -1.88 ± 1.16 | +0.58 ± 0.45 | 0 / 3 |

- Decision: **INCONCLUSIVE** — 1 of 3 base methods pass on ≥ 2 datasets; clearly harmed: k-means on Cora.

## Per cell

| Dataset | Keep | Base | Base acc. | + protection | Gain | Labels lost by base |
|---|---|---|---|---|---|---|
| Cora | 5% | Loukas/SCAL | 76.1 | 76.8 | +0.64 ± 1.55 | 23/140 (16%) |
| Cora | 5% | GB-struct | 72.6 | 75.6 | +3.05 ± 1.33 | 32/140 (23%) |
| Cora | 5% | k-means | 80.9 | 80.2 | -0.76 ± 1.26 | 16/140 (12%) |
| Cora | 2% | Loukas/SCAL | 73.4 | 74.3 | +0.93 ± 4.28 | 43/140 (30%) |
| Cora | 2% | GB-struct | 64.4 | 69.7 | +5.28 ± 3.07 | 51/140 (37%) |
| Cora | 2% | k-means | 79.6 | 76.1 | -3.44 ± 1.62 | 24/140 (17%) |
| Cora | 1% | Loukas/SCAL | 66.5 | 68.6 | +2.03 ± 3.41 | 44/140 (31%) |
| Cora | 1% | GB-struct | 58.9 | 60.3 | +1.42 ± 4.45 | 60/140 (43%) |
| Cora | 1% | k-means | 74.7 | 72.5 | -2.19 ± 2.14 | 29/140 (21%) |
| Citeseer | 5% | Loukas/SCAL | 64.5 | 66.2 | +1.69 ± 1.22 | 13/120 (11%) |
| Citeseer | 5% | GB-struct | 60.8 | 63.3 | +2.45 ± 2.00 | 18/120 (15%) |
| Citeseer | 5% | k-means | 65.8 | 68.1 | +2.26 ± 1.62 | 12/120 (10%) |
| Citeseer | 2% | Loukas/SCAL | 64.0 | 63.8 | -0.19 ± 1.46 | 21/120 (17%) |
| Citeseer | 2% | GB-struct | 58.1 | 61.4 | +3.30 ± 2.68 | 27/120 (23%) |
| Citeseer | 2% | k-means | 66.5 | 64.6 | -1.90 ± 1.65 | 18/120 (15%) |
| Citeseer | 1% | Loukas/SCAL | 64.8 | 61.6 | -3.20 ± 1.21 | 27/120 (23%) |
| Citeseer | 1% | GB-struct | 52.0 | 57.8 | +5.76 ± 3.45 | 41/120 (34%) |
| Citeseer | 1% | k-means | 66.5 | 60.5 | -6.00 ± 1.72 | 23/120 (19%) |
| PubMed | 5% | Loukas/SCAL | 73.2 | 74.0 | +0.79 ± 1.88 | 2/60 (3%) |
| PubMed | 5% | GB-struct | 74.0 | 74.8 | +0.85 ± 1.28 | 1/60 (2%) |
| PubMed | 5% | k-means | 77.1 | 77.2 | +0.04 ± 0.43 | 1/60 (1%) |
| PubMed | 2% | Loukas/SCAL | 70.9 | 72.4 | +1.49 ± 0.96 | 5/60 (8%) |
| PubMed | 2% | GB-struct | 73.5 | 74.1 | +0.61 ± 1.16 | 4/60 (7%) |
| PubMed | 2% | k-means | 76.4 | 76.8 | +0.46 ± 0.66 | 2/60 (3%) |
| PubMed | 1% | Loukas/SCAL | 69.5 | 69.5 | -0.04 ± 2.32 | 7/60 (12%) |
| PubMed | 1% | GB-struct | 72.3 | 71.5 | -0.82 ± 2.20 | 9/60 (14%) |
| PubMed | 1% | k-means | 74.2 | 75.4 | +1.25 ± 1.22 | 3/60 (6%) |

## References (not part of the rule)

| Dataset | Keep | Full graph | Loukas | Loukas + protection | Loukas + Stage 4 separation | protection − Stage 4 |
|---|---|---|---|---|---|---|
| Cora | 5% | 80.4 | 76.1 | 76.8 | 79.2 | -2.40 ± 1.37 |
| Cora | 2% | 80.4 | 73.4 | 74.3 | 74.8 | -0.44 ± 2.82 |
| Cora | 1% | 80.4 | 66.5 | 68.6 | 72.4 | -3.81 ± 3.71 |
| Citeseer | 5% | 66.5 | 64.5 | 66.2 | 66.6 | -0.35 ± 0.94 |
| Citeseer | 2% | 66.5 | 64.0 | 63.8 | 64.3 | -0.54 ± 1.35 |
| Citeseer | 1% | 66.5 | 64.8 | 61.6 | 61.7 | -0.10 ± 1.63 |
| PubMed | 5% | 78.2 | 73.2 | 74.0 | 74.5 | -0.56 ± 1.70 |
| PubMed | 2% | 78.2 | 70.9 | 72.4 | 72.8 | -0.35 ± 2.53 |
| PubMed | 1% | 78.2 | 69.5 | 69.5 | 71.6 | -2.19 ± 4.44 |

