# Stage 3 results — headroom test

640 training runs. Test accuracy (%), mean over seeds; ± = 95% CI of paired per-seed differences.

## Headroom per cell

| Dataset | Keep | Best label-free | Best oracle | H = oracle − label-free | G = oracle − SGBGC-keep | Target cell? |
|---|---|---|---|---|---|---|
| Cora | 10% | Loukas/SCAL (78.5) | Oracle O (78.8) | +0.37 ± 1.60 | +1.73 ± 1.98 | no |
| Cora | 5% | Loukas/SCAL (77.4) | Oracle O (76.4) | -1.01 ± 1.70 | +1.74 ± 3.11 | no |
| Cora | 2% | Loukas/SCAL (72.0) | Oracle O (75.9) | +3.86 ± 2.14 | +2.01 ± 3.51 | **yes** |
| Citeseer | 10% | Loukas/SCAL (67.8) | Oracle O-sep (67.8) | +0.07 ± 1.16 | -1.00 ± 0.73 | no |
| Citeseer | 5% | Loukas/SCAL (66.5) | Oracle O-sep (67.5) | +0.94 ± 1.41 | -0.62 ± 0.62 | no |
| Citeseer | 2% | Loukas/SCAL (65.5) | Oracle O-sep (66.9) | +1.34 ± 1.25 | -0.58 ± 0.64 | no |
| Roman-empire | 10% | Loukas/SCAL (13.9) | Oracle O-sep (15.2) | +1.26 ± 1.17 | +0.46 ± 0.73 | no |
| Roman-empire | 5% | GB-struct (13.0) | Oracle O-sep (13.6) | +0.67 ± 0.64 | -0.63 ± 1.06 | no |
| Roman-empire | 2% | Loukas/SCAL (12.2) | Oracle O-sep (13.3) | +1.13 ± 1.41 | -0.15 ± 0.43 | no |
| Amazon-ratings | 10% | Loukas/SCAL (30.9) | Oracle O-sep (31.5) | +0.60 ± 2.97 | +3.16 ± 2.42 | no |
| Amazon-ratings | 5% | Loukas/SCAL (31.1) | Oracle O (31.3) | +0.18 ± 4.04 | +3.05 ± 2.63 | no |
| Amazon-ratings | 2% | Loukas/SCAL (31.8) | Oracle O (31.6) | -0.18 ± 4.14 | +1.49 ± 3.17 | no |

- Decision: **TARGET FOUND** in: Cora at 2%

## Accuracy by method

### Cora (full graph: 80.5 ± 1.2)

| Method | keep 10% | keep 5% | keep 2% |
|---|---|---|---|
| Loukas/SCAL | 78.5 ± 1.5 | 77.4 ± 3.0 | 72.0 ± 3.3 |
| GB-struct | 76.3 ± 2.2 | 73.5 ± 1.2 | 66.7 ± 2.2 |
| SGBGC-keep (20/class) | 77.1 ± 2.1 | 74.6 ± 2.7 | 73.9 ± 4.4 |
| Oracle O | 78.8 ± 1.7 | 76.4 ± 3.0 | 75.9 ± 2.0 |
| Oracle O-sep | 77.5 ± 1.9 | 75.3 ± 2.6 | 73.8 ± 5.0 |

### Citeseer (full graph: 68.0 ± 2.0)

| Method | keep 10% | keep 5% | keep 2% |
|---|---|---|---|
| Loukas/SCAL | 67.8 ± 1.7 | 66.5 ± 2.3 | 65.5 ± 1.3 |
| GB-struct | 65.5 ± 2.7 | 63.4 ± 2.3 | 61.6 ± 3.0 |
| SGBGC-keep (20/class) | 68.9 ± 1.4 | 68.1 ± 1.8 | 67.4 ± 1.6 |
| Oracle O | 65.3 ± 2.2 | 64.9 ± 3.4 | 66.2 ± 3.1 |
| Oracle O-sep | 67.8 ± 1.0 | 67.5 ± 1.4 | 66.9 ± 1.6 |

### Roman-empire (full graph: 27.5 ± 1.2)

| Method | keep 10% | keep 5% | keep 2% |
|---|---|---|---|
| Loukas/SCAL | 13.9 ± 2.2 | 12.9 ± 1.6 | 12.2 ± 1.5 |
| GB-struct | 13.8 ± 2.1 | 13.0 ± 1.8 | 12.0 ± 1.4 |
| SGBGC-keep (20/class) | 14.7 ± 1.1 | 14.3 ± 1.3 | 13.5 ± 1.6 |
| Oracle O | 13.4 ± 1.3 | 12.8 ± 1.3 | 12.3 ± 1.8 |
| Oracle O-sep | 15.2 ± 1.6 | 13.6 ± 1.9 | 13.3 ± 1.7 |

### Amazon-ratings (full graph: 32.1 ± 4.3)

| Method | keep 10% | keep 5% | keep 2% |
|---|---|---|---|
| Loukas/SCAL | 30.9 ± 4.5 | 31.1 ± 4.5 | 31.8 ± 2.9 |
| GB-struct | 27.6 ± 3.7 | 28.6 ± 4.5 | 31.4 ± 4.7 |
| SGBGC-keep (20/class) | 28.4 ± 3.4 | 28.2 ± 3.8 | 30.1 ± 4.6 |
| Oracle O | 30.3 ± 4.3 | 31.3 ± 5.4 | 31.6 ± 5.5 |
| Oracle O-sep | 31.5 ± 3.6 | 31.2 ± 4.5 | 30.8 ± 4.5 |

