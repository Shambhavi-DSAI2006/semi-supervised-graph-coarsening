# Stage 2 research log

Every experiment, including failed or abandoned ones, newest last. Dates are local (UTC+3).

## 2026-11-13 — Setup

- Wrote [PROTOCOL.md](PROTOCOL.md) before producing any Stage 2 result.
- Baseline code check:
  - SCAL (Huang et al., KDD 2021): official code available (MIT). It uses Loukas's coarsening library.
  - Loukas graph-coarsening library (commit 1d7aad0): runs on current Python after replacing removed NumPy/SciPy aliases (`sp.newaxis`, `np.int`, `np.bool`, `np.float`). No algorithmic change. On Cora's largest component it reaches the target ratios exactly (0.300 and 0.100) in about 1.6 s.
  - GBGC (Xia et al., 2025): only the appendix is public (code link is an anonymous mirror). The structure-only granular-ball baseline (M5) is therefore our own.
- Earlier informal probe (before this protocol, not used for any decision): with SGBGC on Cora at ratio 0.3, at ≤ 20% known labels the coarse graph contained only the labelled nodes. That probe used a different split and evaluation, so its numbers are not comparable with this stage.
