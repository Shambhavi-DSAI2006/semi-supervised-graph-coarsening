# Stage 7 research log

## 2026-11-14 — setup, before the protocol

- Cloned CGC's official code at commit 08ff04dc0d23110ff53445438f30bcff887e7c31 (github.com/XYGaoG/CGC) and ran it unchanged on Cora at its published ratio 0.026: 82.2 (paper: 82.3 ± 1.3).
- Wrote `cgc_adapter.py`, which repeats `main.py`'s call sequence on our data objects. Same Cora run with CGC's trainer: 81.7 / 82.7 / 82.2, so it matches. With the Stage 2–6 trainer: 80.7, so CGC needs its own trainer as an option.
- Size limit confirmed by dry run (Cora, seed 99, random split):
  - 1 label/class at rate 1: 7 nodes, 0 edges;
  - 5/class: 35 nodes, 10 edges;
  - 20/class: 140 nodes, 48 edges;
  - rate 2 crashes in all three, because the per-class clustering cannot ask for more centres than there are training nodes.
- Read CGC's code: condensed nodes are built only from training nodes, and the augmentation step resamples training nodes with 1-hop features. Hyperparameters (alpha, adj_T, tau, aug_ratio, dropout, epochs) are set per dataset and ratio in `scr/para.py`.
- Timing per full-graph evaluation (2 CPUs):

  | Dataset | Time |
  |---|---|
  | Cora | 6 ms |
  | Citeseer | 14 ms |
  | PubMed | 26 ms |
  | Coauthor-CS | 168 ms |
  | Amazon-Photo | 59 ms |
  | Amazon-Computers | 155 ms |
  | Roman-empire | 24 ms |
  | Amazon-ratings | 49 ms |
  | Coauthor-Physics | 401 ms |

  Physics is dropped and the two Amazon co-purchase graphs are added (fetched from shchur/gnn-benchmark, as Coauthor-CS was in Stage 1).
- PubMed timing (5 labels/class): K2 coarsening to 1% 1.7 s; CGC condensation 0.6 s; trainer S 6.6 s; CGC's trainer with per-epoch evaluation 126 s. So trainer C checks validation every 10 epochs.
- Bonsai: like CGC, it builds the condensed graph from training-set computation trees. Official code not located yet; optional.
