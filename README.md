# Semi-Supervised Graph Coarsening

Research project on **graph coarsening when only a few node labels are available**, for scalable GNN training.

## Motivation

Graph coarsening shrinks a large graph into a smaller one, so a GNN can be trained on the small graph and applied to the original. Label-aware methods such as **SGBGC** (Xia et al., AAAI-25) use node labels to decide which nodes to merge, and report strong accuracy at high compression.

Our reproduction (Stage 1) shows these results rest on a **label-rich setting**: coarsening uses the labels of 80% of the nodes (the training and validation sets), and nodes in regions with no known label are discarded from the coarse graph.

Real graphs are usually labelled sparsely; standard semi-supervised benchmarks give only around 20 labels per class. This project asks how to coarsen well in that regime: using the few labels that exist without discarding the structure and features of unlabelled nodes.

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| 1. Reproduce | Reproduce SGBGC's main results with the authors' code | Done: [reproduction/](reproduction/) |
| 2. Diagnose | Measure how label-aware coarsening behaves as labels become scarce | Done: **pivot** - [`stage2/`](stage2/) |
| 3. Headroom | Measure how much labels could add to coarsening at strong compression and on heterophilous graphs | Done: weak target (one cell) - [`stage3/`](stage3/) |
| 4. Label separation | Test one method component: never merge training nodes of different classes | Done: **go**, modest evidence - [`stage4/`](stage4/) |
| 5. General wrapper | Test label protection as a plug-in for any coarsening method | Done: **inconclusive, not general** - [`stage5/`](stage5/) |
| 6. Label-guided k-means | Test whether few labels improve feature k-means coarsening | Done: **inconclusive** (helps PubMed only) - [`stage6/`](stage6/) |
| 7. Baseline gap | Compare against CGC, LAGC, SCAL, SGBGC, Bonsai at very few labels and on large graphs | Next |
| 8. Method and evaluation | Build and fully evaluate the method in the gap found | Planned |

## Stage 1 at a glance

SGBGC's results reproduce on Cora, Citeseer and Co-CS, but only with the code's default of 60 training epochs rather than the 200 stated in the paper. Full results, deviations and instructions: [reproduction/README.md](reproduction/README.md).

## Stage 2 at a glance

Pre-registered test on Cora and Citeseer at 5-20 labels per class. SGBGC's collapse under label scarcity comes from discarding unlabelled nodes; keeping them recovers 4-7 points. But once nothing is discarded, label-aware coarsening is no better than label-free coarsening (Loukas/SCAL) with the same training, so the pre-registered rule says **pivot**: the project needs a different angle on semi-supervised coarsening. Details and candidate directions: [`stage2/README.md`](stage2/README.md).

## Stage 3 at a glance

Pre-registered headroom test on Cora, Citeseer, Roman-empire and Amazon-ratings at 10%, 5% and 2% of nodes with 20 labels per class. Even with many labels, granular-ball coarsening beat label-free coarsening by 3+ points in only one setting (Cora at 2%), with weak evidence; the heterophilous graphs were too hard for the base GCN to be informative. An exploratory hint: at strong compression, keeping the few labelled nodes apart may matter more than having many labels. Details: [`stage3/README.md`](stage3/README.md).

## Stage 4 at a glance

Pre-registered test of one semi-supervised rule added to Loukas/SCAL coarsening: never let a super-node contain training nodes of more than one class. On new seeds and splits it passed the rule on Cora (+3.0 points on average) and PubMed (+2.0, narrowly), not on Citeseer; the largest, robust effect is Cora at 2% of nodes (+6 points). The rule costs budget elsewhere and can hurt when that cost is high, which points to the next design step. Details: [`stage4/README.md`](stage4/README.md).

## Stage 5 at a glance

Pre-registered test of a general "label protection" wrapper (split super-nodes that mix training labels, then rebalance to the same size) on three different coarsening methods. It is not general: it helps a weak structural coarsener (GB-struct, +3 to +4 points), barely changes Loukas, and hurts feature k-means (-2 points on Cora). Exploratory: plain k-means on smoothed features was the most accurate of the three base methods at every compression level. Details: [`stage5/README.md`](stage5/README.md).

## Stage 6 at a glance

Pre-registered test of label guidance for feature k-means coarsening. Label-propagation features helped on PubMed (+2.0 points, matching the full graph at 1-2% of nodes) but not clearly on Cora or Citeseer. A literature check during this stage found closely related published methods (LAGC, CGC, Bonsai), which are now required baselines. Details: [`stage6/README.md`](stage6/README.md).

## Repository layout

- **reproduction/**: Stage 1, the SGBGC reproduction (scripts, raw results, tables, figures)
- **stage2/**: Stage 2, the label-scarcity test (protocol, log, code, results)
- **stage3/**: Stage 3, the headroom test (protocol, log, code, results)
- **stage4/**: Stage 4, the label-separation test (protocol, log, code, results)
- **stage5/**: Stage 5, the general label-protection wrapper test (protocol, log, code, results)
- **stage6/**: Stage 6, the label-guided k-means test (protocol, log, code, results)

## Key references

- Xia et al., *Graph Coarsening via Supervised Granular-Ball for Scalable Graph Neural Network Training*, AAAI-25. Code: [mxjun17/GBNC](https://github.com/mxjun17/GBNC)
- Xia et al., *Efficient and Adaptive Graph Coarsening via Granular-ball Computing* (GBGC, unsupervised), [arXiv:2506.19224](https://arxiv.org/abs/2506.19224), 2025
- Huang et al., *Scaling Up Graph Neural Networks via Graph Coarsening*, KDD 2021

## Author

Shambhavi ([@Shambhavi-DSAI2006](https://github.com/Shambhavi-DSAI2006))

## License

MIT for the code in this repository. Third-party code fetched by the scripts remains under its authors' terms.
