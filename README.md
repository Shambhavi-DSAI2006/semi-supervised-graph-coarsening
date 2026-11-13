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
| 2. Diagnose | Measure how label-aware coarsening behaves as labels become scarce | Planned |
| 3. Method | Design a coarsening method that uses sparse labels and unlabelled structure | Planned |
| 4. Evaluate | Compare against supervised and unsupervised coarsening at realistic label rates and scales | Planned |

## Stage 1 at a glance

SGBGC's results reproduce on Cora, Citeseer and Co-CS, but only with the code's default of 60 training epochs rather than the 200 stated in the paper. Full results, deviations and instructions: [reproduction/README.md](reproduction/README.md).

## Repository layout

- **reproduction/**: Stage 1, the SGBGC reproduction (scripts, raw results, tables, figures)

## Key references

- Xia et al., *Graph Coarsening via Supervised Granular-Ball for Scalable Graph Neural Network Training*, AAAI-25. Code: [mxjun17/GBNC](https://github.com/mxjun17/GBNC)
- Xia et al., *Efficient and Adaptive Graph Coarsening via Granular-ball Computing* (GBGC, unsupervised), [arXiv:2506.19224](https://arxiv.org/abs/2506.19224), 2025
- Huang et al., *Scaling Up Graph Neural Networks via Graph Coarsening*, KDD 2021

## Author

Shambhavi ([@Shambhavi-DSAI2006](https://github.com/Shambhavi-DSAI2006))

## License

MIT for the code in this repository. Third-party code fetched by the scripts remains under its authors' terms.
