# Stage 5 research log

Every experiment, including failed or abandoned ones, newest last. Dates are local (UTC+3).

## 2026-11-14 — Setup

- Requirement from Shambhavi: the method must be **general** — usable with any coarsening algorithm, not tied to one. Stage 4's approach (re-run Loukas at a smaller target) is not: it needs a base method that can be re-run with a different target, and it coarsens the rest of the graph harder.
- Built `protect.py`: separate clashing super-nodes, then rebalance by merging adjacent label-compatible super-nodes back to the original count. It uses only the partition, the adjacency and the training labels.
- **Feasibility checks** (seed 99, public splits, no accuracy computed):
  - On Loukas, GB-struct and k-means over Cora, Citeseer and PubMed at 5%, 2%, 1%: protection always returned exactly the input number of super-nodes, kept every node, and reduced lost training labels to 0 (from up to 60 for Loukas and GB-struct and 22 for k-means on Cora). It takes under 0.1 s per partition, versus re-running Loukas (up to 77 s on PubMed) in Stage 4.
  - **Dropped as base methods:** Stage 2's feature clustering with connectivity splitting could not control its size (PubMed: 1,227 instead of 985 super-nodes at 5%, **1** instead of 197 at 1%); heavy-edge matching (Loukas library) stalled at 412 super-nodes on Cora for every target (101–201) and did not finish on PubMed within 15 minutes. Plain feature k-means hits every target exactly and was chosen as the third, structure-free family.
- Wrote [PROTOCOL.md](PROTOCOL.md) before producing any Stage 5 result.
