#!/usr/bin/env bash
# Run every experiment in the reproduction. Run from the repository root:
#   bash scripts/run_all.sh            # both epoch settings (as reported)
#   EPOCHS="60" bash scripts/run_all.sh  # code-default setting only (faster)
# Results are appended to results/results.jsonl; progress goes to results/run.log.
# Runtime on a 2-core CPU: ~3 hours for both settings, dominated by Co-CS.
set -euo pipefail
EPOCHS="${EPOCHS:-60 200}"     # 60 = authors' code default, 200 = value stated in the paper
P="python repro.py"
LOG=results/run.log
mkdir -p results

for E in $EPOCHS; do
  # Table 1: fixed coarsening ratios
  $P table1 cora     0.5 0.3 0.1  --epochs "$E" | tee -a "$LOG"
  $P table1 citeseer 0.5 0.3 0.1  --epochs "$E" | tee -a "$LOG"
  $P table1 cs       0.3 0.1 0.05 --epochs "$E" | tee -a "$LOG"
  # Table 2: adaptive ratio and full-graph GCN
  for D in cora citeseer cs; do
    $P adaptive "$D" --epochs "$E" | tee -a "$LOG"
    $P full     "$D" --epochs "$E" | tee -a "$LOG"
  done
  # Fig. 4: label noise on Citeseer at ratio 0.3
  $P noise citeseer 0.05 0.10 0.15 0.20 --epochs "$E" | tee -a "$LOG"
done
echo "All experiments finished." | tee -a "$LOG"
