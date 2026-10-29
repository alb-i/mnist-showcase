#!/usr/bin/env bash
# E5 (plan 8.2): regularised CNN (configs/cnn_reg.yaml: AdamW, BN, dropout, augmentation, label
# smoothing 0.1), LR sweep with 5 values x 2 schedules (OneCycle, cosine) = 10 runs of 120 epochs
# without early stopping (D16: patience 5 on the val loss stopped runs before the schedule ended).
# Start in the tmux window "train":   scripts/run_e5.sh
# Extra arguments go to every run, e.g. scripts/run_e5.sh --seed 1
# Results: runs <time>-e5-lr<lr>-<schedule> in $RUNS_DIR/mnist/; pick the best by val accuracy:
#   micromamba run -n mnist-showcase python -m digits.summarize e5- --since <start of the run>
set -euo pipefail
cd "$(dirname "$0")/.."
start=$(date +%Y%m%d-%H%M%S)
for scheduler in onecycle cosine; do
  for lr in 3e-4 1e-3 3e-3 1e-2 3e-2; do
    micromamba run -n mnist-showcase train configs/cnn_reg.yaml --lr "$lr" \
      --set scheduler="$scheduler" --epochs 120 --set early_stopping_patience=null \
      --name "e5-lr$lr-$scheduler" "$@"
  done
done
micromamba run -n mnist-showcase python -m digits.summarize e5- --since "$start"
