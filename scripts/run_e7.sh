#!/usr/bin/env bash
# E7 (plan 8.2): the best config (from E5/E6) with 3 seeds and deterministic kernels (plan 8.1:
# "deterministic cuDNN kernels for final runs"). Mean +- std of the accuracy.
# 120 epochs each, without early stopping (D16).
# Start in the tmux window "train" with the best settings, e.g.
#   scripts/run_e7.sh --lr 1e-2 --set scheduler=onecycle            (plus E6 switches if better)
# Results: runs <time>-e7-seed<n> in $RUNS_DIR/mnist/; mean +- std:
#   micromamba run -n mnist-showcase python -m digits.summarize e7- --since <start of the run>
set -euo pipefail
cd "$(dirname "$0")/.."
start=$(date +%Y%m%d-%H%M%S)
for seed in 0 1 2; do
  micromamba run -n mnist-showcase train configs/cnn_reg.yaml --seed "$seed" --deterministic \
    --epochs 120 --set early_stopping_patience=null --name "e7-seed$seed" "$@"
done
micromamba run -n mnist-showcase python -m digits.summarize e7- --since "$start"
