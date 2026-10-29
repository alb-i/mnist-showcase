#!/usr/bin/env bash
# E6 (plan 8.2): ablation on the best E5 setting. Five runs: the full config, then one component
# switched off at a time: augmentation, BatchNorm, dropout, label smoothing. What actually helps?
# 120 epochs each, without early stopping (D16).
# Start in the tmux window "train" with the best E5 learning rate and schedule, e.g.
#   scripts/run_e6.sh --lr 1e-2 --set scheduler=onecycle
# Results: runs <time>-e6-<variant> in $RUNS_DIR/mnist/; table:
#   micromamba run -n mnist-showcase python -m digits.summarize e6- --since <start of the run>
set -euo pipefail
cd "$(dirname "$0")/.."
start=$(date +%Y%m%d-%H%M%S)
run() { micromamba run -n mnist-showcase train configs/cnn_reg.yaml \
  --epochs 120 --set early_stopping_patience=null "$@"; }
run --name e6-full "$@"
run --name e6-no_aug --set augment=null "$@"
run --name e6-no_bn --set model_args.batchnorm=false "$@"
run --name e6-no_dropout --set model_args.dropout_conv=0 --set model_args.dropout_fc=0 "$@"
run --name e6-no_ls --set label_smoothing=0 "$@"
micromamba run -n mnist-showcase python -m digits.summarize e6- --since "$start"
