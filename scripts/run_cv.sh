#!/usr/bin/env bash
# Out-of-fold predictions (plan 10.2.1, D18): 5 fold models with the E7 config (cnn_reg, lr 1e-2
# OneCycle, 120 epochs, no early stopping, deterministic, seed 0). Model k trains on the 48 000
# training images outside fold k (configs/cv_folds.txt) and keeps its last epoch; its
# predictions/val.parquet holds the out-of-fold predictions of fold k, predictions/test.parquet
# the test set (averaged over the 5 models later). About 3.4 min per fold.
# --no-ls: the same without label smoothing (D22), runs cv-nols-fold<k>, to compare confident
# learning with and without label smoothing.
# Start in the tmux window "train":
#   scripts/run_cv.sh                  # all folds
#   scripts/run_cv.sh 3 4              # only these folds (e.g. after an interruption)
#   scripts/run_cv.sh --no-ls [folds]
# Results: runs <time>-cv-fold<k> (or cv-nols-fold<k>) in $RUNS_DIR/mnist/; then combine them and
# find the label issues (run folder <time>-cv-oof or cv-nols-oof):
#   micromamba run -n mnist-showcase python -m digits.label_issues collect [--prefix cv-nols]
set -euo pipefail
cd "$(dirname "$0")/.."
prefix=cv
extra=()
if [ "${1:-}" = "--no-ls" ]; then
  prefix=cv-nols
  extra=(--set label_smoothing=0.0)
  shift
fi
folds=("$@")
[ ${#folds[@]} -eq 0 ] && folds=(0 1 2 3 4)
for fold in "${folds[@]}"; do
  micromamba run -n mnist-showcase train configs/cnn_reg.yaml --lr 1e-2 --set scheduler=onecycle \
    --epochs 120 --set early_stopping_patience=null --set checkpoint=last --seed 0 \
    --deterministic --fold "$fold" --name "$prefix-fold$fold" "${extra[@]}"
done
