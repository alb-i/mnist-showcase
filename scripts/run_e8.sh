#!/usr/bin/env bash
# E8 (plan 8.2, 10.3): retrain on the curated data. The E7 config (cnn_reg, lr 1e-2 OneCycle,
# 120 epochs without early stopping, deterministic, seeds 0-2) on the data with the overlay
# curation/decisions.csv applied (drops removed, relabels changed, also in val and test); the
# runs record its dataset version. About 3.6 min per seed.
# decisions.csv must be committed (plan 4.4: one commit per curation round), so that the run's git
# commit reproduces the dataset.
# Start in the tmux window "train":
#   scripts/run_e8.sh
# Results: runs <time>-e8-seed<n> in $RUNS_DIR/mnist/, then mean +- std and the table original
# (E7) vs. curated (E8) on the original and the curated test set (digits.curation_report).
set -euo pipefail
cd "$(dirname "$0")/.."
if ! git diff --quiet HEAD -- curation/decisions.csv; then
  echo "curation/decisions.csv has uncommitted changes: commit the curation round first" >&2
  exit 1
fi
start=$(date +%Y%m%d-%H%M%S)
for seed in 0 1 2; do
  micromamba run -n mnist-showcase train configs/cnn_reg.yaml --lr 1e-2 --set scheduler=onecycle \
    --epochs 120 --set early_stopping_patience=null --seed "$seed" --deterministic \
    --name "e8-seed$seed" "$@"
done
micromamba run -n mnist-showcase python -m digits.summarize e8- --since "$start"
micromamba run -n mnist-showcase python -m digits.curation_report --since "$start"
