#!/usr/bin/env bash
# E1 (plan 7.1/8.2): logistic regression and k-NN baselines on flattened pixels.
# Start in the tmux window "train":   scripts/run_e1.sh            (both baselines)
#                                     scripts/run_e1.sh --only knn (just one)
# Results: one run per model in $RUNS_DIR/mnist/, visible in the workbench.
set -euo pipefail
cd "$(dirname "$0")/.."
exec micromamba run -n mnist-showcase python -m digits.baselines "$@"
