#!/usr/bin/env bash
# E3 (plan 7.3/8.2): PyTorch MLP 784-256-128-10, same settings as E2 (SGD vs. Adam). Same numbers?
# Start in the tmux window "train":   scripts/run_e3.sh
# Extra arguments go to both runs, e.g. scripts/run_e3.sh --epochs 30 or --set amp=false
# Results: runs <time>-mlp-sgd and <time>-mlp-adam in $RUNS_DIR/mnist/.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { micromamba run -n mnist-showcase train configs/mlp.yaml "$@"; }
run "$@"                            # configs/mlp.yaml: SGD lr 0.1, momentum 0.9, 20 epochs
run --optimizer adam --lr 1e-3 "$@"
