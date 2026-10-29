#!/usr/bin/env bash
# E2 (plan 7.2/8.2): own NumPy MLP 784-256-128-10, SGD with momentum vs. Adam.
# Start in the tmux window "train":   scripts/run_e2.sh
# Extra arguments go to both runs, e.g. scripts/run_e2.sh --epochs 30
# On the GPU (CuPy):                  scripts/run_e2.sh --device gpu   (runs get the suffix -gpu)
# Results: runs <time>-numpy_mlp-sgd and <time>-numpy_mlp-adam in $RUNS_DIR/mnist/.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { micromamba run -n mnist-showcase python -m digits.numpy_net.train "$@"; }
run --optimizer sgd "$@"            # configs/numpy_mlp.yaml: lr 0.1, momentum 0.9
run --optimizer adam --lr 1e-3 "$@"
