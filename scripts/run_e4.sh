#!/usr/bin/env bash
# E4 (plan 7.3/8.2): small CNN conv(32)-conv(64)-pool-dropout-fc(128)-fc(10), Adam, 15 epochs.
# Start in the tmux window "train":   scripts/run_e4.sh          (expect about 99.2 % test)
# Extra arguments go to the run, e.g. scripts/run_e4.sh --seed 1
# Results: run <time>-cnn_small-adam in $RUNS_DIR/mnist/.
set -euo pipefail
cd "$(dirname "$0")/.."
exec micromamba run -n mnist-showcase train configs/cnn_small.yaml "$@"
