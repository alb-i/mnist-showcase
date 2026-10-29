#!/usr/bin/env bash
# E2b (D14): SGD with Armijo backtracking line search (start step lr = 1, c = 1e-4, halving),
# own NumPy MLP and PyTorch MLP, mini-batch 128, 20 epochs each.
# Start in the tmux window "train":   scripts/run_sgd_ls.sh
#   scripts/run_sgd_ls.sh --full-batch   gradient descent on the whole training set: every step
#                                        uses the gradient and loss over all 54000 images (one
#                                        step per epoch, default 500 epochs = 500 steps)
# Further arguments go to both runs, e.g. --epochs 300 (does SGD still collapse?), --lr 4,
# --device gpu (NumPy net on the GPU via CuPy; the PyTorch run uses the GPU anyway).
# Results: runs <time>-numpy_mlp-sgd_ls[-fullbatch] and <time>-mlp-sgd_ls[-fullbatch].
set -euo pipefail
cd "$(dirname "$0")/.."
full=() numpy_args=() torch_args=()
while (($#)); do
    case "$1" in
        --full-batch) full=(--full-batch --epochs 500) ;;  # a later --epochs N wins
        --device) numpy_args+=("$1" "$2"); shift ;;        # NumPy/CuPy only
        --device=*) numpy_args+=("$1") ;;
        *) numpy_args+=("$1"); torch_args+=("$1") ;;
    esac
    shift
done
micromamba run -n mnist-showcase python -m digits.numpy_net.train --optimizer sgd_ls \
    "${full[@]}" "${numpy_args[@]}"
micromamba run -n mnist-showcase train configs/mlp.yaml --optimizer sgd_ls --set amp=false \
    "${full[@]}" "${torch_args[@]}"
