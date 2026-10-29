"""Sanity check before a long run (plan 9, D17): can the model fit one batch to loss ~ 0?

    micromamba run -n mnist-showcase python -m digits.sanity overfit configs/cnn_reg.yaml
    micromamba run -n mnist-showcase python -m digits.sanity overfit configs/mlp.yaml --lr 1e-3

Trains the configured model on one fixed batch of training images (default 128, chosen with the
config seed) for `--steps` optimiser steps: no augmentation, no label smoothing, constant lr, no
early stopping; dropout and weight decay stay as configured. val and test are the same batch, so
the val curve shows the fit in eval mode. Logged as a normal run (name sanity-overfit), so the
workbench shows it. PASS if the last logged training loss is below 0.01, else exit code 1: then
model, loss, optimiser or lr are broken.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from digits.data import load_dataset
from digits.train import load_config, train

PASS_LOSS = 0.01


def overfit(
    config: dict, dataset: dict, steps: int = 300, batch_size: int = 128, root: Path | None = None
) -> tuple[bool, float, Path]:
    """Returns (passed, last logged training loss, run folder)."""
    ids, images, labels = dataset["train"]
    idx = np.random.default_rng(config["seed"]).choice(len(labels), batch_size, replace=False)
    batch = (ids[idx], images[idx], labels[idx])
    config = {
        **config,
        "augment": None,
        "label_smoothing": 0.0,
        "scheduler": None,
        "early_stopping_patience": None,
        "full_batch": False,
        "batch_size": batch_size,
        "epochs": steps,  # one step per epoch
        "log_every": 10,
    }
    results = train(config, {s: batch for s in ("train", "val", "test")}, root, "sanity-overfit")
    run_dir = results["run_dir"]
    records = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    loss = [r["loss"] for r in records if r["split"] == "train" and "loss" in r][-1]
    return loss < PASS_LOSS, loss, run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="check", required=True)
    p = sub.add_parser("overfit", help="fit one batch of training images")
    p.add_argument("config", type=Path)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--root", type=Path, help="runs folder (default: $RUNS_DIR/mnist)")
    args = parser.parse_args(argv)
    config = load_config(args.config, lr=args.lr, overrides=args.set)
    passed, loss, run_dir = overfit(config, load_dataset(), args.steps, args.batch_size, args.root)
    print(f"{'PASS' if passed else 'FAIL'}: training loss {loss:.4g} after {args.steps} steps "
          f"on {args.batch_size} images (threshold {PASS_LOSS}), run {run_dir.name}")  # fmt: skip
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
