"""Original vs. curated data (experiment E8, plan 8.2 and 10.3, Phase 7 "Done when").

    micromamba run -n mnist-showcase python -m digits.curation_report --since <start of run_e8>
    micromamba run -n mnist-showcase python -m digits.curation_report --curated <run-id> ...

Two tables for the current curation/decisions.csv (its dataset version):
  data      per split (train = all 60 000 training images incl. the val split, test): images in
            the original data, reviewed (with a decision), keep / relabel / drop / unsure, images
            in the curated data.
  accuracy  per run of the groups "original" (trained on the original data: the E7 runs,
            ORIGINAL_RUNS) and "curated" (E8: the finished runs e8-* with an id >= --since), and
            mean +- std per group: the run's own val accuracy, and the test accuracy on the
            original test set (all 10 000 images, original labels) and on the curated test set
            (without the dropped images, the relabelled ones with their new label).

Both test views use the same predictions of a run, predictions/test.parquet (a prediction does
not depend on the label). A run trained on data with dropped test images has no prediction for
them; these few images are predicted from its checkpoint (best.pt, or the model of last.pt with
checkpoint: last) on the CPU.

The original test set stays the reference (plan 10.3: never tune on the curated test set). The
candidates were found with the cross-validation fold models (digits.label_issues), not with the
E7/E8 models, so the curated view of these models is not inflated by the search.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from digits.data import apply_overlay, dataset_version, load_split, read_decisions, sample_ids
from digits.models import build_model
from digits.runlog import runs_dir
from digits.summarize import find_run, run_name
from digits.train import Normalize, evaluate

# E7: cnn_reg, lr 1e-2 OneCycle, 120 epochs, seeds 0-2, original data (the 2nd run of each name)
ORIGINAL_RUNS = ["e7-seed0#2", "e7-seed1#2", "e7-seed2#2"]
CURATED_PREFIX = "e8-"
ACTIONS = ("keep", "relabel", "drop", "unsure")


def data_summary(splits: dict[str, tuple[np.ndarray, np.ndarray]], decisions: dict) -> list[dict]:
    """Per split (ids, original labels): counts of the original and the curated data."""
    rows = []
    for split, (ids, labels) in splits.items():
        known = set(ids)
        actions = Counter(d["action"] for sid, d in decisions.items() if sid in known)
        keep, _ = apply_overlay(ids, labels, decisions)
        rows.append({"split": split, "original": len(ids), "reviewed": sum(actions.values()),
                     **{a: actions[a] for a in ACTIONS}, "curated": int(keep.sum())})  # fmt: skip
    return rows


def predict_from_checkpoint(run_dir: Path, images: np.ndarray) -> np.ndarray:
    """Predicted classes of images (N, 28, 28) uint8 with the run's final model, on the CPU."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    ckpt_dir = run_dir / "checkpoints"
    if config.get("checkpoint", "best") == "last":
        state = torch.load(ckpt_dir / "last.pt", map_location="cpu")["model"]
    else:
        state = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model = build_model(config)
    model.load_state_dict(state)
    x = torch.as_tensor(images[:, None])
    _, _, probs = evaluate(model, x, torch.zeros(len(x), dtype=torch.int64), Normalize(config),
                           amp=False)  # fmt: skip
    return probs.argmax(1)


def run_predictions(run_dir: Path, ids: np.ndarray, images: np.ndarray) -> tuple[np.ndarray, int]:
    """Predicted class for each test id (from predictions/test.parquet, missing ones from the
    checkpoint) and the number of predictions taken from the checkpoint."""
    frame = pd.read_parquet(run_dir / "predictions" / "test.parquet", columns=["id", "pred"])
    unknown = set(frame["id"]) - set(ids)
    if unknown:
        raise ValueError(f"{run_dir.name}: predictions for unknown test ids, e.g. {min(unknown)}")
    pred = pd.Series(frame["pred"].to_numpy(), index=frame["id"]).reindex(ids)
    missing = pred.isna().to_numpy()
    if missing.any():
        pred[missing] = predict_from_checkpoint(run_dir, images[missing])
    return pred.to_numpy(np.int64), int(missing.sum())


def run_row(run_dir: Path, test: tuple, decisions: dict) -> dict:
    """Accuracies of one run: own val, original test, curated test."""
    ids, images, labels = test
    pred, inferred = run_predictions(run_dir, ids, images)
    keep, curated = apply_overlay(ids, labels, decisions)
    val = pd.read_parquet(run_dir / "predictions" / "val.parquet", columns=["label", "pred"])
    meta = json.loads((run_dir / "meta.json").read_text())
    return {
        "run_id": run_dir.name,
        "dataset_version": meta.get("dataset_version"),
        "val_acc": float((val["label"] == val["pred"]).mean()),
        "test_original": float((pred == labels).mean()),
        "errors_original": int((pred != labels).sum()),
        "test_curated": float((pred[keep] == curated[keep]).mean()),
        "errors_curated": int((pred[keep] != curated[keep]).sum()),
        "n_curated": int(keep.sum()),
        "inferred": inferred,
    }


def compare(groups: dict[str, list[Path]], test: tuple, decisions: dict) -> dict:
    """Rows per run (with key group) and mean +- std (None for one run) per group and metric."""
    rows, stats = [], {}
    for group, run_dirs in groups.items():
        group_rows = [{"group": group, **run_row(d, test, decisions)} for d in run_dirs]
        rows += group_rows
        stats[group] = {}
        for key in ("val_acc", "test_original", "test_curated"):
            values = [r[key] for r in group_rows]
            std = statistics.stdev(values) if len(values) > 1 else None
            stats[group][key] = (statistics.mean(values), std)
    return {"runs": rows, "groups": stats}


def curated_runs(root: Path, since: str = "", prefix: str = CURATED_PREFIX) -> list[Path]:
    """Finished runs named <prefix>* with an id >= since, oldest first."""
    out = []
    for d in sorted(root.iterdir()):
        if not (d / "meta.json").is_file() or d.name.count("-") < 2 or d.name < since:
            continue
        status = json.loads((d / "meta.json").read_text()).get("status")
        if run_name(d.name).startswith(prefix) and status == "finished":
            out.append(d)
    return out


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def format_report(summary: list[dict], comparison: dict, version: str) -> str:
    lines = [f"data (decisions.csv, dataset version {version})",
             f"{'split':<6} {'original':>8} {'reviewed':>8} {'keep':>5} {'relabel':>7} "
             f"{'drop':>5} {'unsure':>6} {'curated':>8}"]  # fmt: skip
    for r in summary:
        lines.append(
            f"{r['split']:<6} {r['original']:>8} {r['reviewed']:>8} {r['keep']:>5} "
            f"{r['relabel']:>7} {r['drop']:>5} {r['unsure']:>6} {r['curated']:>8}"
        )
    lines += ["", "accuracy in % (test errors in brackets)",
              f"{'group':<9} {'run':<28} {'data':<8} {'val':>6} {'test orig.':>13} "
              f"{'test cur.':>13}"]  # fmt: skip
    for r in comparison["runs"]:
        note = f"  ({r['inferred']} test images from the checkpoint)" if r["inferred"] else ""
        orig = f"{pct(r['test_original'])} ({r['errors_original']})"
        cur = f"{pct(r['test_curated'])} ({r['errors_curated']})"
        lines.append(f"{r['group']:<9} {r['run_id']:<28} {r['dataset_version'] or '-':<8} "
                     f"{pct(r['val_acc']):>6} {orig:>13} {cur:>13}{note}")  # fmt: skip
    lines.append("")
    for group, stats in comparison["groups"].items():
        cells = []
        for key in ("val_acc", "test_original", "test_curated"):
            mean, std = stats[key]
            cells.append(pct(mean) + ("" if std is None else f" +- {pct(std)}"))
        n = sum(r["group"] == group for r in comparison["runs"])
        lines.append(f"{group:<9} mean ({n} runs): val {cells[0]}, test original {cells[1]}, "
                     f"test curated {cells[2]}")  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--original", nargs="+", default=ORIGINAL_RUNS,
        help="run ids, names (name#k) or folders trained on the original data (default: E7)",
    )  # fmt: skip
    parser.add_argument(
        "--curated",
        nargs="+",
        help=f"runs trained on the curated data (default: {CURATED_PREFIX}*)",
    )
    parser.add_argument("--since", default="", help=f"only {CURATED_PREFIX}* runs with id >= this")
    parser.add_argument("--root", type=Path, help="runs folder (default: $RUNS_DIR/mnist)")
    parser.add_argument("--decisions", type=Path, help="default: curation/decisions.csv")
    args = parser.parse_args(argv)
    root = args.root or runs_dir()

    def resolve(runs: list[str]) -> list[Path]:
        return [Path(r) if Path(r).is_dir() else find_run(root, r) for r in runs]

    curated = resolve(args.curated) if args.curated else curated_runs(root, args.since)
    if not curated:
        parser.error(f"no finished {CURATED_PREFIX}* runs in {root}")
    decisions = read_decisions(args.decisions)
    version = dataset_version(args.decisions)
    splits = {}
    for split in ("train", "test"):
        images, labels = load_split(split)
        splits[split] = (sample_ids(split, len(labels)), images, labels)
    summary = data_summary({s: (ids, labels) for s, (ids, _, labels) in splits.items()}, decisions)
    comparison = compare({"original": resolve(args.original), "curated": curated},
                         splits["test"], decisions)  # fmt: skip
    print(format_report(summary, comparison, version))
    stale = [r["run_id"] for r in comparison["runs"]
             if r["group"] == "curated" and r["dataset_version"] != version]  # fmt: skip
    if stale:
        print(f"\nnote: {', '.join(stale)} trained on another dataset version than {version}")


if __name__ == "__main__":
    main()
