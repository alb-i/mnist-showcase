"""Cross-validated predictions and confident learning (plan 10.2, D18, D19, D21).

    micromamba run -n mnist-showcase python -m digits.label_issues collect     # after run_cv.sh
    micromamba run -n mnist-showcase python -m digits.label_issues collect --prefix cv-nols

collect takes the newest finished run of each <prefix>-fold<k> (or the run folders given with
--runs), checks that their folds cover every training image exactly once, and writes a new run
folder <time>-<prefix>-oof (digits.runlog) with
  predictions/train.parquet  out-of-fold predictions of all training images (column fold),
  predictions/test.parquet   mean probabilities of the fold models on the test set,
  label_issues.parquet       one row per training and test image (see find_label_issues).

Confident learning (Northcutt, Jiang, Chuang: "Confident Learning: Estimating Uncertainty in
Dataset Labels", JAIR 2021), computed per data set (training images with their out-of-fold
probabilities, test images with the ensemble probabilities). With given labels y_i and predicted
probabilities p_i (no model has seen image i, or for the test set: no model has trained on it):

  threshold    t_j = mean of p_ij over the images labelled j (the class's average
               self-confidence);
  confident    the set {j : p_ij >= t_j} of classes image i is confidently predicted as; if it is
               not empty, the suggested class is the most probable class in it, else none;
  joint        C[k, j] = number of images labelled k with suggested class j (confident joint);
  calibrated   C~[k, j] = C[k, j] / sum_j C[k, j] * n_k, so that row k sums to the number n_k of
               images labelled k; Q = C~ / n estimates the joint distribution of given and true
               label, and n (1 - trace Q) = sum of the off-diagonal C~ the number of label errors.

An image is flagged if it has a suggested class and that class differs from its label (an
off-diagonal entry of C): the model is confidently sure of a different class. Score: the
self-confidence p_i,y_i (low = suspicious).

Review candidates (D22): the flagged images plus every image with self-confidence below
LOW_CONFIDENCE (the model gives the label less than even odds), ranked by self-confidence, lowest
first. With label smoothing all confidences are capped near 0.91 and the thresholds t_j sit at
that ceiling, so the flag alone finds hardly anything.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from digits.evaluate import N_CLASSES, classification_metrics, predictions_frame
from digits.runlog import RunLogger, runs_dir

PROB_COLUMNS = [f"p{k}" for k in range(N_CLASSES)]
LOW_CONFIDENCE = 0.5


def per_class_thresholds(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """t_j = mean of p_ij over the images labelled j; inf for a class without images (no image
    can then be confidently that class)."""
    k = probs.shape[1]
    counts = np.bincount(labels, minlength=k)
    sums = np.bincount(labels, weights=probs[np.arange(len(labels)), labels], minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.inf)


def suggested_classes(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per image the most probable class among those with p_ij >= t_j, or -1 if there is none."""
    confident = probs >= thresholds[None, :]
    masked = np.where(confident, probs, -np.inf)
    return np.where(confident.any(axis=1), masked.argmax(axis=1), -1)


def confident_joint(labels: np.ndarray, suggested: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    """C[given label, suggested class], counting only images with a suggested class."""
    joint = np.zeros((k, k), dtype=np.int64)
    has = suggested >= 0
    np.add.at(joint, (labels[has], suggested[has]), 1)
    return joint


def calibrate_joint(joint: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """C~: each row of C rescaled to the number of images with that label (rows without any
    confident image stay 0)."""
    counts = np.bincount(labels, minlength=len(joint))
    rows = joint.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rows > 0, joint / rows * counts[:, None], 0.0)


def find_label_issues(
    ids: np.ndarray, labels: np.ndarray, probs: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """Per image: id, label, pred (argmax), suggested (-1: none), self_confidence, margin
    (p_i,y_i - max over j != y_i of p_ij), flagged, candidate (flagged or self-confidence below
    LOW_CONFIDENCE), rank (1 = most suspicious candidate, <NA> if not a candidate). Summary:
    thresholds, confident joint, calibrated joint, numbers of flagged images and candidates,
    the estimated number of label errors."""
    labels = labels.astype(np.int64)
    thresholds = per_class_thresholds(labels, probs)
    suggested = suggested_classes(probs, thresholds)
    joint = confident_joint(labels, suggested, probs.shape[1])
    calibrated = calibrate_joint(joint, labels)
    rows = np.arange(len(labels))
    self_conf = probs[rows, labels]
    others = probs.copy()
    others[rows, labels] = -np.inf
    flagged = (suggested >= 0) & (suggested != labels)
    candidate = flagged | (self_conf < LOW_CONFIDENCE)
    frame = pd.DataFrame({
        "id": ids,
        "label": labels,
        "pred": probs.argmax(axis=1),
        "suggested": suggested,
        "self_confidence": self_conf,
        "margin": self_conf - others.max(axis=1),
        "flagged": flagged,
        "candidate": candidate,
    })  # fmt: skip
    order = np.flatnonzero(candidate)[np.argsort(self_conf[candidate], kind="stable")]
    rank = pd.array([pd.NA] * len(labels), dtype="Int64")
    rank[order] = np.arange(1, len(order) + 1)
    frame["rank"] = rank
    summary = {
        "thresholds": thresholds.tolist(),
        "confident_joint": joint.tolist(),
        "calibrated_joint": calibrated.round(2).tolist(),
        "n": len(labels),
        "flagged": int(flagged.sum()),
        "candidates": int(candidate.sum()),
        "estimated_errors": float(calibrated.sum() - np.trace(calibrated)),
    }
    return frame, summary


def read_run(run_dir: Path) -> tuple[dict, dict]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    return config, meta


def latest_fold_runs(root: Path, n_folds: int, prefix: str = "cv") -> list[Path]:
    """Newest finished run named <prefix>-fold<k> for each k."""
    runs = []
    for fold in range(n_folds):
        found = sorted(
            d for d in root.glob(f"*-{prefix}-fold{fold}")
            if (d / "meta.json").exists()
            and json.loads((d / "meta.json").read_text()).get("status") == "finished"
        )  # fmt: skip
        if not found:
            raise FileNotFoundError(f"no finished run *-{prefix}-fold{fold} in {root}")
        runs.append(found[-1])
    return runs


def combine_folds(run_dirs: list[Path], folds: dict[str, int] | None = None):
    """Out-of-fold frame (all fold runs' val predictions, column fold) and the test ensemble
    (ids, labels, mean probabilities). Checks: each run has cv_fold and checkpoint last, the same
    dataset version, disjoint val ids (that belong to its fold, if `folds` is given), together
    the folds 0 .. k-1, the same test images."""
    oof, test_probs, test_ref, versions = [], [], None, set()
    for run_dir in run_dirs:
        config, meta = read_run(run_dir)
        fold = config.get("cv_fold")
        if fold is None or config.get("checkpoint") != "last":
            raise ValueError(
                f"{run_dir.name}: not a cross-validation run (cv_fold, checkpoint last)"
            )
        versions.add(meta.get("dataset_version"))
        val = pd.read_parquet(run_dir / "predictions" / "val.parquet")
        if folds is not None and any(folds[i] != fold for i in val["id"]):
            raise ValueError(f"{run_dir.name}: val ids outside fold {fold}")
        oof.append(val.assign(fold=fold))
        test = pd.read_parquet(run_dir / "predictions" / "test.parquet")
        if test_ref is None:
            test_ref = test[["id", "label"]]
        elif not test_ref.equals(test[["id", "label"]]):
            raise ValueError(f"{run_dir.name}: different test images")
        test_probs.append(test[PROB_COLUMNS].to_numpy(np.float64))
    if len(versions) > 1:
        raise ValueError(f"fold runs use different dataset versions: {sorted(versions)}")
    frame = pd.concat(oof, ignore_index=True).sort_values("id", ignore_index=True)
    if frame["id"].duplicated().any():
        raise ValueError("an image is in more than one fold")
    if sorted(frame["fold"].unique()) != list(range(len(run_dirs))):
        raise ValueError("the runs must cover the folds 0 .. k-1 once each")
    test_ids, test_labels = test_ref["id"].to_numpy(), test_ref["label"].to_numpy()
    return frame, (test_ids, test_labels, np.mean(test_probs, axis=0))


def collect(run_dirs: list[Path], root: Path | None = None, folds: dict[str, int] | None = None,
            prefix: str = "cv") -> Path:  # fmt: skip
    """Writes the <prefix>-oof run folder; returns it."""
    oof, (test_ids, test_labels, test_probs) = combine_folds(run_dirs, folds)
    config = {
        "model": "cv-ensemble",
        "source_runs": [d.name for d in run_dirs],
        "cv_folds": len(run_dirs),
        "label_issues": {
            "flag": "confident joint, off-diagonal",
            "candidates": f"flagged or self_confidence < {LOW_CONFIDENCE}",
            "rank": "self_confidence, lowest first",
        },
    }
    with RunLogger(config, name=f"{prefix}-oof", root=root) as run:
        oof.to_parquet(run.dir / "predictions" / "train.parquet", index=False)
        test = predictions_frame(test_ids, test_labels, test_probs)
        test.to_parquet(run.dir / "predictions" / "test.parquet", index=False)
        issues, summaries = [], {}
        for split, frame in (("train", oof), ("test", test)):
            labels = frame["label"].to_numpy()
            probs = frame[PROB_COLUMNS].to_numpy(np.float64)
            found, summary = find_label_issues(frame["id"].to_numpy(), labels, probs)
            issues.append(found.assign(split=split))
            summaries[split] = summary
            metrics = classification_metrics(labels, probs.argmax(axis=1))
            run.log(step=0, split="val" if split == "train" else "test",
                    loss=float(frame["loss"].mean()), acc=metrics["acc"],
                    macro_f1=metrics["macro_f1"], n=len(labels), flagged=summary["flagged"],
                    candidates=summary["candidates"],
                    estimated_errors=round(summary["estimated_errors"], 1))  # fmt: skip
        pd.concat(issues, ignore_index=True).to_parquet(
            run.dir / "label_issues.parquet", index=False
        )
        run.meta["label_issues"] = summaries
    return run.dir


def format_summary(run_dir: Path, top: int = 20) -> str:
    meta = json.loads((run_dir / "meta.json").read_text())
    issues = pd.read_parquet(run_dir / "label_issues.parquet")
    lines = [run_dir.name]
    for split, s in meta["label_issues"].items():
        lines.append(f"  {split}: {s['flagged']} of {s['n']} images flagged, "
                     f"{s['candidates']} candidates, "
                     f"estimated label errors {s['estimated_errors']:.1f}")  # fmt: skip
        ranked = issues[(issues["split"] == split) & issues["candidate"]].sort_values("rank")
        lines.append(ranked.head(top).to_string(index=False))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="combine the fold runs, find label issues")
    p.add_argument("--runs", type=Path, nargs="+", help="fold run folders (default: newest)")
    p.add_argument("--root", type=Path, help="runs folder (default: $RUNS_DIR/mnist)")
    p.add_argument("--prefix", default="cv", help="fold runs <prefix>-fold<k> (cv, cv-nols)")
    p.add_argument("--top", type=int, default=20, help="candidates to print per split")
    args = parser.parse_args(argv)
    from digits.data import cv_folds, split_config

    root = args.root or runs_dir()
    run_dirs = args.runs or latest_fold_runs(root, split_config()["cv_folds"], args.prefix)
    print(format_summary(collect(run_dirs, root, cv_folds(), args.prefix), args.top))


if __name__ == "__main__":
    main()
