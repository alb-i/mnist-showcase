"""Metrics, confusion matrix, calibration (plan 10.1).

    micromamba run -n mnist-showcase python -m digits.evaluate $RUNS_DIR/mnist/<run-id> [...]
    micromamba run -n mnist-showcase python -m digits.evaluate <run-dir> --errors   # + error list

Per run (predictions/{val,test}.parquet): test accuracy with 95 % Wilson interval, macro-F1, NLL
and ECE (15 equal-width confidence bins); then temperature scaling fitted on the val predictions
(T minimises the val NLL of softmax(log p / T)) and NLL / ECE of the test set after scaling.
Scaling does not change the predicted class, so accuracy stays the same. Only reads the run folder.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

N_CLASSES = 10
N_BINS = 15


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    """(n, n) counts; rows are true labels, columns predictions."""
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro-F1 and per-class precision/recall (plan 10.1)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.nan_to_num(tp / cm.sum(axis=0))
        recall = np.nan_to_num(tp / cm.sum(axis=1))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return {
        "acc": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def predictions_frame(ids: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """Plan 6.3 predictions table: id, label, pred, p0..p9, loss (cross-entropy per sample)."""
    eps = 1e-12
    frame = pd.DataFrame({"id": ids, "label": labels.astype(np.int64)})
    frame["pred"] = probs.argmax(axis=1)
    for k in range(probs.shape[1]):
        frame[f"p{k}"] = probs[:, k].astype(np.float32)
    frame["loss"] = -np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0))
    return frame


def load_predictions(run_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """predictions/<split>.parquet and its probabilities (N, 10) as float64."""
    frame = pd.read_parquet(Path(run_dir) / "predictions" / f"{split}.parquet")
    return frame, frame[[f"p{k}" for k in range(N_CLASSES)]].to_numpy(np.float64)


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Confidence = max probability, in n_bins equal-width bins over [0, 1] (right-closed, the first
    bin includes 0). Per bin: count, mean confidence, accuracy (nan if empty); ece = sum over bins
    of count / N * |accuracy - confidence|."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    count = np.bincount(bins, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        conf = np.bincount(bins, confidence, n_bins) / count
        acc = np.bincount(bins, correct, n_bins) / count
    gaps = np.where(count > 0, np.abs(acc - conf), 0.0)
    return {"edges": edges, "count": count, "confidence": conf, "accuracy": acc,
            "ece": float((count * gaps).sum() / len(labels))}  # fmt: skip


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins=N_BINS) -> float:
    return reliability(probs, labels, n_bins)["ece"]


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy of the given labels."""
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-300, 1.0)).mean())


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log p / T): the same as dividing the logits by T (softmax ignores the constant)."""
    z = np.log(np.clip(probs, 1e-300, 1.0)) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """T in [0.05, 20] minimising the NLL of apply_temperature(probs, T) (1-D search on log T)."""
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda log_t: nll(apply_temperature(probs, math.exp(log_t)), labels),
        bounds=(math.log(0.05), math.log(20)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return math.exp(result.x)


def wilson_interval(n_correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % (z = 1.96) Wilson score interval for a proportion; unlike p +- z SE it stays inside
    [0, 1] and is reliable close to 1 (99.68 % on 10 000 images: -0.13 / +0.09 pp)."""
    p = n_correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def top_k(probs: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Classes (N, k) by falling probability and their probabilities."""
    classes = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return classes, np.take_along_axis(probs, classes, axis=1)


def evaluate_run(run_dir: Path) -> dict:
    """Test metrics, interval, calibration before and after temperature scaling (fit on val)."""
    val, val_probs = load_predictions(run_dir, "val")
    test, probs = load_predictions(run_dir, "test")
    labels = test["label"].to_numpy()
    metrics = classification_metrics(labels, probs.argmax(axis=1))
    temperature = fit_temperature(val_probs, val["label"].to_numpy())
    scaled = apply_temperature(probs, temperature)
    n_correct = int(round(metrics["acc"] * len(labels)))
    return {
        **metrics,
        "n": len(labels),
        "errors": len(labels) - n_correct,
        "acc_interval": wilson_interval(n_correct, len(labels)),
        "nll": nll(probs, labels),
        "ece": expected_calibration_error(probs, labels),
        "temperature": temperature,
        "nll_scaled": nll(scaled, labels),
        "ece_scaled": expected_calibration_error(scaled, labels),
    }


def error_table(run_dir: Path, split: str = "test") -> pd.DataFrame:
    """Misclassified samples: id, label, pred, top-3 classes and probabilities, sorted by loss."""
    frame, probs = load_predictions(run_dir, split)
    wrong = (frame["pred"] != frame["label"]).to_numpy()
    classes, values = top_k(probs[wrong])
    out = frame.loc[wrong, ["id", "label", "pred", "loss"]].reset_index(drop=True)
    out["top3"] = [", ".join(f"{c}: {v:.3f}" for c, v in zip(cs, vs, strict=True))
                   for cs, vs in zip(classes, values, strict=True)]  # fmt: skip
    return out.sort_values("loss", ascending=False, ignore_index=True)


def format_evaluation(run_id: str, r: dict) -> str:
    lo, hi = r["acc_interval"]
    return (
        f"{run_id}\n"
        f"  test acc {100 * r['acc']:.2f} % ({r['errors']} errors of {r['n']}), "
        f"95 % CI [{100 * lo:.2f}, {100 * hi:.2f}] %, macro-F1 {100 * r['macro_f1']:.2f} %\n"
        f"  NLL {r['nll']:.4f}, ECE {100 * r['ece']:.2f} %  ->  temperature {r['temperature']:.3f}"
        f" (fit on val): NLL {r['nll_scaled']:.4f}, ECE {100 * r['ece_scaled']:.2f} %"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="run folders with predictions/")
    parser.add_argument("--errors", action="store_true", help="list misclassified test images")
    args = parser.parse_args(argv)
    for run_dir in args.runs:
        print(format_evaluation(run_dir.name, evaluate_run(run_dir)))
        if args.errors:
            print(error_table(run_dir).to_string(index=False))


if __name__ == "__main__":
    main()
