"""Figures for reports/report.tex from run folders.

    micromamba run -n mnist-showcase python -m digits.figures     # -> reports/figures/*.pdf

The runs behind each figure are listed in FIGURES (curves from metrics.jsonl) and EVAL_FIGURES
(from predictions/*.parquet: reliability diagram, confusion matrix, error gallery with the raw
test images; examples of the curation decisions in the order of the review queue of the
candidate run); re-running the script after new runs only needs new run references there. A
reference is a run name, or name#k for the k-th run of that name (digits.summarize.find_run).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from digits.data import REPO_DIR, load_split, read_decisions  # noqa: E402
from digits.evaluate import (  # noqa: E402
    apply_temperature,
    confusion_matrix,
    fit_temperature,
    load_predictions,
    reliability,
    top_k,
)
from digits.runlog import runs_dir  # noqa: E402
from digits.summarize import find_run  # noqa: E402

OUT_DIR = REPO_DIR / "reports" / "figures"

# Series identity: fixed hue per optimiser (categorical slots 1 and 2) plus a line style, so the
# figures also read in greyscale; ink colours for text and axes.
STYLE = {
    "SGD": {"color": "#2a78d6", "ls": "-"},
    "Adam": {"color": "#eb6834", "ls": "--"},
    "NumPy": {"color": "#2a78d6", "ls": "-"},
    "PyTorch": {"color": "#eb6834", "ls": "--"},
    "early stopping": {"color": "#2a78d6", "ls": "-"},
    "120 epochs": {"color": "#eb6834", "ls": "--"},
    "as trained": {"color": "#2a78d6", "ls": "-", "marker": "o"},
    "temperature-scaled": {"color": "#eb6834", "ls": "--", "marker": "s"},
}
INK, MUTED = "#0b0b0b", "#52514e"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FIGURES = {
    "e2_curves": {  # E2, 20 epochs, CPU
        "SGD": "numpy_mlp-sgd#1",
        "Adam": "numpy_mlp-adam#1",
    },
    "e2_long": {  # 300 epochs, GPU: the SGD run collapses
        "SGD": "numpy_mlp-sgd-gpu#2",
        "Adam": "numpy_mlp-adam-gpu#2",
    },
    "e2b_fullbatch": {  # SGD with Armijo line search on the full training set, 20000 steps
        "NumPy": "numpy_mlp-sgd_ls-fullbatch-gpu#2",
        "PyTorch": "mlp-sgd_ls-fullbatch#2",
    },
    "e5_early_stop": {  # E5 lr 1e-3 cosine: 30-epoch schedule stopped at epoch 22 vs. D16
        "early stopping": "e5-lr1e-3-cosine#1",
        "120 epochs": "e5-lr1e-3-cosine#2",
    },
}

EVAL_FIGURES = {
    "calibration": {  # test set; temperature fitted on the run's val predictions
        "with label smoothing (E7, seed 0)": "e7-seed0#2",
        "without label smoothing (E6)": "e6-no_ls#2",
    },
    "confusion": "e7-seed0#2",
    "errors": "e7-seed0#2",
    "curation": "cv-nols-oof",  # the queue reviewed in round 1
}

# Figures also written as PNG for the README (GitHub shows no PDF inline).
README_FIGURES = {"e5_early_stop", "confusion", "curation"}

# Figures with more than two panels are laid out in a grid with this many columns; figures in
# LOG_X get a logarithmic epoch axis.
NCOLS = {"e2b_fullbatch": 2, "e5_early_stop": 3}
LOG_X = {"e2b_fullbatch"}


def read_metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def series(records: list[dict], split: str, key: str) -> tuple[list[float], list[float]]:
    """(epoch, value) for all records of a split that have the key; train steps become
    fractional epochs, using the steps per epoch from the first val record."""
    steps_per_epoch = next(r["step"] for r in records if r["split"] == "val")
    rows = [r for r in records if r["split"] == split and key in r]
    x = [r["epoch"] if split == "val" else r["step"] / steps_per_epoch for r in rows]
    return x, [r[key] for r in rows]


def style_axes(ax, ylabel: str, xlabel: str = "epoch") -> None:
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")


def window_mean(x: list[float], y: list[float], window: int) -> tuple[list[float], list[float]]:
    """Means over consecutive blocks of `window` points, placed at the last x of each block."""
    blocks = range(0, len(y) - window + 1, window)
    return [x[i + window - 1] for i in blocks], [sum(y[i : i + window]) / window for i in blocks]


def plot_runs(runs: dict[str, Path], panels: list[tuple], out: Path,
              ncols: int | None = None, log_x: bool = False) -> None:  # fmt: skip
    """One panel per (split, key, ylabel, log_y[, options]); one line per run; legend in the
    first panel. Options: window (plot means over blocks of that many points), ylim, base2
    (log axis with ticks 1, 1/2, 1/4, ...). With log_x the x axis is the epoch (= step for
    full-batch runs) on a log scale."""
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.6 * nrows), squeeze=False)
    records = {label: read_metrics(path) for label, path in runs.items()}
    for ax, (split, key, ylabel, log_y, *rest) in zip(axes.flat, panels, strict=True):
        options = rest[0] if rest else {}
        for label, recs in records.items():
            x, y = series(recs, split, key)
            if "window" in options:
                x, y = window_mean(x, y, options["window"])
            ax.plot(x, y, label=label, linewidth=1.4, **STYLE.get(label, {}))
        if log_y:
            ax.set_yscale("log", base=2 if options.get("base2") else 10)
        if options.get("base2"):
            ticks = [2.0**-k for k in range(4)]
            ax.set_yticks(ticks, ["1", "1/2", "1/4", "1/8"])
            ax.minorticks_off()
        if "ylim" in options:
            ax.set_ylim(*options["ylim"])
        if log_x:
            ax.set_xscale("log")
        style_axes(ax, ylabel)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    save(fig, out)


def make_figures(root: Path | None = None, out_dir: Path = OUT_DIR, figures=FIGURES) -> list[Path]:
    root = root or runs_dir()
    panels = {
        "e2_curves": [
            ("train", "loss", "train loss", True),
            ("val", "acc", "val accuracy", False),
        ],
        "e2_long": [
            ("val", "acc", "val accuracy", False),
            ("train", "weight_delta_100", "weight change per 100 steps", True),
        ],
        "e2b_fullbatch": [
            ("train", "loss", "train loss", True),
            (
                "train",
                "ls_factor",
                "shortening factor (mean of 50)",
                True,
                {"window": 50, "base2": True},
            ),
            ("val", "loss", "val loss", True, {"ylim": (0.07, 0.3)}),
            ("val", "acc", "val accuracy", False, {"ylim": (0.95, 0.98)}),
        ],
        "e5_early_stop": [
            ("train", "lr", "learning rate", False),
            ("val", "loss", "val loss", False, {"ylim": (0.08, 0.14)}),
            ("val", "acc", "val accuracy", False, {"ylim": (0.99, 0.999)}),
        ],
    }
    written = []
    for name, runs in figures.items():
        out = out_dir / f"{name}.pdf"
        plot_runs({label: find_run(root, run) for label, run in runs.items()}, panels[name], out,
                  ncols=NCOLS.get(name), log_x=name in LOG_X)  # fmt: skip
        written.append(out)
    return written


def save(fig, out: Path) -> Path:
    """Writes the PDF (and for README_FIGURES a PNG next to it); returns the PDF path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None})  # same content -> same file
    if out.stem in README_FIGURES:
        fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out


def plot_calibration(runs: dict[str, Path], out: Path) -> Path:
    """Per run: reliability diagram (accuracy vs. mean confidence per non-empty bin, as trained and
    temperature-scaled) above the number of test images per confidence bin (log scale)."""
    fig, axes = plt.subplots(2, len(runs), figsize=(3.3 * len(runs), 4.6), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.4]})  # fmt: skip
    for col, (title, run_dir) in enumerate(runs.items()):
        val, val_probs = load_predictions(run_dir, "val")
        test, probs = load_predictions(run_dir, "test")
        labels = test["label"].to_numpy()
        t = fit_temperature(val_probs, val["label"].to_numpy())
        top, bottom = axes[0][col], axes[1][col]
        top.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, ls=":")
        scaled = apply_temperature(probs, t)
        for label, p in (("as trained", probs), ("temperature-scaled", scaled)):
            r = reliability(p, labels)
            used = r["count"] > 0
            name = label if label == "as trained" else f"{label} (T = {t:.2f})"
            top.plot(r["confidence"][used], r["accuracy"][used], markersize=4, linewidth=1.2,
                     label=f"{name}: ECE {100 * r['ece']:.2f} %", **STYLE[label])  # fmt: skip
            centres = (r["edges"][:-1] + r["edges"][1:]) / 2
            bottom.plot(centres[used], r["count"][used], markersize=3, linewidth=1, **STYLE[label])
        top.set_xlim(0, 1.02)
        top.set_ylim(0, 1.02)
        top.set_title(title, fontsize=9, color=INK)
        top.legend(frameon=False, fontsize=7, labelcolor=INK, loc="lower right")
        style_axes(top, "accuracy", "mean confidence in bin")
        bottom.set_yscale("log")
        bottom.set_xlim(0, 1.02)
        style_axes(bottom, "test images", "confidence (15 bins)")
    fig.tight_layout()
    return save(fig, out)


def plot_confusion(run_dir: Path, out: Path) -> Path:
    """Test confusion matrix; off-diagonal counts on a blue ramp, the diagonal as plain numbers."""
    test, probs = load_predictions(run_dir, "test")
    cm = confusion_matrix(test["label"].to_numpy(), probs.argmax(axis=1))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    cmap = matplotlib.colors.ListedColormap(["#ffffff", *BLUES])
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, len(BLUES) + 0.5), cmap.N, extend="max")
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    ax.imshow(off, cmap=cmap, norm=norm)
    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5, color=MUTED)
            elif cm[i, j]:
                ink = "#ffffff" if cm[i, j] >= 4 else INK
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color=ink)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.set_xlabel("predicted", color=MUTED)
    ax.set_ylabel("true label", color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return save(fig, out)


def plot_errors(run_dir: Path, images: np.ndarray, out: Path, cols: int = 8) -> Path:
    """All misclassified test images (sorted by loss, highest first), each titled label -> pred
    with the top-3 probabilities below."""
    test, probs = load_predictions(run_dir, "test")
    wrong = np.flatnonzero((test["pred"] != test["label"]).to_numpy())
    wrong = wrong[np.argsort(-test["loss"].to_numpy()[wrong], kind="stable")]
    classes, values = top_k(probs[wrong])
    rows = -(-len(wrong) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(0.95 * cols, 1.25 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i, cs, vs in zip(axes.flat, wrong, classes, values, strict=False):
        index = int(test["id"][i].split("-")[1])
        ax.imshow(images[index], cmap="gray_r", vmin=0, vmax=255)
        ax.set_title(f"{test['label'][i]} → {test['pred'][i]}", fontsize=8, color=INK, pad=2)
        top3 = "  ".join(f"{c}:{v:.2f}".replace("0.", ".") for c, v in zip(cs, vs, strict=True))
        ax.text(14, 31, top3, ha="center", va="top", fontsize=5, color=MUTED)
    fig.tight_layout(h_pad=0.6, w_pad=0.2)
    return save(fig, out)


def curation_examples(issues: pd.DataFrame, decisions: dict[str, dict], n_relabel: int = 12,
                      n_drop: int = 6) -> list[tuple[str, str]]:  # fmt: skip
    """(sample id, title) of the changed samples to show: every changed test image, then the
    first n_relabel relabelled and n_drop dropped training images, each in queue (rank) order."""
    frame = issues[issues["id"].isin(decisions)].copy()
    frame["action"] = [decisions[i]["action"] for i in frame["id"]]
    frame = frame[frame["action"].isin(["relabel", "drop"])]
    frame = frame.sort_values(["rank", "id"], kind="stable")
    train = frame[frame["split"] == "train"]
    chosen = pd.concat([frame[frame["split"] == "test"],
                        train[train["action"] == "relabel"].head(n_relabel),
                        train[train["action"] == "drop"].head(n_drop)])  # fmt: skip
    out = []
    for sample_id, label, action in chosen[["id", "label", "action"]].itertuples(index=False):
        new = decisions[sample_id]["new_label"] if action == "relabel" else "drop"
        out.append((sample_id, f"{label} → {new}"))
    return out


def plot_curation(examples: list[tuple[str, str]], images: dict[str, np.ndarray], out: Path,
                  cols: int = 8) -> Path:  # fmt: skip
    """Grid of the examples, titled given label -> new label (or drop), sample id below."""
    rows = -(-len(examples) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(0.95 * cols, 1.45 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (sample_id, title) in zip(axes.flat, examples, strict=False):
        split, index = sample_id.split("-")
        ax.imshow(images[split][int(index)], cmap="gray_r", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8, color=INK, pad=2)
        ax.text(14, 30.5, sample_id, ha="center", va="top", fontsize=5, color=MUTED)
    fig.tight_layout(h_pad=1.2, w_pad=0.2)
    return save(fig, out)


def make_curation_figures(root: Path | None = None, out_dir: Path = OUT_DIR,
                          figures=EVAL_FIGURES, decisions: Path | None = None,
                          images: dict[str, np.ndarray] | None = None) -> list[Path]:  # fmt: skip
    root = root or runs_dir()
    if images is None:
        images = {split: load_split(split)[0] for split in ("train", "test")}
    issues = pd.read_parquet(find_run(root, figures["curation"]) / "label_issues.parquet")
    examples = curation_examples(issues, read_decisions(decisions))
    return [plot_curation(examples, images, out_dir / "curation.pdf")]


def make_eval_figures(root: Path | None = None, out_dir: Path = OUT_DIR, figures=EVAL_FIGURES,
                      test_images: np.ndarray | None = None) -> list[Path]:  # fmt: skip
    root = root or runs_dir()
    if test_images is None:
        test_images, _ = load_split("test")
    runs = {label: find_run(root, run) for label, run in figures["calibration"].items()}
    return [
        plot_calibration(runs, out_dir / "calibration.pdf"),
        plot_confusion(find_run(root, figures["confusion"]), out_dir / "confusion.pdf"),
        plot_errors(find_run(root, figures["errors"]), test_images, out_dir / "errors.pdf"),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--only", choices=["curves", "eval", "curation"],
                        help="only one group of figures")  # fmt: skip
    args = parser.parse_args(argv)
    paths = make_figures(out_dir=args.out) if args.only in (None, "curves") else []
    paths += make_eval_figures(out_dir=args.out) if args.only in (None, "eval") else []
    paths += make_curation_figures(out_dir=args.out) if args.only in (None, "curation") else []
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
