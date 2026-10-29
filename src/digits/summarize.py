"""Table of runs whose name starts with a prefix, sorted by best validation accuracy (plan 8.2).

    micromamba run -n mnist-showcase python -m digits.summarize e5-      # E5: pick the best config
    micromamba run -n mnist-showcase python -m digits.summarize e7-      # E7: mean +- std
    micromamba run -n mnist-showcase python -m digits.summarize e5- --since <yyyymmdd>-<hhmmss>

The run name is the part of the run id after the timestamp (<yyyymmdd>-<hhmmss>-<name>). The best
run is chosen by validation accuracy, never by test accuracy. With two or more finished runs the
last lines give mean +- sample standard deviation of the best val and the test accuracy.
--since drops runs with an older run id, e.g. earlier runs of the same experiment.
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import yaml

from digits.runlog import runs_dir


def run_name(run_id: str) -> str:
    return run_id.split("-", 2)[2]


def find_run(root: Path, ref: str) -> Path:
    """Run folder for a run id, a run name (must be unique in root) or name#k (the k-th run of
    that name, oldest first, k >= 1), so that code can refer to runs without their timestamps."""
    if (root / ref).is_dir():
        return root / ref
    name, _, k = ref.partition("#")
    runs = [d for d in root.iterdir() if d.is_dir() and d.name.count("-") >= 2]
    matches = sorted(d for d in runs if run_name(d.name) == name)
    if not k and len(matches) > 1:
        raise ValueError(f"{len(matches)} runs named {name} in {root}; use {name}#<k>")
    i = int(k or 1) - 1
    if not 0 <= i < len(matches):
        raise ValueError(f"no run {ref} in {root} ({len(matches)} runs named {name})")
    return matches[i]


def summarize_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    val = [r for r in records if r["split"] == "val" and "acc" in r]
    test = [r for r in records if r["split"] == "test"]
    best = max(val, key=lambda r: r["acc"]) if val else None
    minutes = None
    if meta.get("end_time"):
        start, end = (datetime.fromisoformat(meta[k]) for k in ("start_time", "end_time"))
        minutes = (end - start).total_seconds() / 60
    return {
        "run_id": run_dir.name,
        "status": meta.get("status"),
        "lr": config["optimizer"]["lr"],
        "scheduler": config.get("scheduler"),
        "seed": config.get("seed"),
        "epochs": val[-1]["epoch"] if val else 0,
        "early_stopped": meta.get("early_stopped_epoch"),
        "val_acc": best["acc"] if best else None,
        "val_epoch": best["epoch"] if best else None,
        "test_acc": test[-1]["acc"] if test else None,
        "test_loss": test[-1]["loss"] if test else None,
        "minutes": minutes,
    }


def summarize(prefix: str, root: Path | None = None, since: str = "") -> list[dict]:
    """All runs whose name starts with prefix and whose id is >= since, best val accuracy first."""
    root = root or runs_dir()
    rows = [
        summarize_run(d)
        for d in sorted(root.iterdir())
        if d.is_dir()
        and d.name.count("-") >= 2
        and run_name(d.name).startswith(prefix)
        and d.name >= since
    ]
    return sorted(rows, key=lambda r: -1.0 if r["val_acc"] is None else r["val_acc"], reverse=True)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f}"


def format_table(rows: list[dict]) -> str:
    lines = [
        f"{'run':<42} {'lr':>8} {'sched':<9} {'seed':>4} {'ep':>3} {'val %':>6} {'(ep)':>5} "
        f"{'test %':>6} {'loss':>6} {'min':>5}  status"
    ]
    for r in rows:
        stopped = f" (early stop {r['early_stopped']})" if r["early_stopped"] else ""
        loss = "-" if r["test_loss"] is None else f"{r['test_loss']:.3f}"
        minutes = "-" if r["minutes"] is None else f"{r['minutes']:.1f}"
        val, test = pct(r["val_acc"]), pct(r["test_acc"])
        lines.append(
            f"{r['run_id']:<42} {r['lr']:>8.1e} {r['scheduler'] or '-':<9} {r['seed']!s:>4} "
            f"{r['epochs']:>3} {val:>6} {r['val_epoch']!s:>5} {test:>6} "
            f"{loss:>6} {minutes:>5}  {r['status']}{stopped}"
        )
    done = [r for r in rows if r["status"] == "finished" and r["test_acc"] is not None]
    if len(done) >= 2:
        for key, label in (("val_acc", "best val"), ("test_acc", "test")):
            mean, std = mean_std([r[key] for r in done])
            lines.append(f"{label:>8}: {100 * mean:.2f} +- {100 * std:.2f} % ({len(done)} runs)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix", help="run name prefix, e.g. e5- or e7-")
    parser.add_argument("--root", type=Path, help="runs folder (default: $RUNS_DIR/mnist)")
    parser.add_argument("--since", default="", help="only run ids >= this (<yyyymmdd>-<hhmmss>)")
    args = parser.parse_args(argv)
    rows = summarize(args.prefix, args.root, args.since)
    print(format_table(rows) if rows else f"no runs named {args.prefix}*")


if __name__ == "__main__":
    main()
