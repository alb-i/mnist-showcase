import json

import numpy as np
import pandas as pd
from conftest import run_dir_for
from test_evaluate import fake_predictions

from digits.figures import (
    EVAL_FIGURES,
    FIGURES,
    curation_examples,
    make_curation_figures,
    make_eval_figures,
    make_figures,
    series,
    window_mean,
)


def fake_run(path, n_epochs=3, steps_per_epoch=200):
    path.mkdir(parents=True)
    records = []
    for epoch in range(1, n_epochs + 1):
        for step in range((epoch - 1) * steps_per_epoch + 50, epoch * steps_per_epoch + 1, 50):
            records.append({"step": step, "epoch": epoch, "split": "train", "loss": 1 / step,
                            "ls_factor": 0.5, "lr": 1e-3})  # fmt: skip
            if step % 100 == 0:
                records.append({"step": step, "epoch": epoch, "split": "train",
                                "weight_delta_100": 10.0})  # fmt: skip
        records.append({"step": epoch * steps_per_epoch, "epoch": epoch, "split": "val",
                        "loss": 0.1, "acc": 0.9 + epoch / 100})  # fmt: skip
    (path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    return records


def test_series_uses_fractional_epochs(tmp_path):
    records = fake_run(tmp_path / "run")
    x, y = series(records, "train", "loss")
    assert x[:4] == [0.25, 0.5, 0.75, 1.0] and y[0] == 1 / 50
    assert series(records, "val", "acc")[0] == [1, 2, 3]


def test_make_figures(tmp_path):
    for runs in FIGURES.values():
        for run in runs.values():
            fake_run(run_dir_for(tmp_path / "runs", run), n_epochs=30)  # enough for window 50
    written = make_figures(root=tmp_path / "runs", out_dir=tmp_path / "fig")
    assert sorted(p.name for p in written) == [
        "e2_curves.pdf",
        "e2_long.pdf",
        "e2b_fullbatch.pdf",
        "e5_early_stop.pdf",
    ]
    assert all(p.read_bytes().startswith(b"%PDF") for p in written)
    pngs = sorted(p.name for p in (tmp_path / "fig").glob("*.png"))
    assert pngs == ["e5_early_stop.png"]  # only the README figures


def test_window_mean_drops_incomplete_block():
    assert window_mean([1, 2, 3, 4, 5], [1.0, 3.0, 5.0, 7.0, 9.0], 2) == ([2, 4], [2.0, 6.0])


def test_make_eval_figures(tmp_path):
    rng = np.random.default_rng(0)
    runs = {*EVAL_FIGURES["calibration"].values()}
    runs |= {EVAL_FIGURES["confusion"], EVAL_FIGURES["errors"]}
    for run in runs:
        fake_predictions(run_dir_for(tmp_path / "runs", run), rng)  # ids test-0000 ... test-0499
    images = rng.integers(0, 256, (500, 28, 28), dtype=np.uint8)
    written = make_eval_figures(tmp_path / "runs", tmp_path / "fig", test_images=images)
    assert [p.name for p in written] == ["calibration.pdf", "confusion.pdf", "errors.pdf"]
    assert all(p.read_bytes().startswith(b"%PDF") for p in written)


def test_curation_examples_and_figure(tmp_path):
    issues = pd.DataFrame({
        "id": ["train-00003", "train-00001", "test-0002", "train-00002", "train-00000"],
        "split": ["train", "train", "test", "train", "train"],
        "label": [1, 7, 4, 3, 5],
        "rank": [1, 2, 3, 4, 5],
    })  # fmt: skip
    decisions = {"train-00001": {"action": "relabel", "new_label": "1"},
                 "train-00003": {"action": "relabel", "new_label": "2"},
                 "train-00002": {"action": "drop", "new_label": ""},
                 "train-00000": {"action": "keep", "new_label": ""},
                 "test-0002": {"action": "drop", "new_label": ""}}  # fmt: skip
    # test first, then relabels and drops in rank order; keep is not shown
    assert curation_examples(issues, decisions, n_relabel=1) == [
        ("test-0002", "4 → drop"), ("train-00003", "1 → 2"), ("train-00002", "3 → drop")
    ]  # fmt: skip
    run = run_dir_for(tmp_path / "runs", EVAL_FIGURES["curation"])
    run.mkdir(parents=True)
    issues.to_parquet(run / "label_issues.parquet")
    csv = tmp_path / "decisions.csv"
    csv.write_text("sample_id,action,new_label\n" + "".join(
        f"{k},{v['action']},{v['new_label']}\n" for k, v in decisions.items()))  # fmt: skip
    rng = np.random.default_rng(0)
    images = {s: rng.integers(0, 256, (5, 28, 28), dtype=np.uint8) for s in ("train", "test")}
    (path,) = make_curation_figures(tmp_path / "runs", tmp_path, decisions=csv, images=images)
    assert path.name == "curation.pdf" and path.read_bytes().startswith(b"%PDF")
    assert path.with_suffix(".png").read_bytes().startswith(b"\x89PNG")  # README figure
