import json

import numpy as np
import pandas as pd
import pytest
import yaml

from digits import label_issues
from digits.evaluate import predictions_frame
from digits.label_issues import (
    calibrate_joint,
    collect,
    confident_joint,
    find_label_issues,
    latest_fold_runs,
    per_class_thresholds,
    suggested_classes,
)

# 3 classes, 7 images; image 2 is labelled 0 but looks like a 1
LABELS = np.array([0, 0, 0, 1, 1, 2, 2])
PROBS = np.array([
    [0.9, 0.05, 0.05],
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.1, 0.8],
    [0.3, 0.1, 0.6],
])  # fmt: skip


def test_confident_learning_by_hand():
    # t_0 = (0.9 + 0.7 + 0.1) / 3, t_1 = (0.8 + 0.6) / 2, t_2 = (0.8 + 0.6) / 2
    t = per_class_thresholds(LABELS, PROBS)
    assert t == pytest.approx([1.7 / 3, 0.7, 0.7])
    # image 4: 0.6 < t_1 and 0.2 < t_0, t_2 -> no class; image 6 likewise
    suggested = suggested_classes(PROBS, t)
    assert suggested.tolist() == [0, 0, 1, 1, -1, 2, -1]
    joint = confident_joint(LABELS, suggested, 3)
    assert joint.tolist() == [[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    # rows rescaled to the class sizes 3, 2, 2
    assert calibrate_joint(joint, LABELS).tolist() == [[2, 1, 0], [0, 2, 0], [0, 0, 2]]


def test_find_label_issues_by_hand():
    ids = np.array([f"train-{i:05d}" for i in range(7)])
    frame, summary = find_label_issues(ids, LABELS, PROBS)
    assert frame["flagged"].tolist() == [False, False, True, False, False, False, False]
    assert frame["rank"].tolist() == [pd.NA, pd.NA, 1, pd.NA, pd.NA, pd.NA, pd.NA]
    assert frame.loc[2, "self_confidence"] == pytest.approx(0.1)
    assert frame.loc[2, "margin"] == pytest.approx(0.1 - 0.8)
    assert frame.loc[0, "margin"] == pytest.approx(0.9 - 0.05)
    assert summary["flagged"] == 1 and summary["estimated_errors"] == pytest.approx(1.0)


def test_class_without_images_has_no_confident_images():
    t = per_class_thresholds(np.array([0, 0]), np.array([[0.4, 0.1, 0.5], [0.6, 0.1, 0.3]]))
    assert np.isinf(t[1]) and np.isinf(t[2])


def clean_probs(true_labels: np.ndarray, p: float = 0.9) -> np.ndarray:
    probs = np.full((len(true_labels), 10), (1 - p) / 9)
    probs[np.arange(len(true_labels)), true_labels] = p
    return probs


def test_flipped_labels_are_found_and_counted():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 10, 2000)
    given = true.copy()
    flipped = rng.choice(2000, 30, replace=False)
    given[flipped] = (true[flipped] + rng.integers(1, 10, 30)) % 10
    ids = np.array([f"train-{i:05d}" for i in range(2000)])
    frame, summary = find_label_issues(ids, given, clean_probs(true))
    assert set(np.flatnonzero(frame["flagged"])) == set(flipped)
    assert (frame.loc[flipped, "suggested"] == true[flipped]).all()
    assert summary["estimated_errors"] == pytest.approx(30)
    assert sorted(frame["rank"].dropna()) == list(range(1, 31))


def fake_fold_run(root, name, fold, ids, labels, probs, test, status="finished", version="abc"):
    run = root / name
    (run / "predictions").mkdir(parents=True)
    config = {"model": "cnn_reg", "cv_fold": fold, "checkpoint": "last"}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    meta = {"run_id": name, "status": status, "dataset_version": version}
    (run / "meta.json").write_text(json.dumps(meta))
    predictions_frame(ids, labels, probs).to_parquet(run / "predictions" / "val.parquet")
    predictions_frame(*test).to_parquet(run / "predictions" / "test.parquet")
    return run


@pytest.fixture
def fold_runs(tmp_path):
    """3 folds of 300 synthetic images (10 flipped labels), 100 test images (2 flipped)."""
    rng = np.random.default_rng(1)
    true = rng.integers(0, 10, 300)
    given = true.copy()
    given[:10] = (true[:10] + 1) % 10
    ids = np.array([f"train-{i:05d}" for i in range(300)])
    folds = {i: n % 3 for n, i in enumerate(ids)}
    test_true = rng.integers(0, 10, 100)
    test_given = test_true.copy()
    test_given[[5, 7]] = (test_true[[5, 7]] + 3) % 10
    test_ids = np.array([f"test-{i:04d}" for i in range(100)])
    runs = []
    for fold in range(3):
        idx = np.arange(fold, 300, 3)
        test = (test_ids, test_given, clean_probs(test_true, 0.8 + 0.05 * fold))
        runs.append(fake_fold_run(tmp_path / "runs", f"20261002-12000{fold}-cv-fold{fold}", fold,
                                  ids[idx], given[idx], clean_probs(true[idx]), test))  # fmt: skip
    return tmp_path / "runs", runs, folds


def test_collect_writes_the_cv_oof_run(fold_runs):
    root, runs, folds = fold_runs
    out = collect(runs, root, folds)
    assert out.name.endswith("-cv-oof")
    train = pd.read_parquet(out / "predictions" / "train.parquet")
    assert len(train) == 300 and train["id"].is_monotonic_increasing
    assert (train["fold"] == train["id"].map(folds)).all()
    test = pd.read_parquet(out / "predictions" / "test.parquet")
    assert test[[f"p{k}" for k in range(10)]].sum(axis=1).to_numpy() == pytest.approx(1)
    assert test["p0"].max() == pytest.approx(0.85)  # mean of 0.80, 0.85, 0.90
    issues = pd.read_parquet(out / "label_issues.parquet")
    flagged = issues[issues["flagged"]]
    assert set(flagged.loc[flagged["split"] == "train", "id"]) == {
        f"train-{i:05d}" for i in range(10)
    }
    assert set(flagged.loc[flagged["split"] == "test", "id"]) == {"test-0005", "test-0007"}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["status"] == "finished" and meta["label_issues"]["train"]["flagged"] == 10
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["source_runs"] == [r.name for r in runs]
    records = [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["split"] for r in records] == ["val", "test"]
    assert records[0]["acc"] == pytest.approx(290 / 300) and records[1]["flagged"] == 2


def test_collect_checks_the_folds(fold_runs, tmp_path):
    root, runs, folds = fold_runs
    with pytest.raises(ValueError, match="more than one fold"):
        collect([runs[0], runs[0], runs[1]], root, folds)
    with pytest.raises(ValueError, match="outside fold"):
        collect(runs, root, {**folds, "train-00000": 2})
    with pytest.raises(ValueError, match="folds 0 .. k-1"):
        collect(runs[1:], root, folds)  # folds 1, 2 only
    config = yaml.safe_load((runs[2] / "config.yaml").read_text())
    (runs[2] / "config.yaml").write_text(yaml.safe_dump({**config, "checkpoint": "best"}))
    with pytest.raises(ValueError, match="not a cross-validation run"):
        collect(runs, root, folds)


def test_latest_fold_runs_skips_unfinished(fold_runs):
    root, runs, _ = fold_runs
    newer = root / "20261002-130000-cv-fold1"
    newer.mkdir()
    (newer / "meta.json").write_text(json.dumps({"status": "crashed"}))
    assert latest_fold_runs(root, 3) == runs
    with pytest.raises(FileNotFoundError):
        latest_fold_runs(root, 4)


def test_cli(fold_runs, monkeypatch, capsys):
    root, runs, folds = fold_runs
    monkeypatch.setattr("digits.data.cv_folds", lambda: folds)
    monkeypatch.setattr("digits.data.split_config", lambda: {"cv_folds": 3})
    label_issues.main(["collect", "--root", str(root), "--top", "5"])
    out = capsys.readouterr().out
    assert "train: 10 of 300 images flagged, 10 candidates" in out
    assert "test: 2 of 100 images flagged, 2 candidates" in out


def test_low_self_confidence_makes_a_candidate_without_a_flag():
    # t_0 = (0.95 + 0.4) / 2 = 0.675, t_1 = (0.9 + 0.8) / 2 = 0.85: images 1 and 3 have no class
    labels = np.array([0, 0, 1, 1])
    probs = np.array([[0.95, 0.05], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]])
    frame, summary = find_label_issues(np.array(list("abcd")), labels, probs)
    assert frame["suggested"].tolist() == [0, -1, 1, -1] and not frame["flagged"].any()
    assert frame["candidate"].tolist() == [False, True, False, False]
    assert frame["rank"].tolist() == [pd.NA, 1, pd.NA, pd.NA]
    assert summary["flagged"] == 0 and summary["candidates"] == 1


def test_collect_with_prefix(fold_runs):
    root, runs, folds = fold_runs
    with pytest.raises(FileNotFoundError, match="cv-nols-fold0"):
        latest_fold_runs(root, 3, prefix="cv-nols")
    assert collect(runs, root, folds, prefix="cv-nols").name.endswith("-cv-nols-oof")
