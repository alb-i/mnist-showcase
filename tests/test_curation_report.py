import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_dataset

from digits.curation_report import compare, curated_runs, data_summary, format_report
from digits.train import load_config, train


def decision(action, new_label=""):
    return {"action": action, "new_label": str(new_label)}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A tiny MLP run on the synthetic data (CPU), and that data."""
    config = load_config("configs/mlp.yaml")
    config.update({"device": "cpu", "epochs": 2, "batch_size": 32, "log_every": 5,
                   "layers": [784, 32, 10]})  # fmt: skip
    dataset = synthetic_dataset()
    root = tmp_path_factory.mktemp("runs")
    return train(config, dataset, root=root, name="e7-seed0")["run_dir"], dataset


def test_data_summary():
    ids, labels = np.array(["test-0", "test-1", "test-2", "test-3"]), np.array([0, 1, 2, 3])
    decisions = {"test-0": decision("drop"), "test-1": decision("relabel", 7),
                 "test-2": decision("keep"), "train-9": decision("drop")}  # fmt: skip
    (row,) = data_summary({"test": (ids, labels)}, decisions)
    assert row == {"split": "test", "original": 4, "reviewed": 3, "keep": 1, "relabel": 1,
                   "drop": 1, "unsure": 0, "curated": 3}  # fmt: skip


def test_compare_original_and_curated_test(run):
    run_dir, dataset = run
    ids, images, labels = dataset["test"]
    pred = pd.read_parquet(run_dir / "predictions" / "test.parquet")["pred"].to_numpy()
    # drop a correct image, relabel a correct one (becomes an error), keep and unsure change nothing
    right = np.flatnonzero(pred == labels)
    decisions = {ids[right[0]]: decision("drop"),
                 ids[right[1]]: decision("relabel", (labels[right[1]] + 1) % 10),
                 ids[right[2]]: decision("keep"), ids[right[3]]: decision("unsure")}  # fmt: skip
    result = compare({"original": [run_dir]}, (ids, images, labels), decisions)
    (row,) = result["runs"]
    errors = int((pred != labels).sum())
    assert row["errors_original"] == errors and row["test_original"] == pytest.approx(
        1 - errors / 300
    )
    assert row["n_curated"] == 299 and row["errors_curated"] == errors + 1
    assert row["inferred"] == 0 and row["group"] == "original"
    assert result["groups"]["original"]["test_original"] == (row["test_original"], None)
    text = format_report(data_summary({"test": (ids, labels)}, decisions), result, "abcd1234")
    assert "dataset version abcd1234" in text and "original  mean (1 runs)" in text


def test_missing_test_predictions_come_from_the_checkpoint(run, tmp_path):
    run_dir, dataset = run
    copy = tmp_path / run_dir.name
    shutil.copytree(run_dir, copy)
    path = copy / "predictions" / "test.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[5:].to_parquet(path, index=False)  # as if the first 5 test images were dropped
    full = compare({"a": [run_dir]}, dataset["test"], {})["runs"][0]
    partial = compare({"a": [copy]}, dataset["test"], {})["runs"][0]
    assert partial["inferred"] == 5 and full["inferred"] == 0
    assert partial["errors_original"] == full["errors_original"]
    frame.assign(id="test-99999").to_parquet(path, index=False)
    with pytest.raises(ValueError, match="unknown test ids"):
        compare({"a": [copy]}, dataset["test"], {})


def test_curated_runs(tmp_path):
    for run_id, status in [("20261002-100000-e8-seed0", "finished"),
                           ("20261002-110000-e8-seed0", "finished"),
                           ("20261002-110100-e8-seed1", "crashed"),
                           ("20261002-110200-e7-seed0", "finished")]:  # fmt: skip
        (tmp_path / run_id).mkdir()
        (tmp_path / run_id / "meta.json").write_text(json.dumps({"status": status}))
    (tmp_path / "notes").mkdir()
    assert [d.name for d in curated_runs(tmp_path)] == ["20261002-100000-e8-seed0",
                                                        "20261002-110000-e8-seed0"]  # fmt: skip
    assert [d.name for d in curated_runs(tmp_path, "20261002-110000")] == [
        "20261002-110000-e8-seed0"
    ]


def test_run_e8_script(tmp_path):
    """In a scratch repo: refuses an uncommitted decisions.csv, else 3 seeds, summary, report."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "curation").mkdir()
    shutil.copy("scripts/run_e8.sh", repo / "scripts")
    (repo / "curation" / "decisions.csv").write_text("sample_id,action\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    fake = tmp_path / "bin" / "micromamba"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/calls\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}"}
    script = [str(repo / "scripts" / "run_e8.sh")]
    subprocess.run(["bash", *script], check=True, env=env)
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 5
    for seed, call in enumerate(calls[:3]):
        assert f"--seed {seed} --deterministic --name e8-seed{seed}" in call
        for arg in ["configs/cnn_reg.yaml", "--lr 1e-2", "scheduler=onecycle", "--epochs 120",
                    "early_stopping_patience=null"]:  # fmt: skip
            assert arg in call
        assert "checkpoint=last" not in call  # like E7: best val epoch
    assert "digits.summarize e8- --since" in calls[3]
    assert "digits.curation_report --since" in calls[4]
    (tmp_path / "calls").unlink()
    with open(repo / "curation" / "decisions.csv", "a") as f:
        f.write("train-00001,drop\n")
    r = subprocess.run(["bash", *script], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "uncommitted" in r.stderr
    assert not Path(tmp_path / "calls").exists()
