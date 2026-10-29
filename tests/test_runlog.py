import json
import time

import pytest
import yaml

from digits import data
from digits.runlog import RunLogger


def test_run_folder_layout(tmp_path):
    config = {"model": "mlp", "optimizer": {"lr": 0.1}}
    with RunLogger(config, root=tmp_path) as run:
        run.log(step=1, epoch=0, split="train", loss=2.3, acc=0.1, lr=0.1)
        run.log(step=1, epoch=0, split="val", loss=2.2, acc=0.12)
        meta = json.loads((run.dir / "meta.json").read_text())
        assert meta["status"] == "running"
    assert run.run_id.endswith("-mlp") and run.dir == tmp_path / run.run_id
    for name in ["heartbeat", "checkpoints", "predictions"]:
        assert (run.dir / name).exists()
    assert yaml.safe_load((run.dir / "config.yaml").read_text()) == config
    lines = [json.loads(s) for s in (run.dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["split"] for r in lines] == ["train", "val"]
    assert all(r["ts"] > 0 for r in lines)
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["status"] == "finished" and "end_time" in meta
    assert meta["dataset_version"] == data.dataset_version()
    assert len(meta["git_commit"]) >= 40 or meta["git_commit"] == "unknown"
    assert set(meta) >= {"run_id", "start_time", "host"}


def test_crash_is_recorded(tmp_path):
    with pytest.raises(RuntimeError), RunLogger({"model": "cnn"}, root=tmp_path) as run:
        raise RuntimeError("boom")
    assert json.loads((run.dir / "meta.json").read_text())["status"] == "crashed"


def test_heartbeat_is_touched(tmp_path):
    with RunLogger({}, name="hb", root=tmp_path, heartbeat_seconds=0.05) as run:
        first = (run.dir / "heartbeat").stat().st_mtime_ns
        time.sleep(0.3)
        assert (run.dir / "heartbeat").stat().st_mtime_ns > first


def test_dataset_version(tmp_path):
    (tmp_path / "d.csv").write_text("sample_id,action\n")
    v = data.dataset_version(tmp_path / "d.csv")
    assert len(v) == 8
    (tmp_path / "d.csv").write_text("sample_id,action\ntrain-00001,drop\n")
    assert data.dataset_version(tmp_path / "d.csv") != v
    # only sample_id, action and new_label count, not reason, reviewer, timestamp or round
    header = "sample_id,action,new_label,reason,reviewer,timestamp,round\n"
    (tmp_path / "d.csv").write_text(header + "train-00001,relabel,7,clear,immo,2026-10-01T10,1\n")
    v = data.dataset_version(tmp_path / "d.csv")
    (tmp_path / "d.csv").write_text(header + "train-00001,relabel,7,,demo,2026-10,2\n")
    assert data.dataset_version(tmp_path / "d.csv") == v
    (tmp_path / "d.csv").write_text(header + "train-00001,relabel,1,clear,immo,2026-10-01T10,1\n")
    assert data.dataset_version(tmp_path / "d.csv") != v


def test_resume_appends_to_the_run(tmp_path):
    with pytest.raises(RuntimeError):
        with RunLogger({"model": "m"}, root=tmp_path, heartbeat_seconds=60) as crashed:
            crashed.log(step=1, split="train", loss=1.0)
            crashed.log(step=2, split="train", loss=0.9)  # after the checkpoint at step 1
            raise RuntimeError("killed")
    resume = {"resume": crashed.dir, "resume_step": 1, "heartbeat_seconds": 60}
    with RunLogger({"model": "m", "epochs": 5}, **resume) as run:
        run.log(step=2, split="train", loss=0.5)
    assert run.run_id == crashed.run_id and len(list(tmp_path.iterdir())) == 1
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["status"] == "finished" and len(meta["resumed"]) == 1
    assert "epochs: 5" in (run.dir / "config.yaml").read_text()
    steps = [json.loads(s)["step"] for s in (run.dir / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2]
    assert json.loads((run.dir / "metrics.jsonl").read_text().splitlines()[-1])["loss"] == 0.5
