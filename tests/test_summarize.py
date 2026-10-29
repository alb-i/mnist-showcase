import json

import pytest
import yaml

from digits.summarize import find_run, format_table, main, summarize


def fake_run(root, run_id, val_accs, test_acc, seed=0, lr=3e-3, status="finished"):
    run = root / run_id
    run.mkdir(parents=True)
    config = {"optimizer": {"name": "adamw", "lr": lr}, "scheduler": "onecycle", "seed": seed}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    meta = {"run_id": run_id, "status": status, "start_time": "2026-10-02T10:00:00+01:00"}
    if status == "finished":
        meta["end_time"] = "2026-10-02T10:02:30+01:00"
    (run / "meta.json").write_text(json.dumps(meta))
    records = [{"step": 10 * e, "epoch": e, "split": "val", "loss": 0.1, "acc": a}
               for e, a in enumerate(val_accs, start=1)]  # fmt: skip
    if test_acc is not None:
        records.append({"step": 10 * len(val_accs), "split": "test", "loss": 0.05, "acc": test_acc})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records))


def test_summarize_sorts_by_val_and_filters_by_prefix(tmp_path):
    fake_run(tmp_path, "20261002-100000-e7-seed0", [0.990, 0.994, 0.993], 0.9950, seed=0)
    fake_run(tmp_path, "20261002-100300-e7-seed1", [0.995, 0.996], 0.9940, seed=1)
    fake_run(tmp_path, "20261002-100600-e7-seed2", [0.991], None, seed=2, status="running")
    fake_run(tmp_path, "20261002-090000-e5-lr3e-3-onecycle", [0.999], 0.999)
    rows = summarize("e7-", tmp_path)
    assert [r["seed"] for r in rows] == [1, 0, 2]  # best val first, never by test
    assert rows[1]["val_acc"] == 0.994 and rows[1]["val_epoch"] == 2 and rows[1]["epochs"] == 3
    assert rows[0]["minutes"] == pytest.approx(2.5) and rows[2]["minutes"] is None
    table = format_table(rows)
    # mean +- sample std over the two finished runs only: test 99.45 +- 0.07
    assert "test: 99.45 +- 0.07 % (2 runs)" in table
    assert "best val: 99.50 +- 0.14 % (2 runs)" in table


def test_main_without_runs(tmp_path, capsys):
    main(["e6-", "--root", str(tmp_path)])
    assert "no runs named e6-*" in capsys.readouterr().out


def test_since_drops_older_runs(tmp_path):
    fake_run(tmp_path, "20261002-063643-e5-lr3e-4-onecycle", [0.997], 0.996)
    fake_run(tmp_path, "20261002-080801-e5-lr3e-4-onecycle", [0.998], 0.997)
    rows = summarize("e5-", tmp_path, since="20261002-080000")
    assert [r["run_id"] for r in rows] == ["20261002-080801-e5-lr3e-4-onecycle"]


def test_find_run(tmp_path):
    for run_id in ["20261001-090000-e7-seed0", "20261002-100000-e7-seed0",
                   "20261002-110000-cv-oof"]:  # fmt: skip
        (tmp_path / run_id).mkdir()
    assert find_run(tmp_path, "cv-oof").name == "20261002-110000-cv-oof"
    assert find_run(tmp_path, "e7-seed0#2").name == "20261002-100000-e7-seed0"
    assert find_run(tmp_path, "20261001-090000-e7-seed0").name == "20261001-090000-e7-seed0"
    with pytest.raises(ValueError, match="2 runs named e7-seed0"):
        find_run(tmp_path, "e7-seed0")
    with pytest.raises(ValueError, match="no run e7-seed0#3"):
        find_run(tmp_path, "e7-seed0#3")
    with pytest.raises(ValueError, match="0 runs named e6"):
        find_run(tmp_path, "e6")
