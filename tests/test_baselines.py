import json

import pandas as pd
import pytest
from conftest import synthetic_dataset

from digits.baselines import BASELINES, run_baseline


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_run(name, tmp_path):
    results = run_baseline(BASELINES[name], synthetic_dataset(), root=tmp_path)
    assert results["val"]["acc"] > 0.95 and results["test"]["acc"] > 0.95
    (run_dir,) = tmp_path.iterdir()
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "finished"
    records = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["split"] for r in records] == ["val", "test"]
    assert records[1]["acc"] == pytest.approx(results["test"]["acc"])
    frame = pd.read_parquet(run_dir / "predictions" / "test.parquet")
    assert len(frame) == 300 and {"id", "label", "pred", "p9", "loss"} <= set(frame.columns)
