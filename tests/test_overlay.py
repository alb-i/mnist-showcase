import csv
from pathlib import Path

DECISIONS = Path(__file__).parents[1] / "curation" / "decisions.csv"


def test_decisions_header():
    with DECISIONS.open() as f:
        header = next(csv.reader(f))
    assert header == [
        "sample_id",
        "action",
        "new_label",
        "reason",
        "reviewer",
        "timestamp",
        "round",
    ]


def test_apply_overlay():
    import numpy as np

    from digits.data import apply_overlay

    ids = np.array(["train-00000", "train-00001", "train-00002", "train-00003"])
    labels = np.array([1, 2, 3, 4], dtype=np.uint8)
    decisions = {
        "train-00001": {"action": "relabel", "new_label": "7"},
        "train-00002": {"action": "drop", "new_label": ""},
        "train-00003": {"action": "unsure", "new_label": ""},
        "test-0000": {"action": "drop", "new_label": ""},  # other split: ignored
    }
    keep, curated = apply_overlay(ids, labels, decisions)
    assert keep.tolist() == [True, True, False, True]
    assert curated.tolist() == [1, 7, 3, 4] and labels.tolist() == [1, 2, 3, 4]


def test_read_decisions_later_row_wins(tmp_path):
    from digits.data import read_decisions

    path = tmp_path / "decisions.csv"
    path.write_text(
        "sample_id,action,new_label,reason,reviewer,timestamp,round\n"
        "train-00001,unsure,,,immo,2026-10-01T10:00,1\n"
        "train-00001,relabel,7,clearly a 7,immo,2026-10-02T10:00,2\n"
    )
    assert read_decisions(path)["train-00001"]["action"] == "relabel"
