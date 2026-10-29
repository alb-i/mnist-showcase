import numpy as np
import pytest

from digits import evaluate
from digits.evaluate import (
    apply_temperature,
    classification_metrics,
    confusion_matrix,
    fit_temperature,
    predictions_frame,
    reliability,
    top_k,
    wilson_interval,
)


def test_confusion_matrix_rows_are_true_labels():
    cm = confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), n=2)
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_classification_metrics():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = classification_metrics(y_true, y_pred)
    assert m["acc"] == pytest.approx(3 / 5)
    assert m["precision"][:3] == pytest.approx([1 / 2, 2 / 3, 0.0])
    assert m["recall"][:3] == pytest.approx([1 / 2, 1.0, 0.0])
    f1 = [1 / 2, 0.8, 0.0] + [0.0] * 7  # classes 3-9 absent count as 0
    assert m["macro_f1"] == pytest.approx(np.mean(f1))
    assert np.array(m["confusion_matrix"]).sum() == 5


def test_predictions_frame():
    probs = np.full((2, 10), 0.05)
    probs[0, 3], probs[1, 1] = 0.55, 0.55
    frame = predictions_frame(np.array(["test-0000", "test-0001"]), np.array([3, 2]), probs)
    assert list(frame.columns) == ["id", "label", "pred"] + [f"p{k}" for k in range(10)] + ["loss"]
    assert frame["pred"].tolist() == [3, 1]
    assert frame["loss"].tolist() == pytest.approx([-np.log(0.55), -np.log(0.05)])


def one_hot_probs(confidences, preds, n=10):
    """Probability rows with max `confidence` at `pred`, the rest spread evenly."""
    probs = np.empty((len(preds), n))
    for row, c, k in zip(probs, confidences, preds, strict=True):
        row[:] = (1 - c) / (n - 1)
        row[k] = c
    return probs


def test_reliability_and_ece_by_hand():
    # bins of width 0.1: two samples at 0.95 (one right), one at 0.55 (right), one at 0.15 (wrong)
    probs = one_hot_probs([0.95, 0.95, 0.55, 0.15], [1, 1, 2, 3])
    labels = np.array([1, 0, 2, 4])
    r = reliability(probs, labels, n_bins=10)
    assert r["count"].tolist() == [0, 1, 0, 0, 0, 1, 0, 0, 0, 2]
    assert r["accuracy"][9] == 0.5 and r["confidence"][9] == pytest.approx(0.95)
    assert np.isnan(r["accuracy"][0])
    # ece = (2 |0.5 - 0.95| + 1 |1 - 0.55| + 1 |0 - 0.15|) / 4
    assert r["ece"] == pytest.approx((2 * 0.45 + 0.45 + 0.15) / 4)
    certain = reliability(one_hot_probs([1.0], [0]), np.array([0]))
    assert certain["count"][-1] == 1  # p = 1 falls into the last bin


def test_temperature_scaling_recovers_the_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, (20000, 10))
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    labels = np.array([rng.choice(10, p=row) for row in p])  # calibrated at T = 1
    overconfident = apply_temperature(p, 0.5)  # logits * 2
    assert np.allclose(apply_temperature(p, 1.0), p)
    assert fit_temperature(overconfident, labels) == pytest.approx(2.0, rel=0.05)
    assert (apply_temperature(overconfident, 2.0).argmax(1) == p.argmax(1)).all()


def test_wilson_interval():
    lo, hi = wilson_interval(0, 10)  # textbook value
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-4)
    from scipy.stats import binomtest

    for k, n in [(9968, 10000), (37, 50)]:
        ref = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        assert wilson_interval(k, n) == pytest.approx((ref.low, ref.high), abs=1e-4)


def test_top_k():
    classes, values = top_k(np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]]), k=2)
    assert classes.tolist() == [[1, 2], [0, 2]] and values.tolist() == [[0.6, 0.3], [0.5, 0.3]]


def fake_predictions(run_dir, rng, n=500):
    (run_dir / "predictions").mkdir(parents=True)
    for split in ("val", "test"):
        labels = rng.integers(0, 10, n)
        preds = np.where(rng.random(n) < 0.9, labels, (labels + 1) % 10)
        probs = one_hot_probs(np.full(n, 0.6), preds)  # 90 % right at confidence 0.6
        ids = np.array([f"{split}-{i:04d}" for i in range(n)])
        path = run_dir / "predictions" / f"{split}.parquet"
        predictions_frame(ids, labels, probs).to_parquet(path)


def test_evaluate_run_and_cli(tmp_path, capsys):
    fake_predictions(tmp_path / "run", np.random.default_rng(1))
    r = evaluate.evaluate_run(tmp_path / "run")
    assert r["n"] == 500 and r["errors"] == round((1 - r["acc"]) * 500)
    assert r["ece"] == pytest.approx(r["acc"] - 0.6)  # one bin, underconfident
    assert r["temperature"] < 1 and r["ece_scaled"] < r["ece"] and r["nll_scaled"] < r["nll"]
    errors = evaluate.error_table(tmp_path / "run")
    assert len(errors) == r["errors"] and (errors["label"] != errors["pred"]).all()
    assert errors["top3"][0].startswith(f"{errors['pred'][0]}: 0.600")
    evaluate.main([str(tmp_path / "run"), "--errors"])
    out = capsys.readouterr().out
    assert "95 % CI" in out and "temperature" in out and "top3" in out
