import json

import numpy as np
import pytest
from conftest import synthetic_dataset

from digits.numpy_net.layers import MLP, Dense
from digits.numpy_net.losses import SoftmaxCrossEntropy, softmax
from digits.numpy_net.optim import SGD, Adam, SGDLineSearch, make_optimizer
from digits.numpy_net.train import DEFAULT_CONFIG, load_config, train


def test_he_initialisation():
    layer = Dense(784, 256, np.random.default_rng(0))
    assert layer.W.dtype == np.float32 and not layer.b.any()
    assert layer.W.std() == pytest.approx(np.sqrt(2 / 784), rel=0.02)


def test_softmax_is_stable():
    p = softmax(np.array([[1000.0, 1000.0, -1000.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(p, [[0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]])


def test_cross_entropy_gradient_is_p_minus_y_over_n():
    loss = SoftmaxCrossEntropy()
    logits, y = np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([0, 1])
    p = softmax(logits)
    assert loss.forward(logits, y) == pytest.approx(-np.log(p[[0, 1], y]).mean())
    np.testing.assert_allclose(loss.backward(), (p - np.eye(2)[y]) / 2)


def test_sgd_momentum_steps():
    w, g = np.array([1.0]), np.array([0.5])
    opt = SGD([w], [g], lr=0.1, momentum=0.9)
    opt.step()  # v = 0.5
    opt.step()  # v = 0.9 * 0.5 + 0.5 = 0.95
    assert w[0] == pytest.approx(1.0 - 0.1 * 0.5 - 0.1 * 0.95)


def test_adam_first_step_is_lr_times_sign():
    w, g = np.array([1.0, 1.0]), np.array([3.0, -0.01])
    Adam([w], [g], lr=0.01).step()  # bias correction: m_hat / sqrt(v_hat) = sign(g)
    np.testing.assert_allclose(w, [0.99, 1.01], rtol=1e-6)


@pytest.mark.parametrize(
    "config", [{"name": "sgd", "lr": 0.1, "momentum": 0.9}, {"name": "adam", "lr": 0.1}]
)
def test_optimizers_minimise_a_quadratic(config):
    w = np.array([3.0, -2.0])
    g = np.zeros(2)
    opt = make_optimizer(config, [w], [g])
    for _ in range(300):
        g[...] = 2 * w  # gradient of |w|^2
        opt.step()
    assert np.abs(w).max() < 1e-2


def test_save_and_load(tmp_path):
    a, b = MLP([4, 3, 2], np.random.default_rng(0)), MLP([4, 3, 2], np.random.default_rng(1))
    a.save(tmp_path / "m.npz")
    b.load(tmp_path / "m.npz")
    for p, q in zip(a.params(), b.params(), strict=True):
        np.testing.assert_array_equal(p, q)


def test_config_overrides():
    config = load_config(DEFAULT_CONFIG, optimizer="adam", epochs=2)
    assert config["optimizer"] == {"name": "adam", "lr": 1e-3} and config["epochs"] == 2
    assert load_config(DEFAULT_CONFIG, lr=0.01)["optimizer"]["momentum"] == 0.9


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_train_on_synthetic_data(optimizer, tmp_path):
    config = load_config(DEFAULT_CONFIG, optimizer=optimizer, epochs=3)
    config.update(layers=[784, 32, 16, 10], batch_size=8, log_every=10)
    config["optimizer"]["lr"] = 0.01 if optimizer == "sgd" else 1e-3
    results = train(config, synthetic_dataset(), root=tmp_path, name="np")
    assert results["test"]["acc"] > 0.95
    (run_dir,) = tmp_path.iterdir()
    records = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_records = [r for r in records if "loss" in r and r["split"] == "train"]
    assert train_records[0]["step"] == 10 and {"acc", "lr", "img_s"} <= set(train_records[0])
    assert [r["epoch"] for r in records if r["split"] == "val"] == [1, 2, 3]
    assert any("weight_delta_100" in r for r in records)  # 38 steps per epoch, 114 in total
    assert records[-1]["split"] == "test"
    for name in ["checkpoints/best.npz", "checkpoints/last.npz", "predictions/test.parquet"]:
        assert (run_dir / name).exists()
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "finished"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_divergence_crashes_the_run(tmp_path):
    config = load_config(DEFAULT_CONFIG, lr=1e6, epochs=1)
    config.update(layers=[784, 32, 10], batch_size=8)
    with pytest.raises(FloatingPointError):
        train(config, synthetic_dataset(), root=tmp_path, name="np")
    (run_dir,) = tmp_path.iterdir()
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "crashed"


def quadratic(A, b):
    """f(w) = 1/2 w^T A w - b^T w, gradient A w - b, minimum A^-1 b."""
    return (lambda w: 0.5 * w @ A @ w - b @ w), (lambda w: A @ w - b)


def test_line_search_converges_where_gradient_descent_diverges():
    """Convex quadratic with gradient Lipschitz constant 10: a fixed step of 1 diverges
    (needs < 2/10), Armijo backtracking from t = 1 decreases f in every step and converges."""
    A, b = np.diag([1.0, 10.0]), np.array([1.0, -2.0])
    f, grad = quadratic(A, b)
    w_gd, g_gd = np.zeros(2), np.zeros(2)
    gd = SGD([w_gd], [g_gd], lr=1.0)
    w, g = np.zeros(2), np.zeros(2)
    ls = SGDLineSearch([w], [g], lr=1.0)
    values = [f(w)]
    for _ in range(100):
        g_gd[...] = grad(w_gd)
        gd.step()
        g[...] = grad(w)
        values.append(ls.step(lambda: f(w), values[-1]))
        assert values[-1] <= values[-2] - ls.c * ls.last["step_size"] * (g @ g)  # Armijo
    assert np.linalg.norm(w_gd) > 1e10
    np.testing.assert_allclose(w, np.linalg.solve(A, b), atol=1e-8)
    assert ls.last["backtracks"] > 0


def test_line_search_takes_the_full_step_if_it_decreases_enough():
    w, g = np.array([1.0]), np.array([2.0])
    ls = SGDLineSearch([w], [g], lr=0.1)
    ls.step(lambda: float(w[0] ** 2), 1.0)
    assert ls.last == {"step_size": 0.1, "factor": 1.0, "backtracks": 0}
    assert w[0] == pytest.approx(0.8)


def test_line_search_without_descent_keeps_the_weights():
    w, g = np.array([1.0]), np.array([2.0])
    ls = SGDLineSearch([w], [g], lr=1.0, max_backtracks=5)
    assert ls.step(lambda: 5.0, 1.0) == 1.0  # the loss never decreases
    assert w[0] == 1.0 and ls.last == {"step_size": 0.0, "factor": 0.0, "backtracks": 6}


def test_line_search_training_logs_step_sizes(tmp_path):
    config = load_config(DEFAULT_CONFIG, optimizer="sgd_ls", epochs=2)
    assert config["optimizer"] == {"name": "sgd_ls", "lr": 1.0}
    config.update(layers=[784, 32, 16, 10], batch_size=8, log_every=10)
    results = train(config, synthetic_dataset(), root=tmp_path, name="np")
    assert results["test"]["acc"] > 0.95
    (run_dir,) = tmp_path.iterdir()
    records = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    logged = [r for r in records if "ls_step" in r]
    assert logged and all(0 < r["ls_step"] <= 1.0 and r["ls_backtracks"] >= 0 for r in logged)
    assert all(r["ls_factor"] == pytest.approx(r["ls_step"]) for r in logged)  # lr = 1


def test_full_batch_line_search(tmp_path):
    config = load_config(DEFAULT_CONFIG, optimizer="sgd_ls", epochs=5, full_batch=True)
    config.update(layers=[784, 32, 16, 10])
    train(config, synthetic_dataset(), root=tmp_path, name="np")
    (run_dir,) = tmp_path.iterdir()
    records = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_records = [r for r in records if r["split"] == "train" and "loss" in r]
    assert [r["step"] for r in train_records] == [1, 2, 3, 4, 5]
    losses = [r["loss"] for r in train_records]
    assert losses == sorted(losses, reverse=True)  # Armijo on the full training loss
