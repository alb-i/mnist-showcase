"""The CuPy backend gives the same results as NumPy; skipped without CuPy or a GPU."""

import json

import numpy as np
import pytest
from conftest import synthetic_dataset

from digits.numpy_net.backend import get_xp, gpu_available, to_numpy
from digits.numpy_net.layers import MLP
from digits.numpy_net.losses import SoftmaxCrossEntropy
from digits.numpy_net.optim import make_optimizer
from digits.numpy_net.train import DEFAULT_CONFIG, load_config, train
from digits.weight_delta import WeightDelta

pytestmark = pytest.mark.skipif(not gpu_available(), reason="needs CuPy and a CUDA GPU")


def test_get_xp():
    assert get_xp("cpu") is np and get_xp("gpu").__name__ == "cupy"
    with pytest.raises(ValueError):
        get_xp("tpu")


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_gpu_steps_match_cpu(optimizer):
    """Five training steps in float64 give the same weights on both devices."""
    cp = get_xp("gpu")
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((16, 6)), rng.integers(0, 3, 16)
    config = {"name": optimizer, "lr": 0.05}
    weights = []
    for xp in (np, cp):
        model = MLP([6, 5, 3], np.random.default_rng(0), dtype=np.float64, xp=xp)
        loss, opt = SoftmaxCrossEntropy(), make_optimizer(config, model.params(), model.grads())
        for _ in range(5):
            loss.forward(model.forward(xp.asarray(x)), xp.asarray(y))
            model.backward(loss.backward())
            opt.step()
        weights.append([to_numpy(p) for p in model.params()])
    for a, b in zip(*weights, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_checkpoint_moves_between_devices(tmp_path):
    gpu = MLP([4, 3, 2], np.random.default_rng(0), xp=get_xp("gpu"))
    cpu = MLP([4, 3, 2], np.random.default_rng(1))
    gpu.save(tmp_path / "m.npz")
    cpu.load(tmp_path / "m.npz")
    for p, q in zip(gpu.params(), cpu.params(), strict=True):
        np.testing.assert_array_equal(to_numpy(p), q)


def test_weight_delta_on_cupy_arrays():
    cp = get_xp("gpu")
    w = cp.zeros(3)
    delta = WeightDelta(every=1, lags=[1])
    delta(0, [w])
    w += 1
    assert delta(1, [w]) == {"weight_delta_1": 3.0}


def test_train_on_gpu(tmp_path):
    config = load_config(DEFAULT_CONFIG, optimizer="adam", epochs=3, device="gpu")
    config.update(layers=[784, 32, 16, 10], batch_size=8, log_every=10)
    results = train(config, synthetic_dataset(), root=tmp_path, name="np-gpu")
    assert results["test"]["acc"] > 0.95
    (run_dir,) = tmp_path.iterdir()
    records = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert all("gpu_mem_gb" in r for r in records if r["split"] == "train" and "loss" in r)
    assert any("weight_delta_100" in r for r in records)  # 114 steps
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "finished"
