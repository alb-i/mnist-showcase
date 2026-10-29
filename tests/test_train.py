import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import synthetic_dataset
from PIL import Image

import digits.train
from digits.models import MLP, build_model, count_parameters
from digits.numpy_net.layers import MLP as NumpyMLP
from digits.numpy_net.losses import SoftmaxCrossEntropy
from digits.numpy_net.optim import make_optimizer as numpy_optimizer
from digits.optim import SGDLineSearch
from digits.samples import BACKGROUND, CAPTION_PX, GAP_PX, ZOOM, image_grid
from digits.train import (
    affine_theta,
    line_search_step,
    load_config,
    make_optimizer,
    random_affine,
    train,
    train_step,
)

CONFIGS = "configs/{}.yaml"


def small_config(model: str, **changes) -> dict:
    config = load_config(CONFIGS.format(model))
    config.update({"device": "cpu", "epochs": 2, "batch_size": 32, "log_every": 5, **changes})
    if model == "mlp":
        config["layers"] = [784, 32, 10]
    if config.get("augment"):  # synthetic class k is a bright row k: keep it in its row
        config["augment"] = {"rotation_deg": 1, "shift_px": 0.3, "scale": [0.98, 1.02]}
    return config


def read_records(run_dir):
    return [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize(
    "model,n_params", [("mlp", 235_146), ("cnn_small", 1_199_882), ("cnn_reg", 468_138)]
)
def test_models(model, n_params):
    net = build_model(load_config(CONFIGS.format(model)))
    assert count_parameters(net) == n_params
    net.eval()
    assert net(torch.zeros(3, 1, 28, 28)).shape == (3, 10)


def ablation_runs() -> dict[str, list[str]]:
    """{run name: --set overrides} of the E6 runs, read from scripts/run_e6.sh."""
    runs = {}
    for line in Path("scripts/run_e6.sh").read_text().splitlines():
        if line.startswith("run --name"):
            words = shlex.split(line)
            sets = [words[i + 1] for i, w in enumerate(words) if w == "--set"]
            runs[words[2]] = sets
    return runs


def test_e6_ablation_switches():
    runs = ablation_runs()
    assert list(runs) == ["e6-full", "e6-no_aug", "e6-no_bn", "e6-no_dropout", "e6-no_ls"]
    full = load_config(CONFIGS.format("cnn_reg"))
    for name, sets in runs.items():
        config = load_config(CONFIGS.format("cnn_reg"), overrides=sets)
        net = build_model(config)
        modules = list(net.modules())
        has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in modules)
        dropout = [m.p for m in modules if isinstance(m, torch.nn.Dropout)]
        assert has_bn == (name != "e6-no_bn")
        assert (max(dropout) == 0) == (name == "e6-no_dropout")
        assert (config["augment"] is None) == (name == "e6-no_aug")
        assert (config["label_smoothing"] == 0) == (name == "e6-no_ls")
        changed = {k for k in full if full[k] != config[k]}
        assert len(changed) == (name != "e6-full")  # exactly one component switched off
        net.eval()
        assert net(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_cnn_reg_without_batchnorm(tmp_path):
    net = build_model({"model": "cnn_reg", "model_args": {"batchnorm": False}})
    # BN layers (2 x channels each) replaced by conv/fc biases (channels each)
    assert count_parameters(net) == 468_138 - (32 + 32 + 64 + 64 + 128)
    config = small_config("cnn_reg", epochs=3)
    config["model_args"] = {"batchnorm": False, "dropout_conv": 0, "dropout_fc": 0}
    results = train(config, synthetic_dataset(), root=tmp_path, name="cnn_reg-no_bn")
    assert results["test"]["acc"] > 0.9


def test_mlp_he_init():
    weight = MLP([784, 256, 10])[1].weight
    assert weight.std().item() == pytest.approx(np.sqrt(2 / 784), rel=0.02)


@pytest.mark.parametrize("optimizer", [{"name": "sgd", "lr": 0.1, "momentum": 0.9},
                                       {"name": "adam", "lr": 0.01},
                                       {"name": "sgd_ls", "lr": 20.0}])  # fmt: skip
def test_torch_mlp_matches_numpy_mlp(optimizer):
    """E3 sanity check of both implementations: same weights, same data -> same loss, gradients
    and weights after three optimiser steps (float64); for sgd_ls also the same accepted steps."""
    rng = np.random.default_rng(0)
    ref = NumpyMLP([12, 8, 6, 4], rng, dtype=np.float64)
    for p in ref.params():
        p += 0.1 * rng.standard_normal(p.shape)
    net = MLP([12, 8, 6, 4]).double()
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        pairs = zip(ref.params()[0::2], ref.params()[1::2], strict=True)
        for lin, (W, b) in zip(linears, pairs, strict=True):
            lin.weight.copy_(torch.from_numpy(W.T))  # row-vector convention in NumPy (x W)
            lin.bias.copy_(torch.from_numpy(b))
    x, y = rng.standard_normal((16, 12)), rng.integers(0, 4, 16)
    ce, ref_opt = SoftmaxCrossEntropy(), numpy_optimizer(optimizer, ref.params(), ref.grads())
    opt = make_optimizer(optimizer, net.parameters())
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    backtracks = 0
    for _ in range(3):
        ref_loss = ce.forward(ref.forward(x), y)
        ref.backward(ce.backward())
        opt.zero_grad()
        loss = F.cross_entropy(net(xt), yt)
        loss.backward()
        assert loss.item() == pytest.approx(ref_loss, rel=1e-12)
        np.testing.assert_allclose(linears[0].weight.grad.numpy().T, ref.grads()[0], rtol=1e-10)
        if optimizer["name"] == "sgd_ls":
            ref_opt.step(lambda: SoftmaxCrossEntropy().forward(ref.forward(x), y), ref_loss)
            opt.step(lambda: F.cross_entropy(net(xt), yt).item(), loss.item())
            assert opt.last == ref_opt.last
            backtracks += opt.last["backtracks"]
        else:
            ref_opt.step()
            opt.step()
    if optimizer["name"] == "sgd_ls":
        assert backtracks > 0  # lr 20 is too large, the line search had to shrink it
    for lin, W in zip(linears, ref.params()[0::2], strict=True):
        np.testing.assert_allclose(lin.weight.detach().numpy().T, W, rtol=1e-10, atol=1e-12)


def test_affine_shift_and_identity():
    x = torch.zeros(2, 1, 28, 28)
    x[:, 0, 10, 10] = 1.0
    one, zero = torch.ones(2), torch.zeros(2)
    same = F.grid_sample(x, F.affine_grid(affine_theta(zero, one, zero, zero), [2, 1, 28, 28],
                                          align_corners=False), align_corners=False)  # fmt: skip
    torch.testing.assert_close(same, x)
    theta = affine_theta(zero, one, torch.tensor([2.0, 0.0]), torch.tensor([0.0, -3.0]))
    moved = F.grid_sample(x, F.affine_grid(theta, [2, 1, 28, 28], align_corners=False),
                          align_corners=False)  # fmt: skip
    assert moved[0, 0, 10, 12] == pytest.approx(1.0)  # 2 px to the right
    assert moved[1, 0, 7, 10] == pytest.approx(1.0)  # 3 px up


def test_affine_rotation_direction_and_scale():
    x = torch.zeros(1, 1, 28, 28)
    x[0, 0, 13:15, 20:22] = 1.0  # right of the centre
    rotated = F.grid_sample(x, F.affine_grid(affine_theta(torch.tensor([90.0]), torch.ones(1),
                            torch.zeros(1), torch.zeros(1)), [1, 1, 28, 28], align_corners=False),
                            align_corners=False)  # fmt: skip
    rows, cols = torch.nonzero(rotated[0, 0] > 0.5, as_tuple=True)
    assert cols.float().mean() == pytest.approx(13.5, abs=0.6)  # now on the vertical axis
    assert abs(rows.float().mean().item() - 13.5) > 5  # and 6-7 px away from the centre
    big = random_affine(x, {"rotation_deg": 0, "shift_px": 0, "scale": [2, 2]},
                        torch.Generator().manual_seed(0))  # fmt: skip
    assert (big > 0.5).sum() > (x > 0.5).sum()


def test_random_affine_is_reproducible():
    x = torch.rand(4, 1, 28, 28)
    aug = {"rotation_deg": 10, "shift_px": 2, "scale": [0.9, 1.1]}
    a = random_affine(x, aug, torch.Generator().manual_seed(1))
    b = random_affine(x, aug, torch.Generator().manual_seed(1))
    torch.testing.assert_close(a, b)
    assert a.shape == x.shape and not torch.equal(a, x)


def test_load_config_overrides():
    config = load_config(
        CONFIGS.format("cnn_reg"),
        optimizer="adam",
        lr=0.01,
        seed=3,
        overrides=["augment=null", "model_args.dropout_fc=0.5", "augment_list=[0.9, 1.1]"],
    )
    assert config["optimizer"] == {"name": "adam", "lr": 0.01} and config["seed"] == 3
    assert config["augment"] is None and config["model_args"]["dropout_fc"] == 0.5
    assert config["augment_list"] == [0.9, 1.1]
    with pytest.raises(ValueError):
        load_config(CONFIGS.format("mlp"), overrides=["epochs"])


@pytest.mark.parametrize("model", ["mlp", "cnn_small", "cnn_reg"])
def test_train_on_synthetic_data(model, tmp_path):
    config = small_config(model, epochs=3)
    results = train(config, synthetic_dataset(), root=tmp_path, name=model)
    assert results["test"]["acc"] > 0.9
    (run_dir,) = tmp_path.iterdir()
    records = read_records(run_dir)
    train_records = [r for r in records if r["split"] == "train" and "loss" in r]
    assert train_records[0]["step"] == 5 and {"acc", "lr", "img_s"} <= set(train_records[0])
    assert [r["epoch"] for r in records if r["split"] == "val"] == [1, 2, 3]
    assert records[-1]["split"] == "test" and "macro_f1" in records[-1]
    for name in ["checkpoints/best.pt", "checkpoints/last.pt", "predictions/val.parquet"]:
        assert (run_dir / name).exists()
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "finished"


def test_weight_delta_is_logged(tmp_path):
    config = small_config("mlp", epochs=11)  # 10 steps per epoch
    train(config, synthetic_dataset(), root=tmp_path, name="mlp")
    (run_dir,) = tmp_path.iterdir()
    assert [r["step"] for r in read_records(run_dir) if "weight_delta_100" in r] == [100]


def test_early_stopping(tmp_path):
    config = small_config("mlp", epochs=10, early_stopping_patience=2)
    config["optimizer"]["lr"] = 0.0  # val loss never improves after epoch 1
    train(config, synthetic_dataset(), root=tmp_path, name="mlp")
    (run_dir,) = tmp_path.iterdir()
    assert len([r for r in read_records(run_dir) if r["split"] == "val"]) == 3
    assert json.loads((run_dir / "meta.json").read_text())["early_stopped_epoch"] == 3


def test_resume_gives_the_same_run(tmp_path):
    """3 epochs at once == 2 epochs + resume to 3 epochs (incl. dropout RNG and onecycle)."""
    dataset = synthetic_dataset()
    config = small_config("cnn_small", epochs=3, scheduler="onecycle")
    config["model_args"] = {"dropout": 0.5}
    full = train(config, dataset, root=tmp_path / "full", name="full")
    (full_dir,) = (tmp_path / "full").iterdir()

    class Interrupt(Exception):
        pass

    import digits.train as train_module

    calls = {"n": 0}
    original_save = train_module.torch.save

    def save_then_crash(obj, path):
        original_save(obj, path)
        if path.name == "last.pt":
            calls["n"] += 1
            if calls["n"] == 2:
                raise Interrupt  # killed right after epoch 2

    train_module.torch.save = save_then_crash
    try:
        with pytest.raises(Interrupt):
            train(config, dataset, root=tmp_path / "split", name="split")
    finally:
        train_module.torch.save = original_save
    (split_dir,) = (tmp_path / "split").iterdir()
    assert json.loads((split_dir / "meta.json").read_text())["status"] == "crashed"
    resumed = train(config, dataset, resume=split_dir)
    assert resumed["test"]["acc"] == full["test"]["acc"]
    a = torch.load(full_dir / "checkpoints" / "last.pt")["model"]
    b = torch.load(split_dir / "checkpoints" / "last.pt")["model"]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    val = [r["loss"] for r in read_records(split_dir) if r["split"] == "val"]
    assert val == [r["loss"] for r in read_records(full_dir) if r["split"] == "val"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_divergence_crashes_the_run(tmp_path):
    config = small_config("mlp", epochs=1)
    config["optimizer"]["lr"] = 1e12
    with pytest.raises(FloatingPointError):
        train(config, synthetic_dataset(), root=tmp_path, name="mlp")
    (run_dir,) = tmp_path.iterdir()
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "crashed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_train_on_gpu_with_amp(tmp_path):
    config = small_config("cnn_reg", device="cuda", epochs=3)
    results = train(config, synthetic_dataset(), root=tmp_path, name="cnn_reg")
    assert results["test"]["acc"] > 0.9
    (run_dir,) = tmp_path.iterdir()
    train_records = [r for r in read_records(run_dir) if r["split"] == "train" and "loss" in r]
    assert all(0 < r["gpu_mem_gb"] < 6 for r in train_records)


def test_torch_line_search_converges_where_gradient_descent_diverges():
    """The convex quadratic of test_numpy_net: SGD with lr 1 diverges, sgd_ls converges."""
    A, b = torch.diag(torch.tensor([1.0, 10.0], dtype=torch.float64)), torch.tensor([1.0, -2.0])
    b = b.double()
    w = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    ls = SGDLineSearch([w], lr=1.0)

    def f() -> torch.Tensor:
        return 0.5 * w @ A @ w - b @ w

    for _ in range(100):
        ls.zero_grad()
        loss = f()
        loss.backward()
        new = ls.step(lambda: f().item(), loss.item())
        assert new <= loss.item()
    torch.testing.assert_close(w.detach(), torch.linalg.solve(A, b), atol=1e-8, rtol=0)


@pytest.mark.parametrize("chunk_size", [16, 6, 5])  # 6: chunks 6, 6, 4; 5: 5, 5, 6
def test_line_search_closure_reproduces_the_training_loss(chunk_size):
    """cnn_reg has dropout and BatchNorm: at step size 0 the closure must give exactly the
    training loss (same dropout masks per chunk), and trial forwards must not change the BN
    statistics."""
    torch.manual_seed(0)
    model = build_model(load_config(CONFIGS.format("cnn_reg")))
    model.train()
    x, y = torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))
    optimizer = SGDLineSearch(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    buffers = []
    original = line_search_step

    def check_buffers(*args):  # BN buffers right before the line search
        buffers.extend(b.clone() for b in model.buffers())
        return original(*args)

    import digits.train as train_module

    train_module.line_search_step = check_buffers
    try:
        loss, _ = train_step(model, optimizer, scaler, x, y, chunk_size=chunk_size,
                             label_smoothing=0.1, amp=False, line_search=True)  # fmt: skip
    finally:
        train_module.line_search_step = original
    assert optimizer.last["backtracks"] == 0  # t = 0 accepted at once: L(w) <= L(w)
    for b, saved in zip(model.buffers(), buffers, strict=True):
        torch.testing.assert_close(b, saved, rtol=0, atol=0)


def test_chunked_gradient_is_the_full_gradient():
    """Gradient accumulation over chunks (sizes 7, 7, 7, 7, 2) == gradient on all 30 at once."""
    torch.manual_seed(0)
    model = MLP([784, 16, 10]).double()
    x, y = torch.randn(30, 1, 28, 28, dtype=torch.float64), torch.randint(0, 10, (30,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    losses, grads = [], []
    for chunk_size in (30, 7):
        loss, correct = train_step(model, optimizer, scaler, x, y, chunk_size=chunk_size,
                                   label_smoothing=0.0, amp=False, line_search=False)  # fmt: skip
        losses.append(loss.item())
        grads.append([p.grad.clone() for p in model.parameters()])
    assert losses[0] == pytest.approx(losses[1], rel=1e-14)
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-15)


def test_full_batch_training(tmp_path):
    config = small_config("mlp", epochs=4, amp=False, full_batch=True, chunk_size=64)
    config["optimizer"] = {"name": "sgd_ls", "lr": 1.0}
    results = train(config, synthetic_dataset(), root=tmp_path, name="mlp")
    assert results["test"]["acc"] > 0.9
    (run_dir,) = tmp_path.iterdir()
    records = read_records(run_dir)
    train_records = [r for r in records if r["split"] == "train" and "loss" in r]
    assert [r["step"] for r in train_records] == [1, 2, 3, 4]  # one step per epoch, all logged
    assert all("ls_step" in r for r in train_records)
    losses = [r["loss"] for r in train_records]
    assert losses == sorted(losses, reverse=True)  # Armijo on the full training loss


@pytest.mark.parametrize("model", ["mlp", "cnn_reg"])
def test_train_with_line_search(model, tmp_path):
    config = small_config(model, epochs=3, amp=False)
    config["optimizer"] = {"name": "sgd_ls", "lr": 1.0}
    config.pop("scheduler", None)
    results = train(config, synthetic_dataset(), root=tmp_path, name=model)
    assert results["test"]["acc"] > 0.9
    (run_dir,) = tmp_path.iterdir()
    logged = [r for r in read_records(run_dir) if "ls_step" in r]
    assert logged and all(0 < r["ls_step"] <= 1.0 for r in logged)
    assert all(r["ls_factor"] == pytest.approx(r["ls_step"]) for r in logged)  # lr = 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_line_search_refuses_amp(tmp_path):
    config = small_config("mlp", device="cuda", amp=True)
    config["optimizer"] = {"name": "sgd_ls", "lr": 1.0}
    with pytest.raises(ValueError, match="amp"):
        train(config, synthetic_dataset(), root=tmp_path, name="mlp")


def test_augmented_batch_png(tmp_path, monkeypatch):
    config = small_config("cnn_reg", epochs=1)
    first = train(config, synthetic_dataset(), root=tmp_path / "a", name="cnn_reg")
    with Image.open(first["run_dir"] / "samples" / "augmented.png") as png:
        assert png.mode == "L" and png.size == image_grid(np.zeros((64, 28, 28)), [0] * 64).size
    # the sample has its own generator: without it the run is the same
    monkeypatch.setattr(digits.train, "write_augmented_batch", lambda *args: None)
    second = train(config, synthetic_dataset(), root=tmp_path / "b", name="cnn_reg")
    losses = [[r.get("loss") for r in read_records(r["run_dir"])] for r in (first, second)]
    assert losses[0] == losses[1]
    resumed = train(config, synthetic_dataset(), resume=first["run_dir"])
    assert resumed["run_dir"] == first["run_dir"]  # resume keeps the existing samples/


def test_image_grid_places_tiles_row_by_row():
    images = np.zeros((10, 28, 28))
    images[9] = 1.0  # last image white: row 1, column 1 of a 8-column grid
    grid = np.asarray(image_grid(images, list(range(10))))
    tile_w, tile_h = 28 * ZOOM + GAP_PX, 28 * ZOOM + CAPTION_PX + GAP_PX
    assert grid.shape == (GAP_PX + 2 * tile_h, GAP_PX + 8 * tile_w)
    top = GAP_PX + tile_h + CAPTION_PX  # first image row of the second grid row
    assert grid[top + 5, GAP_PX + tile_w + 5] == 255 and grid[top + 5, GAP_PX + 5] == 0
    assert grid[0, 0] == BACKGROUND


def test_checkpoint_last_keeps_the_last_epoch(tmp_path):
    config = small_config("mlp", epochs=3, checkpoint="last")
    results = train(config, synthetic_dataset(), root=tmp_path, name="mlp")
    run_dir = results["run_dir"]
    assert not (run_dir / "checkpoints" / "best.pt").exists()
    last_val = [r for r in read_records(run_dir) if r["split"] == "val"][-1]
    assert results["val"]["acc"] == pytest.approx(last_val["acc"])
    with pytest.raises(ValueError):
        train({**config, "checkpoint": "first"}, synthetic_dataset(), root=tmp_path, name="x")


def test_fold_option_uses_the_cv_dataset(tmp_path, monkeypatch):
    calls = {}
    load = lambda fold: calls.setdefault("fold", fold)  # noqa: E731
    monkeypatch.setattr(digits.train, "load_cv_dataset", load)
    monkeypatch.setattr(digits.train, "train", lambda config, dataset, **kw: calls.update(
        config=config, dataset=dataset, **kw))  # fmt: skip
    digits.train.main(["configs/cnn_reg.yaml", "--fold", "3", "--set", "checkpoint=last"])
    assert calls["fold"] == 3 and calls["dataset"] == 3 and calls["config"]["cv_fold"] == 3
    assert calls["config"]["checkpoint"] == "last"


def test_run_cv_script():
    text = Path("scripts/run_cv.sh").read_text()
    for arg in ["--fold", "checkpoint=last", "early_stopping_patience=null", "--epochs 120",
                "--deterministic", "--lr 1e-2", "scheduler=onecycle"]:  # fmt: skip
        assert arg in text


@pytest.mark.parametrize(
    "args,name,ls", [([], "cv-fold3", False), (["--no-ls"], "cv-nols-fold3", True)]
)
def test_run_cv_script_passes_the_options(tmp_path, args, name, ls):
    fake = tmp_path / "micromamba"
    fake.write_text('#!/bin/sh\necho "$@" >> "$(dirname "$0")/calls"\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    subprocess.run(["bash", "scripts/run_cv.sh", *args, "3"], check=True, env=env)
    (call,) = (tmp_path / "calls").read_text().splitlines()
    assert f"--fold 3 --name {name}" in call and "checkpoint=last" in call
    assert ("label_smoothing=0.0" in call) == ls
