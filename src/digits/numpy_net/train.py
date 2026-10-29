"""Training loop for the NumPy MLP (plan 7.2, experiment E2).

    micromamba run -n mnist-showcase python -m digits.numpy_net.train
    micromamba run -n mnist-showcase python -m digits.numpy_net.train --optimizer adam --lr 1e-3
    micromamba run -n mnist-showcase python -m digits.numpy_net.train --device gpu    # CuPy
    micromamba run -n mnist-showcase python -m digits.numpy_net.train --optimizer sgd_ls --lr 1
    ... --optimizer sgd_ls --full-batch --epochs 500    # full-batch GD with line search

Logs like every run (digits.runlog): train loss/acc/lr/img_s every `log_every` steps,
weight_delta every 100 steps, val loss/acc every epoch; with sgd_ls also the mean accepted step
size, shortening factor accepted / start step and number of halvings per step (ls_step,
ls_factor, ls_backtracks). Checkpoints best.npz (best val acc) and last.npz; the best checkpoint
is evaluated on val and test and written to predictions/.
full_batch: true (--full-batch) makes every step use the whole training set in one pass (no
shuffling; one step per epoch, logged every step); 54000 images fit at once, on the CPU and with
CuPy (about 0.5 GB).
With device gpu the data, weights and optimiser state live on the GPU (CuPy) and gpu_mem_gb is
logged; the rng (initialisation, batch order) stays on the CPU, so both devices see the same run.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from digits.data import REPO_DIR, load_dataset
from digits.evaluate import classification_metrics, predictions_frame
from digits.numpy_net.backend import DEVICES, array_module, get_xp, gpu_memory_gb, to_numpy
from digits.numpy_net.layers import MLP
from digits.numpy_net.losses import SoftmaxCrossEntropy, softmax
from digits.numpy_net.optim import SGDLineSearch, make_optimizer
from digits.runlog import RunLogger
from digits.weight_delta import WeightDelta

DEFAULT_CONFIG = REPO_DIR / "configs" / "numpy_mlp.yaml"
OPTIMIZERS = ("sgd", "adam", "sgd_ls")
DEFAULT_LR = {"sgd": 0.1, "adam": 1e-3, "sgd_ls": 1.0}  # when --optimizer switches the optimizer


def batch_loss_at_current_weights(model: MLP, x, y) -> float:
    """Closure for the line search: loss of the batch, forward pass only."""
    return SoftmaxCrossEntropy().forward(model.forward(x), y)


def preprocess(images: np.ndarray, config: dict) -> np.ndarray:
    """(N, 28, 28) uint8 -> (N, 784) float32, standardised with the training-set mean/std."""
    norm = config["normalize"]
    x = images.reshape(len(images), -1).astype(np.float32) / 255.0
    return (x - norm["mean"]) / norm["std"]


def predict(model: MLP, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Logits for x in batches."""
    batches = range(0, len(x), batch_size)
    return array_module(x).concatenate([model.forward(x[i : i + batch_size]) for i in batches])


def evaluate(model: MLP, x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Mean loss, accuracy and probabilities (as NumPy array); x and y on the model's device."""
    logits = predict(model, x)
    loss = SoftmaxCrossEntropy().forward(logits, y)
    return loss, float((logits.argmax(axis=1) == y).mean()), to_numpy(softmax(logits))


def train(config: dict, dataset: dict, root: Path | None = None, name: str | None = None) -> dict:
    """Train on dataset["train"], select on "val", report "test"; returns the final metrics."""
    xp = get_xp(config.get("device", "cpu"))
    on_gpu = xp is not np
    rng = np.random.default_rng(config["seed"])
    model = MLP(config["layers"], rng, xp=xp)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = make_optimizer(config["optimizer"], model.params(), model.grads())
    delta = WeightDelta(**config["weight_delta"])
    _, images, y_train = dataset["train"]
    x_train, y_train = xp.asarray(preprocess(images, config)), xp.asarray(y_train)
    x_val = xp.asarray(preprocess(dataset["val"][1], config))
    y_val = xp.asarray(dataset["val"][2])
    batch_size, log_every = config["batch_size"], config["log_every"]
    full_batch = bool(config.get("full_batch"))
    if full_batch:
        batch_size, log_every = len(y_train), 1
    line_search = isinstance(optimizer, SGDLineSearch)

    with RunLogger(config, name=name, root=root) as run:
        best_acc, step = -1.0, 0
        window_loss = window_correct = window_n = 0.0
        window_ls = {"ls_step": 0.0, "ls_factor": 0.0, "ls_backtracks": 0.0}
        t = time.time()
        delta(0, model.params())
        for epoch in range(1, config["epochs"] + 1):
            perm = np.arange(len(y_train)) if full_batch else rng.permutation(len(y_train))
            order = xp.asarray(perm)
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                xb, yb = x_train[idx], y_train[idx]
                logits = model.forward(xb)
                loss = loss_fn.forward(logits, yb)
                if not np.isfinite(loss):
                    raise FloatingPointError(f"loss is {loss} at step {step + 1}")
                model.backward(loss_fn.backward())
                if line_search:
                    batch_loss = partial(batch_loss_at_current_weights, model, xb, yb)
                    optimizer.step(batch_loss, loss)
                    window_ls["ls_step"] += optimizer.last["step_size"]
                    window_ls["ls_factor"] += optimizer.last["factor"]
                    window_ls["ls_backtracks"] += optimizer.last["backtracks"]
                else:
                    optimizer.step()
                step += 1
                window_loss += loss * len(yb)
                window_correct += float((logits.argmax(axis=1) == yb).sum())
                window_n += len(yb)
                if step % log_every == 0:
                    run.log(
                        step=step,
                        epoch=epoch,
                        split="train",
                        loss=window_loss / window_n,
                        acc=window_correct / window_n,
                        lr=optimizer.lr,
                        img_s=round(window_n / (time.time() - t), 1),
                        **({"gpu_mem_gb": round(gpu_memory_gb(), 3)} if on_gpu else {}),
                        **({k: v / log_every for k, v in window_ls.items()} if line_search else {}),
                    )
                    window_loss = window_correct = window_n = 0.0
                    window_ls = dict.fromkeys(window_ls, 0.0)
                    t = time.time()
                if d := delta(step, model.params()):
                    run.log(step=step, epoch=epoch, split="train", **d)
            val_loss, val_acc, _ = evaluate(model, x_val, y_val)
            run.log(step=step, epoch=epoch, split="val", loss=val_loss, acc=val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                model.save(run.dir / "checkpoints" / "best.npz")
            t = time.time()  # evaluation time does not count towards img/s
        model.save(run.dir / "checkpoints" / "last.npz")

        model.load(run.dir / "checkpoints" / "best.npz")
        results, losses = {}, {}
        for split in ("val", "test"):
            ids, images, labels = dataset[split]
            x = xp.asarray(preprocess(images, config))
            loss, acc, probs = evaluate(model, x, xp.asarray(labels))
            frame = predictions_frame(ids, labels, probs)
            frame.to_parquet(run.dir / "predictions" / f"{split}.parquet", index=False)
            results[split] = classification_metrics(labels, frame["pred"].to_numpy())
            losses[split] = loss
        test = results["test"]
        run.log(
            step=step, split="test", loss=losses["test"], acc=test["acc"], macro_f1=test["macro_f1"]
        )
        print(f"{run.run_id}: best val acc {best_acc:.4f}, test acc {test['acc']:.4f}")
    return results


def load_config(
    path: Path,
    optimizer: str | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    full_batch: bool = False,
) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if device is not None:
        config["device"] = device
    if batch_size is not None:
        config["batch_size"] = batch_size
    if full_batch:
        config["full_batch"] = True
    if optimizer and optimizer != config["optimizer"]["name"]:
        config["optimizer"] = {"name": optimizer, "lr": DEFAULT_LR[optimizer]}
    if lr is not None:
        config["optimizer"]["lr"] = lr
    if epochs is not None:
        config["epochs"] = epochs
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--optimizer", choices=OPTIMIZERS)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=DEVICES, help="cpu (NumPy) or gpu (CuPy)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--full-batch", action="store_true", help="each step on the whole train set"
    )
    parser.add_argument("--name", help="run name suffix (default: <model>-<optimizer>[-gpu])")
    args = parser.parse_args(argv)
    config = load_config(
        args.config,
        args.optimizer,
        args.lr,
        args.epochs,
        args.device,
        args.batch_size,
        args.full_batch,
    )
    full = "-fullbatch" if config.get("full_batch") else ""
    gpu = "-gpu" if config.get("device") == "gpu" else ""
    name = args.name or f"{config['model']}-{config['optimizer']['name']}{full}{gpu}"
    train(config, load_dataset(), name=name)


if __name__ == "__main__":
    main()
