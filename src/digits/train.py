"""Training loop for the PyTorch models (plan 8.1; experiments E3-E7).

    micromamba run -n mnist-showcase train configs/cnn_small.yaml
    micromamba run -n mnist-showcase train configs/mlp.yaml --optimizer adam --lr 1e-3
    micromamba run -n mnist-showcase train configs/cnn_reg.yaml --set augment=null --set amp=false
    micromamba run -n mnist-showcase train --resume $RUNS_DIR/mnist/<run-id> [--epochs 40]
    micromamba run -n mnist-showcase train configs/mlp.yaml --optimizer sgd_ls --set amp=false
    micromamba run -n mnist-showcase train configs/mlp.yaml --optimizer sgd_ls --set amp=false \
        --full-batch --epochs 500           # full-batch gradient descent, one step per epoch
    micromamba run -n mnist-showcase train configs/cnn_reg.yaml --fold 2 --set checkpoint=last

The whole dataset lives on the device as uint8 (47 MB); each batch is scaled to [0, 1], augmented
on the device (random affine: rotation, shift, scale) and standardised with config["normalize"].
Logs like every run (digits.runlog): train loss/acc/lr/img_s/gpu_mem_gb every `log_every` steps,
weight_delta every 100 steps (D8), val loss/acc every epoch. Checkpoints: best.pt (best val acc,
weights only) and last.pt (everything needed for --resume, written after every epoch). Optional
early stopping on the val loss. At the end the best checkpoint is evaluated on val and test and
written to predictions/. With augmentation, a new run first writes samples/augmented.png (D17): 64
training images as the model gets them, for the visual check in the workbench.

checkpoint: last keeps the weights of the last epoch instead of the best val epoch (no best.pt).
--fold k (config cv_fold, D18) trains on the cross-validation folds other than k (all 60 000
training images, digits.data.load_cv_dataset) and uses fold k as "val": its predictions/val.parquet
are out-of-fold predictions. Use it with checkpoint: last, so the fold does not select the model.

Batch order and augmentation come from a generator seeded with (seed, epoch), so a resumed run
sees the same batches as an uninterrupted one; weight_delta starts afresh after a resume.

With the optimizer sgd_ls (SGD with Armijo line search, D14) every trial step re-evaluates the
loss of the same batch: with the same dropout masks (the RNG state of the training forward pass
is replayed) and without changing the BatchNorm statistics (restored after the line search).
It needs amp: false, because float16 rounding makes the comparison of losses unreliable.
ls_step, ls_factor and ls_backtracks (mean accepted step size, shortening factor accepted / start
step, halvings per step) are logged as well.

full_batch: true (--full-batch) makes every step use the whole training set (no shuffling; one
step per epoch, logged every step). Forward and backward run in chunks of `chunk_size` images
(default 2048, bounded GPU memory); the chunk gradients, weighted with chunk size / N, add up in
.grad to the gradient of the mean loss over all N images. BatchNorm uses per-chunk statistics.
"""

import argparse
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from digits.data import load_cv_dataset, load_dataset
from digits.evaluate import classification_metrics, predictions_frame
from digits.models import build_model
from digits.optim import SGDLineSearch
from digits.runlog import RunLogger
from digits.samples import image_grid
from digits.weight_delta import WeightDelta

PIXELS_PER_UNIT = 2 / 28  # affine_grid works in [-1, 1] over 28 pixels
OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd_ls": SGDLineSearch,
}
DEFAULT_LR = {"sgd": 0.1, "adam": 1e-3, "adamw": 1e-3, "sgd_ls": 1.0}  # when --optimizer switches


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seeds Python, NumPy and Torch; deterministic kernels for final runs (somewhat slower)."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def affine_theta(angle_deg, scale, shift_x_px, shift_y_px) -> torch.Tensor:
    """(N, 2, 3) matrices for F.affine_grid. The image is rotated by angle, scaled by scale and
    shifted by (x, y) pixels (y down); affine_grid needs the inverse map (output -> input),
    A^-1 = R(-angle) / scale and t_in = -A^-1 t."""
    a = torch.deg2rad(angle_deg)
    cos, sin = torch.cos(a), torch.sin(a)
    inv = torch.stack([torch.stack([cos, sin], 1), torch.stack([-sin, cos], 1)], 1)
    inv = inv / scale[:, None, None]
    t = torch.stack([shift_x_px, shift_y_px], 1)[:, :, None] * PIXELS_PER_UNIT
    return torch.cat([inv, -(inv @ t)], 2)


def random_affine(x: torch.Tensor, config: dict, generator: torch.Generator) -> torch.Tensor:
    """Random rotation (+-rotation_deg), shift (+-shift_px) and scale ([lo, hi]) per image of
    x (N, 1, 28, 28) in [0, 1]; areas from outside the image are black. No flips."""

    def uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(len(x), device=x.device).uniform_(lo, hi, generator=generator)

    r, s = config["rotation_deg"], config["shift_px"]
    theta = affine_theta(uniform(-r, r), uniform(*config["scale"]), uniform(-s, s), uniform(-s, s))
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def to_device(dataset: dict, device: torch.device) -> dict:
    """split -> (images uint8 (N, 1, 28, 28), labels int64) on the device."""
    return {
        split: (
            torch.as_tensor(images[:, None]).to(device),
            torch.as_tensor(labels.astype(np.int64)).to(device),
        )
        for split, (_, images, labels) in dataset.items()
    }


def make_optimizer(config: dict, params) -> torch.optim.Optimizer:
    """From config["optimizer"]: {name: sgd|adam|adamw|sgd_ls, lr, ...} (momentum, weight_decay,
    betas; c, beta, max_backtracks for sgd_ls)."""
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if "betas" in kwargs:
        kwargs["betas"] = tuple(kwargs["betas"])
    if config["name"] not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {config['name']}")
    return OPTIMIZERS[config["name"]](params, **kwargs)


def make_scheduler(name: str | None, optimizer, total_steps: int):
    """Per batch: onecycle (max lr = optimizer lr), cosine to 0, None = constant."""
    if name is None:
        return None
    if name == "onecycle":
        lr = optimizer.param_groups[0]["lr"]
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    raise ValueError(f"unknown scheduler {name}")


class Normalize:
    def __init__(self, config: dict):
        self.mean, self.std = config["normalize"]["mean"], config["normalize"]["std"]

    def __call__(self, x_uint8: torch.Tensor, augment=None, generator=None) -> torch.Tensor:
        x = x_uint8.float() / 255.0
        if augment:
            x = random_affine(x, augment, generator)
        return (x - self.mean) / self.std


def write_augmented_batch(path: Path, x, y, normalize: Normalize, augment: dict, seed: int,
                          n: int = 64) -> None:  # fmt: skip
    """PNG of n training images augmented and standardised like a training batch, mapped back
    with the config mean/std (so a wrong normalisation shows). Own generator: the training RNG
    and thus the run stay unchanged."""
    generator = torch.Generator(x.device).manual_seed(seed)
    idx = torch.randperm(len(y), device=x.device, generator=generator)[:n]
    images = normalize(x[idx], augment, generator) * normalize.std + normalize.mean
    image_grid(images[:, 0].cpu().numpy(), y[idx].tolist()).save(path)


def rng_state(device: torch.device) -> tuple:
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.get_rng_state(), cuda


def line_search_step(optimizer, model, chunks: list, loss: float, label_smoothing: float) -> float:
    """optimizer.step(closure, loss) where the closure reproduces the training loss of the batch
    at the trial weights. chunks: (x, y, rng) per chunk of the batch, rng the RNG state taken
    before the chunk's training forward pass (same dropout masks); the loss is the chunk losses
    weighted with chunk size / batch size. BatchNorm buffers are restored afterwards (trial
    forwards must not count). Returns the loss at the new weights."""
    device = chunks[0][0].device
    n = sum(len(y) for _, y, _ in chunks)
    buffers = [b.clone() for b in model.buffers()]

    def closure() -> float:
        total = torch.zeros((), dtype=torch.float64, device=device)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            for x, y, (cpu_rng, cuda_rng) in chunks:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, device)
                chunk_loss = F.cross_entropy(model(x), y, label_smoothing=label_smoothing)
                total += chunk_loss.double() * (len(y) / n)
        return total.item()

    new_loss = optimizer.step(closure, loss)
    with torch.no_grad():
        for b, saved in zip(model.buffers(), buffers, strict=True):
            b.copy_(saved)
    return new_loss


def train_step(model, optimizer, scaler, xb, yb, *, chunk_size: int, label_smoothing: float,
               amp: bool, line_search: bool):  # fmt: skip
    """One optimiser step on the batch (xb, yb). Forward and backward run in chunks; the
    gradients of the chunk losses weighted with chunk size / batch size add up in .grad to the
    gradient of the mean loss over the batch. Returns the mean loss and the number of correct
    predictions (tensors, no synchronisation)."""
    n, device = len(yb), xb.device
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), device=device)
    chunks = []
    starts = list(range(0, n, chunk_size))
    if len(starts) > 1 and n - starts[-1] == 1:  # no chunk of 1 (BatchNorm needs 2 samples)
        starts.pop()
    for i, end in zip(starts, starts[1:] + [n], strict=True):
        x, y = xb[i:end], yb[i:end]
        rng = rng_state(device) if line_search else None
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        scaler.scale(loss * (len(y) / n) if len(y) < n else loss).backward()
        loss_sum += loss.detach().double() * len(y)
        correct += (logits.argmax(1) == y).sum()
        chunks.append((x, y, rng))
    if line_search:
        line_search_step(optimizer, model, chunks, (loss_sum / n).item(), label_smoothing)
    else:
        scaler.step(optimizer)
    scaler.update()
    return loss_sum / n, correct


@torch.no_grad()
def evaluate(model, x, y, normalize: Normalize, amp: bool, batch_size: int = 4096):
    """Mean cross-entropy (without label smoothing), accuracy and probabilities (NumPy)."""
    model.eval()
    logits = []
    for i in range(0, len(x), batch_size):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
            logits.append(model(normalize(x[i : i + batch_size])).float())
    logits = torch.cat(logits)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc, torch.softmax(logits, 1).cpu().numpy()


def train(
    config: dict,
    dataset: dict,
    root: Path | None = None,
    name: str | None = None,
    resume: Path | None = None,
) -> dict:
    """Train on dataset["train"], select on "val", report "test"; returns the final metrics per
    split and the run folder (key run_dir). With resume (a run folder), training continues from
    its last.pt up to config["epochs"]."""
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = config["seed"]
    seed_everything(seed, config.get("deterministic", False))
    amp = bool(config.get("amp")) and device.type == "cuda"
    model = build_model(config).to(device)
    optimizer = make_optimizer(config["optimizer"], model.parameters())
    line_search = isinstance(optimizer, SGDLineSearch)
    if line_search and amp:
        raise ValueError("sgd_ls needs amp: false (--set amp=false)")
    data = to_device(dataset, device)
    x_train, y_train = data["train"]
    batch_size, log_every = config["batch_size"], config.get("log_every", 50)
    full_batch = bool(config.get("full_batch"))
    chunk_size = config.get("chunk_size", 2048)
    if full_batch:
        batch_size, log_every = len(y_train), 1
    else:
        chunk_size = batch_size
    steps_per_epoch = math.ceil(len(y_train) / batch_size)
    scheduler = make_scheduler(
        config.get("scheduler"), optimizer, config["epochs"] * steps_per_epoch
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    normalize, augment = Normalize(config), config.get("augment")
    label_smoothing = config.get("label_smoothing", 0.0)
    patience = config.get("early_stopping_patience")
    keep_last = config.get("checkpoint", "best") == "last"
    if config.get("checkpoint", "best") not in ("best", "last"):
        raise ValueError(f"checkpoint must be best or last, got {config['checkpoint']}")
    delta = WeightDelta(**config["weight_delta"])
    state = {"epoch": 0, "step": 0, "best_acc": -1.0, "best_loss": math.inf, "bad_epochs": 0}

    if resume is not None:
        ckpt = torch.load(Path(resume) / "checkpoints" / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["rng"]["cpu"])
        if device.type == "cuda" and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(ckpt["rng"]["cuda"], device)
        state = ckpt["state"]
        logger = RunLogger(config, resume=resume, resume_step=state["step"])
    else:
        logger = RunLogger(config, name=name, root=root)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # OneCycle/cosine step after an optimizer step that GradScaler skipped (inf grads) warn
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")

    with logger as run:
        step = state["step"]
        if resume is None:
            delta(0, model.parameters())
            if augment:
                (run.dir / "samples").mkdir()
                path = run.dir / "samples" / "augmented.png"
                write_augmented_batch(path, x_train, y_train, normalize, augment, seed)
        ckpt_dir = run.dir / "checkpoints"
        for epoch in range(state["epoch"] + 1, config["epochs"] + 1):
            model.train()
            generator = torch.Generator(device).manual_seed(seed * 100_003 + epoch)
            if full_batch:
                order = torch.arange(len(y_train), device=device)
            else:
                order = torch.randperm(len(y_train), device=device, generator=generator)
            window_loss = torch.zeros((), device=device)
            window_correct = torch.zeros((), device=device)
            window_n, t = 0, time.time()
            window_ls = {"ls_step": 0.0, "ls_factor": 0.0, "ls_backtracks": 0.0}  # sums
            window_steps = 0
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                xb, yb = normalize(x_train[idx], augment, generator), y_train[idx]
                loss, correct = train_step(
                    model,
                    optimizer,
                    scaler,
                    xb,
                    yb,
                    chunk_size=chunk_size,
                    label_smoothing=label_smoothing,
                    amp=amp,
                    line_search=line_search,
                )
                if line_search:
                    window_ls["ls_step"] += optimizer.last["step_size"]
                    window_ls["ls_factor"] += optimizer.last["factor"]
                    window_ls["ls_backtracks"] += optimizer.last["backtracks"]
                window_steps += 1
                if scheduler is not None:
                    scheduler.step()
                step += 1
                window_loss += loss.float() * len(yb)
                window_correct += correct
                window_n += len(yb)
                if step % log_every == 0:
                    mean_loss = (window_loss / window_n).item()  # synchronises the device
                    if not math.isfinite(mean_loss):
                        raise FloatingPointError(f"loss is {mean_loss} at step {step}")
                    gpu = {}
                    if device.type == "cuda":
                        gpu["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
                    run.log(
                        step=step,
                        epoch=epoch,
                        split="train",
                        loss=mean_loss,
                        acc=window_correct.item() / window_n,
                        lr=optimizer.param_groups[0]["lr"],
                        img_s=round(window_n / (time.time() - t), 1),
                        **gpu,
                        **(
                            {k: v / window_steps for k, v in window_ls.items()}
                            if line_search
                            else {}
                        ),
                    )
                    window_ls, window_steps = dict.fromkeys(window_ls, 0.0), 0
                    window_loss.zero_()
                    window_correct.zero_()
                    window_n, t = 0, time.time()
                if d := delta(step, model.parameters()):
                    run.log(step=step, epoch=epoch, split="train", **d)

            val_loss, val_acc, _ = evaluate(model, *data["val"], normalize, amp)
            run.log(step=step, epoch=epoch, split="val", loss=val_loss, acc=val_acc)
            if val_acc > state["best_acc"]:
                state["best_acc"] = val_acc
                if not keep_last:
                    torch.save(model.state_dict(), ckpt_dir / "best.pt")
            if val_loss < state["best_loss"]:
                state["best_loss"], state["bad_epochs"] = val_loss, 0
            else:
                state["bad_epochs"] += 1
            state.update(epoch=epoch, step=step)
            cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            last = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "rng": {"cpu": torch.get_rng_state(), "cuda": cuda_rng},
                "state": state,
            }
            torch.save(last, ckpt_dir / "last.pt")
            if patience is not None and state["bad_epochs"] >= patience:
                run.meta["early_stopped_epoch"] = epoch
                print(f"early stopping after epoch {epoch} (val loss {patience}x not better)")
                break

        if not keep_last:
            model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
        results, losses = {}, {}
        for split in ("val", "test"):
            ids, _, labels = dataset[split]
            loss, _, probs = evaluate(model, *data[split], normalize, amp)
            frame = predictions_frame(ids, labels, probs)
            frame.to_parquet(run.dir / "predictions" / f"{split}.parquet", index=False)
            results[split] = classification_metrics(labels, frame["pred"].to_numpy())
            losses[split] = loss
        test = results["test"]
        run.log(
            step=step, split="test", loss=losses["test"], acc=test["acc"], macro_f1=test["macro_f1"]
        )
        val_acc = results["val"]["acc"]
        chosen = f"last val acc {val_acc:.4f}" if keep_last else f"best val acc {val_acc:.4f}"
        print(f"{run.run_id}: {chosen}, test acc {test['acc']:.4f}")
    return {**results, "run_dir": run.dir}


def set_key(config: dict, dotted: str, value) -> None:
    *path, last = dotted.split(".")
    for key in path:
        config = config.setdefault(key, {})
    config[last] = value


def load_config(
    path: Path,
    optimizer: str | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    overrides: list[str] = (),
) -> dict:
    """YAML config plus overrides; --set key.sub=value parses YAML (null, 1e-3, [0.9, 1.1])."""
    with open(path) as f:
        config = yaml.safe_load(f)
    if optimizer and optimizer != config["optimizer"]["name"]:
        config["optimizer"] = {"name": optimizer, "lr": DEFAULT_LR[optimizer]}
    if lr is not None:
        config["optimizer"]["lr"] = lr
    if epochs is not None:
        config["epochs"] = epochs
    if seed is not None:
        config["seed"] = seed
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        set_key(config, key, yaml.safe_load(value))
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, nargs="?", help="YAML config (not with --resume)")
    parser.add_argument("--optimizer", choices=list(OPTIMIZERS))
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--deterministic", action="store_true", help="deterministic kernels")
    parser.add_argument(
        "--full-batch", action="store_true", help="each step on the whole train set"
    )
    parser.add_argument("--fold", type=int, help="cross-validation fold used as val (D18)")
    parser.add_argument("--name", help="run name suffix (default: <model>-<optimizer>)")
    parser.add_argument("--resume", type=Path, help="run folder to continue from its last.pt")
    args = parser.parse_args(argv)
    if (args.config is None) == (args.resume is None):
        parser.error("give either a config or --resume")
    path = args.config or args.resume / "config.yaml"
    config = load_config(path, args.optimizer, args.lr, args.epochs, args.seed, args.set)
    if args.deterministic:
        config["deterministic"] = True
    if args.full_batch:
        config["full_batch"] = True
    if args.fold is not None:
        config["cv_fold"] = args.fold
    full = "-fullbatch" if config.get("full_batch") else ""
    name = args.name or f"{config['model']}-{config['optimizer']['name']}{full}"
    fold = config.get("cv_fold")
    dataset = load_dataset() if fold is None else load_cv_dataset(fold)
    train(config, dataset, name=name, resume=args.resume)


if __name__ == "__main__":
    main()
