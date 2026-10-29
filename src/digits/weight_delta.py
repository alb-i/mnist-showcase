"""Weight change between snapshots: sum of |w_now - w_k_steps_ago| over all parameters (D8).

    delta = WeightDelta(**config["weight_delta"])   # every: 100, lags: [100, 500]
    delta(0, model.parameters())                    # snapshot of the initial weights
    for step in range(1, n + 1):
        ...
        if d := delta(step, model.parameters()):    # {"weight_delta_100": ..., ...}
            run.log(step=step, split="train", **d)

Every `every` steps a snapshot is taken and compared with the snapshots `lag` steps ago; a lag only
appears once that many steps have passed (weight_delta_500 first at step 500).
Works for PyTorch tensors and NumPy/CuPy arrays (own NumPy net); snapshots stay on their device.
"""

from collections import deque
from collections.abc import Iterable

import numpy as np


def is_tensor(p) -> bool:
    return hasattr(p, "detach")


def snapshot(p):
    return p.detach().clone() if is_tensor(p) else p.copy()


def l1_distance(a, b) -> float:
    if not is_tensor(a):  # NumPy or CuPy array
        return float(abs(a - b).sum(dtype=np.float64))
    import torch

    return (a - b).abs().sum(dtype=torch.float64).item()


class WeightDelta:
    def __init__(self, every: int = 100, lags: Iterable[int] = (100,)):
        self.every = every
        self.lags = sorted(lags)
        if any(lag <= 0 or lag % every for lag in self.lags):
            raise ValueError(f"lags {self.lags} must be positive multiples of every={every}")
        # snapshots of the last max(lag) steps, newest last
        self.history: deque = deque(maxlen=self.lags[-1] // every)

    def __call__(self, step: int, params: Iterable) -> dict[str, float]:
        """Every `every` steps: {"weight_delta_<lag>": L1 distance to the snapshot `lag` steps
        ago} for all lags reached so far; empty dict otherwise and at the first snapshot."""
        if step % self.every:
            return {}
        current = [snapshot(p) for p in params]
        out = {}
        for lag in self.lags:
            k = lag // self.every
            if len(self.history) >= k:
                previous = self.history[-k]
                if len(current) != len(previous):
                    raise ValueError("parameter list changed between snapshots")
                out[f"weight_delta_{lag}"] = sum(
                    l1_distance(a, b) for a, b in zip(current, previous, strict=True)
                )
        self.history.append(current)
        return out
