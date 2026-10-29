"""Dense and activation layers with hand-derived backward passes.

Row-vector convention: a batch X has shape (N, d_in) and W has shape (d_in, d_out), i.e. W here
is the transpose of the W in z = Wx + b (plan 7.2). Each layer caches what its backward pass
needs; backward receives dL/d(output) and returns dL/d(input).

    Dense: Z = X W + b       dW = X^T dZ,  db = sum_n dZ,  dX = dZ W^T
    ReLU:  A = max(Z, 0)     dZ = dA * [Z > 0]

xp is numpy or cupy (digits.numpy_net.backend); the initial weights always come from the NumPy rng,
so a seed gives the same network on both devices.
"""

import numpy as np

from digits.numpy_net.backend import array_module, to_numpy


class Dense:
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, dtype=np.float32, xp=np):
        # He initialisation: Var(W) = 2 / n_in keeps the activation variance constant under ReLU
        W = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(dtype)
        self.W = xp.asarray(W)
        self.b = xp.zeros(n_out, dtype=dtype)
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, dz: np.ndarray) -> np.ndarray:
        self.dW[...] = self.x.T @ dz
        self.db[...] = dz.sum(axis=0)
        return dz @ self.W.T

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU:
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.mask = z > 0
        return z * self.mask

    def backward(self, da: np.ndarray) -> np.ndarray:
        return da * self.mask

    def params(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []


class MLP:
    """Dense layers with ReLU in between; forward returns the logits (softmax is in the loss)."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, dtype=np.float32, xp=np):
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            self.layers.append(Dense(n_in, n_out, rng, dtype, xp))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dlogits: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dlogits = layer.backward(dlogits)

    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params()]

    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads()]

    def save(self, path) -> None:
        """Always NumPy arrays in the .npz, so a GPU checkpoint loads on the CPU and vice versa."""
        np.savez(path, *[to_numpy(p) for p in self.params()])

    def load(self, path) -> None:
        with np.load(path) as f:
            for i, p in enumerate(self.params()):
                p[...] = array_module(p).asarray(f[f"arr_{i}"])
