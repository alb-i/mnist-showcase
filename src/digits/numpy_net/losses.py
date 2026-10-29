"""Softmax cross-entropy.

For logits z (N, K), one-hot targets y and p = softmax(z):
    L = -(1/N) sum_n log p[n, y_n],   dL/dz = (p - y) / N
log p is computed as z - logsumexp(z) with the row maximum subtracted, so large logits do not
overflow. Works on NumPy and CuPy arrays (y must be on the same device as the logits).
"""

import numpy as np

from digits.numpy_net.backend import array_module


def log_softmax(z: np.ndarray) -> np.ndarray:
    xp = array_module(z)
    z = z - z.max(axis=1, keepdims=True)
    return z - xp.log(xp.exp(z).sum(axis=1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return array_module(z).exp(log_softmax(z))


class SoftmaxCrossEntropy:
    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Mean loss over the batch; y holds class indices (N,)."""
        xp = array_module(logits)
        logp = log_softmax(logits)
        self.probs, self.y = xp.exp(logp), y
        return float(-logp[xp.arange(len(y)), y].mean())

    def backward(self) -> np.ndarray:
        xp = array_module(self.probs)
        g = self.probs.copy()
        g[xp.arange(len(self.y)), self.y] -= 1
        return g / len(self.y)
