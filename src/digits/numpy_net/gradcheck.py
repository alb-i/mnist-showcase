"""Numerical gradient check with central differences.

g_num[i] = (L(w + eps e_i) - L(w - eps e_i)) / (2 eps)
error    = ||g_num - g_bp|| / (||g_num|| + ||g_bp||)      (plan 7.2: < 1e-6 in float64)
"""

from collections.abc import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gradient of f() with respect to the array x, which f reads; x is restored afterwards."""
    g = np.zeros_like(x, dtype=np.float64)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        f_plus = f()
        x[i] = old - eps
        f_minus = f()
        x[i] = old
        g[i] = (f_plus - f_minus) / (2 * eps)
    return g


def relative_error(g_num: np.ndarray, g_bp: np.ndarray) -> float:
    denom = np.linalg.norm(g_num) + np.linalg.norm(g_bp)
    return 0.0 if denom == 0 else float(np.linalg.norm(g_num - g_bp) / denom)


def check_gradients(model, loss, x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> list[float]:
    """Relative error per parameter array of model (backprop vs. central differences)."""
    loss.forward(model.forward(x), y)
    model.backward(loss.backward())
    backprop = [g.copy() for g in model.grads()]

    def f() -> float:
        return loss.forward(model.forward(x), y)

    return [
        relative_error(numerical_gradient(f, p, eps), g)
        for p, g in zip(model.params(), backprop, strict=True)
    ]
