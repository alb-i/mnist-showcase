import numpy as np
import pytest

from digits.numpy_net.gradcheck import check_gradients, numerical_gradient, relative_error
from digits.numpy_net.layers import MLP, Dense
from digits.numpy_net.losses import SoftmaxCrossEntropy


def small_problem(seed=0):
    rng = np.random.default_rng(seed)
    model = MLP([6, 5, 4, 3], rng, dtype=np.float64)
    for p in model.params():  # non-zero biases, so their gradients are tested too
        p += 0.1 * rng.standard_normal(p.shape)
    return model, rng.standard_normal((8, 6)), rng.integers(0, 3, 8)


def test_numerical_gradient_of_quadratic():
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(numerical_gradient(lambda: float((x**2).sum()), x), 2 * x)
    np.testing.assert_array_equal(x, [1.0, -2.0, 3.0])  # restored


def test_backprop_matches_central_differences():
    model, x, y = small_problem()
    errors = check_gradients(model, SoftmaxCrossEntropy(), x, y)
    assert len(errors) == 6  # W and b of three dense layers
    assert max(errors) < 1e-6, errors


def test_gradcheck_catches_a_wrong_backward(monkeypatch):
    def wrong_backward(self, dz):
        self.dW[...] = self.x.T @ dz
        self.db[...] = 0.5 * dz.sum(axis=0)  # bug: factor 0.5
        return dz @ self.W.T

    monkeypatch.setattr(Dense, "backward", wrong_backward)
    model, x, y = small_problem()
    errors = check_gradients(model, SoftmaxCrossEntropy(), x, y)
    assert max(errors[1::2]) > 0.1  # bias gradients are wrong
    assert max(errors[0::2]) < 1e-6  # the dW stay right: dX does not depend on db


def test_relative_error():
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0
    assert relative_error(np.ones(3), -np.ones(3)) == pytest.approx(1.0)
