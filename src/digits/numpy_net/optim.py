"""SGD with momentum, Adam and SGD with Armijo line search.

All update the parameter arrays in place from the gradient arrays of the same order.

    SGD:  v <- mu v + g,  w <- w - lr v                          (PyTorch convention)
    Adam: m <- b1 m + (1 - b1) g,  v <- b2 v + (1 - b2) g^2,
          m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t),
          w <- w - lr m_hat / (sqrt(v_hat) + eps)
    SGD with line search (D14): t <- lr; while L(w - t g) > L(w) - c t ||g||^2: t <- beta t;
          w <- w - t g. L is the loss on the same mini-batch the gradient g was taken on.

The state arrays live on the device of the parameters (NumPy or CuPy).
"""

from digits.numpy_net.backend import array_module


class SGD:
    def __init__(self, params, grads, lr: float, momentum: float = 0.0):
        self.params, self.grads = params, grads
        self.lr, self.momentum = lr, momentum
        self.velocity = [array_module(p).zeros_like(p) for p in params]

    def step(self) -> None:
        for p, g, v in zip(self.params, self.grads, self.velocity, strict=True):
            v *= self.momentum
            v += g
            p -= self.lr * v


class Adam:
    def __init__(
        self,
        params,
        grads,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params, self.grads = params, grads
        self.lr, (self.beta1, self.beta2), self.eps = lr, betas, eps
        self.m = [array_module(p).zeros_like(p) for p in params]
        self.v = [array_module(p).zeros_like(p) for p in params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        c1, c2 = 1 - self.beta1**self.t, 1 - self.beta2**self.t
        for p, g, m, v in zip(self.params, self.grads, self.m, self.v, strict=True):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            p -= self.lr * (m / c1) / (array_module(p).sqrt(v / c2) + self.eps)


class SGDLineSearch:
    """Gradient step with Armijo backtracking: start every iteration at t = lr and halve t
    (factor beta) until the sufficient-decrease condition

        L(w - t g) <= L(w) - c t ||g||^2

    holds. For a smooth L it holds for all small t, because L(w - t g) = L(w) - t ||g||^2 + O(t^2);
    for a convex L with Lipschitz gradient (constant K) every t <= 2 (1 - c) / K is accepted, so
    the accepted steps are bounded below and gradient descent converges to the minimum.
    If no t is accepted after max_backtracks halvings (rounding, kinks of ReLU), w stays unchanged.

    step(closure, loss): loss = L(w) at the current weights, closure() evaluates L at the
    current weights (forward pass only, same mini-batch). After a step, `last` holds the accepted
    step size (0 if none), the shortening factor step size / lr (= beta^halvings) and the number of
    halvings.
    """

    def __init__(
        self,
        params,
        grads,
        lr: float = 1.0,
        c: float = 1e-4,
        beta: float = 0.5,
        max_backtracks: int = 30,
    ):
        self.params, self.grads = params, grads
        self.lr, self.c, self.beta, self.max_backtracks = lr, c, beta, max_backtracks
        self.start = [array_module(p).empty_like(p) for p in params]
        self.last = {"step_size": 0.0, "factor": 0.0, "backtracks": 0}

    def _move(self, t: float) -> None:
        for p, w, g in zip(self.params, self.start, self.grads, strict=True):
            p[...] = w - t * g

    def step(self, closure, loss: float) -> float:
        """Returns the loss at the new weights."""
        g2 = sum(float((g.astype("float64") ** 2).sum()) for g in self.grads)
        for p, w in zip(self.params, self.start, strict=True):
            w[...] = p
        t = self.lr
        for k in range(self.max_backtracks + 1):
            self._move(t)
            new_loss = closure()
            if new_loss <= loss - self.c * t * g2:
                self.last = {"step_size": t, "factor": self.beta**k, "backtracks": k}
                return new_loss
            t *= self.beta
        self._move(0.0)
        self.last = {"step_size": 0.0, "factor": 0.0, "backtracks": self.max_backtracks + 1}
        return loss


def make_optimizer(config: dict, params, grads):
    """From config["optimizer"]: {name: sgd, lr, momentum}, {name: adam, lr, betas, eps} or
    {name: sgd_ls, lr, c, beta, max_backtracks}."""
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if config["name"] == "sgd":
        return SGD(params, grads, **kwargs)
    if config["name"] == "sgd_ls":
        return SGDLineSearch(params, grads, **kwargs)
    if config["name"] == "adam":
        if "betas" in kwargs:
            kwargs["betas"] = tuple(kwargs["betas"])
        return Adam(params, grads, **kwargs)
    raise ValueError(f"unknown optimizer {config['name']}")
