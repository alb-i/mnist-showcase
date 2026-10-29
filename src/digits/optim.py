"""SGD with Armijo backtracking line search for PyTorch (D14); same algorithm as
digits.numpy_net.optim.SGDLineSearch:

    t <- lr;  while L(w - t g) > L(w) - c t ||g||^2:  t <- beta t;   w <- w - t g

    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step(closure, loss.item())    # closure(): loss at the current weights, no backward

L is the loss on the same mini-batch the gradient was taken on, so the closure has to reproduce
that loss exactly (same dropout masks, no update of BatchNorm statistics); digits.train does this.
"""

import torch


class SGDLineSearch(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1.0, c: float = 1e-4, beta: float = 0.5,
                 max_backtracks: int = 30):  # fmt: skip
        super().__init__(params, {"lr": lr, "c": c, "beta": beta, "max_backtracks": max_backtracks})
        self.last = {"step_size": 0.0, "factor": 0.0, "backtracks": 0}

    @torch.no_grad()
    def step(self, closure, loss: float) -> float:
        """Returns the loss at the new weights. One parameter group (its lr, c, beta) is used;
        if no step is accepted after max_backtracks halvings, the weights stay unchanged."""
        (group,) = self.param_groups
        params = [p for p in group["params"] if p.grad is not None]
        start = [p.detach().clone() for p in params]
        g2 = sum(p.grad.double().pow(2).sum() for p in params).item()

        def move(t: float) -> None:
            for p, w in zip(params, start, strict=True):
                p.copy_(w).add_(p.grad, alpha=-t)

        t = group["lr"]
        for k in range(group["max_backtracks"] + 1):
            move(t)
            new_loss = float(closure())
            if new_loss <= loss - group["c"] * t * g2:
                self.last = {"step_size": t, "factor": group["beta"] ** k, "backtracks": k}
                return new_loss
            t *= group["beta"]
        move(0.0)
        self.last = {"step_size": 0.0, "factor": 0.0, "backtracks": group["max_backtracks"] + 1}
        return loss
