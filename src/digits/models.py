"""PyTorch models: MLP, small CNN, regularised CNN (plan 7.3).

All models take normalised images (N, 1, 28, 28) and return logits (N, 10).

    mlp        784-256-128-10 with ReLU and He initialisation, like the NumPy MLP (E3 vs. E2)
    cnn_small  conv(32)-conv(64)-pool-dropout-fc(128)-fc(10)                          (E4)
    cnn_reg    two conv blocks with BatchNorm and dropout, then fc(128)-fc(10)     (E5-E7)
"""

from torch import nn


class MLP(nn.Sequential):
    def __init__(self, layers: list[int]):
        modules: list[nn.Module] = [nn.Flatten()]
        for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:], strict=True)):
            linear = nn.Linear(n_in, n_out)
            # He initialisation as in digits.numpy_net: N(0, 2 / n_in), zero bias
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            if i < len(layers) - 2:
                modules.append(nn.ReLU())
        super().__init__(*modules)


class CNNSmall(nn.Sequential):
    """LeNet-style: 28 -> 26 -> 24 -> pool 12; 64 * 12 * 12 = 9216 features."""

    def __init__(self, dropout: float = 0.25):
        super().__init__(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )


def conv_bn(c_in: int, c_out: int, batchnorm: bool = True) -> list[nn.Module]:
    """conv3x3-BN-ReLU; without BatchNorm the convolution gets its own bias (E6 ablation)."""
    if not batchnorm:
        return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU()]
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]


class CNNReg(nn.Sequential):
    """Two blocks of 2 x (conv3x3-BN-ReLU) + max-pool + dropout: 28 -> 14 -> 7, then
    fc(128)-BN-ReLU-dropout-fc(10). batchnorm=False drops all BatchNorm layers (E6 ablation);
    dropout 0 turns dropout off."""

    def __init__(self, dropout_conv: float = 0.1, dropout_fc: float = 0.3, batchnorm: bool = True):
        fc = [nn.Linear(64 * 7 * 7, 128, bias=not batchnorm)]
        if batchnorm:
            fc.append(nn.BatchNorm1d(128))
        super().__init__(
            *conv_bn(1, 32, batchnorm),
            *conv_bn(32, 32, batchnorm),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_conv),
            *conv_bn(32, 64, batchnorm),
            *conv_bn(64, 64, batchnorm),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_conv),
            nn.Flatten(),
            *fc,
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(128, 10),
        )


def build_model(config: dict) -> nn.Module:
    name, kwargs = config["model"], config.get("model_args", {})
    if name == "mlp":
        return MLP(config["layers"])
    if name == "cnn_small":
        return CNNSmall(**kwargs)
    if name == "cnn_reg":
        return CNNReg(**kwargs)
    raise ValueError(f"unknown model {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
