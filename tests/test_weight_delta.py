import numpy as np
import pytest
import torch

from digits.weight_delta import WeightDelta


def test_numpy_params():
    w, b = np.zeros((3, 2)), np.zeros(2)
    delta = WeightDelta(every=100, lags=[100])
    assert delta(0, [w, b]) == {}  # first snapshot
    w += 0.5  # in-place update, as an optimiser does
    b[0] = -1.0
    assert delta(50, [w, b]) == {}  # not a multiple of `every`
    assert delta(100, [w, b]) == {"weight_delta_100": pytest.approx(6 * 0.5 + 1.0)}
    assert delta(200, [w, b]) == {"weight_delta_100": 0.0}  # compares to step 100, not step 0


def test_two_lags():
    w = np.zeros(4)
    delta = WeightDelta(every=100, lags=[500, 100])
    delta(0, [w])
    out = {}
    for step in range(100, 801, 100):
        w += 1.0  # each parameter moves by 1 per 100 steps
        out[step] = delta(step, [w])
    assert set(out[400]) == {"weight_delta_100"}  # 500-step lag not reached yet
    assert out[500] == {"weight_delta_100": 4.0, "weight_delta_500": 20.0}  # vs. step 0
    assert out[800] == {"weight_delta_100": 4.0, "weight_delta_500": 20.0}  # vs. step 300


def test_lags_must_be_multiples_of_every():
    with pytest.raises(ValueError):
        WeightDelta(every=100, lags=[100, 250])


def test_torch_params_after_sgd_step():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    delta = WeightDelta(every=1, lags=[1])
    before = [p.detach().clone() for p in model.parameters()]
    delta(0, model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.randn(8, 4)).pow(2).sum().backward()
    opt.step()
    expected = sum(
        (p - q).abs().sum().item() for p, q in zip(model.parameters(), before, strict=True)
    )
    assert delta(1, model.parameters())["weight_delta_1"] == pytest.approx(expected)
    assert expected > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_torch_on_gpu():
    p = torch.ones(10, device="cuda")
    delta = WeightDelta(every=10, lags=[10])
    delta(0, [p])
    p.data -= 0.25
    assert delta(10, [p])["weight_delta_10"] == pytest.approx(2.5)


def test_parameter_list_must_not_change():
    delta = WeightDelta(every=1, lags=[1])
    delta(0, [np.zeros(2)])
    with pytest.raises(ValueError):
        delta(1, [np.zeros(2), np.zeros(1)])
