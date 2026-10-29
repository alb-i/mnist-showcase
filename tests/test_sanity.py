import pytest
import yaml
from conftest import synthetic_dataset
from test_train import small_config

from digits import sanity


def test_overfit_passes_on_a_learnable_batch(tmp_path):
    config = small_config("cnn_reg")  # with augmentation and label smoothing: both switched off
    passed, loss, run_dir = sanity.overfit(config, synthetic_dataset(), steps=60, root=tmp_path)
    assert passed and loss < sanity.PASS_LOSS
    assert run_dir.name.endswith("-sanity-overfit")
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved["augment"] is None and saved["label_smoothing"] == 0.0
    assert saved["scheduler"] is None and saved["epochs"] == 60 and saved["batch_size"] == 128
    assert not (run_dir / "samples").exists()


def test_cli_fails_without_learning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sanity, "load_dataset", synthetic_dataset)
    argv = ["overfit", "configs/mlp.yaml", "--steps", "20", "--lr", "0", "--set", "device=cpu"]
    with pytest.raises(SystemExit) as exit_info:
        sanity.main([*argv, "--root", str(tmp_path)])
    assert exit_info.value.code == 1
    assert "\nFAIL: training loss" in capsys.readouterr().out
