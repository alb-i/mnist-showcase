import gzip
import os
import stat
from collections import Counter

import numpy as np
import pytest

from digits import data


def write_idx(path, array, type_code=0x08):
    header = bytes([0, 0, type_code, array.ndim])
    header += np.array(array.shape, dtype=">u4").tobytes()
    path.write_bytes(header + array.astype(data.IDX_DTYPES[type_code]).tobytes())


def test_read_idx_synthetic(tmp_path):
    a = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    write_idx(tmp_path / "a.idx", a)
    np.testing.assert_array_equal(data.read_idx(tmp_path / "a.idx"), a)


def test_read_idx_int32_gzip(tmp_path):
    a = np.array([[-1, 70000], [3, 4]], dtype=np.int32)
    write_idx(tmp_path / "a.idx", a, type_code=0x0C)
    (tmp_path / "a.idx.gz").write_bytes(gzip.compress((tmp_path / "a.idx").read_bytes()))
    np.testing.assert_array_equal(data.read_idx(tmp_path / "a.idx.gz"), a)


def test_read_idx_rejects_truncated(tmp_path):
    write_idx(tmp_path / "a.idx", np.zeros((2, 2), dtype=np.uint8))
    (tmp_path / "a.idx").write_bytes((tmp_path / "a.idx").read_bytes()[:-1])
    with pytest.raises(ValueError):
        data.read_idx(tmp_path / "a.idx")


def test_sample_ids():
    assert list(data.sample_ids("train", 60000)[[0, -1]]) == ["train-00000", "train-59999"]
    assert list(data.sample_ids("test", 10000)[[0, -1]]) == ["test-0000", "test-9999"]


def test_md5s_match_torchvision():
    from torchvision.datasets import MNIST

    assert dict(MNIST.resources) == data.MNIST_FILES


# --- tests on the downloaded raw data -------------------------------------------------------

raw_missing = not (data.raw_dir() / "SHA256SUMS").exists()
needs_raw = pytest.mark.skipif(raw_missing, reason="run scripts/download_mnist.py first")


@needs_raw
def test_checksums():
    data.verify_checksums(data.raw_dir())


@needs_raw
def test_raw_is_read_only():
    raw = data.raw_dir()
    for path in [raw, *raw.iterdir()]:
        assert not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), path
    assert not os.access(raw, os.W_OK)


@needs_raw
@pytest.mark.parametrize("split", ["train", "test"])
def test_reader_matches_torchvision(split, tmp_path):
    from torchvision.datasets.mnist import read_image_file, read_label_file

    images, labels = data.load_split(split)
    images_file, labels_file = data.SPLIT_FILES[split]
    for name in (images_file, labels_file):
        (tmp_path / name[:-3]).write_bytes(gzip.decompress((data.raw_dir() / name).read_bytes()))
    np.testing.assert_array_equal(images, read_image_file(tmp_path / images_file[:-3]).numpy())
    np.testing.assert_array_equal(labels, read_label_file(tmp_path / labels_file[:-3]).numpy())


@needs_raw
def test_shapes_and_statistics():
    train_x, train_y = data.load_split("train")
    test_x, test_y = data.load_split("test")
    assert train_x.shape == (60000, 28, 28) and test_x.shape == (10000, 28, 28)
    assert train_x.dtype == np.uint8 and set(np.unique(train_y)) == set(range(10))
    x = train_x / 255.0
    assert x.mean() == pytest.approx(0.1307, abs=1e-4)
    assert x.std() == pytest.approx(0.3081, abs=1e-4)


@needs_raw
def test_val_split_file():
    cfg = data.split_config()
    _, labels = data.load_split("train")
    ids = data.val_ids()
    assert len(ids) == len(set(ids)) == cfg["n_val"]
    # the committed file is exactly what the configured seed produces
    expected = data.sample_ids("train", len(labels))[
        data.make_val_split(labels, cfg["n_val"], cfg["seed"])
    ]
    assert ids == list(expected)
    # stratified: per-class share in val equals the share in train within one sample
    idx = np.array([int(i.split("-")[1]) for i in ids])
    val_counts, all_counts = Counter(labels[idx]), Counter(labels)
    for c in range(10):
        assert abs(val_counts[c] - all_counts[c] * cfg["n_val"] / len(labels)) <= 1


def empty_decisions(tmp_path):
    """Split sizes without curation, whatever curation/decisions.csv holds."""
    path = tmp_path / "decisions.csv"
    path.write_text("sample_id,action,new_label,reason,reviewer,timestamp,round\n")
    return path


@needs_raw
def test_load_dataset_splits(tmp_path):
    ds = data.load_dataset(decisions=empty_decisions(tmp_path))
    cfg = data.split_config()
    assert [len(ds[s][0]) for s in ("train", "val", "test")] == [
        cfg["n_train"],
        cfg["n_val"],
        10000,
    ]
    assert set(ds["val"][0]) == set(data.val_ids())
    assert not set(ds["train"][0]) & set(ds["val"][0])
    assert ds["train"][1].shape[1:] == (28, 28)


@needs_raw
def test_cv_folds_file():
    cfg = data.split_config()
    _, labels = data.load_split("train")
    folds = data.cv_folds()
    ids = data.sample_ids("train", len(labels))
    # the committed file is exactly what the configured seed produces
    expected = data.make_cv_folds(labels, cfg["cv_folds"], cfg["cv_seed"])
    assert [folds[i] for i in ids] == expected.tolist()
    sizes = Counter(folds.values())
    assert sorted(sizes) == list(range(cfg["cv_folds"])) and set(sizes.values()) == {12000}


def test_make_cv_folds_is_stratified():
    labels = np.repeat(np.arange(10), 50)
    folds = data.make_cv_folds(labels, 5, seed=0)
    for fold in range(5):
        assert Counter(labels[folds == fold].tolist()) == dict.fromkeys(range(10), 10)


@needs_raw
def test_load_cv_dataset(tmp_path):
    ds = data.load_cv_dataset(1, decisions=empty_decisions(tmp_path))
    assert [len(ds[s][0]) for s in ("train", "val", "test")] == [48000, 12000, 10000]
    folds = data.cv_folds()
    assert {folds[i] for i in ds["val"][0]} == {1} and 1 not in {folds[i] for i in ds["train"][0]}
    assert set(ds["val"][0]) & set(data.val_ids())  # the fixed val split takes part in the folds
