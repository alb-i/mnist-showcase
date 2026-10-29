"""IDX reader, checksums, splits and the curation overlay."""

import gzip
import hashlib
import os
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parents[2]

# File names and MD5 checksums as published by torchvision (torchvision.datasets.MNIST.resources)
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
SPLIT_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
ID_WIDTH = {"train": 5, "test": 4}

# IDX type codes (third byte of the magic number) -> numpy dtype, big-endian
IDX_DTYPES = {
    0x08: np.dtype(">u1"),
    0x09: np.dtype(">i1"),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}


def data_dir() -> Path:
    """$DATA_DIR from the environment or the repo's .env file."""
    load_dotenv(REPO_DIR / ".env")
    return Path(os.environ["DATA_DIR"])


def raw_dir() -> Path:
    return data_dir() / "mnist" / "raw"


def file_hash(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(raw: Path) -> None:
    """Check the MD5s published by torchvision and our own SHA256SUMS file."""
    for name, md5 in MNIST_FILES.items():
        if file_hash(raw / name, "md5") != md5:
            raise ValueError(f"MD5 mismatch for {name}")
    for line in (raw / "SHA256SUMS").read_text().splitlines():
        sha256, name = line.split()
        if file_hash(raw / name, "sha256") != sha256:
            raise ValueError(f"SHA256 mismatch for {name}")


def read_idx(path: Path) -> np.ndarray:
    """Read an IDX file (optionally gzipped).

    Format: magic number (2 zero bytes, type code, number of dimensions), then one big-endian
    uint32 per dimension, then the data in C order.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        buf = f.read()
    if buf[0] != 0 or buf[1] != 0:
        raise ValueError(f"{path}: bad magic number")
    dtype = IDX_DTYPES[buf[2]]
    ndim = buf[3]
    shape = tuple(int(d) for d in np.frombuffer(buf, dtype=">u4", count=ndim, offset=4))
    offset = 4 + 4 * ndim
    if len(buf) - offset != dtype.itemsize * int(np.prod(shape)):
        raise ValueError(f"{path}: size does not match header {shape}")
    data = np.frombuffer(buf, dtype=dtype, offset=offset).reshape(shape)
    return data.astype(dtype.newbyteorder("="))


def load_split(split: str, raw: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Raw images (N, 28, 28) uint8 and labels (N,) uint8 of the original 'train' or 'test' set."""
    raw = raw or raw_dir()
    images_file, labels_file = SPLIT_FILES[split]
    images, labels = read_idx(raw / images_file), read_idx(raw / labels_file)
    if len(images) != len(labels):
        raise ValueError(f"{split}: {len(images)} images but {len(labels)} labels")
    return images, labels


def sample_ids(split: str, n: int) -> np.ndarray:
    """Stable IDs: train-00000 ... train-59999, test-0000 ... test-9999."""
    width = ID_WIDTH[split]
    return np.array([f"{split}-{i:0{width}d}" for i in range(n)])


def split_config() -> dict:
    with open(REPO_DIR / "configs" / "split.yaml") as f:
        return yaml.safe_load(f)


def make_val_split(labels: np.ndarray, n_val: int, seed: int) -> np.ndarray:
    """Stratified validation split: sorted indices into the original training set."""
    from sklearn.model_selection import train_test_split

    _, val_idx = train_test_split(
        np.arange(len(labels)), test_size=n_val, stratify=labels, random_state=seed
    )
    return np.sort(val_idx)


def make_cv_folds(labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Stratified fold number (0 .. n_folds-1) per sample."""
    from sklearn.model_selection import StratifiedKFold

    folds = np.empty(len(labels), dtype=np.int64)
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        folds[idx] = fold
    return folds


def cv_folds() -> dict[str, int]:
    """Sample id -> fold of the committed cross-validation folds (configs/split.yaml)."""
    path = REPO_DIR / split_config()["cv_folds_file"]
    lines = path.read_text().splitlines()
    return {sample_id: int(fold) for sample_id, fold in map(str.split, lines)}


def dataset_version(decisions: Path | None = None) -> str:
    """Version of the curated dataset: the first 8 hex digits of the sha256 of the lines
    "sample_id,action,new_label" of curation/decisions.csv (in file order), so that only what
    changes the data counts, not the reason, reviewer, timestamp or round of a decision."""
    import csv

    with open(decisions or REPO_DIR / "curation" / "decisions.csv", newline="") as f:
        rows = [f"{r['sample_id']},{r['action']},{r.get('new_label') or ''}\n"
                for r in csv.DictReader(f)]  # fmt: skip
    return hashlib.sha256("".join(rows).encode()).hexdigest()[:8]


def val_ids() -> list[str]:
    """IDs of the fixed validation split (committed file, see configs/split.yaml)."""
    path = REPO_DIR / split_config()["val_ids_file"]
    return path.read_text().split()


def read_decisions(path: Path | None = None) -> dict[str, dict]:
    """Rows of curation/decisions.csv by sample_id; a later row for the same ID wins."""
    import csv

    with open(path or REPO_DIR / "curation" / "decisions.csv", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def apply_overlay(
    ids: np.ndarray, labels: np.ndarray, decisions: dict[str, dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Keep mask and curated labels: `drop` removes a sample, `relabel` sets new_label.

    keep and unsure only record that a sample was reviewed (curation/README.md).
    """
    keep = np.ones(len(ids), dtype=bool)
    labels = labels.copy()
    index = {sample_id: i for i, sample_id in enumerate(ids)}
    for sample_id, row in decisions.items():
        i = index.get(sample_id)
        if i is None:
            continue
        if row["action"] == "drop":
            keep[i] = False
        elif row["action"] == "relabel":
            labels[i] = int(row["new_label"])
    return keep, labels


def load_dataset(
    raw: Path | None = None, decisions: Path | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Curated splits {"train", "val", "test"}: (ids, images (N, 28, 28) uint8, labels (N,))."""
    overlay = read_decisions(decisions)
    out = {}
    for split in ("train", "test"):
        images, labels = load_split(split, raw)
        ids = sample_ids(split, len(labels))
        keep, labels = apply_overlay(ids, labels, overlay)
        if split == "train":
            is_val = np.isin(ids, val_ids())
            out["val"] = (ids[keep & is_val], images[keep & is_val], labels[keep & is_val])
            keep &= ~is_val
        out[split] = (ids[keep], images[keep], labels[keep])
    return out


def load_cv_dataset(
    fold: int, raw: Path | None = None, decisions: Path | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Cross-validation splits (D18): "train" = all curated training images (incl. the fixed val
    split) outside `fold`, "val" = the images of `fold` (out-of-fold predictions), "test" = test."""
    ds = load_dataset(raw, decisions)
    pairs = zip(ds["train"], ds["val"], strict=True)
    ids, images, labels = (np.concatenate(pair) for pair in pairs)
    folds = cv_folds()
    in_fold = np.array([folds[i] == fold for i in ids])
    return {
        "train": (ids[~in_fold], images[~in_fold], labels[~in_fold]),
        "val": (ids[in_fold], images[in_fold], labels[in_fold]),
        "test": ds["test"],
    }
