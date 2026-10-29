import numpy as np


def synthetic_dataset(n=300, seed=0):
    """Ten separable 'digits': class k lights up row k of the image."""
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("train", "val", "test"):
        labels = rng.integers(0, 10, n).astype(np.uint8)
        images = rng.integers(0, 40, (n, 28, 28)).astype(np.uint8)
        images[np.arange(n), labels] = 255
        out[split] = (np.array([f"{split}-{i:05d}" for i in range(n)]), images, labels)
    return out


def run_dir_for(root, ref):
    """Creates the run folders that digits.summarize.find_run resolves `ref` (name or name#k) to,
    with placeholder timestamps, and returns the folder for ref."""
    name, _, k = ref.partition("#")
    dirs = [root / f"20261001-{i:06d}-{name}" for i in range(int(k or 1))]
    for d in dirs[:-1]:
        d.mkdir(parents=True, exist_ok=True)
    return dirs[-1]
