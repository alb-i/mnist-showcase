"""Array backend: NumPy on the CPU, CuPy on the GPU (plan 7.2: "same code, xp = cupy").

    xp = get_xp("gpu")              # cupy; get_xp("cpu") is numpy
    x = xp.asarray(x_numpy)         # upload
    y = to_numpy(x)                 # download (no-op for NumPy arrays)

The layers, loss and optimisers never import cupy themselves: they take the module of their
arrays with array_module(), so the same code runs on both devices.
"""

import numpy as np

try:
    import cupy
except ImportError:  # CPU-only environment
    cupy = None

DEVICES = ("cpu", "gpu")


def get_xp(device: str):
    if device == "cpu":
        return np
    if device == "gpu":
        if cupy is None:
            raise RuntimeError("device gpu needs CuPy (micromamba install -c conda-forge cupy)")
        return cupy
    raise ValueError(f"unknown device {device!r}, expected one of {DEVICES}")


def gpu_available() -> bool:
    try:
        return cupy is not None and cupy.cuda.runtime.getDeviceCount() > 0
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


def array_module(x):
    """numpy or cupy, whichever x belongs to."""
    return np if cupy is None else cupy.get_array_module(x)


def to_numpy(x) -> np.ndarray:
    return x.get() if cupy is not None and isinstance(x, cupy.ndarray) else np.asarray(x)


def gpu_memory_gb() -> float:
    """Bytes held by CuPy's memory pool, in GB (logged as gpu_mem_gb like the PyTorch runs)."""
    return cupy.get_default_memory_pool().total_bytes() / 1e9
