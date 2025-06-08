"""Memory-mapped NumPy array helpers."""

import mmap
import numpy as np
from typing import Tuple


def mmap_array(path: str, dtype=np.float32, shape: Tuple[int, ...] = None):
    """Memory-map a NumPy array from ``path``.

    Parameters
    ----------
    path:
        File path to the array on disk.
    dtype:
        NumPy dtype of the array.
    shape:
        Expected shape of the array.  If ``None``, the array is 1-D.
    """
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)
