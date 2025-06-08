"""Safe on-disk writers."""

import os
from contextlib import contextmanager


@contextmanager
def atomic_write(path: str, mode: str = "wb"):
    """Write to ``path`` atomically using a temporary file."""
    tmp = f"{path}.tmp"
    with open(tmp, mode) as f:
        yield f
    os.replace(tmp, path)
