"""File system abstraction used by the HKM search toolkit."""

import os
from dataclasses import dataclass
from typing import Iterable


@dataclass
class FileSystem:
    """Simple local file-system implementation.

    This is intentionally minimal and suitable for unit tests.  A real
    implementation could provide remote object store access while
    preserving the same interface.
    """

    root: str = ""

    # ------------------------------------------------------------------
    # path helpers
    def join(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    # ------------------------------------------------------------------
    # filesystem helpers
    def mkdir(self, path: str, exist_ok: bool = False) -> None:
        os.makedirs(path, exist_ok=exist_ok)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def listdir(self, path: str) -> Iterable[str]:
        return os.listdir(path)

    def open(self, path: str, mode: str = "r"):
        return open(path, mode)
