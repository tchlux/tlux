# File system abstraction used by the HKM search toolkit.
# 
# This module provides a minimal, test-friendly file system interface.
# Designed for local use, but easily extensible to remote backends
# with the same API. All operations are deterministic and side-effect-free
# unless explicitly invoked by the user.
# 
# Example usage:
# 
#     fs = FileSystem(root="/tmp")
#     path = fs.join("example.txt")
#     fs.write(path, b"hello")
#     assert fs.read(path) == b"hello"
#     fs.rename(path, fs.join("renamed.txt"))


import os
import shutil
from dataclasses import dataclass
from typing import Iterable


# Simple file system interface rooted at a specified base path.
# All methods operate relative to `root`, except for absolute paths.
#
# Parameters:
#   root (str): Optional base directory prepended to all paths.
#
# Example:
#   fs = FileSystem("/tmp")
#   fs.mkdir(fs.join("subdir"))
# 
@dataclass(frozen=True)
class FileSystem:
    root: str = ""

    # Join one or more path components with the root.
    #
    # Parameters:
    #   parts (str): Path components to join.
    #
    # Returns:
    #   str: Joined path relative to root.
    # 
    def join(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    # Create a directory at the specified path.
    #
    # Parameters:
    #   path (str): Directory path.
    #   exist_ok (bool): Allow existing directory without error.
    #
    # Raises:
    #   OSError: If creation fails.
    # 
    def mkdir(self, path: str, exist_ok: bool = False) -> None:
        os.makedirs(path, exist_ok=exist_ok)

    # Check whether the path exists.
    #
    # Parameters:
    #   path (str): Path to check.
    #
    # Returns:
    #   bool: True if path exists.
    # 
    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    # List entries in a directory.
    #
    # Parameters:
    #   path (str): Directory path.
    #
    # Returns:
    #   Iterable[str]: List of entry names.
    # 
    def listdir(self, path: str) -> Iterable[str]:
        return os.listdir(path)

    # Read the full contents of a file as bytes.
    #
    # Parameters:
    #   path (str): File path.
    #
    # Returns:
    #   bytes: File contents.
    #
    # Raises:
    #   IOError: If read fails.
    # 
    def read(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    # Write bytes to a file.
    #
    # Parameters:
    #   path (str): File path.
    #   data (bytes): Data to write.
    #   overwrite (bool): If False, raises on existing files.
    #
    # Raises:
    #   RuntimeError: If refusing to overwrite existing file.
    # 
    def write(self, path: str, data: bytes, overwrite: bool = True, mkdir: bool = True) -> None:
        if not overwrite and os.path.exists(path):
            raise RuntimeError(
                f"Refusing to overwrite existing contents at '{path}'."
            )
        if mkdir: os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    # Rename or move a file or directory.
    #
    # Parameters:
    #   source (str): Existing file or directory path.
    #   destination (str): Target path after rename.
    #
    # Returns:
    #   bool: True on success, False on failure.
    # 
    def rename(self, source: str, destination: str) -> bool:
        try:
            shutil.move(source, destination)
            return True
        except (OSError, shutil.Error):
            return False


if __name__ == "__main__":
    # Sanity check: write-read-rename cycle
    _fs = FileSystem("/tmp")
    _path = _fs.join("demo.txt")
    _fs.write(_path, b"check")
    assert _fs.read(_path) == b"check"
    _new_path = _fs.join("demo_renamed.txt")
    assert _fs.rename(_path, _new_path)
    assert _fs.exists(_new_path)
