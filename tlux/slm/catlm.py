#!/usr/bin/env python3
"""
catlm.py – Summarize a directory, show a tree view, and optionally emit the
           first 50 text files as fenced Markdown blocks.

This script provides a detailed summary of a directory's structure and contents
using only the Python standard library. It is compatible with CPython 3.8+.

Usage:
  python catlm.py [directory] [--exclude PATTERN]... [--clear-cache] [--structure]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

CACHE_FILE = ".catlm_exclusions"
MAX_FILES = 50  # Maximum number of files to process for content dumping

# --------------------------------------------------------------------------- #
# Helper Functions                                                          #
# --------------------------------------------------------------------------- #

# Load and manage exclusion patterns from a cache file.
#
# This function manages a persistent list of exclusion patterns:
# - Clears the cache file if requested.
# - Appends new patterns from the command line to the cache.
# - Reads existing patterns, removes duplicates (preserving order), and updates the file.
# - Returns the current list of patterns.
#
# Parameters:
# - clear (bool): If True, delete the cache file before processing.
# - new_patterns (List[str]): New glob patterns to add to the cache.
#
# Returns:
# - List[str]: A list of unique exclusion patterns.
def load_exclusions(clear: bool, new_patterns: List[str]) -> List[str]:
    if clear and Path(CACHE_FILE).exists():
        Path(CACHE_FILE).unlink()
        print("Cleared exclusion cache")

    if new_patterns:
        with open(CACHE_FILE, "a", encoding="utf-8") as fh:
            fh.write("\n".join(new_patterns) + "\n")

    patterns: List[str] = []
    if Path(CACHE_FILE).is_file():
        with open(CACHE_FILE, encoding="utf-8") as fh:
            patterns = [ln.strip() for ln in fh if ln.strip()]

    # Remove duplicates while preserving order
    seen = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]
    if patterns:
        print("Active exclusion patterns:", " ".join(patterns), end="\n\n")
        with open(CACHE_FILE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(patterns) + "\n")

    return patterns

# Determine if a path should be excluded based on its name or relative path.
#
# A path is excluded if its name starts with a dot (hidden) or its relative path
# matches any exclusion pattern (e.g., "build/*" excludes all files under "build").
#
# Parameters:
# - path (Path): The file or directory path to check.
# - root (Path): The root directory for computing the relative path.
# - exclusions (List[str]): List of glob-style exclusion patterns.
#
# Returns:
# - bool: True if the path should be excluded, False otherwise.
def is_excluded(path: Path, root: Path, exclusions: List[str]) -> bool:
    rel_path = path.relative_to(root).as_posix()
    return (
        path.name.startswith(".")
        or any(fnmatch.fnmatch(rel_path, pat) for pat in exclusions)
    )

# Walk the directory tree, filtering out excluded and hidden items.
#
# This generator yields tuples of (current directory, files, subdirectories) for
# each directory in the tree, similar to os.walk, but applies filters:
# - Excludes hidden items (starting with ".") and items matching exclusion patterns.
# - Sorts subdirectories alphabetically for consistent traversal order.
#
# Parameters:
# - root (Path): The starting directory for the walk.
# - exclusions (List[str]): List of glob patterns to exclude.
#
# Yields:
# - Tuple[Path, List[Path], List[Path]]: Current directory, list of file paths,
#   and list of subdirectory paths.
def filtered_walk(
    root: Path, exclusions: List[str]
) -> Iterable[Tuple[Path, List[Path], List[Path]]]:
    for cur_dir, dirnames, filenames in os.walk(root):
        cur_path = Path(cur_dir)

        # Filter and sort subdirectories in-place to prune the walk
        dirnames[:] = sorted(
            [d for d in dirnames if not is_excluded(cur_path / d, root, exclusions)]
        )

        # Filter files based on exclusions and hidden status
        files = [f for f in filenames if not is_excluded(cur_path / f, root, exclusions)]
        yield cur_path, [cur_path / f for f in files], [cur_path / d for d in dirnames]

# Define a sorting key for directory entries.
#
# Sorts entries to prioritize files over directories, then by extension and name
# (both case-insensitive), ensuring a consistent and readable order.
#
# Parameters:
# - entry (Path): The file or directory path to sort.
#
# Returns:
# - tuple: A tuple (is_dir, ext, name) for sorting.
def sort_key(entry: Path) -> tuple:
    is_dir = entry.is_dir()
    ext = entry.suffix.lower()
    name = entry.stem.lower()
    return (is_dir, ext, name)  # False < True, so files come before dirs

# Display the directory structure as a tree.
#
# Prints a hierarchical view of the directory:
# - Files are listed before subdirectories at each level.
# - Entries are sorted by extension and name (case-insensitive).
# - Uses indentation and symbols to represent the tree structure.
#
# Parameters:
# - root (Path): The root directory to display.
# - exclusions (List[str]): List of glob patterns to exclude.
def print_tree(root: Path, exclusions: List[str]) -> None:
    def walk(dir_path: Path, depth: int) -> None:
        children = [c for c in dir_path.iterdir() if not is_excluded(c, root, exclusions)]
        children.sort(key=sort_key)
        for child in children:
            prefix = "| " * depth + "|___ "
            print(f"{prefix}{child.name}" + ("/" if child.is_dir() else ""))
            if child.is_dir():
                walk(child, depth + 1)

    print("Directory structure")
    print("=" * 31)
    print()
    print(root.name + "/")
    walk(root, 0)
    print()

# Check if a file is likely a text file.
#
# Reads the first 1024 bytes of the file:
# - If it contains a null byte (\0), assumes binary.
# - If empty or no null bytes, assumes text.
# - Returns False on read errors (e.g., permissions).
#
# Parameters:
# - path (Path): The file to check.
#
# Returns:
# - bool: True if likely text, False if likely binary or unreadable.
def is_text(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except (OSError, PermissionError):
        return False
    if not data:
        return True
    return b"\0" not in data[:1024]

# Output the contents of the first 50 text files in Markdown format.
#
# Processes files from the directory tree:
# - Sorts files within each directory by extension and name.
# - Prints each text file's contents in a fenced code block, up to MAX_FILES.
# - Skips binary files and handles read errors gracefully.
#
# Parameters:
# - root (Path): The root directory to process.
# - exclusions (List[str]): List of glob patterns to exclude.
def dump_files(root: Path, exclusions: List[str]) -> None:
    print("\nFile contents")
    print("==============")

    count = 0
    for cur_dir, files, _ in filtered_walk(root, exclusions):
        for f in sorted(files, key=sort_key):
            if count >= MAX_FILES:
                break
            full_path = f  # Already a Path from filtered_walk
            rel_path = full_path.relative_to(root)
            print(f"\n--- Start of file: {rel_path} ---")
            print("```")
            if is_text(full_path):
                try:
                    text = full_path.read_text(errors="replace")
                    print(text.replace("```", "``"))  # Avoid breaking fences
                except Exception as exc:
                    print(f"[Error reading file: {exc}]")
            else:
                print("[Binary file – content not displayed]")
            print("```")
            print(f"--- End of file: {rel_path} ---")
            count += 1
        if count >= MAX_FILES:
            break

    print(f"\nProcessed {count} file(s)")

# --------------------------------------------------------------------------- #
#   Main Function                                                             #
# --------------------------------------------------------------------------- #

# Entry point for the script.
#
# Parses command-line arguments, sets up the directory processing, and
# coordinates the tree display and file content dumping.
#
# Command-line options:
# - directory: Directory to process (default: current directory).
# - --exclude: Glob patterns to exclude files/directories.
# - --clear-cache: Remove the exclusion cache file.
# - --structure: Show only the tree, not file contents.
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="catlm.py",
        description="Summarize a directory tree and show up to 50 text files "
                    "as fenced Markdown blocks."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="directory to process (default: current directory)",
    )
    parser.add_argument(
        "--exclude",
        metavar="PATTERN",
        action="append",
        default=[],
        help="glob pattern to exclude (can be given multiple times)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="wipe the exclusion cache before running",
    )
    parser.add_argument(
        "--structure",
        action="store_true",
        help="only show the tree structure and not contents",
    )

    args = parser.parse_args()

    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"'{root}' is not a directory")

    patterns = load_exclusions(args.clear_cache, args.exclude or [])
    print_tree(root, patterns)
    if not args.structure:
        dump_files(root, patterns)

if __name__ == "__main__":
    sys.exit(main())
