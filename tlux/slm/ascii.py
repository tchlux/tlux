#!/usr/bin/env python3
"""
ascii.py - Recursively strip or report non-ASCII characters.

This script scans a directory tree (default: current directory) and either
replaces non-ASCII characters in-place or reports them, using only the Python
standard library.  Compatible with CPython 3.8+.

Usage::

  python ascii.py [directory]               # rewrite files in-place
  python ascii.py [directory] --report      # list offending chars only

Options:
  --replace OLD=NEW   Add or update a replacement mapping (can repeat)
  --clear-config      Remove the replacement config file before running
  --report            Show a report instead of rewriting files
  --dry-run           Show what would change but don't write files

Replacement mappings are persisted in *~/.ascii_replacements* (one
``OLD=NEW`` pair per line).  Defaults include smart quotes, en/em dashes, the
ellipsis character, and non-breaking space.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CONFIG_FILE = Path.home() / ".ascii_replacements"
EXCLUDE_CACHE = Path.home() / ".ascii_exclusions"
DEFAULT_REPLACEMENTS = {
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201C": '"',    # left double quote
    "\u201D": '"',    # right double quote
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2212": "-",    # dash
    "\u2022": "-",    # bullet
    "\u2026": "...",  # ellipsis
    "\u00A0": " ",    # non-breaking space
    "\u279e": "->",   # right arrow
    "\u2192": "->",   # right arrow
    "\u2190": "<-",   # left arrow
    "\u00B2": "^2",   # squared
    "\uFF5E": "~",    # wide tilde
    "\u223C": "~",    # tilde
    "\u223D": "~",    # tilde
    "\u226A": "<<",   # double left arrow
    "\u2265": ">=",   # greater equals
    "\uA78C": "'",    # apostrophe
    "\uFF07": "'",    # apostrophe
    "\uFF0B": "+",    # big plus
    "\uFF0D": "-",    # wide dash
    "\u2248": "~",    # approx equals
    "\u2215": "/",    # slash
    "\u00B2": "^2",   # superscript two
    "\u00B4": "'",    # acute accent
    "\u00B7": "*",    # middle dot
    "\u00B9": "^1",   # superscript one
    "\u00D7": "x",    # multiplication sign
    "\u02DC": "~",    # small tilde
    "\u055A": "'",    # Armenian apostrophe
    "\u1D40": "^T",   # superscript capital T
    "\u2010": "-",    # hyphen
    "\u2011": "-",    # non-breaking hyphen
    "\u2012": "-",    # figure dash
    "\u2015": "-",    # horizontal bar
    "\u201A": "'",    # single low-9 quote
    "\u201B": "'",    # single high-reversed-9 quote
    "\u201E": "\"",   # double low-9 quote
    "\u201F": "\"",   # double high-reversed-9 quote
    "\u2022": "*",    # bullet
    "\u2032": "'",    # prime
    "\u2033": "''",   # double prime
    "\u2034": "'''",  # triple prime
    "\u2035": "'",    # reversed prime
    "\u2036": "''",   # reversed double prime
    "\u2037": "'''",  # reversed triple prime
    "\u2043": "-",    # hyphen bullet
    "\u2044": "/",    # fraction slash
    "\u2053": "~",    # swung dash
    "\u2057": "''''", # quadruple prime
    "\u2079": "^9",   # superscript nine
    "\u207A": "+",    # superscript plus
    "\u207B": "-",    # superscript minus
    "\u2192": "->",   # right arrow
    "\u2212": "-",    # minus sign
    "\u2215": "/",    # division slash
    "\u223C": "~",    # tilde operator
    "\u223D": "~",    # reversed tilde
    "\u223F": "~",    # sine wave
    "\u2248": "~=",   # almost equal
    "\u2264": "<=",   # less-than-or-equal
    "\u2265": ">=",   # greater-than-or-equal
    "\u226A": "<<",   # much less than
    "\u301C": "~",    # wave dash
    "\uA78B": "'",    # Latin capital saltillo
    "\uA78C": "'",    # Latin small  saltillo
    "\uFF07": "'",    # full-width apostrophe
    "\uFF0B": "+",    # full-width plus
    "\uFF0D": "-",    # full-width hyphen
    "\uFF5E": "~",    # full-width tilde
}
SNIPPET_RADIUS = 30  # characters of context on each side when reporting

# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #

def load_replacements(clear: bool, updates: List[str]) -> Dict[str, str]:
    """Load replacement dictionary, optionally clearing or updating it."""
    if clear and CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        print("Cleared replacement config\n")

    # Append new pairs to config file for persistence
    if updates:
        with CONFIG_FILE.open("a", encoding="utf-8") as fh:
            for pair in updates:
                if "=" not in pair:
                    print(f"[warn] invalid --replace argument '{pair}'; expected OLD=NEW")
                    continue
                fh.write(pair + "\n")

    # Start with defaults, then layer persisted overrides
    repl: Dict[str, str] = DEFAULT_REPLACEMENTS.copy()
    if CONFIG_FILE.is_file():
        with CONFIG_FILE.open(encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.rstrip("\n")
                if "=" in ln:
                    old, new = ln.split("=", 1)
                    repl[old] = new

    # Show active mapping summary (limited length)
    preview = "\n  ".join(f"{k!r} -> {v!r}" for k, v in list(repl.items())[:8])
    print(f"Active replacements ({len(repl)}):\n  {preview}")
    return repl


def is_text(path: Path) -> bool:
    """Heuristic: file is text if first 1024 bytes contain no NULs."""
    try:
        chunk = path.read_bytes()[:1024]
    except (OSError, PermissionError):
        return False
    return b"\0" not in chunk


def find_offenders(content: str) -> List[Tuple[str, int]]:
    """Return list of (char, index) tuples for non-ASCII chars in *content*."""
    return [(ch, i) for i, ch in enumerate(content) if ord(ch) > 127]


# --------------------------------------------------------------------------- #
# Exclusion helpers                                                           #
# --------------------------------------------------------------------------- #

def load_exclusions(clear: bool, new_patterns: List[str]) -> List[str]:
    """Persist and retrieve glob exclusion patterns."""
    if clear and EXCLUDE_CACHE.exists():
        EXCLUDE_CACHE.unlink()
        print("Cleared exclusion cache\n")
    if new_patterns:
        with EXCLUDE_CACHE.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(new_patterns) + "\n")
    patterns: List[str] = []
    if EXCLUDE_CACHE.is_file():
        with EXCLUDE_CACHE.open(encoding="utf-8") as fh:
            patterns = [ln.strip() for ln in fh if ln.strip()]
    # de-dupe while preserving order
    seen = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]
    if patterns:
        print("Active exclusion patterns:", " ".join(patterns))
        with EXCLUDE_CACHE.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(patterns) + "\n")
    return patterns


def is_excluded(path: Path, root: Path, patterns: List[str]) -> bool:
    """Return True if *path* (or any parent) should be skipped."""
    rel_path = path.relative_to(root)
    # Skip everything inside hidden directories (e.g., `.env/**`)
    if any(part.startswith(".") for part in rel_path.parts):
        return True
    # Skip if the full relative path matches an exclusion glob
    rel = rel_path.as_posix()
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


# --------------------------------------------------------------------------- #
# Report Mode                                                                 #
# --------------------------------------------------------------------------- #

def report_directory(root: Path, exclusions: List[str], repl: Dict[str, str]) -> None:
    """Report unique non-ASCII chars and where they occur."""
    offenders: Dict[str, List[Tuple[Path, int, str]]] = {}
    for path in root.rglob("*"):
        if is_excluded(path, root, exclusions) or path.is_dir() or not is_text(path):
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        for ch, idx in find_offenders(text):
            snippet_start = max(0, idx - SNIPPET_RADIUS)
            snippet_end = idx + SNIPPET_RADIUS
            snippet = text[snippet_start:snippet_end].replace("\n", " ")
            offenders.setdefault(ch, []).append((path, idx, snippet))

    print()
    if not offenders:
        print("No non-ASCII characters found.")
        return

    print("Non-ASCII character report")
    print("=" * 26)
    for ch, occurrences in sorted(offenders.items(), key=lambda t: t[0]):
        print(f"\n\"\\u{ord(ch):04X}\": \"{ch}\" - {len(occurrences)} occurrence(s)" +
              ("  [NO RULE]" if (ch not in repl) else f"  [replaced by {repr(repl[ch])}]"))
        for path, idx, snippet in occurrences:
            print(f"  {path.relative_to(root)}:{idx}:  {repr(snippet)}")
    print()


# --------------------------------------------------------------------------- #
# Rewrite Mode                                                                #
# --------------------------------------------------------------------------- #


def rewrite_text(text: str, repl: Dict[str, str]) -> Tuple[bool, str]:
    """Perform character replacement on text and return a copy after applying *repl* mapping.."""
    new_text = []
    changed = False
    for ch in text:
        code = ord(ch)
        if code < 128:
            new_text.append(ch)
        else:
            if ch in repl:
                new_text.append(repl[ch])
            else:
                # Unmapped chars are simply dropped
                continue
            changed = True
    return changed, new_text


def rewrite_directory(root: Path, exclusions: List[str], repl: Dict[str, str], dry_run: bool) -> None:
    """Replace non-ASCII chars in-place using *repl* mapping."""
    total_files = total_changed = 0
    for path in root.rglob("*"):
        if is_excluded(path, root, exclusions) or path.is_dir() or not is_text(path):
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            print(f"[skip] {path.relative_to(root)} ({exc})")
            continue

        changed, new_text = rewrite_text(text, repl)
        new_text = []
        changed = False
        for ch in text:
            code = ord(ch)
            if code < 128:
                new_text.append(ch)
            else:
                if ch in repl:
                    new_text.append(repl[ch])
                else:
                    # Unmapped chars are simply dropped
                    continue
                changed = True
        if changed:
            total_changed += 1
            if dry_run:
                print(f"[dry] {path.relative_to(root)} would be updated")
            else:
                path.write_text("".join(new_text))
                print(f"[fix] {path.relative_to(root)} updated")
        total_files += 1

    print(f"\nProcessed {total_files} file(s); updated {total_changed} file(s)")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ascii.py",
        description="Recursively replace or report non-ASCII characters in a "
        "directory tree."
    )
    parser.add_argument("directory", nargs="?", default=Path.cwd(), type=Path,
                        help="directory to process (default: current directory)")
    parser.add_argument("--replace", metavar="OLD=NEW", action="append", default=[],
                        help="add/update a persistent replacement mapping")
    parser.add_argument("--exclude", metavar="PATTERN", action="append", default=[],
                        help="glob pattern to exclude (can repeat)")
    parser.add_argument("--clear-exclude", action="store_true",
                        help="remove all persisted exclusions before running")
    parser.add_argument("--clear-config", action="store_true",
                        help="remove all persisted replacements before running")
    parser.add_argument("--report", action="store_true",
                        help="only list offending characters; do not modify files")
    parser.add_argument("--dry-run", action="store_true",
                        help="show actions without writing files")

    args = parser.parse_args()

    root: Path = args.directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"'{root}' is not a directory")

    exclusions   = load_exclusions(args.clear_exclude, args.exclude or [])
    replacements = load_replacements(args.clear_config, args.replace or [])

    if args.report:
        report_directory(root, exclusions, replacements)
    else:
        rewrite_directory(root, exclusions, replacements, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
