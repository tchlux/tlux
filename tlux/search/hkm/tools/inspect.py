"""Inspect one published HKM index and its build evidence."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from ..search.searcher import Searcher, audit_index
from .benchmark import _build_evidence, _iter_windows, _storage_components, _summary, _tree_bytes


# Time warm public searches and return stable percentile evidence.
#
# Arguments:
#   searcher (Searcher): Open index to query.
#   queries (list[str]): Text queries to measure.
#   repeats (int): Warm repetitions per query.
#
# Returns:
#   (list[dict[str, Any]]): Query timings and result counts.
#
def _search_evidence(searcher: Searcher, queries: list[str], repeats: int) -> list[dict[str, Any]]:
    rows = []
    for text in queries:
        searcher.search({"text": text, "mode": "hybrid", "top_k": 10})
        timings = []
        count = 0
        for _ in range(repeats):
            start = time.perf_counter()
            result = searcher.search({"text": text, "mode": "hybrid", "top_k": 10})
            timings.append((time.perf_counter() - start) * 1000.0)
            count = result.count
        rows.append({"query": text, "results": count, "timing_ms": _summary(timings)})
    return rows


# Inspect a validated index without mutating it.
#
# Arguments:
#   index_root (str): Index directory.
#   queries (list[str]): Optional public queries to time.
#   repeats (int): Warm repetitions per query.
#
# Returns:
#   (dict[str, Any]): Audit, build, storage, resource, and search evidence.
#
def inspect_index(index_root: str, queries: list[str] | None = None, repeats: int = 3) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    root = audit_index(index_root)
    searcher = Searcher.from_index_root(str(root))
    generation_root = Path(searcher.generation_root or root)
    jobs_root = root / ".hkm_jobs"
    return {
        "index_root": str(root),
        "build_id": searcher.build_id,
        "backend": searcher.backend_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "documents": searcher._doc_count(),
        "embedding_windows": sum(1 for _ in _iter_windows(searcher)),
        "generation_root": str(generation_root),
        "canonical_index_bytes": _tree_bytes(generation_root, {".hkm_jobs", ".hkm_cache"}),
        "embedding_cache_bytes": _tree_bytes(root / ".hkm_cache", set()),
        "storage_components": _storage_components(generation_root),
        "build": _build_evidence(root),
        "search": _search_evidence(searcher, queries or [], repeats),
        "audit": "PASS",
        "jobs_root": str(jobs_root),
    }


# Parse CLI arguments, print JSON, and optionally persist the report.
#
# Arguments:
#   None.
#
# Returns:
#   (None): Prints one JSON report.
#
def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HKM build and search evidence.")
    parser.add_argument("index_root")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    report = inspect_index(args.index_root, args.query, args.repeat)
    payload = json.dumps(report, indent=2)
    if args.json_output:
        Path(args.json_output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
