"""Measure HKM build time, resources, storage, and small corpus-size sweeps.

The report labels observed values separately from estimates and is intended for
repeatable local capacity checks, not as evidence of web-scale performance.

Example:
    bin/hkm-capacity /tmp/hkm-capacity --documents 8 --documents 16
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from ..builder.launcher import build_search_index_from_documents
from ..search.searcher import Searcher, audit_index
from .benchmark import _build_evidence, _resource_evidence, _storage_components, _tree_bytes


# Return deterministic numeric documents accepted by the fake and real backends.
#
# Arguments:
#   count (int): Number of documents to generate.
#   tokens_per_document (int): Numeric tokens in each document.
#   seed (int): Starting token offset.
#
# Returns:
#   (list[dict[str, Any]]): Documents with stable source paths.
def _documents(count: int, tokens_per_document: int, seed: int) -> list[dict[str, Any]]:
    return [
        {
            "text": " ".join(
                str(seed + index * tokens_per_document + offset)
                for offset in range(tokens_per_document)
            ),
            "metadata": {"source_path": f"doc-{index:06d}.txt"},
        }
        for index in range(count)
    ]


# Return the current maximum resident bytes reported for child processes.
#
# Returns:
#   (int): Platform-normalized child RSS, or zero when unavailable.
def _child_peak_rss_bytes() -> int:
    try:
        value = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except (AttributeError, OSError):
        return 0
    return value if sys.platform == "darwin" else value * 1024


# Measure one deterministic HKM build and its persisted artifacts.
#
# Arguments:
#   index_root (str): Destination index directory.
#   document_count (int): Number of generated documents.
#   tokens_per_document (int): Numeric tokens per generated document.
#   max_k (int): Maximum cluster count for the build.
#   seed (int): Deterministic document token seed.
#   hourly_rate (float | None): Optional cost rate in local currency per hour.
#
# Returns:
#   (dict[str, Any]): Measured build, resource, storage, hardware, and cost data.
def measure_capacity(
    index_root: str,
    document_count: int = 16,
    tokens_per_document: int = 32,
    max_k: int = 2,
    seed: int = 42,
    hourly_rate: float | None = None,
) -> dict[str, Any]:
    if document_count < 1 or tokens_per_document < 1 or max_k < 1:
        raise ValueError("document_count, tokens_per_document, and max_k must be positive")
    if hourly_rate is not None and hourly_rate < 0:
        raise ValueError("hourly_rate must be non-negative")
    root = Path(index_root).expanduser().absolute()
    documents = _documents(document_count, tokens_per_document, seed)
    source_bytes = sum(len(str(document["text"]).encode("utf-8")) for document in documents)
    child_rss_before = _child_peak_rss_bytes()
    started = time.perf_counter()
    job = build_search_index_from_documents(
        str(root), documents, num_workers=1, max_k=max_k, seed=seed
    )
    build_seconds = time.perf_counter() - started
    child_rss_after = _child_peak_rss_bytes()
    published_root = audit_index(str(root))
    searcher = Searcher.from_index_root(str(published_root))
    generation_root = Path(searcher.generation_root or searcher.hkm_root)
    persisted_resources = _resource_evidence(published_root)
    canonical_bytes = _tree_bytes(generation_root, set())
    jobs_bytes = _tree_bytes(published_root / ".hkm_jobs", set())
    builds_bytes = _tree_bytes(published_root / ".hkm_builds", set())
    cache_bytes = _tree_bytes(published_root / ".hkm_cache", set())
    child_peak = max(child_rss_before, child_rss_after)
    cost = build_seconds * hourly_rate / 3600.0 if hourly_rate is not None else None
    return {
        "measurement": {
            "status": "measured",
            "corpus": "deterministic_numeric_documents",
            "extrapolated": False,
        },
        "build": {
            "status": str(getattr(job, "status", "unknown")),
            "seconds": build_seconds,
            "documents_requested": document_count,
            "documents_indexed": searcher._doc_count(),
            "tokens_per_document": tokens_per_document,
            "stage_evidence": _build_evidence(published_root),
        },
        "resources": {
            "child_peak_rss_bytes": child_peak,
            "child_rss_scope": "maximum child RSS reported since parent process start",
            "persisted": persisted_resources,
            "rss_sample_status": (
                "measured_nonzero"
                if persisted_resources.get("nonzero_rss_samples", 0)
                else "unavailable_or_zero"
            ),
        },
        "storage": {
            "source_bytes": source_bytes,
            "canonical_index_bytes": canonical_bytes,
            "public_root_bytes": _tree_bytes(
                published_root, {".hkm_jobs", ".hkm_cache", ".hkm_builds"}
            ),
            "jobs_bytes": jobs_bytes,
            "build_roots_bytes": builds_bytes,
            "cache_bytes": cache_bytes,
            "bytes_per_source_byte": canonical_bytes / source_bytes if source_bytes else None,
            "components": _storage_components(generation_root),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "cost": {
            "hourly_rate": hourly_rate,
            "estimated_build_cost": cost,
            "basis": "wall-clock build seconds multiplied by supplied hourly rate",
        },
    }


# Measure several deterministic corpus sizes beneath one output directory.
#
# Arguments:
#   output_root (str): Parent directory for size-specific indexes.
#   document_counts (Iterable[int]): Positive corpus sizes to build.
#   tokens_per_document (int): Numeric tokens per generated document.
#   max_k (int): Maximum cluster count for each build.
#   seed (int): Deterministic document token seed.
#   hourly_rate (float | None): Optional cost rate in local currency per hour.
#
# Returns:
#   (dict[str, Any]): Measured rows with no web-scale extrapolation claim.
def measure_capacity_sweep(
    output_root: str,
    document_counts: Iterable[int] = (16, 64),
    tokens_per_document: int = 32,
    max_k: int = 2,
    seed: int = 42,
    hourly_rate: float | None = None,
) -> dict[str, Any]:
    counts = tuple(int(count) for count in document_counts)
    if not counts or any(count < 1 for count in counts):
        raise ValueError("document_counts must contain positive values")
    root = Path(output_root).expanduser().absolute()
    rows = [
        measure_capacity(
            str(root / f"documents-{count}"),
            count,
            tokens_per_document,
            max_k,
            seed,
            hourly_rate,
        )
        for count in counts
    ]
    return {
        "measurement": "observed_build_sweep",
        "document_counts": list(counts),
        "rows": rows,
        "scaling_note": "Every row is measured; no larger-corpus extrapolation is included.",
    }


# Parse arguments and print one JSON capacity report.
#
# Returns:
#   (None): Prints and optionally writes the report.
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure HKM build capacity and storage.")
    parser.add_argument("output_root")
    parser.add_argument("--documents", type=int, action="append")
    parser.add_argument("--tokens-per-document", type=int, default=32)
    parser.add_argument("--max-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hourly-rate", type=float, default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    build_logs = io.StringIO()
    with contextlib.redirect_stdout(build_logs):
        report = measure_capacity_sweep(
            args.output_root,
            args.documents or (16, 64),
            args.tokens_per_document,
            args.max_k,
            args.seed,
            args.hourly_rate,
        )
    if build_logs.getvalue():
        sys.stderr.write(build_logs.getvalue())
    payload = json.dumps(report, indent=2)
    if args.json_output:
        Path(args.json_output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
