"""Consolidate worker chunk outputs into a global doc_index."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from .chunk_io import ChunkReader
from ..fs import FileSystem, make_filesystem
from ..schema import DOC_INDEX_DTYPE


# Merge pre-worker and worker ingest reports into the public build summary.
#
# Arguments:
#   index_root (str): Root directory of the index.
#   workers (List[Path]): Worker output directories.
#
# Returns:
#   (None): Updates manifests/ingest_summary.json when it exists.
#
def _merge_ingest_summary(index_root: str, workers: List[Path]) -> None:
    manifest_dir = Path(index_root) / "manifests"
    summary_path = manifest_dir / "ingest_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    skipped_files = list(summary.get("skipped_files", []))
    failed_files = list(summary.get("failed_files", []))
    reasons: Dict[str, int] = dict(summary.get("skip_reasons", {}))
    indexed = 0
    for worker_path in workers:
        report_path = worker_path / "ingest_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        indexed += int(report.get("indexed", 0))
        for item in report.get("skipped_files", []):
            reason = item.get("reason", "worker_skip")
            reasons[reason] = reasons.get(reason, 0) + 1
            skipped_files.append(item)
        failed_files.extend(report.get("failed_files", []))
    summary["indexed"] = indexed
    summary["skipped"] = len(skipped_files)
    summary["failed"] = len(failed_files)
    summary["skip_reasons"] = reasons
    summary["skipped_files"] = skipped_files
    summary["failed_files"] = failed_files
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def consolidate(fs: FileSystem, index_root: str) -> None:
    """Build doc_index.npy under <index_root>/docs from worker chunk dirs."""
    docs_root = fs.join(index_root, "docs")
    workers = sorted(Path(docs_root).glob("worker_*"))
    rows: List[np.ndarray] = []
    for worker_path in workers:
        worker_id = int(worker_path.name.split("_")[-1])
        shard_id = 0
        for chunk_dir in sorted(worker_path.glob("chunk_*.hkmchunk")):
            tokens_path = chunk_dir / "tokens.bin"
            if tokens_path.exists() and tokens_path.stat().st_size == 0:
                continue
            reader = ChunkReader(str(chunk_dir), metadata_schema=[])
            for local_idx in range(reader.document_count):
                doc_id = (reader.chunk_metadata().get("min_document_id", 0) or 0) + local_idx
                row = np.array((doc_id, worker_id, shard_id, local_idx), dtype=DOC_INDEX_DTYPE)
                rows.append(row)
            # rename to shard naming
            target = worker_path / f"shard_{shard_id:08d}.hkmchunk"
            if chunk_dir != target:
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                os.rename(chunk_dir, target)
            shard_id += 1
    if rows:
        doc_index = np.stack(rows).astype(DOC_INDEX_DTYPE, copy=False)
        doc_index.sort(order="doc_id")
        np.save(fs.join(docs_root, "doc_index.npy"), doc_index)
    _merge_ingest_summary(index_root, workers)


def run_consolidate(index_root: str, fs_root: str | None = None) -> None:
    """Entry point for job execution."""
    fs = make_filesystem(fs_root)
    consolidate(fs, index_root)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate worker chunk outputs.")
    parser.add_argument("index_root")
    args = parser.parse_args()
    consolidate(make_filesystem(), args.index_root)
