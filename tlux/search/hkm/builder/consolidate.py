"""Consolidate worker chunk outputs into a global doc_index."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

import numpy as np

from .chunk_io import ChunkReader
from ..fs import FileSystem
from ..schema import DOC_INDEX_DTYPE


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


def run_consolidate(index_root: str, fs_root: str | None = None) -> None:
    """Entry point for job execution."""
    fs = FileSystem() if fs_root is None else FileSystem(root=fs_root)
    consolidate(fs, index_root)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate worker chunk outputs.")
    parser.add_argument("index_root")
    args = parser.parse_args()
    consolidate(FileSystem(), args.index_root)
