"""
Consolidate worker chunk outputs into root-level shards for HKM build.

Input layout (per worker):
  docs/worker_XXXX/chunk_*.hkmchunk/   # directory chunks from ChunkWriter

Emitted under index_root/docs:
  doc_index.npy     # mapping doc_id -> (worker, shard, idx)
  embed_root.npy    # stacked embeddings for root clustering
  worker_XXXX/shard_YYYYYYYY.hkmchunk/  # relocated chunks (unchanged contents)

Chunks remain directory-based; consolidation only rewrites index/embedding shards.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from ..fs import FileSystem
    from ..schema import DOC_INDEX_DTYPE
    from .chunk_io import ChunkReader
except ImportError:  # pragma: no cover
    from tlux.search.hkm.fs import FileSystem
    from tlux.search.hkm.schema import DOC_INDEX_DTYPE
    from tlux.search.hkm.builder.chunk_io import ChunkReader


def _iter_chunks(worker_dir: Path) -> List[Path]:
    return sorted(p for p in worker_dir.glob("chunk_*.hkmchunk") if p.is_dir())


def consolidate_worker(fs: FileSystem, worker_dir: str, docs_root: str, shard_start: int) -> Tuple[np.ndarray, int, list]:
    """Relocate worker chunks to docs_root/worker_x and collect doc index rows and embeddings."""
    worker_path = Path(worker_dir)
    worker_id = int(worker_path.name.split("_")[-1]) if worker_path.name.startswith("worker_") else shard_start
    dest_worker = Path(fs.join(docs_root, f"worker_{worker_id:04d}"))
    os.makedirs(dest_worker, exist_ok=True)

    rows = []
    embeddings = []
    shard_id = shard_start
    for chunk_path in _iter_chunks(worker_path):
        reader = ChunkReader(str(chunk_path), metadata_schema=[])
        shard_name = f"shard_{shard_id:08d}.hkmchunk"
        dest_chunk = dest_worker / shard_name
        if dest_chunk.exists():
            # remove existing directory if overwriting
            for root, dirs, files in os.walk(dest_chunk, topdown=False):
                for f in files:
                    os.remove(Path(root) / f)
                for d in dirs:
                    os.rmdir(Path(root) / d)
            os.rmdir(dest_chunk)
        os.rename(chunk_path, dest_chunk)

        for local_idx in range(reader.document_count):
            doc_id = reader.chunk_metadata().get("min_document_id", 0) + local_idx
            row = np.array(
                (doc_id, worker_id, shard_id, local_idx),
                dtype=DOC_INDEX_DTYPE,
            )
            rows.append(row)
        emb = reader.embeddings
        if emb is not None:
            embeddings.append(np.asarray(emb, dtype=np.float32))
        shard_id += 1
    return (np.stack(rows) if rows else np.empty((0,), dtype=DOC_INDEX_DTYPE), shard_id, embeddings)


def consolidate(fs: FileSystem, index_root: str) -> None:
    """Main entry: consolidate all worker chunks under index_root/docs."""
    docs_root = fs.join(index_root, "docs")
    worker_dirs = sorted(Path(docs_root).glob("worker_*"))
    all_rows: List[np.ndarray] = []
    all_embs: List[np.ndarray] = []
    shard_id = 0
    for worker_dir in worker_dirs:
        rows, shard_id, embs = consolidate_worker(fs, str(worker_dir), docs_root, shard_id)
        if rows.size:
            all_rows.append(rows)
        all_embs.extend(embs)
    if all_rows:
        doc_index = np.concatenate(all_rows).astype(DOC_INDEX_DTYPE, copy=False)
        doc_index.sort(order="doc_id")
        np.save(fs.join(index_root, "docs", "doc_index.npy"), doc_index)
    if all_embs:
        embed_root = np.concatenate(all_embs, axis=0)
        np.save(fs.join(index_root, "docs", "embed_root.npy"), embed_root)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate worker chunk outputs.")
    parser.add_argument("index_root")
    args = parser.parse_args()
    consolidate(FileSystem(), args.index_root)
