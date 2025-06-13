"""
DocumentStore - zero-copy loader for the HKM corpus.

*  Designed for >10^9 docs: doc_index.npy is memory-mapped and binary-searched.
*  Supports micro-shards (<=8 MB) packing many documents per file.
*  Provides text and metadata access for production search and filtering.

Build-time requirements
-----------------------
The IndexBuilder must emit:
1. docs/worker_XXXX/shard_XXXXXXXX.bin      - Document texts.
2. docs/worker_XXXX/shard_XXXXXXXX.meta.npy - Metadata (DOC_META_DTYPE rows).
3. docs/doc_index.npy                       - Mapping doc_id -> (worker, shard, idx).

All files are NumPy `.npy` v2 or binary, <=8 MB, for efficient whole-file reads.

Usage
-----
    fs   = FileSystem()
    store = DocumentStore(fs, "idx")
    text = store.get(42)          # Get document text
    meta = store.get_metadata(42) # Get metadata dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..fs import FileSystem
from ..schema import DOC_META_DTYPE, DOC_INDEX_DTYPE


# --------------------------------------------------------------------- #
#  Main loader                                                          #
# --------------------------------------------------------------------- #
@dataclass
class DocumentStore:
    fs: FileSystem
    index_root: str

    # Internal state (created lazily)
    _doc_index: np.memmap = field(init=False, repr=False)
    _meta_cache: Dict[Tuple[int, int], np.memmap] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        idx_path = Path(self.fs.join(self.index_root, "docs", "doc_index.npy"))
        if not idx_path.exists():
            raise FileNotFoundError(f"Required file {idx_path} not found")
        self._doc_index = np.memmap(idx_path, mode="r", dtype=DOC_INDEX_DTYPE)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def get(self, doc_id: int) -> str:
        """Return UTF-8 text for *doc_id*."""
        worker, shard, rec_idx = self._lookup(doc_id)
        meta = self._load_meta(worker, shard)[rec_idx]
        text_path = Path(
            self.fs.join(
                self.index_root,
                "docs",
                f"worker_{worker:04d}",
                f"shard_{shard:08d}.bin",
            )
        )
        mm = np.memmap(text_path, dtype=np.uint8, mode="r")
        lo, hi = int(meta["text_off"]), int(meta["text_off"] + meta["text_len"])
        return bytes(mm[lo:hi]).decode("utf-8")

    def get_metadata(self, doc_id: int) -> Dict[str, float]:
        """Return metadata dictionary for *doc_id*."""
        worker, shard, rec_idx = self._lookup(doc_id)
        meta = self._load_meta(worker, shard)[rec_idx]
        return {
            "num_token_count": float(meta["num_token_count"]),
        }

    __call__ = get  # allow store(42)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #
    def _lookup(self, doc_id: int) -> Tuple[int, int, int]:
        """Binary-search doc_index for *doc_id* -> (worker, shard, idx)."""
        pos = np.searchsorted(self._doc_index["doc_id"], doc_id)
        if pos >= self._doc_index.size or self._doc_index["doc_id"][pos] != doc_id:
            raise KeyError(f"doc_id {doc_id} not found")
        rec = self._doc_index[pos]
        return int(rec["worker"]), int(rec["shard"]), int(rec["idx"])

    def _load_meta(self, worker: int, shard: int) -> np.memmap:
        """Load shard metadata with LRU caching."""
        meta_path = Path(
            self.fs.join(
                self.index_root,
                "docs",
                f"worker_{worker:04d}",
                f"shard_{shard:08d}.meta.npy",
            )
        )
        return _load_meta(meta_path)


@lru_cache(maxsize=128)
def _load_meta(meta_path: Path) -> np.memmap:
    """Load shard metadata with LRU caching."""
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return np.memmap(meta_path, mode="r", dtype=DOC_META_DTYPE)
