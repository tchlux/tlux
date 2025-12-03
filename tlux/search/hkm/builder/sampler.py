"""Uniform reservoir sampling of embeddings from chunk directories."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .chunk_io import ChunkReader
from ..fs import FileSystem
from ..tools.preview import select_random, select_diverse


def iter_doc_embeddings(chunk_path: Path) -> Iterable[np.ndarray]:
    reader = ChunkReader(str(chunk_path), metadata_schema=[])
    for i in range(reader.document_count):
        tokens, emb, _, _ = reader[i]
        if emb.size > 0:
            yield emb[0]  # representative per doc


def sample_embeddings(docs_root: str, target: int = 256_000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random.seed(seed)
    paths = sorted(Path(docs_root).rglob("*.hkmchunk"))
    reservoir = []
    count = 0
    for p in paths:
        for emb in iter_doc_embeddings(p):
            count += 1
            if len(reservoir) < target:
                reservoir.append(emb)
            else:
                j = random.randint(0, count - 1)
                if j < target:
                    reservoir[j] = emb
    if not reservoir:
        return (
            np.empty((0,)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )
    sample = np.stack(reservoir).astype(np.float32, copy=False)
    k = min(512, sample.shape[0])
    rnd_idx = np.array(select_random(range(sample.shape[0]), k, seed=seed), dtype=int)
    div_idx = np.array(select_diverse(sample, k, seed=seed), dtype=int)
    preview_rnd = sample[rnd_idx]
    preview_div = sample[div_idx]
    return sample, rnd_idx, div_idx, preview_rnd, preview_div
