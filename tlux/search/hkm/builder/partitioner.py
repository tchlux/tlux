"""Assign documents to clusters and write per-cluster chunks + stats."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .chunk_io import ChunkReader, ChunkWriter
from ..fs import FileSystem
from ..tools.preview import select_random, select_diverse


def _doc_assignment(reader: ChunkReader, centroids: np.ndarray) -> Dict[int, list]:
    """Assign each doc to nearest centroid based on first embedding."""
    assignments: Dict[int, list] = {i: [] for i in range(centroids.shape[0])}
    for i in range(reader.document_count):
        tokens, emb, emb_meta, _ = reader[i]
        if emb.size == 0:
            continue
        dist = np.linalg.norm(centroids - emb[0], axis=1)
        cid = int(np.argmin(dist))
        assignments[cid].append((i, tokens, emb, emb_meta))
    return assignments


def route_embeddings(docs_dir: str, hkm_dir: str, centroids_path: str, seed: int = 42, file_system: FileSystem | None = None, force_balance: bool = False) -> None:
    fs = file_system or FileSystem()
    centroids = np.load(centroids_path)
    cluster_count = centroids.shape[0]

    # prepare writers per cluster
    writers: Dict[int, ChunkWriter] = {}
    cluster_embeddings: Dict[int, List[np.ndarray]] = {i: [] for i in range(cluster_count)}

    for chunk_path in Path(docs_dir).rglob("*.hkmchunk"):
        reader = ChunkReader(str(chunk_path), metadata_schema=[])
        doc_assign = _doc_assignment(reader, centroids)
        if force_balance:
            all_docs = []
            for docs in doc_assign.values():
                all_docs.extend(docs)
            doc_assign = {i: [] for i in range(cluster_count)}
            for idx, item in enumerate(all_docs):
                doc_assign[idx % cluster_count].append(item)
        for cid, docs in doc_assign.items():
            if not docs:
                continue
            if cid not in writers:
                cluster_dir = os.path.join(hkm_dir, f"cluster_{cid:04d}", "data", "worker_0000")
                os.makedirs(cluster_dir, exist_ok=True)
                writers[cid] = ChunkWriter(fs, cluster_dir, chunk_size_limit=8 * 2**20, metadata_schema=[], emit_worker_stats=True)
            writer = writers[cid]
            for doc_local_idx, tokens, emb, emb_meta in docs:
                doc_id = reader.chunk_metadata().get("min_document_id", 0) + doc_local_idx
                emb_windows = [
                    (int(m["token_start"]), int(m["token_end"]), int(m["window_size"])) for m in emb_meta
                ]
                writer.add_document(doc_id, tokens.tolist(), emb, emb_windows, [])
                cluster_embeddings[cid].append(emb)

    # close writers and write cluster stats
    for cid, writer in writers.items():
        writer.save_chunk()
        writer.finalize_worker()
        cluster_dir = Path(os.path.join(hkm_dir, f"cluster_{cid:04d}"))
        all_emb = np.concatenate(cluster_embeddings[cid], axis=0) if cluster_embeddings[cid] else np.empty((0, 0))
        k = min(512, all_emb.shape[0]) if all_emb.size else 0
        rnd_idx = np.array(select_random(range(all_emb.shape[0]), k, seed=seed), dtype=int) if k else np.empty((0,), dtype=int)
        div_idx = np.array(select_diverse(all_emb, k, seed=seed), dtype=int) if k else np.empty((0,), dtype=int)
        if k:
            np.save(cluster_dir / "preview_random.npy", all_emb[rnd_idx])
            np.save(cluster_dir / "preview_diverse.npy", all_emb[div_idx])

        stats = {"doc_count": int(all_emb.shape[0]) if all_emb.size else 0}
        stats_path = cluster_dir / "stats.json"
        with open(stats_path, "w", encoding="ascii") as f:
            json.dump(stats, f)
