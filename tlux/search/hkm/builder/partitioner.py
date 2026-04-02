"""Assign documents to clusters and write per-cluster chunks + stats."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from .chunk_io import ChunkReader, ChunkWriter
from .sampler import sample_embeddings
from ..fs import FileSystem
from ..tools.preview import select_diverse, select_random
from ..tools.unique_count_estimator import UniqueCounter
from ..tools.value_seen_estimator import ValueObserver


def _doc_assignment(reader: ChunkReader, centroids: np.ndarray) -> Dict[int, list]:
    assignments: Dict[int, list] = {i: [] for i in range(centroids.shape[0])}
    for i in range(reader.document_count):
        tokens, emb, emb_meta, _ = reader[i]
        if emb.size == 0:
            continue
        dist = np.linalg.norm(centroids[None, :, :] - emb[:, None, :], axis=2)
        assignments[int(np.argmin(dist.mean(axis=0)))].append((i, tokens, emb, emb_meta))
    return assignments


def _iter_ngram_bytes(tokens: List[int], max_n_gram: int) -> List[bytes]:
    return [
        b"".join(int(tok & 0xFFFFFFFF).to_bytes(4, "little") for tok in tokens[i : i + n])
        for n in range(1, max_n_gram + 1)
        for i in range(len(tokens) - n + 1)
    ]


def _load_counter(counter_path: str) -> UniqueCounter:
    return UniqueCounter.from_bytes(Path(counter_path).read_bytes()) if Path(counter_path).exists() else UniqueCounter()


def _node_observer(counter: UniqueCounter, fp_rate: float) -> ValueObserver:
    _, _, upper = counter.estimate()
    return ValueObserver.create(capacity=max(1, int(math.ceil(upper))), fp_rate=fp_rate)


def route_chunk(
    chunk_path: str,
    hkm_dir: str,
    centroids_path: str = "",
    worker_index: int = 0,
    max_n_gram: int = 3,
    n_gram_fp_rate: float = 0.01,
    seed: int = 42,
    fs_root: str | None = None,
    force_balance: bool = False,
) -> None:
    fs = FileSystem() if fs_root is None else FileSystem(root=fs_root)
    reader = ChunkReader(str(chunk_path), metadata_schema=[])
    centroids = np.load(centroids_path) if centroids_path else np.empty((0, 0), dtype=np.float32)
    node_counter = _load_counter(os.path.join(hkm_dir, "n_gram_counter.bytes"))
    node_observer = _node_observer(node_counter, n_gram_fp_rate)
    child_counters = {i: UniqueCounter(precision=node_counter.precision) for i in range(centroids.shape[0])}
    writers: Dict[int, ChunkWriter] = {}
    doc_assign = _doc_assignment(reader, centroids) if centroids.size else {}
    if force_balance and centroids.size:
        all_docs = []
        for docs in doc_assign.values():
            all_docs.extend(docs)
        doc_assign = {i: [] for i in range(centroids.shape[0])}
        for idx, item in enumerate(all_docs):
            doc_assign[idx % centroids.shape[0]].append(item)
    for local_idx in range(reader.document_count):
        tokens, emb, emb_meta, _ = reader[local_idx]
        token_list = tokens.tolist()
        ngrams = _iter_ngram_bytes(token_list, max_n_gram)
        for ngram in ngrams:
            node_observer.add(ngram)
        if not centroids.size or emb.size == 0:
            continue
        dist = np.linalg.norm(centroids[None, :, :] - emb[:, None, :], axis=2)
        cid = int(np.argmin(dist.mean(axis=0)))
        for ngram in ngrams:
            child_counters[cid].add(ngram)
        if cid not in writers:
            cluster_dir = os.path.join(hkm_dir, f"cluster_{cid:04d}", "data", f"worker_{worker_index:04d}")
            os.makedirs(cluster_dir, exist_ok=True)
            writers[cid] = ChunkWriter(fs, cluster_dir, chunk_size_limit=8 * 2**20, metadata_schema=[])
        doc_id = int(reader.chunk_metadata().get("min_document_id", 0) or 0) + local_idx
        emb_windows = [(int(m["token_start"]), int(m["token_end"]), int(m["window_size"])) for m in emb_meta]
        writers[cid].add_document(doc_id, token_list, emb, emb_windows, [])
    for writer in writers.values():
        writer.save_chunk()
    temp_dir = Path(hkm_dir) / "_assign" / f"worker_{worker_index:04d}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "n_gram_exists.bytes").write_bytes(node_observer.to_bytes())
    for cid, counter in child_counters.items():
        (temp_dir / f"child_{cid:04d}.n_gram_counter.bytes").write_bytes(counter.to_bytes())


def route_embeddings(
    docs_dir: str,
    hkm_dir: str,
    centroids_path: str,
    seed: int = 42,
    fs_root: str | None = None,
    force_balance: bool = False,
) -> None:
    fs = FileSystem() if fs_root is None else FileSystem(root=fs_root)
    centroids = np.load(centroids_path)
    cluster_count = centroids.shape[0]
    writers: Dict[int, ChunkWriter] = {}
    cluster_embeddings: Dict[int, List[np.ndarray]] = {i: [] for i in range(cluster_count)}
    cluster_doc_counts: Dict[int, int] = {i: 0 for i in range(cluster_count)}
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
                emb_windows = [(int(m["token_start"]), int(m["token_end"]), int(m["window_size"])) for m in emb_meta]
                writer.add_document(doc_id, tokens.tolist(), emb, emb_windows, [])
                cluster_embeddings[cid].append(emb)
                cluster_doc_counts[cid] += 1
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
        with open(cluster_dir / "stats.json", "w", encoding="ascii") as f:
            json.dump({"doc_count": int(cluster_doc_counts[cid]), "emb_count": int(all_emb.shape[0]) if all_emb.size else 0}, f)


def finalize_node(
    hkm_dir: str,
    docs_dir: str,
    seed: int = 42,
    n_gram_fp_rate: float = 0.01,
    fs_root: str | None = None,
) -> None:
    node_dir = Path(hkm_dir)
    temp_root = node_dir / "_assign"
    node_manifest_path = node_dir / "node.json"
    node = json.loads(node_manifest_path.read_text(encoding="utf-8")) if node_manifest_path.exists() else {}
    node_counter = _load_counter(str(node_dir / "n_gram_counter.bytes"))
    node_exists = _node_observer(node_counter, n_gram_fp_rate)
    for shard_path in sorted(temp_root.glob("worker_*/n_gram_exists.bytes")):
        node_exists.merge(ValueObserver.from_bytes(shard_path.read_bytes()))
    (node_dir / "n_gram_exists.bytes").write_bytes(node_exists.to_bytes())
    node.update({
        "n_gram_counter_path": "n_gram_counter.bytes",
        "n_gram_exists_path": "n_gram_exists.bytes",
        "estimated_unique_ngrams": int(math.ceil(node_counter.estimate(0.0)[0])),
    })
    node_manifest_path.write_text(json.dumps(node, separators=(",", ":")), encoding="utf-8")
    depth = int(node.get("depth", 0)) + 1
    for cid, child_name in enumerate(node.get("children", [])):
        child_dir = node_dir / child_name
        child_counter = UniqueCounter(precision=node_counter.precision)
        for shard_path in sorted(temp_root.glob(f"worker_*/child_{cid:04d}.n_gram_counter.bytes")):
            child_counter.merge(UniqueCounter.from_bytes(shard_path.read_bytes()))
        (child_dir / "n_gram_counter.bytes").write_bytes(child_counter.to_bytes())
        child_docs = child_dir / "data"
        doc_count = 0
        emb_count = 0
        for chunk_path in child_docs.rglob("*.hkmchunk") if child_docs.exists() else []:
            reader = ChunkReader(str(chunk_path), metadata_schema=[])
            doc_count += reader.document_count
            emb_count += int(reader.embeddings.shape[0])
        _, _, _, preview_rnd, preview_div = sample_embeddings(str(child_docs), target=256_000, seed=seed) if child_docs.exists() else (
            np.empty((0,)), np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
        )
        if preview_rnd.size:
            np.save(child_dir / "preview_random.npy", preview_rnd)
        if preview_div.size:
            np.save(child_dir / "preview_diverse.npy", preview_div)
        (child_dir / "node.json").write_text(json.dumps({
            "depth": depth,
            "doc_count": doc_count,
            "embedding_count": emb_count,
            "is_leaf": False,
            "children": [],
            "preview_files": [name for name in ("preview_random.npy", "preview_diverse.npy") if (child_dir / name).exists()],
            "has_data": child_docs.exists(),
            "chunk_roots": [],
            "n_gram_counter_path": "n_gram_counter.bytes",
            "n_gram_exists_path": "",
            "estimated_unique_ngrams": int(math.ceil(child_counter.estimate(0.0)[0])),
        }, separators=(",", ":")), encoding="utf-8")
    shutil.rmtree(temp_root, ignore_errors=True)
