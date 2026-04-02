"""Recursive HKM tree construction through the shared job manager."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..jobs import run_job
from ..tools.kmeans import kmeans
from ..tools.unique_count_estimator import UniqueCounter
from .chunk_io import ChunkReader
from .sampler import sample_embeddings


MAX_CLUSTER_COUNT: int = 1024
RANDOM_SEED: int = 42


def _count_documents(docs_dir: str) -> int:
    total = 0
    for chunk_path in Path(docs_dir).rglob("*.hkmchunk"):
        total += ChunkReader(str(chunk_path), metadata_schema=[]).document_count
    return total


def _count_embeddings(docs_dir: str) -> int:
    total = 0
    for chunk_path in Path(docs_dir).rglob("*.hkmchunk"):
        total += int(ChunkReader(str(chunk_path), metadata_schema=[]).embeddings.shape[0])
    return total


def _merge_counters(paths: List[Path]) -> UniqueCounter:
    merged = None
    for path in paths:
        counter = UniqueCounter.from_bytes(path.read_bytes())
        if merged is None:
            merged = counter
        else:
            merged.merge(counter)
    return merged or UniqueCounter()


def _ensure_node_counter(docs_dir: str, hkm_dir: str) -> Path:
    path = Path(hkm_dir) / "n_gram_counter.bytes"
    if not path.exists():
        path.write_bytes(_merge_counters(sorted(Path(docs_dir).glob("worker_*/n_gram_counter.bytes"))).to_bytes())
    return path


def _write_node_manifest(node_dir: str, docs_dir: str, depth: int, doc_count: int, embedding_count: int, is_leaf: bool, children: List[str]) -> None:
    node_path = Path(node_dir)
    data_dir = node_path / "data"
    counter_path = node_path / "n_gram_counter.bytes"
    exists_path = node_path / "n_gram_exists.bytes"
    previews = [name for name in ("preview_random.npy", "preview_diverse.npy") if (node_path / name).exists()]
    counter_estimate = int(math.ceil(UniqueCounter.from_bytes(counter_path.read_bytes()).estimate(0.0)[0])) if counter_path.exists() else 0
    with open(node_path / "node.json", "w", encoding="utf-8") as f_node:
        json.dump({
            "depth": int(depth),
            "doc_count": int(doc_count),
            "embedding_count": int(embedding_count),
            "is_leaf": bool(is_leaf),
            "children": children,
            "preview_files": previews,
            "has_data": data_dir.is_dir(),
            "chunk_roots": [os.path.relpath(docs_dir, node_dir)] if is_leaf and os.path.isdir(docs_dir) else [],
            "n_gram_counter_path": "n_gram_counter.bytes" if counter_path.exists() else "",
            "n_gram_exists_path": "n_gram_exists.bytes" if exists_path.exists() else "",
            "estimated_unique_ngrams": counter_estimate,
        }, f_node, separators=(",", ":"))


def build_cluster_index(
    index_root_directory: str,
    max_cluster_count: int = MAX_CLUSTER_COUNT,
    leaf_doc_limit: int = 2,
    max_n_gram: int = 3,
    n_gram_fp_rate: float = 0.01,
    seed: int = RANDOM_SEED,
    fs_root: Optional[str] = None,
    max_depth: int = 3,
    depth: int = 0,
) -> None:
    if not isinstance(index_root_directory, str):
        raise TypeError("index_root_directory must be a string")
    root_path = Path(index_root_directory)
    if not root_path.exists():
        raise ValueError(f"Directory does not exist: {index_root_directory}")

    docs_dir = os.path.join(index_root_directory, "docs")
    if os.path.isdir(os.path.join(index_root_directory, "data")):
        docs_dir = os.path.join(index_root_directory, "data")
    hkm_dir = index_root_directory if os.path.basename(index_root_directory).startswith("cluster_") else os.path.join(index_root_directory, "hkm")
    os.makedirs(hkm_dir, exist_ok=True)
    _ensure_node_counter(docs_dir, hkm_dir)

    sample, _, _, preview_rnd, preview_div = sample_embeddings(docs_dir, target=256_000, seed=seed)
    document_count = _count_documents(docs_dir)
    embedding_count = _count_embeddings(docs_dir)
    if preview_rnd.size:
        np.save(os.path.join(hkm_dir, "preview_random.npy"), preview_rnd)
    if preview_div.size:
        np.save(os.path.join(hkm_dir, "preview_diverse.npy"), preview_div)

    cluster_limit = min(int(sample.shape[0]) if sample.ndim else 0, max_cluster_count)
    is_leaf = sample.size == 0 or cluster_limit <= 1 or document_count <= leaf_doc_limit or depth >= max_depth
    children = [] if is_leaf else [f"cluster_{cluster_id:04d}" for cluster_id in range(cluster_limit)]
    stats = {"doc_count": int(document_count), "emb_count": int(embedding_count), "leaf": bool(is_leaf), "depth": depth}
    with open(os.path.join(hkm_dir, "stats.json"), "w", encoding="ascii") as f_stats:
        json.dump(stats, f_stats, separators=(",", ":"))
    if is_leaf:
        with open(os.path.join(hkm_dir, "chunk_meta.json"), "w", encoding="ascii") as f_meta:
            json.dump({"leaf": True, "doc_count": int(document_count)}, f_meta, separators=(",", ":"))
    else:
        if cluster_limit >= 2 and np.allclose(sample, sample[0]):
            base = sample[0]
            cluster_centers = np.stack([base + 1e-3, base - 1e-3], axis=0).astype(np.float32)[:cluster_limit]
        else:
            cluster_centers, _ = kmeans(sample, cluster_limit, seed=seed)
        np.save(os.path.join(hkm_dir, "centroids.npy"), cluster_centers)
    _write_node_manifest(hkm_dir, docs_dir, depth, document_count, embedding_count, is_leaf, children)

    assignment_jobs: List[Any] = []
    for worker_index, chunk_path in enumerate(sorted(Path(docs_dir).rglob("*.hkmchunk"))):
        assignment_jobs.append(run_job(
            "tlux.search.hkm.builder.partitioner.route_chunk",
            str(chunk_path),
            hkm_dir,
            "" if is_leaf else os.path.join(hkm_dir, "centroids.npy"),
            worker_index=worker_index,
            max_n_gram=max_n_gram,
            n_gram_fp_rate=n_gram_fp_rate,
            seed=seed,
            fs_root=fs_root,
        ))
    finalize_job = run_job(
        "tlux.search.hkm.builder.partitioner.finalize_node",
        hkm_dir,
        docs_dir,
        seed=seed,
        n_gram_fp_rate=n_gram_fp_rate,
        fs_root=fs_root,
        dependencies=assignment_jobs,
    )
    if is_leaf:
        return
    for cluster_id in range(cluster_limit):
        sub_hkm_dir = os.path.join(hkm_dir, f"cluster_{cluster_id:04d}")
        os.makedirs(sub_hkm_dir, exist_ok=True)
        run_job(
            "tlux.search.hkm.builder.recursive_index_builder.build_cluster_index",
            sub_hkm_dir,
            max_cluster_count,
            leaf_doc_limit,
            max_n_gram,
            n_gram_fp_rate,
            seed + cluster_id + 1,
            fs_root,
            max_depth,
            depth + 1,
            dependencies=[finalize_job],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an HKM index tree for a cluster.")
    parser.add_argument("index_root_directory", type=str, help="Root directory for the index structure")
    args = parser.parse_args()
    build_cluster_index(args.index_root_directory)
