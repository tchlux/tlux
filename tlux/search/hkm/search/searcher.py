"""Token n-gram searcher over directory chunks."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import heapq

import numpy as np

from ..fs import FileSystem
from ..schema import Hit, QuerySpec, SearchResult, DOC_INDEX_DTYPE
from ..builder.chunk_io import ChunkReader
from .planner import parse_query


def _seq_to_bytes(seq: List[int]) -> bytes:
    return struct.pack("<" + "I" * len(seq), *[int(x) for x in seq])


@dataclass
class Searcher:
    fs: FileSystem
    docs_root: str
    hkm_root: str = ""

    def _load_doc_index(self) -> np.ndarray:
        path = self.fs.join(self.docs_root, "doc_index.npy")
        return np.load(path)

    def _group_by_shard(self, doc_index: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        for row in doc_index:
            groups[(int(row["worker"]), int(row["shard"]))].append((int(row["doc_id"]), int(row["idx"])))
        return groups

    def search(self, query_dict) -> SearchResult:
        spec = parse_query(json.dumps(query_dict)) if isinstance(query_dict, dict) else query_dict
        if spec.embeddings:
            emb = np.array(spec.embeddings[0], dtype=np.float32)
            hits = self._search_embeddings(emb, spec.top_k)
            return SearchResult(docs=hits)
        if spec.token_sequence:
            return self._search_tokens(spec)
        return SearchResult(docs=[])

    def _search_tokens(self, spec: QuerySpec) -> SearchResult:
        target = _seq_to_bytes(spec.token_sequence)
        doc_index = self._load_doc_index()
        groups = self._group_by_shard(doc_index)

        hits: List[Hit] = []
        for (worker, shard), entries in groups.items():
            chunk_path = self.fs.join(self.docs_root, f"worker_{worker:04d}", f"shard_{shard:08d}.hkmchunk")
            if not os.path.exists(chunk_path):
                continue
            reader = ChunkReader(chunk_path, metadata_schema=[])
            for doc_id, local_idx in entries:
                tokens, _, _, _ = reader[local_idx]
                tok_bytes = tokens.tobytes()
                pos = tok_bytes.find(target)
                if pos != -1:
                    hit_pos = pos // 4
                    hits.append(Hit(doc_id, 1.0, (hit_pos, hit_pos + len(spec.token_sequence))))
                    if len(hits) >= spec.top_k:
                        return SearchResult(docs=hits)
        return SearchResult(docs=hits)

    def _search_embeddings(self, query_emb: np.ndarray, top_k: int) -> List[Hit]:
        heap: List[Tuple[float, int]] = []
        seen: set[int] = set()

        def _search_node(node_dir: str) -> None:
            cluster_dirs = sorted(Path(node_dir).glob("cluster_*"))
            centroids_path = Path(node_dir) / "centroids.npy"
            if cluster_dirs and centroids_path.exists():
                centroids = np.load(centroids_path)
                dists = np.linalg.norm(centroids - query_emb[None, :], axis=1)
                top_children = np.argsort(dists)[: min(2, len(cluster_dirs))]
                for cid in top_children:
                    _search_node(str(cluster_dirs[cid]))
                return

            data_dir = Path(node_dir) / "data"
            chunk_paths = list(data_dir.rglob("*.hkmchunk")) if data_dir.exists() else []
            for chunk_path in chunk_paths:
                reader = ChunkReader(str(chunk_path), metadata_schema=[])
                emb = reader.embeddings
                if emb.size == 0:
                    continue
                dists = np.linalg.norm(emb - query_emb[None, :], axis=1)
                embed_idx = reader.embed_index
                for dist, meta in zip(dists, embed_idx):
                    doc_id = int(meta["document_id"])
                    if doc_id in seen:
                        continue
                    heapq.heappush(heap, (-float(dist), doc_id))
                    if len(heap) > top_k:
                        heapq.heappop(heap)
                    seen.add(doc_id)

        _search_node(self.hkm_root if self.hkm_root else self.docs_root)
        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [Hit(doc_id, score, (0, 0)) for score, doc_id in results]
