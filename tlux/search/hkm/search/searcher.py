"""Token n-gram searcher over directory chunks."""

from __future__ import annotations

import json
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
        if not spec.token_sequence:
            return SearchResult(docs=[])

        target = _seq_to_bytes(spec.token_sequence)
        doc_index = self._load_doc_index()
        groups = self._group_by_shard(doc_index)

        hits: List[Hit] = []
        for (worker, shard), entries in groups.items():
            chunk_path = self.fs.join(self.docs_root, f"worker_{worker:04d}", f"shard_{shard:08d}.hkmchunk")
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
