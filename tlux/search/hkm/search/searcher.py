"""Top-level search facade."""

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from ..builder.bloom import BloomFilter
from ..fs import FileSystem
from ..schema import Hit, SearchResult, QuerySpec
from .planner import parse_query


# ------------------------------------------------------------------ #
#  Bloom-filter loader (LRU cached)                                  #
# ------------------------------------------------------------------ #
@lru_cache(maxsize=1024)
def _load_bloom(path: str) -> BloomFilter:
    with open(path, "rb") as f:
        data = f.read()
    return BloomFilter.from_bytes(data)


@dataclass
class Searcher:
    """Simple searcher that performs substring search on stored docs."""

    fs: FileSystem
    docs_root: str
    hkm_root: str

    def _load_manifest(self) -> List[dict]:
        manifest_path = self.fs.join(self.hkm_root, "manifest.jsonl")
        if not self.fs.exists(manifest_path):
            return []
        with self.fs.open(manifest_path) as f:
            return [json.loads(line) for line in f]


    @staticmethod
    def _ngrams_to_le_bytes(tokens: List[int]) -> List[bytes]:
        """Return packed 1- to 3-grams from tokens as little-endian byte sequences."""
        out = []
        for i in range(len(tokens)):
            for j in range(i + 1, min(len(tokens) + 1, i + 4)):
                out.append(struct.pack("<" + "I" * (j - i), *(int(t) for t in tokens[i:j])))
        return out


    def search(self, query_dict) -> SearchResult:
        # Normalise input
        spec = (
            parse_query(json.dumps(query_dict))
            if isinstance(query_dict, dict)
            else query_dict
        )
        hits: List[Hit] = []
        manifest = self._load_manifest()
        # Decide search mode
        if spec.text:
            query_bytes = spec.text.lower().encode("utf-8")
            for entry in manifest:
                with self.fs.open(entry["path"], "rb") as f:
                    data = f.read().lower()
                pos = data.find(query_bytes)
                if pos != -1:
                    hits.append(Hit(entry["doc_id"], 1.0, (pos, pos + len(query_bytes))))
                    if len(hits) >= spec.top_k:
                        break
        elif spec.token_sequence:
            ngrams = self._ngrams_to_le_bytes(spec.token_sequence)
            for entry in manifest:
                bloom_path = entry.get("bloom")
                if bloom_path:  # quick reject if Bloom says impossible
                    bf = _load_bloom(bloom_path)
                    if any(ngram not in bf for ngram in ngrams):
                        continue  # definitely absent, skip I/O
                with self.fs.open(entry["path"], "rb") as f:
                    data = f.read()
                pos = data.find(seq_bytes)
                if pos != -1:
                    hits.append(Hit(entry["doc_id"], 1.0, (pos, pos + len(seq_bytes))))
                    if len(hits) >= spec.top_k:
                        break
        return SearchResult(docs=hits)
