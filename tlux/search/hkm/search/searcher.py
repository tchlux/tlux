"""Top-level search facade."""

import json
from dataclasses import dataclass
from typing import List

from ..fs import FileSystem
from ..schema import Hit, SearchResult, QuerySpec
from .planner import parse_query


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

    def search(self, query_dict) -> SearchResult:
        spec = parse_query(json.dumps(query_dict)) if isinstance(query_dict, dict) else query_dict
        hits: List[Hit] = []
        manifest = self._load_manifest()
        token_bytes = bytes(spec.token_sequence)
        for entry in manifest:
            with self.fs.open(entry["path"], "rb") as f:
                data = f.read()
            idx = data.find(token_bytes)
            if idx != -1:
                hits.append(Hit(doc_id=entry["doc_id"], score=1.0, span=(idx, idx + len(token_bytes))))
                if len(hits) >= spec.top_k:
                    break
        return SearchResult(docs=hits)
