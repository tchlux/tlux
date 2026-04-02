"""Canonical HKM search entrypoint."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..builder.chunk_io import ChunkReader
from ..embedder import get_backend
from ..fs import FileSystem
from ..schema import Hit, QuerySpec, SearchResult
from ..tools.value_seen_estimator import ValueObserver


def _seq_to_bytes(seq: List[int]) -> bytes:
    return struct.pack("<" + "I" * len(seq), *[int(x) for x in seq]) if seq else b""


def _parse_metadata_schema(schema: List[List[str]]) -> List[Tuple[str, type]]:
    type_map = {"str": str, "float": float, "int": int, "json": dict, "bytes": bytes, "list": list, "dict": dict}
    return [(name, type_map.get(kind, str)) for name, kind in schema]


def _snippet(text: str, start: int, end: int, radius: int = 120) -> str:
    if not text:
        return ""
    lo = max(0, start - radius)
    hi = min(len(text), max(end, start) + radius)
    return " ".join(text[lo:hi].split()).strip()


def _default_span(tokens: np.ndarray, limit: int = 64) -> Tuple[int, int]:
    return (0, min(int(tokens.shape[0]), limit))


@dataclass
class Searcher:
    fs: FileSystem
    index_root: str
    source_root: str
    docs_root: str
    hkm_root: str
    metadata_schema: List[Tuple[str, type]]
    backend_name: str
    max_n_gram: int = 3
    _doc_index: np.ndarray | None = field(default=None, init=False, repr=False)
    _doc_rows: Dict[int, np.void] = field(default_factory=dict, init=False, repr=False)
    _reader_cache: Dict[str, ChunkReader] = field(default_factory=dict, init=False, repr=False)
    _observer_cache: Dict[str, ValueObserver | None] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_index_root(cls, index_root: str, fs: FileSystem | None = None) -> "Searcher":
        root_path = Path(index_root).resolve()
        fs = fs or FileSystem(root=str(root_path))
        manifest = root_path / "index.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing canonical index manifest: {manifest}")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return cls(
            fs=fs,
            index_root=str(root_path),
            source_root=data["source_root"],
            docs_root=str(root_path / data.get("docs_path", "docs")),
            hkm_root=str(root_path / data.get("hkm_path", "hkm")),
            metadata_schema=_parse_metadata_schema(data.get("metadata_schema", [])),
            backend_name=data.get("embedder_backend", "drama"),
            max_n_gram=int(data.get("max_n_gram", data.get("build_config", {}).get("max_n_gram", 3))),
        )

    def _backend(self):
        return get_backend(self.backend_name)

    def _load_doc_index(self) -> np.ndarray:
        if self._doc_index is None:
            self._doc_index = np.load(Path(self.docs_root) / "doc_index.npy")
            self._doc_rows = {int(row["doc_id"]): row for row in self._doc_index}
        return self._doc_index

    def _chunk_reader(self, chunk_path: str, metadata_schema: List[Tuple[str, type]]) -> ChunkReader:
        if chunk_path not in self._reader_cache:
            self._reader_cache[chunk_path] = ChunkReader(chunk_path, metadata_schema=metadata_schema)
        return self._reader_cache[chunk_path]

    def _doc_reader(self, doc_id: int) -> Tuple[ChunkReader, int]:
        self._load_doc_index()
        row = self._doc_rows[doc_id]
        chunk_path = str(
            Path(self.docs_root)
            / f"worker_{int(row['worker']):04d}"
            / f"shard_{int(row['shard']):08d}.hkmchunk"
        )
        return self._chunk_reader(chunk_path, self.metadata_schema), int(row["idx"])

    def _doc_context(self, doc_id: int) -> Tuple[np.ndarray, Dict[str, object]]:
        reader, idx = self._doc_reader(doc_id)
        tokens, _, _, meta_values = reader[idx]
        return tokens, {name: value for (name, _), value in zip(self.metadata_schema, meta_values)}

    def _source_path(self, meta: Dict[str, object]) -> str:
        value = meta.get("source_path")
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value
        return ""

    def _preview(self, source_path: str, tokens: np.ndarray, span: Tuple[int, int], query_text: str) -> str:
        needle = self._backend().detokenize([tokens[slice(*span)].tolist()])[0].strip() if span[1] > span[0] else ""
        source_file = Path(self.source_root) / source_path if source_path else None
        if source_file and source_file.exists():
            text = source_file.read_text(encoding="utf-8", errors="ignore")
            if query_text:
                idx = text.find(query_text)
                if idx >= 0:
                    return _snippet(text, idx, idx + len(query_text))
            if needle:
                idx = text.find(needle)
                if idx >= 0:
                    return _snippet(text, idx, idx + len(needle))
            return _snippet(text, 0, min(len(text), 160))
        span_tokens = tokens[slice(*span)] if span[1] > span[0] else tokens[: min(len(tokens), 64)]
        return self._backend().detokenize([span_tokens.tolist()])[0].strip()

    def _hit(self, doc_id: int, score: float, span: Tuple[int, int], mode: str, query_text: str) -> Hit:
        tokens, meta = self._doc_context(doc_id)
        source_path = self._source_path(meta)
        return Hit(
            doc_id=doc_id,
            score=float(score),
            span=span,
            source_path=source_path,
            preview_text=self._preview(source_path, tokens, span, query_text),
            query_mode=mode,
        )

    def _leaf_manifest(self, node_dir: str | Path) -> Tuple[Path, Dict[str, object]]:
        path = Path(node_dir)
        node = json.loads((path / "node.json").read_text(encoding="utf-8"))
        if not node.get("is_leaf", False):
            raise ValueError(f"Node is not a leaf: {path}")
        if not node.get("chunk_roots"):
            try:
                if path.resolve() == Path(self.hkm_root).resolve():
                    node["chunk_roots"] = [os.path.relpath(self.docs_root, path)]
            except Exception:
                pass
        return path, node

    def _leaf_embeddings(self, node_dir: str | Path) -> Dict[int, Tuple[np.ndarray, Tuple[int, int]]]:
        docs: Dict[int, Tuple[np.ndarray, Tuple[int, int]]] = {}
        for doc_id, windows in self._leaf_windows(node_dir).items():
            total = np.zeros_like(windows[0][0], dtype=np.float32)
            for emb, _ in windows:
                total += emb
            docs[doc_id] = (total / float(len(windows)), windows[0][1])
        return docs

    def _leaf_windows(self, node_dir: str | Path) -> Dict[int, List[Tuple[np.ndarray, Tuple[int, int]]]]:
        path, node = self._leaf_manifest(node_dir)
        docs: Dict[int, List[Tuple[np.ndarray, Tuple[int, int]]]] = {}
        for chunk_root in node.get("chunk_roots", []):
            for chunk_path in sorted((path / chunk_root).rglob("*.hkmchunk")):
                reader = self._chunk_reader(str(chunk_path), [])
                if reader.embeddings.size == 0:
                    continue
                for emb, meta in zip(reader.embeddings, reader.embed_index):
                    doc_id = int(meta["document_id"])
                    span = (int(meta["token_start"]), int(meta["token_end"]))
                    docs.setdefault(doc_id, []).append((emb.astype(np.float32, copy=True), span))
        return docs

    def leaf_docs(self, node_dir: str | Path) -> List[Hit]:
        docs = []
        for doc_id in sorted(self._leaf_embeddings(node_dir)):
            tokens, _ = self._doc_context(doc_id)
            docs.append(self._hit(doc_id, 0.0, _default_span(tokens), "browse", ""))
        return sorted(docs, key=lambda hit: (hit.source_path, hit.doc_id))

    def leaf_neighbors(self, node_dir: str | Path, doc_id: int, top_k: int = 10) -> List[Hit]:
        docs = self._leaf_windows(node_dir)
        if doc_id not in docs:
            return []
        anchor_tokens, anchor_meta = self._doc_context(doc_id)
        anchor_path = self._source_path(anchor_meta)
        anchor_windows = docs[doc_id]
        ranked = []
        for other_id, other_windows in docs.items():
            if other_id == doc_id:
                continue
            anchor_embs = np.stack([emb for emb, _ in anchor_windows], axis=0)
            other_embs = np.stack([emb for emb, _ in other_windows], axis=0)
            dists = np.linalg.norm(anchor_embs[:, None, :] - other_embs[None, :, :], axis=2)
            best = np.unravel_index(np.argmin(dists), dists.shape)
            ranked.append((
                float(dists[best]),
                other_id,
                other_windows[int(best[1])][1],
                anchor_windows[int(best[0])][1],
            ))
        ranked.sort(key=lambda item: item[0])
        hits = []
        for dist, other_id, other_span, anchor_span in ranked[:top_k]:
            hit = self._hit(other_id, 1.0 / (1.0 + dist), other_span, "neighbor", "")
            hit.anchor_span = anchor_span
            hit.anchor_source_path = anchor_path
            hit.anchor_preview_text = self._preview(anchor_path, anchor_tokens, anchor_span, "")
            hits.append(hit)
        return hits

    def _node_manifest(self, node_dir: str | Path) -> Dict[str, object]:
        return json.loads((Path(node_dir) / "node.json").read_text(encoding="utf-8"))

    def _node_observer(self, node_dir: str | Path) -> ValueObserver | None:
        path = str(Path(node_dir))
        if path not in self._observer_cache:
            node = self._node_manifest(node_dir)
            rel = node.get("n_gram_exists_path") or ""
            obs_path = Path(node_dir) / rel if rel else Path(node_dir) / "n_gram_exists.bytes"
            self._observer_cache[path] = ValueObserver.from_bytes(obs_path.read_bytes()) if obs_path.exists() else None
        return self._observer_cache[path]

    def _query_ngrams(self, token_sequence: List[int]) -> List[bytes]:
        if not token_sequence:
            return []
        n = min(len(token_sequence), self.max_n_gram)
        return [_seq_to_bytes(token_sequence[i : i + n]) for i in range(len(token_sequence) - n + 1)]

    def _scan_leaf_tokens(self, node_dir: str | Path, target: bytes, token_sequence: List[int], query_text: str) -> List[Hit]:
        hits = []
        path, node = self._leaf_manifest(node_dir)
        for chunk_root in node.get("chunk_roots", []):
            for chunk_path in sorted((path / chunk_root).rglob("*.hkmchunk")):
                reader = self._chunk_reader(str(chunk_path), [])
                base = int(reader.chunk_metadata().get("min_document_id", 0) or 0)
                for idx in range(reader.document_count):
                    tokens, _, _, _ = reader[idx]
                    pos = tokens.tobytes().find(target)
                    if pos >= 0:
                        hit_pos = pos // 4
                        hits.append(self._hit(base + idx, 1.0, (hit_pos, hit_pos + len(token_sequence)), "token", query_text))
        return hits

    def _search_token_node(self, node_dir: Path, target: bytes, token_sequence: List[int], query_text: str, hits: List[Hit]) -> None:
        node = self._node_manifest(node_dir)
        if node.get("is_leaf", False):
            hits.extend(self._scan_leaf_tokens(node_dir, target, token_sequence, query_text))
            return
        grams = self._query_ngrams(token_sequence)
        for child in node.get("children", []):
            child_dir = node_dir / child
            observer = self._node_observer(child_dir)
            if observer is not None and grams and not all(gram in observer for gram in grams):
                continue
            self._search_token_node(child_dir, target, token_sequence, query_text, hits)

    def search(self, query_dict) -> SearchResult:
        spec = query_dict if isinstance(query_dict, QuerySpec) else QuerySpec(
            text=query_dict.get("text", ""),
            mode=query_dict.get("mode", "semantic"),
            embeddings=query_dict.get("embeddings", []),
            token_sequence=query_dict.get("token_sequence", []),
            label_include=query_dict.get("label_include", {}),
            numeric_range=query_dict.get("numeric_range", {}),
            top_k=query_dict.get("top_k", 10),
        )
        if spec.token_sequence:
            return SearchResult(docs=self._search_tokens(spec.token_sequence, spec.top_k, spec.text))
        if spec.embeddings:
            return SearchResult(docs=self._search_embeddings(np.asarray(spec.embeddings[0], dtype=np.float32), spec.top_k, spec.text))
        if spec.text and spec.mode == "token":
            return SearchResult(docs=self._search_tokens(self._backend().tokenize([spec.text])[0], spec.top_k, spec.text))
        if spec.text and spec.mode == "semantic":
            query_ids = self._backend().tokenize([spec.text])
            query_emb = self._backend().embed(query_ids, role="query")[0]
            return SearchResult(docs=self._search_embeddings(query_emb, spec.top_k, spec.text))
        return SearchResult(docs=[])

    def _search_tokens(self, token_sequence: List[int], top_k: int, query_text: str) -> List[Hit]:
        target = _seq_to_bytes(token_sequence)
        hits: List[Hit] = []
        self._search_token_node(Path(self.hkm_root), target, token_sequence, query_text, hits)
        hits.sort(key=lambda hit: hit.doc_id)
        return hits[:top_k]

    def _search_node(self, node_dir: Path, query_emb: np.ndarray, best: Dict[int, Tuple[float, Tuple[int, int]]]) -> None:
        node = json.loads((node_dir / "node.json").read_text(encoding="utf-8"))
        if node.get("is_leaf", False):
            for chunk_root in node.get("chunk_roots", []):
                for chunk_path in sorted((node_dir / chunk_root).rglob("*.hkmchunk")):
                    reader = self._chunk_reader(str(chunk_path), [])
                    if reader.embeddings.size == 0:
                        continue
                    dists = np.linalg.norm(reader.embeddings - query_emb[None, :], axis=1)
                    for dist, meta in zip(dists, reader.embed_index):
                        doc_id = int(meta["document_id"])
                        span = (int(meta["token_start"]), int(meta["token_end"]))
                        if doc_id not in best or dist < best[doc_id][0]:
                            best[doc_id] = (float(dist), span)
            return
        centroids = np.load(node_dir / "centroids.npy")
        dists = np.linalg.norm(centroids - query_emb[None, :], axis=1)
        for idx in np.argsort(dists)[: min(2, len(node.get("children", [])))]:
            self._search_node(node_dir / node["children"][int(idx)], query_emb, best)

    def _search_embeddings(self, query_emb: np.ndarray, top_k: int, query_text: str) -> List[Hit]:
        best: Dict[int, Tuple[float, Tuple[int, int]]] = {}
        self._search_node(Path(self.hkm_root), query_emb, best)
        ranked = sorted(best.items(), key=lambda item: item[1][0])[:top_k]
        return [self._hit(doc_id, 1.0 / (1.0 + dist), span, "semantic", query_text) for doc_id, (dist, span) in ranked]
