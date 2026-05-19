"""Canonical HKM search entrypoint."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..builder.chunk_io import ChunkReader
from ..embedder import get_backend
from ..fs import FileSystem, make_filesystem
from ..schema import DocumentRecord, Hit, QuerySpec, SearchResult
from ..tools.value_seen_estimator import ValueObserver

VALID_QUERY_KEYS = {
    "text",
    "mode",
    "embeddings",
    "token_sequence",
    "label_include",
    "numeric_range",
    "top_k",
    "offset",
    "filters",
}
VALID_MODES = {"hybrid", "token", "semantic"}
VALID_FILTERS = {"path_include", "path_exclude", "file_kind"}


# Resolve an index root from an exact path or an unambiguous containing path.
#
# Arguments:
#   index_root (str): Index root, hkm directory, or parent containing one index.
#
# Returns:
#   (Path): Resolved directory containing index.json.
#
def resolve_index_root(index_root: str) -> Path:
    root = Path(index_root).expanduser().resolve()
    if (root / "index.json").exists():
        return root
    if root.name == "hkm" and (root.parent / "index.json").exists():
        return root.parent
    if root.is_dir():
        candidates = sorted(path.parent for path in root.glob("*/index.json"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Multiple indexes under {root}; choose one explicitly.")
    return root


# Audit a built HKM index for the minimal files needed by query traversal.
#
# Arguments:
#   index_root (str): Root directory containing index.json.
#
# Returns:
#   (Path): Resolved index root path.
#
# Raises:
#   FileNotFoundError: If a required manifest or centroid file is missing.
#   ValueError: If a node manifest cannot be decoded.
#
def audit_index(index_root: str) -> Path:
    root = resolve_index_root(index_root)
    manifest = root / "index.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing canonical index manifest: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    hkm_root = root / data.get("hkm_path", "hkm")
    root_node = hkm_root / "node.json"
    if not root_node.exists():
        raise FileNotFoundError(f"Missing root HKM node manifest: {root_node}")
    for node_path in sorted(hkm_root.rglob("node.json")):
        try:
            node = json.loads(node_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid node manifest: {node_path}: {exc}") from exc
        node_dir = node_path.parent
        for child in node.get("children", []):
            child_node = node_dir / str(child) / "node.json"
            if not child_node.exists():
                raise FileNotFoundError(f"Missing child node manifest: {child_node}")
        if not node.get("is_leaf", False) and not (node_dir / "centroids.npy").exists():
            raise FileNotFoundError(f"Missing non-leaf centroids: {node_dir / 'centroids.npy'}")
    return root


# Audit and open a built HKM index.
#
# Arguments:
#   index_root (str): Root directory containing index.json.
#
# Returns:
#   (Searcher): Loaded searcher for the existing index.
#
def open_index(index_root: str) -> "Searcher":
    return Searcher.from_index_root(str(audit_index(index_root)))


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


# Find the case-insensitive full-query or first-term match in text.
#
# Arguments:
#   text (str): Candidate text to scan.
#   query_text (str): User query.
#
# Returns:
#   (Tuple[int, int]): Start/end character offsets, or (-1, -1).
#
def _match_start(text: str, query_text: str) -> Tuple[int, int]:
    haystack = text.lower()
    query = query_text.strip().lower()
    if not haystack or not query:
        return (-1, -1)
    idx = haystack.find(query)
    if idx >= 0:
        return (idx, idx + len(query))
    best = (-1, -1)
    for term in query.split():
        idx = haystack.find(term)
        if idx >= 0 and (best[0] < 0 or idx < best[0]):
            best = (idx, idx + len(term))
    return best


def _default_span(tokens: np.ndarray, limit: int = 64) -> Tuple[int, int]:
    return (0, min(int(tokens.shape[0]), limit))


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return ""


def _decode_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# Normalize a file-kind value for exact metadata filtering.
#
# Arguments:
#   value (str): File kind or suffix.
#
# Returns:
#   (str): Lowercase suffix-style kind.
#
def _normalize_file_kind(value: str) -> str:
    kind = value.strip().lower()
    if kind and kind != "none" and not kind.startswith("."):
        kind = "." + kind
    return kind


# Convert a hit to the stable JSON-compatible response shape.
#
# Arguments:
#   hit (Hit): Search result hit.
#
# Returns:
#   (Dict[str, object]): JSON-compatible hit record.
#
def hit_to_dict(hit: Hit) -> Dict[str, object]:
    return {
        "doc_id": hit.doc_id,
        "score": hit.score,
        "span": hit.span,
        "source_path": hit.source_path,
        "preview_text": hit.preview_text,
        "query_mode": hit.query_mode,
        "anchor_span": hit.anchor_span,
        "anchor_source_path": hit.anchor_source_path,
        "anchor_preview_text": hit.anchor_preview_text,
        "match_reasons": hit.match_reasons,
        "semantic_score": hit.semantic_score,
        "token_score": hit.token_score,
        "document": asdict(hit.document),
    }


# Convert a search result to the stable JSON-compatible response shape.
#
# Arguments:
#   result (SearchResult): Search result container.
#
# Returns:
#   (Dict[str, object]): JSON-compatible response record.
#
def search_result_to_dict(result: SearchResult) -> Dict[str, object]:
    return {
        "docs": [hit_to_dict(hit) for hit in result.docs],
        "offset": result.offset,
        "limit": result.limit,
        "count": result.count,
        "next_offset": result.next_offset,
        "query": result.query,
    }


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
    append_only: bool = False
    _doc_index: np.ndarray | None = field(default=None, init=False, repr=False)
    _doc_rows: Dict[int, np.void] = field(default_factory=dict, init=False, repr=False)
    _reader_cache: Dict[str, ChunkReader] = field(default_factory=dict, init=False, repr=False)
    _observer_cache: Dict[str, ValueObserver | None] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_index_root(cls, index_root: str, fs: FileSystem | None = None) -> "Searcher":
        root_path = Path(index_root).resolve()
        fs = fs or make_filesystem(str(root_path))
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
            append_only=bool(data.get("append_only", False)),
        )

    def _backend(self):
        return get_backend(self.backend_name)

    def _load_doc_index(self) -> np.ndarray:
        if self._doc_index is None:
            self._doc_index = np.load(Path(self.docs_root) / "doc_index.npy")
            self._doc_rows = {int(row["doc_id"]): row for row in self._doc_index}
        return self._doc_index

    def _doc_count(self) -> int:
        return int(self._load_doc_index().shape[0])

    def _active_doc_ids(self) -> set[int]:
        self._load_doc_index()
        return set(self._doc_rows)

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
        return _decode_text(meta.get("source_path"))

    def _document_record(self, doc_id: int, meta: Dict[str, object], tokens: np.ndarray) -> DocumentRecord:
        source_path = self._source_path(meta)
        source_url = _decode_text(meta.get("source_url"))
        file_kind = _decode_text(meta.get("file_kind")) or (Path(source_path).suffix if source_path else "")
        return DocumentRecord(
            doc_id=doc_id,
            source_path=source_path,
            source_type=_decode_text(meta.get("source_type")) or ("web" if source_url else ("file" if source_path else "")),
            file_kind=file_kind,
            title=_decode_text(meta.get("title")),
            section_path=_decode_text(meta.get("section_path")),
            byte_start=_decode_int(meta.get("byte_start")),
            byte_end=_decode_int(meta.get("byte_end")),
            token_start=_decode_int(meta.get("token_start")),
            token_end=_decode_int(meta.get("token_end")) or int(tokens.shape[0]),
            content_hash=_decode_text(meta.get("content_hash")),
            build_id=_decode_text(meta.get("build_id")),
            ingested_at=_decode_text(meta.get("ingested_at")),
            source_id=_decode_text(meta.get("source_id")),
            source_url=source_url,
            source_date=_decode_text(meta.get("source_date")),
            source_token_count=_decode_int(meta.get("source_token_count")),
            num_bytes=_decode_int(meta.get("num_bytes")),
            document_preview=_decode_text(meta.get("document_preview")),
        )

    def _preview(
        self,
        source_path: str,
        tokens: np.ndarray,
        span: Tuple[int, int],
        query_text: str,
        document_preview: str = "",
    ) -> str:
        needle = self._backend().detokenize([tokens[slice(*span)].tolist()])[0].strip() if span[1] > span[0] else ""
        source_file = Path(self.source_root) / source_path if source_path else None
        if source_file and source_file.exists():
            text = source_file.read_text(encoding="utf-8", errors="ignore")
            if needle:
                idx = text.find(needle)
                if idx >= 0:
                    return _snippet(text, idx, idx + len(needle))
            start, end = _match_start(text, query_text)
            if start >= 0:
                return _snippet(text, start, end)
            return _snippet(text, 0, min(len(text), 160))
        if document_preview:
            if needle:
                idx = document_preview.find(needle)
                if idx >= 0:
                    return _snippet(document_preview, idx, idx + len(needle))
            start, end = _match_start(document_preview, query_text)
            if start >= 0:
                return _snippet(document_preview, start, end)
            return _snippet(document_preview, 0, min(len(document_preview), 160))
        span_tokens = tokens[slice(*span)] if span[1] > span[0] else tokens[: min(len(tokens), 64)]
        return self._backend().detokenize([span_tokens.tolist()])[0].strip()

    def _hit(self, doc_id: int, score: float, span: Tuple[int, int], mode: str, query_text: str) -> Hit:
        tokens, meta = self._doc_context(doc_id)
        document = self._document_record(doc_id, meta, tokens)
        return Hit(
            doc_id=doc_id,
            score=float(score),
            span=span,
            document=document,
            source_path=document.source_path,
            preview_text=self._preview(document.source_path, tokens, span, query_text, document.document_preview),
            query_mode=mode,
            match_reasons=[mode] if mode in {"semantic", "token"} else [],
            semantic_score=float(score) if mode == "semantic" else 0.0,
            token_score=float(score) if mode == "token" else 0.0,
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
        active = self._active_doc_ids()
        for chunk_root in node.get("chunk_roots", []):
            for chunk_path in sorted((path / chunk_root).rglob("*.hkmchunk")):
                reader = self._chunk_reader(str(chunk_path), [])
                if reader.embeddings.size == 0:
                    continue
                for emb, meta in zip(reader.embeddings, reader.embed_index):
                    doc_id = int(meta["document_id"])
                    if doc_id not in active:
                        continue
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
        anchor_document = self._document_record(doc_id, anchor_meta, anchor_tokens)
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
            hit.anchor_source_path = anchor_document.source_path
            hit.anchor_preview_text = self._preview(
                anchor_document.source_path,
                anchor_tokens,
                anchor_span,
                "",
                anchor_document.document_preview,
            )
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
        active = self._active_doc_ids()
        for chunk_root in node.get("chunk_roots", []):
            for chunk_path in sorted((path / chunk_root).rglob("*.hkmchunk")):
                reader = self._chunk_reader(str(chunk_path), [])
                base = int(reader.chunk_metadata().get("min_document_id", 0) or 0)
                for idx in range(reader.document_count):
                    doc_id = base + idx
                    if doc_id not in active:
                        continue
                    tokens, _, _, _ = reader[idx]
                    token_bytes = tokens.tobytes()
                    pos = token_bytes.find(target)
                    while pos >= 0:
                        if pos % 4 == 0:
                            hit_pos = pos // 4
                            hits.append(self._hit(doc_id, 1.0, (hit_pos, hit_pos + len(token_sequence)), "token", query_text))
                        pos = token_bytes.find(target, pos + 4)
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

    # Normalize and validate the public query dictionary.
    #
    # Arguments:
    #   query (dict | QuerySpec): Raw query request.
    #
    # Returns:
    #   (QuerySpec): Normalized query request.
    #
    def _query_spec(self, query: Dict[str, object] | QuerySpec) -> QuerySpec:
        if isinstance(query, QuerySpec):
            spec = query
        else:
            if not isinstance(query, dict):
                raise TypeError("query must be a dict or QuerySpec")
            unknown = set(query) - VALID_QUERY_KEYS
            if unknown:
                raise ValueError(f"unknown query keys: {sorted(unknown)}")
            spec = QuerySpec(
                text=str(query.get("text", "")),
                mode=str(query.get("mode", "hybrid")),
                embeddings=query.get("embeddings", []),
                token_sequence=query.get("token_sequence", []),
                label_include=query.get("label_include", {}),
                numeric_range=query.get("numeric_range", {}),
                top_k=query.get("top_k", 10),
                offset=query.get("offset", 0),
                filters=query.get("filters", {}),
            )
        spec.text = str(spec.text or "")
        spec.mode = str(spec.mode or "hybrid")
        try:
            spec.top_k = int(spec.top_k)
            spec.offset = int(spec.offset)
        except (TypeError, ValueError):
            raise ValueError("top_k and offset must be integers")
        if spec.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        if spec.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if spec.offset < 0:
            raise ValueError("offset must be non-negative")
        if not isinstance(spec.filters, dict):
            raise ValueError("filters must be an object")
        unknown_filters = set(spec.filters) - VALID_FILTERS
        if unknown_filters:
            raise ValueError(f"unknown filters: {sorted(unknown_filters)}")
        for name, values in spec.filters.items():
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"filters.{name} must be a list of strings")
        spec.filters = {
            "path_include": list(spec.filters.get("path_include", [])),
            "path_exclude": list(spec.filters.get("path_exclude", [])),
            "file_kind": [_normalize_file_kind(value) for value in spec.filters.get("file_kind", [])],
        }
        if not (spec.text.strip() or spec.token_sequence or spec.embeddings):
            raise ValueError("query requires text, token_sequence, or embeddings")
        return spec

    # Convert a normalized query to a stable JSON-compatible record.
    #
    # Arguments:
    #   spec (QuerySpec): Normalized query.
    #
    # Returns:
    #   (Dict[str, object]): Public query record.
    #
    def _query_dict(self, spec: QuerySpec) -> Dict[str, object]:
        return {
            "text": spec.text,
            "mode": spec.mode,
            "token_sequence": list(spec.token_sequence),
            "embeddings": spec.embeddings,
            "top_k": spec.top_k,
            "offset": spec.offset,
            "filters": spec.filters,
        }

    # Check whether a hit satisfies normalized metadata filters.
    #
    # Arguments:
    #   hit (Hit): Candidate hit.
    #   filters (Dict[str, List[str]]): Normalized filters.
    #
    # Returns:
    #   (bool): True when the hit should remain in the result set.
    #
    def _filter_hit(self, hit: Hit, filters: Dict[str, List[str]]) -> bool:
        source_path = hit.source_path or ""
        includes = filters.get("path_include", [])
        excludes = filters.get("path_exclude", [])
        if includes and not any(fnmatch.fnmatch(source_path, pattern) for pattern in includes):
            return False
        if excludes and any(fnmatch.fnmatch(source_path, pattern) for pattern in excludes):
            return False
        kinds = filters.get("file_kind", [])
        if kinds and _normalize_file_kind(hit.document.file_kind) not in kinds:
            return False
        return True

    # Apply filters and offset pagination to ranked hits.
    #
    # Arguments:
    #   hits (List[Hit]): Ranked candidate hits.
    #   spec (QuerySpec): Normalized query.
    #
    # Returns:
    #   (SearchResult): Stable paginated result.
    #
    def _page(self, hits: List[Hit], spec: QuerySpec) -> SearchResult:
        filtered = [hit for hit in hits if self._filter_hit(hit, spec.filters)]
        end = spec.offset + spec.top_k
        return SearchResult(
            docs=filtered[spec.offset:end],
            offset=spec.offset,
            limit=spec.top_k,
            count=len(filtered),
            next_offset=end if end < len(filtered) else None,
            query=self._query_dict(spec),
        )

    def search(self, query_dict) -> SearchResult:
        spec = self._query_spec(query_dict)
        candidate_count = max(self._doc_count(), spec.offset + spec.top_k)
        if spec.token_sequence:
            return self._page(self._search_tokens(spec.token_sequence, candidate_count, spec.text), spec)
        if spec.embeddings:
            hits = self._search_embeddings(np.asarray(spec.embeddings[0], dtype=np.float32), candidate_count, spec.text)
            return self._page(hits, spec)
        if spec.text and spec.mode == "token":
            return self._page(self._search_tokens(self._backend().tokenize([spec.text])[0], candidate_count, spec.text), spec)
        if spec.text and spec.mode == "semantic":
            query_ids = self._backend().tokenize([spec.text])
            query_emb = self._backend().embed(query_ids, role="query")[0]
            return self._page(self._search_embeddings(query_emb, candidate_count, spec.text), spec)
        if spec.text and spec.mode == "hybrid":
            return self._page(self._search_hybrid(spec.text, candidate_count), spec)
        return self._page([], spec)

    def _search_tokens(self, token_sequence: List[int], top_k: int, query_text: str) -> List[Hit]:
        if not token_sequence:
            return []
        target = _seq_to_bytes(token_sequence)
        hits: List[Hit] = []
        self._search_token_node(Path(self.hkm_root), target, token_sequence, query_text, hits)
        if self.append_only:
            seen = {(hit.doc_id, hit.span) for hit in hits}
            for hit in self._scan_active_tokens(target, token_sequence, query_text):
                key = (hit.doc_id, hit.span)
                if key not in seen:
                    hits.append(hit)
                    seen.add(key)
        hits.sort(key=lambda hit: (hit.doc_id, hit.span))
        return hits[:top_k]

    # Verify token matches against active canonical documents.
    #
    # Arguments:
    #   target (bytes): Encoded query token sequence.
    #   token_sequence (List[int]): Query token ids.
    #   query_text (str): Raw query text.
    #
    # Returns:
    #   (List[Hit]): Active document token hits.
    #
    def _scan_active_tokens(self, target: bytes, token_sequence: List[int], query_text: str) -> List[Hit]:
        hits = []
        for doc_id in sorted(self._active_doc_ids()):
            tokens, _ = self._doc_context(doc_id)
            token_bytes = tokens.tobytes()
            pos = token_bytes.find(target)
            while pos >= 0:
                if pos % 4 == 0:
                    hit_pos = pos // 4
                    hits.append(self._hit(doc_id, 1.0, (hit_pos, hit_pos + len(token_sequence)), "token", query_text))
                pos = token_bytes.find(target, pos + 4)
        return hits

    def _search_node(self, node_dir: Path, query_emb: np.ndarray, ranked: List[Tuple[float, int, Tuple[int, int]]]) -> None:
        node = json.loads((node_dir / "node.json").read_text(encoding="utf-8"))
        if node.get("is_leaf", False):
            active = self._active_doc_ids()
            for chunk_root in node.get("chunk_roots", []):
                for chunk_path in sorted((node_dir / chunk_root).rglob("*.hkmchunk")):
                    reader = self._chunk_reader(str(chunk_path), [])
                    if reader.embeddings.size == 0:
                        continue
                    dists = np.linalg.norm(reader.embeddings - query_emb[None, :], axis=1)
                    for dist, meta in zip(dists, reader.embed_index):
                        doc_id = int(meta["document_id"])
                        if doc_id not in active:
                            continue
                        span = (int(meta["token_start"]), int(meta["token_end"]))
                        ranked.append((float(dist), doc_id, span))
            return
        centroids = np.load(node_dir / "centroids.npy")
        dists = np.linalg.norm(centroids - query_emb[None, :], axis=1)
        for idx in np.argsort(dists)[: min(2, len(node.get("children", [])))]:
            self._search_node(node_dir / node["children"][int(idx)], query_emb, ranked)

    def _search_embeddings(self, query_emb: np.ndarray, top_k: int, query_text: str) -> List[Hit]:
        ranked: List[Tuple[float, int, Tuple[int, int]]] = []
        self._search_node(Path(self.hkm_root), query_emb, ranked)
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [self._hit(doc_id, 1.0 / (1.0 + dist), span, "semantic", query_text) for dist, doc_id, span in ranked[:top_k]]

    # Find documents whose stable metadata contains the query text.
    #
    # Arguments:
    #   query_text (str): User query.
    #
    # Returns:
    #   (List[Hit]): Metadata-backed candidate hits.
    #
    def _metadata_hits(self, query_text: str) -> List[Hit]:
        hits: List[Hit] = []
        self._load_doc_index()
        for doc_id in sorted(self._doc_rows):
            tokens, meta = self._doc_context(doc_id)
            document = self._document_record(doc_id, meta, tokens)
            fields = {
                "path": document.source_path,
                "title": document.title,
                "section": document.section_path,
                "preview": document.document_preview,
            }
            reasons = [name for name, value in fields.items() if _match_start(value, query_text)[0] >= 0]
            if reasons:
                hit = self._hit(doc_id, 0.0, _default_span(tokens), "hybrid", query_text)
                hit.match_reasons = reasons
                hits.append(hit)
        return hits

    # Compute a simple additive hybrid score from evidence fields.
    #
    # Arguments:
    #   hit (Hit): Candidate hit with reason and score components.
    #
    # Returns:
    #   (float): Combined ranking score.
    #
    def _hybrid_score(self, hit: Hit) -> float:
        reasons = set(hit.match_reasons)
        score = 0.6 * hit.semantic_score + hit.token_score
        score += 0.75 if "path" in reasons else 0.0
        score += 0.50 if "title" in reasons else 0.0
        score += 0.25 if "section" in reasons else 0.0
        score += 0.35 if "preview" in reasons else 0.0
        return score

    # Merge a candidate hit into the passage-level grouped result map.
    #
    # Arguments:
    #   grouped (Dict[Tuple[int, Tuple[int, int]], Hit]): Passage-keyed best hits.
    #   hit (Hit): New candidate hit.
    #
    # Returns:
    #   (None): Mutates grouped in place.
    #
    def _merge_hybrid_hit(self, grouped: Dict[Tuple[int, Tuple[int, int]], Hit], hit: Hit) -> None:
        key = (hit.doc_id, hit.span)
        existing = grouped.get(key)
        if existing is None:
            hit.query_mode = "hybrid"
            hit.score = self._hybrid_score(hit)
            grouped[key] = hit
            return
        replace_preview = hit.token_score > existing.token_score or hit.semantic_score > existing.semantic_score
        existing.semantic_score = max(existing.semantic_score, hit.semantic_score)
        existing.token_score = max(existing.token_score, hit.token_score)
        existing.match_reasons = sorted(set(existing.match_reasons) | set(hit.match_reasons))
        if hit.span != (0, 0) and replace_preview:
            existing.span = hit.span
            existing.preview_text = hit.preview_text
        existing.score = self._hybrid_score(existing)

    # Search by combining semantic, token, and metadata evidence.
    #
    # Arguments:
    #   query_text (str): User query.
    #   top_k (int): Maximum result count.
    #
    # Returns:
    #   (List[Hit]): Ranked passage hits.
    #
    def _search_hybrid(self, query_text: str, top_k: int) -> List[Hit]:
        candidate_count = max(top_k * 4, 20)
        grouped: Dict[Tuple[int, Tuple[int, int]], Hit] = {}
        query_ids = self._backend().tokenize([query_text])
        query_tokens = query_ids[0] if query_ids else []
        query_emb = self._backend().embed(query_ids, role="query")[0]
        for hit in self._search_embeddings(query_emb, candidate_count, query_text):
            self._merge_hybrid_hit(grouped, hit)
        if query_tokens:
            for hit in self._search_tokens(query_tokens, candidate_count, query_text):
                self._merge_hybrid_hit(grouped, hit)
        for hit in self._metadata_hits(query_text):
            self._merge_hybrid_hit(grouped, hit)
        hits = sorted(grouped.values(), key=lambda hit: (-hit.score, hit.source_path, hit.doc_id, hit.span))
        return hits[:top_k]


# Resolve a user node argument to a concrete HKM node directory.
#
# Arguments:
#   searcher (Searcher): Open index searcher.
#   node_arg (str): Absolute, index-relative, or hkm-relative node path.
#
# Returns:
#   (Path): Node directory containing node.json.
#
def _resolve_node_dir(searcher: Searcher, node_arg: str) -> Path:
    raw = Path(node_arg or "hkm")
    candidates = [raw] if raw.is_absolute() else [
        Path(searcher.index_root) / raw,
        Path(searcher.hkm_root) / raw,
    ]
    if str(raw) in ("", ".", "hkm"):
        candidates.insert(0, Path(searcher.hkm_root))
    for candidate in candidates:
        if (candidate / "node.json").exists():
            return candidate
    raise FileNotFoundError(f"Missing node manifest for node path: {node_arg}")


# Convert a node path to a stable index-relative display path.
#
# Arguments:
#   searcher (Searcher): Open index searcher.
#   node_dir (Path): Node directory.
#
# Returns:
#   (str): Relative node path when possible.
#
def _relative_node_path(searcher: Searcher, node_dir: Path) -> str:
    try:
        return str(node_dir.resolve().relative_to(Path(searcher.index_root).resolve()))
    except ValueError:
        return str(node_dir)


# Build a query dictionary from direct CLI flags.
#
# Arguments:
#   args (argparse.Namespace): Parsed command-line flags.
#
# Returns:
#   (Dict[str, object]): Query request for Searcher.search().
#
def _query_from_args(args: argparse.Namespace) -> Dict[str, object]:
    filters = {
        "path_include": args.path_include,
        "path_exclude": args.path_exclude,
        "file_kind": args.file_kind,
    }
    return {
        "text": args.text,
        "mode": args.mode,
        "top_k": args.top_k,
        "offset": args.offset,
        "filters": filters,
    }


# Run the search CLI against an existing index root.
#
# Arguments:
#   None.
#
# Returns:
#   (None): Result is printed as one JSON object.
#
def main() -> None:
    parser = argparse.ArgumentParser(description="Search HKM index")
    parser.add_argument("index_root", help="Index root containing index.json")
    parser.add_argument("query_json", nargs="?", help="Path to JSON query file")
    parser.add_argument("--text", help="Plain-text query")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="hybrid", help="Search mode")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum results")
    parser.add_argument("--offset", type=int, default=0, help="Result offset")
    parser.add_argument("--path-include", action="append", default=[], help="Source path glob to include")
    parser.add_argument("--path-exclude", action="append", default=[], help="Source path glob to exclude")
    parser.add_argument("--file-kind", action="append", default=[], help="File suffix to include")
    parser.add_argument("--node", nargs="?", const="hkm", help="Print a node manifest as JSON")
    parser.add_argument("--docs", help="Print leaf documents for a node")
    parser.add_argument("--neighbors", help="Print leaf neighbors for a node")
    parser.add_argument("--doc-id", type=int, help="Anchor document id for --neighbors")
    args = parser.parse_args()

    actions = [
        bool(args.query_json),
        bool(args.text),
        args.node is not None,
        bool(args.docs),
        bool(args.neighbors),
    ]
    if sum(actions) != 1:
        parser.error("choose exactly one of query_json, --text, --node, --docs, or --neighbors")
    if args.neighbors and args.doc_id is None:
        parser.error("--neighbors requires --doc-id")
    try:
        searcher = open_index(args.index_root)
        if args.query_json:
            with open(args.query_json, "r", encoding="utf-8") as f_query:
                payload = search_result_to_dict(searcher.search(json.load(f_query)))
        elif args.text:
            payload = search_result_to_dict(searcher.search(_query_from_args(args)))
        elif args.node is not None:
            node_dir = _resolve_node_dir(searcher, args.node)
            payload = {
                "path": _relative_node_path(searcher, node_dir),
                "node": json.loads((node_dir / "node.json").read_text(encoding="utf-8")),
            }
        elif args.docs:
            node_dir = _resolve_node_dir(searcher, args.docs)
            docs = [hit_to_dict(hit) for hit in searcher.leaf_docs(node_dir)]
            payload = {"node": _relative_node_path(searcher, node_dir), "count": len(docs), "docs": docs}
        else:
            node_dir = _resolve_node_dir(searcher, args.neighbors)
            docs = [hit_to_dict(hit) for hit in searcher.leaf_neighbors(node_dir, int(args.doc_id), args.top_k)]
            payload = {
                "node": _relative_node_path(searcher, node_dir),
                "doc_id": int(args.doc_id),
                "count": len(docs),
                "docs": docs,
            }
        print(json.dumps(payload))
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
