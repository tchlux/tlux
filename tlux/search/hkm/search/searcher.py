"""Canonical HKM search entrypoint."""

from __future__ import annotations

import argparse
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
            start, end = _match_start(text, query_text)
            if start >= 0:
                return _snippet(text, start, end)
            if needle:
                idx = text.find(needle)
                if idx >= 0:
                    return _snippet(text, idx, idx + len(needle))
            return _snippet(text, 0, min(len(text), 160))
        if document_preview:
            start, end = _match_start(document_preview, query_text)
            if start >= 0:
                return _snippet(document_preview, start, end)
            if needle:
                idx = document_preview.find(needle)
                if idx >= 0:
                    return _snippet(document_preview, idx, idx + len(needle))
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
            mode=query_dict.get("mode", "hybrid"),
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
        if spec.text and spec.mode == "hybrid":
            return SearchResult(docs=self._search_hybrid(spec.text, spec.top_k))
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

    # Merge a candidate hit into the source-level grouped result map.
    #
    # Arguments:
    #   grouped (Dict[str, Hit]): Source-keyed best hits.
    #   hit (Hit): New candidate hit.
    #
    # Returns:
    #   (None): Mutates grouped in place.
    #
    def _merge_hybrid_hit(self, grouped: Dict[str, Hit], hit: Hit) -> None:
        key = hit.source_path or str(hit.doc_id)
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
    #   (List[Hit]): Ranked, source-deduplicated hybrid hits.
    #
    def _search_hybrid(self, query_text: str, top_k: int) -> List[Hit]:
        candidate_count = max(top_k * 4, 20)
        grouped: Dict[str, Hit] = {}
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
        hits = sorted(grouped.values(), key=lambda hit: (-hit.score, hit.source_path, hit.doc_id))
        return hits[:top_k]


# Run the search CLI against an existing index root.
#
# Arguments:
#   None.
#
# Returns:
#   (None): Results are printed as JSON lines.
#
def main() -> None:
    parser = argparse.ArgumentParser(description="Search HKM index")
    parser.add_argument("index_root", help="Index root containing index.json")
    parser.add_argument("query_json", help="Path to JSON query file")
    args = parser.parse_args()

    with open(args.query_json, "r", encoding="utf-8") as f_query:
        query = json.load(f_query)
    for hit in Searcher.from_index_root(args.index_root).search(query).docs:
        print(json.dumps({
            "doc_id": hit.doc_id,
            "score": hit.score,
            "span": hit.span,
            "source_path": hit.source_path,
            "preview_text": hit.preview_text,
            "query_mode": hit.query_mode,
            "match_reasons": hit.match_reasons,
            "semantic_score": hit.semantic_score,
            "token_score": hit.token_score,
            "document": asdict(hit.document),
        }))


if __name__ == "__main__":
    main()
