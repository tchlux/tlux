"""Tokenize and embed documents into chunk-directory outputs."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    from .. import embedder
    from ..fs import FileSystem, make_filesystem
    from ..schema import DEFAULT_METADATA_SCHEMA
    from ..tools.unique_count_estimator import UniqueCounter
    from ..tools.rank_estimator import RankEstimator
except ImportError:  # pragma: no cover
    from tlux.search.hkm import embedder
    from tlux.search.hkm.fs import FileSystem, make_filesystem
    from tlux.search.hkm.schema import DEFAULT_METADATA_SCHEMA
    from tlux.search.hkm.tools.unique_count_estimator import UniqueCounter
    from tlux.search.hkm.tools.rank_estimator import RankEstimator

from .chunk_io import (
    CATEGORY_NULL,
    NUMBER_NULL,
    ChunkReader,
    ChunkWriter,
)

DocumentValue = Union[str, float, int, bytes, list, dict, None]
DocumentBatch = Iterable[Tuple[List[str], List[List[DocumentValue]]]]
MetadataSchema = List[Tuple[str, type]]
DEFAULT_METADATA_SCHEMA_TEXT = json.dumps(DEFAULT_METADATA_SCHEMA)
EMBED_CACHE_WINDOWS = (32, 128, 512, 1024)
EMBED_CACHE_OVERLAP = 0.5


# Return the current UTC timestamp as a compact ISO string.
#
# Arguments:
#   None.
#
# Returns:
#   (str): Timestamp ending in Z.
#
def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# Resolve the source metadata manifest, if one exists.
#
# Arguments:
#   document_directory (str): Directory holding indexed documents.
#   source_manifest (str | None): Explicit manifest path.
#
# Returns:
#   (Path | None): Manifest path when available.
#
def _source_manifest_path(document_directory: str, source_manifest: str | None) -> Path | None:
    if source_manifest:
        return Path(source_manifest)
    root = Path(document_directory)
    for candidate in (root / "manifest.jsonl", root.parent / "manifest.jsonl"):
        if candidate.exists():
            return candidate
    return None


# Read a JSONL source manifest and index rows by relative file path.
#
# Arguments:
#   document_directory (str): Directory holding indexed documents.
#   source_manifest (str | None): Explicit manifest path.
#
# Returns:
#   (dict[str, dict]): Manifest rows keyed by document-relative path.
#
def _load_source_manifest(document_directory: str, source_manifest: str | None) -> Dict[str, dict]:
    manifest_path = _source_manifest_path(document_directory, source_manifest)
    if manifest_path is None or not manifest_path.exists():
        return {}
    rows: Dict[str, dict] = {}
    doc_root = Path(document_directory).resolve()
    base = manifest_path.parent.resolve()
    with manifest_path.open("r", encoding="utf-8") as f_manifest:
        for line in f_manifest:
            row = json.loads(line)
            file_value = str(row.get("file", "")).replace(os.sep, "/")
            if not file_value:
                continue
            keys = {file_value, Path(file_value).name}
            try:
                keys.add((base / file_value).resolve().relative_to(doc_root).as_posix())
            except ValueError:
                pass
            for key in keys:
                rows[key] = row
    return rows


# Update a metadata field when it is present in the active schema.
#
# Arguments:
#   metadata (list): Metadata values aligned to schema.
#   field_names (dict[str, int]): Metadata field positions.
#   name (str): Field name.
#   value (object): New value.
#
# Returns:
#   (None): Mutates metadata in place.
#
def _set_metadata(metadata: List[DocumentValue], field_names: Dict[str, int], name: str, value: DocumentValue) -> None:
    if name in field_names:
        metadata[field_names[name]] = value


# Return a stable cache key for a document embedding artifact.
#
# Arguments:
#   backend_name (str): Active embedder backend.
#   content_hash (str): SHA256 of the source bytes.
#
# Returns:
#   (str): Hex cache key.
#
def _embedding_cache_key(backend_name: str, content_hash: str) -> str:
    payload = {
        "backend": backend_name,
        "windows": EMBED_CACHE_WINDOWS,
        "overlap": EMBED_CACHE_OVERLAP,
        "content_hash": content_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("ascii")).hexdigest()


# Load cached tokens and embeddings for a source content hash.
#
# Arguments:
#   cache_root (str | None): Embedding cache directory.
#   backend_name (str): Active embedder backend.
#   content_hash (str): SHA256 of the source bytes.
#
# Returns:
#   (tuple[list[int], np.ndarray, list[tuple[int, int, int]]] | None): Cached
#   tokens, embeddings, and window metadata.
#
def _load_embedding_cache(
    cache_root: str | None,
    backend_name: str,
    content_hash: str,
) -> tuple[List[int], np.ndarray, List[Tuple[int, int, int]]] | None:
    if cache_root is None:
        return None
    path = Path(cache_root) / f"{_embedding_cache_key(backend_name, content_hash)}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            tokens = data["tokens"].astype(np.uint32, copy=False).tolist()
            embeddings = data["embeddings"].astype(np.float32, copy=False)
            windows = [tuple(int(v) for v in row) for row in data["windows"]]
        return tokens, embeddings, windows
    except Exception:
        return None


# Persist tokens and embeddings for reuse by later builds.
#
# Arguments:
#   cache_root (str | None): Embedding cache directory.
#   backend_name (str): Active embedder backend.
#   content_hash (str): SHA256 of the source bytes.
#   tokens (list[int]): Tokenized source document.
#   embeddings (np.ndarray): Window embedding matrix.
#   windows (list[tuple[int, int, int]]): Embedding window metadata.
#
# Returns:
#   (None): Writes the cache entry when a cache root is configured.
#
def _write_embedding_cache(
    cache_root: str | None,
    backend_name: str,
    content_hash: str,
    tokens: List[int],
    embeddings: np.ndarray,
    windows: List[Tuple[int, int, int]],
) -> None:
    if cache_root is None:
        return
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_embedding_cache_key(backend_name, content_hash)}.npz"
    if path.exists():
        return
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        tokens=np.asarray(tokens, dtype=np.uint32),
        embeddings=embeddings.astype(np.float32, copy=False),
        windows=np.asarray(windows, dtype=np.uint32),
    )
    os.replace(tmp, path)


def process_documents(
    document_output_directory: str,
    summary_output_directory: str,
    document_batches: DocumentBatch,
    metadata_schema: MetadataSchema,
    chunk_size_limit: int = 8 * 2**20,
    n_gram: int = 3,
    fs_root: Optional[str] = None,
    document_id_base: int = 0,
    max_tokens: int | None = None,
    ingest_report_path: str | None = None,
    planned_count: int = 0,
    skipped_files: List[Dict[str, str]] | None = None,
    failed_files: List[Dict[str, str]] | None = None,
    embedding_cache_dir: str | None = None,
) -> Tuple[str, str]:
    """Tokenize + embed batches, emit chunk directories and summary stats."""
    file_system = make_filesystem(fs_root)
    document_output_directory = file_system.mkdir(document_output_directory, exist_ok=True)
    summary_output_directory = file_system.mkdir(summary_output_directory, exist_ok=True)
    print(f"[worker] start output_dir={document_output_directory} summary_dir={summary_output_directory}", flush=True)

    ngram_counter = UniqueCounter()
    category_ids: Dict[str, Dict[str, int]] = {}
    category_counts: Dict[str, Dict[int, int]] = {}
    number_dists: Dict[str, RankEstimator] = {
        name: RankEstimator() for (name, typ) in metadata_schema if typ is float
    }

    chunk_writer = ChunkWriter(
        file_system,
        document_output_directory,
        chunk_size_limit,
        metadata_schema,
        emit_worker_stats=True,
    )
    document_id = int(document_id_base)

    total_docs = 0
    total_chunks = 0
    cache_hits = 0
    cache_misses = 0
    skipped_files = [] if skipped_files is None else skipped_files
    failed_files = [] if failed_files is None else failed_files

    field_names = {name: i for i, (name, _typ) in enumerate(metadata_schema)}

    def _metadata_path(metadata: List[DocumentValue]) -> str:
        for (field_name, field_type), value in zip(metadata_schema, metadata):
            if field_name == "source_path":
                if field_type is bytes and isinstance(value, bytes):
                    return value.decode("utf-8", errors="ignore")
                return str(value)
        return ""

    for batch_idx, (texts, metadata_list) in enumerate(document_batches):
        print("-"*40, flush=True)
        print(f"Document {batch_idx+1}", flush=True)
        for text, metadata in zip(texts, metadata_list):
            print("  ", repr(str(metadata)[:40]), flush=True)
            source_path = _metadata_path(metadata)
            content_hash = ""
            if "content_hash" in field_names:
                raw_hash = metadata[field_names["content_hash"]]
                content_hash = raw_hash.decode("ascii", errors="ignore") if isinstance(raw_hash, bytes) else str(raw_hash)
            backend_name = embedder.get_backend().name
            cached = _load_embedding_cache(embedding_cache_dir, backend_name, content_hash)
            if cached is None:
                try:
                    tokens = embedder.tokenize([text])[0]
                except Exception as exc:
                    failed_files.append({"path": source_path, "reason": "tokenize_error", "error": str(exc)})
                    continue
                cache_misses += 1
                embeddings = np.empty((0, 0), dtype=np.float32)
                embedding_windows: List[Tuple[int, int, int]] = []
            else:
                tokens, embeddings, embedding_windows = cached
                cache_hits += 1
            if max_tokens is not None and len(tokens) > max_tokens:
                skipped_files.append({"path": source_path, "reason": "max_tokens"})
                continue
            _set_metadata(metadata, field_names, "token_start", 0)
            _set_metadata(metadata, field_names, "token_end", len(tokens))
            if "source_token_count" in field_names and metadata[field_names["source_token_count"]] in (None, 0):
                _set_metadata(metadata, field_names, "source_token_count", len(tokens))
            total_docs += 1
            document_id += 1
            if len(tokens) == 0:
                tokens = [0]
            for n in range(1, n_gram + 1):
                for i in range(len(tokens) - n + 1):
                    ngram_bytes = b"".join(int(token & 0xFFFFFFFF).to_bytes(4, "little") for token in tokens[i : i + n])
                    ngram_counter.add(ngram_bytes)
            if cached is None:
                embeddings, embedding_windows = embedder.embed_windows(
                    [tokens],
                    window_sizes=list(EMBED_CACHE_WINDOWS),
                    window_overlap=EMBED_CACHE_OVERLAP,
                )
                _write_embedding_cache(
                    embedding_cache_dir,
                    backend_name,
                    content_hash,
                    tokens,
                    embeddings,
                    embedding_windows,
                )
            doc_metadata: List[DocumentValue] = []
            for (field_name, field_type), value in zip(metadata_schema, metadata):
                if field_type is float:
                    value = float(value)
                    number_dists[field_name].add(value)
                elif field_type is int:
                    value = None if value is None else int(value)
                elif field_type is bytes:
                    if value is not None and not isinstance(value, bytes):
                        raise ValueError(f"Expected bytes for field {field_name!r}")
                elif field_type in (list, dict, tuple):
                    # pass through for ValueObserver handling inside ChunkWriter
                    pass
                else:
                    if value is not None:
                        hash_value = int.from_bytes(
                            hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "little"
                        )
                        if field_name not in category_ids:
                            category_ids[field_name] = {}
                            category_counts[field_name] = {}
                        category_ids_for_field = category_ids[field_name]
                        if value not in category_ids_for_field:
                            category_ids_for_field[value] = hash_value
                        counts = category_counts[field_name]
                        counts[hash_value] = counts.get(hash_value, 0) + 1
                        value = hash_value
                doc_metadata.append(value)
            chunk_writer.add_document(document_id, tokens, embeddings, embedding_windows, doc_metadata)

    chunk_writer.save_chunk()
    chunk_writer.finalize_worker()
    total_chunks = getattr(chunk_writer, "chunk_index", -1) + 1

    print(f"[worker] processed_docs={total_docs} chunks={total_chunks} out={document_output_directory}", flush=True)

    if ingest_report_path is not None:
        report = {
            "planned": int(planned_count),
            "indexed": total_docs,
            "skipped": len(skipped_files),
            "failed": len(failed_files),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
        }
        Path(ingest_report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    file_system.write(
        file_system.join(summary_output_directory, "n_gram_counter.bytes"),
        ngram_counter.to_bytes(),
    )
    file_system.write(
        file_system.join(summary_output_directory, "category_map.json"),
        json.dumps(category_ids, indent=2).encode("ascii"),
    )
    for name in category_ids:
        buf = bytearray()
        cat_counts = category_counts[name]
        buf.extend(struct.pack("<Q", len(cat_counts)))
        for cat_id, count in sorted(cat_counts.items()):
            buf.extend(struct.pack("<Q", cat_id))
            buf.extend(struct.pack("<Q", count))
        file_system.write(
            file_system.join(summary_output_directory, f"categorical-dist.{name}.bytes"),
            bytes(buf),
        )
    for name, dist in number_dists.items():
        file_system.write(
            file_system.join(summary_output_directory, f"numeric-dist.{name}.bytes"),
            dist.to_bytes(),
        )
    return document_output_directory, summary_output_directory


def default_worker(
    document_directory: str,
    output_directory: str,
    metadata_schema: str = DEFAULT_METADATA_SCHEMA_TEXT,
    worker_index: int = 0,
    total_workers: int = 1,
    chunk_size_limit: int = 8 * 2**20,
    n_gram: int = 3,
    manifest_path: str | None = None,
    fs_root: str | None = None,
    doc_id_base: int = 0,
    max_tokens: int | None = None,
    source_manifest: str | None = None,
    build_id: str | None = None,
    ingested_at: str | None = None,
    embedding_cache_dir: str | None = None,
) -> None:
    """Process a shard of files in document_directory or an explicit manifest."""
    try:
        schema = json.loads(metadata_schema)
    except Exception:
        import ast
        schema = ast.literal_eval(metadata_schema)
    type_map = {"str": str, "float": float, "int": int, "json": dict, "bytes": bytes, "list": list, "dict": dict}
    parsed_schema = [(name, type_map.get(typ, str)) for name, typ in schema]

    if manifest_path is not None:
        with open(manifest_path, "r", encoding="utf-8") as f_manifest:
            manifest_rows = json.load(f_manifest)
            all_files = [Path(row.get("path", row) if isinstance(row, dict) else row) for row in manifest_rows]
    else:
        all_files = sorted(Path(document_directory).rglob("*"))
        all_files = [p for p in all_files if p.is_file()]
    # simple byte-balanced selection when manifest not provided falls back to modulo
    my_files = [file for i, file in enumerate(all_files) if (manifest_path is not None) or (i % total_workers == worker_index)]
    source_rows = _load_source_manifest(document_directory, source_manifest)
    failed_files: List[Dict[str, str]] = []
    build_id = build_id or _utc_now()
    ingested_at = ingested_at or build_id

    def get_document_batches() -> Iterable[Tuple[List[str], List[List[DocumentValue]]]]:
        for file in my_files:
            source_path = os.path.relpath(file, document_directory).replace(os.sep, "/")
            try:
                raw = file.read_bytes()
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                failed_files.append({"path": source_path, "reason": "decode_error", "error": str(exc)})
                continue
            except OSError as exc:
                failed_files.append({"path": source_path, "reason": "read_error", "error": str(exc)})
                continue
            size_bytes = len(raw)
            source_row = source_rows.get(source_path, {})
            value_map = {
                "path": file.name,
                "name": file.name,
                "source_path": source_path.encode("utf-8"),
                "source_type": ("web" if source_row.get("url") else "file").encode("utf-8"),
                "file_kind": (file.suffix or "none").encode("utf-8"),
                "title": file.stem.encode("utf-8"),
                "section_path": b"",
                "byte_start": 0,
                "byte_end": size_bytes,
                "token_start": 0,
                "token_end": 0,
                "content_hash": hashlib.sha256(raw).hexdigest().encode("ascii"),
                "build_id": build_id.encode("utf-8"),
                "ingested_at": ingested_at.encode("utf-8"),
                "source_id": str(source_row.get("id", "") or "").encode("utf-8"),
                "source_url": str(source_row.get("url", "") or "").encode("utf-8"),
                "source_date": str(source_row.get("date", "") or "").encode("utf-8"),
                "source_token_count": int(source_row.get("token_count", 0) or 0),
                "num_bytes": size_bytes,
                "document_preview": text[:512].encode("utf-8"),
            }
            metadata_row: List[DocumentValue] = []
            for field_name, _field_type in parsed_schema:
                metadata_row.append(value_map.get(field_name))
            yield [text], [metadata_row]

    process_documents(
        output_directory,
        output_directory,
        get_document_batches(),
        parsed_schema,
        chunk_size_limit=chunk_size_limit,
        n_gram=n_gram,
        fs_root=fs_root,
        document_id_base=doc_id_base,
        max_tokens=max_tokens,
        ingest_report_path=str(Path(output_directory) / "ingest_report.json"),
        planned_count=len(my_files),
        failed_files=failed_files,
        embedding_cache_dir=embedding_cache_dir,
    )


def doc_chunk_dict(
    chunk_reader: ChunkReader,
    category_map: Dict[str, Dict[int, str]] | None = None,
) -> Dict[str, list]:
    """Return a column-oriented dict of embeddings and metadata for a chunk."""
    schema = chunk_reader._metadata_schema
    cat_fields = [i for i, (_, typ) in enumerate(schema) if typ is not float]
    num_fields = [i for i, (_, typ) in enumerate(schema) if typ is float]
    blob_fields = [i for i, (_, typ) in enumerate(schema) if typ is bytes]
    cat_names = [schema[i][0] for i in cat_fields]
    num_names = [schema[i][0] for i in num_fields]
    blob_names = [schema[i][0] for i in blob_fields]

    embed_index = chunk_reader.embed_index
    embeddings = chunk_reader.embeddings
    n_embeddings = embed_index.shape[0]
    doc_count = chunk_reader.document_count

    out: Dict[str, list] = {
        "doc_id": [],
        "tokens": [],
        "embedding": [],
        "window_start": [],
        "window_end": [],
    }
    for name in cat_names + num_names + blob_names:
        out[name] = []

    tokens_array = [chunk_reader._get_tokens(i) for i in range(doc_count)]
    min_doc_id = chunk_reader.chunk_metadata().get("min_document_id", 0)
    docid_to_idx = {(min_doc_id + i): i for i in range(doc_count)}

    for emb_idx in range(n_embeddings):
        meta = embed_index[emb_idx]
        doc_id = int(meta["document_id"])
        doc_idx = docid_to_idx.get(doc_id, doc_id)
        tokens = tokens_array[doc_idx]
        out["doc_id"].append(doc_id)
        out["tokens"].append(tokens)
        out["embedding"].append(embeddings[emb_idx])
        out["window_start"].append(int(meta["token_start"]))
        out["window_end"].append(int(meta["token_end"]))

        meta_row = chunk_reader._get_metadata_row(doc_idx)
        for (field_name, field_type) in schema:
            if field_type is float:
                v = meta_row[field_name]
                value = None if np.isnan(v) or v.tobytes() == NUMBER_NULL.tobytes() else float(v)
            elif field_type is bytes:
                start = meta_row[field_name + "_blob_start"]
                size = meta_row[field_name + "_blob_size"]
                value = None if size == 0 else chunk_reader._blobs[start : start + size]
            elif field_type in (list, dict, tuple):
                value = None
            else:
                v = meta_row[field_name]
                value = None if v == CATEGORY_NULL else int(v)
                if category_map and field_name in category_map and value is not None:
                    value = category_map[field_name].get(value, value)
            out[field_name].append(value)

    return out
