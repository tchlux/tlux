"""
Tokenize and embed documents, writing directory-based .hkmchunk outputs.

Public surface:
- process_documents(...)  -> writes chunk dirs + per-worker summaries
- default_worker(...)     -> convenience CLI-style entry
- doc_chunk_dict(...)     -> helper to view chunk contents column-wise
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    from ..fs import FileSystem
    from ..tools.unique_count_estimator import UniqueCounter
    from ..tools.rank_estimator import RankEstimator
except ImportError:  # pragma: no cover
    from tlux.search.hkm.fs import FileSystem
    from tlux.search.hkm.tools.unique_count_estimator import UniqueCounter
    from tlux.search.hkm.tools.rank_estimator import RankEstimator

from .chunk_io import (
    CATEGORY_NULL,
    NUMBER_NULL,
    ChunkReader,
ChunkWriter,
)

DocumentBatch = Iterable[Tuple[List[str], List[List[Union[str, float]]]]]
MetadataSchema = List[Tuple[str, type]]


def _load_embedder():
    import os

    if os.getenv("HKM_FAKE_EMBEDDER") == "1":
        def _tok(texts: List[str]) -> List[List[int]]:
            return [[int(t) for t in txt.split() if t.isdigit()] for txt in texts]

        def _emb(tok_lists: List[List[int]]):
            metas = []
            all_vecs = []
            for toks in tok_lists:
                if len(toks) == 0:
                    toks = [0]
                vec = np.zeros((1, 4), dtype=np.float32)
                all_vecs.append(vec)
                metas.append((0, len(toks), len(toks)))
            embeddings = np.vstack(all_vecs)
            return embeddings, metas

        return _tok, _emb

    try:
        from ..embedder import tokenize, embed_windows  # type: ignore
    except Exception:
        from tlux.search.hkm.embedder import tokenize, embed_windows  # type: ignore
    return tokenize, embed_windows


def process_documents(
    document_output_directory: str,
    summary_output_directory: str,
    document_batches: DocumentBatch,
    metadata_schema: MetadataSchema,
    chunk_size_limit: int = 8 * 2**20,
    n_gram: int = 3,
    fs_root: Optional[str] = None,
) -> Tuple[str, str]:
    tokenize, embed_windows = _load_embedder()
    """Tokenize + embed batches, emit chunk directories and summary stats."""
    file_system = FileSystem() if fs_root is None else FileSystem(root=fs_root)
    document_output_directory = file_system.mkdir(document_output_directory, exist_ok=True)
    summary_output_directory = file_system.mkdir(summary_output_directory, exist_ok=True)

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
    document_id = 0

    for texts, metadata_list in document_batches:
        for text, metadata in zip(texts, metadata_list):
            document_id += 1
            tokens = tokenize([text])[0]
            for n in range(1, n_gram + 1):
                for i in range(len(tokens) - n + 1):
                    ngram_bytes = b"".join(token.to_bytes(4, "little") for token in tokens[i : i + n])
                    ngram_counter.add(ngram_bytes)
            embeddings, embedding_windows = embed_windows([tokens])
            doc_metadata: List[Union[int, float]] = []
            for (field_name, field_type), value in zip(metadata_schema, metadata):
                if field_type is float:
                    value = float(value)
                    number_dists[field_name].add(value)
                elif field_type is bytes:
                    if not isinstance(value, bytes):
                        raise ValueError(f"Expected bytes for field {field_name!r}")
                elif field_type in (list, dict, tuple):
                    # pass through for ValueObserver handling inside ChunkWriter
                    pass
                else:
                    hash_value = int.from_bytes(
                        hashlib.sha256(str(value).encode("ascii")).digest()[:8], "little"
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
    metadata_schema: str = "[('name', 'str'), ('num_bytes', 'int')]",
    worker_index: int = 0,
    total_workers: int = 1,
    chunk_size_limit: int = 8 * 2**20,
) -> None:
    """Process a shard of files in document_directory."""
    schema = json.loads(metadata_schema)
    type_map = {"str": str, "float": float, "json": dict, "bytes": bytes}
    parsed_schema = [(name, type_map.get(typ, str)) for name, typ in schema]
    all_files = sorted(Path(document_directory).glob("*"))
    my_files = [file for i, file in enumerate(all_files) if i % total_workers == worker_index]

    def get_document_batches() -> Iterable[Tuple[List[str], List[List[Union[str, float]]]]]:
        for file in my_files:
            text = file.read_text(encoding="utf-8")
            yield [text], [(file.name, float(len(text.encode("utf-8"))))]

    process_documents(
        output_directory,
        output_directory,
        get_document_batches(),
        parsed_schema,
        chunk_size_limit=chunk_size_limit,
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
