"""Chunk I/O utilities for HKM.

This module isolates the minimal read/write surface for chunk data.
Chunks are directories named ``chunk_<min>_<max>.hkmchunk`` containing:

- tokens.bin (uint32 tokens concatenated)
- tokens_index.npy (offset, length per document)
- embeddings.npy (float32, n_embeddings x dim)
- embed_index.npy (document_id, token_start, token_end, window_size)
- metadata.npy (structured per-doc rows matching the provided schema)
- blobs.bin (optional raw blobs)
- chunk_meta.json (min/max/doc_count/embedding_dim)
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    from ..fs import FileSystem
    from ..tools.value_seen_estimator import ValueObserver
    from ..tools.unique_count_estimator import UniqueCounter
except ImportError:  # pragma: no cover
    from tlux.search.hkm.fs import FileSystem
    from tlux.search.hkm.tools.value_seen_estimator import ValueObserver
    from tlux.search.hkm.tools.unique_count_estimator import UniqueCounter

MetadataSchema = List[Tuple[str, type]]

CATEGORY_NULL = np.uint64(0xFFFFFFFFFFFFFFFF)
NUMBER_NULL = np.float32(np.nan)
OBSERVER_CAPACITY = 1 << 20
UNIQUE_PRECISION = 18  # 2**18 registers ~1.57e5, supports ~1e6 with low error
EMBEDDING_INDEX_DTYPE = np.dtype(
    [
        ("document_id", np.uint32),
        ("token_start", np.uint32),
        ("token_end", np.uint32),
        ("window_size", np.uint32),
    ]
)


def _flatten_json_pairs(obj: object, prefix: str = "") -> Iterable[Tuple[str, object]]:
    if isinstance(obj, tuple):
        obj = list(obj)
    if isinstance(obj, list):
        path = prefix if prefix else "item"
        for v in obj:
            if isinstance(v, (list, dict, tuple)):
                yield from _flatten_json_pairs(v, path)
            else:
                yield path, v
    elif isinstance(obj, dict):
        for k, v in obj.items():
            k_str = str(k)
            path = f"{prefix}.{k_str}" if prefix else k_str
            if isinstance(v, (list, dict, tuple)):
                yield from _flatten_json_pairs(v, path)
            else:
                yield path, v
    else:
        yield prefix if prefix else "", obj


def _encode_observation(path: str, value: object) -> bytes:
    path_b = path.encode("utf-8")
    val_b = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    return struct.pack("<I", len(path_b)) + path_b + struct.pack("<I", len(val_b)) + val_b


def _observe_iterable(observer: "ValueObserver", json_input: object) -> None:
    for path, val in _flatten_json_pairs(json_input):
        if isinstance(val, (type(None), bool, int, float, str)):
            observer.add(_encode_observation(path, val))
        else:
            raise ValueError("Iterable JSON contains non-scalar leaf; only None, bool, int, float, str are allowed.")


class ChunkWriter:
    """Buffer documents then flush to a directory-based .hkmchunk."""

    _BUFFER_SIZE = 8 * 2**10  # 8 KB

    def __init__(
        self,
        file_system: FileSystem,
        output_directory: str,
        chunk_size_limit: int,
        metadata_schema: MetadataSchema,
        n_gram: int = 3,
    ):
        self._fs = file_system
        self._output_directory = output_directory
        self._chunk_size_limit = chunk_size_limit
        self._metadata_schema = metadata_schema
        self._iterable_fields = [name for (name, typ) in metadata_schema if typ in (list, dict, tuple)]
        self._n_gram = max(1, int(n_gram))
        self._reset()

    def _reset(self) -> None:
        self._total_bytes = 0
        self._min_document_id = float("inf")
        self._max_document_id = -1
        self._embedding_dim: Optional[int] = None

        self._token_offsets: List[int] = []
        self._token_sizes: List[int] = []
        self._token_file = tempfile.TemporaryFile()
        self._token_writer = open(self._token_file.fileno(), "wb", buffering=self._BUFFER_SIZE, closefd=False)

        self._embedding_file = tempfile.TemporaryFile()
        self._embedding_writer = open(self._embedding_file.fileno(), "wb", buffering=self._BUFFER_SIZE, closefd=False)

        self._index_file = tempfile.TemporaryFile()
        self._index_writer = open(self._index_file.fileno(), "wb", buffering=self._BUFFER_SIZE, closefd=False)

        self._metadata_file = tempfile.TemporaryFile()
        self._metadata_writer = open(self._metadata_file.fileno(), "wb", buffering=self._BUFFER_SIZE, closefd=False)

        self._blob_file = tempfile.TemporaryFile()
        self._blob_writer = open(self._blob_file.fileno(), "wb", buffering=self._BUFFER_SIZE, closefd=False)

        self._iterable_observers = {
            name: ValueObserver.create(OBSERVER_CAPACITY) for name in self._iterable_fields
        }
        self._iterable_uniques = {
            name: UniqueCounter(precision=UNIQUE_PRECISION) for name in self._iterable_fields
        }
        self._ngram_counter = UniqueCounter(precision=UNIQUE_PRECISION)

    def _metadata_dtype(self) -> np.dtype:
        fields = []
        for field_name, field_type in self._metadata_schema:
            if field_type is float:
                fields.append((field_name, np.float32))
            elif field_type is bytes:
                fields.append((field_name + "_blob_start", np.uint64))
                fields.append((field_name + "_blob_size", np.uint64))
            elif field_type in (list, dict, tuple):
                fields.append((field_name + "_json_pad0", np.uint64))
                fields.append((field_name + "_json_pad1", np.uint64))
            else:
                fields.append((field_name, np.uint64))
        return np.dtype(fields)

    def add_document(
        self,
        document_id: int,
        tokens: List[int],
        embeddings: np.ndarray,
        embedding_windows: List[Tuple[int, int, int]],
        metadata: List[Union[int, float]],
    ) -> None:
        token_data = struct.pack(f"<{len(tokens)}I", *tokens)
        self._token_writer.flush()
        offset = self._token_file.tell()
        self._token_offsets.append(offset)
        self._token_sizes.append(len(token_data))
        self._token_writer.write(token_data)
        self._total_bytes += len(token_data)
        self._min_document_id = min(self._min_document_id, document_id)
        self._max_document_id = max(self._max_document_id, document_id)

        # n-gram counting for token sequences
        for n in range(1, self._n_gram + 1):
            for i in range(len(tokens) - n + 1):
                ngram_bytes = b"".join(int(tok).to_bytes(4, "little") for tok in tokens[i : i + n])
                self._ngram_counter.add(ngram_bytes)

        for i, (start, end, window_size) in enumerate(embedding_windows):
            embedding_row = embeddings[i]
            if self._embedding_dim is None:
                self._embedding_dim = int(embedding_row.shape[-1])
            elif self._embedding_dim != int(embedding_row.shape[-1]):
                raise RuntimeError("Embedding dimensionality changed within a chunk")
            self._embedding_writer.write(embedding_row.tobytes())
            self._total_bytes += embedding_row.nbytes
            index_entry = struct.pack("<4I", document_id, start, end, window_size)
            self._index_writer.write(index_entry)
            self._total_bytes += EMBEDDING_INDEX_DTYPE.itemsize

        meta_bytes = bytearray()
        for (field_name, field_type), value in zip(self._metadata_schema, metadata):
            if field_type is float:
                meta_bytes += NUMBER_NULL.tobytes() if value is None else np.float32(float(value)).tobytes()
            elif field_type is bytes:
                if value is None:
                    meta_bytes += struct.pack("<QQ", 0, 0)
                else:
                    self._blob_writer.flush()
                    b_off = self._blob_file.tell()
                    self._blob_writer.write(value)
                    b_size = len(value)
                    meta_bytes += struct.pack("<QQ", b_off, b_size)
                    self._total_bytes += b_size
            elif field_type in (list, dict, tuple):
                if value is not None:
                    _observe_iterable(self._iterable_observers[field_name], value)
                    for path, val in _flatten_json_pairs(value):
                        if isinstance(val, (type(None), bool, int, float, str)):
                            self._iterable_uniques[field_name].add(
                                _encode_observation(path, val)
                            )
                meta_bytes += struct.pack("<QQ", 0, 0)
            else:
                meta_bytes += CATEGORY_NULL.tobytes() if value is None else np.uint64(int(value)).tobytes()
        self._metadata_writer.write(meta_bytes)
        self._total_bytes += len(meta_bytes)

        if self._total_bytes >= self._chunk_size_limit:
            self.save_chunk()

    def save_chunk(self) -> None:
        if self._min_document_id == float("inf"):
            return
        for writer in (
            self._token_writer,
            self._embedding_writer,
            self._index_writer,
            self._metadata_writer,
        ):
            writer.flush()

        chunk_name = f"chunk_{int(self._min_document_id):08d}_{int(self._max_document_id):08d}.hkmchunk"
        chunk_dir = os.path.join(self._output_directory, chunk_name)
        if os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)

        # tokens
        self._token_file.seek(0)
        with open(os.path.join(chunk_dir, "tokens.bin"), "wb") as f_tokens:
            f_tokens.write(self._token_file.read())
        token_index = np.zeros(
            len(self._token_offsets),
            dtype=np.dtype([("offset", np.uint64), ("length", np.uint32)]),
        )
        for i, (offset, size) in enumerate(zip(self._token_offsets, self._token_sizes)):
            token_index[i]["offset"] = offset
            token_index[i]["length"] = size
        np.save(os.path.join(chunk_dir, "tokens_index.npy"), token_index)

        # embeddings
        self._embedding_file.seek(0)
        emb_bytes = self._embedding_file.read()
        if emb_bytes and self._embedding_dim is not None:
            emb_arr = np.frombuffer(emb_bytes, dtype=np.float32).reshape((-1, self._embedding_dim))
            np.save(os.path.join(chunk_dir, "embeddings.npy"), emb_arr)

        # embed index
        self._index_file.seek(0)
        index_bytes = self._index_file.read()
        if index_bytes:
            idx_arr = np.frombuffer(index_bytes, dtype=EMBEDDING_INDEX_DTYPE)
            np.save(os.path.join(chunk_dir, "embed_index.npy"), idx_arr)

        # metadata
        self._metadata_file.seek(0)
        meta_bytes = self._metadata_file.read()
        if meta_bytes:
            meta_arr = np.frombuffer(meta_bytes, dtype=self._metadata_dtype())
            np.save(os.path.join(chunk_dir, "metadata.npy"), meta_arr)

        # blobs
        self._blob_file.seek(0)
        blob_bytes = self._blob_file.read()
        if blob_bytes:
            with open(os.path.join(chunk_dir, "blobs.bin"), "wb") as f_blobs:
                f_blobs.write(blob_bytes)

        for field_name, observer in self._iterable_observers.items():
            obs_bytes = observer.to_bytes()
            if obs_bytes:
                with open(os.path.join(chunk_dir, f"observer.{field_name}.bytes"), "wb") as f_obs:
                    f_obs.write(obs_bytes)
        for field_name, counter in self._iterable_uniques.items():
            with open(os.path.join(chunk_dir, f"unique.{field_name}.bytes"), "wb") as f_uc:
                f_uc.write(counter.to_bytes())
        # token n-gram unique counter
        with open(os.path.join(chunk_dir, "n_gram_counter.bytes"), "wb") as f_ng:
            f_ng.write(self._ngram_counter.to_bytes())

        chunk_meta = {
            "min_document_id": int(self._min_document_id),
            "max_document_id": int(self._max_document_id),
            "document_count": len(self._token_offsets),
            "embedding_dim": int(self._embedding_dim) if self._embedding_dim is not None else None,
        }
        with open(os.path.join(chunk_dir, "chunk_meta.json"), "w", encoding="ascii") as f_meta:
            json.dump(chunk_meta, f_meta, separators=(",", ":"))

        for writer in (
            self._token_writer,
            self._embedding_writer,
            self._index_writer,
            self._metadata_writer,
            self._blob_writer,
        ):
            writer.close()
        self._reset()


class ChunkReader:
    """Mmap-based reader for directory chunks."""

    def __init__(self, chunk_path: str, metadata_schema: MetadataSchema):
        if not os.path.isdir(chunk_path):
            raise FileNotFoundError(chunk_path)
        self._path = Path(chunk_path)
        self._metadata_schema = metadata_schema
        self._tokens_index = np.load(self._path / "tokens_index.npy", mmap_mode="r")
        self._tokens_blob = np.memmap(self._path / "tokens.bin", mode="r", dtype=np.uint8)
        self._embeddings = np.load(self._path / "embeddings.npy", mmap_mode="r")
        self._embed_index = np.load(self._path / "embed_index.npy", mmap_mode="r")
        self._metadata = np.load(self._path / "metadata.npy", mmap_mode="r")
        blobs_path = self._path / "blobs.bin"
        self._blobs = blobs_path.read_bytes() if blobs_path.exists() else b""
        meta_path = self._path / "chunk_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="ascii") as f_meta:
                self._chunk_meta = json.load(f_meta)
        else:
            self._chunk_meta = {
                "min_document_id": None,
                "max_document_id": None,
                "document_count": int(self._tokens_index.shape[0]),
            }

    @property
    def document_count(self) -> int:
        return int(self._tokens_index.shape[0])

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @property
    def embed_index(self) -> np.ndarray:
        return self._embed_index

    def chunk_metadata(self) -> Dict[str, Union[int, None]]:
        return dict(self._chunk_meta)

    def _get_tokens(self, idx: int) -> np.ndarray:
        entry = self._tokens_index[idx]
        start = int(entry["offset"])
        length = int(entry["length"])
        raw = self._tokens_blob[start : start + length]
        return np.frombuffer(raw, dtype=np.uint32)

    def _get_metadata_row(self, idx: int):
        return self._metadata[idx]

    def __getitem__(self, i: int):
        if not (0 <= i < self.document_count):
            raise IndexError(f"Document index {i} out of range [0, {self.document_count})")

        tokens = self._get_tokens(i)
        doc_id_base = self._chunk_meta.get("min_document_id")
        doc_id = (doc_id_base + i) if doc_id_base is not None else i
        mask = self._embed_index["document_id"] == doc_id
        embedding_meta = self._embed_index[mask]
        embeddings = self._embeddings[np.where(mask)[0]] if np.any(mask) else np.empty((0,), dtype=np.float32)

        meta_row = self._get_metadata_row(i)
        meta_values: List[Union[int, float, None]] = []
        for (field_name, field_type) in self._metadata_schema:
            if field_type is float:
                v = meta_row[field_name]
                meta_values.append(None if np.isnan(v) or v.tobytes() == NUMBER_NULL.tobytes() else float(v))
            elif field_type is bytes:
                start = meta_row[field_name + "_blob_start"]
                size = meta_row[field_name + "_blob_size"]
                meta_values.append(None if size == 0 else self._blobs[start : start + size])
            elif field_type in (list, dict, tuple):
                meta_values.append(None)
            else:
                v = meta_row[field_name]
                meta_values.append(None if v == CATEGORY_NULL else int(v))
        return tokens, embeddings, embedding_meta, meta_values

    def arrays(self) -> Dict[str, object]:
        return {
            "tokens": [self._get_tokens(i) for i in range(self.document_count)],
            "embeddings": self._embeddings,
            "embed_index": self._embed_index,
            "metadata": self._metadata,
        }
