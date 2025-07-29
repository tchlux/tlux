"""
worker_tokenize_embed.py

Processes documents by tokenizing and embedding them in a single pass,
organizing results into size-limited chunk files (.hkmchunk).

Overview:
- Reads document batches once, buffering data in RAM, and saves to
  disk when chunk size limit is reached.
- Each chunk contains: tokens_{i:08d}.bin (token sequences as packed 32-bit int),
  embeddings.mmap (float32), embed_index.mmap (document id, start, end, window size),
  categories.mmap (int64 categorical metadata), numbers.mmap (float64 numerical metadata).
- Metadata schema defined as list of (field_name, type); chunk naming reflects doc id range.
- Global outputs: category_map.json (categorical value mappings), numeric_map.json
  (percentiles for numerical values).
- N-gram tracking is stubbed but included for future work.

Example usage:
    process_documents(
        output_directory="index",
        document_batches=doc_batches(),
        metadata_schema=[("name", str), ("num_bytes", float)],
        chunk_size_limit=8 * 2**20,
    )
"""

import os
import hashlib
import io
import json
import mmap
import struct
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    from ..fs import FileSystem
    from ..embedder import tokenize, embed_windows
    from ..tools.unique_count_estimator import UniqueCounter
    from ..tools.rank_estimator import RankEstimator
except ImportError:
    from tlux.search.hkm.fs import FileSystem
    from tlux.search.hkm.embedder import tokenize, embed_windows
    from tlux.search.hkm.tools.unique_count_estimator import UniqueCounter
    from tlux.search.hkm.tools.rank_estimator import RankEstimator

# Type aliases for clarity
DocumentBatch = Iterable[Tuple[List[str], List[List[Union[str, float]]]]]
MetadataSchema = List[Tuple[str, type]]

CATEGORY_NULL = np.uint64(0xFFFFFFFFFFFFFFFF)
NUMBER_NULL = np.float32(np.nan)
METADATA_NUMPY_DTYPES = {
    float: np.float32,
    int: np.int64,
    bytes: None,  # blobs are handled specially
}
EMBEDDING_INDEX_TYPE = np.dtype([
    ('document_id', np.uint32),
    ('token_start', np.uint32),
    ('token_end', np.uint32),
    ('window_size', np.uint32),
])

# ChunkBuffer: Buffers token, embedding, and metadata for documents, flushing to disk as a .hkmchunk when full.
class ChunkBuffer:
    _BUFFER_SIZE = 8 * 1024  # 8KB

    def __init__(self, file_system: FileSystem, output_directory: str, chunk_size_limit: int, metadata_schema: MetadataSchema):
        self._file_system = file_system
        self._output_directory = output_directory
        self._chunk_size_limit = chunk_size_limit
        self._metadata_schema = metadata_schema
        self._reset()
        self._document_count = 0

    def _reset(self) -> None:
        # Tracking
        self._total_bytes = 0
        self._min_document_id = float('inf')
        self._max_document_id = -1
        # Token data
        self._token_offsets: List[int] = []
        self._token_sizes: List[int] = []
        self._token_file = tempfile.TemporaryFile()
        self._token_writer = io.BufferedWriter(self._token_file, buffer_size=self._BUFFER_SIZE)
        # Embedding data
        self._embedding_file = tempfile.TemporaryFile()
        self._embedding_writer = io.BufferedWriter(self._embedding_file, buffer_size=self._BUFFER_SIZE)
        # Index data
        self._index_file = tempfile.TemporaryFile()
        self._index_writer = io.BufferedWriter(self._index_file, buffer_size=self._BUFFER_SIZE)
        # Meta data
        self._metadata_file = tempfile.TemporaryFile()
        self._metadata_writer = io.BufferedWriter(self._metadata_file, buffer_size=self._BUFFER_SIZE)
        # Blob data
        self._blob_file = tempfile.TemporaryFile()
        self._blob_writer = io.BufferedWriter(self._blob_file, buffer_size=self._BUFFER_SIZE)

    # Add document's token, embedding, and metadata to chunk.
    #
    # Parameters:
    #   document_id (int): Document index.
    #   tokens (List[int]): Token sequence.
    #   embeddings (np.ndarray): Embedding matrix, shape (n_windows, d).
    #   embedding_windows (List[Tuple[int, int, int]]): (start, end, window_size) for each embedding.
    #   metadata (List[Union[int, float]]): Document metadata, already mapped to int/float.
    #
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
        for i, (start, end, window_size) in enumerate(embedding_windows):
            embedding_row = embeddings[i]
            self._embedding_writer.write(embedding_row.tobytes())
            self._total_bytes += embedding_row.nbytes
            index_entry = struct.pack("<4I", document_id, start, end, window_size)
            self._index_writer.write(index_entry)
            self._total_bytes += EMBEDDING_INDEX_TYPE.itemsize
        meta_bytes = bytearray()
        blob_offsets_for_doc: List[Tuple[int, int]] = []
        for (field_name, field_type), value in zip(self._metadata_schema, metadata):
            if field_type is float:
                v = float(value)
                arr = np.array([v], dtype=np.float32)
                if np.isnan(v):  # already null
                    arr[0] = NUMBER_NULL
                meta_bytes += arr.tobytes()
            elif field_type is bytes:
                if value is None:
                    meta_bytes += struct.pack("<QQ", 0, 0)  # Null: offset 0, size 0
                    blob_offsets_for_doc.append((0, 0))
                else:
                    self._blob_writer.flush()
                    offset = self._blob_file.tell()
                    self._blob_writer.write(value)
                    size = len(value)
                    meta_bytes += struct.pack("<QQ", offset, size)
                    blob_offsets_for_doc.append((offset, size))
                    self._total_bytes += size
            else:
                v = int(value)
                arr = np.array([v], dtype=np.uint64)
                if v == CATEGORY_NULL:
                    arr[0] = CATEGORY_NULL
                meta_bytes += arr.tobytes()
        self._metadata_writer.write(meta_bytes)
        self._total_bytes += len(meta_bytes)
        if self._total_bytes >= self._chunk_size_limit:
            self.save_chunk()

    # Save the current chunk as a .hkmchunk file and reset buffers.
    # 
    def save_chunk(self) -> None:
        if self._min_document_id == float('inf'):
            return
        for writer in (self._token_writer, self._embedding_writer, self._index_writer, self._metadata_writer):
            writer.flush()
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_STORED) as zip_file:
            for i, (offset, size) in enumerate(zip(self._token_offsets, self._token_sizes)):
                self._token_file.seek(offset)
                data = self._token_file.read(size)
                zip_file.writestr(f"tokens_{i:08d}.bin", data)
            for filename, temp_file in [
                ("embeddings.mmap", self._embedding_file),
                ("embed_index.mmap", self._index_file),
                ("metadata.mmap", self._metadata_file),
                ("blobs.mmap", self._blob_file),  # add blobs
            ]:
                temp_file.seek(0)
                data = temp_file.read()
                if data:
                    zip_file.writestr(filename, data)
        chunk_name = f"chunk_{int(self._min_document_id):08d}_{int(self._max_document_id):08d}.hkmchunk"
        self._file_system.write(
            os.path.join(self._output_directory, chunk_name),
            zip_buffer.getvalue()
        )
        for writer in (self._token_writer, self._embedding_writer, self._index_writer, self._metadata_writer, self._blob_writer):
            writer.close()
        self._reset()



# ChunkReader: exposes token, embedding, and metadata arrays from a .hkmchunk file as memmap-style views.
class ChunkReader:
    # Create a reader for a .hkmchunk file.
    #
    # Parameters:
    #   chunk_path (str): Path to .hkmchunk file.
    #   metadata_schema (MetadataSchema): List of (name, type).
    #
    # Raises:
    #   FileNotFoundError: if chunk_path missing.
    #   RuntimeError: if required fields missing from archive.
    #
    # Returns: ChunkReader instance.
    def __init__(self, chunk_path: str, metadata_schema: MetadataSchema):
        if not os.path.isfile(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        self._chunk_path = chunk_path
        self._metadata_schema = metadata_schema
        self._zipfile = zipfile.ZipFile(chunk_path, "r")
        self._file_handles: Dict[str, Any] = {}
        self._arrays: Dict[str, np.ndarray] = {}
        self._tokens_index: List[Tuple[int, int]] = []

        self._load_token_index()
        self._load_array("embeddings.mmap", dtype=np.float32)
        self._load_array("embed_index.mmap", dtype=np.dtype([
            ('document_id', np.uint32),
            ('token_start', np.uint32),
            ('token_end', np.uint32),
            ('window_size', np.uint32),
        ]))
        self._load_array("metadata.mmap", dtype=self._metadata_row_dtype())
        self._load_blob_data("blobs.mmap")
        self._chunk_metadata = self._extract_chunk_metadata()

    # Returns a compound dtype matching the metadata schema.
    def _metadata_row_dtype(self):
        fields = []
        for field_name, field_type in self._metadata_schema:
            if field_type is float:
                fields.append((field_name, np.float32))
            elif field_type is bytes:
                fields.append((field_name + "_blob_start", np.uint64))
                fields.append((field_name + "_blob_size", np.uint64))
            else:
                fields.append((field_name, np.uint64))
        return np.dtype(fields)

    # Load blob data from the archive.
    def _load_blob_data(self, filename: str) -> None:
        if filename in self._zipfile.namelist():
            self._blobs = self._zipfile.read(filename)
        else:
            self._blobs = b""

    # Loads offsets/sizes for per-document token files within the archive.
    def _load_token_index(self) -> None:
        token_files = sorted([
            name for name in self._zipfile.namelist()
            if name.startswith("tokens_") and name.endswith(".bin")
        ])
        self._tokens_index = []
        self._token_file_names = token_files

    # Loads a memory-mapped array for a file in the ZIP archive.
    def _load_array(self, filename: str, dtype: np.dtype) -> None:
        if filename not in self._zipfile.namelist():
            self._arrays[filename] = None
            return
        data = self._zipfile.read(filename)
        arr = np.frombuffer(data, dtype=dtype)
        self._arrays[filename] = arr

    # Extracts min/max document ID and chunk doc count from chunk filename and files.
    def _extract_chunk_metadata(self) -> Dict[str, Any]:
        base = os.path.basename(self._chunk_path)
        if base.startswith("chunk_") and base.endswith(".hkmchunk"):
            parts = base[6:-9].split("_")
            if len(parts) == 2:
                try:
                    min_id = int(parts[0])
                    max_id = int(parts[1])
                    return {
                        "min_document_id": min_id,
                        "max_document_id": max_id,
                        "document_count": len(self._token_file_names),
                    }
                except Exception:
                    pass
        return {
            "min_document_id": None,
            "max_document_id": None,
            "document_count": len(self._token_file_names),
        }

    # Number of documents in chunk.
    @property
    def document_count(self) -> int:
        return len(self._token_file_names)

    # Returns: tokens, embeddings, embedding_index, metadata for document i
    #
    # Parameters:
    #   i (int): Document index in chunk [0, N)
    #
    # Returns:
    #   tokens (np.ndarray[int32]), embeddings (np.ndarray[float32, 2d]), embedding_meta (np.ndarray), metadata (List[Union[int, float]])
    #
    # Raises:
    #   IndexError: if i out of range
    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Union[int, float]]]:
        if not (0 <= i < self.document_count):
            raise IndexError(f"Document index {i} out of range [0, {self.document_count})")

        # Load tokens from per-doc .bin file in the ZIP
        token_bin = self._zipfile.read(self._token_file_names[i])
        n_tokens = len(token_bin) // 4
        tokens = np.frombuffer(token_bin, dtype=np.uint32, count=n_tokens)

        # Find all embeddings in this doc (embed_index uses document_id)
        embed_index = self._arrays["embed_index.mmap"]
        doc_id = self._chunk_metadata["min_document_id"] + i if self._chunk_metadata["min_document_id"] is not None else i
        mask = embed_index["document_id"] == doc_id
        embedding_meta = embed_index[mask]
        embeddings = self._arrays["embeddings.mmap"][np.where(mask)[0]] if embedding_meta.size > 0 else np.empty((0,))

        # Metadata
        meta_row = self._arrays["metadata.mmap"][i]
        meta_values: List[Union[int, float, None]] = []
        for (field_name, field_type) in self._metadata_schema:
            if field_type is float:
                v = meta_row[field_name]
                if np.isnan(v) or v.tobytes() == np.float32(NUMBER_NULL).tobytes():
                    meta_values.append(None)
                else:
                    meta_values.append(float(v))
            elif field_type is bytes:
                start = meta_row[field_name + "_blob_start"]
                size = meta_row[field_name + "_blob_size"]
                if size == 0:
                    meta_values.append(None)
                else:
                    meta_values.append(self._blobs[start:start+size])
            else:
                v = meta_row[field_name]
                if v == CATEGORY_NULL:
                    meta_values.append(None)
                else:
                    meta_values.append(int(v))

        return tokens, embeddings, embedding_meta, meta_values

    # Returns all per-array views.
    #
    # Returns:
    #   Dict[str, Any]: keys are array names.
    def arrays(self) -> Dict[str, Any]:
        result = dict(self._arrays)
        result["tokens"] = [self._zipfile.read(fname) for fname in self._token_file_names]
        return result

    # Returns chunk metadata dict.
    def chunk_metadata(self) -> Dict[str, Any]:
        return dict(self._chunk_metadata)

    # Close underlying ZIP handles, releases resources.
    def close(self) -> None:
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None
        self._file_handles.clear()
        self._arrays.clear()


# Process documents: tokenize, embed, and store in chunked files with metadata and n-gram tracking.
#
# Parameters:
#   document_output_directory (str): Path to store raw document tokens, embeddings, and metadata.
#   summary_output_directory (str): Path to store summary files describing whole document baches.
#   document_batches (DocumentBatch): Iterator yielding ([texts], [metadata]).
#   metadata_schema (MetadataSchema): Field name and type list.
#   chunk_size_limit (int): Chunk size in bytes.
#   n_gram (int): Max n-gram for uniqueness counting.
#
# Returns:
#   (Tuple[str, str]): The (document, summary) output directories that contains the processed data.
# 
def process_documents(
    document_output_directory: str,
    summary_output_directory: str,
    document_batches: DocumentBatch,
    metadata_schema: MetadataSchema,
    chunk_size_limit: int = 8 * 2**20,
    n_gram: int = 3,
    fs_root: Optional[str] = None,
) -> Tuple[str, str]:
    if fs_root is None:
        file_system = FileSystem()
    else:
        file_system = FileSystem(root=fs_root)
    document_output_directory = file_system.mkdir(document_output_directory, exist_ok=True)
    summary_output_directory = file_system.mkdir(summary_output_directory, exist_ok=True)
    ngram_counter = UniqueCounter()
    category_ids: Dict[str, Dict[str, int]] = {}
    category_counts: Dict[str, Dict[int, int]] = {}
    number_dists: Dist[str, RankEstimator] = {name: RankEstimator() for (name, typ) in metadata_schema if typ is float}

    # Buffered chunk writer.
    chunk_buffer = ChunkBuffer(file_system, document_output_directory, chunk_size_limit, metadata_schema)
    document_id = 0

    for texts, metadata_list in document_batches:
        for text, metadata in zip(texts, metadata_list):
            # Increment for the next document.
            document_id += 1
            # Tokenize text.
            tokens = tokenize([text])[0]
            # Count all n-grams.
            for n in range(1, n_gram + 1):
                for i in range(len(tokens) - n + 1):
                    ngram_bytes = b''.join(token.to_bytes(4, "little") for token in tokens[i:i+n])
                    ngram_counter.add(ngram_bytes)
            # Generate sliding windowed embeddings for tokens.
            embeddings, embedding_windows = embed_windows([tokens])
            # Count frequency of metadata values and ID mappings.
            doc_metadata: List[Union[int, float]] = []
            for (field_name, field_type), value in zip(metadata_schema, metadata):
                # Parse the value based on whether it's a numeric or categorical column.
                if field_type is float:
                    value = float(value)
                    number_dists[field_name].add(value)
                elif field_type is bytes:
                    if type(value) is not bytes:
                        raise(ValueError(f"Expected 'bytes' for field {repr(field_name)}, but got {type(value)}."))
                else:
                    hash_value = int.from_bytes(
                        hashlib.sha256(str(value).encode("ascii")).digest()[:8], 'little'
                    )
                    if field_name not in category_ids:
                        category_ids[field_name] = {}
                        category_counts[field_name] = {}
                    category_ids_for_field = category_ids[field_name]
                    if value not in category_ids_for_field:
                        category_ids_for_field[value] = hash_value
                    category_counts_for_field = category_counts[field_name]
                    if hash_value not in category_counts_for_field:
                        category_counts_for_field[hash_value] = 1
                    else:
                        category_counts_for_field[hash_value] += 1
                    value = hash_value
                # Add to the metadata list.
                doc_metadata.append(value)
            # Add this document, its tokens, embeddings, and metadata into the buffered writer.
            chunk_buffer.add_document(document_id, tokens, embeddings, embedding_windows, doc_metadata)

    # Make sure the chunks are all written.
    chunk_buffer.save_chunk()
    # Write the summary data.
    #   n-gram unique count
    file_system.write(
        file_system.join(summary_output_directory, 'n_gram_counter.bytes'),
        ngram_counter.to_bytes()
    )
    #   category ID mapping (string -> int64)
    file_system.write(
        file_system.join(summary_output_directory, 'category_map.json'),
        json.dumps(category_ids, indent=2).encode("ascii")
    )
    #   categorical metadata column value counts
    for name in category_ids:
        num_cats = len(category_counts[name])
        buf = bytearray()
        buf.extend(struct.pack("<Q", num_cats))  # uint64 number of categories
        for cat_id, count in sorted(category_counts[name].items()):
            buf.extend(struct.pack("<Q", cat_id))
            buf.extend(struct.pack("<Q", count))
        file_system.write(
            file_system.join(summary_output_directory, f'categorical-dist.{name}.bytes'),
            bytes(buf)
        )
    #   numeric metadata column distribution
    for name in number_dists:
        file_system.write(
            file_system.join(summary_output_directory, f'numeric-dist.{name}.bytes'),
            number_dists[name].to_bytes()
        )
    # Return the now populated output directories.
    return document_output_directory, summary_output_directory

# Entrypoint for distributed/parallel worker.
#
# Parameters:
#   document_directory (str): Input directory.
#   output_directory (str): Output directory.
#   metadata_schema (str): JSON-encoded schema.
#   worker_index (int): Worker index (for sharding).
#   total_workers (int): Number of workers.
#   chunk_size_limit (int): Chunk size in bytes.
#
def default_worker(
    document_directory: str,
    output_directory: str,
    metadata_schema: str = "[('name', 'str'), ('num_bytes', 'int')]",
    worker_index: int = 0,
    total_workers: int = 1,
    chunk_size_limit: int = 8 * 2**20,
) -> None:
    schema = json.loads(metadata_schema)
    parsed_schema = [(name, str if typ == 'str' else float if typ == 'float' else bytes) for name, typ in schema]
    all_files = sorted(Path(document_directory).glob('*'))
    my_files = [file for i, file in enumerate(all_files) if i % total_workers == worker_index]

    def get_document_batches() -> Iterable[Tuple[List[str], List[List[Union[str, float]]]]]:
        for file in my_files:
            text = file.read_text(encoding="utf-8")
            yield [text], [(file.name, float(len(text.encode("utf-8"))))]

    process_documents(
        output_directory,
        get_document_batches(),
        parsed_schema,
        chunk_size_limit=chunk_size_limit,
    )


# Construct a column-oriented dictionary representing all embedding rows in the chunk.
#
# Parameters:
#   chunk_reader (ChunkReader): Loaded chunk instance.
#   category_map_path (Optional[str]): Path to category_map.json for decoding categorical fields.
#
# Returns:
#   Dict[str, List]: Keys are column names, values are lists for DataFrame construction.
#
def doc_chunk_dict(
    chunk_reader: "ChunkReader",
    category_map: Dict[str, Dict[int, str]] = None,
) -> Dict[str, list]:
    schema = chunk_reader._metadata_schema
    cat_fields = [i for i, (_, typ) in enumerate(schema) if typ is not float]
    num_fields = [i for i, (_, typ) in enumerate(schema) if typ is float]
    blob_fields = [i for i, (_, typ) in enumerate(schema) if typ is bytes]
    cat_names = [schema[i][0] for i in cat_fields]
    num_names = [schema[i][0] for i in num_fields]
    blob_names = [schema[i][0] for i in blob_fields]
    
    embed_index = chunk_reader._arrays["embed_index.mmap"]
    embeddings = chunk_reader._arrays["embeddings.mmap"]
    n_embeddings = embed_index.shape[0]
    doc_count = chunk_reader.document_count

    # Prepare lists for each column
    out: Dict[str, list] = {}
    out["doc_id"] = []
    out["tokens"] = []
    out["embedding"] = []
    out["window_start"] = []
    out["window_end"] = []
    for name in cat_names:
        out[name] = []
    for name in num_names:
        out[name] = []
    for name in blob_names:
        out[name] = []

    # Preload all tokens to avoid repeated ZIP reads
    tokens_list = [chunk_reader._zipfile.read(fname) for fname in chunk_reader._token_file_names]
    tokens_array = [np.frombuffer(tok, dtype=np.uint32) for tok in tokens_list]

    # Map from document_id (in embed_index) to local index
    min_doc_id = chunk_reader._chunk_metadata.get("min_document_id", 0)
    docid_to_idx = {
        (min_doc_id + i): i for i in range(doc_count)
    }

    # Build embedding-wise columns
    for emb_idx in range(n_embeddings):
        meta = embed_index[emb_idx]
        doc_id = int(meta["document_id"])
        doc_idx = docid_to_idx.get(doc_id, doc_id)
        window_start = int(meta["token_start"])
        window_end = int(meta["token_end"])
        tokens = tokens_array[doc_idx]
        embedding_vec = embeddings[emb_idx]
        out["doc_id"].append(doc_id)
        out["tokens"].append(tokens)
        out["embedding"].append(embedding_vec)
        out["window_start"].append(window_start)
        out["window_end"].append(window_end)

        # Add metadata for this embedding/document
        meta_row = chunk_reader._arrays["metadata.mmap"][doc_idx]
        for (field_name, field_type) in schema:
            if field_type is float:
                v = meta_row[field_name]
                value = None if np.isnan(v) or v.tobytes() == np.float32(NUMBER_NULL).tobytes() else float(v)
            elif field_type is bytes:
                start = meta_row[field_name + "_blob_start"]
                size = meta_row[field_name + "_blob_size"]
                value = None if size == 0 else chunk_reader._blobs[start:start+size]
            else:
                v = meta_row[field_name]
                value = None if v == CATEGORY_NULL else int(v)
                # Optionally decode category
                if field_name in category_map and value is not None:
                    value = category_map[field_name].get(value, value)
            out[field_name].append(value)

    return out


# CLI / demo entry: processes all files in current directory, outputs
# to ./index, metadata = (filename, byte length).
if __name__ == "__main__":
    input_dir = Path(".")
    output_dir = "tokenized"
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in input_dir.glob("*.*") if f.is_file()])
    metadata_schema = [("name", str), ("num_bytes", float)] # , ("contents", bytes)]

    def doc_batches() -> Iterable[Tuple[List[str], List[List[Union[str, float, bytes]]]]]:
        for file in files:
            text = file.read_text(encoding="utf-8")
            info = [file.name, float(len(text.encode("utf-8")))] # , text.encode()]
            yield [text], [info]

    document_output_dir, summary_output_dir = process_documents(
        document_output_directory=output_dir,
        summary_output_directory=output_dir,
        document_batches=doc_batches(),
        metadata_schema=metadata_schema,
        fs_root="./tests",
    )
    output_dir = document_output_dir

    # Verification: iterate over all chunks, print out basic info from ChunkReader
    chunk_files = sorted(Path(output_dir).glob("chunk_*.hkmchunk"))
    print("chunk_files: ", chunk_files, flush=True)
    for chunk_file in chunk_files:
        reader = ChunkReader(str(chunk_file), metadata_schema)
        chunk_info = reader.chunk_metadata()
        print(f"Chunk: {chunk_file}")
        print(f"  Documents: {chunk_info['document_count']}")
        print(f"  Min ID: {chunk_info['min_document_id']}, Max ID: {chunk_info['max_document_id']}")
        for i in range(reader.document_count):
            tokens, embeddings, embedding_meta, meta_values = reader[i]
            print(f"    Document {i+1}:")
            print(f"      Token count: {len(tokens)}")
            print(f"      Embeddings: shape {getattr(embeddings, 'shape', None)}")
            print(f"      Metadata: {[repr(v)[:42] for v in meta_values]}")

        print()
        print("-"*100)
        print()
        import pandas as pd
        # Load category decoding map if available
        category_map_path: str = os.path.join(summary_output_dir, "category_map.json")
        category_map: Dict[str, Dict[int, str]] = {}
        with open(category_map_path, "r", encoding="ascii") as f:
            category_data = json.load(f)
        category_map = {name: {i: s for (s, i) in cat_map.items()} for (name, cat_map) in category_data.items()}
        #   get the dict form
        chunk_dict = doc_chunk_dict(reader, category_map=category_map)
        df = pd.DataFrame(chunk_dict)
        reader.close()
        print("df.columns: ", df.columns, flush=True)
        print("df.shape: ", df.shape, flush=True)
        print(df)

