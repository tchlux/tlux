# worker_tokenize_embed.py
#
# This module processes documents by tokenizing and embedding them in a single pass,
# organizing the results into size-limited chunk files (.hkmchunk).
#
# Key features:
#   - Processes documents once: reads document_batches only once, building data in memory
#     and saving it to disk when a chunk reaches its size limit.
#   - Memory-efficient: buffers all data in RAM using ChunkBuffer, avoiding temporary
#     disk writes until a chunk is complete.
#   - Chunk contents:
#     - tokens_{i:08d}.bin: token sequences as packed 32-bit integers
#     - embeddings.mmap: float32 embeddings for all windows
#     - embed_index.mmap: array describing each embedding (doc_id, start, end, window_size)
#     - categories.mmap: int64 categorical metadata per document
#     - numbers.mmap: float64 numerical metadata per document
#   - Metadata schema: defined by the user as a list of (field_name, type) pairs,
#     where type is str or float, setting the number of categorical and numerical fields.
#   - Chunk naming: files are named chunk_<min_doc_id>_<max_doc_id>.hkmchunk based on
#     document ID range.
#   - Global outputs: category_map.json maps categorical values to integers, and
#     numeric_map.json provides percentiles for numerical values.
#   - N-gram tracking: includes a placeholder for counting unique n-grams (not implemented).
#
# Usage:
#     process_documents(
#         output_directory: str,
#         document_batches: Iterable[([texts], [infos])],
#         metadata_schema: List[Tuple[field_name, type]],
#         chunk_size_limit: int = 8 * 2**20,
#     )

import os
import argparse
import io
import json
import hashlib
import struct
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable, Tuple, Union, List, Dict

import numpy as np
try:
    from ..fs import FileSystem
    from ..embedder import tokenize, embed_windows
    from ..tools.unique_count_estimator import UniqueCounter
except ImportError:
    from tlux.search.hkm.fs import FileSystem
    from tlux.search.hkm.embedder import tokenize, embed_windows
    from tlux.search.hkm.tools.unique_count_estimator import UniqueCounter
    

# Type definitions for better readability
DocumentBatch = Iterable[Tuple[List[str], List[List[Union[str, float]]]]]
MetadataSchema = List[Tuple[str, type]]

# Format for embedding index entries: document ID, token start, token end, window size
EMBEDDING_INDEX_TYPE = np.dtype([
    ('document_id', np.uint32),
    ('token_start', np.uint32),
    ('token_end', np.uint32),
    ('window_size', np.uint32),
])


# Class to buffer document data in memory before saving it to disk.
# Manages tokens, embeddings, and metadata, writing them to a chunk file when the
# size limit is reached.
class ChunkBuffer:
    BUFFER_SIZE = 8 * 1024  # 8 KB buffer for writing data

    def __init__(self, file_system: FileSystem, output_directory: str, chunk_size_limit: int, metadata_schema: MetadataSchema):
        self.file_system = file_system
        self.output_directory = output_directory
        self.chunk_size_limit = chunk_size_limit
        self.metadata_schema = metadata_schema
        self._setup_new_buffers()
        self.document_count = 0

    def _setup_new_buffers(self):
        # Initialize tracking for document IDs and data size
        self.min_document_id = float('inf')
        self.max_document_id = -1
        self.total_bytes = 0
        # Track where token data starts and its size
        self.token_offsets: List[int] = []
        self.token_sizes: List[int] = []
        # Set up temporary files and buffers for data
        self.token_file = tempfile.TemporaryFile()
        self.token_writer = io.BufferedWriter(self.token_file, buffer_size=self.BUFFER_SIZE)
        self.embedding_file = tempfile.TemporaryFile()
        self.embedding_writer = io.BufferedWriter(self.embedding_file, buffer_size=self.BUFFER_SIZE)
        self.index_file = tempfile.TemporaryFile()
        self.index_writer = io.BufferedWriter(self.index_file, buffer_size=self.BUFFER_SIZE)
        self.category_file = tempfile.TemporaryFile()
        self.category_writer = io.BufferedWriter(self.category_file, buffer_size=self.BUFFER_SIZE)
        self.number_file = tempfile.TemporaryFile()
        self.number_writer = io.BufferedWriter(self.number_file, buffer_size=self.BUFFER_SIZE)

    def add_document(
        self,
        document_id: int,
        tokens: List[int],
        embeddings: np.ndarray,
        embedding_windows: List[Tuple[int, int, int]],
        metadata: List[Union[int, float]],
    ):
        # Save token sequence
        token_data = struct.pack(f"<{len(tokens)}I", *tokens)
        self.token_writer.flush()  # Make sure position is current
        offset = self.token_file.tell()
        self.token_offsets.append(offset)
        self.token_sizes.append(len(token_data))
        self.token_writer.write(token_data)
        self.total_bytes += len(token_data)

        # Track document ID range
        self.min_document_id = min(self.min_document_id, document_id)
        self.max_document_id = max(self.max_document_id, document_id)

        # Save embeddings and their index entries
        for i, (start, end, window_size) in enumerate(embedding_windows):
            embedding_row = embeddings[i]
            self.embedding_writer.write(embedding_row.tobytes())
            self.total_bytes += embedding_row.nbytes

            index_entry = struct.pack("<4I", document_id, start, end, window_size)
            self.index_writer.write(index_entry)
            self.total_bytes += EMBEDDING_INDEX_TYPE.itemsize

        # Split metadata into categorical and numerical parts
        category_values: List[int] = []
        number_values: List[float] = []
        for (field_name, field_type), value in zip(self.metadata_schema, metadata):
            if field_type is float:
                number_values.append(float(value))
            else:
                category_values.append(int(value))  # Already an integer from mapping

        # Save categorical data
        if category_values:
            packed_categories = struct.pack(f"<{len(category_values)}q", *category_values)
            self.category_writer.write(packed_categories)
            self.total_bytes += len(packed_categories)
        # Save numerical data
        if number_values:
            packed_numbers = struct.pack(f"<{len(number_values)}d", *number_values)
            self.number_writer.write(packed_numbers)
            self.total_bytes += len(packed_numbers)

        # Write to disk if size limit is reached
        if self.total_bytes >= self.chunk_size_limit:
            self.save_chunk()

    def save_chunk(self):
        if self.min_document_id == float('inf'):
            return  # Nothing to save

        # Ensure all data is written to temporary files
        for writer in (self.token_writer, self.embedding_writer, self.index_writer,
                       self.category_writer, self.number_writer):
            writer.flush()

        # Create a ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_STORED) as zip_file:
            # Save each document's tokens separately
            for i, (offset, size) in enumerate(zip(self.token_offsets, self.token_sizes)):
                self.token_file.seek(offset)
                data = self.token_file.read(size)
                zip_file.writestr(f"tokens_{i:08d}.bin", data)

            # Save all embeddings, index, categories, and numbers for the chunk
            for filename, temp_file in [
                ("embeddings.mmap", self.embedding_file),
                ("embed_index.mmap", self.index_file),
                ("categories.mmap", self.category_file),
                ("numbers.mmap", self.number_file),
            ]:
                temp_file.seek(0)
                data = temp_file.read()
                if data:
                    zip_file.writestr(filename, data)

        # Write the ZIP to disk
        chunk_name = f"chunk_{int(self.min_document_id):08d}_{int(self.max_document_id):08d}.hkmchunk"
        with self.file_system.open(os.path.join(self.output_directory, chunk_name), 'wb') as output_file:
            output_file.write(zip_buffer.getvalue())

        # Clean up and prepare for next chunk
        for writer in (self.token_writer, self.embedding_writer, self.index_writer,
                       self.category_writer, self.number_writer):
            writer.close()
        self._setup_new_buffers()


# Function to process documents by tokenizing, embedding, and organizing them into chunks.
# Handles everything in one pass, saving data to disk when chunks are full.
# Parameters:
# - output_directory: Where to save chunk files.
# - document_batches: Source of document texts and metadata.
# - metadata_schema: Defines metadata fields and their types (str or float).
# - chunk_size_limit: Maximum size in bytes for each chunk.
def process_documents(
    output_directory: str,
    document_batches: DocumentBatch,
    metadata_schema: MetadataSchema,
    chunk_size_limit: int = 8 * 2**20,
    n_gram: int = 3,
) -> None:
    file_system = FileSystem()
    file_system.mkdir(output_directory, exist_ok=True)

    # Track categories and their frequencies globally
    category_ids: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    # Collect numerical values for percentiles
    number_values: Dict[str, List[float]] = {name: [] for name, typ in metadata_schema if typ is float}

    chunk_buffer = ChunkBuffer(file_system, output_directory, chunk_size_limit, metadata_schema)
    document_id = 0
    ngram_counter = UniqueCounter()

    for texts, metadata_list in document_batches:
        for text, metadata in zip(texts, metadata_list):
            # Convert text to tokens
            tokens = tokenize([text])[0]
            # Add the n-grams to the counter.
            for n in range(1, n_gram + 1):
                for i in range(len(tokens) - n + 1):
                    ngram_bytes = b''.join(token.to_bytes() for token in tokens[i:i+n])
                    ngram_counter.add(ngram_bytes)
            # Create embeddings for token windows
            embeddings, embedding_windows = embed_windows([tokens])
            # Process metadata into integers (categories) and floats (numbers)
            processed_metadata = []
            for (field_name, field_type), value in zip(metadata_schema, metadata):
                if field_type is float:
                    processed_metadata.append(float(value))
                    number_values[field_name].append(value)
                else:
                    key = str((field_name, value))
                    if key not in category_ids:
                        # Create a unique integer ID for this category
                        hash_value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
                        category_ids[key] = hash_value
                        category_counts[key] = 1
                    else:
                        category_counts[key] += 1
                    processed_metadata.append(category_ids[key])
            # Add document data to buffer
            chunk_buffer.add_document(document_id, tokens, embeddings, embedding_windows, processed_metadata)
            document_id += 1

    # Save any remaining data
    chunk_buffer.save_chunk()
    # Save unique n-gram count.
    with file_system.open(file_system.join(output_directory, 'n_gram_counter.bytes'), 'w') as output_file:
        output_file.write(ngram_counter.dumps())
    # Save category mappings
    category_data = [(key, (category_ids[key], category_counts[key])) for key in category_ids]
    with file_system.open(file_system.join(output_directory, 'category_map.json'), 'w') as output_file:
        json.dump(category_data, output_file, indent=2)
    # Save numerical percentiles
    number_data = []
    for field_name, values in number_values.items():
        if values:
            percentiles = np.percentile(values, np.linspace(0, 100, 101)).tolist()
            number_data.append((field_name, percentiles))
    with file_system.open(file_system.join(output_directory, 'numeric_map.json'), 'w') as output_file:
        json.dump(number_data, output_file, indent=2)


# Default worker called by invocation of `main()`.
def default_worker(
    document_directory: str,
    output_directory: str,
    metadata_schema: str = "[('name', 'str'), ('num_bytes', 'int')]",
    worker_index: int = 0,
    total_workers: int = 1,
    chunk_size_limit: int = 8 * 2**20,
):
    # Convert metadata schema from JSON
    schema = json.loads(metadata_schema)
    metadata_schema = [(name, str if typ == 'str' else float) for name, typ in schema]

    # Find and split document files among workers
    all_files = sorted(Path(document_directory).glob('*'))
    my_files = [
        file for i, file in enumerate(all_files)
        if i % total_workers == worker_index
    ]

    # Create an iterator for document batches
    def get_document_batches():
        for file in my_files:
            text = file.read_text(encoding='utf-8')
            # Simplified: no metadata; real use would include metadata here
            yield [text], [(file.name, len(text))]

    process_documents(
        output_directory,
        get_document_batches(),
        metadata_schema,
        chunk_size_limit=chunk_size_limit,
    )


# Main function to handle command-line inputs and start processing.
def main():
    parser = argparse.ArgumentParser(
        description="Worker script to tokenize and embed documents with metadata"
    )
    parser.add_argument('document_directory', help='Folder with document files')
    parser.add_argument('output_directory', help='Folder to save chunk files')
    parser.add_argument('--metadata_schema', type=str, default='[]',
                        help='JSON list of [field_name, "str" or "float"] pairs')
    parser.add_argument('--worker_index', type=int, default=0,
                        help='Index of this worker (0-based)')
    parser.add_argument('--total_workers', type=int, default=1,
                        help='Number of workers running in parallel')
    parser.add_argument('--chunk_size_limit', type=int, default=8 * 2**20,
                        help='Maximum chunk size in bytes')
    args = parser.parse_args()
    default_worker(
        document_directory=args.document_directory,
        output_directory=args.output_directory,
        metadata_schema=args.metadata_schema,
        worker_index=args.worker_index,
        total_workers=args.total_workers,
        chunk_size_limit=args.chunk_size_limit,
    )



if __name__ == '__main__':
    main()


# Process all .txt files in the current directory and write chunks to ./tests/index.
# Metadata includes file name and text length (in bytes).
if __name__ == "__main__":
    from pathlib import Path
    import os
    from typing import List, Tuple
    input_dir = Path(".")
    output_dir = "./tests/index"
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in input_dir.glob("*.txt") if f.is_file()])
    metadata_schema = [("name", str), ("num_bytes", float)]
    def doc_batches() -> Iterable[Tuple[List[str], List[List[object]]]]:
        for file in files:
            text = file.read_text(encoding="utf-8")
            info = [file.name, float(len(text.encode("utf-8")))]
            yield [text], [info]
    process_documents(
        output_directory=output_dir,
        document_batches=doc_batches(),
        metadata_schema=metadata_schema,
    )
