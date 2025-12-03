import hashlib
import os
import tempfile

import numpy as np

from tlux.search.hkm.builder.chunk_io import ChunkReader, ChunkWriter
from tlux.search.hkm.fs import FileSystem


def _expected_hash(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(str(value).encode("ascii")).digest()[:8],
        "little",
    )


def test_chunk_writer_and_reader_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = FileSystem(root=tmpdir)
        schema = [("cat", str), ("value", float), ("tags", list)]
        writer = ChunkWriter(fs, tmpdir, chunk_size_limit=1_000_000, metadata_schema=schema)

        tokens_a = [10, 11, 12, 13]
        embeds_a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        windows_a = [(0, 2, 2), (2, 4, 2)]
        cat_a = _expected_hash("blue")
        writer.add_document(1, tokens_a, embeds_a, windows_a, [cat_a, 1.5, ["blue", "green"]])

        tokens_b = [5, 6]
        embeds_b = np.array([[1.0, 1.1]], dtype=np.float32)
        windows_b = [(0, 2, 2)]
        cat_b = _expected_hash("red")
        writer.add_document(2, tokens_b, embeds_b, windows_b, [cat_b, 2.5, ["red"]])

        writer.save_chunk()

        chunk_dirs = [d for d in os.listdir(tmpdir) if d.endswith(".hkmchunk")]
        assert len(chunk_dirs) == 1
        chunk_path = os.path.join(tmpdir, chunk_dirs[0])

        expected_files = {
            "tokens.bin",
            "tokens_index.npy",
            "embeddings.npy",
            "embed_index.npy",
            "metadata.npy",
            "chunk_meta.json",
            "observer.tags.bytes",
            "unique.tags.bytes",
        }
        assert expected_files.issubset(set(os.listdir(chunk_path)))

        reader = ChunkReader(chunk_path, schema)
        assert reader.document_count == 2

        tokens0, emb0, meta0, meta_vals0 = reader[0]
        assert np.array_equal(tokens0, np.array(tokens_a, dtype=np.uint32))
        assert emb0.shape == (2, 2)
        assert np.allclose(emb0, embeds_a)
        assert meta0["document_id"].tolist() == [1, 1]
        assert meta_vals0[0] == cat_a
        assert meta_vals0[1] == 1.5

        tokens1, emb1, meta1, meta_vals1 = reader[1]
        assert np.array_equal(tokens1, np.array(tokens_b, dtype=np.uint32))
        assert emb1.shape == (1, 2)
        assert np.allclose(emb1, embeds_b)
        assert meta1["document_id"].tolist() == [2]
        assert meta_vals1[0] == cat_b
        assert meta_vals1[1] == 2.5

        arrays = reader.arrays()
        assert len(arrays["tokens"]) == 2
        assert arrays["embeddings"].shape[0] == 3


if __name__ == "__main__":
    test_chunk_writer_and_reader_roundtrip()
