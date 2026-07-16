"""Verify new document builds use the consolidated window defaults."""

from pathlib import Path

import numpy as np

from tlux.search.hkm.builder.chunk_io import ChunkReader
from tlux.search.hkm.builder.tokenize_and_embed import process_documents


def test_new_build_windows_have_no_sub_128_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    text = " ".join(str(value) for value in range(200))

    process_documents(
        str(tmp_path / "docs"),
        str(tmp_path / "summary"),
        [([text], [[b"doc.txt"]])],
        [("source_path", bytes)],
        fs_root=str(tmp_path),
    )

    chunk_path = next((tmp_path / "docs").rglob("*.hkmchunk"))
    reader = ChunkReader(str(chunk_path), [("source_path", bytes)])
    window_sizes = reader.embed_index["window_size"]

    assert window_sizes.size > 0
    assert np.all(window_sizes >= 128)
