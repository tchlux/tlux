import json
import os
from pathlib import Path

from tlux.search.hkm.builder.launcher import build_search_index_inline


def test_skip_paths_excludes_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs = tmp_path / "docs"
    docs.mkdir()
    keep = docs / "keep.txt"
    skip_dir = docs / "skipme"
    skip_dir.mkdir()
    skip_file = skip_dir / "skip.txt"
    keep.write_text("keep me", encoding="utf-8")
    skip_file.write_text("skip me", encoding="utf-8")

    index_root = tmp_path / "idx"
    index_root.mkdir()

    build_search_index_inline(
        docs_dir=str(docs),
        index_root=str(index_root),
        num_workers=1,
        max_docs=10,
        fs_root=str(index_root),
        skip_paths=[str(skip_dir)],
    )

    manifest = Path(index_root / "manifests" / "worker_0000.json")
    assert manifest.exists()
    files = json.loads(manifest.read_text(encoding="utf-8"))
    assert str(keep) in files
    assert str(skip_file) not in files
