from pathlib import Path

import pytest

from tlux.search.hkm import audit_index, build_search_index_from_documents


def test_audit_rejects_missing_leaf_array(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    index_root = tmp_path / "idx"
    build_search_index_from_documents(
        str(index_root),
        [{"text": "1 2 3", "metadata": {"source_path": "one.txt"}}],
        num_workers=1,
        max_k=1,
    )
    chunk = next(index_root.rglob("*.hkmchunk"))
    (chunk / "embeddings.npy").unlink()
    with pytest.raises(ValueError, match="Unreadable leaf chunk"):
        audit_index(str(index_root))
