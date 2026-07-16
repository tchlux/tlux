from pathlib import Path

import json

import numpy as np

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


def test_audit_rejects_populated_node_with_empty_children(tmp_path) -> None:
    index_root = tmp_path / "idx"
    hkm_root = index_root / "hkm"
    docs_root = index_root / "docs"
    child = hkm_root / "cluster_0000"
    child.mkdir(parents=True)
    docs_root.mkdir()
    np.save(docs_root / "doc_index.npy", np.empty(0, dtype=[("doc_id", "u8"), ("worker", "u4"), ("shard", "u4"), ("idx", "u4")]))
    (index_root / "index.json").write_text(json.dumps({"hkm_path": "hkm", "docs_path": "docs"}), encoding="utf-8")
    (hkm_root / "node.json").write_text(json.dumps({
        "doc_count": 1,
        "embedding_count": 1,
        "is_leaf": False,
        "children": ["cluster_0000"],
    }), encoding="utf-8")
    (child / "node.json").write_text(json.dumps({"doc_count": 0, "has_data": False, "is_leaf": True}), encoding="utf-8")
    np.save(hkm_root / "centroids.npy", np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="no populated children"):
        audit_index(str(index_root))
