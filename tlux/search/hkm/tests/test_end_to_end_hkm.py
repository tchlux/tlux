"""
End-to-end HKM integration test using inline builder.
"""

import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pytest

from tlux.search.hkm.fs import FileSystem
from tlux.search.hkm.builder.launcher import build_search_index_inline
from tlux.search.hkm.search.searcher import Searcher

os.environ["HKM_FAKE_EMBEDDER"] = "1"


def test_hkm_integration_repo_corpus(tmp_path: Path) -> None:
    fs = FileSystem(root=str(tmp_path))

    # create synthetic corpus with guaranteed numeric tokens including 99
    docs_src = Path(tmp_path) / "corpus"
    docs_src.mkdir()
    contents = [
        "0 1 2 3 4",
        "5 6 7 8 9 99",
        "10 11 12 13 99",
        "20 21 22 23",
        "30 31 32 33 34 35",
        "40 41 42 43 44 45 99",
    ]
    for i, text in enumerate(contents):
        (docs_src / f"doc{i}.txt").write_text(text, encoding="utf-8")

    build_search_index_inline(
        docs_dir=str(docs_src),
        index_root=str(tmp_path),
        num_workers=2,
        metadata_schema=json.dumps([
            ["path", "str"],
            ["num_bytes", "float"],
            ["file_kind", "str"],
            ["tags", "list"],
            ["attrs", "dict"],
        ]),
        max_k=2,
        leaf_doc_limit=1,
        seed=0,
        max_docs=20,
    )

    doc_index_path = Path(tmp_path) / "docs" / "doc_index.npy"
    assert doc_index_path.exists(), "doc_index.npy missing after consolidate"
    doc_index = np.load(doc_index_path)
    assert doc_index.shape[0] >= 4, "expected multiple documents indexed"
    assert np.all(np.diff(doc_index["doc_id"]) >= 0), "doc_index should be sorted by doc_id"

    docs_root = fs.join(str(tmp_path), "docs")
    hkm_root = fs.join(str(tmp_path), "hkm")
    searcher = Searcher(fs, docs_root, hkm_root)

    hits = searcher.search({"token_sequence": [0], "top_k": 5})
    assert hits.docs, "token search should return at least one hit"
    assert hits.docs[0].span[0] == 0
    hits2 = searcher.search({"token_sequence": [99], "top_k": 5})
    assert hits2.docs, "shared token query should return hits"

    # embedding search via HKM traversal
    query_emb = np.array([44.0, 7.0, 5.0, 40.0], dtype=np.float32)
    hits_emb = searcher.search({"embeddings": [query_emb.tolist()], "top_k": 3})
    assert hits_emb.docs, "embedding search should return hits"

    root_centroids = Path(hkm_root) / "centroids.npy"
    assert root_centroids.exists(), "root centroids should be saved"

    child_clusters = sorted(Path(hkm_root).glob("cluster_*"))
    assert len(child_clusters) >= 2, "expected multiple child clusters"

    grand_children = []
    for child in child_clusters:
        grand_children.extend(child.glob("cluster_*"))
    assert grand_children, "expected at least one deeper cluster layer"

    assert (Path(hkm_root) / "preview_random.npy").exists()
    assert (Path(hkm_root) / "preview_diverse.npy").exists()
    with open(Path(hkm_root) / "stats.json", "r", encoding="ascii") as f:
        root_stats = json.load(f)
    assert root_stats.get("doc_count", 0) >= doc_index.shape[0]

    for child in child_clusters:
        stats_path = child / "stats.json"
        assert stats_path.exists(), f"stats missing for {child}"
        stats = json.loads(stats_path.read_text())
        assert "doc_count" in stats
        if not stats.get("leaf", False):
            assert (child / "centroids.npy").exists()
            assert any(child.glob("cluster_*")), "non-leaf should have children"
        assert (child / "preview_random.npy").exists()
        assert (child / "preview_diverse.npy").exists()

    chunk_dirs = sorted(Path(docs_root).rglob("*.hkmchunk"))
    assert chunk_dirs, "no chunks written"
    first_chunk = chunk_dirs[0]
    assert (first_chunk / "tokens.bin").exists()
    assert (first_chunk / "tokens_index.npy").exists()
    assert (first_chunk / "embeddings.npy").exists()
    assert (first_chunk / "embed_index.npy").exists()
    assert (first_chunk / "metadata.npy").exists()
    assert (first_chunk / "n_gram_counter.bytes").exists()
    assert (first_chunk / "observer.tags.bytes").exists()
    assert (first_chunk / "unique.tags.bytes").exists()
    assert (first_chunk / "unique.attrs.bytes").exists()
