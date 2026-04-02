"""End-to-end HKM integration test using the job-managed builder."""

import json
from pathlib import Path

import numpy as np
import pytest

from tlux.search.hkm import Searcher, build_search_index, drain_jobs
from tlux.search.hkm.fs import FileSystem


def test_hkm_integration_repo_corpus(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
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
        "40 41 42 43 44 45 99 777",
    ]
    for i, text in enumerate(contents):
        (docs_src / f"doc{i}.txt").write_text(text, encoding="utf-8")

    root_job = build_search_index(
        docs_dir=str(docs_src),
        index_root=str(tmp_path),
        num_workers=2,
        max_k=2,
        leaf_doc_limit=1,
        seed=0,
    )
    drain_jobs(FileSystem(root=str(Path(tmp_path) / ".hkm_jobs")), max_workers=1)
    root_job.reload()
    assert root_job.status == "SUCCEEDED", root_job.stderr

    doc_index_path = Path(tmp_path) / "docs" / "doc_index.npy"
    assert doc_index_path.exists(), "doc_index.npy missing after consolidate"
    doc_index = np.load(doc_index_path)
    assert doc_index.shape[0] >= 4, "expected multiple documents indexed"
    assert np.all(np.diff(doc_index["doc_id"]) >= 0), "doc_index should be sorted by doc_id"

    hkm_root = fs.join(str(tmp_path), "hkm")
    searcher = Searcher.from_index_root(str(tmp_path), fs=fs)
    searcher_default_fs = Searcher.from_index_root(str(tmp_path))
    assert searcher_default_fs.hkm_root == str(Path(tmp_path) / "hkm")

    index_manifest = json.loads((Path(tmp_path) / "index.json").read_text(encoding="utf-8"))
    assert index_manifest["source_root"] == str(docs_src)
    assert index_manifest["docs_path"] == "docs"
    assert index_manifest["hkm_path"] == "hkm"
    assert index_manifest["max_n_gram"] == 3
    assert index_manifest["n_gram_fp_rate"] == pytest.approx(0.01)
    assert index_manifest["metadata_schema"][0] == ["source_path", "bytes"]

    root_node = json.loads((Path(hkm_root) / "node.json").read_text(encoding="utf-8"))
    assert "children" in root_node
    assert "preview_files" in root_node
    assert root_node["n_gram_counter_path"] == "n_gram_counter.bytes"
    assert root_node["n_gram_exists_path"] == "n_gram_exists.bytes"
    assert (Path(hkm_root) / "n_gram_counter.bytes").exists()
    assert (Path(hkm_root) / "n_gram_exists.bytes").exists()

    hits = searcher.search({"mode": "token", "text": "0", "top_k": 5})
    assert hits.docs, "token search should return at least one hit"
    assert hits.docs[0].span[0] == 0
    assert hits.docs[0].source_path == "doc0.txt"
    assert hits.docs[0].preview_text
    hits2 = searcher.search({"mode": "token", "text": "99", "top_k": 5})
    assert hits2.docs, "shared token query should return hits"
    assert all(hit.source_path.endswith(".txt") and not hit.source_path.startswith("/") for hit in hits2.docs)

    hits_emb = searcher.search({"mode": "semantic", "text": "40 41 42 43 44 45 99 777", "top_k": 3})
    assert hits_emb.docs, "embedding search should return hits"
    assert hits_emb.docs[0].source_path == "doc5.txt"
    assert hits_emb.docs[0].preview_text

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
        node_path = child / "node.json"
        assert node_path.exists(), f"node manifest missing for {child}"
        node = json.loads(node_path.read_text())
        assert "doc_count" in node
        assert (child / "n_gram_counter.bytes").exists()
        if not node.get("is_leaf", False):
            assert (child / "centroids.npy").exists()
            assert node["children"], "non-leaf should have children"
        else:
            assert (child / "n_gram_exists.bytes").exists()
        if "preview_random.npy" in node.get("preview_files", []):
            assert (child / "preview_random.npy").exists()
        if "preview_diverse.npy" in node.get("preview_files", []):
            assert (child / "preview_diverse.npy").exists()

    leaf_nodes = [path.parent for path in Path(hkm_root).rglob("node.json") if json.loads(path.read_text()).get("is_leaf", False)]
    assert leaf_nodes, "expected at least one leaf node"
    leaf_docs = searcher.leaf_docs(leaf_nodes[0])
    assert leaf_docs, "leaf browsing should list documents"
    assert leaf_docs[0].source_path
    leaf_neighbors = searcher.leaf_neighbors(leaf_nodes[0], leaf_docs[0].doc_id, top_k=3)
    assert all(hit.source_path and hit.preview_text for hit in leaf_neighbors)
    if len(leaf_docs) > 1:
        assert leaf_neighbors, "multi-doc leaf should yield neighbors"

    chunk_dirs = sorted((Path(tmp_path) / "docs").rglob("*.hkmchunk"))
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

    (docs_src / "doc5.txt").unlink()
    missing_source = searcher.search({"mode": "token", "text": "777", "top_k": 1})
    assert missing_source.docs[0].source_path == "doc5.txt"
    assert "777" in missing_source.docs[0].preview_text


def test_searcher_requires_index_manifest(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "hkm").mkdir()
    with pytest.raises(FileNotFoundError):
        Searcher.from_index_root(str(tmp_path))


def test_leaf_neighbors_use_best_passage_match(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs_src = tmp_path / "corpus"
    docs_src.mkdir()

    def _text(values) -> str:
        return " ".join(str(v) for v in values)

    (docs_src / "anchor.txt").write_text(_text(range(40)), encoding="utf-8")
    (docs_src / "good.txt").write_text(_text(list(range(32)) + list(range(1000, 1008))), encoding="utf-8")
    (docs_src / "bad.txt").write_text(_text(range(4, 44)), encoding="utf-8")

    root_job = build_search_index(
        docs_dir=str(docs_src),
        index_root=str(tmp_path),
        num_workers=1,
        max_k=2,
        leaf_doc_limit=100,
        seed=0,
    )
    drain_jobs(FileSystem(root=str(tmp_path / ".hkm_jobs")), max_workers=1)
    root_job.reload()
    assert root_job.status == "SUCCEEDED", root_job.stderr

    searcher = Searcher.from_index_root(str(tmp_path))
    leaf_root = tmp_path / "hkm"
    docs = {hit.source_path: hit for hit in searcher.leaf_docs(leaf_root)}
    hits = searcher.leaf_neighbors(leaf_root, docs["anchor.txt"].doc_id, top_k=2)
    assert hits[0].source_path == "good.txt"
    assert hits[0].score > hits[1].score
    assert hits[0].anchor_span == (0, 32)
    assert hits[0].span == (0, 32)
    assert "0 1 2 3" in hits[0].anchor_preview_text


def test_token_search_uses_hierarchical_filters(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs_src = tmp_path / "corpus"
    docs_src.mkdir()
    for i, text in enumerate(["1 2 3 4", "1 2 9 10", "20 21 22 23", "30 31 32 33"]):
        (docs_src / f"doc{i}.txt").write_text(text, encoding="utf-8")

    root_job = build_search_index(
        docs_dir=str(docs_src),
        index_root=str(tmp_path),
        num_workers=2,
        max_k=2,
        leaf_doc_limit=1,
        seed=0,
    )
    drain_jobs(FileSystem(root=str(tmp_path / ".hkm_jobs")), max_workers=1)
    root_job.reload()
    assert root_job.status == "SUCCEEDED", root_job.stderr

    searcher = Searcher.from_index_root(str(tmp_path))
    hits = searcher.search({"token_sequence": [1, 2], "top_k": 10})
    assert sorted(hit.source_path for hit in hits.docs) == ["doc0.txt", "doc1.txt"]
    assert not searcher.search({"token_sequence": [99, 100], "top_k": 10}).docs
