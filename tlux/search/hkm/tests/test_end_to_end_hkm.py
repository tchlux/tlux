"""End-to-end HKM integration test using the job-managed builder."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tlux.search.hkm import Searcher, build_search_index, drain_jobs, open_index, resolve_index_root
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
        leaf_embedding_limit=1,
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
    assert index_manifest["build_config"]["leaf_embedding_limit"] == 1
    assert index_manifest["build_config"]["leaf_doc_limit"] == 1

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
    metadata_dtype = np.load(first_chunk / "metadata.npy", mmap_mode="r").dtype
    assert "content_hash_blob_start" in metadata_dtype.names
    assert "document_preview_blob_start" in metadata_dtype.names

    (docs_src / "doc5.txt").unlink()
    missing_source = searcher.search({"mode": "token", "text": "777", "top_k": 1})
    assert missing_source.docs[0].source_path == "doc5.txt"
    assert "777" in missing_source.docs[0].preview_text


def test_searcher_requires_index_manifest(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "hkm").mkdir()
    with pytest.raises(FileNotFoundError):
        Searcher.from_index_root(str(tmp_path))


def test_fineweb_manifest_enriches_document_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    corpus = tmp_path / "fineweb"
    docs_src = corpus / "docs"
    docs_src.mkdir(parents=True)
    raw = b"1 2 3 4 5\n"
    doc_path = docs_src / "sample.txt"
    doc_path.write_bytes(raw)
    (corpus / "manifest.jsonl").write_text(json.dumps({
        "file": "docs/sample.txt",
        "id": "doc-123",
        "url": "https://example.com/sample",
        "date": "2020-01-02T03:04:05Z",
        "token_count": 99,
    }) + "\n", encoding="utf-8")

    index_root = tmp_path / "idx"
    root_job = build_search_index(
        docs_dir=str(docs_src),
        index_root=str(index_root),
        num_workers=1,
        fs_root=str(index_root),
    )
    drain_jobs(FileSystem(root=str(index_root / ".hkm_jobs")), max_workers=1)
    root_job.reload()
    assert root_job.status == "SUCCEEDED", root_job.stderr

    hit = Searcher.from_index_root(str(index_root)).search({"mode": "token", "text": "3", "top_k": 1}).docs[0]
    assert hit.document.source_path == "sample.txt"
    assert hit.document.source_type == "web"
    assert hit.document.source_id == "doc-123"
    assert hit.document.source_url == "https://example.com/sample"
    assert hit.document.source_date == "2020-01-02T03:04:05Z"
    assert hit.document.source_token_count == 99
    assert hit.document.content_hash == hashlib.sha256(raw).hexdigest()
    assert hit.document.byte_start == 0
    assert hit.document.byte_end == len(raw)
    assert hit.document.token_start == 0
    assert hit.document.token_end == 5
    assert hit.document.num_bytes == len(raw)

    doc_path.unlink()
    missing_source_hit = Searcher.from_index_root(str(index_root)).search({"mode": "token", "text": "4", "top_k": 1}).docs[0]
    assert "4" in missing_source_hit.preview_text


def test_hybrid_search_ranks_metadata_and_explains_matches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs_src = tmp_path / "corpus"
    docs_src.mkdir()
    (docs_src / "alpha_report.txt").write_text("90 91 92 Alpha launch notes and exact preview text", encoding="utf-8")
    (docs_src / "near_numbers.txt").write_text("0 1 2 3 semantic only baseline", encoding="utf-8")
    (docs_src / "other.txt").write_text("50 51 52 unrelated", encoding="utf-8")

    index_root = tmp_path / "idx"
    root_job = build_search_index(
        docs_dir=str(docs_src),
        index_root=str(index_root),
        num_workers=1,
        max_k=2,
        leaf_doc_limit=100,
        fs_root=str(index_root),
        seed=0,
    )
    drain_jobs(FileSystem(root=str(index_root / ".hkm_jobs")), max_workers=1)
    root_job.reload()
    assert root_job.status == "SUCCEEDED", root_job.stderr

    searcher = Searcher.from_index_root(str(index_root))
    default_hits = searcher.search({"text": "alpha", "top_k": 3}).docs
    assert default_hits[0].query_mode == "hybrid"
    assert default_hits[0].source_path == "alpha_report.txt"
    assert {"semantic", "path", "title", "preview"} <= set(default_hits[0].match_reasons)
    assert "Alpha" in default_hits[0].preview_text
    assert default_hits[0].semantic_score > 0.0

    token_hits = searcher.search({"mode": "token", "text": "90 91", "top_k": 2}).docs
    semantic_hits = searcher.search({"mode": "semantic", "text": "0 1 2 3", "top_k": 2}).docs
    hybrid_token_hits = searcher.search({"text": "90 91", "top_k": 2}).docs
    assert token_hits[0].query_mode == "token"
    assert semantic_hits[0].query_mode == "semantic"
    assert hybrid_token_hits[0].token_score > 0.0
    assert "token" in hybrid_token_hits[0].match_reasons

    duplicate = searcher._hit(default_hits[0].doc_id, 1.0, default_hits[0].span, "token", "alpha")
    grouped = {default_hits[0].source_path: default_hits[0]}
    searcher._merge_hybrid_hit(grouped, duplicate)
    assert len(grouped) == 1
    assert {"token", "semantic", "path", "title", "preview"} <= set(grouped[default_hits[0].source_path].match_reasons)

    page = searcher.search({"text": "alpha", "top_k": 1})
    assert page.offset == 0
    assert page.limit == 1
    assert page.count >= 1
    assert page.query["mode"] == "hybrid"
    assert page.docs[0].source_path == "alpha_report.txt"
    if page.next_offset is not None:
        next_page = searcher.search({"text": "alpha", "top_k": 1, "offset": page.next_offset})
        assert next_page.offset == page.next_offset
        assert next_page.docs[0].source_path != page.docs[0].source_path

    filtered = searcher.search({
        "text": "alpha",
        "top_k": 3,
        "filters": {"path_include": ["alpha_*"], "file_kind": ["txt"]},
    })
    assert [hit.source_path for hit in filtered.docs] == ["alpha_report.txt"]
    assert not searcher.search({
        "text": "alpha",
        "top_k": 3,
        "filters": {"path_exclude": ["alpha_*"], "file_kind": [".txt"]},
    }).docs[0].source_path == "alpha_report.txt"

    for query in (
        {"text": "alpha", "mode": "bad"},
        {"text": "alpha", "top_k": 0},
        {"text": "alpha", "offset": -1},
        {"text": "alpha", "filters": {"path_include": "alpha_*"}},
        {"text": ""},
    ):
        with pytest.raises((TypeError, ValueError)):
            searcher.search(query)

    query_path = tmp_path / "query.json"
    query_path.write_text(json.dumps({
        "text": "alpha",
        "top_k": 1,
        "filters": {"path_include": ["alpha_*"]},
    }), encoding="utf-8")
    command = [str(Path(__file__).resolve().parents[1] / "bin" / "hkm-search"), str(index_root), str(query_path)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert set(payload) == {"docs", "offset", "limit", "count", "next_offset", "query"}
    assert payload["docs"][0]["source_path"] == "alpha_report.txt"
    assert payload["docs"][0]["document"]["file_kind"] == ".txt"

    opened = open_index(str(index_root))
    assert opened.index_root == str(index_root.resolve())

    command = [
        str(Path(__file__).resolve().parents[1] / "bin" / "hkm-search"),
        str(index_root),
        "--text",
        "alpha",
        "--top-k",
        "1",
        "--path-include",
        "alpha_*",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert set(payload) == {"docs", "offset", "limit", "count", "next_offset", "query"}
    assert payload["docs"][0]["source_path"] == "alpha_report.txt"

    command = [str(Path(__file__).resolve().parents[1] / "bin" / "hkm-search"), str(index_root), "--node", "hkm"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["path"] == "hkm"
    assert payload["node"]["is_leaf"]

    command = [str(Path(__file__).resolve().parents[1] / "bin" / "hkm-search"), str(index_root), "--docs", "hkm"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["node"] == "hkm"
    assert "alpha_report.txt" in {hit["source_path"] for hit in payload["docs"]}

    command = [
        str(Path(__file__).resolve().parents[1] / "bin" / "hkm-search"),
        str(index_root),
        "--neighbors",
        "hkm",
        "--doc-id",
        str(default_hits[0].doc_id),
        "--top-k",
        "2",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["node"] == "hkm"
    assert payload["doc_id"] == default_hits[0].doc_id
    assert payload["docs"]


def test_open_index_reports_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing canonical index manifest"):
        open_index(str(tmp_path))


def test_resolve_index_root_accepts_parent_or_hkm_dir(tmp_path: Path) -> None:
    index_root = tmp_path / "idx"
    (index_root / "hkm").mkdir(parents=True)
    (index_root / "index.json").write_text(json.dumps({
        "source_root": str(tmp_path),
        "metadata_schema": [],
        "hkm_path": "hkm",
        "docs_path": "docs",
    }), encoding="utf-8")
    (index_root / "hkm" / "node.json").write_text(json.dumps({
        "is_leaf": True,
        "children": [],
    }), encoding="utf-8")

    assert resolve_index_root(str(tmp_path)) == index_root.resolve()
    assert open_index(str(index_root / "hkm")).index_root == str(index_root.resolve())


def test_open_index_reports_missing_child_node(tmp_path: Path) -> None:
    (tmp_path / "hkm").mkdir()
    (tmp_path / "index.json").write_text(json.dumps({
        "source_root": str(tmp_path),
        "metadata_schema": [],
        "hkm_path": "hkm",
        "docs_path": "docs",
    }), encoding="utf-8")
    (tmp_path / "hkm" / "node.json").write_text(json.dumps({
        "is_leaf": False,
        "children": ["cluster_0000"],
    }), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing child node manifest"):
        open_index(str(tmp_path))


def test_content_hash_is_stable_when_doc_id_changes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs_src = tmp_path / "corpus"
    docs_src.mkdir()
    target = docs_src / "target.txt"
    target.write_text("10 11 12", encoding="utf-8")

    first_root = tmp_path / "idx1"
    first_job = build_search_index(str(docs_src), str(first_root), 1, fs_root=str(first_root))
    drain_jobs(FileSystem(root=str(first_root / ".hkm_jobs")), max_workers=1)
    first_job.reload()
    assert first_job.status == "SUCCEEDED", first_job.stderr
    first_hit = Searcher.from_index_root(str(first_root)).search({"mode": "token", "text": "12", "top_k": 1}).docs[0]

    (docs_src / "aaa.txt").write_text("1 2 3 4 5 6 7 8 9", encoding="utf-8")
    second_root = tmp_path / "idx2"
    second_job = build_search_index(str(docs_src), str(second_root), 1, fs_root=str(second_root))
    drain_jobs(FileSystem(root=str(second_root / ".hkm_jobs")), max_workers=1)
    second_job.reload()
    assert second_job.status == "SUCCEEDED", second_job.stderr
    second_hits = Searcher.from_index_root(str(second_root)).search({"mode": "token", "text": "12", "top_k": 3}).docs
    second_hit = next(hit for hit in second_hits if hit.source_path == "target.txt")

    assert first_hit.doc_id != second_hit.doc_id
    assert first_hit.document.content_hash == second_hit.document.content_hash


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
        leaf_embedding_limit=1,
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


def test_leaf_split_uses_embedding_count_not_doc_count(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    docs_src = tmp_path / "corpus"
    docs_src.mkdir()
    for i, start in enumerate((0, 1000)):
        (docs_src / f"doc{i}.txt").write_text(" ".join(str(v) for v in range(start, start + 80)), encoding="utf-8")

    root_job = build_search_index(
        docs_dir=str(docs_src),
        index_root=str(tmp_path),
        num_workers=1,
        max_k=2,
        leaf_embedding_limit=4,
        leaf_doc_limit=100,
        seed=0,
    )
    drain_jobs(FileSystem(root=str(tmp_path / ".hkm_jobs")), max_workers=1)
    root_job.reload()
    assert root_job.status == "SUCCEEDED", root_job.stderr

    root_node = json.loads((tmp_path / "hkm" / "node.json").read_text(encoding="utf-8"))
    assert root_node["doc_count"] == 2
    assert root_node["embedding_count"] == 8
    assert not root_node["is_leaf"]
    assert root_node["children"]
