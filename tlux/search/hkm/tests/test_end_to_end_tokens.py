import os
import tempfile
import json
from pathlib import Path
import pytest

from tlux.search.hkm.fs import FileSystem
from tlux.search.hkm.builder.tokenize_and_embed import process_documents
from tlux.search.hkm.builder.consolidate import consolidate
from tlux.search.hkm.search.searcher import Searcher


def test_end_to_end_token_sequence(monkeypatch):
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = FileSystem(root=tmpdir)
        docs_root = fs.join(tmpdir, "docs")
        os.makedirs(docs_root, exist_ok=True)

        texts = ["1 2 3 4", "0 2 3 5"]
        # tokens = [[1, 2, 3, 4], [0, 2, 3, 5]]

        def batches():
            for t in texts:
                yield [t], [[t, float(len(t))]]

        process_documents(
            document_output_directory=fs.join(docs_root, "worker_0000"),
            summary_output_directory=fs.join(tmpdir, "summary"),
            document_batches=batches(),
            metadata_schema=[("name", str), ("num_bytes", float)],
            chunk_size_limit=1_000_000,
            fs_root=tmpdir,
        )

        consolidate(fs, tmpdir)

        (Path(tmpdir) / "hkm").mkdir(exist_ok=True)
        (Path(tmpdir) / "index.json").write_text(json.dumps({
            "version": 1,
            "source_root": tmpdir,
            "jobs_root": os.path.join(tmpdir, ".hkm_jobs"),
            "embedder_backend": "fake",
            "metadata_schema": [["name", "str"], ["num_bytes", "float"]],
            "build_config": {},
            "docs_path": "docs",
            "hkm_path": "hkm",
        }), encoding="utf-8")
        searcher = Searcher.from_index_root(tmpdir, fs=fs)
        hits = searcher.search({"token_sequence": [2, 3], "top_k": 10})
        assert [h.doc_id for h in hits.docs] == [1, 2]
        assert hits.docs[0].span == (1, 3)
        assert hits.docs[1].span == (1, 3)
