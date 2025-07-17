"""Full system test."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tlux.search.hkm import FileSystem
from tlux.search.hkm import IndexBuilder, BuildConfig
from tlux.search.hkm import Searcher, SearchResult, Hit
from tlux.search.hkm.embedder import embed_windows, tokenize
from tlux.search.hkm.search.filters import apply_filters
from tlux.search.hkm.search.loader import DocumentStore
from tlux.search.hkm.search.planner import parse_query


class FilteredSearcher(Searcher):
    def search(self, query_dict) -> SearchResult:
        """Search with filtering using index metadata."""
        raw = super().search(query_dict)
        spec = parse_query(json.dumps(query_dict))
        if not (spec.label_include or spec.numeric_range):
            return raw

        index_root = os.path.dirname(self.docs_root)
        store = DocumentStore(self.fs, index_root)

        docs_for_filter = []
        for h in raw.docs:
            meta = store.get_metadata(h.doc_id)
            docs_for_filter.append({
                "doc_id": h.doc_id,
                "numeric": {"token_count": meta["num_token_count"]},
            })

        kept = {d["doc_id"] for d in apply_filters(docs_for_filter, spec)}
        return SearchResult(docs=[h for h in raw.docs if h.doc_id in kept])


@pytest.mark.skipif(os.getenv("SKIP_SLOW"), reason="slow test")
def test_full(tmp_path: Path) -> None:
    index_root = tmp_path / "idx"
    fs = FileSystem()
    doc_dir = tmp_path / "raw"
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir, exist_ok=True)

    # -- 1. prepare test docs ------------------------------------------
    paths = []
    for name, text in [
        ("en_short", "Hello world"),
        ("fr_short", "Bonjour le monde"),
        ("en_long",  "Hello big world, this is a doc"),
    ]:
        path = doc_dir / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))

    # -- 2. build index ------------------------------------------------
    builder = IndexBuilder(fs, BuildConfig(str(index_root), paths))
    builder.run()

    # -- 3. search -----------------------------------------------------
    docs_root = fs.join(index_root, "docs")
    hkm_root  = fs.join(index_root, "hkm")
    searcher = FilteredSearcher(fs, docs_root, hkm_root)
    hits = searcher.search({
        "text": "hello",
        "numeric_range": {
            "token_count": [4, 10]
        },
    })
    assert [h.doc_id for h in hits.docs] == [2]  # only en_long has "hello"

    store = DocumentStore(fs, str(index_root))
    text = store.get(hits.docs[0].doc_id)
    assert "Hello big world" in text

    # # -- 4. cleanup ----------------------------------------------------
    # shutil.rmtree(index_root)


if __name__ == "__main__":
    temp_dir = Path(os.path.expanduser("~/Git/tlux/tlux/search/hkm/tests/tmp"))
    os.makedirs(temp_dir, exist_ok=True)
    test_full(temp_dir)
    # shutil.rmtree(temp_dir)

    # cd .. && source .env/bin/activate && pytest tests/ --pdb
