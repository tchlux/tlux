import os
import tempfile

from tlux.search.hkm.hkm_search import FileSystem, BuildConfig, IndexBuilder, Searcher


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        doc1 = os.path.join(tmp, "d1.txt")
        doc2 = os.path.join(tmp, "d2.txt")
        with open(doc1, "w") as f:
            f.write("hello world")
        with open(doc2, "w") as f:
            f.write("second document")

        index_root = os.path.join(tmp, "index")
        fs = FileSystem()
        cfg = BuildConfig(index_root=index_root, raw_paths=[doc1, doc2])
        IndexBuilder(fs, cfg).run()

        searcher = Searcher(fs, fs.join(index_root, "docs"), fs.join(index_root, "hkm"))
        hits = searcher.search({"token_sequence": [ord(c) for c in "hello"], "top_k": 10})
        assert len(hits.docs) == 1
        assert hits.docs[0].doc_id == 0
