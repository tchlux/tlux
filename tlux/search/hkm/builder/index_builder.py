"""Minimal IndexBuilder orchestrator using directory chunks."""

from __future__ import annotations

import os
from typing import List

from .tokenize_and_embed import process_documents
from .consolidate import consolidate
from .recursive_index_builder import build_cluster_index
from ..fs import FileSystem
from ..schema import BuildConfig


class IndexBuilder:
    """Sequential index builder for small datasets."""

    def __init__(self, fs: FileSystem, config: BuildConfig):
        self.fs = fs
        self.cfg = config

    def run(self) -> None:
        docs_root = self.fs.join(self.cfg.index_root, "docs")
        os.makedirs(docs_root, exist_ok=True)

        # Single worker pass over input paths
        def doc_batches():
            for path in self.cfg.raw_paths:
                text = open(path, "r", encoding="utf-8").read()
                yield [text], [[os.path.basename(path), float(len(text.encode('utf-8')))]]

        process_documents(
            document_output_directory=self.fs.join(docs_root, "worker_0000"),
            summary_output_directory=self.fs.join(self.cfg.index_root, "summary"),
            document_batches=doc_batches(),
            metadata_schema=[("name", str), ("num_bytes", float)],
            fs_root=self.fs.root,
        )

        consolidate(self.fs, self.cfg.index_root)
        build_cluster_index(self.cfg.index_root)
