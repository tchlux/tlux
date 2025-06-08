"""Index building orchestrator."""

import json
from dataclasses import dataclass

from ..fs import FileSystem
from ..schema import BuildConfig


@dataclass
class IndexBuilder:
    """Simple example index builder.

    This implementation is intentionally tiny and only stores raw
    documents.  It exists to provide a working end-to-end demo for the
    tests.  A production implementation would follow the full HKM
    specification described in the README.
    """

    fs: FileSystem
    config: BuildConfig

    def run(self) -> None:
        docs_dir = self.fs.join(self.config.index_root, "docs", "worker_0000")
        hkm_dir = self.fs.join(self.config.index_root, "hkm")
        self.fs.mkdir(docs_dir, exist_ok=True)
        self.fs.mkdir(hkm_dir, exist_ok=True)
        manifest_path = self.fs.join(hkm_dir, "manifest.jsonl")
        with self.fs.open(manifest_path, "w") as mf:
            for i, path in enumerate(self.config.raw_paths):
                with open(path, "rb") as f:
                    data = f.read()
                shard_path = self.fs.join(docs_dir, f"shard_{i:08d}.bin")
                with self.fs.open(shard_path, "wb") as out:
                    out.write(data)
                mf.write(json.dumps({"doc_id": i, "path": shard_path}) + "\n")

