"""Index building orchestrator."""

import json
from dataclasses import dataclass

import numpy as np

from ..fs import FileSystem
from ..schema import BuildConfig, WINDOW_SIZES
from ..embedder import embed_text


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
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Store raw document
                shard_path = self.fs.join(docs_dir, f"shard_{i:08d}.bin")
                with self.fs.open(shard_path, "wb") as out:
                    out.write(text.encode("utf-8"))

                # Generate embeddings for all window sizes
                embed_paths = {}
                embeddings = embed_text(text)
                for w, arr in embeddings.items():
                    epath = self.fs.join(hkm_dir, f"embed_{i:08d}_w{w}.npy")
                    np.save(epath, arr)
                    embed_paths[w] = epath

                mf.write(
                    json.dumps({
                        "doc_id": i,
                        "path": shard_path,
                        "embeddings": embed_paths,
                    })
                    + "\n"
                )

