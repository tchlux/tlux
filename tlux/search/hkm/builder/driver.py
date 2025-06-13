"""Index building orchestrator."""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..fs import FileSystem
from ..schema import (
    BuildConfig,
    WINDOW_SIZES,
    DOC_META_DTYPE,
    DOC_INDEX_DTYPE,
    BLOOM_FP_RATE,
)
from ..embedder import embed_text, tokenize
from ..builder.bloom import BloomFilter


@dataclass
class IndexBuilder:
    """Index builder for HKM search system.

    Builds the index by storing document texts in shard files, metadata in
    shard_*.meta.npy, and a global doc_index.npy mapping. Currently stores
    one document per shard for simplicity; a full production system would
    pack multiple documents per shard up to SHARD_MAX_BYTES.
    """

    fs: FileSystem
    config: BuildConfig

    def run(self) -> None:
        """Build the index from raw documents."""
        docs_dir = self.fs.join(self.config.index_root, "docs", "worker_0000")
        hkm_dir = self.fs.join(self.config.index_root, "hkm")
        self.fs.mkdir(docs_dir, exist_ok=True)
        self.fs.mkdir(hkm_dir, exist_ok=True)
        manifest_path = self.fs.join(hkm_dir, "manifest.jsonl")

        doc_index_path = self.fs.join(self.config.index_root, "docs", "doc_index.npy")
        doc_index_records = np.memmap(
            doc_index_path,
            dtype=DOC_INDEX_DTYPE,
            mode="w+",
            shape=(len(self.config.raw_paths),)
        )

        with self.fs.open(manifest_path, "w") as mf:
            for i, path in enumerate(self.config.raw_paths):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                text_bytes = text.encode("utf-8")

                # Extract metadata
                tokens = tokenize(text)
                num_token_count = float(tokens.size)

                # ------------------------------------------------------
                # Bloom Filter (token-ids, 4-byte little-endian each)
                # ------------------------------------------------------
                bf = BloomFilter.create(capacity=tokens.size, fp_rate=BLOOM_FP_RATE)
                pack = struct.pack
                for tid in tokens:
                    bf.add(pack("<I", int(tid)))
                bloom_path = self.fs.join(hkm_dir, f"bloom_{i:08d}.bin")
                with self.fs.open(bloom_path, "wb") as bf_out:
                    bf_out.write(bf.to_bytes())

                # Store document in shard
                shard_path = self.fs.join(docs_dir, f"shard_{i:08d}.bin")
                with self.fs.open(shard_path, "wb") as out:
                    out.write(text_bytes)

                # Store metadata in shard_*.meta.npy
                meta_path = self.fs.join(docs_dir, f"shard_{i:08d}.meta.npy")
                meta = np.memmap(
                    meta_path,
                    dtype=DOC_META_DTYPE,
                    shape=(1,),
                    mode="w+",
                )
                meta[0] = (i, num_token_count, 0, len(text_bytes))
                meta.flush()

                # Add to doc_index
                doc_index_records[i] = (i, 0, i, 0)  # doc_id, worker, shard, idx

                # Generate embeddings
                embed_paths = {}
                embeddings = embed_text(text)
                for w, arr in embeddings.items():
                    epath = self.fs.join(hkm_dir, f"embed_{i:08d}_w{w}.npy")
                    # Save the memmap.
                    embs = np.memmap(
                        epath,
                        dtype=arr.dtype,
                        shape=arr.shape,
                        mode="w+",
                    )
                    embs[:] = arr[:]
                    embs.flush()
                    # Save the path.
                    embed_paths[w] = epath

                # Write manifest (minimal, metadata is in index)
                mf.write(
                    json.dumps({
                        "doc_id": i,
                        "path": shard_path,
                        "embeddings": embed_paths,
                        "bloom": bloom_path,
                    })
                    + "\n"
                )

        # Write doc_index.npy
        doc_index_records.flush()
