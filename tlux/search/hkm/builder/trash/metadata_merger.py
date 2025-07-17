
"""
Worker that reads all metadata from shards in provided docs directory
and writes a unified metadata file to the current directory.
"""

import argparse
from pathlib import Path

import numpy as np

from ..fs import FileSystem
from ..schema import DOC_META_DTYPE


def merge_metadata(docs_root: str, out_path: str):
    """
    Concatenate every .meta.npy under each worker_* in docs_root
    and write a single merged array to out_path.
    """
    fs = FileSystem()
    all_meta = []
    for worker_dir in Path(docs_root).iterdir():
        if not worker_dir.is_dir():
            continue
        for meta_file in worker_dir.glob("*.meta.npy"):
            arr = np.load(str(meta_file), mmap_mode="r")
            all_meta.append(arr)
    merged = np.concatenate(all_meta, axis=0) if all_meta else np.empty((0,), dtype=DOC_META_DTYPE)
    mm = np.memmap(out_path, dtype=DOC_META_DTYPE, shape=merged.shape, mode="w+")
    mm[:] = merged
    mm.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Worker: merge per-shard metadata into one file."
    )
    parser.add_argument("docs_root")
    parser.add_argument("out_path")
    args = parser.parse_args()

    merge_metadata(args.docs_root, args.out_path)

