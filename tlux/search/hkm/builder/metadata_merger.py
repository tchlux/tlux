# Worker that reads all metadata from shards in provided docs directory
# and writes a unified metadata file to the current directory.
# 
# Overview:
# - Scans all `worker_*` directories under `docs_root`.
# - Reads all `.meta.npy` files within each worker directory.
# - Concatenates their contents into a single array.
# - Writes the merged array to `out_path` using np.memmap.
# 
# Example usage:
#   $ python merge_metadata.py ./docs ./merged.meta.npy
# 

import argparse
import logging
from pathlib import Path

import numpy as np

from ..fs import FileSystem
from ..schema import DOC_META_DTYPE


# Description:
#   Merge all .meta.npy arrays under subdirectories of a given root
#   and save a unified metadata file using memory mapping.
#
# Parameters:
#   docs_root (str): Path to the root directory containing worker subdirectories.
#   out_path (str): Path to save the merged metadata file.
#
# Raises:
#   ValueError: If docs_root does not exist or contains no metadata files.
def merge_metadata(docs_root: str, out_path: str) -> None:
    fs = FileSystem()
    metadata_dir = docs_root + "/metadata"
    # Validate inputs early
    if not fs.exists(docs_root):
        raise ValueError(f"docs_root does not exist: {docs_root}")
    if not fs.exists(metadata_dir):
        raise ValueError(f"docs_root metadata directory does not exist: {metadata_dir}")

    all_meta = []

    # Collect all .meta.npy files within this worker directory
    for meta_file in fs.listdir(metadata_dir ):
        arr = np.load(str(meta_file), mmap_mode="r")
        all_meta.append(arr)

    # Concatenate all metadata arrays (empty array fallback)
    if all_meta:
        merged = np.concatenate(all_meta, axis=0)
    else:
        merged = np.empty((0,), dtype=DOC_META_DTYPE)

    # Write the result using memory-mapped I/O
    mm = np.memmap(out_path, dtype=DOC_META_DTYPE, shape=merged.shape, mode="w+")
    mm[:] = merged
    mm.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge all per-shard metadata into one file."
    )
    parser.add_argument("docs_root", type=str, help="Directory containing worker_* subdirectories")
    parser.add_argument("out_path", type=str, help="Path to write merged metadata file")
    args = parser.parse_args()

    merge_metadata(args.docs_root, args.out_path)
