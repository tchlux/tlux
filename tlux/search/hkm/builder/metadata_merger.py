"""
Merges all summary/statistics files (n-gram, categorical, numeric, category maps)
from multiple worker output subdirectories under a shared summary directory,
producing a unified set of merged summary files.

Overview:
- Finds all per-worker summary subdirectories under the given root.
- Merges all `n_gram_counter.bytes` using UniqueCounter, all
  `categorical-dist.*.bytes` and `category_map.json` into a unified
  category map and counts, and all `numeric-dist.*.bytes` using RankEstimator.
- Writes outputs in the same format to the specified output directory.
- Ensures deterministic, minimal, and clear merging. Reports contract violations.

Example usage:
    python metadata_merger.py --summary-root ./summaries --output-dir ./summary_merged

"""

import os
import sys
import json
import struct
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

try:
    from ..tools.unique_count_estimator import UniqueCounter
    from ..tools.rank_estimator import RankEstimator
except ImportError:
    from tlux.search.hkm.tools.unique_count_estimator import UniqueCounter
    from tlux.search.hkm.tools.rank_estimator import RankEstimator

# Description:
#   Merges all summary statistics files from per-worker subdirectories
#   under summary_root, and writes merged versions to output_dir.
#
# Parameters:
#   summary_root (str): Directory containing all worker summary subdirectories.
#   output_dir (str): Directory where merged summary files are written.
#
# Raises:
#   ValueError: If summary_root is missing or contains no worker directories.
#   RuntimeError: If inconsistent category mappings are detected.
#
def merge_summaries(summary_root: str, output_dir: str) -> None:
    # Validate input directories
    if not os.path.isdir(summary_root):
        raise ValueError(f"summary_root does not exist: {summary_root}")
    os.makedirs(output_dir, exist_ok=True)

    # Discover all worker summary subdirs (must be directories, not files)
    subdirs = sorted([
        d for d in (Path(summary_root).iterdir())
        if d.is_dir()
    ])
    if not subdirs:
        raise ValueError(f"No worker summary subdirectories found in: {summary_root}")

    # Find all summary files by type
    ngram_files: List[Path] = []
    category_map_files: List[Path] = []
    categorical_files: Dict[str, List[Path]] = {}
    numeric_files: Dict[str, List[Path]] = {}

    for subdir in subdirs:
        for f in subdir.glob("*"):
            if f.name == "n_gram_counter.bytes":
                ngram_files.append(f)
            elif f.name == "category_map.json":
                category_map_files.append(f)
            elif f.name.startswith("categorical-dist.") and f.name.endswith(".bytes"):
                field = f.name[len("categorical-dist."):-len(".bytes")]
                categorical_files.setdefault(field, []).append(f)
            elif f.name.startswith("numeric-dist.") and f.name.endswith(".bytes"):
                field = f.name[len("numeric-dist."):-len(".bytes")]
                numeric_files.setdefault(field, []).append(f)

    # --- Merge n_gram_counter.bytes ---
    if ngram_files:
        counter = None
        for path in ngram_files:
            uc = UniqueCounter.load_bytes(path.read_bytes())
            if counter is None:
                counter = uc
            else:
                counter.merge(uc)
        with open(os.path.join(output_dir, "n_gram_counter.bytes"), "wb") as f:
            f.write(counter.to_bytes())
    else:
        # Write empty counter if missing
        counter = UniqueCounter()
        with open(os.path.join(output_dir, "n_gram_counter.bytes"), "wb") as f:
            f.write(counter.to_bytes())

    # --- Merge category_map.json ---
    # Merges all worker maps to a unified map: {field: {str: int}}
    unified_category_map: Dict[str, Dict[str, int]] = {}
    for path in category_map_files:
        with open(path, "r", encoding="ascii") as f:
            cm = json.load(f)
        for field, str2id in cm.items():
            if field not in unified_category_map:
                unified_category_map[field] = dict(str2id)
            else:
                # Check for string->id consistency
                for s, i in str2id.items():
                    if s in unified_category_map[field]:
                        if unified_category_map[field][s] != i:
                            raise RuntimeError(f"Conflicting category mapping for {field!r} value {s!r}: {unified_category_map[field][s]} vs {i}")
                    else:
                        unified_category_map[field][s] = i
    # Write merged category_map.json
    with open(os.path.join(output_dir, "category_map.json"), "w", encoding="ascii") as f:
        json.dump(unified_category_map, f, indent=2)

    # --- Merge categorical-dist.*.bytes ---
    for field, file_list in categorical_files.items():
        cat_counts: Dict[int, int] = {}
        for path in file_list:
            with open(path, "rb") as f:
                buf = f.read()
            if len(buf) < 8:
                continue
            n_cat = struct.unpack_from("<Q", buf, 0)[0]
            offset = 8
            for _ in range(n_cat):
                cat_id = struct.unpack_from("<Q", buf, offset)[0]
                offset += 8
                count = struct.unpack_from("<Q", buf, offset)[0]
                offset += 8
                cat_counts[cat_id] = cat_counts.get(cat_id, 0) + count
        # Write merged result
        out_buf = bytearray()
        out_buf.extend(struct.pack("<Q", len(cat_counts)))
        for cat_id, count in sorted(cat_counts.items()):
            out_buf.extend(struct.pack("<Q", cat_id))
            out_buf.extend(struct.pack("<Q", count))
        with open(os.path.join(output_dir, f"categorical-dist.{field}.bytes"), "wb") as f:
            f.write(bytes(out_buf))

    # --- Merge numeric-dist.*.bytes ---
    for field, file_list in numeric_files.items():
        estimator = None
        for path in file_list:
            re = RankEstimator.load_bytes(path.read_bytes())
            if estimator is None:
                estimator = re
            else:
                estimator.merge(re)
        with open(os.path.join(output_dir, f"numeric-dist.{field}.bytes"), "wb") as f:
            f.write(estimator.to_bytes())

# CLI & demo entrypoint (no side effects on import)
if __name__ == "__main__":
    # Description:
    #   Merge summary/statistics files from all subdirectories under --summary-root,
    #   outputting merged files to --output-dir.
    #
    # Example:
    #   python metadata_merger.py --summary-root ./summaries --output-dir ./summary_merged
    #
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge all summary/statistics files from worker summary directories."
    )
    parser.add_argument(
        "--summary-root",
        type=str,
        required=True,
        help="Parent directory containing all per-worker summary subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write merged summary files"
    )
    args = parser.parse_args()

    merge_summaries(
        summary_root=args.summary_root,
        output_dir=args.output_dir,
    )
