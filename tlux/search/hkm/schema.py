"""Shared constants and lightweight data structures."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Configuration constants
LEAF_MAX_CHUNKS = 256_000
PREVIEW_CHUNKS = 128  # 64 random + 64 diverse
WINDOW_SIZES = (8, 32, 128, 512)
STRIDE_FACTOR = 0.5
BLOOM_FP_RATE = 0.01
KMEANS_MAX_K = 4096
DESCEND_K = 8
HEAP_FACTOR = 4
SHARD_MAX_BYTES = 8 * 2**20

# ----------------------------------------------------------------------
# Binary layout dtypes (shared by builder & loader)

DOC_META_DTYPE = np.dtype(
    [
        ("doc_id",   np.uint64),
        ("num_token_count", np.float32),  # Number of tokens in document
        ("text_off", np.uint64),
        ("text_len", np.uint32),
    ]
)

DOC_INDEX_DTYPE = np.dtype(
    [
        ("doc_id", np.uint64),
        ("worker", np.uint32),
        ("shard",  np.uint32),
        ("idx",    np.uint32),
    ]
)

# ---------------------------------------------------------------------------
# Data structures
@dataclass
class BuildConfig:
    """Configuration for ``IndexBuilder``."""

    index_root: str
    raw_paths: List[str]


@dataclass
class QuerySpec:
    """Normalized query specification.

    * ``text`` - raw UTF-8 substring to search for.  
    * ``token_sequence`` - low-level token IDs (reserved for later HKM path).  
    """
    text: str = ""
    embeddings: List = field(default_factory=list)
    token_sequence: List[int] = field(default_factory=list)
    label_include: Dict[str, List[str]] = field(default_factory=dict)
    numeric_range: Dict[str, Tuple] = field(default_factory=dict)
    top_k: int = 10


@dataclass
class Hit:
    """Single document hit."""

    doc_id: int
    score: float
    span: Tuple[int, int]


@dataclass
class SearchResult:
    """Container for search hits."""

    docs: List[Hit]
