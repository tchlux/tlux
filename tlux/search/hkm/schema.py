"""Shared constants and lightweight data structures."""

from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Data structures
@dataclass
class BuildConfig:
    """Configuration for ``IndexBuilder``."""

    index_root: str
    raw_paths: List[str]


@dataclass
class QuerySpec:
    """Normalized query specification."""

    embeddings: List
    token_sequence: List[int]
    label_include: Dict[str, List[str]]
    numeric_range: Dict[str, Tuple]
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
