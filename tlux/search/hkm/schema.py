"""Shared constants and lightweight data structures."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

# ---------------------------------------------------------------------------
# Configuration constants
LEAF_MAX_CHUNKS = 256_000
PREVIEW_CHUNKS = 1024  # 512 random + 512 diverse
WINDOW_SIZES = (8, 32, 128, 512)
STRIDE_FACTOR = 0.5
HASHBITMASK_FP_RATE = 0.01
KMEANS_MAX_K = 4096
DESCEND_K = 8
HEAP_FACTOR = 4
SHARD_MAX_BYTES = 8 * 2**20

DEFAULT_METADATA_SCHEMA = [
    ["source_path", "bytes"],
    ["source_type", "bytes"],
    ["file_kind", "bytes"],
    ["title", "bytes"],
    ["section_path", "bytes"],
    ["byte_start", "int"],
    ["byte_end", "int"],
    ["token_start", "int"],
    ["token_end", "int"],
    ["content_hash", "bytes"],
    ["build_id", "bytes"],
    ["ingested_at", "bytes"],
    ["source_id", "bytes"],
    ["source_url", "bytes"],
    ["source_date", "bytes"],
    ["source_token_count", "int"],
    ["num_bytes", "int"],
    ["document_preview", "bytes"],
]

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
    mode: str = "hybrid"
    embeddings: List = field(default_factory=list)
    token_sequence: List[int] = field(default_factory=list)
    text_ast: Dict[str, Any] = field(default_factory=dict)
    label_include: Dict[str, List[str]] = field(default_factory=dict)
    numeric_range: Dict[str, Tuple] = field(default_factory=dict)
    where: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 10
    offset: int = 0
    probe_count: int = 0
    filters: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DocumentRecord:
    doc_id: int = 0
    source_path: str = ""
    source_type: str = ""
    file_kind: str = ""
    title: str = ""
    section_path: str = ""
    byte_start: int = 0
    byte_end: int = 0
    token_start: int = 0
    token_end: int = 0
    content_hash: str = ""
    build_id: str = ""
    ingested_at: str = ""
    source_id: str = ""
    source_url: str = ""
    source_date: str = ""
    source_token_count: int = 0
    num_bytes: int = 0
    document_preview: str = ""


@dataclass
class Hit:
    """Single document hit."""

    doc_id: int
    score: float
    span: Tuple[int, int]
    source_path: str = ""
    preview_text: str = ""
    query_mode: str = ""
    anchor_span: Tuple[int, int] = (0, 0)
    anchor_source_path: str = ""
    anchor_preview_text: str = ""
    document: DocumentRecord = field(default_factory=DocumentRecord)
    match_reasons: List[str] = field(default_factory=list)
    semantic_score: float = 0.0
    token_score: float = 0.0
    window_size: int = 0


@dataclass
class SearchResult:
    """Container for search hits."""

    docs: List[Hit]
    offset: int = 0
    limit: int = 10
    count: int = 0
    next_offset: int | None = None
    query: Dict[str, Any] = field(default_factory=dict)
