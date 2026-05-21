"""HKM library surface."""

from .fs import FileSystem
from .jobs import Job, drain_jobs, run_job, set_jobs_root
from .schema import BuildConfig, DEFAULT_METADATA_SCHEMA, DocumentRecord, Hit, QuerySpec, SearchResult
from .builder.launcher import build_search_index, build_search_index_from_documents
from .search.searcher import Searcher, audit_index, open_index, resolve_index_root

__all__ = [
    "BuildConfig",
    "DEFAULT_METADATA_SCHEMA",
    "DocumentRecord",
    "FileSystem",
    "Hit",
    "Job",
    "QuerySpec",
    "SearchResult",
    "Searcher",
    "audit_index",
    "build_search_index",
    "build_search_index_from_documents",
    "drain_jobs",
    "open_index",
    "resolve_index_root",
    "run_job",
    "set_jobs_root",
]
