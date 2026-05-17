"""HKM library surface."""

from .fs import FileSystem
from .jobs import Job, drain_jobs, run_job, set_jobs_root
from .schema import BuildConfig, DEFAULT_METADATA_SCHEMA, DocumentRecord, Hit, QuerySpec, SearchResult
from .builder.launcher import build_search_index
from .search.searcher import Searcher

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
    "build_search_index",
    "drain_jobs",
    "run_job",
    "set_jobs_root",
]
