# HKM

HKM is a Python library for building and searching a hierarchical chunk index over a shared filesystem. The library is the core product. The TUI is the default interface today, and future UIs should call the same library and job APIs.

## Current supported model

- All real work is done through the filesystem-backed job manager in [`jobs.py`](jobs.py).
- Local-machine usage is the default: enqueue jobs and let local workers execute them.
- Distributed usage uses the same job directory on a shared filesystem and workers started on every participating host.
- The current index format is the directory-based `.hkmchunk` layout written by [`builder/chunk_io.py`](builder/chunk_io.py).
- The current query surface supports hybrid text search, hierarchical token pruning, and semantic text queries, with exact token verification at leaves.
- Metadata filters, preview streaming, and richer retrieval planning are not current supported features even if older prototype code mentioned them.

## Installation

The package is intended to be pip-installable, but this directory is still in cleanup. For the current code, install the runtime dependencies listed in [`requirements.txt`](requirements.txt).

The default embedder backend is `drama`. Alternate backends can be selected with `HKM_EMBEDDER`, and the deterministic fake backend used in tests can be enabled with `HKM_FAKE_EMBEDDER=1`.

## Default interface

Run the TUI:

```bash
tlux/search/hkm/bin/hkm-tui
```

The TUI:

- initializes a job root under `<index_root>/.hkm_jobs`
- enqueues build jobs through the library
- watches local job execution
- browses the resulting HKM tree
- runs token or semantic search against a built index

## Library usage

```python
from tlux.search.hkm import Searcher, build_search_index, drain_jobs
from tlux.search.hkm.fs import FileSystem

root_job = build_search_index(
    docs_dir="data/raw_docs",
    index_root="idx",
    num_workers=4,
)

drain_jobs(FileSystem(root="idx/.hkm_jobs"), max_workers=1)
root_job.reload()

searcher = Searcher.from_index_root("idx")
hits = searcher.search({"mode": "semantic", "text": "job scheduler", "top_k": 5})
```

## CLI usage

Enqueue a build:

```bash
tlux/search/hkm/bin/hkm-index idx data/raw_docs --workers 4
```

Search an existing index:

```bash
tlux/search/hkm/bin/hkm-search idx query.json
```

The build CLI prints the root build job id. Jobs are stored under `idx/.hkm_jobs` by default.

## Current on-disk layout

```text
index_root/
  index.json
  .hkm_jobs/
  manifests/
    worker_0000.json
  docs/
    doc_index.npy
    worker_0000/
      shard_00000000.hkmchunk/
        chunk_meta.json
        tokens.bin
        tokens_index.npy
        embeddings.npy
        embed_index.npy
        metadata.npy
        n_gram_counter.bytes
        ...
  hkm/
    node.json
    n_gram_counter.bytes
    n_gram_exists.bytes
    stats.json
    centroids.npy
    preview_random.npy
    preview_diverse.npy
    cluster_0000/
      node.json
      n_gram_counter.bytes
      n_gram_exists.bytes
      data/
      stats.json
      ...
```

`.hkmchunk` directories are the canonical current storage unit. Search and build code should agree with that format exactly.

The canonical query-time manifests and token artifacts are:

- `index.json` at the root with source root, jobs root, embedder backend, metadata schema, build config, `max_n_gram`, `n_gram_fp_rate`, and relative `docs/` + `hkm/` paths
- `node.json` at each HKM node with child order, counts, preview files, token artifact paths, and whether local `data/` exists
- `n_gram_counter.bytes` at each node with the node's merged unique-count sketch
- `n_gram_exists.bytes` at each searchable node with the node's Bloom filter for token-pruned descent

Search results currently return:

- `doc_id`
- `score`
- `span`
- `source_path`
- `preview_text`
- `query_mode`
- `match_reasons`
- `semantic_score`
- `token_score`
- `document` with stable source path, type, title, byte/token spans, content
  hash, build timestamps, optional source URL metadata, and stored preview text

## Manual TUI validation

Use the repository itself as a corpus:

```bash
HKM_EMBEDDER=drama tlux/search/hkm/bin/hkm-tui
```

Set:

- docs dir: `/Users/thomaslux/Git/tlux`
- index root: `/Users/thomaslux/Git/tlux/tlux/search/hkm/tmp_repo_index`
- workers: `4`

Skip at least:

- `.git`
- `tlux/search/hkm/.env`
- existing `tmp_index*`
- `__pycache__`

Build the index, browse the tree, then switch to search mode with `/`.

Useful token queries:

- `build_search_index`
- `ChunkWriter`
- `watcher(`
- `Searcher`

Useful semantic queries:

- `job scheduler`
- `hierarchical k means`
- `token search`
- `recursive index builder`

## Public surface

The supported package-level exports are:

- `FileSystem`
- `Job`
- `QuerySpec`
- `SearchResult`
- `Hit`
- `build_search_index`
- `run_job`
- `set_jobs_root`
- `drain_jobs`
- `Searcher`

## Forward-looking architecture

The future-oriented design lives in [`ARCHITECTURE.md`](ARCHITECTURE.md). That document describes the intended scalable destination. This README describes only the current supported surface.
