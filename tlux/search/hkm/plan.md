# HKM Work Summary (Dec 2025)

This file captures the current state of tlux/search/hkm after recent work, plus remaining tasks to reach a full end-to-end HKM search pipeline.

## Repository file roles & status

- **README.md** — Project design spec. Updated to describe directory-based `.hkmchunk/` chunks. Status: needs refresh to reflect new sampler/partitioner/searcher behavior and removal of legacy ZIP.
- **__init__.py** — Exposes core types (FileSystem, BuildConfig, QuerySpec, SearchResult, Hit). Status: minimal; IndexBuilder not exported; needs revisit once driver stabilizes.

- **schema.py** — Constants and simple data classes (BuildConfig, QuerySpec, Hit, SearchResult, dtypes). Status: mostly final; may need alignment with new search path (e.g., PREVIEW counts already 1024).
- **fs.py** — Simple filesystem helper (join/mkdir/read/write/etc.). Status: stable.

### Builder
- **builder/chunk_io.py** — Core chunk writer/reader. Writes directory-based `.hkmchunk/` with tokens, tokens_index, embeddings, embed_index, metadata, blobs, per-chunk iter/unique and n-gram counters. Optional worker-level mergeable stats (categorical, numeric, iterable uniques, n-grams) via `emit_worker_stats` and `finalize_worker()`. Status: near-final; could add stricter schemas and window-aware sampling helpers.
- **builder/tokenize_and_embed.py** — Processes document batches into chunk dirs and per-worker summaries. Supports `HKM_FAKE_EMBEDDER=1` for offline tests; otherwise loads real embedder. Status: functional; needs docstring cleanup and alignment with final driver.
- **builder/sampler.py** — Reservoir sample up to 256K embeddings; produces random/diverse previews and saves sample/previews. Currently samples one embedding per doc (first window). Status: works; needs window-aware sampling and metadata tracking.
- **builder/consolidate.py** — Builds `doc_index.npy` under `docs/` from worker chunk dirs; renames chunk dirs to shard_*. Status: functional, simple; may expand to merge worker summaries or emit manifest.
- **builder/recursive_index_builder.py** — Root sampler + k-means + preview save; leaf marker when cluster_limit <= 1; spawns partitioner then recurses. Status: partial; recursion lacks leaf stats/previews and child sample generation; only root-level meaningful today.
- **builder/partitioner.py** — Assigns docs to centroids (first embedding), writes per-cluster chunk dirs via ChunkWriter, emits per-cluster previews and stats stub. Status: partial; single worker, doc-level assignment, no deeper leaf handling, stats minimal.
- **builder/launcher.py** — Orchestrator placeholder; still references consolidate + recursive builder jobs. Status: stale; needs update once driver solidifies.
- **builder/metadata_merger.py** — Merges per-worker categorical/numeric/ngram stats. Status: unchanged; compatible with per-worker outputs.
- **builder/trash/** — Old experiments. Status: delete/ignore.

### Search
- **search/searcher.py** — New token-sequence searcher that scans chunk tokens using doc_index and ChunkReader; ignores HKM tree. Status: works for token subsequence tests; lacks HKM traversal, Bloom/unique prefilters, ranking.
- **search/planner.py**, **search/filters.py**, **search/descent.py**, **search/ranker.py**, **search/loader.py** — Legacy helpers (planner/filters OK; descent assumes HKM centroids; loader expects text shards). Status: need revisit/rewrite to match new storage and search path.

### Tests
- **tests/test_chunk_io.py** — Chunk writer/reader round-trip with iterable stats. Status: passing.
- **tests/test_end_to_end_tokens.py** — End-to-end tokenize → consolidate → token search using HKM_FAKE_EMBEDDER. Status: passing (set env inside test now).
- Legacy tests removed (`test_end_to_end.py`, `test_token_searcher.py`) to avoid stale pipelines.

### CLI / bin
- **bin/hkm-index** — Still references old IndexBuilder driver. Status: broken until driver is rebuilt.
- **bin/hkm-search** — Uses legacy searcher. Status: obsolete.

## Remaining tasks to complete implementation
1) **Node stats & previews:** Persist stats.json (label/numeric bounds, counts) and previews at every HKM node; ensure chunk_meta for internal nodes.
2) **Partitioning & recursion:** Make partitioner window-aware, support multiple workers, and write full per-cluster stats/previews; recursive builder should sample per cluster, recurse until leaf (<= LEAF_MAX_CHUNKS), and emit leaf chunk_meta/stats.
3) **Sampling fidelity:** Improve sampler to draw windows (not just first embedding per doc) with minimal doc overlap; track source indices for previews.
4) **Searcher rewrite:** Add HKM traversal (using centroids/stats), Bloom/unique prefilters, and top-k ranking; optionally surface previews.
5) **Driver/CLI:** Reintroduce IndexBuilder/launcher wired to new pipeline (tokenize → consolidate → sample/cluster → partition → recurse), export via __init__, fix bin scripts.
6) **Integration test:** Add a small HKM integration test exercising the full build + HKM traversal search path (with fake embedder).
7) **Docs:** Refresh README to reflect the new chunk format, pipeline steps, and search path; document env var HKM_FAKE_EMBEDDER for offline runs.
8) **Cleanup:** Remove stale trash/legacy files once new pipeline is stable.

