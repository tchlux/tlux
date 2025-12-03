# HKM Work Summary (Dec 2025)

This file captures the current state of tlux/search/hkm after recent work, plus remaining tasks to reach a full end-to-end HKM search pipeline.

## Repository file roles & status

- **README.md** — Project design spec. Updated; now includes inline builder quick-start. Status: mostly current; expand once job-scheduler path is documented.
- **__init__.py** — Exposes FileSystem, BuildConfig, QuerySpec, SearchResult, Hit, build_search_index_inline/searcher. Status: current.

- **schema.py** — Constants and simple data classes (BuildConfig, QuerySpec, Hit, SearchResult, dtypes). Status: mostly final; may need alignment with new search path (e.g., PREVIEW counts already 1024).
- **fs.py** — Simple filesystem helper (join/mkdir/read/write/etc.). Status: stable.

### Builder
- **builder/chunk_io.py** — Core chunk writer/reader. Writes directory-based `.hkmchunk/` with tokens, tokens_index, embeddings, embed_index, metadata, blobs, per-chunk iter/unique and n-gram counters. Optional worker-level mergeable stats (categorical, numeric, iterable uniques, n-grams) via `emit_worker_stats` and `finalize_worker()`. Status: near-final; could add stricter schemas and window-aware sampling helpers.
- **builder/tokenize_and_embed.py** — Processes document batches into chunk dirs and per-worker summaries. Supports `HKM_FAKE_EMBEDDER=1` for offline tests; otherwise loads real embedder. Status: functional; minor cleanup/doc pass still useful.
- **builder/sampler.py** — Reservoir sample up to 256K embeddings; produces random/diverse previews and saves sample/previews. Currently samples one embedding per doc (first window). Status: works; window-aware sampling/metadata tracking still pending.
- **builder/consolidate.py** — Builds `doc_index.npy` under `docs/` from worker chunk dirs; renames chunk dirs to shard_*. Status: functional, simple; may expand to merge worker summaries or emit manifest.
- **builder/recursive_index_builder.py** — Sampling + k-means + previews + recursive build; supports inline mode; depth guards. Status: functional for tests; needs richer stats/previews and full window-aware sampling.
- **builder/partitioner.py** — Assigns docs to centroids, writes per-cluster chunks, emits previews/stats stub; supports balanced routing. Status: functional; still single-worker and doc-level assignment.
- **builder/launcher.py** — Orchestrator; now offers inline builder; jobs path still minimal. Status: inline path good; jobs-based path needs refresh once scheduler story is finalized.
- **builder/metadata_merger.py** — Merges per-worker categorical/numeric/ngram stats. Status: unchanged; compatible with per-worker outputs.
- **builder/trash/** — Old experiments. Status: delete/ignore.

### Search
- **search/searcher.py** — Token search + HKM traversal with embedding ranking; top-k from leaf embeddings. Status: functional; add Bloom/unique prefilters and better ranking later.
- **search/planner.py**, **search/filters.py**, **search/descent.py**, **search/ranker.py**, **search/loader.py** — Legacy helpers (planner OK; others stale). Status: update/trim as traversal matures.

### Tests
- **tests/test_chunk_io.py** — Chunk writer/reader round-trip with iterable stats. Status: passing.
- **tests/test_end_to_end_tokens.py** — End-to-end tokenize → consolidate → token search using HKM_FAKE_EMBEDDER. Status: passing.
- **tests/test_end_to_end_hkm.py** — Integration test builds inline index, checks hierarchy, token search, and HKM embedding traversal. Status: passing.
- Legacy tests removed (`test_end_to_end.py`, `test_token_searcher.py`) to avoid stale pipelines.

### CLI / bin
- **bin/hkm-index** — Still references old IndexBuilder driver. Status: broken until driver is rebuilt.
- **bin/hkm-search** — Uses legacy searcher. Status: obsolete.

## Remaining tasks to complete implementation
1) **Node stats & previews:** Enrich stats.json (label/numeric bounds, counts), ensure chunk_meta/stats for all nodes.  
2) **Partitioning & recursion:** Window-aware partitioning; per-cluster stats/previews; true multi-worker support; recurse until leaf thresholds.  
3) **Sampling fidelity:** Window-aware sampler with source indices for previews; reduce overlap.  
4) **Searcher polish:** Bloom/unique prefilters, better ranking/scoring; expose previews.  
5) **Driver/CLI (jobs path):** Refresh jobs-based orchestration, bin scripts, and manifest emission.  
6) **Docs:** Expand README/jobs docs; add guidance on HKM traversal and config knobs.  
7) **Cleanup:** Remove stale trash/legacy files after new pipeline solidifies.
