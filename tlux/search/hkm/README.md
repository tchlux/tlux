```

===============================================================================
hkm - Hierarchical-K-Means Large-Scale Search Toolkit
===============================================================================

Author : Thomas C.H. Lux                  Version : 0.1-spec  
Date   : 2025-06-07                       License : MIT (proposed)

-------------------------------------------------------------------------------
TABLE OF CONTENTS
-------------------------------------------------------------------------------
  1.  Problem Statement and Hard Requirements
  2.  Design Rationale and Key Assumptions
  3.  High-Level Architecture
  4.  Data Model and On-Disk Layout
  5.  Build-Time Workflow
  6.  Query-Time Workflow
  7.  Public Python API
  8.  Package File Map
  9.  Configuration Constants
 10.  Performance Targets
 11.  Future Work and Open Questions

===============================================================================
1.  PROBLEM STATEMENT AND HARD REQUIREMENTS
===============================================================================

We must index **up to 100 B documents** (< 1 MB UTF-8 each) and support
*composable* search modes:

  *  Embedding nearest-neighbour (1024-D float32, L2 / cosine / inner-product)  
  *  Exact token-sequence match (caller supplies the tokenizer)  
  *  Category filters (<= 100 distinct labels per metadata column)  
  *  Numeric range filters (float32 / int32; arbitrary distributions)  
  *  Any combination above in a single query

Hard constraints

  *  Pure CPython 3.x, **only** external dependency is NumPy  
  *  Works on CPU or GPU, but cheap enough that CPU is realistic
  *  RAM per process 4 - 32 GB  
  *  Storage is a remote file/object store with *whole-file reads only*  
     (no range get). Atomic `mkdir` is available.  
  *  Preferred block sizes: 8 MB or 64 MB; bandwidth ~= 64 MB s-^1  
  *  Latency goal: < 1 s p95; first preview ~= 100 ms  
  *  Index is rebuilt offline; no live mutations required  
  *  Sliding-window embeddings - window sizes {8, 32, 128, 512} tokens,  
     stride = window / 2, ~= 100 windows per doc  
  *  Code must be portable (no AWS, Docker, POSIX tricks)  
  *  Deliverable is an installable open-source PyPI package

===============================================================================
2.  DESIGN RATIONALE AND KEY ASSUMPTIONS
===============================================================================

To meet latency and RAM targets with pure Python, we adopt a **recursive
k-means (HKM) tree**:

  *  Sample 256 k embeddings -> <= 4096-way k-means -> assign -> recurse  
     until a leaf <= 256 k chunks.  
  *  A query visits O(log N) nodes and touches << 1 % of embeddings.  
  *  All binary files are bounded to 8-64 MB

**Instant feedback requirement**

  *  Each HKM node stores **preview chunks from 1024 unique documents**  
       - 512 randomly sampled (representative)  
       - 512 chosen by greedy farthest-point (diverse)
       - all sliding window embeddings for each chunk
  *  Representative previews stream first (< 120 ms), then diverse.

**Strict filters**

  *  Each node holds label counts and numeric distributions.  
  *  Planner computes an upper-bound candidate count *before* descent;  
     if bound < top-k, it may relax filters or fail fast.

===============================================================================
3.  HIGH-LEVEL ARCHITECTURE
===============================================================================

            user / API
                |
                v

             Searcher
    planner - retriever - ranker
         heavy use of mmaps
                |
                v

        HKM tree (embeddings)
       token store (postings)

                ^
                |
             build jobs
                |
           IndexBuilder          


Both build and search work with **whole-file** reads/writes only.

===============================================================================
4.  DATA MODEL AND FILE SYSTEM LAYOUT
===============================================================================

Directory structure (fixed):

    index/                    # Nearest Cluster Centroid hierarchy
      config.json             # Search index config, column names, size, ...
      summary/
        centroids.npy         # (<= 4096 x 1024 float32)
        preview_random.npy    # (512 x uint64 chunk_id, 512-token doc preview, embedding)
        preview_diverse.npy   # (512 x uint64 chunk_id, 512-token doc preview, embedding)
        category_map.json     # maps categorical values to hashed integers
        categorical-dist.<name>.bytes # per-field categorical frequency tables
        numeric-dist.<name>.bytes     # per-field numeric value distributions
        n_gram_counter.bytes  # total unique n-gram counter
        hashbitmask.bin       # [optional] token n-gram presence hash bit mask for all docs in this node
        part_0000/            # summary for a part of the data (before having been merged above)
          category_map.json # maps categorical values to hashed integers
          categorical-dist.<name>.bytes  # per-field categorical frequency tables
          numeric-dist.<name>.bytes      # per-field numeric value distributions
          n_gram_counter.bytes # total unique n-gram counter
        part_0001/
          ...
      data/                   # randomized data broken up by number of workers
        part_0000/
          chunk_00000000_00000042.hkmchunk/  # directory: tokens.bin, tokens_index.npy, embeddings.npy, embed_index.npy, metadata.npy
          chunk_00000043_00000099.hkmchunk/
          ...
        ...
      cluster_0000/     # recursively structured subdirectories
        summary/
          ...
        data/
          ...
        cluster_0000/
          summary/
            ...
          data/
            ...
        ...
      ...

Binary format: directory-based `.hkmchunk` with NumPy v2 arrays and raw binaries.

Structured dtypes:

  TO DO (based on code in `/build`)

All per-node files are <= 8 MB.

===============================================================================
5.  BUILD-TIME WORKFLOW
===============================================================================

PHASE A - Producer  
  *  Scans raw corpus, assigns `doc_id`, writes build_job_N.json
     (<= 32 k docs each).

PHASE B - Shard Worker (parallel)  
  *  Claims a job via `mkdir`.  
  *  Tokenises docs; generates sliding-window embeddings.  
  *  Appends tokens + metadata to 8 MB `docs/worker_*/shard_*.bin`.  
  *  Appends embeddings to matching `embeddings/shard_*.npy` in **root node**.

PHASE C - HKM Builder (parallel by subtree)  
  1.  For a subtree: sample 256 k embeddings -> <= 4096-way k-means.  
  2.  Assign embeddings, write `centroids.npy`, `stats.json`, `hashbitmask.bin`.  
  3.  Choose 512 random + 512 farthest-point previews, save.  
  4.  Copy subtree embeddings into `embeddings/shard_*.npy` (<= 8 MB each).  
  5.  Recurse until leaf <= 256 k chunks, then write `embed.npy`,
      `chunk_meta.npy`.  
  6.  Append node path to `manifest.jsonl`.

Quick-start (local, single process):  
  *  ``build_search_index_inline(docs_dir="data/raw", index_root="idx", num_workers=8, max_k=8, leaf_doc_limit=1024)``  
    - bin-packs files by byte size across workers  
    - tokenizes/embeds -> consolidates -> builds HKM inline (no external scheduler)

Jobs-based builds:  
  *  Use `build_search_index` to enqueue tokenize/embed -> consolidate -> HKM build via the filesystem-backed `jobs/` queue.  
  *  For purely serial runs, `spawn_job(..., inline=True)` executes job functions in-process while keeping the same interface.

Configuration knobs (common):  
  * `max_k`: max clusters per level (default 8)  
  * `leaf_doc_limit`: max docs before stopping recursion (default 1024)  
  * `num_workers`: tokenize/embed workers (bin-packed by bytes)  
  * `max_depth`: recursion limit (default 3 in inline builder)  
  * `chunk_size_limit`: bytes per chunk (default 8 MiB)  
  * `seed`: RNG seed for sampling/k-means  

All writes are "write-temp -> rename" for crash safety.

===============================================================================
6.  QUERY-TIME WORKFLOW
===============================================================================

 0  Parse request JSON -> QuerySpec.  
 1  **Upper-bound test** using `labels_count` and `numeric_hist`.  
 2  **HKM descent**: at each level, keep 8 closest children per embedding,
    prune by metadata.  
 3  **Stream previews** when depth >= 2 node accepted  
       - first `preview_random.npy`, then `preview_diverse.npy`.  
 4  **Leaf ranking**: load `embed.npy`, compare query vectors to matching
    `win_size`, keep heap of top-k chunks.  
 5  **Exact token check** (optional): use `hashbitmask.bin`; load full text only
    if hash bit mask says "possible".  
 6  Aggregate best chunk per document -> return top-k docs.

===============================================================================
7.  PUBLIC PYTHON API (HIGH-LEVEL)
===============================================================================

    from tlux.search.hkm import FileSystem, build_search_index_inline, Searcher

    # Build (one call, local)
    fs   = FileSystem()
    build_search_index_inline(
        docs_dir="data/raw_docs",
        index_root="idx",
        num_workers=8,
    )

    # Search (token + embedding)
    searcher = Searcher(fs, "idx/docs", "idx/hkm")
    hits     = searcher.search({
                 "embeddings"     : [[...]],  # list of vectors
                 "token_sequence" : [101, 202, 303],
                 "label_include"  : {"lang": ["en"]},
                 "numeric_range"  : {"year": [2010, 2022]},
                 "top_k"          : 100
               })
    for h in hits.docs:
        print(h.doc_id, h.score, h.span)

Heavy arrays are memory-mapped internally; API returns lightweight
namedtuples and lists.

===============================================================================
8.  PACKAGE FILE MAP
===============================================================================

    hkm/
      __init__.py          (re-exports FS, IndexBuilder, Searcher)
      fs.py                (FileSystem base + LocalFS impl)
      schema.py            (dtypes + constants)
      utils/
        array.py           (mmap helpers)
        bitset.py          (<= 100-bit masks)
      builder/
        driver.py          (IndexBuilder orchestrator)
        sampler.py         (reservoir & k-means++ samplers)
        kmeans.py          (<= 4096-way Lloyd in NumPy)
        stats.py           (label & numeric aggregations)
        hashbitmask.py     (bit-array hash bit mask)
        preview.py         (random + farthest-point selection)
        writer.py          (atomic directory emitters)
      search/
        planner.py         (JSON -> QuerySpec)
        filters.py         (predicates, hashbitmask check, bound estimate)
        descent.py         (HKM traversal + preview streaming)
        ranker.py          (NumPy distance + heap top-k)
        searcher.py        (top-level facade)
        loader.py          (doc content loader with caching)
      bin/
        hkm-index          (CLI stub)
        hkm-search         (CLI stub)
      tests/
        ...                (all test files)

Each module currently holds only imports, dataclasses, function/class
signatures, and docstrings explaining their contract.

===============================================================================
9.  CONFIGURATION CONSTANTS (schema.py)
===============================================================================

    LEAF_MAX_CHUNKS        = 256_000
    PREVIEW_CHUNKS         = 1024        # 512 random + 512 diverse
    WINDOW_SIZES           = (8, 32, 128, 512)
    STRIDE_FACTOR          = 0.5
    HASHBITMASK_FP_RATE    = 0.01
    KMEANS_MAX_K           = 4096
    DESCEND_K              = 8
    SHARD_MAX_BYTES        = 8 * 2**20  # 8 MB

===============================================================================
10. PERFORMANCE TARGETS  (32 vCPU, 16 GB RAM reference)
===============================================================================

  *  Build throughput   > 100K docs / min / worker
  *  Search cold-cache  p50 ~= 300 ms, p95 ~= 900 ms
  *  Preview first byte < 120 ms
  *  Serve RAM          < 512 MB resident + mmap
  *  Disk footprint     ~ 3 copies of all data

===============================================================================
11. FUTURE WORK / OPEN QUESTIONS
===============================================================================

  *  PQ compression for ANN if disk becomes limiting.  
  *  Filter-relaxation heuristics (drop rarest label vs widen numeric range).  
  *  Incremental rebuilds (delta ingest) vs full rebuild.  
  *  Suffix / wildcard search layer (FM-index).  
  *  Horizontal query fan-out; current design is single stateless worker.

```
