===============================================================================
hkm_search – Hierarchical-K-Means Large-Scale Search Toolkit
===============================================================================

Author : Thomas C.H. Lux                  Version : 0.1-spec  
Date   : 2025-06-07                       License : MIT (proposed)

This file is a **stand-alone specification**.  Hand it to any engineer with
Python + NumPy experience and they can build the system exactly as intended,
without needing prior context or extra guidance.

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

  •  Embedding nearest-neighbour (1024-D float32, L2 / cosine / inner-product)  
  •  Exact token-sequence match (caller supplies the tokeniser)  
  •  Category filters (≤ 100 distinct labels per metadata column)  
  •  Numeric range filters (float32 / int32; arbitrary distributions)  
  •  Any combination above in a single query

Hard constraints

  •  Pure CPython 3.x, **only** external dependency is NumPy  
  •  CPU-only; no GPU, no AVX512 contract  
  •  RAM per process 4 – 32 GB  
  •  Storage is a remote file/object store with *whole-file reads only*  
     (no range GET).  Atomic `mkdir` is available.  
  •  Preferred block sizes: 8 MB or 64 MB; bandwidth ≈ 64 MB s⁻¹  
  •  Latency goal: < 1 s p95; first preview ≈ 100 ms  
  •  Index is rebuilt offline; no live mutations required  
  •  Sliding-window embeddings — window sizes {8, 32, 128, 512} tokens,  
     stride = window / 2, ≈ 100 windows per doc  
  •  Code must be portable (no AWS, Docker, POSIX tricks)  
  •  Deliverable is an installable open-source PyPI package

===============================================================================
2.  DESIGN RATIONALE AND KEY ASSUMPTIONS
===============================================================================

To meet latency and RAM targets with pure Python, we adopt a **recursive
k-means (HKM) tree**:

  •  Sample 256 k embeddings → ≤ 4096-way k-means → assign → recurse  
     until a leaf ≤ 256 k chunks.  
  •  A query visits O(log N) nodes and touches ≪ 1 % of embeddings.  
  •  All binary files are bounded to 8 MB, fitting the whole-file rule.

**Instant feedback requirement**

  •  Each HKM node stores **128 preview chunks**  
       – 64 randomly sampled (representative)  
       – 64 chosen by greedy farthest-point (diverse)  
  •  Representative previews stream first (< 120 ms), then diverse.

**Strict filters**

  •  Each node holds label counts and numeric histograms.  
  •  Planner computes an upper-bound candidate count *before* descent;  
     if bound < top-k, it may relax filters or fail fast.

===============================================================================
3.  HIGH-LEVEL ARCHITECTURE
===============================================================================

            user / API
                 │
  ┌──────────────▼──────────────┐
  │          Searcher           │
  │  planner · descent · rank   │
  └──────────────┬──────────────┘
                 │ mmap / stream
                 ▼
     HKM tree (embeddings)      token store (postings)
                 ▲ build jobs
                 │
  ┌──────────────┴──────────────┐
  │        IndexBuilder         │
  └─────────────────────────────┘

Both build and search work with **whole-file** reads/writes only.

===============================================================================
4.  DATA MODEL AND ON-DISK LAYOUT
===============================================================================

Directory structure (fixed):

    index_root/
      docs/                                # immutable micro-shards
        worker_0000/
          shard_00000000.bin   (≤ 8 MB)
          shard_00000001.bin
          …
        worker_0001/
          …
      hkm/                                 # ANN hierarchy
        node_root/                         # depth 0
          embeddings/
            shard_0000.npy      (≤ 8 MB, 256 k × 1024 × 4 B)
            …
          centroids.npy         (≤ 4096 × 1024 float32)
          stats.json
          preview_random.npy    (64 × uint64 chunk_id)
          preview_diverse.npy   (64 × uint64 chunk_id)
          bloom.bin
          node_root_0003/                  # depth-first children
            embeddings/
            centroids.npy
            stats.json
            …
            node_root_0003_0019/
              embeddings/
              centroids.npy
              stats.json
              preview_random.npy
              preview_diverse.npy
              bloom.bin
              embed.npy          (leaf copy ≤ 8 MB)
              chunk_meta.npy     (leaf meta ≤ 8 MB)

Binary format: little-endian NumPy `.npy` v2.

Structured dtypes

  DOC_META_DTYPE (stored inside each docs/ shard)

        [('doc_id',   uint64),
         ('cat0',     uint32), … up to 20 cols …,
         ('num0',     float32), …                ,
         ('text_off', uint64),
         ('text_len', uint32)]

  CHUNK_META_DTYPE (stored in leaf)

        [('chunk_id', uint64),
         ('doc_id',   uint64),
         ('tok_off',  uint32),
         ('tok_len',  uint32),
         ('win_size', uint16),
         padding … to 64 B]

`stats.json` schema (per node)

    {
      "labels_bitset" : "<hex>",
      "numeric_min"   : [...],
      "numeric_max"   : [...],
      "labels_count"  : {"lang:en": 31231, ...},
      "numeric_hist"  : {"year": [[1900,1950,12], ...]},
      "doc_id_min"    : 123,
      "doc_id_max"    : 456
    }

All per-node files are ≤ 8 MB.

===============================================================================
5.  BUILD-TIME WORKFLOW
===============================================================================

PHASE A – Producer  
  •  Scans raw corpus, assigns `doc_id`, writes build_job_N.json
     (≤ 32 k docs each).

PHASE B – Shard Worker (parallel)  
  •  Claims a job via `mkdir`.  
  •  Tokenises docs; generates sliding-window embeddings.  
  •  Appends tokens + metadata to 8 MB `docs/worker_*/shard_*.bin`.  
  •  Appends embeddings to matching `embeddings/shard_*.npy` in **root node**.

PHASE C – HKM Builder (parallel by subtree)  
  1.  For a subtree: sample 256 k embeddings → ≤ 4096-way k-means.  
  2.  Assign embeddings, write `centroids.npy`, `stats.json`, `bloom.bin`.  
  3.  Choose 64 random + 64 farthest-point previews, save.  
  4.  Copy subtree embeddings into `embeddings/shard_*.npy` (≤ 8 MB each).  
  5.  Recurse until leaf ≤ 256 k chunks, then write `embed.npy`,
      `chunk_meta.npy`.  
  6.  Append node path to `manifest.jsonl`.

All writes are “write-temp → rename” for crash safety.

===============================================================================
6.  QUERY-TIME WORKFLOW
===============================================================================

 0  Parse request JSON → QuerySpec.  
 1  **Upper-bound test** using `labels_count` and `numeric_hist`.  
 2  **HKM descent**: at each level, keep 8 closest children per embedding,
    prune by metadata.  
 3  **Stream previews** when depth ≥ 2 node accepted  
       – first `preview_random.npy`, then `preview_diverse.npy`.  
 4  **Leaf ranking**: load `embed.npy`, compare query vectors to matching
    `win_size`, keep heap of top-k chunks.  
 5  **Exact token check** (optional): use `bloom.bin`; load full text only
    if bloom says “possible”.  
 6  Aggregate best chunk per document → return top-k docs.

===============================================================================
7.  PUBLIC PYTHON API (HIGH-LEVEL)
===============================================================================

    from hkm_search import FileSystem, IndexBuilder, Searcher

    # Build
    fs   = FileSystem()                  # Local file system by default
    cfg  = BuildConfig(index_root="idx", raw_paths=[...])
    IndexBuilder(fs, cfg).run()

    # Search
    searcher = Searcher(fs, "idx/docs", "idx/hkm")
    hits     = searcher.search({
                 "embeddings"     : [q32, q128],
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

    hkm_search/
      __init__.py          (re-exports FS, IndexBuilder, Searcher)
      fs.py                (FileSystem base + LocalFS impl)
      schema.py            (dtypes + constants)
      utils/
        array.py           (mmap helpers)
        bitset.py          (≤ 100-bit masks)
      builder/
        driver.py          (IndexBuilder orchestrator)
        sampler.py         (reservoir & k-means++ samplers)
        kmeans.py          (≤ 4096-way Lloyd in NumPy)
        stats.py           (label & numeric aggregations)
        bloom.py           (bit-array bloom filter)
        preview.py         (random + farthest-point selection)
        writer.py          (atomic directory emitters)
      search/
        planner.py         (JSON → QuerySpec)
        filters.py         (predicates, bloom check, bound estimate)
        descent.py         (HKM traversal + preview streaming)
        ranker.py          (NumPy distance + heap top-k)
        searcher.py        (top-level facade)
      bin/
        hkm-index          (CLI stub)
        hkm-search         (CLI stub)

Each module currently holds only imports, dataclasses, function/class
signatures, and docstrings explaining their contract.

===============================================================================
9.  CONFIGURATION CONSTANTS (schema.py)
===============================================================================

    LEAF_MAX_CHUNKS        = 256_000
    PREVIEW_CHUNKS         = 128        # 64 random + 64 diverse
    WINDOW_SIZES           = (8, 32, 128, 512)
    STRIDE_FACTOR          = 0.5
    BLOOM_FP_RATE          = 0.01
    KMEANS_MAX_K           = 4096
    DESCEND_K              = 8
    HEAP_FACTOR            = 4          # candidate inflation
    SHARD_MAX_BYTES        = 8 * 2**20  # 8 MB

===============================================================================
10. PERFORMANCE TARGETS  (32 vCPU, 16 GB RAM reference)
===============================================================================

  •  Build throughput   > 1 M docs / min / worker  
  •  Search cold-cache  p50 ≈ 300 ms, p95 ≈ 900 ms  
  •  Preview first byte < 120 ms  
  •  Serve RAM          < 512 MB resident + mmap  
  •  Disk footprint     ~ 1.3 × raw embeddings (+metadata)

===============================================================================
11. FUTURE WORK / OPEN QUESTIONS
===============================================================================

  •  PQ compression for ANN if disk becomes limiting.  
  •  Filter-relaxation heuristics (drop rarest label vs widen numeric range).  
  •  Incremental rebuilds (delta ingest) vs full rebuild.  
  •  Suffix / wildcard search layer (FM-index).  
  •  Horizontal query fan-out; current design is single stateless worker.

Enjoy building!
