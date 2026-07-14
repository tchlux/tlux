# HKM

HKM is a Python library for building and searching a hierarchical chunk index over a shared filesystem. The library is the core product. The TUI is the default interface today, and future UIs should call the same library and job APIs.

## Current supported model

- All real work is done through the filesystem-backed job manager in [`jobs.py`](jobs.py).
- Local-machine usage is the default: enqueue jobs and let local workers execute them.
- Distributed usage uses the same job directory on a shared filesystem and workers started on every participating host.
- The current index format is the directory-based `.hkmchunk` layout written by [`builder/chunk_io.py`](builder/chunk_io.py).
- The current query surface supports hybrid text search, hierarchical token pruning, semantic text queries, exact boolean text ASTs, and metadata filters with exact token verification at leaves.

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
from tlux.search.hkm import Searcher, build_search_index, build_search_index_from_documents, drain_jobs
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

Directory builds are convenient for file corpora. Library users with document
iterators and metadata should build directly from records:

```python
schema = [
    ["source_path", "bytes"],
    ["document_preview", "bytes"],
    ["category", "bytes"],
    ["year", "int"],
]

build_search_index_from_documents(
    "idx",
    [
        {"text": "hello world", "metadata": {"source_path": "a.txt", "category": "demo", "year": 2024}},
        {"text": "HELLO WORLD", "metadata": {"source_path": "b.txt", "category": "demo", "year": 2025}},
    ],
    metadata_schema=schema,
)

searcher = Searcher.from_index_root("idx")
hits = searcher.search({
    "text_ast": {"or": [{"phrase": "hello world"}, {"phrase": "HELLO WORLD"}]},
    "where": {"category": "demo", "year": {"gte": 2024}},
    "top_k": 10,
})
```

Metadata filters narrow candidates but do not affect ranking. Custom metadata
fields must be listed in `metadata_schema`; store filterable text metadata as
`bytes` when users need plain equality checks.

## CLI usage

Enqueue a build:

```bash
tlux/search/hkm/bin/hkm-index idx data/raw_docs --workers 4
```

Search an existing index:

```bash
tlux/search/hkm/bin/hkm-search idx query.json
```

`query.json` uses the stable v1 query shape:

```json
{
  "top_k": 10,
  "offset": 0,
  "text_ast": {
    "and": [
      {"or": [{"phrase": "hello"}, {"phrase": "Hello"}, {"phrase": "HELLO"}]},
      {"or": [{"phrase": "world"}, {"phrase": "World"}, {"phrase": "WORLD"}]}
    ]
  },
  "where": {
    "source_path": {"include": ["*.py"], "exclude": ["tests/*"]},
    "file_kind": {"in": [".py"]},
    "year": {"gte": 2024}
  }
}
```

`mode` defaults to `hybrid` and may be `token`, `semantic`, or `hybrid`.
Semantic and hybrid searches are exhaustive by default (`probe_count: 0`) so
the default is optimized for agent-quality recall. Set a positive
`probe_count` to probe only that many nearest child clusters per tree level;
benchmark the resulting recall before using it for a production workload.
`text_ast` runs exact tokenizer-based phrase search with `and` and `or`.
`top_k` is the page size, `offset` is zero-based pagination, and `where` holds
optional metadata filters. `where.FIELD` may be a scalar exact match, an
`{"in": [...]}` list, a numeric range using `gt`, `gte`, `lt`, `lte`, `min`, or
`max`, or source-path glob rules with `include` and `exclude`. The older
`filters` object for `path_include`, `path_exclude`, and `file_kind` is still
accepted and normalized into `where`.
The CLI prints one JSON object with `docs`, `offset`, `limit`, `count`,
`next_offset`, and the normalized `query`.

The build CLI prints the root build job id. Jobs are stored under `idx/.hkm_jobs` by default.
Builds are incremental by default when a compatible `manifests/source_snapshot.json`
exists. Use `--full-rebuild` to rebuild the HKM tree while preserving cached
document embeddings.

`--full-rebuild` is also the compaction operation: it removes stale append-only
chunks and republishes only active documents while reusing the embedding cache.

## Current on-disk layout

```text
index_root/
  index.json
  .hkm_jobs/
  .hkm_cache/
    embeddings/
  .hkm_builds/<generation>/
    index.json
    manifests/
      ingest_summary.json
      source_snapshot.json
      worker_0000.json
    docs/
      doc_index.npy
      worker_0000/shard_00000000.hkmchunk/...
    hkm/
      node.json
      cluster_0000/...
  docs -> .hkm_builds/<generation>/docs
  hkm -> .hkm_builds/<generation>/hkm
  manifests -> .hkm_builds/<generation>/manifests
```

`.hkmchunk` directories are the canonical current storage unit. Search and build code should agree with that format exactly.

The canonical query-time manifests and token artifacts are:

- `index.json` at the root with the published `generation_path`, source root, stable jobs root, embedder backend, metadata schema, build config, `max_n_gram`, `n_gram_fp_rate`, relative `docs/` + `hkm/` paths, and whether the tree has append-only incremental chunks
- `manifests/source_snapshot.json` with active source paths, content hashes, doc ids, canonical chunk rows, and leaf paths for incremental reuse
- `.hkm_cache/embeddings/` with reusable per-document tokens, embedding windows, and embeddings keyed by backend/window settings and content hash
- `node.json` at each HKM node with child order, counts, preview files, token artifact paths, and whether local `data/` exists
- `n_gram_counter.bytes` at each node with the node's merged unique-count sketch
- `n_gram_exists.bytes` at each searchable node with the node's Bloom filter for token-pruned descent

Search results currently return:

- `offset`, `limit`, `count`, `next_offset`, and normalized `query`
- `docs`, each containing:
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

## Retrieval guidance

A local book-sized source file was used as a passage-retrieval quality probe.
The built index contained 395 indexed document chunks and 19,771 embedding
windows: 15,342 at 32 tokens, 3,691 at 128 tokens, 734 at 512 tokens, and 4 at
1024 tokens.

For ten random source excerpts, short generalized semantic queries were written
without relying on specific names or source phrasing. The expected target was
the best matching embedding window from the source chunk containing the sampled
excerpt. Ranked within each window size:

- 32-token windows: median target rank 22, worst rank 111 of 15,342, hit@50 8/10.
- 128-token windows: median target rank 29, worst rank 226 of 3,691, hit@50 7/10.
- 512-token windows: median target rank 15, worst rank 107 of 734, hit@50 7/9.

Keep all embedding window sizes. Smaller windows are the best primary recall
layer for short semantic queries because they match query granularity and are
cheap to pass to a language model. Larger windows still matter because broad
scene-level matches can rank very high, but they should usually be a secondary
recall channel rather than the main prompt payload.

A practical result-surfacing path is:

1. Retrieve the top 200 32-token windows as primary anchors.
2. Add a smaller set of 128-token and 512-token hits as secondary anchors.
3. Normalize ranks per window size before merging candidates.
4. Deduplicate or merge overlapping windows by document id and token span.
5. Expand surviving anchors to enough surrounding context for display or LM
   reranking.

Do not treat the current document-level collapse as the only retrieval
abstraction for UI quality. Window-level hits with `window_size`, `doc_id`,
`token_start`, `token_end`, and score are the better substrate for reranking and
for deciding how much context to surface.

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

The local CLI now includes `hkm-audit`, `hkm-inspect`, and `hkm-benchmark`.
`hkm-inspect` reports build stages, failures, storage composition, and warm
search p50/p95/p99 timings; `hkm-benchmark` adds exhaustive quality oracles,
probe work, exact checks, quantization/window ablations, and 1K-to-1B scale
estimates. Builds stage under `.hkm_builds/`, keep `.hkm_jobs/` stable, audit
the complete generation, and atomically replace the public `index.json`
pointer. Compatibility aliases for `docs/`, `hkm/`, and `manifests/` are
swapped after publication, so a failed build leaves the previous generation
searchable.

`hkm-agent-benchmark` samples raw indexed passages, asks a local LM Studio
OpenAI-compatible endpoint for queries, and reports model first-pass quality
separately from exact target-document and content-evidence relevance. Use
`--stub` for deterministic offline regression; probe counts are diagnostic until
a real-model run proves a lower-work setting preserves recall. For example:

    bin/hkm-agent-benchmark data/fourth_wing_hkm_index --samples 50 --stub

Add `--deterministic-first` to search with cheap lexical evidence before calling
the model; the model is used only when that first result is not evidence-relevant.
Add `--initial-probe 2` to try a small HKM probe first and retry exhaustive
search only when the evidence check misses.

Use `--tool-agent` to exercise the complete local model/tool/model protocol:
the model calls `search_index`, HKM returns ranked snippets, and the model
answers from that payload. The report separates tool evidence relevance (the
grounding gate) from whether the model copied the source path into its final
JSON answer:

    bin/hkm-agent-benchmark data/fourth_wing_hkm_index --tool-agent --samples 10 --initial-probe 2

For short model-generated keyword queries, `--tool-mode token` avoids the
embedding pass. The tool bounds model queries to a small 16-word budget and
retained 1.000 tool evidence recall/precision on the live cross-corpus gate;
keep `hybrid` as the correctness default until a larger corpus gate supports
switching it.

For the lowest result latency, add `--tool-only --tool-mode token`. This stops
after one model function call and returns the grounded HKM result directly,
avoiding a second model completion. On 50 random repository passages and 50
random prose passages, this path reached 1.000 content-evidence recall with
zero tool errors; median one-completion agent latency was about 2.7-3.0 s.
The report still records model citation quality separately.

A current 20-sample live Gemma 4 E4B repository run made one tool call for
every sample and retained 1.000 content-evidence recall and precision@1;
median agent latency was 1.70 s (p95 2.90 s). The tool query response is
bounded to 32 output tokens, and this expensive model lane should be reserved
for passages that fail the cheap first pass.

Add `--deterministic-first` to skip Gemma when a cheap lexical result already
contains the raw passage. On the same 50-sample gates it reduced model calls
to 0% on both prose and repository after evidence-aware reranking, with
content-evidence recall and precision@1 at 1.000; median result latency was
about 351 ms on prose and 163 ms on the repository. The model-assisted path
remains available for low-confidence passages.

The larger all-active deterministic gates now cover 75 eligible repository
passages and all 395 prose chunks. Both retain 1.000 content-evidence recall
and precision@1 with zero model calls. The deterministic planner uses six
keyword terms and bounds rescue to four distinctive terms. Median/p95 result
latency is 88/1,122 ms on the repository and 256/358 ms on prose. Exact
document-id precision is
lower on the repository because repeated boilerplate produces content-equivalent
duplicate chunks; evidence relevance is the product gate.

The benchmark accepts any OpenAI-compatible local endpoint. When the LM Studio
desktop server is unavailable, its installed llama.cpp backend can serve the
same GGUF directly:

    runtime="$HOME/.cache/lm-studio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.24.0"
    model="$HOME/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
    DYLD_LIBRARY_PATH="$runtime" "$runtime/llama-server" -m "$model" --host 127.0.0.1 --port 1234 -c 4096 --n-predict 32 --reasoning off --jinja

## Forward-looking architecture

The future-oriented design lives in [`ARCHITECTURE.md`](ARCHITECTURE.md). That document describes the intended scalable destination. This README describes only the current supported surface.
