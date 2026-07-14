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

The local CLI now includes `hkm-audit`, `hkm-inspect`, `hkm-benchmark`, and
the persistent JSONL `hkm-agent`.
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

Add `--require-grounded` to make the command fail if any final evidence recall,
precision@1, or MRR metric is below 1.000; this is the local correctness gate
used before comparing latency.

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

Gemma 4's native tool template can truncate on long raw passages. Use
`--planner-tool` to have the model emit a structured query first, then execute
`search_index` once in the wrapper; combine it with `--deterministic-first` for
the low-compute path. Across all 395 prose passages this path returned rank-1
evidence with 1.000 recall/precision@1/MRR at 77/104 ms median/p95 and zero
model calls. The all-75 repository gate likewise kept 1.000 evidence metrics
at 90/466 ms median/p95; exact document identity remains limited by duplicate
source content.

The independent `data/fineweb_sample` corpus provides a second grounded gate
across 64 source files and 101 indexed passages. Built with the real drama
backend, its deterministic token tool path returned 1.000 target/evidence
recall, precision@1, and MRR at 19/56 ms median/p95. A live all-101 Gemma 4
planner-tool run generated a query for every passage and retained the same
perfect metrics with zero recovery at 795/1,017 ms median/p95 using a
diagnostic five-second planner timeout. The normal one-second budget remains
the production fail-fast setting and uses the same grounded deterministic
rescue when model generation misses the budget.

The larger `data/fineweb_profile` corpus is a 512-file, 785-passage gate over
about 227,000 words of heterogeneous web text. A real-drama index audited
cleanly at about 1.2 GiB. The all-512 deterministic-first token-tool run
returned 1.000 target/evidence recall, precision@1, and MRR with zero errors,
0.2% bounded fallback, and 157/494 ms median/p95 agent latency. This is the
strongest current local quality gate, but it remains a small corpus; repeat the
model-first gate and larger-scale cost tests before making a B2B performance
claim.

The model-first planner-tool path is the correctness gate when a real model
call is required: a post-parser 50-sample prose run and a 75-sample mixed
repository run both returned 1.000 evidence recall/precision@1/MRR with zero
errors and one model call per sample. With the 16-token planner budget, the
latest warmed Gemma 4 prose gate measured 0.880/1.055 seconds median/p95 with
zero planner recovery and 28% bounded fallback. Planner output rejects generic
instruction-word overlap, accepts truncated JSON arguments, and falls back
after the bounded 1.0-second LM Studio timeout. If every bounded search lane misses
the raw evidence, the tool fails closed with an empty result rather than
returning an ungrounded candidate.

Evidence ranking now requires the normalized raw passage, or all of at least
four distinct request terms after punctuation/word-order normalization, in a
readable indexed source before accepting a hit; partial generic overlap is
still rejected. With this stricter precision gate, deterministic
token retrieval still returns 1.000 evidence recall/precision@1/MRR across all
395 prose passages (77/104 ms median/p95) and all 75 repository passages
(90/466 ms median/p95). Duplicate repository files can still lower exact
document identity without lowering content relevance.

The current 50-sample live Gemma 4 planner-to-tool gate on random prose ended
at 1.000 target and evidence recall/precision@1/MRR with zero errors and one
planner completion per request. The structured path recovered 24% of omitted
model tool calls and used bounded fallback lanes on 18% of samples; median/p95
agent latency was 609/1,179 ms and search latency was 208/553 ms.

The same live gate with `--deterministic-first` made zero model calls while
retaining 1.000 target/evidence recall, precision@1, and MRR with zero errors;
median/p95 agent latency fell to 77/104 ms in the latest all-395 gate. This is
the recommended low-compute
path when indexed raw passages usually contain enough lexical evidence.

For lower warm latency, pass `--model google/gemma-3-4b` with
`--planner-tool`; this smaller installed model is not tool-trained, but its
structured planner retained the same 1.000 evidence metrics at 0.728/1.107 s
median/p95 on prose and 0.765/1.427 s on the repository. Keep Gemma 4 E4B for
the native-tool experiment; use the structured wrapper for Gemma 3.

For short model-generated keyword queries, `--tool-mode token` avoids the
embedding pass. The tool bounds model queries to a small 16-word budget and
retained 1.000 tool evidence recall/precision on the live cross-corpus gate;
keep `hybrid` as the correctness default until a larger corpus gate supports
switching it.

Use `--native-planner-tool` when the model itself must emit the native
`search_index` function call after planning. The compact two-completion bridge
reached a 0.980 raw model tool-call rate on 50 prose passages and 0.907 on 75
repository passages, while wrapper tool calls and evidence recall/precision@1/
MRR stayed at 1.000 with zero errors. Its median/p95 latency was 1.904/2.282 s
on prose and 1.897/2.690 s on the repository; use plain `--planner-tool` for
the lower-latency one-completion path.

For a long-running local service, `hkm-agent` keeps the index and LM Studio
client alive and accepts one JSON object per line. The response is also JSONL
and contains the bounded query, grounded HKM hits, tool/recovery flags, and
timings:

    printf '%s\n' '{"text":"raw passage"}' | bin/hkm-agent data/fourth_wing_hkm_index \
        --base-url http://192.168.8.222:1234/v1 --model google/gemma-4-e4b \
        --deterministic-first --warmup --top-k 5

The persistent CLI defaults to deterministic-first routing; pass `--model-first`
when every passage must go through the LM Studio planner. Use `--warmup` to pay
index and model initialization before the first request. Warmup uses a one-time
15-second planner budget, then restores the bounded 1.0-second request timeout.
The LM Studio client lazily reuses one HTTP/1.1 connection and reconnects once
after a dropped socket, avoiding connection setup on each planner request;
timeouts are never retried, so the one-second planner budget remains bounded.
Current source-index smoke runs took 3.8-7.6 seconds to start and then handled
known deterministic-first hits in roughly 56-90 ms; these are local reference
measurements, not a service-level guarantee. The smaller `google/gemma-3-4b`
model remains available for a lower warm model latency through `--model-first`.

A fresh persistent JSONL run over 101 random FineWeb passages returned 101/101
grounded responses with zero errors at 20.8/60.2 ms median/p95 per request;
the process completed in 6.96 seconds including startup.

A persistent model-first smoke gate through the active LM Studio server sent 12
random repository passages through one warmed Gemma 4 process. The planner-shaped
warmup returned grounded, exact-source evidence for every request with one
planner completion per request; median/p95 agent time was 536/1,013 ms and
deterministic recovery handled 8.3% of responses. The p95 remains the main
model-serving optimization target.

Use `--language-query` for remembered, conversational requests rather than raw
passages. The agent searches the original request, inspects bounded snippets,
asks the local model for up to four paraphrase/antipattern lanes, adds bounded
deterministic clause lanes, then searches those alternatives in a second round.
Natural-language lanes use semantic
search even when `--tool-mode hybrid` is selected; explicit forgotten-entity
requests also get a compact token lane for rare remembered terms. The first
lane is anchored,
and antipatterns are soft penalties so a useful hit is never hard-filtered:

    printf '%s\n' \
      '{"text":"A scenario where something funny is said"}' \
      '{"text":"Crowds gathered to watch in horror"}' \
      '{"text":"A person climbs a structure at night while people gather below, but I cannot remember who the person is"}' \
    | bin/hkm-agent data/fourth_wing_hkm_index --language-query \
        --base-url http://192.168.8.222:1234/v1 --model google/gemma-3-4b \
        --model-first --tool-mode semantic --top-k 5 --warmup

Language responses add `agentic`, `queries`, `antipatterns`, and `rounds` to
the normal JSONL contract. `grounded` means that the index returned evidence;
for vague requests, review the returned previews and query trace rather than
treating non-empty output as a perfect relevance guarantee. The reviewed
challenge set and concept-group gate live in
`plan/benchmark_language_queries.md`.
Run `bin/hkm-language-benchmark INDEX --model-first --jsonl report.jsonl`
to evaluate that gate and record per-case coverage plus latency.
The grounded conditional fixture in
`plan/benchmark_language_conditionals.md` adds seven increasingly conditional
requests with forgotten-name phrasing; its latest deterministic run passes
7/7 coverage/coherence checks; the latest labeled sample is 0.929 precision,
with unknown-hit precision still open.

For a reproducible harder audit, run
`bin/hkm-random-language-benchmark INDEX --samples 10 --top-k 5`. It samples
raw indexed passages and evaluates vague, specific, conditional, and
missing-entity requests against exact source evidence. The fixed Fourth Wing
baseline is documented in `plan/benchmark_random_language.md`; use
`--query-source lm` to generate requests through LM Studio and
`--require-evidence` to fail closed on an imperfect evidence gate.
The current 40-case deterministic run reaches 0.850 evidence recall@5 and
0.625 precision@1 (MRR 0.719). A latest eight-case Gemma 3 model-first smoke reaches
1.000 recall@5, 0.875 precision@1, and 0.938 MRR, including two rejected
missing-entity generations recovered with three retained clues; larger random
gates remain open.
The valid 512-file FineWeb profile index passes a 32-case model-first language
gate at 1.000 recall/precision/MRR across all four styles, with one validated
deterministic rescue.

The current warmed 50-sample Gemma 4 prose gate uses the 1.0-second timeout and
returns 1.000 evidence recall/precision@1/MRR with zero errors at 880/1,055
ms median/p95 agent latency. The 16-token prompt asks for only 1-3 exact words,
leaving enough room for complete JSON on both installed Gemma models; this run
had zero planner recoveries and 28% bounded fallback.
The timeout is a bounded failure budget, not a quality shortcut: failed planner
calls use the same evidence-checked deterministic rescue path.

With the active LM Studio server, a warmed 20-request persistent Gemma 4
model-first run grounded all 20 requests; 17 used model-generated queries and
3 used deterministic rescue. Request latency was 1,067 ms median and 1,306 ms
maximum after warmup.

For the planner-only path, the smaller `google/gemma-3-4b` is a lower-cost
option: a warmed all-395-passage run grounded 395/395 requests, used 390
model-generated queries and 5 rescues, and measured 734/891 ms median/p95 with
the 16-token planner budget and 1.0-second timeout.
Use Gemma 4 when native model-emitted tool calls are required.

Pass `--native-planner-tool` with `--model-first` when the model itself must
emit the `search_index` call. This explicit two-completion bridge is slower,
but a live 10-sample Gemma 4 gate emitted the native tool call on 10/10
requests while retaining 1.000 evidence recall/precision@1/MRR.
The persistent five-request smoke also emitted the native call on 5/5 requests
and grounded every result at 2,091 ms median agent latency.

The persistent planner also keeps a bounded 256-entry cache keyed by normalized
raw passage text. In a repeated-passage smoke test, the first grounded request took
578 ms and the next four took 35-39 ms each, with zero additional model calls.
The cache stores only the generated query; HKM search and evidence validation
still run for every request.

Native mode reuses the same planner cache while still requiring a fresh model
tool call: a repeated-passage smoke dropped from 1,579 ms and two completions
to 674 ms and one completion, with both results grounded.

Planner prompts are bounded to 64 words (head and tail) for long raw inputs;
the complete passage still drives deterministic fallback and exact evidence
validation. A 2,048-word source passage stayed exactly grounded after warmup at
1,483 ms agent time and 422 ms search time, rather than timing out or exposing
an ungrounded candidate.

For the lowest result latency, add `--tool-only --tool-mode token`. This stops
after one model function call and returns the grounded HKM result directly,
avoiding a second model completion. On 50 random repository passages and 50
random prose passages, this path reached 1.000 content-evidence recall with
zero tool errors; median one-completion agent latency was about 2.7-3.0 s.
The report still records model citation quality separately.

A current 20-sample live Gemma 4 E4B repository run made one tool call for
every sample and retained 1.000 content-evidence recall and precision@1;
median agent latency was 1.70 s (p95 2.90 s). The tool query response is
bounded to 16 output tokens, and this expensive model lane should be reserved
for passages that fail the cheap first pass.

A fresh 50-sample repository gate against the installed llama.cpp 2.24 server
with `--reasoning off` made one tool call for every sample and reached 1.000
content-evidence recall@5/precision@1 with zero errors; target-document
recall@5 was 0.980 and median/p95 one-completion latency was 1.57/2.85 s.
Disable Gemma reasoning for direct llama.cpp serving: otherwise hidden
reasoning can consume the bounded completion before the tool call is emitted.
The wrapper still guarantees a grounded result if that happens: a 20-sample
default-reasoning stress run recovered every omitted tool call deterministically,
with 1.000 evidence recall/precision@1 and zero errors. Reports separate raw
model tool-call rate from wrapper recovery rate.

The same 50-sample gate through the live LM Studio endpoint reached wrapper
tool-call rate 1.000, raw model tool-call rate 0.280, recovery rate 0.720, and
1.000 content-evidence recall/precision@1 with zero errors; median/p95 agent
latency was 2.09/3.83 s.

The captured 50-sample Gemma query set that exposed the fallback bug now replays
through the corrected tool path at 1.000 content-evidence recall and
precision@1 with zero errors; this replay excludes new model-generation time.

LM Studio query planning uses JSON-schema output and `reasoning_effort: none`,
so the model returns an actual bounded query instead of hidden instruction text.
Across all 395 random prose passages, model first-pass recall/precision@1 were
0.924/0.830 with no planner errors; the evidence-aware fallback raised final
recall and precision@1 to 1.000 with a 7.6% fallback rate.
Planner responses are also required to quote evidence terms; unsupported
endpoints that return instruction text are rejected before search and use the
same deterministic recovery path.

Add `--deterministic-first` to skip Gemma when a cheap lexical result already
contains the raw passage. On the same 50-sample gates it reduced model calls
to 0% on both prose and repository after evidence-aware reranking, with
content-evidence recall and precision@1 at 1.000; median result latency was
about 351 ms on prose and 163 ms on the repository. The model-assisted path
remains available for low-confidence passages.

The larger all-active deterministic gates now cover 75 eligible repository
passages and all 395 prose chunks. Both retain 1.000 content-evidence recall
and precision@1 with zero model calls. The deterministic planner uses six
keyword terms and bounds rescue to eight distinctive terms. Median/p95 result
latency is 88/1,122 ms on the repository and 77/104 ms on prose. Exact
document-id precision is
lower on the repository because repeated boilerplate produces content-equivalent
duplicate chunks; evidence relevance is the product gate.

Token-mode fallback now tries up to five cheap lexical alternates before paying
for semantic recovery, and returns immediately once the raw passage is
evidence-ranked first. A fresh 50-sample live LM Studio deterministic-first
repository gate kept 1.000 content-evidence recall/precision@1 with zero model
calls at 112.73/1,561.82 ms median/p95 latency; semantic recovery remains the
correctness path for the few misses.

The stronger all-395 random-passage live LM Studio tool-only gate on the real
`drama` corpus returned the target and evidence result at rank 1 for every
sample, with 1.000 recall/precision@1/MRR, zero errors, and 100% deterministic
recovery after one bounded LM Studio request per sample. Agent latency was
1,485/1,603 ms median/p95; the deterministic-first path avoids that generation
cost and measures 77/104 ms median/p95 on the same corpus after ranking
candidates before constructing source previews.
The 16-token tool cap intentionally treats a truncated Gemma tool response as
recoverable; on this long-passage prompt the raw model tool-call rate was 0.0,
while the wrapper recovery rate was 1.0. The separate structured planner above
is the model-generated query path.

The all-75 repository tool-only gate also returned 1.000 evidence recall,
precision@1, and MRR with zero errors; raw model tool calls were 1.3% and
deterministic recovery handled 98.7%. Exact target-document recall/precision@1
were 0.960/0.787 because duplicate code and metadata files are indistinguishable
from the sampled text alone.

The benchmark accepts any OpenAI-compatible local endpoint. When the LM Studio
desktop server is unavailable, its installed llama.cpp backend can serve the
same GGUF directly:

    runtime="$HOME/.cache/lm-studio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.24.0"
    model="$HOME/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
    DYLD_LIBRARY_PATH="$runtime" "$runtime/llama-server" -m "$model" --host 127.0.0.1 --port 1234 -c 4096 --n-predict 32 --reasoning off --jinja

When LM Studio is serving on its LAN address, use the advertised endpoint and
model id, for example `--base-url http://192.168.8.222:1234/v1 --model
google/gemma-4-e4b`.

## Forward-looking architecture

The future-oriented design lives in [`ARCHITECTURE.md`](ARCHITECTURE.md). That document describes the intended scalable destination. This README describes only the current supported surface.
