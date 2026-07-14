# HKM Evidence-First Roadmap

This is the working implementation record for turning HKM into a high-quality,
agent-first private search product. The quality objective dominates latency:
the system must return the best supported passages, preserve provenance, and
make recall/cost tradeoffs measurable. Cheapness is a scaling hypothesis to be
tested at large synthetic sizes, not a claim inferred from the current laptop
corpora.

## Status at start

The repository is a credible pre-alpha engine: hierarchical token and semantic
search, hybrid ranking, metadata filters, incremental builds, a shared-
filesystem job manager, CLI/TUI clients, and 42 passing tests. It has no
repeatable retrieval-quality harness, no scaling report, and no validated wheel
installation. A 1.08 MB book currently produces about 65 MB of canonical index
data and 118 MB of embedding cache. Several old bug reports describe behavior
that now has regression tests; the remaining actionable reports are the
non-positive file-size flag and macOS resource sampling.

Current implementation status: 51 tests pass; the quality harness, retrieval
contract, scaling evidence, inspection commands, local wheel packaging, and
atomic generation publication are complete. A clean wheel build/install also
passes outside the repository tree.

## Gates and work queue

### A. Quality/evidence harness — complete

- [x] Add a deterministic benchmark command with JSON and human-readable output.
- [x] Add an exhaustive semantic oracle and compare HKM recall at multiple probe budgets.
- [x] Add exact phrase/AST/filter correctness cases, including adversarial near misses.
- [x] Report source/index/cache bytes, windows, build/update work, candidate counts, and p50/p95/p99 timings.
- [x] Produce a scaling chart for deterministic 1K, 10K, 100K, 1M, and 1B-window estimates.
- [x] Keep agent-facing results window-level, provenance-rich, deterministic, and auditable.

Quality gate: exact retrieval is 100% correct on the fixture suite; semantic
Recall@10 is measured against exhaustive search; no optimization is accepted
without a recorded quality delta. Latency is diagnostic, not the primary gate.

### B. Retrieval quality and agent contract — complete

- [x] Replace hybrid full-corpus lexical candidate scans with indexed candidate generation.
- [x] Make semantic probe breadth explicit and test recall/cost curves.
- [x] Preserve window size, token span, source identity, score components, and match evidence in every hit.
- [x] Define filtered top-k and pagination semantics before freezing v1.
- [x] Add deterministic tie-breaking and query/result provenance suitable for agents.

### First benchmark evidence

The first run used `tmp_hkm_source_index` (76 passages, 6,615 embedding
windows) and the local DRAMA backend. With `probe_count=2`, document Recall@10
was 0.30, 0.70, and 0.60 for three queries while inspecting 14% to 26% of
windows. With `probe_count=0` (exhaustive), Recall@10 was 1.00 for both tested
queries. This confirms that the quality-first default must remain exhaustive
until a larger benchmark identifies a safe probe policy. The same report
estimated 15,446.85 GiB for one billion windows at the observed 16,585.93
bytes/window; this is a storage warning and an extrapolation, not a measured
1B-document result.

An exact phrase benchmark initially exposed a correctness bug: routed leaf
chunks can contain non-contiguous global document ids, so
`min_document_id + local_index` misidentified exact matches. Token verification
now derives ids from each chunk's embedding index, and the repository-corpus
test asserts the complete expected source set.

### C. Storage and scale — complete

- [x] Measure overlap ablations against quality.
- [x] Measure window-size ablations against quality.
- [x] Measure float16 and int8 ablations against quality.
- [x] Measure append-only growth and compact stale incremental data.
- [x] Generate scaling charts with observed small-data slopes and clearly marked extrapolation to 1B documents/windows.
- [x] Explain which costs are linear, logarithmic, bounded by probe budget, or dominated by embedding generation.
- [x] Measure append-only generation staging bytes and elapsed copy time so update
  work is explicit rather than hidden in the latency story.

### D. Publication and operations — complete

- [x] Extend index audit to every referenced artifact and count/shape invariant.
- [x] Publish audited generations atomically so failed builds never replace a valid index.
- [x] Add build/search inspection with stage timings, failures, resources, and storage composition.
- [x] Resolve `--max-file-bytes <= 0` semantics and harden macOS RSS/CPU sampling.

The launcher now keeps the public index directory stable, stages each build in
`.hkm_builds/<generation>`, and leaves `.hkm_jobs` and source directories in
place. The staged manifest is audited before one `os.replace()` of the public
`index.json`; compatibility aliases are swapped afterward and older active
generations are pruned. A failed worker or tree job therefore leaves the prior
manifest and generation searchable. Incremental staging records copied bytes
and seconds in `build_config`, making its linear update cost visible for future
copy-on-write/reflink work.

The macOS sampler has a tested whitespace-tolerant `ps` parser; a live
long-running executor sample remains an environment-level verification item.

### E. Embedded alpha packaging — complete

- [x] Validate a clean wheel install with all HKM subpackages and one `hkm` command.
- [x] Expose `build`, `search`, `audit`, `inspect`, and `benchmark` through one coherent CLI.
- [x] Keep the first deployment embedded/private; defer service, auth, replication, and tenancy until a design partner requires them.
- [x] Publish a benchmark report with hardware, corpus revisions, commands, raw results, and known limits.

### F. Agentic retrieval quality — in progress

- [x] Sample deterministic raw passages from active HKM documents and record the exact target document.
- [x] Add `hkm-agent-benchmark` with an OpenAI-compatible LM Studio query planner, deterministic offline planner, evidence-aware reranking, and probe curves.
- [x] Preserve exact-retrieval recall for active documents absent from searchable HKM leaves; these are now explicitly routed through a safe fallback path.
- [x] Run a 20-passage repository-corpus baseline: final evidence-assisted recall@10 and precision@1 are both 1.000 with the deterministic planner.
- [x] Run a 50-passage Fourth Wing prose baseline: final evidence-assisted recall@10 and precision@1 are both 1.000; model first-pass precision@1 is 0.960.
- [x] Run the harness with Gemma 4 E4B served by LM Studio's bundled llama.cpp 2.24 backend; 100 random Fourth Wing passages reached final target-doc and evidence relevance recall@10/precision@1 of 1.000, with 1% fallback.
- [x] Evaluate 174 non-empty random passages across prose, code, and metadata-heavy corpora: 100 Fourth Wing prose passages plus 74 repository passages. The repository final evidence relevance recall@10/precision@1 is 1.000/1.000; exact document identity is 0.919/0.784 because several files contain identical evidence.
- [x] Remove target-document knowledge from fallback decisions, disable model reasoning, shorten the planner prompt, cap query output at 24 tokens, and cache source slices during evidence ranking. The 74-passage repository run measured about 1.1 s median planning and 162 ms median exhaustive search with a 9.5% fallback rate.
- [x] Add an optional deterministic-first planner path. On a 20-passage real Gemma run it called the model for 20% of samples while retaining final evidence relevance recall@10/precision@1 of 1.000; the full current source-index sample still exposes two stale leaf-artifact misses.
- [x] Preserve non-contiguous document IDs while routing recursive chunks. The old source index had silently reassigned later documents to `min_id + local_index`, causing the reproducible `tests/test_chunk_io.py` and `embedder.py` misses.
- [x] Rebuild the 76-file source corpus after that repair: 169 active passages pass audit, and deterministic-first real Gemma retrieval reaches final content-evidence recall@10/precision@1/MRR of 1.000/1.000/1.000.
- [x] Add adaptive probe escalation. With `--initial-probe 2`, the 74-passage real Gemma run used the model for 16.2% of samples, reached exact target recall@10 of 1.000 and final evidence relevance recall@10/precision@1 of 1.000/1.000, with about 212 ms median initial search and 64 ms median exhaustive escalation.
- [x] Validate adaptive probe 2 on 100 random Fourth Wing prose passages: final target and evidence recall@10/precision@1/MRR were all 1.000, with no model calls or fallback searches required.
- [x] Add a complete LM Studio model/tool/model harness. Gemma 4 E4B made real `search_index` calls on a five-passage repository sample; tool-call rate, target-doc recall@5, and content-evidence recall/precision/MRR were all 1.000. The harness reports model citation matches separately from grounded tool results.
- [x] Add explicit tool search modes. Token-only retrieval avoids the embedding pass; hybrid remains the correctness default for general callers.
- [x] Add the one-completion grounded tool-only path, a 16-word query budget, and score/short-query fallback lanes. Across 50 random repository passages and 50 random Fourth Wing passages, live Gemma tool calls achieved 1.000 content-evidence recall with zero errors; evidence precision@1 was 0.920 and 0.980 respectively.
- [x] Wire deterministic-first routing and evidence-aware reranking into the tool path. The same 50-sample gates retained 1.000 content-evidence recall and precision@1 with zero model calls; median result latency was about 351 ms on prose and 163 ms on the repository.
- [ ] Benchmark adaptive defaults on larger corpora and minimize the remaining search/model tail latency.

Evidence relevance is content-based: a returned source must contain the sampled
passage after whitespace normalization or at least 90% of its distinct terms.
This separates a genuinely relevant duplicate file from an exact document-id
miss and keeps the target-id audit visible.

The LM Studio application and multiple GGUF models are installed locally, but its
desktop server is not running in the current locked-Mac session; `lms server
start` waits for that service. Its installed llama.cpp 2.24 backend does serve
the Gemma 4 GGUF directly with reasoning disabled, so the real-model gate is
now measurable without installing another runtime. The separate audio environment has
`llama-cpp-python==0.3.9`, but it cannot load these newer GGUF architectures.

## Decisions

- The first milestone is evidence and quality, not a hosted service.
- Exhaustive search is the correctness oracle; HKM is allowed to trade work for recall only when the tradeoff is visible.
- The current Mac is the initial reference machine; every report records software and hardware details.
- Benchmark-only datasets/tools may add optional dependencies; HKM runtime dependencies stay minimal.
- Backward compatibility is not required before the first external alpha; unused query plumbing may be deleted.

## Change log

- 2026-07-13: Created from repository audit. Confirmed 42 tests pass. Added the quality-first objective, agent-result requirements, storage/scaling gates, and explicit deferral of service features.
- 2026-07-13: Added `tools/benchmark.py`, `bin/hkm-benchmark`, exhaustive-vs-probe recall measurement, scaling Markdown/Mermaid output, and `Hit.window_size`. Focused semantic/hybrid search defaults on exhaustive traversal; positive `probe_count` is now explicit and measurable. Added benchmark unit coverage; 18 focused tests pass.
- 2026-07-13: Added independent exact phrase and metadata-filter checks to the benchmark. Fixed routed-chunk document-id mapping and added a regression assertion. Made recursive tree depth derive from embedding count and leaf size instead of the hard-coded depth-3 cap, so scale estimates are structurally meaningful.
- 2026-07-13: Expanded `audit_index()` to validate active rows, node artifacts, centroids, chunk readability, array shapes, and finite values; added corruption coverage. Rejected non-positive file limits and hardened macOS `ps` parsing with a pure parser test.
- The non-positive file limit is now complete; macOS sampling remains unverified on a live long-running executor because this environment cannot provide that external run as a test oracle.
- 2026-07-13: Hybrid lexical candidate generation now traverses indexed token/Bloom paths and scores only candidate documents. Added adaptive-depth unit coverage and a routed-chunk exact-match regression.
- 2026-07-13: Benchmark now reports persisted build-stage durations, ingest/cache work, storage-component breakdowns, deterministic 1K/10K/100K exhaustive-scan points, and float16/int8 retrieval-quality/storage ablations. Current source-index evidence shows previews and token sketches are larger than raw token storage, making them primary compression targets.
- 2026-07-13: Added per-window-size removal ablations. On the current source index, removing 32-token windows reduced one tested query's Recall@10 to 0.10, while removing 128-token windows reduced it to 0.90; other removals were neutral for that query. This supports retaining multiple sizes until a broader query set is evaluated.
- 2026-07-13: Measured incremental storage and verified `--full-rebuild` compacts stale append-only chunks while reusing cached embeddings; added a regression assertion that compacted canonical bytes do not exceed the incremental form.
- 2026-07-13: Added filtered pagination/provenance checks (`query_id`, `index_build_id`), p50/p95/p99 timing output, overlap-removal ablations, `hkm-audit`/`hkm-inspect`, a clean local wheel/console-script package, and a checked-in source-index benchmark report. Atomic generation publication was initially documented as blocked pending a stable sidecar job-root design.
- 2026-07-13: Replaced in-place publication with stable-root generations under `.hkm_builds/`; `.hkm_jobs` and source roots remain stable, audited generations publish through one public-manifest replacement, aliases are swapped afterward, and failed-build coverage proves the prior generation remains searchable. Incremental staging bytes and elapsed copy time are now part of build evidence.
- 2026-07-13: Added `hkm-agent-benchmark` for raw-passage sampling, LM Studio-compatible query planning, deterministic offline regression, evidence-aware reranking, and probe curves. A 20-passage repository run reached 1.000 final recall@10 and precision@1 after repairing token-context variants and active documents omitted from HKM leaves.
- 2026-07-13: Served Gemma 4 E4B through the installed LM Studio llama.cpp 2.24 backend with reasoning disabled. Final content-evidence relevance was 1.000 recall@10/precision@1 across 100 prose passages and 74 non-empty code/metadata repository passages; exact document identity remains lower for duplicate files.
- 2026-07-13: Added the real model/tool/model agent harness and validated five live Gemma tool calls at 1.000 tool-call rate and 1.000 target/evidence recall@5; normalized truncated tool arguments and kept grounded evidence separate from optional model source-path citations.
- 2026-07-13: Added token-only tool retrieval, a one-completion grounded result path, bounded model queries, confidence-triggered semantic/lexical fallback lanes, deterministic-first routing, and evidence-aware reranking. A 100-sample live cross-corpus gate retained 1.000 content-evidence recall and precision@1; deterministic-first reduced median result latency to about 163-351 ms with zero model calls on this gate.
