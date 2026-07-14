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
- [x] Benchmark adaptive defaults on larger corpora: all 75 eligible repository passages and all 395 prose chunks retain 1.000 content-evidence recall and precision@1 with deterministic-first routing and zero model calls.
- [x] Guarantee a grounded tool result when a model emits no or malformed tool call: the wrapper falls back to the bounded deterministic planner and evidence reranker. A 20-sample default-reasoning llama.cpp stress run recovered 100% of calls with zero errors and 1.000 content-evidence recall/precision@1.
- [x] Enforce structured JSON query output for LM Studio, reject unquoted planner text, and disable hidden reasoning in the planner request. The all-395 live Gemma gate reached 0.924 first-pass recall and 0.830 precision@1, then 1.000 final evidence recall/precision@1 with no planner errors and 7.6% fallback.
- [x] Add the structured planner-to-tool agent path. With deterministic-first routing, all 395 prose passages and all 75 repository passages retain 1.000 evidence recall/precision@1/MRR with zero model calls; median/p95 latency is 251/354 ms on prose and 89/382 ms on the repository.
- [x] Verify the model-first planner-tool gate after parser and timeout hardening: 50 random prose passages and 75 mixed repository passages each retained 1.000 evidence recall/precision@1/MRR with zero errors and one model call per sample. With a 16-token planner budget and 1.0-second timeout, the latest warmed Gemma 4 prose gate measured 0.880/1.055 s median/p95 with zero planner recovery; exact repository identity remains limited by duplicate content.
- [x] Add the compact native planner/tool bridge: a structured query is followed by a short native `search_index` call. Live Gemma 4 gates made native calls on 98.0% of 50 prose samples and 90.7% of 75 repository samples; wrapper tool calls and evidence recall/precision@1/MRR remained 1.000 with zero errors.
- [x] Add `hkm-agent`, a persistent JSONL local-agent entry point that reuses one HKM searcher and LM Studio client; warm deterministic-first smoke requests complete in roughly 56-90 ms after startup.
- [x] Tighten evidence acceptance to require the normalized raw passage in readable sources, then rank verified evidence before lexical score. All 395 prose and all 75 repository passages retain 1.000 evidence recall/precision@1/MRR; duplicate repository content still limits exact document identity.
- [x] Add a bounded persistent planner-query cache for repeated raw passages; cache hits retain full HKM search and evidence validation while avoiding another LM Studio completion.
- [x] Bound long planner prompts to 64 raw words while retaining the complete passage for fallback/evidence checks; a 2,048-word passage now remains exactly grounded instead of failing closed.
- [x] Default the persistent `hkm-agent` CLI to deterministic-first routing and add an explicit `--model-first` override; the live 50-sample low-compute gate makes zero model calls at 252/326 ms median/p95.
- [x] Bound the default LM Studio planner timeout to 1.0 second; a live 50-sample prose gate retained 1.000 evidence recall/precision@1/MRR with zero errors at 880/1,055 ms median/p95 model-first agent latency.
- [x] Raise the planner budget to 16 tokens while retaining the 1-3 exact-word prompt; the warmed Gemma 4 50-sample gate retained perfect evidence metrics with zero planner recovery and 28% bounded fallback.
- [x] Order deterministic rare-term fallback before phrase lanes; the live 50-sample gate retained perfect evidence metrics while reducing model-first agent latency to 578/1,227 ms median/p95 and search latency to 42/384 ms.
- [x] Add a `--require-grounded` benchmark gate that exits nonzero unless final evidence recall, precision@1, and MRR are all 1.000.
- [x] Separate persistent startup from request latency with a one-time 15-second planner warmup budget; a live 20-request model-first run grounded 20/20 requests, with 17 model queries and 3 deterministic rescues.
- [x] Expose the native planner/tool bridge through persistent `hkm-agent --native-planner-tool`; a live 10-sample gate emitted `search_index` on 10/10 requests with perfect evidence metrics.
- [x] Reuse the bounded planner cache in native mode while retaining a fresh `search_index` completion; repeated native requests drop from two completions to one without skipping evidence validation.
- [x] Re-measure the smaller Gemma 3 planner-only path: a warmed all-395-passage run grounded 395/395 requests with 390 model queries, 5 rescues, and 734/891 ms median/p95 latency under the 16-token, 1.0-second budget; Gemma 4 remains the native-tool choice.
- [x] Normalize whitespace in planner-cache keys so equivalent raw passages reuse one query without changing evidence validation.
- [x] Collapse multi-term lexical candidate scans into one HKM traversal; all 395 prose chunks retain 1.000 evidence recall/precision@1/MRR with zero model calls at 83/117 ms median/p95.
- [ ] Reduce the remaining tail latency. The deterministic path is now below 120 ms p95 on the prose corpus; the latest warmed 50-sample Gemma 4 planner gate measures 0.880/1.055 s median/p95, so local model generation remains the main tail.

Evidence relevance is content-based: when a readable source snapshot exists, a
returned source must contain the sampled passage after whitespace
normalization. If the source snapshot is unavailable, the audit accepts at
least 90% of distinct terms. This separates a genuinely relevant duplicate
file from an exact document-id miss and keeps the target-id audit visible.

The LM Studio application and multiple GGUF models are installed locally. Its
bundled llama.cpp 2.24 backend serves the Gemma 4 E4B GGUF with reasoning
disabled, and the live model/tool gate is repeatable. The separately installed
Python 3.12 `llama_cpp` binding is older (0.3.16) and aborts while loading these
newer GGUF architectures; use the LM Studio backend or upgrade the binding
before relying on direct Python serving.

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
- 2026-07-13: Added token-only tool retrieval, a one-completion grounded result path, bounded model queries, six-term deterministic planning, eight-term rescue bounds including repeated identifiers, evidence-first fallback short-circuiting, confidence-triggered semantic/lexical fallback lanes, deterministic-first routing, and evidence-aware reranking. The current live 20-sample Gemma tool-only gate retains 1.000 content-evidence recall and precision@1 at 1.70/2.92 s median/p95. All-active deterministic gates cover 75 repository passages and 395 prose chunks at the same evidence quality; p95 latency remains the next optimization target.
- 2026-07-13: Replayed the captured 50-sample Gemma query set after the high-score/no-evidence fix: content-evidence recall and precision@1 are both 1.000 with zero errors. A fresh direct llama.cpp 2.24 generation gate with reasoning disabled also reached 1.000 content-evidence recall@5/precision@1 and 1.000 tool-call rate; target-document recall@5 was 0.980 at 1.57/2.85 s median/p95 agent latency.
- 2026-07-13: Added deterministic recovery for missing or malformed model tool calls and separated model tool-call, wrapper tool-call, and recovery rates. Under default llama.cpp reasoning, a 20-sample repository stress run recovered every call while retaining 1.000 content-evidence recall/precision@1 with zero errors.
- 2026-07-13: Re-ran the 50-sample gate through the live LM Studio endpoint at `192.168.8.222:1234` using `google/gemma-4-e4b`: wrapper tool-call rate 1.000, raw model tool-call rate 0.280, recovery rate 0.720, content-evidence recall/precision@1 1.000/1.000, zero errors, and 2.09/3.83 s median/p95 agent latency.
- 2026-07-13: Token-mode fallback now tries five bounded lexical lanes before semantic recovery and short-circuits on rank-1 evidence. The live LM Studio deterministic-first 50-sample repository gate retained 1.000 content-evidence recall/precision@1 with zero model calls at 112.73/1,561.82 ms median/p95 latency; tail reduction remains open.
- 2026-07-14: Ranked lexical candidates before constructing source previews, preserving perfect evidence quality on all 395 prose chunks while reducing deterministic-first latency to 254/361 ms median/p95. The all-395 live LM Studio tool-only gate also returned rank-1 target/evidence results for every sample with zero errors; raw model tool calls were 6.3% and deterministic recovery handled 93.7%.
- 2026-07-14: Reduced the LM Studio tool completion budget from 32 to 16 tokens. The all-395 live gate retained 1.000 target/evidence recall, precision@1, and MRR with zero errors and 100% deterministic recovery; median/p95 agent latency fell to 1,485/1,603 ms.
- 2026-07-14: Added LM Studio JSON-schema query planning with `reasoning_effort: none` and compatibility retries for endpoints that reject either option. Across all 395 random prose passages, model first-pass recall/precision@1 were 0.924/0.830; evidence fallback restored 1.000/1.000 with a 7.6% fallback rate and no planner errors.
- 2026-07-14: Rejected planner outputs with no lexical overlap against the raw passage, preventing unsupported endpoints from searching instruction text; deterministic recovery remains the correctness fallback.
- 2026-07-14: Added `--planner-tool`, which turns structured model query output into one grounded `search_index` call without relying on Gemma's long-prompt native tool template. Deterministic-first all-corpus gates remain perfect on evidence quality and avoid model calls.
- 2026-07-14: Ran the all-75 repository LM Studio tool-only gate across code and metadata passages. Evidence recall/precision@1/MRR were 1.000/1.000/1.000 with zero errors; raw model tool calls were 1.3%, recovery was 98.7%, and exact target-document recall/precision@1 were 0.960/0.787 because duplicate files share content.
- 2026-07-14: Hardened the model-first planner-tool lane: generic instruction words no longer satisfy the evidence guard, truncated JSON query strings are parsed safely, planner output is bounded, the default LM Studio timeout is two seconds, lexical rescue stops once rank-1 evidence is proven, and ungrounded candidates fail closed. Current 50-prose and 75-repository live gates remain perfect on evidence quality with one model call per sample.
- 2026-07-14: Reduced structured planner output to 8 tokens after live comparison. Current 50-prose and 75-repository model-first gates retain 1.000 evidence recall/precision@1/MRR; median/p95 agent latency is 0.953/1.281 s and 0.958/1.368 s respectively, with deterministic recovery for truncated outputs.
- 2026-07-14: Compared the installed smaller `google/gemma-3-4b` planner. Its warm model-first gates retain 1.000 evidence recall/precision@1/MRR at 0.728/1.107 s prose median/p95 and 0.765/1.427 s repository median/p95; it is recommended only through `--planner-tool` because it is not tool-trained.
- 2026-07-14: Added `--native-planner-tool`, which uses two compact completions to get an actual Gemma 4 `search_index` call without sending long raw passages through the native template. Raw model tool-call rates were 98.0% prose and 90.7% repository; grounded evidence remained perfect, at the cost of roughly 1.9/2.3 s and 1.9/2.7 s median/p95 latency.
- 2026-07-14: Re-measured deterministic-first after rescue short-circuiting: all 395 prose and all 75 repository passages retain perfect evidence quality at 0.251/0.354 s and 0.089/0.382 s median/p95 respectively, with zero model calls.
- 2026-07-14: Reduced the local LM Studio default timeout to two seconds. A warm 50-sample Gemma 4 planner gate retained 1.000 evidence recall/precision@1/MRR at 0.936/1.277 s median/p95; unloaded-model requests now fail fast into deterministic recovery instead of extending the result tail.
- 2026-07-14: Added the persistent `hkm-agent` JSONL CLI. It keeps the index and LM Studio client alive, supports startup warmup and deterministic-first routing, and returned grounded results in a local smoke test at roughly 56-90 ms per warmed request.
- 2026-07-14: Rejected generic term-overlap false positives when a readable source snapshot exists and ranked verified raw-passage evidence first. All 395 prose and 75 repository deterministic gates retain 1.000 evidence recall/precision@1/MRR; a persistent 12-sample Gemma 4 model-first gate also returned 1.000 exact-source evidence with 0.536/1.013 s median/p95 latency.
- 2026-07-14: Changed persistent-agent warmup from a generic one-token completion to a representative structured planner request. The same 12-sample Gemma 4 process retained 1.000 grounded/exact evidence while reducing recovery to 8.3% and roughly halving the observed request tail.
- 2026-07-14: Re-ran the full live Gemma 4 planner-to-tool gate after the precision fix on 50 random prose passages. Final target/evidence recall/precision@1/MRR remained 1.000 with zero errors and one planner completion per sample; recovery was 24%, fallback was 18%, and median/p95 agent latency was 609/1,179 ms.
- 2026-07-14: Re-ran the same live 50-sample gate with deterministic-first routing. It made zero model calls while retaining 1.000 target/evidence recall, precision@1, and MRR; median/p95 agent latency was 252/326 ms.
- 2026-07-14: Added a bounded 256-entry planner-query cache to the persistent agent. Repeated full raw passages stayed grounded while dropping from 578 ms on the first request to 35-39 ms on subsequent requests with no additional model calls.
- 2026-07-14: Bounded LM Studio planner prompts to 64 words and added full-source validation for long passages spanning multiple HKM chunks. A 2,048-word raw passage now returns exact grounded evidence at 1,483 ms agent/422 ms search time after warmup.
- 2026-07-14: Made the persistent `hkm-agent` CLI deterministic-first by default, with `--model-first` as the explicit LM Studio override, matching the measured zero-model-call low-compute gate.
- 2026-07-14: Reduced the default LM Studio planner timeout from 2.0 to 1.5 seconds. A live 50-sample repository gate retained perfect evidence metrics at 0.903/1.411 seconds median/p95; timeout failures still use evidence-checked deterministic rescue.
- 2026-07-14: Shortened the 8-token planner instruction to request 1-3 exact evidence words. A live 50-sample repository gate retained perfect evidence metrics with zero planner recoveries, 36% bounded fallback, and 0.897/1.542 seconds median/p95 agent latency.
- 2026-07-14: Ordered the deterministic rare-term fallback before phrase lanes. The same live 50-sample gate retained perfect evidence metrics while reducing model-first agent latency to 0.578/1.227 seconds median/p95 and search latency to 42/384 ms.
- 2026-07-14: Added `--require-grounded` to make the benchmark fail closed when final evidence recall, precision@1, or MRR drops below 1.000; the 75-sample deterministic repository gate passes this check.
- 2026-07-14: Added a one-time 15-second persistent-agent planner warmup that restores the 1.5-second request timeout. A live 20-request Gemma 4 run grounded every request, using 17 model queries and 3 deterministic rescues at 1,067 ms median/1,306 ms maximum request latency.
- 2026-07-14: Exposed the native planner/tool bridge through persistent `hkm-agent --native-planner-tool`. A live 10-sample Gemma 4 gate emitted `search_index` on every request while retaining perfect evidence metrics; the faster wrapper remains the default.
- 2026-07-14: Verified the persistent native mode with five warmed requests: 5/5 model-emitted `search_index` calls, 5/5 grounded results, two completions per request, and 2,091 ms median agent latency.
- 2026-07-14: Extended the planner cache to native mode. A repeated native request retained a fresh model tool call and grounding while dropping from 1,579 ms/two completions to 674 ms/one completion.
- 2026-07-14: Normalized whitespace in planner-cache keys; equivalent raw passages now reuse one bounded query while search and evidence checks still receive the original text.
- 2026-07-14: Re-measured warmed Gemma 3 as the lower-cost planner-only option: 50/50 grounded requests, 48 model-generated queries, 2 rescues, and 0.917/1.124 seconds median/p95 latency; native mode remains on Gemma 4.
- 2026-07-14: Raised the structured planner budget from 8 to 16 tokens after direct LM Studio responses showed Gemma JSON truncation at the smaller cap. A warmed 50-sample Gemma 4 gate had zero planner recovery with 1.000 evidence metrics at 1.001/1.329 seconds median/p95; Gemma 3 remained grounded 50/50 with 48 model queries and 780/1.160 seconds median/p95.
- 2026-07-14: Changed token candidate collection to scan each HKM node once for all query terms. The all-395 grounded prose gate stayed perfect while deterministic-first latency fell to 83/117 ms median/p95.
- 2026-07-14: Re-ran the native Gemma 4 bridge after the search optimization: 10/10 model-emitted tool calls and grounded results, with search latency at 114/197 ms median/p95.
- 2026-07-14: Reduced the default planner timeout from 1.5 to 1.0 seconds. A warmed 50-sample Gemma 4 planner gate retained perfect grounded metrics with zero planner recovery while reducing agent latency to 880/1,055 ms median/p95; timeout rescue remains evidence-checked.
- 2026-07-14: Re-measured the lower-cost Gemma 3 planner at the same 1.0-second timeout: 50/50 grounded, 48 model queries, 2 rescues, and 721/881 ms median/p95 latency.
- 2026-07-14: Extended the lower-cost Gemma 3 gate to all 395 prose chunks: 395/395 grounded with 390 model queries, 5 rescues, and 734/891 ms median/p95 latency.
