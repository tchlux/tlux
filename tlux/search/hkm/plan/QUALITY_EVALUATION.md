"""Evaluation protocol for separating retrieval quality from system quality."""

# Purpose

The checked-in quality harness is a diagnostic benchmark, not a claim of
web-scale or external-benchmark superiority. It keeps corpus, query, qrels,
and system outputs separate so that retrieval improvements can be measured
without turning target recovery into comprehensive recall.

# Corpus and query design

Use a deterministic heterogeneous fixture for fast regression tests and keep
large downloaded corpora in ignored paths. Each query records its category,
origin, source-blind status, expected answerability, and required evidence
facets. Natural queries are written from corpus metadata or user intent without
passing the target text to the query generator. Passage-derived queries are
diagnostic only and must not enter a reported holdout.

The fixture includes short and long text, duplicates, conflicting passages,
hard negatives, code, tables, noisy OCR-like text, malformed text, Spanish,
and an unanswerable query. The qrels use grades rather than a binary target and
include multiple relevant documents and complementary evidence requirements.

# Metrics

Report precision@k, recall@k over judged relevant documents, AP/MAP, MRR,
nDCG, judged and unjudged counts, facet coverage, complementary-hit rate,
source diversity, redundancy, confidence calibration, and abstention quality.
Unknown results are never silently removed: every report states whether the
metric excludes them, treats them as non-relevant, or rejects incomplete qrels.

# Statistical protocol

Compare systems on the same query rows. Use paired bootstrap confidence
intervals and paired randomization tests for metric deltas. Record the seed,
number of resamples, query count, and a simple power estimate. Do not tune on
the private holdout; publish a corpus/query hash with every result.

# System ablations

The benchmark names the available lanes independently: lexical/token,
BM25, dense/semantic, sparse, late interaction, hybrid, reranking, query
rewriting, agentic retrieval, and fallback. An unavailable lane is reported as
unavailable rather than silently represented by another implementation. HKM's
existing `tools/benchmark.py` remains the source for index audit, probe-count,
storage, build-stage, and quantization measurements.

# Operational and product contracts

Run latency at multiple concurrency levels and report p50, p95, p99, errors,
throughput, and saturation. Record build duration, stage durations, peak
resource samples when available, index bytes by component, and whether a
number is measured or extrapolated. Validate metadata filters and provenance
fields separately from authorization: descriptive tenant or source metadata
is not an ACL, and no authorization behavior is inferred by this benchmark.

Answer-quality evaluation is downstream of retrieval. It must score answerable,
unanswerable, and conflicting-source questions using claim-level correctness,
unsupported claims, citation precision/recall, and abstention; retrieval
scores alone are not answer-faithfulness evidence.
