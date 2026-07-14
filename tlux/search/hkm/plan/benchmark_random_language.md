# Random language-query gate

`bin/hkm-random-language-benchmark` samples active indexed documents, builds
memory-style requests from each raw passage, and runs the persistent language
agent. Four styles are included: vague, specific, conditional, and
missing-entity. The last style deliberately drops the first content term to
model forgetting a subject or object. A fixed seed makes every query and
target reproducible.

The report separates exact target-document ranks from raw-evidence ranks. The
evidence rank is the quality gate: a duplicate or neighboring chunk can still
contain the sampled passage even when its document ID differs. It also records
query origin per case, so deterministic fallback is never counted as a model
generation.

Offline deterministic smoke:

```text
bin/hkm-random-language-benchmark data/fourth_wing_hkm_index \
  --samples 10 --seed 20260714 --top-k 5 \
  --json-output /private/tmp/random_language.json
```

On 10 Fourth Wing samples (40 cases), the current deterministic baseline
reached evidence recall@5 0.725, precision@1 0.450, and MRR 0.571. By style,
evidence recall was 0.900 vague, 0.800 specific, 0.700 conditional, and 0.500
missing-entity. Median agent/search latency was 650/586 ms.
This is a challenge baseline, not a production quality claim; improve the
agent and rerun the same seed before changing the index or query templates.
The deterministic templates intentionally retain lexical clues, so they test
agent ranking and condition handling rather than claim human-level paraphrase
generation. The LM mode is the stronger query-generation experiment.

The merge keeps a small agreement bonus for repeated lanes but caps it, so a
focused hit cannot be displaced solely because a decoy appeared in more
paraphrase lanes. Model-backed refinement also protects a high-coverage first
pass, while deterministic routing remains recall-first. One-word clause lanes
are skipped as low-signal searches. Missing-entity requests send their compact
lexical lane through token search, preserving rare remembered details without
changing the semantic path for ordinary requests.

A fresh five-sample LM Studio model-first run (20 cases, Gemma 3 4B, seed
20260714, timeout 15 seconds) generated 20/20 queries without fallback and
reached evidence recall@5 1.000, precision@1 0.800, and MRR 0.888. By style,
precision@1 was 0.600 vague, 1.000 specific, 0.800 conditional, and 0.800
missing-entity. Median agent/search latency was 3,437/872 ms; removing
single-word clause lanes reduced p95 agent latency to 3,883 ms in this run.

Use `--query-source lm --base-url URL --model MODEL` to ask LM Studio to write
each request from the raw evidence. Generated requests must quote at least two
evidence terms; timeout, malformed output, or unsupported endpoint responses
fall back per case and are reported in `planner_errors` and `query_origin`.
Use `--model-first` to let the language-search agent perform its own LM Studio
refinement after the generated request. Add `--require-evidence` when a run
should exit nonzero unless recall@k, precision@1, and MRR are all 1.0.

The checked-in [LM trace](benchmark_random_language_lm_sample.json) is a
historical two-sample model-first failure trace retained as a regression
fixture; rerun the command against the live endpoint for current model and
latency measurements.

The checked-in `plan/benchmark_random_language_lm_sample.json` records an
8-case model-first run, including the failed second-round traces and top
document IDs needed to diagnose refinement regressions.
