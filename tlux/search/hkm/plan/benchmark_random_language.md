# Random language-query gate

`bin/hkm-random-language-benchmark` samples active indexed documents, builds
memory-style requests from each raw passage, and runs the persistent language
agent. Four styles are included: vague, specific, conditional, and
missing-entity. The last style deliberately drops the first content term and
retains three later clues to model forgetting a subject or object. A fixed
seed makes every query and target reproducible.

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
reached evidence recall@5 0.850, precision@1 0.600, and MRR 0.708. By style,
evidence recall was 0.900 vague, 0.800 specific, 0.700 conditional, and 1.000
missing-entity; missing-entity precision@1 was 0.800. Median agent/search
latency was 707/621 ms.
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

A current two-sample LM Studio model-first run (8 cases, Gemma 3 4B, seed
20260714, timeout 15 seconds) reached evidence recall@5, precision@1, and MRR
of 1.000/1.000/1.000. Six requests used model-generated queries; two invalid
missing-entity generations used deterministic fallback and still ranked the
evidence first. Median agent/search latency was 657/381 ms.

The heterogeneous FineWeb profile index at `/tmp/hkm_fineweb_profile_drama`
now passes an eight-sample model-first slice (32 cases) at 1.000 recall@5,
precision@1, and MRR across all four styles. Gemma 3 generated 31/32 requests;
one invalid missing-entity generation used the evidence-checked deterministic
rescue. Median agent/search latency was 5.94/2.74 s (p95 6.34/2.96 s).
Its deterministic-only slice remains a useful hard gate at 0.906/0.750/0.807,
where generic terms such as `authors`, `reading`, and `Update` expose the next
rarity-stratified challenge.

The current model-first language agent stops after a confident first semantic
hit, caps model refinement at four lanes, and reserves a compact clue lane for
missing-entity requests. A live eight-case Fourth Wing smoke (two sampled
passages, Gemma 3, seed 20260714) reached 1.000 recall@5, precision@1, and MRR;
both rejected missing-entity generations recovered rank-one evidence.

Use `--query-source lm --base-url URL --model MODEL` to ask LM Studio to write
each request from the raw evidence. Generated requests must quote at least two
evidence terms. Missing-entity requests must also omit the first distinctive
sampled clue; otherwise they are reported as planner failures and use
deterministic fallback. Timeout, malformed output, or unsupported endpoint
responses fall back per case and are reported in `planner_errors` and
`query_origin`.
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
