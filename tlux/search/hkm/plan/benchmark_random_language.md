# Random language-query gate

`bin/hkm-random-language-benchmark` samples active indexed documents, builds
memory-style requests from each raw passage, and runs the persistent language
agent. Four styles are included: vague, specific, conditional, and
missing-entity. The last style deliberately drops the first content term and
retains four later clues to model forgetting a subject or object. A fixed
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

On 20 Fourth Wing samples (80 cases), the deterministic generator reaches
evidence recall@5 1.000, precision@1 0.9625, and MRR 0.9813. By style, evidence
recall is 1.000 for every style; precision@1 is 0.950 vague, 1.000 specific,
0.950 conditional, and 0.950 missing-entity. Median agent/search latency is
649/430 ms (p95 955/727 ms). The generator normalizes contractions, drops
common dialogue glue, keeps four grounded clues for ordinary styles, and adds
a bounded lexical rescue page; missing-entity requests omit the first clue and
the LM prompt redacts it as `[unknown]`.
This remains a challenge baseline, not a production quality claim; the
conditional and forgotten-entity styles still need larger quality gates.
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

A latest two-sample LM Studio model-first run (8 cases, Gemma 3 4B, seed
20260714, timeout 15 seconds) reached evidence recall@5/precision@1/MRR of
1.000/0.875/0.938. Six requests used model-generated queries; two invalid
missing-entity generations used deterministic fallback and still recovered the
target evidence. Median agent/search latency was 2,266/599 ms; LM Studio load
causes substantial warm-run variance.

The heterogeneous FineWeb profile index at `/tmp/hkm_fineweb_profile_drama`
now passes an eight-sample model-first slice (32 cases) at 1.000 recall@5,
precision@1, and MRR across all four styles. Gemma 3 generated 31/32 requests;
one invalid missing-entity generation used the evidence-checked deterministic
rescue. Median agent/search latency was 5.94/2.74 s (p95 6.34/2.96 s).
Its deterministic-only slice remains a useful hard gate at 0.906/0.750/0.807,
where generic terms such as `authors`, `reading`, and `Update` expose the next
rarity-stratified challenge.

The current adaptive model-first language agent stops after a confident first
semantic hit, caps refinement at four lanes, and reserves a compact clue lane
for missing-entity requests. `--always-refine` forces one inspection/refinement
round for every request; this is useful for auditing the agentic trace but adds
model latency and is not the default.

The checked-in FineWeb challenge fixture uses 10 manually reviewed cases over
the 512-file profile index. It passes 10/10 coverage/coherence and 1.000 judged
precision over 20 labels with the deterministic agent; use it as the stable
heterogeneous regression gate while expanding the broader random sample.

The extended FineWeb fixture in
`plan/benchmark_fineweb_language_challenges_extended.md` adds 20 unrelated
documents and domains with harder conditionals and forgotten entities. It
passes 20/20 coverage/coherence and 1.000 judged precision over 43 labeled
hits at top-k 5 (2.81/2.60 seconds median agent/search, 3.58/3.36 seconds
p95).

Use `--query-source lm --base-url URL --model MODEL` to ask LM Studio to write
each request from the raw evidence. Generated requests must quote at least two
evidence terms. Missing-entity requests must also omit the first distinctive
sampled clue; otherwise they are reported as planner failures and use
deterministic fallback. Timeout, malformed output, or unsupported endpoint
responses fall back per case and are reported in `planner_errors` and
`query_origin`.
Use `--model-first` to let the language-search agent perform its own LM Studio
refinement after the generated request. Add `--always-refine` to force the
inspect/refine round even after a confident first result, and use
`--timeout 1.5` or higher so the trace measures genuine LM planning rather than
the one-second deterministic rescue budget. Add `--require-evidence` when a
run should exit nonzero unless recall@k, precision@1, and MRR are all 1.0.

A live Gemma 3 run over two random Fourth Wing passages (eight cases) with
`--query-source lm --model-first --always-refine --timeout 1.5` generated all
eight first memory queries, completed one planner call per case, recorded
alternative lanes and antipatterns, and returned rank-one evidence for every
case. Median agent/search latency was 2.46/0.84 seconds (p95 3.42/1.82 s).
This is the strongest end-to-end raw-passage autonomy trace; the broader
quality gate remains open because model-generated lanes are stochastic.

The checked-in [LM trace](benchmark_random_language_lm_sample.json) is a
historical two-sample model-first failure trace retained as a regression
fixture; rerun the command against the live endpoint for current model and
latency measurements.

The checked-in `plan/benchmark_random_language_lm_sample.json` records an
8-case model-first run, including the failed second-round traces and top
document IDs needed to diagnose refinement regressions.
