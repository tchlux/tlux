# Expanded local FineWeb benchmark

This benchmark records the larger local corpus used for the next language-search
quality pass. The source files are intentionally ignored by Git; this plan keeps
their provenance, build command, and measured baseline reproducible.

## Corpus

- Source: `data/fineweb_expanded/docs`
- Documents: 1,024 plain-text files
- Size: 6.5 MB and about 646,882 words
- Domains: 1,013 unique domains
- Overlap with `data/fineweb_profile`: 0 URLs
- Provenance: FineWeb `sample-10BT/train`, shard `000_00000.parquet`, rows
  100,000 through 101,023, extracted with bounded HTTP range reads

## Index

```text
HKM_EMBEDDER=drama bin/hkm-index /private/tmp/hkm_fineweb_expanded_drama \
  data/fineweb_expanded/docs --workers 1 --max-k 8 \
  --leaf-embedding-limit 512 --leaf-doc-limit 64 --max-tokens 2048
bin/hkm-audit /private/tmp/hkm_fineweb_expanded_drama
```

The audited index contains 2,108 passages and 63,588 embedding windows. Its
canonical generation is about 1.90 GB; this storage amplification is itself a
performance target for later work.

## Baseline

The deterministic 40-case language run (`--samples 10 --seed 20260715`) reports
recall@5 1.000, precision@1 0.975, and MRR 0.9875. Median agent/search latency
is 2.37/2.17 seconds, with 3.80/3.62 seconds at p95. Missing-entity precision@1
is 0.900; the other three styles are perfect in this slice.

A live LM Studio diagnostic (`--samples 1 --seed 20260719 --model-first
--always-refine --timeout 6`) generated all four queries through Gemma 3 with
no fallback and retained 1.000 recall, precision@1, and MRR. Median
agent/search latency was 5.62/2.65 seconds. The six-second timeout is diagnostic;
the normal one-second budget is expected to fall back on this larger corpus.

The next suite should add fixed challenging cases with distractors, reordered
clues, multi-condition logic, omitted entities, and adversarially generic terms,
then measure both deterministic recovery and genuine LM refinement.

## Deep fixed suite

`plan/benchmark_fineweb_expanded_deep.md` contains 24 hand-authored queries
across vehicles, politics, medicine, security, travel, finance, arts, and
fiction. They deliberately vary clue order, use nested conditionals and
negation, and omit a remembered subject or object. Stable text-pattern
judgements live in the adjacent JSON sidecar so rebuilt document IDs do not
become the oracle.

The first deterministic run on the audited index passed all 24 cases with full
group coverage and 43/43 judged top-five hits relevant. The aliases intentionally
include normalized spellings and passage-window variants (for example
`Aedesaegypti` and `monthly payment`) so the oracle tests meaning rather than a
single exact token. Median agent/search latency was 5.37/5.16 seconds (p95
5.76/5.50 seconds). Re-run it with:

```text
bin/hkm-language-benchmark /private/tmp/hkm_fineweb_expanded_drama \
  --benchmark plan/benchmark_fineweb_expanded_deep.md \
  --judgements plan/benchmark_fineweb_expanded_deep_judgements.json \
  --tool-mode semantic --top-k 5 --timeout 15 \
  --require-gate --require-precision
```

The long pytest gate is opt-in because the local index is ignored and takes
about two minutes to load and search. Set `HKM_EXPANDED_INDEX` and run
`bin/hkm-python -m pytest tests/test_fineweb_expanded_gate.py -q`; its process
timeout defaults to 900 seconds and can be raised with
`HKM_EXPANDED_TEST_TIMEOUT`. To exercise the LM Studio planner as well, set
`HKM_EXPANDED_MODEL` and optionally `HKM_EXPANDED_BASE_URL` and
`HKM_EXPANDED_LM_TIMEOUT` (the latter defaults to 15 seconds). The normal
one-second LM default remains unchanged for product behavior.

The companion `benchmark_fineweb_expanded_stress.md` contains 25 harder cases
from additional domains. Its current deterministic floor is 13/25 full-group
passes and 25/25 coherent top-five results; the twelve misses are retained as
regression targets instead of being hidden by permissive aliases. Run that
diagnostic with `HKM_RUN_EXPANDED_STRESS=1` alongside `HKM_EXPANDED_INDEX`.
The current misses are LQ-52, LQ-53, LQ-54, LQ-56, LQ-62, LQ-63, LQ-65,
LQ-66, LQ-70, LQ-72, LQ-73, and LQ-74; each still returns a coherent passage,
so they isolate multi-clue coverage rather than total retrieval failure.
