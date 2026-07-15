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
