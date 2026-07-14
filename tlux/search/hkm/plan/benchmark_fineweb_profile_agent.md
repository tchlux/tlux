# FineWeb profile agent gate

This report records the larger independent corpus gate for the evidence-first
agent path. It complements the 64-file `data/fineweb_sample` gate without
claiming broad production-scale coverage.

## Corpus

- Source: `data/fineweb_profile/docs`
- Files: 512
- Source bytes: 1,359,407
- Source words: about 227,435
- Indexed passages: 785
- Embedder: real `drama`
- Published canonical index: about 1.2 GiB
- Audit: `PASS`
- Build failures: 0

The documents are a heterogeneous web snapshot spanning news, health,
government, education, software, sports, recipes, retail, travel, forums,
fiction, and personal blogs. It is still a small historical sample rather than
a representative enterprise corpus.

## Deterministic grounded gate

Command:

    HKM_EMBEDDER=drama bin/hkm-index /tmp/hkm_fineweb_profile_drama data/fineweb_profile/docs --workers 2 --max-k 8 --leaf-embedding-limit 64 --leaf-doc-limit 32 --max-tokens 2048
    bin/hkm-agent-benchmark /tmp/hkm_fineweb_profile_drama --samples 512 --seed 20260716 --tool-agent --stub --deterministic-first --tool-only --tool-mode token --top-k 5 --require-grounded

Result:

- Samples: 512/512
- Target-document recall@5 / precision@1 / MRR: 1.000 / 1.000 / 1.000
- Content-evidence recall@5 / precision@1 / MRR: 1.000 / 1.000 / 1.000
- Errors: 0
- Fallback rate: 0.002
- Agent latency median/p95: 156.70 / 493.79 ms
- HKM search latency median/p95: 154.68 / 491.61 ms
- Model calls: 0

The gate proves the current deterministic-first tool wrapper on this corpus;
it does not prove model-generation quality or billion-document scaling. A
live LM Studio planner run should be repeated when the server is reachable.
