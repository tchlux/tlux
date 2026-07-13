# HKM Benchmark Report

- Backend: `drama`
- Python/platform: `3.12.12` / `macOS-26.5.2-arm64-arm-64bit`
- Documents/windows: 76 / 6615
- Source bytes: 2,563,196,370
- Canonical bytes: 109,715,954
- Cache bytes: 0
- Bytes/window: 16585.93
- Probe count: 0 (0 means exhaustive)

## Query evidence

| Query | Recall@k | HKM windows | Exact windows | Work fraction | Warm p50/p95/p99 ms |
|---|---:|---:|---:|---:|---:|
| 0 1 2 3 | 1.000 | 6615 | 6615 | 1.000 | 19.60/19.93/19.93 |
  Quantized recall: float16=1.000, int8=1.000.
  Window ablation recall: drop 1: 1.000, drop 10: 1.000, drop 1024: 1.000, drop 11: 1.000, drop 128: 0.900, drop 13: 1.000, drop 17: 1.000, drop 18: 1.000, drop 2: 1.000, drop 32: 0.100, drop 512: 1.000, drop 9: 1.000.
  Alternate-overlap removal recall: 1: 1.000, 10: 1.000, 1024: 1.000, 11: 1.000, 128: 0.900, 13: 1.000, 17: 1.000, 18: 1.000, 2: 1.000, 32: 0.600, 512: 1.000, 9: 1.000.
| job scheduler | 1.000 | 6615 | 6615 | 1.000 | 19.21/19.23/19.23 |
  Quantized recall: float16=1.000, int8=1.000.
  Window ablation recall: drop 1: 1.000, drop 10: 1.000, drop 1024: 1.000, drop 11: 1.000, drop 128: 0.900, drop 13: 1.000, drop 17: 1.000, drop 18: 1.000, drop 2: 1.000, drop 32: 0.100, drop 512: 1.000, drop 9: 1.000.
  Alternate-overlap removal recall: 1: 1.000, 10: 1.000, 1024: 1.000, 11: 1.000, 128: 1.000, 13: 1.000, 17: 1.000, 18: 1.000, 2: 1.000, 32: 0.400, 512: 1.000, 9: 1.000.
| source index | 1.000 | 6615 | 6615 | 1.000 | 19.34/19.35/19.35 |
  Quantized recall: float16=1.000, int8=1.000.
  Window ablation recall: drop 1: 1.000, drop 10: 1.000, drop 1024: 1.000, drop 11: 1.000, drop 128: 1.000, drop 13: 1.000, drop 17: 1.000, drop 18: 1.000, drop 2: 1.000, drop 32: 0.000, drop 512: 1.000, drop 9: 1.000.
  Alternate-overlap removal recall: 1: 1.000, 10: 1.000, 1024: 1.000, 11: 1.000, 128: 1.000, 13: 1.000, 17: 1.000, 18: 1.000, 2: 1.000, 32: 0.700, 512: 1.000, 9: 1.000.

## Exact checks

- `job scheduler`: `PASS`

## Build evidence

| Stage | Jobs | Total seconds | Max seconds |
|---|---:|---:|---:|
| build_cluster_index | 17 | 11.741 | 2.673 |
| default_worker | 4 | 453.601 | 131.285 |
| finalize_node | 17 | 5.505 | 1.807 |
| route_chunk | 44 | 13.162 | 0.971 |
| run_consolidate | 1 | 0.357 | 0.357 |

## Canonical storage components

| Component | Bytes |
|---|---:|
| centroids | 49,408 |
| embeddings | 46,139,536 |
| metadata_and_manifests | 4,823,614 |
| previews | 32,431,616 |
| token_sketches | 25,554,008 |
| tokens | 717,772 |

Estimated canonical bytes after embedding-only quantization:
- `float16`: 86,769,906
- `int8`: 75,296,882

## Scaling estimate

These rows extrapolate observed storage per embedding window. They do not claim that a laptop has built a billion-document index.

| Windows | Estimated index GiB | Modeled candidate windows | Tree levels |
|---:|---:|---:|---:|
| 1,000 | 0.02 | 1,000 | 1 |
| 10,000 | 0.15 | 10,000 | 2 |
| 100,000 | 1.54 | 100,000 | 3 |
| 1,000,000 | 15.45 | 1,000,000 | 4 |
| 1,000,000,000 | 15446.85 | 1,000,000,000 | 7 |

```mermaid
xychart-beta
    title "Observed-storage extrapolation"
    x-axis [1K, 10K, 100K, 1M, 1B]
    y-axis "GiB" 0 --> auto
    line [0.02, 0.15, 1.54, 15.45, 15446.85]
```

At positive probe counts, candidate work is modeled as the measured candidate fraction held constant while tree depth grows logarithmically with the corpus. This supports a scaling hypothesis, not a claim that the laptop measured a billion-document build.

## Deterministic exhaustive-scan scaling

These points are measured NumPy scans on this machine; they are not HKM query results.

| Vectors | Dimension | Vector bytes | Exact scan ms |
|---:|---:|---:|---:|
| 1,000 | 8 | 32,000 | 0.082 |
| 10,000 | 8 | 320,000 | 0.152 |
| 100,000 | 8 | 3,200,000 | 2.040 |
