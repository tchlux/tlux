# Incremental Builds

Status: done.

Goal: avoid full rebuilds when only part of a dataset changed.

Why it matters:

Large datasets make full rebuilds expensive. A useful search product must reuse
unchanged token and embedding work, detect deletions, and update the index with
bounded effort.

Scope:

- Store source snapshot metadata in the index manifest.
- Hash file contents during ingestion.
- Reuse unchanged document shards.
- Detect new, changed, and deleted files.
- Rebuild affected HKM branches.
- Keep full rebuild as the simple fallback path.

Done when:

- Re-running an index over mostly unchanged data skips unchanged files.
- Changed and deleted files are reflected in search results.
- Incremental behavior is tested against a small synthetic corpus.

Completed:

- Build workers persist per-document token and embedding artifacts under
  `.hkm_cache/embeddings`, keyed by backend/window settings and content hash.
- Full rebuilds and recovery builds reuse cached embeddings instead of
  recomputing unchanged source content.
- Builds write `manifests/source_snapshot.json` and use it to classify reused,
  new, changed, and deleted files.
- Incremental builds append new and changed documents through the existing HKM
  tree, update active `doc_index.npy`, and keep stale chunks out of results by
  filtering every search path through active document membership.
- Oversized touched leaves are split locally; full rebuild remains the intended
  way to rebalance the full tree.
- `hkm-index --full-rebuild` forces the old full rebuild path while preserving
  the embedding cache.
