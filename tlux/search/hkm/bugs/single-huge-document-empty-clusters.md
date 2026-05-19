# A Single Huge Document Can Publish Empty HKM Child Clusters

Status: open
Severity: critical

## Summary

With one source document and 20,915 embedding windows, the default
`leaf_embedding_limit=1024` caused the recursive builder to publish the HKM root
as a non-leaf with eight child cluster names. The child directories existed, but
they did not contain `node.json`. All search modes then failed with
`FileNotFoundError`.

## Reproduction

Build a one-file corpus with a high token limit but default leaf embedding
limit:

```bash
env HF_HUB_OFFLINE=1 HKM_EMBEDDER=drama HKM_MAX_WORKERS=4 HKM_DISABLE_WATCHER_LAUNCH=1 \
  bin/hkm-index /abs/index /abs/fourth_wing_markdown \
  --workers 4 --full-rebuild --include '*.md' \
  --max-file-bytes 20000000 --max-tokens 1000000
```

Then drain jobs and inspect:

```bash
find data/fourth_wing_hkm_index/hkm -maxdepth 2 -print
```

Observed:

- `hkm/node.json` existed.
- `hkm/node.json` had `"is_leaf": false` and children `cluster_0000` through
  `cluster_0007`.
- The child directories existed.
- No child directory had `node.json`.

Search failures:

```text
FileNotFoundError: .../hkm/cluster_0000/node.json
FileNotFoundError: .../hkm/cluster_0004/node.json
```

## Expected

The builder should never publish a `node.json` that references child nodes
without valid child manifests. For one source document, it should probably stay
a leaf even if it has many embedding windows, or it should split into valid
passage-level child nodes that the searcher can traverse.

## Actual

The root advertised child clusters that were not valid HKM nodes. Token,
semantic, and hybrid search all crashed before returning results.

## Impact

This is a correctness blocker for large single-document corpora. The job graph
reported success, but the built index was not queryable.

## Workaround Used

Rebuild from the cached embeddings with a leaf embedding limit above the
observed window count:

```bash
--leaf-embedding-limit 50000
```

That produced a searchable root leaf:

```text
doc_count=1
embedding_count=20915
is_leaf=True
chunk_roots=['../docs']
```

## Likely Fix Area

In the recursive builder, prevent splitting when a node contains only one active
document unless the implementation can publish valid passage-level children.
Add an index publication audit that walks every `node.json`, verifies every
referenced child node exists, and runs before the root build job succeeds.
