# Stable Document Model

Status: done.

Goal: define durable document and passage metadata for heterogeneous search.

Why it matters:

Search quality, previews, filtering, deduplication, and incremental rebuilds all
depend on stable metadata. The current file-level metadata is enough for a
prototype, but not enough for a product over mixed data.

Scope:

- Define a canonical document record.
- Track `source_path`, `source_type`, `file_kind`, title, section path, byte
  offsets, token spans, content hash, build id, and ingest timestamp.
- Preserve enough metadata to rebuild previews without reaching into chunk
  internals.
- Keep metadata serializable and compatible with `.hkmchunk` storage.
- Add tests around metadata round trips and search result fields.

Done when:

- Every hit has stable source and passage identity.
- The same document can be recognized across rebuilds by content hash.
- Search result previews can be built from canonical metadata.

Completed:

- Hits now include a canonical document record with source identity, source
  type, file kind, title, byte spans, token spans, content hash, build id,
  ingest timestamp, optional source manifest fields, byte counts, and stored
  preview text.
- Default ingestion enriches document records from FineWeb-style
  `manifest.jsonl` files beside the docs root or its parent, with an explicit
  `--source-manifest` override.
- Preview generation falls back to stored document preview text when source
  files are unavailable.
