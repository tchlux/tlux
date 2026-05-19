# HKM Bug Reports

These reports capture concrete bugs found during the 2026-05-19 local
Fourth Wing single-document DRAMA build.

The test corpus was:

- EPUB: `data/Fourth-Wing-by-Rebecca-Yarros.epub`
- Converted corpus: `data/fourth_wing_markdown/Fourth-Wing-by-Rebecca-Yarros.md`
- Intended index root: `data/fourth_wing_hkm_index`
- Embedder: `HKM_EMBEDDER=drama`
- Search mode validation: `token`, `semantic`, and `hybrid`

Reports:

- [Relative index paths can publish worker artifacts under the wrong root](relative-index-root-duplicates-path.md)
- [CLI auto-launched watchers can be orphaned after enqueue](cli-auto-watcher-orphaned.md)
- [`--max-file-bytes 0` skips every non-empty file](max-file-bytes-zero-skips-all.md)
- [Long documents can be silently skipped by the token limit](long-document-token-limit-silent-empty-index.md)
- [A single huge document can publish empty HKM child clusters](single-huge-document-empty-clusters.md)
- [macOS resource samples report zero RSS and CPU](macos-resource-sampler-zero-rss-cpu.md)
