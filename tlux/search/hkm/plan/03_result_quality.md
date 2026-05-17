# Search Result Quality

Status: next.

Goal: make results useful, explainable, and ranked well enough for daily use.

Why it matters:

HKM can currently prove token and semantic search work, but product search needs
better ranking, deduplication, and snippets. Users judge the system by the first
page of results.

Scope:

- Add hybrid ranking that combines semantic distance and exact token evidence.
- Boost filename, heading, symbol, and path matches.
- Deduplicate repeated hits from the same file.
- Return passage-level snippets centered on matched query terms.
- Expose basic match reason fields.
- Support result grouping by source file.

Done when:

- Plain text queries return sensible top results without choosing a mode.
- Repeated passages from the same source do not crowd out the result page.
- Token matches and semantic matches both contribute to ranking.
