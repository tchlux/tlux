# Query API

Status: next.

Goal: provide one stable query surface for CLI, TUI, and future UIs.

Why it matters:

The current query shape is close, but users should not need to know internals.
The product API should support simple default search while still exposing
explicit modes and filters.

Scope:

- Document and stabilize the existing plain-text hybrid search default.
- Keep explicit `token`, `semantic`, and `hybrid` modes.
- Add path and file-kind filters.
- Add pagination or cursor support.
- Stabilize the JSON request and response schema.
- Keep CLI and TUI using the same `Searcher` API.

Done when:

- `hkm-search INDEX query.json` works with a documented stable schema.
- UIs can render results without reading chunk internals.
- Query validation errors are explicit and actionable.
