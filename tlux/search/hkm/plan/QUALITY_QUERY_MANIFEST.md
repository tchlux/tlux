# Quality query manifest

`tools/query_manifest.py` accepts one JSON object per non-empty JSONL line.
Every record must contain:

```json
{
  "query_id": "q-001",
  "text": "Which source explains the storage tradeoff?",
  "natural": true,
  "source_blind": true,
  "category": "conceptual",
  "required_facets": ["mechanism", "tradeoff"],
  "answerable": true,
  "relevance": {"doc-a": 3, "doc-b": 1}
}
```

The loader rejects duplicate IDs, missing provenance labels, malformed grades,
negative grades, and disagreement between `answerable` and positive qrels.
Use `answerable: false` with no positive grade for no-answer cases. A
`source_blind: false` record is allowed only as an explicitly labeled
diagnostic case and must not be treated as a blind holdout.

Use `evaluate_query_manifest` to run any retriever against the validated cases.
The returned report includes manifest composition beside the normal graded
retrieval metrics and judged/unjudged accounting.
