"""Evaluate a local query-planning agent against sampled HKM passages.

The benchmark uses an OpenAI-compatible LM Studio endpoint when available and
falls back to a deterministic query planner for offline regression tests. It
reports model first-pass quality separately from the final evidence-assisted
tool result so a perfect fallback cannot hide a weak model query.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol

from ..search.searcher import Searcher


STOP_WORDS = {
    "about", "after", "again", "also", "because", "before", "being", "could",
    "every", "first", "from", "have", "into", "just", "more", "other", "over",
    "said", "some", "than", "that", "their", "there", "these", "they", "this",
    "through", "under", "what", "when", "where", "which", "while", "with", "would",
}


# One sampled raw passage and its exact indexed document target.
@dataclass
class Sample:
    sample_id: int
    doc_id: int
    source_path: str
    excerpt: str
    token_start: int
    token_end: int


# Query planner interface shared by LM Studio and deterministic tests.
class QueryGenerator(Protocol):
    name: str

    def generate(self, excerpt: str) -> str:
        ...


# Return a compact phrase from evidence for exact fallback retrieval.
def _phrase_query(excerpt: str, limit: int = 8) -> str:
    if len(excerpt.split()) <= limit:
        return excerpt.strip()
    return " ".join(excerpt.split()[:limit])


# Return several short raw phrases for evidence-assisted exact retrieval.
def _phrase_queries(excerpt: str, limit: int = 8, count: int = 6) -> List[str]:
    words = excerpt.split()
    if len(words) <= limit:
        return [excerpt.strip()]
    starts = list(dict.fromkeys([0, len(words) // 4, len(words) // 2, (3 * len(words)) // 4, len(words) - limit]))
    return [" ".join(words[start:start + limit]) for start in starts[:count]]


# Return bounded fallback queries, including distinctive code identifiers.
def _fallback_queries(excerpt: str) -> List[str]:
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_'-]{3,}", excerpt)
    distinctive = list(dict.fromkeys(
        word for word in identifiers
        if "_" in word or any(char.isdigit() for char in word) or len(word) >= 12
    ))
    queries = _phrase_queries(excerpt) + [_keyword_query(excerpt)] + distinctive[:8]
    return list(dict.fromkeys(query for query in queries if query.strip()))


# Return distinctive terms for a deterministic offline query planner.
def _keyword_query(excerpt: str, limit: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_'-]*", excerpt)
    unique = list(dict.fromkeys(word for word in words if len(word) >= 4 and word.lower() not in STOP_WORDS))
    ranked = sorted(
        enumerate(unique),
        key=lambda item: (
            0 if "_" in item[1] or any(char.isdigit() for char in item[1]) else 1,
            -len(item[1]),
            item[0],
        ),
    )
    return " ".join(word for _, word in ranked[:limit] or list(enumerate(words[:limit])))


# Parse a model response into one bounded search query.
def parse_query(response: str) -> str:
    text = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            text = str(value.get("query", ""))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                text = str(value.get("query", ""))
            except json.JSONDecodeError:
                pass
    text = " ".join(text.split())
    if not text:
        raise ValueError("query planner returned an empty query")
    return text[:256]


# Call the local OpenAI-compatible LM Studio server.
class LMStudioQueryGenerator:
    name = "lmstudio"

    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model: str | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request(self, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _model_name(self) -> str:
        if self.model:
            return self.model
        models = self._request("models").get("data", [])
        if not models:
            raise RuntimeError("LM Studio has no model available at the configured endpoint")
        self.model = str(models[0]["id"])
        return self.model

    def generate(self, excerpt: str) -> str:
        prompt = (
            "You plan one search query for a retrieval tool. Read the evidence passage and "
            "return JSON only as {\"query\": \"...\"}. Use 3-8 concrete words that would "
            "retrieve this passage. Do not answer the passage or mention these instructions.\n\n"
            f"Evidence passage:\n{excerpt}"
        )
        response = self._request("chat/completions", {
            "model": self._model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
        })
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("LM Studio returned no chat completion choices")
        content = choices[0].get("message", {}).get("content", "")
        return parse_query(str(content))


# Deterministic planner used for offline tests and endpoint recovery.
class StubQueryGenerator:
    name = "deterministic_stub"

    def generate(self, excerpt: str) -> str:
        return _keyword_query(excerpt)


# Sample raw text from actual active documents in the index.
def sample_passages(searcher: Searcher, count: int, seed: int, max_tokens: int = 48) -> List[Sample]:
    if count < 1:
        raise ValueError("count must be positive")
    doc_ids = sorted(searcher._active_doc_ids())
    if not doc_ids:
        raise ValueError("index contains no active documents")
    rng = random.Random(seed)
    selected = rng.sample(doc_ids, min(count, len(doc_ids)))
    backend = searcher._backend()
    samples = []
    for sample_id, doc_id in enumerate(selected):
        tokens, metadata = searcher._doc_context(doc_id)
        if tokens.size == 0:
            continue
        document = searcher._document_record(doc_id, metadata, tokens)
        source_file = Path(searcher.source_root) / document.source_path
        raw_text = ""
        if source_file.exists():
            raw = source_file.read_bytes()
            raw = raw[document.byte_start:document.byte_end or len(raw)]
            raw_text = raw.decode("utf-8", errors="ignore")
        words = raw_text.split()
        token_start = 0
        token_end = int(tokens.shape[0])
        if words and len(words) <= max_tokens:
            excerpt = raw_text.strip()
        elif words:
            width = min(max_tokens, len(words))
            word_start = rng.randrange(max(1, len(words) - width + 1))
            excerpt = " ".join(words[word_start:word_start + width])
        else:
            width = min(max_tokens, int(tokens.shape[0]))
            start = rng.randrange(max(1, int(tokens.shape[0]) - width + 1))
            end = min(int(tokens.shape[0]), start + width)
            excerpt = " ".join(backend.detokenize([tokens[start:end].tolist()])[0].split())
            token_start, token_end = start, end
        if not excerpt:
            continue
        samples.append(Sample(sample_id, doc_id, document.source_path, excerpt, token_start, token_end))
    if not samples:
        raise ValueError("sampled documents contained no readable text")
    return samples


# Return a target rank or None when the target is absent.
def _target_rank(result: Any, doc_id: int) -> int | None:
    for rank, hit in enumerate(result.docs, 1):
        if int(hit.doc_id) == int(doc_id):
            return rank
    return None


# Re-rank returned snippets against the raw evidence held by the agent.
def _rerank_with_evidence(result: Any, excerpt: str, searcher: Searcher | None = None) -> Any:
    terms = set(re.findall(r"[A-Za-z0-9]+", excerpt.lower()))
    if not terms:
        return result

    def rank_key(hit: Any) -> tuple[float, float, float, int]:
        text = f"{hit.preview_text} {getattr(hit.document, 'document_preview', '')}".lower()
        if searcher is not None and hit.document.source_path:
            source = Path(searcher.source_root) / hit.document.source_path
            if source.exists():
                raw = source.read_bytes()
                raw = raw[hit.document.byte_start:hit.document.byte_end or len(raw)]
                text += " " + raw.decode("utf-8", errors="ignore").lower()
        overlap = len(terms.intersection(re.findall(r"[A-Za-z0-9]+", text))) / len(terms)
        return (-overlap, -float(hit.score), -float(getattr(hit, "token_score", 0.0)), int(hit.doc_id))

    result.docs = sorted(result.docs, key=rank_key)
    return result


# Merge candidate result pages while preserving the strongest lexical score.
def _merge_results(results: List[Any], top_k: int, excerpt: str = "") -> Any:
    base = results[0]
    hits: Dict[tuple[int, tuple[int, int]], Any] = {}
    for result in results:
        for hit in result.docs:
            key = (int(hit.doc_id), tuple(hit.span))
            if key not in hits or float(hit.score) > float(hits[key].score):
                hits[key] = hit
    terms = set(re.findall(r"[A-Za-z0-9]+", excerpt.lower()))

    def rank_key(hit: Any) -> tuple[float, float, str, int, tuple[int, int]]:
        text = f"{hit.preview_text} {getattr(hit.document, 'document_preview', '')}".lower()
        overlap = len(terms.intersection(re.findall(r"[A-Za-z0-9]+", text))) / len(terms) if terms else 0.0
        return (-float(hit.score), -overlap, hit.source_path, hit.doc_id, hit.span)

    base.docs = sorted(hits.values(), key=rank_key)[:top_k]
    return base


# Search one query and measure only the HKM call.
def _search(searcher: Searcher, query: str, top_k: int, probe_count: int, mode: str = "hybrid") -> tuple[Any, float]:
    started = time.perf_counter()
    result = searcher.search({
        "mode": mode,
        "text": query,
        "top_k": top_k,
        "probe_count": probe_count,
    })
    return result, (time.perf_counter() - started) * 1000.0


# Aggregate ranks into precision and recall metrics.
def _metrics(rows: Iterable[Dict[str, Any]], key: str = "final_rank") -> Dict[str, float]:
    values = list(rows)
    ranks = [row.get(key) for row in values]
    found = [rank is not None for rank in ranks]
    reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]
    top_one = [rank == 1 for rank in ranks]
    return {
        "recall_at_k": sum(found) / len(found) if found else 0.0,
        "precision_at_1": sum(top_one) / len(top_one) if top_one else 0.0,
        "mrr": sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
    }


# Evaluate model queries, evidence fallback, and probe budgets.
def evaluate_agent(
    index_root: str,
    generator: QueryGenerator,
    samples: int = 8,
    seed: int = 42,
    top_k: int = 10,
    max_tokens: int = 48,
    probe_counts: List[int] | None = None,
) -> Dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    probe_counts = probe_counts or [0, 1, 2, 4]
    if any(probe < 0 for probe in probe_counts):
        raise ValueError("probe counts must be non-negative")
    searcher = Searcher.from_index_root(index_root)
    sampled = sample_passages(searcher, samples, seed, max_tokens)
    rows = []
    probe_rows: Dict[int, List[Dict[str, Any]]] = {probe: [] for probe in probe_counts}
    planner_errors = []
    for sample in sampled:
        planner_started = time.perf_counter()
        try:
            query = generator.generate(sample.excerpt)
        except Exception as exc:
            planner_errors.append({"sample_id": sample.sample_id, "error": str(exc)})
            query = _keyword_query(sample.excerpt)
        planner_ms = (time.perf_counter() - planner_started) * 1000.0
        first, first_ms = _search(searcher, query, top_k, 0)
        first_rank = _target_rank(first, sample.doc_id)
        fallback_query = ""
        fallback_ms = 0.0
        final = first if first_rank == 1 else _rerank_with_evidence(first, sample.excerpt, searcher)
        final_rank = _target_rank(final, sample.doc_id)
        if final_rank != 1:
            phrase_results = []
            for phrase in _fallback_queries(sample.excerpt):
                fallback_query = phrase
                fallback, elapsed = _search(searcher, phrase, top_k, 0, mode="token")
                phrase_results.append(fallback)
                fallback_ms += elapsed
            fallback = _merge_results(phrase_results, top_k, sample.excerpt) if phrase_results else final
            fallback = _rerank_with_evidence(fallback, sample.excerpt, searcher)
            fallback_rank = _target_rank(fallback, sample.doc_id)
            if fallback_rank is not None and (final_rank is None or fallback_rank < final_rank):
                final = fallback
        final_rank = _target_rank(final, sample.doc_id)
        rows.append({
            "sample_id": sample.sample_id,
            "doc_id": sample.doc_id,
            "source_path": sample.source_path,
            "excerpt": sample.excerpt,
            "query": query,
            "first_rank": first_rank,
            "final_rank": final_rank,
            "fallback_query": fallback_query,
            "planner_ms": planner_ms,
            "first_search_ms": first_ms,
            "fallback_search_ms": fallback_ms,
        })
        for probe in probe_counts:
            result, elapsed = _search(searcher, query, top_k, probe)
            probe_rows[probe].append({
                "rank": _target_rank(result, sample.doc_id),
                "latency_ms": elapsed,
            })
    curve = []
    for probe, values in probe_rows.items():
        ranks = [value["rank"] for value in values]
        curve.append({
            "probe_count": probe,
            "recall_at_k": sum(rank is not None for rank in ranks) / len(ranks) if ranks else 0.0,
            "precision_at_1": sum(rank == 1 for rank in ranks) / len(ranks) if ranks else 0.0,
            "median_search_ms": statistics.median(value["latency_ms"] for value in values) if values else 0.0,
        })
    return {
        "index_root": str(Path(index_root).expanduser().absolute()),
        "generator": generator.name,
        "samples": len(rows),
        "seed": seed,
        "top_k": top_k,
        "planner_errors": planner_errors,
        "first_pass": _metrics(rows, "first_rank"),
        "final": _metrics(rows, "final_rank"),
        "fallback_rate": sum(bool(row["fallback_query"]) for row in rows) / len(rows) if rows else 0.0,
        "probe_curve": curve,
        "rows": rows,
    }


# Render a compact agent-quality report.
def render_report(report: Dict[str, Any]) -> str:
    first = report["first_pass"]
    final = report["final"]
    lines = [
        "# HKM Agent Search Benchmark",
        "",
        f"- Generator: `{report['generator']}`",
        f"- Samples/seed/top-k: {report['samples']} / {report['seed']} / {report['top_k']}",
        f"- Model first-pass recall@k: `{first['recall_at_k']:.3f}`; precision@1: `{first['precision_at_1']:.3f}`; MRR: `{first['mrr']:.3f}`",
        f"- Final evidence-assisted recall@k: `{final['recall_at_k']:.3f}`; precision@1: `{final['precision_at_1']:.3f}`; MRR: `{final['mrr']:.3f}`",
        f"- Fallback rate: `{report['fallback_rate']:.3f}`",
        "",
        "## Probe curve (model query only)",
        "",
        "| Probe count | Recall@k | Precision@1 | Median search ms |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["probe_curve"]:
        lines.append(
            f"| {row['probe_count']} | {row['recall_at_k']:.3f} | "
            f"{row['precision_at_1']:.3f} | {row['median_search_ms']:.2f} |"
        )
    lines.extend(["", "## Samples", "", "| ID | Target doc | Query | First rank | Final rank | Fallback |", "|---:|---:|---|---:|---:|---|"])
    for row in report["rows"]:
        lines.append(
            f"| {row['sample_id']} | {row['doc_id']} | {row['query'].replace('|', ' ')} | "
            f"{row['first_rank'] or '-'} | {row['final_rank'] or '-'} | "
            f"{'yes' if row['fallback_query'] else 'no'} |"
        )
    return "\n".join(lines)


# Parse CLI arguments and write machine/human-readable evidence.
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LM Studio query agent against HKM samples.")
    parser.add_argument("index_root")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--probes", default="0,1,2,4")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--stub", action="store_true", help="Use the deterministic planner without LM Studio")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()
    generator: QueryGenerator = StubQueryGenerator() if args.stub else LMStudioQueryGenerator(args.base_url, args.model, args.timeout)
    if not args.stub:
        try:
            generator._model_name()  # type: ignore[attr-defined]
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            print(f"LM Studio unavailable; using deterministic fallback: {exc}")
            generator = StubQueryGenerator()
    report = evaluate_agent(
        args.index_root,
        generator,
        args.samples,
        args.seed,
        args.top_k,
        args.max_tokens,
        [int(value) for value in args.probes.split(",") if value.strip()],
    )
    markdown = render_report(report)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.report_output:
        Path(args.report_output).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":  # pragma: no cover
    main()
