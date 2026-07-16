"""Run a persistent grounded HKM agent over JSONL input.

The process keeps one index and one LM Studio client alive so model loading and
index initialization are amortized across requests.

Example:
    echo '{"text":"raw passage"}' | hkm-agent INDEX --warmup
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, TextIO

from ..search.searcher import Searcher
from .agent_benchmark import (
    DeterministicToolAgent,
    LanguageSearchAgent,
    LMSTUDIO_TIMEOUT,
    LMSTUDIO_WARMUP_TIMEOUT,
    LMStudioPlannerToolAgent,
    LMStudioQueryGenerator,
)


# Read a bounded source window so agents can inspect evidence beyond the index preview.
#
# Arguments:
#   hit (Any): Search hit containing source path and byte bounds.
#   source_root (str): Root directory allowed for source reads.
#   limit (int): Maximum UTF-8 payload size in bytes.
#
# Returns:
#   (str): Bounded decoded source context, or an empty string.
#
def _source_context(hit: Any, source_root: str, limit: int = 2048) -> str:
    source_path = str(getattr(hit, "source_path", "") or "")
    document = getattr(hit, "document", None)
    if not source_path or not source_root or document is None:
        return ""
    try:
        root = Path(source_root).resolve()
        path = (root / source_path).resolve()
        path.relative_to(root)
        start = max(0, int(getattr(document, "byte_start", 0)))
        end = int(getattr(document, "byte_end", 0))
        with path.open("rb") as source:
            if end <= start:
                source.seek(start)
                raw = source.read(limit)
            elif end - start <= limit:
                source.seek(start)
                raw = source.read(end - start)
            else:
                marker = b"\n...\n"
                width = max(1, (limit - len(marker)) // 2)
                source.seek(start)
                head = source.read(width)
                source.seek(max(start, end - width))
                raw = head + marker + source.read(width)
        return raw.decode("utf-8", errors="ignore").strip()
    except (OSError, ValueError):
        return ""


# Keep compact stable fields plus bounded source evidence for downstream agents.
#
# Arguments:
#   hit (Any): Search hit to serialize.
#   source_root (str): Root directory used for bounded source evidence.
#
# Returns:
#   (Dict[str, Any]): JSON-compatible hit payload.
#
def _hit_payload(hit: Any, source_root: str = "") -> Dict[str, Any]:
    context = _source_context(hit, source_root)
    return {
        "doc_id": int(hit.doc_id),
        "source_path": hit.source_path,
        "score": float(hit.score),
        "preview_text": hit.preview_text,
        "document": {
            "document_preview": hit.document.document_preview,
            "source_context": context,
        },
    }


# Persistent local agent that reuses one searcher and one model client.
class LocalSearchAgent:
    def __init__(
        self,
        index_root: str,
        base_url: str = "http://127.0.0.1:4321/v1",
        model: str | None = None,
        timeout: float = LMSTUDIO_TIMEOUT,
        mode: str = "token",
        deterministic_first: bool = False,
        runner: Any | None = None,
        native_tool: bool = False,
        language_query: bool = False,
        always_refine: bool = False,
        language_rounds: int = 2,
    ) -> None:
        self.searcher = Searcher.from_index_root(index_root)
        self.language_query = language_query
        self.deterministic = DeterministicToolAgent(mode) if deterministic_first else None
        if runner is not None:
            self.client = None
            self.runner = runner
        elif language_query and deterministic_first:
            self.client = None
            self.runner = LanguageSearchAgent(
                None, mode, max_rounds=language_rounds, always_refine=always_refine
            )
        else:
            self.client = LMStudioQueryGenerator(base_url, model, timeout)
            self.runner = (
                LanguageSearchAgent(
                    self.client,
                    mode,
                    max_rounds=language_rounds,
                    always_refine=always_refine,
                )
                if language_query
                else LMStudioPlannerToolAgent(self.client, mode, native_tool)
            )

    # Warm HKM search and the structured planner with representative requests.
    def warmup(self) -> bool:
        search_ready = True
        try:
            self.searcher.search({"mode": "token", "text": "warmup", "top_k": 1})
        except Exception:
            search_ready = False
        if self.client is None:
            return search_ready
        previous_timeout = getattr(self.client, "timeout", None)
        try:
            if previous_timeout is not None:
                self.client.timeout = max(float(previous_timeout), LMSTUDIO_WARMUP_TIMEOUT)
            if self.language_query:
                self.client.plan_language_query("warmup language query", "[]")
            else:
                self.client.generate("warmup evidence token")
            return search_ready
        except Exception:
            return False
        finally:
            if previous_timeout is not None:
                self.client.timeout = previous_timeout

    # Run one raw passage through the grounded search tool.
    def run(self, excerpt: str, top_k: int = 10) -> Dict[str, Any]:
        if not excerpt.strip():
            raise ValueError("text must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.language_query:
            return self._serialize(self.runner.run(excerpt, self.searcher, top_k))
        if self.deterministic is not None:
            cheap = self.deterministic.run(excerpt, self.searcher, top_k)
            if cheap["result"].docs:
                return self._serialize(cheap)
        return self._serialize(self.runner.run(excerpt, self.searcher, top_k))

    # Convert the internal result to the stable JSONL response contract.
    def _serialize(self, run: Dict[str, Any]) -> Dict[str, Any]:
        result = run["result"]
        payload = {
            "query": run.get("query", ""),
            "grounded": bool(result.docs),
            "docs": [_hit_payload(hit, self.searcher.source_root) for hit in result.docs],
            "tool_called": bool(run.get("tool_called")),
            "model_tool_called": bool(run.get("model_tool_called")),
            "recovered": bool(run.get("recovered")),
            "planner_called": bool(run.get("planner_called")),
            "planner_cache_hit": bool(run.get("planner_cache_hit")),
            "completion_calls": int(run.get("completion_calls", 0)),
            "search_ms": float(run.get("search_ms", 0.0)),
            "agent_ms": float(run.get("agent_ms", 0.0)),
        }
        if self.language_query:
            payload.update({
                "agentic": True,
                "queries": run.get("queries", []),
                "antipatterns": run.get("antipatterns", []),
                "rounds": int(run.get("rounds", 0)),
            })
        return payload


# Process JSONL requests until standard input closes.
def process_lines(agent: LocalSearchAgent, lines: Iterable[str], output: TextIO, top_k: int = 10) -> None:
    for line in lines:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            excerpt = request if isinstance(request, str) else request.get("text", request.get("excerpt", ""))
            response = agent.run(str(excerpt), top_k)
        except Exception as exc:
            response = {"grounded": False, "docs": [], "error": str(exc)}
        output.write(json.dumps(response, ensure_ascii=True) + "\n")
        output.flush()


# Parse CLI arguments and serve JSONL requests.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a persistent grounded HKM search agent.")
    parser.add_argument("index_root")
    parser.add_argument("--base-url", default="http://127.0.0.1:4321/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=LMSTUDIO_TIMEOUT)
    parser.add_argument("--tool-mode", choices=["hybrid", "token", "semantic"], default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--native-planner-tool", action="store_true", help="Require the model to emit search_index after planning")
    parser.add_argument("--language-query", action="store_true", help="Iteratively search natural-language queries and paraphrases")
    parser.add_argument("--always-refine", action="store_true", help="Force one LM inspection/refinement round for every language query")
    parser.add_argument("--language-rounds", type=int, choices=[1, 2, 3], default=2, help="Maximum result-conditioned search rounds")
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--deterministic-first", dest="deterministic_first", action="store_true")
    routing.add_argument("--model-first", dest="deterministic_first", action="store_false")
    parser.set_defaults(deterministic_first=True)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    mode = args.tool_mode or ("semantic" if args.language_query else "token")
    agent = LocalSearchAgent(
        args.index_root,
        args.base_url,
        args.model,
        args.timeout,
        mode,
        args.deterministic_first,
        native_tool=args.native_planner_tool,
        language_query=args.language_query,
        always_refine=args.always_refine,
        language_rounds=args.language_rounds,
    )
    if args.warmup:
        agent.warmup()
    process_lines(agent, sys.stdin, sys.stdout, args.top_k)


if __name__ == "__main__":
    main()
