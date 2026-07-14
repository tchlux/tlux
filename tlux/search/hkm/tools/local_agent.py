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
from typing import Any, Dict, Iterable, TextIO

from ..search.searcher import Searcher
from .agent_benchmark import (
    DeterministicToolAgent,
    LMSTUDIO_TIMEOUT,
    LMStudioPlannerToolAgent,
    LMStudioQueryGenerator,
)


# Keep the compact, stable fields needed by a downstream agent or service.
def _hit_payload(hit: Any) -> Dict[str, Any]:
    return {
        "doc_id": int(hit.doc_id),
        "source_path": hit.source_path,
        "score": float(hit.score),
        "preview_text": hit.preview_text,
    }


# Persistent local agent that reuses one searcher and one model client.
class LocalSearchAgent:
    def __init__(
        self,
        index_root: str,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str | None = None,
        timeout: float = LMSTUDIO_TIMEOUT,
        mode: str = "token",
        deterministic_first: bool = False,
        runner: Any | None = None,
    ) -> None:
        self.searcher = Searcher.from_index_root(index_root)
        self.deterministic = DeterministicToolAgent(mode) if deterministic_first else None
        if runner is not None:
            self.client = None
            self.runner = runner
        else:
            self.client = LMStudioQueryGenerator(base_url, model, timeout)
            self.runner = LMStudioPlannerToolAgent(self.client, mode)

    # Warm HKM search and the structured planner with representative requests.
    def warmup(self) -> bool:
        search_ready = True
        try:
            self.searcher.search({"mode": "token", "text": "warmup", "top_k": 1})
        except Exception:
            search_ready = False
        if self.client is None:
            return search_ready
        try:
            self.client.generate("warmup evidence token")
            return search_ready
        except Exception:
            return False

    # Run one raw passage through the grounded search tool.
    def run(self, excerpt: str, top_k: int = 10) -> Dict[str, Any]:
        if not excerpt.strip():
            raise ValueError("text must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.deterministic is not None:
            cheap = self.deterministic.run(excerpt, self.searcher, top_k)
            if cheap["result"].docs:
                return self._serialize(cheap)
        return self._serialize(self.runner.run(excerpt, self.searcher, top_k))

    # Convert the internal result to the stable JSONL response contract.
    def _serialize(self, run: Dict[str, Any]) -> Dict[str, Any]:
        result = run["result"]
        return {
            "query": run.get("query", ""),
            "grounded": bool(result.docs),
            "docs": [_hit_payload(hit) for hit in result.docs],
            "tool_called": bool(run.get("tool_called")),
            "model_tool_called": bool(run.get("model_tool_called")),
            "recovered": bool(run.get("recovered")),
            "completion_calls": int(run.get("completion_calls", 0)),
            "search_ms": float(run.get("search_ms", 0.0)),
            "agent_ms": float(run.get("agent_ms", 0.0)),
        }


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
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=LMSTUDIO_TIMEOUT)
    parser.add_argument("--tool-mode", choices=["hybrid", "token", "semantic"], default="token")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--deterministic-first", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    agent = LocalSearchAgent(
        args.index_root,
        args.base_url,
        args.model,
        args.timeout,
        args.tool_mode,
        args.deterministic_first,
    )
    if args.warmup:
        agent.warmup()
    process_lines(agent, sys.stdin, sys.stdout, args.top_k)


if __name__ == "__main__":
    main()
