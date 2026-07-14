import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import tlux.search.hkm.tools.agent_benchmark as benchmark
import tlux.search.hkm.tools.local_agent as local_agent

from tlux.search.hkm import build_search_index_from_documents
from tlux.search.hkm.tools.agent_benchmark import (
    DeterministicToolAgent,
    LMStudioQueryGenerator,
    LMStudioPlannerToolAgent,
    LMStudioToolAgent,
    StubQueryGenerator,
    _evidence_coverage,
    _fallback_queries,
    _grounded_quality_failed,
    _parse_tool_query,
    _keyword_query,
    _planner_excerpt,
    _rerank_with_evidence,
    evaluate_agent,
    evaluate_tool_agent,
    parse_query,
)
from tlux.search.hkm.tools.local_agent import LocalSearchAgent, process_lines


class CountingGenerator(StubQueryGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, excerpt: str) -> str:
        self.calls += 1
        return super().generate(excerpt)


class FakeToolAgent(LMStudioToolAgent):
    def __init__(self) -> None:
        client = StubQueryGenerator()
        self.client = client
        self.model = "fake"
        self.responses = [
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call-1",
                            "function": {"name": "search_index", "arguments": '{"query":"alpha dragon"}'},
                        }]
                    }
                }]
            },
            {"choices": [{"message": {"content": '{"answer":"found","source_path":"a.txt"}'}}]},
        ]

    def _request(self, path, payload=None):
        return self.responses.pop(0)

    def _model_name(self):
        return self.model


class NoToolAgent(LMStudioToolAgent):
    def __init__(self) -> None:
        client = StubQueryGenerator()
        self.client = client
        self.model = "fake"

    def _request(self, path, payload=None):
        return {"choices": [{"message": {"content": "truncated reasoning"}}]}

    def _model_name(self):
        return self.model


class NativePlannerClient(StubQueryGenerator):
    model = "fake"

    def _model_name(self):
        return self.model

    def _request(self, path, payload=None):
        return {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "arguments": '{"query":"alpha dragon"}',
                        },
                    }],
                },
            }],
        }


def test_parse_query_accepts_json_and_code_fences() -> None:
    assert parse_query('{"query": "dragon wardstone"}') == "dragon wardstone"
    assert parse_query('```json\n{"query":"sky fortress"}\n```') == "sky fortress"
    assert parse_query('{"query":"truncated evidence') == "truncated evidence"
    assert _keyword_query("!") == "!"


def test_planner_excerpt_bounds_long_raw_input() -> None:
    words = [f"word{index}" for index in range(100)]
    excerpt = _planner_excerpt(" ".join(words), limit=10)
    assert excerpt.split() == words[:5] + words[-5:]


def test_fallback_queries_try_rare_terms_before_phrases() -> None:
    excerpt = "alpha dragon fortress with repeated identifier_123"
    queries = _fallback_queries(excerpt)
    assert queries[0] == _keyword_query(excerpt)


def test_grounded_quality_gate_requires_perfect_evidence() -> None:
    passing = {"errors": [], "evidence": {"recall_at_k": 1.0, "precision_at_1": 1.0, "mrr": 1.0}}
    failing = {"errors": [], "evidence": {"recall_at_k": 1.0, "precision_at_1": 0.5, "mrr": 1.0}}
    assert _grounded_quality_failed(passing) is False
    assert _grounded_quality_failed(failing) is True


def test_tool_query_is_bounded_and_recovers_truncated_json() -> None:
    assert _parse_tool_query('{"query":"one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"}') == (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    )
    assert _parse_tool_query('{"query":"alpha dragon') == "alpha dragon"


def test_lmstudio_query_rejects_unquoted_planner_text(monkeypatch) -> None:
    client = LMStudioQueryGenerator(model="fake")
    monkeypatch.setattr(
        client,
        "_request",
        lambda path, payload: {"choices": [{"message": {"content": "The user wants a query."}}]},
    )
    with pytest.raises(ValueError, match="did not quote"):
        client.generate("alpha dragon fortress")


def test_lmstudio_query_rejects_generic_instruction_overlap(monkeypatch) -> None:
    client = LMStudioQueryGenerator(model="fake")
    monkeypatch.setattr(
        client,
        "_request",
        lambda path, payload: {"choices": [{"message": {"content": '{"query":"exact words from"}'}}]},
    )
    with pytest.raises(ValueError, match="did not quote"):
        client.generate("alpha dragon fortress")


def test_lmstudio_reuses_one_http_connection() -> None:
    class Connection:
        sock = None

        def __init__(self) -> None:
            self.requests = []
            self.timeout = None

        def request(self, method, target, body=None, headers=None):
            self.requests.append((method, target, body, headers))

        def getresponse(self):
            return SimpleNamespace(
                status=200,
                reason="OK",
                headers={},
                read=lambda: b'{"data": []}',
            )

    client = LMStudioQueryGenerator("http://localhost:1234/v1", model="fake")
    connection = Connection()
    client._connection = connection
    assert client._request("models") == {"data": []}
    assert client._request("models") == {"data": []}
    assert len(connection.requests) == 2
    assert all(request[0:2] == ("GET", "/v1/models") for request in connection.requests)


def test_stub_agent_recovers_sampled_documents(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [
            {"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}},
            {"text": "beta river valley", "metadata": {"source_path": "b.txt"}},
        ],
        max_k=1,
    )
    report = evaluate_agent(
        str(tmp_path / "index"),
        generator=StubQueryGenerator(),
        samples=2,
        top_k=3,
        probe_counts=[0],
        initial_probe_count=1,
    )
    assert report["final"]["recall_at_k"] == 1.0
    assert report["final"]["mrr"] > 0.0
    assert report["initial_probe_count"] == 1


def test_deterministic_first_skips_model_when_evidence_hits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    generator = CountingGenerator()
    report = evaluate_agent(
        str(tmp_path / "index"),
        generator=generator,
        samples=1,
        top_k=1,
        probe_counts=[0],
        deterministic_first=True,
    )
    assert generator.calls == 0
    assert report["planner_call_rate"] == 0.0
    assert report["final_relevance"]["precision_at_1"] == 1.0


def test_tool_agent_reports_grounded_tool_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    report = evaluate_tool_agent(
        str(tmp_path / "index"),
        agent=DeterministicToolAgent(),
        samples=1,
        top_k=1,
    )
    assert report["tool_call_rate"] == 1.0
    assert report["evidence"]["recall_at_k"] == 1.0
    assert report["grounded_source_match_rate"] == 1.0


def test_lmstudio_tool_protocol_executes_search_and_answer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    report = evaluate_tool_agent(
        str(tmp_path / "index"),
        agent=FakeToolAgent(),
        samples=1,
        top_k=1,
    )
    assert report["tool_call_rate"] == 1.0
    assert report["evidence"]["recall_at_k"] == 1.0
    assert report["grounded_source_match_rate"] == 1.0
    assert report["answer_source_match_rate"] == 1.0


def test_lmstudio_tool_recovery_handles_missing_tool_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    report = evaluate_tool_agent(
        str(tmp_path / "index"), agent=NoToolAgent(), samples=1, top_k=1
    )
    assert report["tool_call_rate"] == 1.0
    assert report["model_tool_call_rate"] == 0.0
    assert report["recovery_rate"] == 1.0
    assert report["evidence"]["precision_at_1"] == 1.0
    assert report["answer_source_match_rate"] == 0.0
    assert report["rows"][0]["recovered"] is True


def test_structured_planner_runs_grounded_tool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    client = StubQueryGenerator()
    client.model = "fake"
    report = evaluate_tool_agent(
        str(tmp_path / "index"),
        agent=LMStudioPlannerToolAgent(client, mode="token"),
        samples=1,
        top_k=1,
    )
    assert report["model_call_rate"] == 1.0
    assert report["tool_call_rate"] == 1.0
    assert report["completion_calls_per_sample"] == 1.0
    assert report["evidence"]["precision_at_1"] == 1.0
    assert report["recovery_rate"] == 0.0


def test_structured_planner_caches_repeated_raw_passage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    client = CountingGenerator()
    client.model = "fake"
    agent = LMStudioPlannerToolAgent(client, mode="token")
    searcher = benchmark.Searcher.from_index_root(str(tmp_path / "index"))
    first = agent.run("alpha dragon fortress", searcher, top_k=1)
    second = agent.run("  alpha\n dragon   fortress ", searcher, top_k=1)
    assert client.calls == 1
    assert first["completion_calls"] == 1
    assert second["completion_calls"] == 0
    assert second["planner_cache_hit"] is True
    assert second["result"].docs


def test_structured_planner_can_use_native_tool_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    report = evaluate_tool_agent(
        str(tmp_path / "index"),
        agent=LMStudioPlannerToolAgent(NativePlannerClient(), mode="token", native_tool=True),
        samples=1,
        top_k=1,
    )
    assert report["agent"] == "lmstudio_planner_native_tool_agent"
    assert report["model_tool_call_rate"] == 1.0
    assert report["completion_calls_per_sample"] == 2.0
    assert report["evidence"]["precision_at_1"] == 1.0


def test_persistent_local_agent_returns_grounded_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    (tmp_path / "index" / "b.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [
            {"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}},
            {"text": "alpha dragon fortress", "metadata": {"source_path": "b.txt"}},
        ],
        max_k=2,
    )
    agent = LocalSearchAgent(
        str(tmp_path / "index"),
        runner=DeterministicToolAgent(mode="token"),
    )
    output = StringIO()
    process_lines(agent, ['{"text":"alpha dragon fortress"}\n'], output, top_k=1)
    response = json.loads(output.getvalue())
    assert response["grounded"] is True
    assert len(response["docs"]) == 1
    assert response["docs"][0]["source_path"] == "a.txt"


def test_persistent_local_agent_warmup_uses_planner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    calls: list[str] = []

    def generate(text: str) -> str:
        calls.append(text)
        return text

    client = SimpleNamespace(model=None, timeout=1.5, generate=generate)
    monkeypatch.setattr(local_agent, "LMStudioQueryGenerator", lambda *args: client)
    agent = LocalSearchAgent(str(tmp_path / "index"))
    assert agent.warmup() is True
    assert calls == ["warmup evidence token"]
    assert client.timeout == 1.5


def test_persistent_local_agent_can_require_native_tool_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    agent = LocalSearchAgent(str(tmp_path / "index"), native_tool=True)
    assert agent.runner.native_tool is True


def test_native_planner_caches_structured_query_but_calls_tool_again(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    agent = LMStudioPlannerToolAgent(NativePlannerClient(), mode="token", native_tool=True)
    searcher = benchmark.Searcher.from_index_root(str(tmp_path / "index"))
    first = agent.run("alpha dragon fortress", searcher, top_k=1)
    second = agent.run("alpha dragon fortress", searcher, top_k=1)
    assert first["completion_calls"] == 2
    assert second["completion_calls"] == 1
    assert second["planner_cache_hit"] is True
    assert second["model_tool_called"] is True


def test_tool_only_skips_second_completion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    agent = FakeToolAgent()
    agent.final_answer = False
    report = evaluate_tool_agent(str(tmp_path / "index"), agent=agent, samples=1, top_k=1)
    assert report["answer_mode"] == "tool-only"
    assert report["completion_calls_per_sample"] == 1.0
    assert report["grounded_source_match_rate"] == 1.0


def test_tool_deterministic_first_skips_model_on_exact_hit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    agent = FakeToolAgent()
    report = evaluate_tool_agent(
        str(tmp_path / "index"), agent=agent, samples=1, top_k=1, deterministic_first=True
    )
    assert report["model_call_rate"] == 0.0
    assert report["completion_calls_per_sample"] == 0.0
    assert report["grounded_source_match_rate"] == 1.0


def test_high_score_without_evidence_still_escalates(monkeypatch) -> None:
    hit = SimpleNamespace(doc_id=1, span=(0, 1), score=0.99)
    searches = []
    ranks = iter([None, 1])

    def fake_search(*args, **kwargs):
        searches.append(kwargs.get("mode", args[-1] if args else ""))
        return SimpleNamespace(docs=[hit]), 1.0

    monkeypatch.setattr(benchmark, "_search", fake_search)
    monkeypatch.setattr(benchmark, "_rerank_with_evidence", lambda result, *args: result)
    monkeypatch.setattr(benchmark, "_evidence_rank", lambda *args: next(ranks))
    raw = " ".join(f"term{index}" for index in range(48))
    result, elapsed, fallback_calls = benchmark._adaptive_tool_search(
        object(), "long model query with many words", 1, 0, "token", raw, {}
    )
    assert result.docs
    assert elapsed == 2.0
    assert fallback_calls == 1
    assert len(searches) == 2


def test_adaptive_tool_search_fails_closed_without_evidence(monkeypatch) -> None:
    hit = SimpleNamespace(doc_id=1, span=(0, 1), score=0.99)

    def fake_search(*args, **kwargs):
        return SimpleNamespace(docs=[hit]), 1.0

    monkeypatch.setattr(benchmark, "_search", fake_search)
    monkeypatch.setattr(benchmark, "_fallback_queries", lambda text: [])
    monkeypatch.setattr(benchmark, "_rerank_with_evidence", lambda result, *args: result)
    monkeypatch.setattr(benchmark, "_evidence_rank", lambda *args: None)
    result, _, _ = benchmark._adaptive_tool_search(
        object(), "query", 1, 0, "token", "raw evidence", {}
    )
    assert result.docs == []
    assert result.count == 0


def test_evidence_guard_rejects_generic_overlap_when_source_exists(tmp_path: Path) -> None:
    target = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    (tmp_path / "target.txt").write_text(target, encoding="utf-8")
    (tmp_path / "decoy.txt").write_text(
        "alpha beta gamma delta epsilon zeta eta theta iota lambda", encoding="utf-8"
    )
    searcher = SimpleNamespace(source_root=str(tmp_path))

    def hit(path: str, score: float = 0.0, byte_end: int = 0) -> SimpleNamespace:
        return SimpleNamespace(
            preview_text="",
            score=score,
            token_score=score,
            doc_id=1 if path == "target.txt" else 2,
            document=SimpleNamespace(source_path=path, byte_start=0, byte_end=byte_end),
        )
    assert _evidence_coverage(hit("target.txt"), target, searcher) == 1.0
    assert _evidence_coverage(hit("decoy.txt"), target, searcher) == 0.0
    punctuated = "Industry praise for: Suspenseful, sexy, and entertaining storytelling -- first in Yarros Empyrean series."
    (tmp_path / "punctuated.txt").write_text(punctuated, encoding="utf-8")
    assert _evidence_coverage(
        hit("punctuated.txt"),
        "Industry praise for Suspenseful sexy entertaining storytelling first Yarros Empyrean series",
        searcher,
    ) == 1.0
    result = SimpleNamespace(docs=[hit("decoy.txt", 1.0), hit("target.txt", 0.1)])
    _rerank_with_evidence(result, target, searcher)
    assert result.docs[0].document.source_path == "target.txt"
    long_target = " ".join(f"term{index}" for index in range(80))
    (tmp_path / "long.txt").write_text("prefix " + long_target + " suffix", encoding="utf-8")
    assert _evidence_coverage(hit("long.txt", byte_end=6), long_target, searcher) == 1.0
