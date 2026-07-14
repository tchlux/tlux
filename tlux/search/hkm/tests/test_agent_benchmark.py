from pathlib import Path
from types import SimpleNamespace

import tlux.search.hkm.tools.agent_benchmark as benchmark

from tlux.search.hkm import build_search_index_from_documents
from tlux.search.hkm.tools.agent_benchmark import (
    DeterministicToolAgent,
    LMStudioToolAgent,
    StubQueryGenerator,
    _parse_tool_query,
    _keyword_query,
    evaluate_agent,
    evaluate_tool_agent,
    parse_query,
)


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


def test_parse_query_accepts_json_and_code_fences() -> None:
    assert parse_query('{"query": "dragon wardstone"}') == "dragon wardstone"
    assert parse_query('```json\n{"query":"sky fortress"}\n```') == "sky fortress"
    assert _keyword_query("!") == "!"


def test_tool_query_is_bounded_and_recovers_truncated_json() -> None:
    assert _parse_tool_query('{"query":"one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"}') == (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    )
    assert _parse_tool_query('{"query":"alpha dragon') == "alpha dragon"


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
    assert elapsed == 6.0
    assert fallback_calls == 5
    assert len(searches) == 6
