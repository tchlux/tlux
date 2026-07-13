from pathlib import Path

from tlux.search.hkm import build_search_index_from_documents
from tlux.search.hkm.tools.agent_benchmark import StubQueryGenerator, _keyword_query, evaluate_agent, parse_query


class CountingGenerator(StubQueryGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, excerpt: str) -> str:
        self.calls += 1
        return super().generate(excerpt)


def test_parse_query_accepts_json_and_code_fences() -> None:
    assert parse_query('{"query": "dragon wardstone"}') == "dragon wardstone"
    assert parse_query('```json\n{"query":"sky fortress"}\n```') == "sky fortress"
    assert _keyword_query("!") == "!"


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
    )
    assert report["final"]["recall_at_k"] == 1.0
    assert report["final"]["mrr"] > 0.0


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
