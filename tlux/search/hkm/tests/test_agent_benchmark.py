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
    LanguageSearchAgent,
    LMStudioQueryGenerator,
    LMStudioPlannerToolAgent,
    LMStudioToolAgent,
    StubQueryGenerator,
    _evidence_coverage,
    _fallback_queries,
    _grounded_quality_failed,
    _parse_tool_query,
    _keyword_query,
    _language_query_clauses,
    _language_first_pass_confident,
    _language_query_variants,
    _language_negative_clauses,
    _language_unless_clauses,
    _language_missing_entity_query,
    _language_missing_entity_clue,
    _language_compact_concept_query,
    _language_named_contrast_query,
    _language_positive_clauses,
    _language_should_abstain,
    _language_token_pairs,
    _language_word_forms,
    _merge_language_results,
    _parse_language_plan,
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


class FakeLanguageClient:
    model = "fake"

    def __init__(self) -> None:
        self.calls = []

    def plan_language_query(self, query: str, snippets: str):
        self.calls.append((query, snippets))
        return {"queries": ["nighttime climbing gathered crowd horror"], "exclude_terms": ["wooden"]}


def test_parse_query_accepts_json_and_code_fences() -> None:
    assert parse_query('{"query": "dragon wardstone"}') == "dragon wardstone"
    assert parse_query('```json\n{"query":"sky fortress"}\n```') == "sky fortress"
    assert parse_query('{"query":"truncated evidence') == "truncated evidence"
    assert _keyword_query("!") == "!"


def test_language_plan_bounds_variants_and_exclusions() -> None:
    plan = _parse_language_plan(
        '{"queries":["one two", "three four five"], "exclude_terms":["wooden ladder"]}'
    )
    assert plan == {
        "queries": ["one two", "three four five"],
        "exclude_terms": ["wooden ladder"],
    }
    assert _language_query_variants("A large wall stands over us while crowds watch in horror")


def test_compact_language_query_uses_one_guarded_concept_lane() -> None:
    assert _language_compact_concept_query("A large wall stands over us") == "large wall structure"
    assert _language_compact_concept_query("A large wall stands over crowds watching in horror") == ""
    assert _language_compact_concept_query("Find a large wall, not a tower") == ""


def test_language_variants_skip_single_word_clause_lanes() -> None:
    variants = _language_query_variants("Find the Scribe Quadrant, like, right now, if needed")
    assert "like" not in variants
    assert "right now" in variants
    missing = _language_query_variants(
        "I remember make near pretty, but not who or what was involved"
    )
    assert "pretty make" in missing
    assert "pretty make passage" in missing


def test_language_clauses_consume_multiword_condition_markers() -> None:
    query = (
        "Only if someone climbs the chimney as the crowd watches below, "
        "and provided it is after dark, return the memory"
    )
    clauses = _language_query_clauses(query)
    variants = _language_query_variants(query)
    assert "someone climbs the chimney as the crowd watches below" in clauses
    assert "dark" in clauses
    assert "Only" not in clauses
    assert "provided it is" not in clauses
    assert "provided it is" not in variants
    assert "dark" not in variants
    assert "crowd people below" in variants
    assert "night dark" in variants
    assert "night dark" in _language_query_variants("after darkness or nightfall")


def test_language_token_pairs_keep_bounded_content_pairs() -> None:
    assert _language_token_pairs(
        "a Padres slugger clears loaded bases against Arizona"
    ) == [
        "padres slugger",
        "slugger clears",
        "clears loaded",
        "loaded bases",
        "bases against",
        "against arizona",
    ]


def test_language_named_contrast_query_keeps_anchor_and_tail_clues() -> None:
    assert _language_named_contrast_query(
        "If a Montessori teacher observes children lining up instead of yelling, "
        "find the classroom example where pushing and arguing occur."
    ) == "montessori example pushing arguing"
    assert _language_named_contrast_query(
        "if a montessori teacher observes children lining up instead of yelling, "
        "find the classroom example where pushing and arguing occur."
    ) == "montessori example pushing arguing"


def test_lm_memory_query_generation_is_structured_and_grounded() -> None:
    class MemoryGenerator(LMStudioQueryGenerator):
        def __init__(self) -> None:
            self.model = "fake"
            self.payload = None

        def _model_name(self) -> str:
            return self.model

        def _request(self, path, payload=None):
            self.payload = payload
            return {"choices": [{"message": {"content": '{"query":"find the basalt bridge"}'}}]}

    generator = MemoryGenerator()
    assert generator.generate_memory_query("A basalt bridge crosses the ravine", "specific") == (
        "find the basalt bridge"
    )
    assert generator.payload["max_tokens"] == 32
    with pytest.raises(ValueError, match="unknown memory-query style"):
        generator.generate_memory_query("A basalt bridge", "unknown")


def test_lm_missing_entity_prompt_redacts_first_clue() -> None:
    class MissingGenerator(LMStudioQueryGenerator):
        def __init__(self) -> None:
            self.model = "fake"
            self.payload = None

        def _model_name(self) -> str:
            return self.model

        def _request(self, path, payload=None):
            self.payload = payload
            return {"choices": [{"message": {"content": '{"query":"Quadrant Dain eyebrows"}'}}]}

    assert _language_missing_entity_clue("Scribe Quadrant Dain eyebrows") == "Scribe"
    generator = MissingGenerator()
    assert generator.generate_memory_query(
        "Scribe Quadrant Dain eyebrows", "missing_entity"
    ) == "Quadrant Dain eyebrows"
    assert "Scribe" not in generator.payload["messages"][0]["content"]


def test_language_ranking_uses_document_preview_for_condition_coverage() -> None:
    document = lambda text: SimpleNamespace(document_preview=text)
    weak = SimpleNamespace(
        doc_id=1,
        span=(0, 1),
        score=0.55,
        source_path="weak.txt",
        preview_text="people stand over us",
        anchor_preview_text="",
        document=document("A gathering fills the room."),
    )
    strong = SimpleNamespace(
        doc_id=2,
        span=(0, 1),
        score=0.54,
        source_path="strong.txt",
        preview_text="courtyard below",
        anchor_preview_text="",
        document=document("A large wall stands over the courtyard."),
    )
    result = SimpleNamespace(docs=[weak, strong], count=2, limit=2, next_offset=None)
    merged = _merge_language_results([result], "A large wall stands over us", [], 1)
    assert merged.docs[0].doc_id == 2
    assert "a" not in _language_word_forms("A large wall stands over us")


def test_missing_entity_merge_rewards_all_condition_groups() -> None:
    def hit(doc_id: int, score: float, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            doc_id=doc_id,
            span=(0, 1),
            score=score,
            source_path=f"{doc_id}.txt",
            preview_text=text,
            anchor_preview_text="",
            document=SimpleNamespace(document_preview=text),
        )

    query = (
        "I remember someone climbing something tall at night while people watched "
        "below, but I forgot who the person was and what the structure was called"
    )
    result = SimpleNamespace(
        docs=[
            hit(0, 0.90, "A routine office discussion."),
            hit(1, 0.80, "Someone climbs at night while people watch below."),
            hit(2, 0.50, "A towering parapet rises above the scene."),
            hit(3, 0.60, "A funny remark is exchanged."),
        ],
        count=4,
        limit=4,
        next_offset=None,
    )
    merged = _merge_language_results([result], query, [], 3)
    assert [hit.doc_id for hit in merged.docs] == [1, 0, 2]


def test_language_first_pass_confidence_requires_positive_coverage() -> None:
    hit = SimpleNamespace(
        preview_text="A crowd watches in horror.",
        anchor_preview_text="",
        document=SimpleNamespace(document_preview=""),
    )
    result = SimpleNamespace(docs=[hit])
    assert _language_first_pass_confident("A crowd watches in horror", result)
    assert not _language_first_pass_confident("A crowd watches in horror, not dragons", result)


def test_language_abstention_rejects_low_coverage_token_decoys() -> None:
    hit = SimpleNamespace(
        doc_id=1,
        preview_text="A battery is mentioned during a briefing.",
        anchor_preview_text="",
        semantic_score=0.0,
        document=SimpleNamespace(document_preview=""),
    )
    result = SimpleNamespace(docs=[hit])
    assert _language_should_abstain("Which document explains lunar battery gardening?", result)
    hit.semantic_score = 0.4
    assert not _language_should_abstain("Which document explains lunar battery gardening?", result)


def test_language_merge_preserves_first_pass_candidates_for_refinement() -> None:
    hits = [
        SimpleNamespace(
            doc_id=index,
            span=(0, 1),
            score=1.0 - index / 100.0,
            source_path=f"{index}.txt",
            preview_text=f"candidate {index}",
            anchor_preview_text="",
            document=SimpleNamespace(document_preview=""),
        )
        for index in range(2)
    ]
    first = SimpleNamespace(docs=hits, count=2, limit=2, next_offset=None)
    merged = _merge_language_results([first], "candidate", [], 1)
    assert len(first.docs) == 2
    assert len(merged.docs) == 1


def test_language_search_cache_reuses_index_lane(monkeypatch) -> None:
    calls = []
    hit = SimpleNamespace(doc_id=1)
    result = SimpleNamespace(docs=[hit], count=1, limit=1, next_offset=None)

    def fake_search(searcher, query, top_k, probe_count, mode):
        calls.append((query, top_k, probe_count, mode))
        return result, 12.0

    monkeypatch.setattr(benchmark, "_search", fake_search)
    agent = LanguageSearchAgent(None, "semantic")
    searcher = SimpleNamespace(index_root="/tmp/index")
    first, first_ms = agent._cached_search(searcher, "alpha", 3, 0, "semantic")
    second, second_ms = agent._cached_search(searcher, "alpha", 3, 0, "semantic")
    assert len(calls) == 1
    assert first_ms == 12.0
    assert second_ms == 0.0
    assert first is not second
    assert first.docs == second.docs


def test_language_merge_keeps_stronger_focused_lane_over_repeated_decoy() -> None:
    def hit(doc_id: int, score: float, preview: str) -> SimpleNamespace:
        return SimpleNamespace(
            doc_id=doc_id,
            span=(0, 1),
            score=score,
            source_path=f"{doc_id}.txt",
            preview_text=preview,
            anchor_preview_text="",
            document=SimpleNamespace(document_preview=preview),
        )

    target = hit(1, 0.60, "Dain enters the Scribe Quadrant")
    decoy = hit(2, 0.57, "Dain enters the Scribe Quadrant with another detail")
    first = SimpleNamespace(docs=[target, decoy], count=2, limit=2, next_offset=None)
    second = SimpleNamespace(docs=[target], count=1, limit=1, next_offset=None)
    third = SimpleNamespace(docs=[decoy], count=1, limit=1, next_offset=None)
    merged = _merge_language_results(
        [first, second, third, third],
        "Dain enters the Scribe Quadrant",
        [],
        1,
    )
    assert merged.docs[0].doc_id == 1


def test_language_merge_preserves_lane_specific_expansion_evidence() -> None:
    def hit(doc_id: int, score: float, preview: str) -> SimpleNamespace:
        return SimpleNamespace(
            doc_id=doc_id,
            span=(0, 1),
            score=score,
            source_path=f"{doc_id}.txt",
            preview_text=preview,
            anchor_preview_text="",
            document=SimpleNamespace(document_preview=preview),
        )

    first = SimpleNamespace(
        docs=[hit(1, 0.80, "generic immune response")], count=1, limit=1, next_offset=None
    )
    expansion = SimpleNamespace(
        docs=[hit(2, 0.45, "JAM-A intestinal immune compensation")],
        count=1,
        limit=1,
        next_offset=None,
    )
    merged = _merge_language_results(
        [first, expansion],
        "inflammatory Th17 cells and anti-inflammatory iTregs",
        [],
        1,
        lane_queries=[
            "inflammatory Th17 cells and anti-inflammatory iTregs",
            "JAM-A intestinal immune compensation",
        ],
    )
    assert merged.docs[0].doc_id == 2


def test_language_negative_clause_demotes_matching_anchor() -> None:
    bad = SimpleNamespace(
        doc_id=1,
        span=(0, 1),
        score=0.60,
        source_path="bad.txt",
        preview_text="A funny Archives research remark.",
        anchor_preview_text="",
        document=SimpleNamespace(document_preview=""),
    )
    good = SimpleNamespace(
        doc_id=2,
        span=(0, 1),
        score=0.55,
        source_path="good.txt",
        preview_text="A funny remark in an office.",
        anchor_preview_text="",
        document=SimpleNamespace(document_preview=""),
    )
    result = SimpleNamespace(docs=[bad, good], count=2, limit=2, next_offset=None)
    query = "A funny remark in an office, not the Archives research passage"
    merged = _merge_language_results([result], query, [], 1)
    assert _language_negative_clauses(query)
    assert merged.docs[0].doc_id == 2


def test_language_contrast_ignores_negative_document_context() -> None:
    bad = SimpleNamespace(
        doc_id=1,
        span=(0, 1),
        score=0.56,
        source_path="bad.txt",
        preview_text="A funny remark.",
        anchor_preview_text="",
        document=SimpleNamespace(document_preview="Archives research passage"),
    )
    good = SimpleNamespace(
        doc_id=2,
        span=(0, 1),
        score=0.55,
        source_path="good.txt",
        preview_text="A funny remark in an office.",
        anchor_preview_text="",
        document=SimpleNamespace(document_preview=""),
    )
    result = SimpleNamespace(docs=[bad, good], count=2, limit=2, next_offset=None)
    merged = _merge_language_results(
        [result],
        "A funny remark in an office, not the Archives research passage",
        [],
        1,
    )
    assert merged.docs[0].doc_id == 2
    assert _language_positive_clauses("Find a funny remark without old Archives research") == [
        "Find a funny remark",
    ]


def test_language_negative_parser_keeps_event_negation() -> None:
    query = "Tairn tells me not to leave the field while the crowd waits"
    assert _language_negative_clauses(query) == []


def test_language_unless_parser_extracts_soft_exclusion() -> None:
    clauses = _language_unless_clauses(
        "Find the office joke unless this is the Archives research discussion"
    )
    assert any("archive" in clause and "research" in clause for clause in clauses)


def test_language_negative_parser_targets_explicit_unless_contrast() -> None:
    clauses = _language_negative_clauses(
        "Find the office joke unless this is Archives research"
    )
    assert any("archive" in clause and "research" in clause for clause in clauses)
    assert _language_negative_clauses(
        "Find it unless this is not dragon spectators"
    ) == [{"dragon", "spectator", "spectators"}]
    assert not _language_negative_clauses(
        "Find the office joke unless Archives is a separate scene"
    )


def test_language_unless_penalty_reorders_only_selected_page() -> None:
    def hit(doc_id: int, score: float, preview: str) -> SimpleNamespace:
        return SimpleNamespace(
            doc_id=doc_id,
            span=(0, 1),
            score=score,
            source_path=f"{doc_id}.txt",
            preview_text=preview,
            anchor_preview_text="",
            document=SimpleNamespace(document_preview=preview),
        )

    bad = hit(1, 0.60, "A funny Archives research remark")
    good = hit(2, 0.55, "A funny remark in an office")
    result = SimpleNamespace(docs=[bad, good], count=2, limit=2, next_offset=None)
    merged = _merge_language_results(
        [result],
        "A funny remark in an office unless this is the Archives research passage",
        [],
        2,
    )
    assert merged.docs[0].doc_id == 2
    assert {hit.doc_id for hit in merged.docs} == {1, 2}


def test_language_missing_entity_parser_covers_memory_phrasings() -> None:
    assert _language_missing_entity_query("I cannot recall who climbed the tower")
    assert _language_missing_entity_query("I don't remember who climbed the tower")
    assert _language_missing_entity_query("I do not recall what happened near the tower")
    assert _language_missing_entity_query("I have no memory of who climbed the tower")
    assert _language_missing_entity_query("I remember the scene, but not who was there")
    assert _language_missing_entity_query("The name escapes me; search the chimney climb")
    assert _language_missing_entity_query("My name has escaped me; search the chimney climb")
    assert not _language_missing_entity_query("Xaden climbs the tower at night")


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


def test_lmstudio_timeout_does_not_retry() -> None:
    class TimeoutConnection:
        sock = None

        def __init__(self) -> None:
            self.requests = 0
            self.timeout = None

        def request(self, *args, **kwargs):
            self.requests += 1
            raise TimeoutError("planner timeout")

        def close(self) -> None:
            pass

    client = LMStudioQueryGenerator("http://localhost:1234/v1", model="fake")
    connection = TimeoutConnection()
    client._connection = connection
    with pytest.raises(TimeoutError, match="planner timeout"):
        client._request("models")
    assert connection.requests == 1
    assert client._connection is None


def test_lmstudio_reconnects_once_after_dropped_socket(monkeypatch) -> None:
    class DroppedConnection:
        sock = None

        def __init__(self) -> None:
            self.timeout = None
            self.closed = False

        def request(self, *args, **kwargs):
            raise ConnectionResetError("socket dropped")

        def close(self) -> None:
            self.closed = True

    class WorkingConnection:
        sock = None

        def __init__(self) -> None:
            self.timeout = None
            self.requests = 0

        def request(self, *args, **kwargs):
            self.requests += 1

        def getresponse(self):
            return SimpleNamespace(
                status=200,
                reason="OK",
                headers={},
                read=lambda: b'{"data": []}',
            )

    client = LMStudioQueryGenerator("http://localhost:1234/v1", model="fake")
    dropped = DroppedConnection()
    working = WorkingConnection()
    connections = iter((dropped, working))

    def next_connection():
        client._connection = next(connections)
        return client._connection

    monkeypatch.setattr(client, "_http_connection", next_connection)
    assert client._request("models") == {"data": []}
    assert dropped.closed is True
    assert working.requests == 1


def test_lmstudio_reconnects_after_empty_response(monkeypatch) -> None:
    class EmptyConnection:
        sock = None

        def __init__(self) -> None:
            self.timeout = None
            self.closed = False

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return SimpleNamespace(
                status=200,
                reason="OK",
                headers={},
                read=lambda: b"",
            )

        def close(self) -> None:
            self.closed = True

    class WorkingConnection(EmptyConnection):
        def getresponse(self):
            return SimpleNamespace(
                status=200,
                reason="OK",
                headers={},
                read=lambda: b'{"data": []}',
            )

    client = LMStudioQueryGenerator("http://localhost:1234/v1", model="fake")
    empty = EmptyConnection()
    working = WorkingConnection()
    connections = iter((empty, working))
    def next_connection():
        client._connection = next(connections)
        return client._connection

    monkeypatch.setattr(client, "_http_connection", next_connection)
    assert client._request("models") == {"data": []}
    assert empty.closed is True


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
    assert response["docs"][0]["document"]["document_preview"]
    assert response["docs"][0]["document"]["source_context"] == "alpha dragon fortress"


def test_source_context_is_bounded_and_root_contained(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.txt").write_text("head " + ("x" * 3000) + " tail", encoding="utf-8")
    document = SimpleNamespace(byte_start=0, byte_end=3010, document_preview="")
    hit = SimpleNamespace(source_path="a.txt", document=document)
    context = local_agent._source_context(hit, str(source_root))
    assert len(context.encode("utf-8")) <= 2048
    assert context.startswith("head")
    assert context.endswith("tail")
    hit.source_path = "../outside.txt"
    assert local_agent._source_context(hit, str(source_root)) == ""


def test_persistent_local_agent_supports_language_queries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text(
        "At night a large wall stands over crowds watching in horror.",
        encoding="utf-8",
    )
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "At night a large wall stands over crowds watching in horror.", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    agent = LocalSearchAgent(
        str(tmp_path / "index"),
        runner=LanguageSearchAgent(mode="hybrid"),
        language_query=True,
    )
    output = StringIO()
    process_lines(agent, ['{"text":"A large structure towers over a horrified crowd at night"}\n'], output, top_k=1)
    response = json.loads(output.getvalue())
    assert response["agentic"] is True
    assert response["planner_called"] is False
    assert response["rounds"] == 2
    assert response["docs"][0]["source_path"] == "a.txt"


def test_language_deterministic_first_avoids_model_client(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "A large wall stands over us.", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    monkeypatch.setattr(
        local_agent,
        "LMStudioQueryGenerator",
        lambda *args: pytest.fail("deterministic language search created a model client"),
    )
    agent = LocalSearchAgent(
        str(tmp_path / "index"),
        mode="semantic",
        deterministic_first=True,
        language_query=True,
    )
    response = agent.run("A large wall stands over us", top_k=1)
    assert response["agentic"] is True
    assert response["grounded"] is True


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


def test_language_agent_handles_conditional_partial_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [
            {
                "text": "At night Xaden climbs the side of a large stone structure while crowds gather below to watch in horror.",
                "metadata": {"source_path": "target.txt"},
            },
            {
                "text": "A quiet wooden bridge spans a stream.",
                "metadata": {"source_path": "decoy.txt"},
            },
        ],
        max_k=1,
    )
    searcher = benchmark.Searcher.from_index_root(str(tmp_path / "index"))
    query = "The person is forgotten, but at night something climbs a large structure while crowds watch in horror"
    run = LanguageSearchAgent(mode="hybrid").run(query, searcher, top_k=1)
    assert run["result"].docs[0].source_path == "target.txt"
    assert len(run["queries"]) >= 2


def test_language_agent_refines_after_first_search(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "nighttime climbing gathered crowd horror", "metadata": {"source_path": "target.txt"}}],
        max_k=1,
    )
    searcher = benchmark.Searcher.from_index_root(str(tmp_path / "index"))
    client = FakeLanguageClient()
    run = LanguageSearchAgent(client, mode="hybrid").run(
        "A scenario where a crowd watches in horror", searcher, top_k=1
    )
    assert len(client.calls) == 1
    assert run["rounds"] == 2
    assert run["planner_called"] is True
    assert run["antipatterns"]
    assert "nighttime climbing gathered crowd horror" in run["queries"]
    assert run["result"].docs[0].source_path == "target.txt"


def test_language_agent_can_force_refinement_after_confident_first_pass(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "nighttime climbing gathered crowd horror", "metadata": {"source_path": "target.txt"}}],
        max_k=1,
    )
    searcher = benchmark.Searcher.from_index_root(str(tmp_path / "index"))
    monkeypatch.setattr(benchmark, "_language_first_pass_confident", lambda *_: True)
    client = FakeLanguageClient()
    run = LanguageSearchAgent(client, mode="hybrid", always_refine=True).run(
        "A scenario where a crowd watches in horror", searcher, top_k=1
    )
    assert run["rounds"] == 2
    assert len(client.calls) == 1


def test_language_model_cap_keeps_deterministic_recovery_budget(monkeypatch) -> None:
    empty = SimpleNamespace(docs=[], count=0, limit=1, next_offset=None)
    monkeypatch.setattr(benchmark, "_search", lambda *args, **kwargs: (empty, 0.0))
    query = "The person is forgotten, but at night something climbs a large structure while crowds watch in horror"

    run = LanguageSearchAgent(FakeLanguageClient(), mode="hybrid").run(query, None, top_k=1)
    assert len(run["queries"]) == 4
    assert "structure climbs crowds horror night large watch" in run["queries"]
    missing = LanguageSearchAgent(FakeLanguageClient(), mode="hybrid").run(
        "I cannot recall the person or object; search for the part with Quadrant and right",
        None,
        top_k=1,
    )
    assert "Quadrant right" in missing["queries"]

    class FailingLanguageClient(FakeLanguageClient):
        def plan_language_query(self, query: str, snippets: str):
            raise ValueError("planner unavailable")

    recovered = LanguageSearchAgent(FailingLanguageClient(), mode="hybrid").run(
        query, None, top_k=1
    )
    assert len(recovered["queries"]) == 8


def test_deterministic_tool_agent_fails_closed_for_unindexed_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HKM_FAKE_EMBEDDER", "1")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "a.txt").write_text("alpha dragon fortress", encoding="utf-8")
    build_search_index_from_documents(
        str(tmp_path / "index"),
        [{"text": "alpha dragon fortress", "metadata": {"source_path": "a.txt"}}],
        max_k=1,
    )
    searcher = benchmark.Searcher.from_index_root(str(tmp_path / "index"))
    run = DeterministicToolAgent(mode="token").run(
        "unindexed glacier observatory evidence", searcher, top_k=1
    )
    assert run["result"].docs == []


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
