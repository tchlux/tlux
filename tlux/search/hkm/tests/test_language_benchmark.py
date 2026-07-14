from pathlib import Path

from tlux.search.hkm.tools.language_benchmark import evaluate_case, parse_cases, summarize


def test_parse_language_benchmark_table() -> None:
    path = Path(__file__).parents[1] / "plan" / "benchmark_language_queries.md"
    cases = parse_cases(path.read_text(encoding="utf-8"))
    assert [case["id"] for case in cases] == [f"LQ-{index:02d}" for index in range(1, 7)]
    assert cases[3]["groups"][-1] == ["watch", "watching", "horror", "fear"]
    assert cases[4]["antipatterns"] == []


def test_evaluate_case_matches_aliases_and_reports_antipattern_rank() -> None:
    case = {
        "id": "LQ-test",
        "query": "night climb",
        "groups": [["night", "dark"], ["climb", "scale"]],
        "antipatterns": ["archives"],
    }
    response = {
        "grounded": True,
        "agent_ms": 12,
        "search_ms": 4,
        "docs": [
            {"preview_text": "A dark person is climbing a wall."},
            {"preview_text": "The archives are old."},
        ],
    }
    result = evaluate_case(case, response)
    assert result["group_coverage_at_5"] == 1.0
    assert result["coherent_hit_at_5"] is True
    assert result["top_hit_groups"] == 2
    assert result["anti_rank"] == 2


def test_summarize_requires_coverage_and_coherence() -> None:
    passing = {
        "group_coverage_at_5": 1.0,
        "coherent_hit_at_5": True,
        "agent_ms": 10.0,
        "search_ms": 2.0,
    }
    failing = dict(passing, group_coverage_at_5=0.5, coherent_hit_at_5=False)
    report = summarize([passing, failing], [12.0, 20.0])
    assert report["passed"] == 1
    assert report["gate_passed"] is False
    assert report["wall_ms"]["median"] == 16.0
