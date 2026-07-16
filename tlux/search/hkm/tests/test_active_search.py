from __future__ import annotations

import json

import numpy as np

from tlux.search.hkm.tools.active_search import (
    ActiveSearchAgent,
    ActiveSearchConfig,
    SmallLM,
    WindowCandidate,
    WindowPool,
    _fit_svc,
    _bootstrap_choice,
    _lane_choice,
)


def _window(index: int, embedding: list[float], size: int = 128) -> WindowCandidate:
    return WindowCandidate(index, 0, size, size, np.asarray(embedding, dtype=np.float32), f"window {index}", f"c{index % 3}")


def test_window_identity_includes_size() -> None:
    short = _window(1, [1.0, 0.0], 128)
    long = _window(1, [1.0, 0.0], 512)
    assert short.key == (1, 0, 128, 128)
    assert short.key != long.key


def test_explicit_pool_cache_preserves_windows_and_lanes(tmp_path) -> None:
    window = _window(1, [1.0, 0.0])
    window.lanes["semantic"] = 0.5
    pool = WindowPool([window])
    path = tmp_path / "pool.pkl"
    pool.save_cache(path, "query", str(tmp_path))
    loaded = WindowPool.load_cache(path, "query", str(tmp_path), lambda text: np.asarray([1.0, 0.0]))
    assert loaded.windows[0].key == window.key
    assert loaded.windows[0].lanes == {"semantic": 0.5}
    assert np.array_equal(loaded.embeddings, pool.embeddings)


def test_small_lm_judgment_receives_only_query_and_one_window(monkeypatch) -> None:
    model = SmallLM.__new__(SmallLM)
    window = _window(7, [1.0, 0.0])
    window.lanes["semantic"] = 0.987
    captured: list[str] = []
    monkeypatch.setattr(model, "_chat", lambda system, user: captured.extend([system, user]) or "**Label:** relevant\n**Evidence:** direct match")
    assert model.judge("original request", window)[0] == "relevant"
    assert captured[1] == "USER REQUEST:\noriginal request\n\nONE PASSAGE:\nwindow 7"
    assert "JSON" not in captured[0]


def test_small_lm_parses_loose_markdown_lists(monkeypatch) -> None:
    model = SmallLM.__new__(SmallLM)
    monkeypatch.setattr(model, "_chat", lambda system, user: """```markdown
### Paraphrases
- alternate wording
2. another wording
### HyDE:
* matching passage
```""")
    assert model.initial_queries("request") == (["alternate wording", "another wording"], ["matching passage"])


def test_active_search_returns_only_explicit_small_lm_positives() -> None:
    windows = [_window(index, [1.0, index / 10.0]) for index in range(30)]
    for index, window in enumerate(windows):
        window.lanes["semantic"] = 1.0 - index / 30.0
    labels = {0: "relevant", 4: "relevant"}
    judged: list[int] = []

    def judge(_: str, window: WindowCandidate) -> tuple[str, str]:
        judged.append(window.document_id)
        return labels.get(window.document_id, "not_relevant"), "small LM"

    result = ActiveSearchAgent(
        WindowPool(windows),
        judge,
        config=ActiveSearchConfig(max_seconds=10, max_calls=30, temporary_negatives=4),
    ).run("remembered request")

    assert {window.document_id for window in result.windows} == {0, 4}
    assert all(step.key[0] in judged for step in result.trace)
    assert result.stop_reason in {"all_windows_judged", "three_cycles_without_new_positives"}
    json.dumps(result.to_dict())


def test_active_cycle_uses_exploitation_boundary_and_cluster_routes() -> None:
    windows = [_window(index, [1.0, float(index % 7), float(index)]) for index in range(40)]
    for index, window in enumerate(windows):
        window.lanes["semantic"] = 1.0 - index / 40.0

    result = ActiveSearchAgent(
        WindowPool(windows),
        lambda query, window: ("relevant", "seed") if window.document_id == 0 else ("not_relevant", "no"),
        config=ActiveSearchConfig(max_seconds=30, max_calls=40, temporary_negatives=4),
    ).run("query")

    routes = {step.acquisition for step in result.trace}
    assert {"exploitation", "boundary", "cluster_exploration"}.issubset(routes)
    assert any(step.acquisition == "lane_exploration:semantic" for step in result.trace)
    assert result.stop_reason == "three_cycles_without_new_positives"


def test_bootstrap_scans_paraphrase_and_hyde_lanes() -> None:
    windows = [_window(index, [1.0, float(index)]) for index in range(4)]
    for window in windows:
        window.lanes["semantic"] = 0.1
        window.lanes["lexical"] = 0.1
        window.lanes["hybrid"] = 0.1
        window.lanes["paraphrase:0"] = 0.1
        window.lanes["hyde:0"] = 0.1
    windows[2].lanes["paraphrase:0"] = 1.0
    windows[3].lanes["hyde:0"] = 1.0
    ranks = {
        lane: sorted(range(len(windows)), key=lambda index: -windows[index].lanes[lane])
        for lane in windows[0].lanes
    }
    selected, route = _bootstrap_choice(
        list(range(len(windows))), windows, list(ranks), ranks, {}, 6,
    )
    assert selected == 2
    assert route == "bootstrap_lane:paraphrase:0"


def test_bootstrap_deep_probe_uses_original_lane_rank() -> None:
    windows = [_window(index, [1.0, float(index)]) for index in range(80)]
    for index, window in enumerate(windows):
        window.lanes["semantic"] = 1.0 - index / 80.0
    ranks = {"semantic": list(range(len(windows)))}
    selected, route = _bootstrap_choice(
        list(range(len(windows))), windows, ["semantic"], ranks, {}, 28,
    )
    assert selected == 48
    assert route == "bootstrap_lane:semantic"


def test_bootstrap_hyde_probe_reaches_deep_rank_early() -> None:
    windows = [_window(index, [1.0, float(index)]) for index in range(80)]
    for index, window in enumerate(windows):
        window.lanes["semantic"] = 0.1
        window.lanes["hyde:0"] = 1.0 - index / 80.0
    ranks = {lane: list(range(len(windows))) for lane in windows[0].lanes}
    selected, route = _bootstrap_choice(
        list(range(len(windows))), windows, list(ranks), ranks, {}, 44,
    )
    assert selected == 48
    assert route == "bootstrap_lane:hyde:0"


def test_lane_exploration_probes_hyde_frontier() -> None:
    windows = [_window(index, [1.0, float(index)]) for index in range(80)]
    for index, window in enumerate(windows):
        window.lanes["semantic"] = 0.1
        window.lanes["hyde:0"] = 1.0 - index / 80.0
    ranks = {lane: list(range(len(windows))) for lane in windows[0].lanes}
    selected, route = _lane_choice(list(range(len(windows))), ["semantic", "hyde:0"], ranks, 1)
    assert selected == 48
    assert route == "lane_exploration:hyde:0"


def test_active_trace_resumes_without_rejudging_windows(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"

    def pool() -> WindowPool:
        windows = [_window(index, [1.0, float(index)]) for index in range(8)]
        for index, window in enumerate(windows):
            window.lanes["semantic"] = 1.0 - index / 8.0
        return WindowPool(windows)

    calls = []
    judge = lambda query, window: calls.append(window.key) or ("not_relevant", "no")
    ActiveSearchAgent(pool(), judge, config=ActiveSearchConfig(max_seconds=10, max_calls=2, trace_path=str(path))).run("query")
    ActiveSearchAgent(pool(), judge, config=ActiveSearchConfig(
        max_seconds=10, max_calls=4, trace_path=str(path), resume_trace=True,
    )).run("query")
    assert len(calls) == 4
    assert len(set(calls)) == 4


def test_uncertain_128_window_uses_existing_larger_context() -> None:
    short = _window(1, [1.0, 0.0], 128)
    long = WindowCandidate(1, 0, 512, 512, np.asarray([1.0, 0.0]), "larger context", "c0")
    short.lanes["semantic"] = 1.0
    long.lanes["semantic"] = 0.5

    def judge(_: str, window: WindowCandidate) -> tuple[str, str]:
        return ("uncertain", "need context") if window.window_size == 128 else ("relevant", "clear in context")

    result = ActiveSearchAgent(
        WindowPool([short, long]), judge,
        config=ActiveSearchConfig(max_seconds=10, max_calls=2, temporary_negatives=1),
    ).run("query")

    assert result.windows == (long,)
    assert [step.acquisition for step in result.trace] == ["bootstrap_lane:semantic", "larger_context"]


def test_svc_uses_normalized_embedding_rows_and_requested_kernel() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    model, elapsed = _fit_svc(embeddings, [0], [1], [2], "poly")
    assert model.kernel == "poly"
    assert model.degree == 2
    assert model.class_weight == "balanced"
    assert elapsed >= 0.0
