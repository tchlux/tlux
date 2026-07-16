"""Read evaluation-only audits and replay exhaustive small-LM relevance maps.

Runtime search never imports this module. Oracle files accept only labels marked
as produced by the small LM; larger-model assessments stay in the registry.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..search.searcher import Searcher
from .active_search import WindowPool, _cluster_choice, _fit_svc


STATUSES = {
    "confirmed_search_failure",
    "suspected_bad_qrel",
    "ambiguous_relevance",
    "sparse_qrels_artifact",
    "bounded_corpus_artifact",
}


# Hold one large-model benchmark assessment outside runtime search.
@dataclass(frozen=True)
class FailureRecord:
    query_id: str
    qrels_targets: tuple[str, ...]
    observed_behavior: str
    assessment: str
    status: str


# Return the repository's evaluation-only registry path.
def default_failure_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "evaluation" / "scifact_failure_registry.json"


# Load the fixed development-query inventory used for exhaustive maps.
def load_hard_queries(path: str | Path | None = None) -> tuple[dict[str, str], ...]:
    source = Path(path).expanduser() if path is not None else Path(__file__).resolve().parents[1] / "evaluation" / "active_search_hard_queries.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"hard-query registry must contain a queries list: {source}")
    result = tuple({str(key): str(value) for key, value in row.items()} for row in rows if isinstance(row, dict))
    ids = [row.get("id", "") for row in result]
    if len(result) != len(rows) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("hard-query IDs must be unique and non-empty")
    return result


# Validate audit rows without interpreting them as runtime relevance labels.
def validate_failure_registry(rows: Sequence[Mapping[str, Any]]) -> tuple[FailureRecord, ...]:
    records: list[FailureRecord] = []
    seen: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        targets = tuple(str(value).strip() for value in row.get("qrels_targets", ()))
        observed = str(row.get("observed_behavior", "")).strip()
        assessment = str(row.get("assessment", "")).strip()
        status = str(row.get("status", ""))
        if not query_id or query_id in seen:
            raise ValueError(f"query_id must be unique and non-empty: {query_id!r}")
        if any(not value for value in targets) or len(set(targets)) != len(targets):
            raise ValueError(f"qrels_targets must be unique and non-empty: {query_id}")
        if not observed or not assessment:
            raise ValueError(f"observed_behavior and assessment are required: {query_id}")
        if status not in STATUSES:
            raise ValueError(f"invalid status for {query_id}: {status}")
        seen.add(query_id)
        records.append(FailureRecord(query_id, targets, observed, assessment, status))
    return tuple(records)


# Load the physically separate benchmark-failure registry.
def load_failure_registry(path: str | Path | None = None) -> tuple[FailureRecord, ...]:
    source = Path(path).expanduser() if path is not None else default_failure_registry_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("issues"), list):
        raise ValueError(f"registry must contain an issues list: {source}")
    return validate_failure_registry(payload["issues"])


# Load an exhaustive oracle while rejecting any non-small-LM label source.
def load_small_lm_map(path: str | Path) -> dict[str, dict[tuple[int, int, int, int], str]]:
    result: dict[str, dict[tuple[int, int, int, int], str]] = {}
    for number, line in enumerate(Path(path).expanduser().read_text(encoding="ascii").splitlines(), 1):
        row = json.loads(line)
        key = tuple(row.get("key", ()))
        label = str(row.get("label", ""))
        query_id = str(row.get("query_id", "")).strip()
        if row.get("judge") != "small_lm":
            raise ValueError(f"oracle line {number} was not labeled by the small LM")
        if len(key) != 4 or not all(isinstance(value, int) for value in key):
            raise ValueError(f"invalid window identity on oracle line {number}")
        if label not in {"relevant", "not_relevant", "uncertain"} or not query_id:
            raise ValueError(f"invalid oracle label on line {number}")
        labels = result.setdefault(query_id, {})
        if key in labels:
            raise ValueError(f"duplicate oracle window on line {number}")
        labels[key] = label
    return result


# Load the one query text repeated in an exhaustive oracle file.
def _oracle_query(path: str | Path) -> str:
    queries = {str(json.loads(line).get("query", "")).strip()
               for line in Path(path).expanduser().read_text(encoding="ascii").splitlines() if line.strip()}
    if len(queries) != 1 or not next(iter(queries), ""):
        raise ValueError(f"oracle must contain exactly one non-empty query: {path}")
    return queries.pop()


# Replay one exhaustive label map without making any language-model calls.
def replay_active_labels(
    pool: WindowPool,
    labels: Mapping[tuple[int, int, int, int], str],
    baseline_order: Sequence[int],
    temporary_negatives: int,
    kernel: str,
    seed: int = 0,
) -> dict[str, float | int | str]:
    relevant = {index for index, window in enumerate(pool.windows) if labels.get(window.key) == "relevant"}
    if not relevant:
        return {"kernel": kernel, "temporary_negatives": temporary_negatives, "calls_to_final_positive": 0,
                "baseline_calls_to_final_positive": 0, "fit_seconds": 0.0}
    ordered = list(dict.fromkeys([*baseline_order, *range(len(pool.windows))]))
    baseline = max(ordered.index(index) + 1 for index in relevant)
    rng = np.random.default_rng(seed)
    judged: set[int] = set()
    positives: list[int] = []
    negatives: list[int] = []
    scores = np.zeros(len(pool.windows), dtype=np.float32)
    counts: dict[str, int] = {}
    fit_seconds = 0.0
    calls = 0

    while not relevant.issubset(positives):
        unjudged = [index for index in range(len(pool.windows)) if index not in judged]
        if not unjudged:
            break
        if not positives:
            selected = next(index for index in ordered if index not in judged)
        else:
            hard = sorted(negatives, key=lambda index: -scores[index])[:128]
            temporary = rng.choice(unjudged, min(temporary_negatives, len(unjudged)), replace=False).tolist()
            model, elapsed = _fit_svc(pool.embeddings, positives, hard, temporary, kernel)
            fit_seconds += elapsed
            scores = np.asarray(model.decision_function(pool.embeddings), dtype=np.float32)
            position = calls % 8
            if position < 6:
                selected = max(unjudged, key=lambda index: scores[index])
            elif position == 6:
                selected = min(unjudged, key=lambda index: abs(scores[index]))
            else:
                selected = _cluster_choice(unjudged, pool.windows, counts, scores)
        judged.add(selected)
        calls += 1
        cluster = pool.windows[selected].cluster
        counts[cluster] = counts.get(cluster, 0) + 1
        if selected in relevant:
            positives.append(selected)
        elif labels.get(pool.windows[selected].key) == "not_relevant":
            negatives.append(selected)
    return {
        "kernel": kernel,
        "temporary_negatives": temporary_negatives,
        "calls_to_final_positive": calls,
        "baseline_calls_to_final_positive": baseline,
        "fit_seconds": fit_seconds,
    }


# Compare all requested classifiers and choose one global bounded configuration.
def benchmark_svc_configurations(
    cases: Sequence[tuple[WindowPool, Mapping[tuple[int, int, int, int], str], Sequence[int]]],
    negative_counts: Sequence[int] = (32, 64, 128),
    kernels: Sequence[str] = ("linear", "rbf", "poly"),
) -> dict[str, Any]:
    configurations: list[dict[str, Any]] = []
    for kernel in kernels:
        for count in negative_counts:
            rows = [replay_active_labels(pool, labels, order, count, kernel, index)
                    for index, (pool, labels, order) in enumerate(cases)]
            calls = [int(row["calls_to_final_positive"]) for row in rows]
            configurations.append({
                "kernel": kernel,
                "temporary_negatives": count,
                "worst_calls_to_final_positive": max(calls, default=0),
                "median_calls_to_final_positive": statistics.median(calls) if calls else 0.0,
                "fit_seconds": sum(float(row["fit_seconds"]) for row in rows),
                "cases": rows,
            })
    best_worst = min(row["worst_calls_to_final_positive"] for row in configurations)
    best_median = min(row["median_calls_to_final_positive"] for row in configurations
                      if row["worst_calls_to_final_positive"] == best_worst)
    eligible = [row for row in configurations
                if row["worst_calls_to_final_positive"] == best_worst
                and row["median_calls_to_final_positive"] <= best_median * 1.02]
    selected = min(eligible, key=lambda row: (row["temporary_negatives"], row["fit_seconds"], row["kernel"]))
    baseline_calls = [int(row["cases"][index]["baseline_calls_to_final_positive"])
                      for row in configurations[:1] for index in range(len(row["cases"]))]
    return {"selected": {key: selected[key] for key in ("kernel", "temporary_negatives")},
            "beats_baseline": bool(baseline_calls) and selected["worst_calls_to_final_positive"] < max(baseline_calls),
            "configurations": configurations}


# Replay one or more exhaustive maps over the same rebuilt HKM index.
def main() -> None:
    parser = argparse.ArgumentParser(description="Replay exhaustive small-LM labels through active SVC acquisition.")
    parser.add_argument("index_root")
    parser.add_argument("oracles", nargs="+")
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    pool = WindowPool.from_searcher(Searcher.from_index_root(args.index_root), args.max_windows)
    cases = []
    for path in args.oracles:
        query = _oracle_query(path)
        maps = load_small_lm_map(path)
        if len(maps) != 1:
            raise ValueError(f"each oracle file must contain exactly one query ID: {path}")
        pool.add_semantic_lane("replay_semantic", query)
        pool.add_lexical_lane("replay_lexical", query)
        semantic = sorted(range(len(pool.windows)), key=lambda index: -pool.windows[index].lanes["replay_semantic"])
        lexical = sorted(range(len(pool.windows)), key=lambda index: -pool.windows[index].lanes["replay_lexical"])
        baseline = list(dict.fromkeys(value for pair in zip(semantic, lexical) for value in pair))
        cases.append((pool, next(iter(maps.values())), baseline))
    print(json.dumps(benchmark_svc_configurations(cases), sort_keys=True))


__all__ = [
    "FailureRecord",
    "STATUSES",
    "benchmark_svc_configurations",
    "default_failure_registry_path",
    "load_failure_registry",
    "load_hard_queries",
    "load_small_lm_map",
    "replay_active_labels",
    "validate_failure_registry",
]


if __name__ == "__main__":
    main()
