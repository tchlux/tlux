"""Run self-contained active search with HKM windows and a small local LM.

The local model alone labels relevance and writes query reformulations. An
optional query-specific SVC only orders which unlabeled window it sees next.
Evaluation labels are neither imported nor accepted by this module.

Example:
    hkm-active-search INDEX "a large wall stands over us" --max-seconds 600
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..builder.chunk_io import ChunkReader
from ..search.searcher import Searcher


RELEVANCE_RUBRIC = """Judge only whether this one passage is useful evidence for the
user request. You are labeling an evidence fragment that the search engine may combine
with neighboring fragments, not requiring this passage to satisfy every condition alone.
Match direct synonyms, inflections, and ordinary paraphrases (for example, large/giant or
wall/towering structure). Treat ordinary relational paraphrases as equivalent: a large
wall or structure that towers above, looms over, or surrounds people can answer a request
that says it stands over them; do not demand the exact perspective words. A giant or massive
outer wall enclosing a courtyard is direct evidence even without the words "over us"; the
adjective-plus-wall pair itself is sufficient for this simple description. For example, for
"A large wall stands over us", "a massive outer wall around the courtyard" is relevant
evidence even if the passage does not repeat "over us". A wall mentioned only as a perch or
background, without a relation to people or a surrounding courtyard, is not enough. For a
multi-clue
scene, accept at least two distinctive requested clues (such as chimney plus rope or crowd
plus horror), or an explicit relationship/event, even when another clue is in a neighboring
passage. Reject one generic noun and unsupported inference; do not invent absent details.
For temporal wording such as after or before, do not reject a nearby overlapping fragment
solely because it begins just before the action; accept the scene when the passage contains
both the entry/room event and the requested remark in the same local event. Still require
the requested events and reject a separate later discussion that lacks the entry scene. In
particular, an explicit "funny" or "kidding" remark next to an office, room, door, or
entering event is relevant even when the 128-token window straddles the door motion.
Require a remembered subject or object unless the request says it is forgotten or uncertain.
Use no outside knowledge and do not discuss these judging instructions. Reply in this Markdown
format:
Label: relevant, not_relevant, or uncertain
Evidence: brief reason"""

QUERY_RUBRIC = """Create independent retrieval routes for the request. Preserve its
conditions while varying wording. Reply in Markdown with a `Paraphrases:` heading and
up to four bullets, then a `HyDE:` heading and up to two hypothetical matching-passage
bullets. Never name documents, IDs, rankings, labels, or benchmark answers."""

FOLLOWUP_RUBRIC = """The passage below was judged relevant. Reply in Markdown with a
`Queries:` heading and up to four bullet queries that describe its relevance mode while
retaining every explicit condition in the original request. A relevant passage supplies
alternate wording, not permission to broaden the request. Do not mention rankings,
labels, document IDs, or benchmark answers."""


# Hold one exact stored HKM embedding window and its retrieval-lane evidence.
@dataclass
class WindowCandidate:
    document_id: int
    token_start: int
    token_end: int
    window_size: int
    embedding: np.ndarray
    text: str
    cluster: str
    source_path: str = ""
    lanes: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.document_id, self.token_start, self.token_end, self.window_size)

    # Return a JSON-safe result without exposing classifier state.
    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "window_size": self.window_size,
            "source_path": self.source_path,
            "text": self.text,
            "lanes": dict(self.lanes),
        }


# Bound runtime work and the repeated SVC fits.
@dataclass(frozen=True)
class ActiveSearchConfig:
    max_seconds: float = 600.0
    max_calls: int | None = None
    retrieval_depth: int = 512
    temporary_negatives: int = 64
    hard_negatives: int = 128
    neighbor_count: int = 32
    stagnant_cycles: int = 3
    seed: int = 0
    kernel: str = "rbf"
    trace_path: str | None = None
    resume_trace: bool = False

    def __post_init__(self) -> None:
        if self.max_seconds <= 0.0:
            raise ValueError("max_seconds must be positive")
        if self.max_calls is not None and self.max_calls < 1:
            raise ValueError("max_calls must be positive")
        for name in ("retrieval_depth", "temporary_negatives", "hard_negatives", "neighbor_count", "stagnant_cycles"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.kernel not in {"linear", "rbf", "poly"}:
            raise ValueError("kernel must be linear, rbf, or poly")


# Record one independent small-LM judgment and its acquisition route.
@dataclass(frozen=True)
class ActiveSearchStep:
    call: int
    key: tuple[int, int, int, int]
    acquisition: str
    label: str
    evidence: str
    elapsed_seconds: float
    fit_seconds: float
    queries: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": self.call,
            "key": list(self.key),
            "acquisition": self.acquisition,
            "label": self.label,
            "evidence": self.evidence,
            "elapsed_seconds": self.elapsed_seconds,
            "fit_seconds": self.fit_seconds,
            "queries": list(self.queries),
        }


# Return only explicitly relevant windows plus a complete acquisition trace.
@dataclass(frozen=True)
class ActiveSearchResult:
    query: str
    windows: tuple[WindowCandidate, ...]
    trace: tuple[ActiveSearchStep, ...]
    queries: tuple[str, ...]
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "windows": [window.to_dict() for window in self.windows],
            "trace": [step.to_dict() for step in self.trace],
            "queries": list(self.queries),
            "stop_reason": self.stop_reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)


# Normalize rows for cosine retrieval and SVC training.
def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, np.finfo(np.float32).eps)


# Extract named fields and bullet lists from loose Markdown.
def _markdown_fields(text: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    current = ""
    names = "label|evidence|paraphrases|hyde|queries"
    for raw_line in text.splitlines():
        line = re.sub(r"^#+\s*", "", raw_line.strip().strip("`")).replace("**", "").replace("__", "")
        match = re.match(rf"^({names})\s*:?\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            current = match.group(1).lower()
            fields.setdefault(current, [])
            if match.group(2).strip():
                fields[current].append(match.group(2).strip())
        elif current and line:
            fields[current].append(re.sub(r"^(?:[-*+] |\d+[.)]\s*)", "", line).strip())
    return fields


# Call an OpenAI-compatible local server for all runtime language decisions.
class SmallLM:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4321/v1",
        model: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model or self._models()[0]

    # Send one deterministic chat request and decode its text.
    def _chat(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"])

    # Discover the first locally served chat model.
    def _models(self) -> list[str]:
        with urllib.request.urlopen(f"{self.base_url}/models", timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [str(row["id"]) for row in payload.get("data", []) if row.get("id")]
        models = [model for model in models if "embed" not in model.lower()]
        if not models:
            raise RuntimeError("LM Studio returned no models")
        return models

    # Judge one window without ranks, scores, prior labels, or other candidates.
    def judge(self, query: str, window: WindowCandidate) -> tuple[str, str]:
        response = _markdown_fields(self._chat(
            RELEVANCE_RUBRIC,
            f"USER REQUEST:\n{query}\n\nONE PASSAGE:\n{window.text}",
        ))
        label = response.get("label", ["uncertain"])[0].lower().strip(" .`*")
        if label not in {"relevant", "not_relevant", "uncertain"}:
            label = "uncertain"
        return label, " ".join(response.get("evidence", []))[:1000]

    # Generate initial paraphrase and HyDE retrieval lanes.
    def initial_queries(self, query: str) -> tuple[list[str], list[str]]:
        response = _markdown_fields(self._chat(QUERY_RUBRIC, query))
        return _strings(response.get("paraphrases"), 4), _strings(response.get("hyde"), 2)

    # Generate relevance-mode queries after a new positive.
    def followups(self, query: str, window: WindowCandidate) -> list[str]:
        response = _markdown_fields(self._chat(
            FOLLOWUP_RUBRIC,
            f"ORIGINAL REQUEST:\n{query}\n\nRELEVANT PASSAGE:\n{window.text}",
        ))
        return _strings(response.get("queries"), 4)


# Normalize bounded string arrays returned by the local model.
def _strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


# Hold every eligible stored window and independent retrieval-lane scores.
class WindowPool:
    def __init__(self, windows: Sequence[WindowCandidate], query_embed: Callable[[str], np.ndarray] | None = None) -> None:
        if not windows:
            raise ValueError("window pool must not be empty")
        self.windows = list(windows)
        self.by_key = {window.key: window for window in windows}
        self.index_by_key = {window.key: index for index, window in enumerate(windows)}
        if len(self.by_key) != len(self.windows):
            raise ValueError("window identities must be unique")
        self.embeddings = _normalize(np.stack([window.embedding for window in windows]))
        for window in self.windows:
            window.embedding = np.empty(0, dtype=np.float32)
        self.query_embed = query_embed

    # Load active >=128-token windows directly from published HKM leaves.
    @classmethod
    def from_searcher(cls, searcher: Searcher, max_windows: int | None = None) -> "WindowPool":
        active = searcher._active_doc_ids()
        root = Path(searcher.generation_hkm_root or searcher.hkm_root)
        windows: list[WindowCandidate] = []
        seen: set[tuple[int, int, int, int]] = set()
        document_cache: dict[int, tuple[np.ndarray, str]] = {}
        for node_path in sorted(root.rglob("node.json")):
            node = json.loads(node_path.read_text(encoding="utf-8"))
            if not node.get("is_leaf", False):
                continue
            cluster = str(node_path.parent.relative_to(root))
            for chunk_root in node.get("chunk_roots", []):
                for chunk_path in sorted((node_path.parent / str(chunk_root)).rglob("*.hkmchunk")):
                    reader = ChunkReader(str(chunk_path), metadata_schema=[])
                    for embedding, meta in zip(reader.embeddings, reader.embed_index):
                        doc_id = int(meta["document_id"])
                        start = int(meta["token_start"])
                        end = int(meta["token_end"])
                        size = int(meta["window_size"])
                        key = (doc_id, start, end, size)
                        if doc_id not in active or size < 128 or key in seen:
                            continue
                        if doc_id not in document_cache:
                            tokens, values = searcher._doc_context(doc_id)
                            document_cache[doc_id] = (tokens, searcher._source_path(values))
                        tokens, source_path = document_cache[doc_id]
                        text = searcher._backend().detokenize([tokens[start:end].tolist()])[0].strip()
                        windows.append(WindowCandidate(doc_id, start, end, size, embedding, text, cluster, source_path))
                        seen.add(key)
                        if max_windows is not None and len(windows) >= max_windows:
                            return cls(windows, lambda text: _query_embedding(searcher, text))
        return cls(windows, lambda text: _query_embedding(searcher, text))

    # Score one independent semantic retrieval lane over the full pool.
    def add_semantic_lane(self, name: str, text: str) -> None:
        if self.query_embed is None:
            raise RuntimeError("query embedding is unavailable")
        query = _normalize(self.query_embed(text))[0]
        for window, score in zip(self.windows, self.embeddings @ query):
            window.lanes[name] = float(score)

    # Score a deterministic lexical lane over exact stored-window text.
    def add_lexical_lane(self, name: str, text: str) -> None:
        terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        for window in self.windows:
            present = terms.intersection(re.findall(r"[a-z0-9]+", window.text.lower()))
            window.lanes[name] = len(present) / max(1, len(terms))

    # Score a hybrid lane without collapsing window identity.
    def add_hybrid_lane(self, name: str, semantic: str, lexical: str) -> None:
        for window in self.windows:
            window.lanes[name] = 0.5 * window.lanes[semantic] + 0.5 * window.lanes[lexical]

    # Find the smallest stored context window containing an uncertain span.
    def containing(self, window: WindowCandidate) -> WindowCandidate | None:
        values = [candidate for candidate in self.windows
                  if candidate.document_id == window.document_id
                  and candidate.token_start <= window.token_start
                  and candidate.token_end >= window.token_end
                  and candidate.window_size > window.window_size]
        return min(values, key=lambda candidate: candidate.window_size, default=None)

    # Persist an explicit local experiment cache for interruption-safe replay.
    def save_cache(self, path: str | Path, query: str, index_root: str) -> None:
        payload = {
            "query": query,
            "index_root": str(Path(index_root).expanduser().resolve()),
            "windows": self.windows,
            "embeddings": self.embeddings,
        }
        with Path(path).expanduser().open("wb") as output:
            pickle.dump(payload, output, protocol=pickle.HIGHEST_PROTOCOL)

    # Load only an explicitly requested local experiment cache.
    @classmethod
    def load_cache(
        cls,
        path: str | Path,
        query: str,
        index_root: str,
        query_embed: Callable[[str], np.ndarray],
    ) -> "WindowPool":
        with Path(path).expanduser().open("rb") as source:
            payload = pickle.load(source)
        expected = str(Path(index_root).expanduser().resolve())
        if payload.get("query") != query or payload.get("index_root") != expected:
            raise ValueError("pool cache query or index does not match")
        pool = cls.__new__(cls)
        pool.windows = payload["windows"]
        pool.by_key = {window.key: window for window in pool.windows}
        pool.index_by_key = {window.key: index for index, window in enumerate(pool.windows)}
        pool.embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        pool.query_embed = query_embed
        return pool


# Embed a query using the index's configured backend.
def _query_embedding(searcher: Searcher, text: str) -> np.ndarray:
    backend = searcher._backend()
    return np.asarray(backend.embed(backend.tokenize([text]), role="query")[0], dtype=np.float32)


# Fit exactly one bounded acquisition classifier.
def _fit_svc(
    embeddings: np.ndarray,
    positives: Sequence[int],
    hard_negatives: Sequence[int],
    temporary_negatives: Sequence[int],
    kernel: str = "rbf",
) -> tuple[Any, float]:
    try:
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError("active search requires the experimental scikit-learn dependency") from exc
    negative = list(dict.fromkeys([*hard_negatives, *temporary_negatives]))
    indexes = [*positives, *negative]
    labels = np.asarray([1] * len(positives) + [0] * len(negative), dtype=np.int8)
    if not positives or not negative:
        raise ValueError("SVC fit requires positive and negative examples")
    model = SVC(C=1, gamma="scale", kernel=kernel, degree=2, class_weight="balanced")
    started = time.monotonic()
    model.fit(embeddings[indexes], labels)
    return model, time.monotonic() - started


Judge = Callable[[str, WindowCandidate], tuple[str, str]]
Reformulator = Callable[[str, WindowCandidate], Sequence[str]]


# Normalize a callback judgment to the runtime's three explicit states.
def _judge(judge: Judge, query: str, window: WindowCandidate) -> tuple[str, str]:
    label, evidence = judge(query, window)
    label = label if label in {"relevant", "not_relevant", "uncertain"} else "uncertain"
    return label, str(evidence)[:1000]


# Discover small-LM positives while minimizing independent passage judgments.
class ActiveSearchAgent:
    def __init__(
        self,
        pool: WindowPool,
        judge: Judge,
        reformulator: Reformulator | None = None,
        config: ActiveSearchConfig | None = None,
    ) -> None:
        self.pool = pool
        self.judge = judge
        self.reformulator = reformulator
        self.config = config or ActiveSearchConfig()

    # Run the per-query label and acquisition loop.
    def run(self, query: str) -> ActiveSearchResult:
        query = " ".join(query.split())
        if not query:
            raise ValueError("query must not be empty")
        rng = np.random.default_rng(self.config.seed)
        judged: dict[int, str] = {}
        positives: list[int] = []
        negatives: list[int] = []
        queries = [query]
        trace: list[ActiveSearchStep] = []
        lane_order = list(dict.fromkeys(lane for window in self.pool.windows for lane in window.lanes))
        lane_ranks = {lane: sorted(range(len(self.pool.windows)), key=lambda i: -self.pool.windows[i].lanes.get(lane, -np.inf))
                      for lane in lane_order}
        available = {index for lane in lane_order for index in lane_ranks[lane][:self.config.retrieval_depth]}
        if not available:
            available = set(range(len(self.pool.windows)))
        previous_scores = np.zeros(len(self.pool.windows), dtype=np.float32)
        cluster_counts: dict[str, int] = {}
        started = time.monotonic()
        cycle_positive = False
        acquisition_calls = 0
        lane_turn = 0
        empty_cycles = 0
        stop_reason = "all_windows_judged"

        # Restore flushed labels after an interrupted local run.
        if self.config.resume_trace and self.config.trace_path and Path(self.config.trace_path).exists():
            for line in Path(self.config.trace_path).read_text(encoding="ascii").splitlines():
                row = json.loads(line)
                key = tuple(int(value) for value in row["key"])
                index = self.pool.index_by_key[key]
                label = str(row["label"])
                step = ActiveSearchStep(
                    int(row["call"]), key, str(row["acquisition"]), label,
                    str(row.get("evidence", "")), float(row.get("elapsed_seconds", 0.0)),
                    float(row.get("fit_seconds", 0.0)), tuple(row.get("queries", ())),
                )
                trace.append(step)
                judged[index] = label
                cluster = self.pool.windows[index].cluster
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                if label == "relevant":
                    positives.append(index)
                elif label == "not_relevant":
                    negatives.append(index)
                if step.queries:
                    queries = list(step.queries)
                if step.acquisition != "larger_context":
                    acquisition_calls += 1
            for index in positives:
                scores = self.pool.embeddings @ self.pool.embeddings[index]
                available.update(int(value) for value in np.argsort(-scores)[:self.config.neighbor_count])
            for number, text in enumerate(queries[1:], 1):
                lane = f"restored:{number}"
                self.pool.add_semantic_lane(lane, text)
                ranked = sorted(range(len(self.pool.windows)), key=lambda value: -self.pool.windows[value].lanes[lane])
                available.update(ranked[:self.config.retrieval_depth])
            rng = np.random.default_rng(self.config.seed + len(trace))
            standard = [step for step in trace if step.acquisition != "larger_context"]
            seen_positive = False
            for start in range(0, len(standard), 8):
                rows = standard[start:start + 8]
                cycle_has_positive = any(step.label == "relevant" for step in rows)
                seen_positive = seen_positive or cycle_has_positive
                if len(rows) == 8 and seen_positive:
                    empty_cycles = 0 if cycle_has_positive else empty_cycles + 1
                elif len(rows) < 8:
                    cycle_positive = cycle_has_positive

        while len(judged) < len(self.pool.windows):
            elapsed = time.monotonic() - started
            if elapsed >= self.config.max_seconds:
                stop_reason = "time_budget_exhausted"
                break
            if self.config.max_calls is not None and len(trace) >= self.config.max_calls:
                stop_reason = "call_budget_exhausted"
                break
            unjudged = [index for index in range(len(self.pool.windows)) if index not in judged]
            acquisition = "bootstrap"
            fit_seconds = 0.0
            if positives:
                hard = sorted(negatives, key=lambda i: -previous_scores[i])[:self.config.hard_negatives]
                temporary_pool = [index for index in unjudged if index not in positives]
                temporary = rng.choice(
                    temporary_pool,
                    size=min(self.config.temporary_negatives, len(temporary_pool)),
                    replace=False,
                ).tolist()
                model, fit_seconds = _fit_svc(self.pool.embeddings, positives, hard, temporary, self.config.kernel)
                previous_scores = np.asarray(model.decision_function(self.pool.embeddings), dtype=np.float32)
                position = acquisition_calls % 8
                acquisition_pool = [index for index in unjudged if index in available] or unjudged
                if position < 5:
                    acquisition = "exploitation"
                    selected = max(acquisition_pool, key=lambda i: previous_scores[i])
                elif position == 5:
                    selected, acquisition = _lane_choice(
                        unjudged, lane_order, lane_ranks,
                        lane_turn,
                    )
                    lane_turn += 1
                elif position == 6:
                    acquisition = "boundary"
                    selected = min(acquisition_pool, key=lambda i: abs(previous_scores[i]))
                else:
                    acquisition = "cluster_exploration"
                    selected = _cluster_choice(unjudged, self.pool.windows, cluster_counts, previous_scores)
                    available.add(selected)
            else:
                selected, acquisition = _bootstrap_choice(
                    unjudged, self.pool.windows, lane_order, lane_ranks, cluster_counts, acquisition_calls)

            label, reason = _judge(self.judge, query, self.pool.windows[selected])
            judged[selected] = label
            cluster = self.pool.windows[selected].cluster
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            if label == "relevant":
                positives.append(selected)
                cycle_positive = True
                available.update(self._expand_positive(query, selected, queries))
                # Add newly generated follow-up lanes to the next lane-diversity slot.
                for lane in dict.fromkeys(lane for window in self.pool.windows for lane in window.lanes):
                    if lane not in lane_ranks and not lane.startswith("neighbor:"):
                        lane_order.append(lane)
                        lane_ranks[lane] = sorted(
                            range(len(self.pool.windows)),
                            key=lambda i: -self.pool.windows[i].lanes.get(lane, -np.inf),
                        )
            elif label == "not_relevant":
                negatives.append(selected)
            trace.append(ActiveSearchStep(
                len(trace) + 1,
                self.pool.windows[selected].key,
                acquisition,
                label,
                reason,
                time.monotonic() - started,
                fit_seconds,
                tuple(queries),
            ))
            self._write_step(trace[-1])
            acquisition_calls += 1

            # Inspect the existing larger HKM context only after an uncertain short window.
            call_available = self.config.max_calls is None or len(trace) < self.config.max_calls
            time_available = time.monotonic() - started < self.config.max_seconds
            if label == "uncertain" and self.pool.windows[selected].window_size == 128 and call_available and time_available:
                context = self.pool.containing(self.pool.windows[selected])
                if context is not None and self.pool.index_by_key[context.key] not in judged:
                    context_index = self.pool.index_by_key[context.key]
                    context_label, context_reason = _judge(self.judge, query, context)
                    judged[context_index] = context_label
                    if context_label == "relevant":
                        positives.append(context_index)
                        cycle_positive = True
                        available.update(self._expand_positive(query, context_index, queries))
                    elif context_label == "not_relevant":
                        negatives.append(context_index)
                    trace.append(ActiveSearchStep(
                        len(trace) + 1, context.key, "larger_context", context_label,
                        context_reason, time.monotonic() - started, 0.0, tuple(queries),
                    ))
                    self._write_step(trace[-1])

            # A complete 6/1/1 cycle must contain a positive to reset convergence.
            if positives and acquisition_calls % 8 == 0:
                empty_cycles = 0 if cycle_positive else empty_cycles + 1
                cycle_positive = False
                if empty_cycles >= self.config.stagnant_cycles:
                    stop_reason = "three_cycles_without_new_positives"
                    break

        result_windows = tuple(self.pool.windows[index] for index in positives)
        return ActiveSearchResult(query, result_windows, tuple(trace), tuple(queries), stop_reason)

    # Flush one optional JSONL trace row so interrupted long runs remain visible.
    def _write_step(self, step: ActiveSearchStep) -> None:
        if self.config.trace_path:
            with Path(self.config.trace_path).expanduser().open("a", encoding="ascii") as output:
                output.write(json.dumps(step.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")

    # Add neighbors and small-LM relevance-mode lanes after a positive.
    def _expand_positive(self, query: str, index: int, queries: list[str]) -> set[int]:
        scores = self.pool.embeddings @ self.pool.embeddings[index]
        added = {int(value) for value in np.argsort(-scores)[:self.config.neighbor_count]}
        for neighbor in added:
            self.pool.windows[int(neighbor)].lanes[f"neighbor:{len(queries)}"] = float(scores[neighbor])
        if self.reformulator is None or self.pool.query_embed is None:
            return added
        try:
            reformulations = self.reformulator(query, self.pool.windows[index])
        except (OSError, KeyError, RuntimeError, ValueError):
            return added
        for text in reformulations:
            normalized = " ".join(str(text).split())
            if normalized and normalized not in queries:
                queries.append(normalized)
                lane = f"followup:{len(queries)}"
                self.pool.add_semantic_lane(lane, normalized)
                ranked = sorted(range(len(self.pool.windows)), key=lambda value: -self.pool.windows[value].lanes[lane])
                added.update(ranked[:self.config.retrieval_depth])
        return added


# Choose alternating deep-lane and under-sampled-cluster bootstrap candidates.
def _bootstrap_choice(
    unjudged: Sequence[int],
    windows: Sequence[WindowCandidate],
    lane_order: Sequence[str],
    lane_ranks: dict[str, list[int]],
    cluster_counts: dict[str, int],
    call: int,
) -> tuple[int, str]:
    if call % 2 or not lane_order:
        return _cluster_choice(unjudged, windows, cluster_counts), "bootstrap_cluster"
    # Interleave base, paraphrase, and HyDE lanes before deepening any one lane.
    primary = [lane for lane in ("semantic", "lexical", "hybrid") if lane in lane_ranks]
    groups = ("semantic", "lexical", "hybrid", "paraphrase", "hyde", "semantic",
              "lexical", "hybrid", "followup")
    round_number = call // 2
    lanes = primary + [lane for lane in lane_order if lane not in primary]
    group = groups[round_number % len(groups)] if primary else ""
    matches = [lane for lane in lanes if lane == group or lane.startswith(f"{group}:")]
    lane = matches[0] if matches else lanes[round_number % len(lanes)]
    available = set(unjudged)
    ranked = lane_ranks[lane]
    # Probe progressively deeper ranks before a positive seed, while retaining
    # the earliest candidates for the first few passes.
    occurrence = sum(
        1 for index in range(round_number + 1)
        if groups[index % len(groups)] == group
    )
    depths = (0, 1, 2, 48, 96, 192, 384, 768)
    if group == "hyde":
        depths = (0, 1, 48, 96, 192, 384, 768, 1024)
    depth = depths[min(occurrence - 1, len(depths) - 1)] if group else 0
    target = min(depth, len(ranked) - 1)
    for offset in range(len(ranked)):
        for position in (target + offset, target - offset):
            if 0 <= position < len(ranked) and ranked[position] in available:
                return ranked[position], f"bootstrap_lane:{lane}"
    return unjudged[0], f"bootstrap_lane:{lane}"


# Select a frontier result from a rotating retrieval lane.
def _lane_choice(
    unjudged: Sequence[int],
    lane_order: Sequence[str],
    lane_ranks: dict[str, list[int]],
    cycle: int,
) -> tuple[int, str]:
    base = [lane for lane in ("semantic", "lexical", "hybrid") if lane in lane_ranks]
    hyde = sorted(lane for lane in lane_order if lane.startswith("hyde:") and lane in lane_ranks)
    paraphrase = sorted(lane for lane in lane_order if lane.startswith("paraphrase:") and lane in lane_ranks)
    other = [lane for lane in lane_order if lane in lane_ranks and lane not in base + hyde + paraphrase]
    lanes = base + hyde + paraphrase + other
    if not lanes:
        return unjudged[0], "lane_exploration"
    start = cycle % len(lanes)
    for offset in range(len(lanes)):
        lane = lanes[(start + offset) % len(lanes)]
        ranked = [index for index in lane_ranks[lane] if index in unjudged]
        if ranked:
            target = 48 if lane.startswith("hyde:") else 0
            target = min(target, len(lane_ranks[lane]) - 1)
            for distance in range(len(lane_ranks[lane])):
                for position in (target + distance, target - distance):
                    if 0 <= position < len(lane_ranks[lane]):
                        candidate = lane_ranks[lane][position]
                        if candidate in unjudged:
                            return candidate, f"lane_exploration:{lane}"
    return unjudged[0], "lane_exploration"


# Select the best candidate from the least-inspected HKM cluster.
def _cluster_choice(
    indexes: Sequence[int],
    windows: Sequence[WindowCandidate],
    counts: dict[str, int],
    scores: np.ndarray | None = None,
) -> int:
    minimum = min(counts.get(windows[index].cluster, 0) for index in indexes)
    eligible = [index for index in indexes if counts.get(windows[index].cluster, 0) == minimum]
    if scores is None:
        return min(eligible)
    return max(eligible, key=lambda index: scores[index])


# Build independent original, lexical, hybrid, paraphrase, and HyDE lanes.
def prepare_pool(searcher: Searcher, model: SmallLM, query: str, max_windows: int | None = None) -> WindowPool:
    pool = WindowPool.from_searcher(searcher, max_windows)
    pool.add_semantic_lane("semantic", query)
    pool.add_lexical_lane("lexical", query)
    pool.add_hybrid_lane("hybrid", "semantic", "lexical")
    paraphrases, hyde = model.initial_queries(query)
    for index, text in enumerate(paraphrases):
        pool.add_semantic_lane(f"paraphrase:{index}", text)
    for index, text in enumerate(hyde):
        pool.add_semantic_lane(f"hyde:{index}", text)
    return pool


# Exhaustively persist small-LM labels as an evaluation-only JSONL oracle.
def write_exhaustive_map(
    path: str | Path,
    query_id: str,
    query: str,
    pool: WindowPool,
    judge: Judge,
    workers: int = 1,
    resume: bool = True,
    retries: int = 3,
) -> dict[str, int]:
    if workers < 1 or retries < 0:
        raise ValueError("workers must be positive and retries must be non-negative")
    counts = {"relevant": 0, "not_relevant": 0, "uncertain": 0}
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[int, int, int, int]] = set()
    if resume and destination.exists():
        for line in destination.read_text(encoding="ascii").splitlines():
            row = json.loads(line)
            if row.get("query_id") != query_id or row.get("query") != query or row.get("judge") != "small_lm":
                raise ValueError(f"existing oracle does not match this small-LM query: {destination}")
            key = tuple(int(value) for value in row["key"])
            label = str(row["label"])
            if len(key) != 4 or label not in counts or key in completed:
                raise ValueError(f"existing oracle has an invalid or duplicate row: {destination}")
            completed.add(key)
            counts[label] += 1
    pending = [window for window in pool.windows if window.key not in completed]
    mode = "a" if resume and destination.exists() else "w"

    def evaluate(window: WindowCandidate) -> tuple[WindowCandidate, str, str]:
        for attempt in range(retries + 1):
            try:
                label, evidence = _judge(judge, query, window)
                return window, label, evidence
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable judgment retry state")

    with destination.open(mode, encoding="ascii") as output, ThreadPoolExecutor(max_workers=workers) as executor:
        batch_size = workers * 4
        for start in range(0, len(pending), batch_size):
            for window, label, evidence in executor.map(evaluate, pending[start:start + batch_size]):
                counts[label] += 1
                output.write(json.dumps({
                    "query_id": query_id,
                    "query": query,
                    "key": list(window.key),
                    "cluster": window.cluster,
                    "source_path": window.source_path,
                    "label": label,
                    "evidence": evidence,
                    "judge": "small_lm",
                }, ensure_ascii=True, sort_keys=True) + "\n")
                output.flush()
    return counts


# Run one active query or emit an exhaustive small-LM relevance map.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run small-LM active search over exact HKM windows.")
    parser.add_argument("index_root")
    parser.add_argument("query")
    parser.add_argument("--base-url", default="http://127.0.0.1:4321/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--temporary-negatives", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--kernel", choices=["linear", "rbf", "poly"], default="rbf")
    parser.add_argument("--exhaustive-map", default=None)
    parser.add_argument("--query-id", default="query")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--resume-trace", action="store_true")
    parser.add_argument("--pool-cache", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    searcher = Searcher.from_index_root(args.index_root)
    model = SmallLM(args.base_url, args.model, args.timeout)
    if args.exhaustive_map:
        pool = WindowPool.from_searcher(searcher, args.max_windows)
        counts = write_exhaustive_map(
            args.exhaustive_map, args.query_id, args.query, pool, model.judge,
            args.workers, not args.overwrite, args.retries,
        )
        print(json.dumps({"path": args.exhaustive_map, "counts": counts}, sort_keys=True))
        return
    cache = Path(args.pool_cache).expanduser() if args.pool_cache else None
    if cache is not None and cache.exists():
        pool = WindowPool.load_cache(cache, args.query, args.index_root, lambda text: _query_embedding(searcher, text))
    else:
        pool = prepare_pool(searcher, model, args.query, args.max_windows)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            pool.save_cache(cache, args.query, args.index_root)
    if args.prepare_only:
        print(json.dumps({"pool_cache": str(cache or ""), "windows": len(pool.windows)}, sort_keys=True))
        return
    if args.trace_output and not args.resume_trace:
        Path(args.trace_output).expanduser().write_text("", encoding="ascii")
    result = ActiveSearchAgent(
        pool,
        model.judge,
        model.followups,
        ActiveSearchConfig(
            max_seconds=args.max_seconds,
            max_calls=args.max_calls,
            temporary_negatives=args.temporary_negatives,
            kernel=args.kernel,
            trace_path=args.trace_output,
            resume_trace=args.resume_trace,
        ),
    ).run(args.query)
    print(result.to_json())


if __name__ == "__main__":
    main()
