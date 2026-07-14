"""Evaluate local agents against sampled HKM passages.

The benchmark uses an OpenAI-compatible LM Studio endpoint when available and
falls back to deterministic agents for offline regression tests. It reports
model first-pass quality separately from an evidence-grounded tool result.
"""

from __future__ import annotations

import argparse
import copy
import http.client
import json
import math
import random
import re
import socket
import statistics
import time
import urllib.error
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol
from urllib.parse import urlsplit

from ..search.searcher import Searcher


STOP_WORDS = {
    "about", "after", "again", "also", "because", "before", "being", "could",
    "every", "first", "from", "have", "into", "just", "more", "other", "over",
    "said", "some", "than", "that", "their", "there", "these", "they", "this",
    "through", "under", "what", "when", "where", "which", "while", "with", "would",
}
TOOL_QUERY_WORDS = 16
TOOL_MAX_TOKENS = 16
PLANNER_MAX_TOKENS = 16
PLANNER_INPUT_WORDS = 64
PLANNER_QUERY_CACHE_SIZE = 256
NATIVE_TOOL_MAX_TOKENS = 32
TOOL_KEYWORD_WORDS = 6
TOOL_RESCUE_QUERIES = 8
TOOL_FALLBACK_SCORE = 0.7
TOOL_EXPANSION_FACTOR = 3
LMSTUDIO_TIMEOUT = 1.0
LMSTUDIO_WARMUP_TIMEOUT = 15.0
LANGUAGE_QUERY_MAX_WORDS = 24
LANGUAGE_QUERY_MAX_VARIANTS = 7
LANGUAGE_PLAN_MAX_VARIANTS = 4
LANGUAGE_QUERY_MAX_ROUNDS = 2
LANGUAGE_CONCEPT_GROUPS = (
    frozenset({"amusing", "funny", "humorous", "laugh", "laughter", "joke", "silly"}),
    frozenset({"enter", "entered", "enters", "door", "room", "office"}),
    frozenset({"climb", "climbs", "climbing", "ascending", "scaling"}),
    frozenset({"night", "nighttime", "midnight", "dark", "dusk"}),
    frozenset({"wall", "tower", "structure", "cliff", "chimney", "parapet"}),
    frozenset({"crowd", "crowds", "people", "gather", "gathers", "below"}),
    frozenset({"watch", "watched", "watching", "spectators", "horror", "horrified", "fear", "scream"}),
)
LANGUAGE_CONCEPT_LANES = (
    (LANGUAGE_CONCEPT_GROUPS[0], "funny humorous joke"),
    (LANGUAGE_CONCEPT_GROUPS[1], "door room office"),
    (LANGUAGE_CONCEPT_GROUPS[2], "climb chimney"),
    (LANGUAGE_CONCEPT_GROUPS[3], "night dark"),
    (LANGUAGE_CONCEPT_GROUPS[4], "large wall structure"),
    (LANGUAGE_CONCEPT_GROUPS[5], "crowd people below"),
    (LANGUAGE_CONCEPT_GROUPS[6], "watch horror"),
)
QUERY_GUARD_WORDS = STOP_WORDS | {
    "a", "an", "and", "as", "at", "by", "for", "in", "is", "it", "of",
    "on", "or", "the", "to", "was", "were", "will", "you", "your",
    "us", "falls", "even",
}
QUERY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "search_query",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
LANGUAGE_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "language_search_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}},
                "exclude_terms": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["queries", "exclude_terms"],
        },
    },
}


# One sampled raw passage and its exact indexed document target.
@dataclass
class Sample:
    sample_id: int
    doc_id: int
    source_path: str
    excerpt: str
    token_start: int
    token_end: int


# Query planner interface shared by LM Studio and deterministic tests.
class QueryGenerator(Protocol):
    name: str

    def generate(self, excerpt: str) -> str:
        ...


# Return a compact phrase from evidence for exact fallback retrieval.
def _phrase_query(excerpt: str, limit: int = 8) -> str:
    if len(excerpt.split()) <= limit:
        return excerpt.strip()
    return " ".join(excerpt.split()[:limit])


# Bound model prompt size while preserving both ends of a raw passage.
def _planner_excerpt(excerpt: str, limit: int = PLANNER_INPUT_WORDS) -> str:
    words = excerpt.split()
    if len(words) <= limit:
        return excerpt
    head = limit // 2
    return " ".join(words[:head] + words[-(limit - head):])


# Return several short raw phrases for evidence-assisted exact retrieval.
def _phrase_queries(excerpt: str, limit: int = 8, count: int = 6) -> List[str]:
    words = excerpt.split()
    if len(words) <= limit:
        return [excerpt.strip()]
    starts = list(dict.fromkeys([0, len(words) // 4, len(words) // 2, (3 * len(words)) // 4, len(words) - limit]))
    return [" ".join(words[start:start + limit]) for start in starts[:count]]


# Return bounded fallback queries, including distinctive code identifiers.
def _fallback_queries(excerpt: str) -> List[str]:
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_'-]{3,}", excerpt)
    repeated = list(dict.fromkeys(
        word for word in identifiers if identifiers.count(word) > 1 and len(word) >= 6
    ))
    distinctive = repeated + [
        word for word in identifiers
        if "_" in word or any(char.isdigit() for char in word) or len(word) >= 12
    ]
    queries = [_keyword_query(excerpt)] + _phrase_queries(excerpt) + list(dict.fromkeys(distinctive))[:8]
    return list(dict.fromkeys(query for query in queries if query.strip()))


# Return distinctive terms for a deterministic offline query planner.
def _keyword_query(excerpt: str, limit: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_'-]*", excerpt)
    unique = list(dict.fromkeys(word for word in words if len(word) >= 4 and word.lower() not in STOP_WORDS))
    ranked = sorted(
        enumerate(unique),
        key=lambda item: (
            0 if "_" in item[1] or any(char.isdigit() for char in item[1]) else 1,
            -len(item[1]),
            item[0],
        ),
    )
    selected = ranked[:limit] or list(enumerate(words[:limit]))
    return " ".join(word for _, word in selected) or _phrase_query(excerpt, limit)


# Split a natural-language request into condition clauses.
#
# Arguments:
#   query (str): Natural-language user query.
#
# Returns:
#   (list[str]): Non-empty clauses in source order.
#
def _language_query_clauses(query: str) -> List[str]:
    normalized = " ".join(query.split())
    if not normalized:
        return []
    return [part.strip() for part in re.split(
        r"\b(?:and|but|while|when|where|because|although|though|with|without|if|unless|despite|after|before|except|until)\b|[,;:]",
        normalized,
        flags=re.IGNORECASE,
    ) if part.strip()]


# Add compact synonym lanes for concepts explicitly present in a request.
def _language_concept_queries(query: str) -> List[str]:
    forms = _language_word_forms(query)
    return [phrase for group, phrase in LANGUAGE_CONCEPT_LANES if forms.intersection(group)]


# Return short language-query variants while preserving content words.
def _language_query_variants(query: str) -> List[str]:
    normalized = " ".join(query.split())
    if not normalized:
        return []
    clauses = _language_query_clauses(normalized)
    variants = [normalized]
    variants.extend(clauses)
    variants.extend(_language_concept_queries(normalized))
    variants.extend(
        f"{clauses[index]} {clauses[index + 1]}"
        for index in range(len(clauses) - 1)
        if len(_language_word_forms(clauses[index])) >= 2
        and len(_language_word_forms(clauses[index + 1])) >= 2
    )
    variants.append(_keyword_query(normalized, limit=12))
    return list(dict.fromkeys(
        " ".join(value.split()[:LANGUAGE_QUERY_MAX_WORDS])
        for value in variants
        if value.strip()
    ))[:LANGUAGE_QUERY_MAX_VARIANTS]


# Return simple lexical forms for language-query coverage and exclusions.
def _language_word_forms(text: str) -> set[str]:
    terms = set(re.findall(r"[A-Za-z0-9]+", text.lower())) - QUERY_GUARD_WORDS
    forms = set(terms)
    for term in terms:
        if len(term) > 5:
            for suffix in ("ing", "ed", "es", "s"):
                if term.endswith(suffix) and len(term) - len(suffix) >= 4:
                    forms.add(term[:-len(suffix)])
    return forms


# Extract lexical clauses introduced by an explicit negative condition.
def _language_negative_clauses(query: str) -> List[set[str]]:
    clauses: List[set[str]] = []
    event_negation = {
        "to", "be", "being", "been", "have", "has", "had", "do", "does",
        "did", "leave", "leaving", "go", "going", "remember", "remembered",
        "forget", "forgot", "want", "need", "like", "know", "stand", "say",
        "said", "see", "seen", "take", "taking",
    }
    match_pattern = (
        r"\b(?:not|without|except|excluding|rather than|instead of|no)\b"
        r"(.+?)(?=\b(?:and|but|while|when|where|because|although|though)\b|[,;:.]|$)"
    )
    for match in re.finditer(match_pattern, query, flags=re.IGNORECASE):
        remainder = match.group(1).strip()
        marker = match.group(0).split(None, 1)[0].lower()
        first_word = remainder.split(None, 1)[0].lower() if remainder else ""
        if marker == "not" and first_word in event_negation:
            continue
        terms = _language_word_forms(remainder)
        if terms:
            clauses.append(terms)
    return clauses


# Keep explicit negative clauses out of positive condition scoring.
def _language_positive_clauses(query: str) -> List[str]:
    negative_marker = re.compile(
        r"\b(?:not|without|except|excluding|rather than|instead of|no)\b",
        flags=re.IGNORECASE,
    )
    negative_terms = _language_negative_clauses(query)
    return [
        clause for clause in _language_query_clauses(query)
        if not negative_marker.search(clause)
        and not any(
            len(_language_word_forms(clause).intersection(terms)) >= min(2, len(terms))
            for terms in negative_terms
        )
    ]


# Expand lexical forms into a small set of condition concepts for reranking.
def _language_concept_forms(text: str) -> set[str]:
    forms = _language_word_forms(text)
    expanded = set(forms)
    for group in LANGUAGE_CONCEPT_GROUPS[:2]:
        if forms.intersection(group):
            expanded.update(group)
    return expanded


# Expand every known condition group when a query contains an explicit contrast.
def _language_full_concept_forms(text: str) -> set[str]:
    forms = _language_word_forms(text)
    expanded = set(forms)
    for group in LANGUAGE_CONCEPT_GROUPS:
        if forms.intersection(group):
            expanded.update(group)
    return expanded


# Decode a bounded language-query plan from a model response.
#
# Arguments:
#   response (str): Model response containing JSON.
#
# Returns:
#   (dict[str, list[str]]): Queries and soft exclusion terms.
#
def _parse_language_plan(response: str) -> Dict[str, List[str]]:
    text = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        value = json.loads(match.group(0)) if match else {}
    if not isinstance(value, dict):
        raise ValueError("language planner returned a non-object")
    queries = [
        " ".join(str(item).split()[:LANGUAGE_QUERY_MAX_WORDS])
        for item in value.get("queries", [])
        if str(item).strip()
    ]
    exclude_terms = [
        " ".join(str(item).split()[:3])
        for item in value.get("exclude_terms", [])
        if str(item).strip()
    ]
    if not queries:
        raise ValueError("language planner returned no queries")
    return {
        "queries": list(dict.fromkeys(queries))[:LANGUAGE_PLAN_MAX_VARIANTS],
        "exclude_terms": list(dict.fromkeys(exclude_terms))[:LANGUAGE_PLAN_MAX_VARIANTS],
    }


# Parse a model response into one bounded search query.
def parse_query(response: str) -> str:
    text = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            text = str(value.get("query", ""))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                text = str(value.get("query", ""))
            except json.JSONDecodeError:
                pass
    if text.startswith("{"):
        match = re.search(r'"query"\s*:\s*"((?:\\.|[^"\\])*)', text, re.DOTALL)
        if match:
            text = match.group(1).replace('\\"', '"').replace("\\\\", "\\")
    text = " ".join(text.split())
    if not text or text.lower().startswith("thinking process:"):
        raise ValueError("query planner returned an empty query")
    return text[:256]


# Call the local OpenAI-compatible LM Studio server.
class LMStudioQueryGenerator:
    name = "lmstudio"

    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model: str | None = None, timeout: float = LMSTUDIO_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("base_url must be an http(s) URL")
        self._http_parts = parts
        self._connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None

    # Return a lazy persistent connection to the configured LM Studio server.
    def _http_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        if self._connection is None:
            connection_type = http.client.HTTPSConnection if self._http_parts.scheme == "https" else http.client.HTTPConnection
            self._connection = connection_type(
                self._http_parts.hostname,
                self._http_parts.port,
                timeout=self.timeout,
            )
        self._connection.timeout = self.timeout
        if self._connection.sock is not None:
            self._connection.sock.settimeout(self.timeout)
        return self._connection

    # Close a broken persistent connection before the next request recreates it.
    def _close_http_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _request(self, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        target = f"{self._http_parts.path.rstrip('/')}/{path.lstrip('/')}"
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(2):
            try:
                connection = self._http_connection()
                connection.request(
                    "POST" if body is not None else "GET",
                    target,
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                data = response.read()
                if response.status >= 400:
                    raise urllib.error.HTTPError(
                        url,
                        response.status,
                        response.reason,
                        response.headers,
                        None,
                    )
                return json.loads(data.decode("utf-8"))
            except urllib.error.HTTPError:
                if payload is None:
                    raise
                retry = dict(payload)
                removed = False
                for option in ("reasoning_effort", "response_format"):
                    if option in retry:
                        retry.pop(option)
                        removed = True
                        break
                if not removed:
                    raise
                return self._request(path, retry)
            except json.JSONDecodeError:
                self._close_http_connection()
                if attempt:
                    raise
            except (TimeoutError, socket.timeout):
                self._close_http_connection()
                raise
            except (OSError, http.client.HTTPException):
                self._close_http_connection()
                if attempt:
                    raise

    def _model_name(self) -> str:
        if self.model:
            return self.model
        models = self._request("models").get("data", [])
        if not models:
            raise RuntimeError("LM Studio has no model available at the configured endpoint")
        self.model = str(models[0]["id"])
        return self.model

    def generate(self, excerpt: str) -> str:
        planner_excerpt = _planner_excerpt(excerpt)
        if planner_excerpt != excerpt:
            planner_excerpt += "\nCandidate terms from full passage: " + _keyword_query(excerpt)
        prompt = (
            "Extract a search query. Return only {\"query\":\"...\"}; copy 1-3 exact "
            "words from evidence, preferring rare names, identifiers, or numbers. No explanation.\n"
            f"Evidence:\n{planner_excerpt}"
        )
        response = self._request("chat/completions", {
            "model": self._model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": PLANNER_MAX_TOKENS,
            "reasoning_effort": "none",
            "response_format": QUERY_RESPONSE_FORMAT,
            "stream": False,
        })
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("LM Studio returned no chat completion choices")
        message = choices[0].get("message", {})
        content = message.get("content") or message.get("reasoning_content", "")
        query = " ".join(parse_query(str(content)).split()[:TOOL_QUERY_WORDS])
        query_terms = set(re.findall(r"[A-Za-z0-9]+", query.lower())) - QUERY_GUARD_WORDS
        evidence_terms = set(re.findall(r"[A-Za-z0-9]+", excerpt.lower())) - QUERY_GUARD_WORDS
        if not query_terms.intersection(evidence_terms):
            raise ValueError("LM Studio query did not quote the supplied evidence")
        return query

    # Turn raw evidence into a bounded natural-language memory request.
    def generate_memory_query(self, excerpt: str, style: str = "specific") -> str:
        guidance = {
            "vague": "Use an imprecise conversational description while retaining concrete clues.",
            "specific": "State the concrete event and relevant entities clearly.",
            "conditional": "Use an if, after, unless, or while condition.",
            "missing_entity": "Omit one named subject or object and say it is forgotten.",
        }
        if style not in guidance:
            raise ValueError(f"unknown memory-query style: {style}")
        prompt = (
            "Write one natural-language memory search request from the evidence. "
            f"{guidance[style]} Preserve the other details. Return only "
            '{"query":"..."}; do not answer or explain.\nEvidence:\n'
            f"{_planner_excerpt(excerpt, 160)}"
        )
        response = self._request("chat/completions", {
            "model": self._model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": PLANNER_MAX_TOKENS,
            "reasoning_effort": "none",
            "response_format": QUERY_RESPONSE_FORMAT,
            "stream": False,
        })
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("LM Studio returned no memory-query choices")
        message = choices[0].get("message", {})
        content = message.get("content") or message.get("reasoning_content", "")
        query = " ".join(parse_query(str(content)).split()[:LANGUAGE_QUERY_MAX_WORDS])
        query_terms = set(re.findall(r"[A-Za-z0-9]+", query.lower())) - QUERY_GUARD_WORDS
        evidence_terms = set(re.findall(r"[A-Za-z0-9]+", excerpt.lower())) - QUERY_GUARD_WORDS
        if not query_terms.intersection(evidence_terms):
            raise ValueError("LM Studio memory query did not quote the supplied evidence")
        return query

    # Propose paraphrases and soft exclusions after inspecting search results.
    #
    # Arguments:
    #   query (str): Natural-language user query.
    #   snippets (str): Bounded first-pass result snippets.
    #
    # Returns:
    #   (dict[str, list[str]]): Alternative queries and soft exclusions.
    #
    def plan_language_query(self, query: str, snippets: str = "") -> Dict[str, List[str]]:
        prompt = (
            "Plan a grounded search. Return only {\"queries\":[...],\"exclude_terms\":[...]}. "
            "Rewrite the request in several short ways, preserving every condition. "
            "Give each positive condition at least one synonym lane (for example, "
            "watch in horror may become horrified spectators). If the request says "
            "not, without, or excluding, keep the positive target in a separate lane "
            "and use the negative scene only as an exclusion; never spend every lane "
            "on the excluded scene. "
            "A subject or object may be unknown; do not invent one. Exclusions are soft "
            "antipatterns seen in the results, not facts. Use at most four queries and four "
            "short exclusion terms.\nRequest:\n"
            f"{_planner_excerpt(query, 48)}\nInitial results:\n{_planner_excerpt(snippets, 160)}"
        )
        response = self._request("chat/completions", {
            "model": self._model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 64,
            "reasoning_effort": "none",
            "response_format": LANGUAGE_PLAN_RESPONSE_FORMAT,
            "stream": False,
        })
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("LM Studio returned no language-plan choices")
        message = choices[0].get("message", {})
        content = message.get("content") or message.get("reasoning_content", "")
        return _parse_language_plan(str(content))


# Deterministic planner used for offline tests and endpoint recovery.
class StubQueryGenerator:
    name = "deterministic_stub"

    def generate(self, excerpt: str) -> str:
        return _keyword_query(excerpt)


# OpenAI-compatible function schema exposed to the local model.
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_index",
        "description": "Search the HKM index for relevant source passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Short search query."},
                "top_k": {"type": "integer", "description": "Maximum number of results."},
            },
            "required": ["query"],
        },
    },
}


# Extract plain text from an OpenAI-compatible message.
def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content") or message.get("reasoning_content", "")
    if isinstance(content, list):
        content = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


# Parse a model-selected source path from its final JSON answer.
def _answer_source_path(response: str) -> str:
    text = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return str(value.get("source_path", ""))
    except json.JSONDecodeError:
        pass
    match = re.search(r'"source_path"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else ""


# Return the small JSON payload given to an agent after a search tool call.
def _tool_result_payload(result: Any, limit: int) -> Dict[str, Any]:
    return {
        "docs": [
            {
                "doc_id": int(hit.doc_id),
                "source_path": hit.source_path,
                "score": float(hit.score),
                "preview_text": hit.preview_text[:400],
            }
            for hit in result.docs[:limit]
        ],
        "count": int(result.count),
    }


# Search cheaply first, then add semantic and lexical lanes for low-confidence queries.
def _adaptive_tool_search(
    searcher: Searcher,
    query: str,
    top_k: int,
    probe_count: int,
    mode: str,
    fallback_text: str = "",
    source_cache: Dict[tuple[str, int, int], str] | None = None,
) -> tuple[Any, float, int]:
    result, elapsed = _search(searcher, query, top_k, probe_count, mode)
    fallback_calls = 0
    # Do not spend fallback work when the initial page already proves relevance.
    if fallback_text:
        _rerank_with_evidence(result, fallback_text, searcher, source_cache)
        if _evidence_rank(result, fallback_text, searcher, source_cache) == 1:
            return result, elapsed, fallback_calls
    if not fallback_text and (
        mode != "token" or (
            result.docs
            and float(result.docs[0].score) >= TOOL_FALLBACK_SCORE
            and len(query.split()) > 5
        )
    ):
        return result, elapsed, fallback_calls
    docs = list(result.docs)
    seen = {(int(hit.doc_id), tuple(hit.span)) for hit in docs}

    # Merge one fallback page at a time so proven evidence stops further work.
    def append(source: Any) -> None:
        for hit in source.docs:
            key = (int(hit.doc_id), tuple(hit.span))
            if key not in seen:
                docs.append(hit)
                seen.add(key)

    def evidence_is_first() -> bool:
        result.docs = docs
        _rerank_with_evidence(result, fallback_text, searcher, source_cache)
        return _evidence_rank(result, fallback_text, searcher, source_cache) == 1

    alternates = _fallback_queries(fallback_text or query)
    if mode != "token":
        semantic, semantic_ms = _search(searcher, query, top_k, probe_count, "semantic")
        elapsed += semantic_ms
        fallback_calls += 1
        append(semantic)
        if fallback_text and evidence_is_first():
            result.docs = result.docs[:top_k * TOOL_EXPANSION_FACTOR]
            return result, elapsed, fallback_calls
    for alternate in alternates[:5]:
        if alternate == query:
            continue
        lexical, lexical_ms = _search(searcher, alternate, top_k, probe_count, "token")
        append(lexical)
        elapsed += lexical_ms
        fallback_calls += 1
        if fallback_text and evidence_is_first():
            result.docs = result.docs[:top_k * TOOL_EXPANSION_FACTOR]
            return result, elapsed, fallback_calls
    result.docs = docs
    # Rank all first-pass lanes before truncating so later phrase hits survive.
    if fallback_text:
        _rerank_with_evidence(result, fallback_text, searcher, source_cache)
        if mode == "token" and _evidence_rank(result, fallback_text, searcher, source_cache) == 1:
            result.docs = result.docs[:top_k * TOOL_EXPANSION_FACTOR]
            return result, elapsed, fallback_calls
    if mode == "token" and fallback_text and _evidence_rank(
        result, fallback_text, searcher, source_cache
    ) is None:
        semantic, semantic_ms = _search(searcher, query, top_k, probe_count, "semantic")
        elapsed += semantic_ms
        fallback_calls += 1
        append(semantic)
        result.docs = docs
        _rerank_with_evidence(result, fallback_text, searcher, source_cache)
    # Probe remaining distinctive terms only when the first lanes lack evidence.
    if fallback_text and _evidence_rank(result, fallback_text, searcher, source_cache) is None:
        for alternate in alternates[5:5 + TOOL_RESCUE_QUERIES]:
            if alternate == query:
                continue
            rescue, rescue_ms = _search(searcher, alternate, top_k * 2, probe_count, "token")
            append(rescue)
            elapsed += rescue_ms
            fallback_calls += 1
            if evidence_is_first():
                result.docs = result.docs[:top_k * TOOL_EXPANSION_FACTOR]
                return result, elapsed, fallback_calls
        result.docs = docs
        _rerank_with_evidence(result, fallback_text, searcher, source_cache)
    # Never expose an ungrounded candidate as a successful tool result.
    if fallback_text and _evidence_rank(result, fallback_text, searcher, source_cache) is None:
        result.docs = []
        result.count = 0
    result.docs = result.docs[:top_k * TOOL_EXPANSION_FACTOR]
    return result, elapsed, fallback_calls


# Recover a query when a small local model truncates its JSON arguments.
def _parse_tool_query(arguments: Any) -> str:
    def bounded(value: Any) -> str:
        query = parse_query(json.dumps({"query": value}))
        return " ".join(query.split()[:TOOL_QUERY_WORDS])

    if isinstance(arguments, dict):
        return bounded(arguments.get("query", ""))
    text = str(arguments)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return bounded(value.get("query", ""))
    except json.JSONDecodeError:
        pass
    match = re.search(r'"query"\s*:\s*"((?:\\.|[^"\\])*)', text, re.DOTALL)
    if not match:
        raise ValueError("tool call did not contain a query")
    value = match.group(1).replace('\\"', '"').replace("\\\\", "\\")
    return bounded(value)


# Run one model/tool/model turn against the HKM search index.
class LMStudioToolAgent:
    name = "lmstudio_tool_agent"

    def __init__(self, client: LMStudioQueryGenerator, mode: str = "hybrid", final_answer: bool = True):
        self.client = client
        self.model = client.model
        self.mode = mode
        self.final_answer = final_answer
        self.source_cache: Dict[tuple[str, int, int], str] = {}

    def _request(self, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.client._request(path, payload)

    def _model_name(self) -> str:
        return self.client._model_name()

    # Recover a grounded result when the model omits or truncates its tool call.
    def _recover_tool_result(
        self,
        excerpt: str,
        searcher: Searcher,
        top_k: int,
        probe_count: int,
        started: float,
    ) -> Dict[str, Any]:
        query = _keyword_query(excerpt, limit=TOOL_KEYWORD_WORDS)
        if not query:
            raise ValueError("cannot recover a tool query from empty evidence")
        result, search_ms, fallback_calls = _adaptive_tool_search(
            searcher,
            query,
            top_k,
            probe_count,
            getattr(self, "mode", "hybrid"),
            excerpt,
            getattr(self, "source_cache", None),
        )
        _rerank_with_evidence(result, excerpt, searcher, getattr(self, "source_cache", None))
        return {
            "tool_called": True,
            "model_tool_called": False,
            "recovered": True,
            "query": query,
            "result": result,
            "answer": "",
            "answer_source_path": "",
            "completion_calls": 1,
            "search_ms": search_ms,
            "fallback_calls": fallback_calls,
            "expanded_docs": len(result.docs),
            "agent_ms": (time.perf_counter() - started) * 1000.0,
        }

    def run(self, excerpt: str, searcher: Searcher, top_k: int = 10, probe_count: int = 0) -> Dict[str, Any]:
        started = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer from indexed evidence. You must call search_index before answering. "
                    "After the tool result, return only JSON {\"answer\":\"...\",\"source_path\":\"...\"}."
                ),
            },
            {"role": "user", "content": f"Find the indexed source for this raw passage:\n{excerpt}"},
        ]
        response = self._request("chat/completions", {
            "model": self._model_name(),
            "messages": messages,
            "tools": [SEARCH_TOOL],
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": TOOL_MAX_TOKENS,
            "reasoning_effort": "none",
            "stream": False,
        })
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("LM Studio returned no tool-agent choices")
        assistant = choices[0].get("message", {})
        calls = assistant.get("tool_calls", [])
        if not calls and assistant.get("function_call"):
            calls = [{"id": "legacy-call", "function": assistant["function_call"]}]
        if not calls:
            return self._recover_tool_result(excerpt, searcher, top_k, probe_count, started)
        call = calls[0]
        function = call.get("function", {})
        try:
            query = _parse_tool_query(function.get("arguments", {}))
        except (TypeError, ValueError):
            return self._recover_tool_result(excerpt, searcher, top_k, probe_count, started)
        result, search_ms, fallback_calls = _adaptive_tool_search(
            searcher,
            query,
            top_k,
            probe_count,
            getattr(self, "mode", "hybrid"),
            excerpt,
            getattr(self, "source_cache", None),
        )
        _rerank_with_evidence(result, excerpt, searcher, getattr(self, "source_cache", None))
        call_id = str(call.get("id", "tool-call"))
        if not getattr(self, "final_answer", True):
            return {
                "tool_called": True,
                "model_tool_called": True,
                "recovered": False,
                "query": query,
                "result": result,
                "answer": "",
                "answer_source_path": "",
                "completion_calls": 1,
                "search_ms": search_ms,
                "fallback_calls": fallback_calls,
                "expanded_docs": len(result.docs),
                "agent_ms": (time.perf_counter() - started) * 1000.0,
            }
        normalized_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "search_index",
                "arguments": json.dumps({"query": query, "top_k": top_k}),
            },
        }
        messages.extend([
            {"role": "assistant", "content": assistant.get("content"), "tool_calls": [normalized_call]},
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "search_index",
                "content": json.dumps(_tool_result_payload(result, len(result.docs))),
            },
        ])
        final_response = self._request("chat/completions", {
            "model": self._model_name(),
            "messages": messages,
            "temperature": 0,
            "max_tokens": 96,
            "stream": False,
        })
        final_choices = final_response.get("choices", [])
        answer = _message_text(final_choices[0].get("message", {})) if final_choices else ""
        return {
            "tool_called": True,
            "model_tool_called": True,
            "recovered": False,
            "query": query,
            "result": result,
            "answer": answer,
            "answer_source_path": _answer_source_path(answer),
            "completion_calls": 2,
            "search_ms": search_ms,
            "fallback_calls": fallback_calls,
            "expanded_docs": len(result.docs),
            "agent_ms": (time.perf_counter() - started) * 1000.0,
        }


# Run a structured model query followed by one grounded search-tool call.
class LMStudioPlannerToolAgent:
    name = "lmstudio_planner_tool_agent"
    final_answer = False

    def __init__(self, client: LMStudioQueryGenerator, mode: str = "hybrid", native_tool: bool = False):
        self.client = client
        self.model = client.model
        self.mode = mode
        self.native_tool = native_tool
        self.name = "lmstudio_planner_native_tool_agent" if native_tool else "lmstudio_planner_tool_agent"
        self.source_cache: Dict[tuple[str, int, int], str] = {}
        self.query_cache: Dict[str, str] = {}

    # Reuse a bounded planner result for repeated raw passages in one process.
    def _plan_query(self, excerpt: str) -> tuple[str, bool]:
        cache_key = " ".join(excerpt.split())
        if cache_key in self.query_cache:
            query = self.query_cache.pop(cache_key)
            self.query_cache[cache_key] = query
            return query, True
        query = self.client.generate(excerpt)
        if len(self.query_cache) >= PLANNER_QUERY_CACHE_SIZE:
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = query
        return query, False

    def run(self, excerpt: str, searcher: Searcher, top_k: int = 10, probe_count: int = 0) -> Dict[str, Any]:
        started = time.perf_counter()
        recovered = False
        planner_cache_hit = False
        try:
            query, planner_cache_hit = self._plan_query(excerpt)
        except (OSError, RuntimeError, ValueError, urllib.error.URLError):
            query = _keyword_query(excerpt, limit=TOOL_KEYWORD_WORDS)
            recovered = True
        model_tool_called = False
        completion_calls = 0 if planner_cache_hit else 1
        if self.native_tool:
            completion_calls += 1
            try:
                response = self.client._request("chat/completions", {
                    "model": self.client._model_name(),
                    "messages": [{
                        "role": "user",
                        "content": "Call search_index with this exact short query: " + query,
                    }],
                    "tools": [SEARCH_TOOL],
                    "tool_choice": "required",
                    "temperature": 0,
                    "max_tokens": NATIVE_TOOL_MAX_TOKENS,
                    "reasoning_effort": "none",
                    "stream": False,
                })
                choices = response.get("choices", [])
                assistant = choices[0].get("message", {}) if choices else {}
                calls = assistant.get("tool_calls", [])
                if not calls and assistant.get("function_call"):
                    calls = [{"function": assistant["function_call"]}]
                if calls:
                    query = _parse_tool_query(calls[0].get("function", {}).get("arguments", {}))
                    model_tool_called = True
                else:
                    recovered = True
            except (OSError, RuntimeError, ValueError, urllib.error.URLError):
                recovered = True
        result, search_ms, fallback_calls = _adaptive_tool_search(
            searcher, query, top_k, probe_count, self.mode, excerpt, self.source_cache
        )
        _rerank_with_evidence(result, excerpt, searcher, self.source_cache)
        return {
            "tool_called": True,
            "model_tool_called": model_tool_called,
            "recovered": recovered,
            "planner_cache_hit": planner_cache_hit,
            "query": query,
            "result": result,
            "answer": "",
            "answer_source_path": "",
            "completion_calls": completion_calls,
            "search_ms": search_ms,
            "fallback_calls": fallback_calls,
            "expanded_docs": len(result.docs),
            "agent_ms": (time.perf_counter() - started) * 1000.0,
        }


# Use the same tool boundary without a model for offline tests and recovery.
class DeterministicToolAgent:
    name = "deterministic_tool_agent"
    model = None

    def __init__(self, mode: str = "hybrid"):
        self.mode = mode
        self.source_cache: Dict[tuple[str, int, int], str] = {}

    def run(self, excerpt: str, searcher: Searcher, top_k: int = 10, probe_count: int = 0) -> Dict[str, Any]:
        started = time.perf_counter()
        query = _keyword_query(excerpt, limit=TOOL_KEYWORD_WORDS)
        result, search_ms, fallback_calls = _adaptive_tool_search(
            searcher, query, top_k, probe_count, self.mode, excerpt, self.source_cache
        )
        _rerank_with_evidence(result, excerpt, searcher, self.source_cache)
        answer_source_path = result.docs[0].source_path if result.docs else ""
        return {
            "tool_called": True,
            "model_tool_called": False,
            "recovered": False,
            "query": query,
            "result": result,
            "answer": "",
            "answer_source_path": answer_source_path,
            "completion_calls": 0,
            "search_ms": search_ms,
            "fallback_calls": fallback_calls,
            "expanded_docs": len(result.docs),
            "agent_ms": (time.perf_counter() - started) * 1000.0,
        }


# Return recurring non-query terms from first-pass snippets as soft antipatterns.
#
# Arguments:
#   query (str): Original user query.
#   hits (list[Any]): First-pass search hits.
#
# Returns:
#   (list[str]): Bounded terms that may describe a misleading result cluster.
#
def _language_antipatterns(query: str, hits: List[Any]) -> List[str]:
    query_terms = _language_word_forms(query)
    counts: Dict[str, int] = {}
    for hit in hits[:6]:
        text = hit.preview_text
        terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{4,}", text.lower()))
        for term in terms:
            if term.count("-") >= 2:
                continue
            if not _language_word_forms(term).intersection(query_terms) and term not in STOP_WORDS:
                counts[term] = counts.get(term, 0) + 1
    return [term for term, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if count > 1][:LANGUAGE_QUERY_MAX_VARIANTS]


# Return visible hit text used to score remembered conditions.
def _language_hit_text(hit: Any) -> str:
    document = getattr(hit, "document", None)
    return " ".join(
        value for value in (
            getattr(hit, "preview_text", ""),
            getattr(document, "document_preview", ""),
            getattr(hit, "anchor_preview_text", ""),
        ) if value
    ).lower()


# Return the local snippet text used for explicit contrast checks.
def _language_local_hit_text(hit: Any) -> str:
    return " ".join(
        value for value in (
            getattr(hit, "preview_text", ""),
            getattr(hit, "anchor_preview_text", ""),
        ) if value
    ).lower()


# Merge language-query result lanes and apply soft antipattern penalties.
#
# Arguments:
#   results (list[Any]): Search results from query variants.
#   query (str): Original natural-language query.
#   exclude_terms (list[str]): Terms to penalize, never hard-filter.
#   top_k (int): Number of result documents to keep.
#
# Returns:
#   (Any): A SearchResult-like object with agent-ranked documents.
#
def _merge_language_results(
    results: List[Any],
    query: str,
    exclude_terms: List[str],
    top_k: int,
) -> Any:
    # Keep the full first-pass candidate page for later refinement rounds.
    base = copy.copy(results[0])
    grouped: Dict[tuple[int, tuple[int, int]], tuple[Any, int, bool]] = {}
    positive_clauses = _language_positive_clauses(query)
    query_terms = _language_word_forms(" ".join(positive_clauses))
    if not query_terms:
        query_terms = _language_word_forms(query)
    exclusions = set(term.lower() for term in exclude_terms)
    negative_clauses = _language_negative_clauses(query)
    for result in results:
        for hit in result.docs:
            key = (int(hit.doc_id), tuple(hit.span))
            if key not in grouped:
                grouped[key] = (hit, 0, result is results[0])
            best_hit, lanes, anchor = grouped[key]
            if float(hit.score) > float(best_hit.score):
                best_hit = hit
            grouped[key] = (best_hit, lanes + 1, anchor or result is results[0])

    def rank_key(item: tuple[Any, int, bool]) -> tuple[float, float, int, str]:
        hit, lanes, anchor = item
        text = _language_hit_text(hit)
        local_terms = _language_word_forms(_language_local_hit_text(hit))
        terms = _language_word_forms(text)
        coverage = len(query_terms.intersection(terms)) / float(max(1, len(query_terms)))
        negative_penalty = sum(
            len(clause.intersection(local_terms)) >= min(2, len(clause))
            for clause in negative_clauses
        )
        penalty = 0 if anchor else sum(1 for term in exclusions if term in text)
        score = (
            float(hit.score) + 0.04 * anchor + 0.005 * min(lanes, 2) + 0.03 * coverage
            - 0.14 * negative_penalty - 0.03 * penalty
        )
        return (-score, -coverage, int(hit.doc_id), hit.source_path)

    ranked = sorted(grouped.values(), key=rank_key)
    contrast_query = bool(negative_clauses)
    condition_terms = [
        (_language_full_concept_forms if contrast_query else _language_concept_forms)(clause)
        for clause in positive_clauses
    ]
    condition_terms = [terms for terms in condition_terms if terms]
    masks = {
        id(item): sum(
            1 << index
            for index, terms in enumerate(condition_terms)
            if len((
                _language_word_forms(_language_local_hit_text(item[0]))
                if contrast_query else _language_concept_forms(_language_hit_text(item[0]))
            ).intersection(terms))
            >= (1 if contrast_query else (2 if len(terms) >= 2 else 1))
        )
        for item in ranked
    }
    selected: List[tuple[Any, int, bool]] = []
    remaining = list(ranked)
    covered_mask = 0
    coherence_weight = 0.08 if contrast_query else 0.0
    while remaining and len(selected) < top_k:
        if not selected:
            choice = remaining.pop(0)
        else:
            choice_index = max(
                range(len(remaining)),
                key=lambda index: (
                    -rank_key(remaining[index])[0]
                    + 0.15 * (masks[id(remaining[index])] & ~covered_mask).bit_count()
                    + coherence_weight * masks[id(remaining[index])].bit_count(),
                    -rank_key(remaining[index])[0],
                ),
            )
            choice = remaining.pop(choice_index)
        selected.append(choice)
        covered_mask |= masks[id(choice)]
    base.docs = [hit for hit, _, _ in selected]
    base.count = len(ranked)
    base.limit = top_k
    base.next_offset = top_k if len(ranked) > top_k else None
    return base


# Search natural-language requests through iterative query and antipattern lanes.
class LanguageSearchAgent:
    name = "language_search_agent"

    def __init__(
        self,
        client: LMStudioQueryGenerator | None = None,
        mode: str = "hybrid",
        max_rounds: int = LANGUAGE_QUERY_MAX_ROUNDS,
    ) -> None:
        if mode not in {"hybrid", "semantic", "token"}:
            raise ValueError("mode must be hybrid, semantic, or token")
        if max_rounds < 1 or max_rounds > LANGUAGE_QUERY_MAX_ROUNDS:
            raise ValueError("max_rounds must be in [1, 2]")
        self.client = client
        self.model = client.model if client is not None else None
        self.mode = mode
        self.max_rounds = max_rounds

    # Run bounded first-pass, refinement, and alternative-query searches.
    def run(self, query: str, searcher: Searcher, top_k: int = 10) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        started = time.perf_counter()
        base_variants = _language_query_variants(query)
        round_queries = base_variants[:1] if self.client is not None else base_variants
        exclude_terms: List[str] = []
        trace: List[Dict[str, Any]] = []
        results: List[Any] = []
        search_ms = 0.0
        recovered = False
        completion_calls = 0
        for round_index in range(self.max_rounds):
            for variant in list(dict.fromkeys(round_queries))[:LANGUAGE_QUERY_MAX_VARIANTS]:
                variant_mode = "semantic" if self.mode == "hybrid" else self.mode
                if (
                    variant != query
                    and variant == _keyword_query(query, limit=12)
                    and variant_mode == "semantic"
                    and re.search(
                        r"\b(?:forgot|forgotten|forget|remember|recall|unknown)\b",
                        query,
                        flags=re.IGNORECASE,
                    )
                ):
                    variant_mode = "token"
                result, elapsed = _search(searcher, variant, top_k * 3, 0, variant_mode)
                results.append(result)
                search_ms += elapsed
                trace.append({
                    "round": round_index,
                    "query": variant,
                    "mode": variant_mode,
                    "hits": len(result.docs),
                })
            merged = _merge_language_results(results, query, exclude_terms, top_k)
            seen_exclusions = _language_antipatterns(query, merged.docs)
            exclude_terms = list(dict.fromkeys(exclude_terms + seen_exclusions))[:LANGUAGE_QUERY_MAX_VARIANTS]
            if self.client is None or round_index + 1 >= self.max_rounds:
                break
            snippets = json.dumps(_tool_result_payload(merged, min(8, len(merged.docs))))
            completion_calls += 1
            try:
                plan = self.client.plan_language_query(query, snippets)
                query_forms = _language_word_forms(query)
                safe_exclusions = [
                    term for term in plan["exclude_terms"]
                    if not _language_word_forms(term).intersection(query_forms)
                ]
                exclude_terms = list(dict.fromkeys(exclude_terms + safe_exclusions))[:LANGUAGE_QUERY_MAX_VARIANTS]
                deterministic_limit = max(0, LANGUAGE_QUERY_MAX_VARIANTS - 2)
                deterministic_queries = base_variants[1:1 + deterministic_limit]
                model_queries = plan["queries"][:max(0, LANGUAGE_QUERY_MAX_VARIANTS - len(deterministic_queries))]
                round_queries = list(dict.fromkeys(model_queries + deterministic_queries))[:LANGUAGE_QUERY_MAX_VARIANTS]
            except (OSError, RuntimeError, ValueError, urllib.error.URLError):
                recovered = True
                round_queries = base_variants[1:]
                if not round_queries:
                    break
        final = _merge_language_results(results, query, exclude_terms, top_k)
        return {
            "tool_called": True,
            "query": query,
            "queries": [row["query"] for row in trace],
            "antipatterns": exclude_terms,
            "rounds": max((row["round"] for row in trace), default=-1) + 1,
            "recovered": recovered,
            "completion_calls": completion_calls,
            "result": final,
            "search_ms": search_ms,
            "agent_ms": (time.perf_counter() - started) * 1000.0,
        }


# Sample raw text from actual active documents in the index.
def sample_passages(searcher: Searcher, count: int, seed: int, max_tokens: int = 48) -> List[Sample]:
    if count < 1:
        raise ValueError("count must be positive")
    doc_ids = sorted(searcher._active_doc_ids())
    if not doc_ids:
        raise ValueError("index contains no active documents")
    rng = random.Random(seed)
    backend = searcher._backend()
    eligible = []
    for doc_id in doc_ids:
        tokens, metadata = searcher._doc_context(doc_id)
        if tokens.size == 0:
            continue
        document = searcher._document_record(doc_id, metadata, tokens)
        source_file = Path(searcher.source_root) / document.source_path
        if source_file.exists():
            raw = source_file.read_bytes()
            raw = raw[document.byte_start:document.byte_end or len(raw)]
            if not raw.decode("utf-8", errors="ignore").strip():
                continue
        eligible.append(doc_id)
    selected = rng.sample(eligible, min(count, len(eligible)))
    samples = []
    for sample_id, doc_id in enumerate(selected):
        tokens, metadata = searcher._doc_context(doc_id)
        if tokens.size == 0:
            continue
        document = searcher._document_record(doc_id, metadata, tokens)
        source_file = Path(searcher.source_root) / document.source_path
        raw_text = ""
        if source_file.exists():
            raw = source_file.read_bytes()
            raw = raw[document.byte_start:document.byte_end or len(raw)]
            raw_text = raw.decode("utf-8", errors="ignore")
            if not raw_text.strip():
                continue
        words = raw_text.split()
        token_start = 0
        token_end = int(tokens.shape[0])
        if words and len(words) <= max_tokens:
            excerpt = raw_text.strip()
        elif words:
            width = min(max_tokens, len(words))
            word_start = rng.randrange(max(1, len(words) - width + 1))
            excerpt = " ".join(words[word_start:word_start + width])
        else:
            width = min(max_tokens, int(tokens.shape[0]))
            start = rng.randrange(max(1, int(tokens.shape[0]) - width + 1))
            end = min(int(tokens.shape[0]), start + width)
            excerpt = " ".join(backend.detokenize([tokens[start:end].tolist()])[0].split())
            token_start, token_end = start, end
        if not excerpt:
            continue
        samples.append(Sample(sample_id, doc_id, document.source_path, excerpt, token_start, token_end))
    if not samples:
        raise ValueError("sampled documents contained no readable text")
    return samples


# Return a target rank or None when the target is absent.
def _target_rank(result: Any, doc_id: int) -> int | None:
    for rank, hit in enumerate(result.docs, 1):
        if int(hit.doc_id) == int(doc_id):
            return rank
    return None


# Read one immutable source snapshot while its size and mtime remain unchanged.
@lru_cache(maxsize=64)
def _cached_source_bytes(path: str, size: int, mtime_ns: int) -> bytes:
    return Path(path).read_bytes()


# Read large sources without retaining their full contents in the process cache.
def _source_bytes(path: str, size: int, mtime_ns: int) -> bytes:
    if size > 16 * 1024 * 1024:
        return Path(path).read_bytes()
    return _cached_source_bytes(path, size, mtime_ns)


# Return lower-case evidence text for a hit, including its canonical source file.
def _hit_evidence_text(
    hit: Any,
    searcher: Searcher,
    source_cache: Dict[tuple[str, int, int], str] | None = None,
) -> str:
    text = f"{hit.preview_text} {getattr(hit.document, 'document_preview', '')}"
    if hit.document.source_path:
        source = Path(searcher.source_root) / hit.document.source_path
        if source.exists():
            start = int(getattr(hit.document, "byte_start", 0))
            end = int(getattr(hit.document, "byte_end", 0)) or source.stat().st_size
            key = (str(source), start, end)
            if source_cache is not None and key in source_cache:
                decoded = source_cache[key]
            else:
                stat = source.stat()
                raw = _source_bytes(str(source), stat.st_size, stat.st_mtime_ns)
                decoded = raw[start:end].decode("utf-8", errors="ignore")
                if source_cache is not None:
                    source_cache[key] = decoded
            text += " " + decoded
    return text.lower()


# Return the fraction of sampled evidence terms present in a hit's source.
def _evidence_coverage(
    hit: Any,
    excerpt: str,
    searcher: Searcher,
    source_cache: Dict[tuple[str, int, int], str] | None = None,
) -> float:
    text = _hit_evidence_text(hit, searcher, source_cache)
    normalized_excerpt = " ".join(excerpt.split()).lower()
    normalized_text = " ".join(text.split())
    if normalized_excerpt and normalized_excerpt in normalized_text:
        return 1.0
    source_path = getattr(getattr(hit, "document", None), "source_path", "")
    if source_path and searcher is not None:
        source = Path(searcher.source_root) / source_path
        if source.exists():
            excerpt_terms = set(re.findall(r"[A-Za-z0-9]+", normalized_excerpt))
            evidence_terms = set(re.findall(r"[A-Za-z0-9]+", normalized_text))
            if len(excerpt_terms) >= 4 and excerpt_terms.issubset(evidence_terms):
                return 1.0
            if len(normalized_excerpt.split()) > PLANNER_INPUT_WORDS:
                size = source.stat().st_size
                key = (str(source), 0, size)
                if source_cache is not None and key in source_cache:
                    full_text = source_cache[key]
                else:
                    stat = source.stat()
                    full_text = " ".join(
                        _source_bytes(str(source), stat.st_size, stat.st_mtime_ns)
                        .decode("utf-8", errors="ignore")
                        .split()
                    ).lower()
                    if source_cache is not None:
                        source_cache[key] = full_text
                if normalized_excerpt in full_text:
                    return 1.0
            return 0.0
    terms = set(re.findall(r"[A-Za-z0-9]+", normalized_excerpt))
    if len(terms) < 2:
        return 0.0
    return len(terms.intersection(re.findall(r"[A-Za-z0-9]+", normalized_text))) / len(terms)


# Return the first result with substantial raw-evidence agreement.
def _evidence_rank(
    result: Any,
    excerpt: str,
    searcher: Searcher,
    source_cache: Dict[tuple[str, int, int], str] | None = None,
) -> int | None:
    for rank, hit in enumerate(result.docs, 1):
        if _evidence_coverage(hit, excerpt, searcher, source_cache) >= 0.9:
            return rank
    return None


# Re-rank returned snippets against the raw evidence held by the agent.
def _rerank_with_evidence(
    result: Any,
    excerpt: str,
    searcher: Searcher | None = None,
    source_cache: Dict[tuple[str, int, int], str] | None = None,
) -> Any:
    terms = set(re.findall(r"[A-Za-z0-9]+", excerpt.lower()))
    if not terms:
        return result

    def rank_key(hit: Any) -> tuple[float, float, float, int]:
        text = _hit_evidence_text(hit, searcher, source_cache) if searcher is not None else (
            f"{hit.preview_text} {getattr(hit.document, 'document_preview', '')}".lower()
        )
        overlap = len(terms.intersection(re.findall(r"[A-Za-z0-9]+", text))) / len(terms)
        evidence = _evidence_coverage(hit, excerpt, searcher, source_cache) if searcher is not None else overlap
        return (-evidence, -overlap, -float(hit.score), int(hit.doc_id))

    result.docs = sorted(result.docs, key=rank_key)
    return result


# Merge candidate result pages while preserving the strongest lexical score.
def _merge_results(results: List[Any], top_k: int, excerpt: str = "") -> Any:
    base = results[0]
    hits: Dict[tuple[int, tuple[int, int]], Any] = {}
    for result in results:
        for hit in result.docs:
            key = (int(hit.doc_id), tuple(hit.span))
            if key not in hits or float(hit.score) > float(hits[key].score):
                hits[key] = hit
    terms = set(re.findall(r"[A-Za-z0-9]+", excerpt.lower()))

    def rank_key(hit: Any) -> tuple[float, float, str, int, tuple[int, int]]:
        text = f"{hit.preview_text} {getattr(hit.document, 'document_preview', '')}".lower()
        overlap = len(terms.intersection(re.findall(r"[A-Za-z0-9]+", text))) / len(terms) if terms else 0.0
        return (-overlap, -float(hit.score), hit.source_path, hit.doc_id, hit.span)

    # Keep one full page per fallback query until evidence ranking can inspect it.
    base.docs = sorted(hits.values(), key=rank_key)[:max(top_k * len(results), top_k)]
    return base


# Search one query and measure only the HKM call.
def _search(searcher: Searcher, query: str, top_k: int, probe_count: int, mode: str = "hybrid") -> tuple[Any, float]:
    started = time.perf_counter()
    result = searcher.search({
        "mode": mode,
        "text": query,
        "top_k": top_k,
        "probe_count": probe_count,
    })
    return result, (time.perf_counter() - started) * 1000.0


# Aggregate ranks into precision and recall metrics.
def _metrics(rows: Iterable[Dict[str, Any]], key: str = "final_rank") -> Dict[str, float]:
    values = list(rows)
    ranks = [row.get(key) for row in values]
    found = [rank is not None for rank in ranks]
    reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]
    top_one = [rank == 1 for rank in ranks]
    return {
        "recall_at_k": sum(found) / len(found) if found else 0.0,
        "precision_at_1": sum(top_one) / len(top_one) if top_one else 0.0,
        "mrr": sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
    }


# Summarize per-sample elapsed times without hiding slow tail requests.
def _timing_metrics(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = sorted(float(row[key]) for row in rows)
    if not values:
        return {"median_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, max(0, math.ceil(0.95 * len(values)) - 1))],
        "mean_ms": sum(values) / len(values),
    }


# Return whether a report fails the exact grounded-evidence quality gate.
def _grounded_quality_failed(report: Dict[str, Any]) -> bool:
    metrics = report.get("evidence", report.get("final_relevance", {}))
    return bool(report.get("errors")) or any(
        float(metrics.get(name, 0.0)) < 1.0
        for name in ("recall_at_k", "precision_at_1", "mrr")
    )


# Evaluate model queries, evidence fallback, and probe budgets.
def evaluate_agent(
    index_root: str,
    generator: QueryGenerator,
    samples: int = 8,
    seed: int = 42,
    top_k: int = 10,
    max_tokens: int = 48,
    probe_counts: List[int] | None = None,
    deterministic_first: bool = False,
    initial_probe_count: int = 0,
) -> Dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if initial_probe_count < 0:
        raise ValueError("initial probe count must be non-negative")
    probe_counts = probe_counts or [0, 1, 2, 4]
    if any(probe < 0 for probe in probe_counts):
        raise ValueError("probe counts must be non-negative")
    searcher = Searcher.from_index_root(index_root)
    sampled = sample_passages(searcher, samples, seed, max_tokens)
    source_cache: Dict[tuple[str, int, int], str] = {}
    rows = []
    probe_rows: Dict[int, List[Dict[str, Any]]] = {probe: [] for probe in probe_counts}
    planner_errors = []
    for sample in sampled:
        planner_started = time.perf_counter()
        planner_calls = 0
        planner_source = "deterministic" if deterministic_first else generator.name
        query = _keyword_query(sample.excerpt) if deterministic_first else ""
        if not query:
            planner_source = generator.name
        if not deterministic_first or not query:
            planner_calls = 1
            try:
                query = generator.generate(sample.excerpt)
            except Exception as exc:
                planner_errors.append({"sample_id": sample.sample_id, "error": str(exc)})
                query = _keyword_query(sample.excerpt)
                planner_source = "deterministic_fallback"
        planner_ms = (time.perf_counter() - planner_started) * 1000.0
        first, first_ms = _search(searcher, query, top_k, initial_probe_count)
        first_rank = _target_rank(first, sample.doc_id)
        fallback_query = ""
        fallback_ms = 0.0
        first_relevant_rank = _evidence_rank(first, sample.excerpt, searcher, source_cache)
        current = first
        current_relevant_rank = first_relevant_rank
        exhaustive_ms = 0.0
        if initial_probe_count and current_relevant_rank != 1:
            exhaustive, elapsed = _search(searcher, query, top_k, 0)
            exhaustive_ms += elapsed
            current = _merge_results([current, exhaustive], top_k, sample.excerpt)
            current_relevant_rank = _evidence_rank(current, sample.excerpt, searcher, source_cache)
        if deterministic_first and current_relevant_rank != 1:
            planner_started = time.perf_counter()
            planner_calls = 1
            planner_source = generator.name
            try:
                query = generator.generate(sample.excerpt)
            except Exception as exc:
                planner_errors.append({"sample_id": sample.sample_id, "error": str(exc)})
                query = _keyword_query(sample.excerpt)
                planner_source = "deterministic_fallback"
            planner_ms += (time.perf_counter() - planner_started) * 1000.0
            model_result, model_ms = _search(searcher, query, top_k, 0)
            current = _merge_results([current, model_result], top_k, sample.excerpt)
            current_relevant_rank = _evidence_rank(current, sample.excerpt, searcher, source_cache)
            first_ms += model_ms
        final = current if current_relevant_rank == 1 else _rerank_with_evidence(
            current, sample.excerpt, searcher, source_cache
        )
        final_rank = _target_rank(final, sample.doc_id)
        final_relevant_rank = _evidence_rank(final, sample.excerpt, searcher, source_cache)
        if final_relevant_rank != 1:
            phrase_results = []
            for phrase in _fallback_queries(sample.excerpt):
                fallback_query = phrase
                fallback, elapsed = _search(searcher, phrase, top_k, 0, mode="token")
                phrase_results.append(fallback)
                fallback_ms += elapsed
            fallback = _merge_results(phrase_results, top_k, sample.excerpt) if phrase_results else final
            fallback = _rerank_with_evidence(fallback, sample.excerpt, searcher, source_cache)
            fallback.docs = fallback.docs[:top_k]
            fallback_relevant_rank = _evidence_rank(fallback, sample.excerpt, searcher, source_cache)
            if fallback_relevant_rank is not None and (
                final_relevant_rank is None or fallback_relevant_rank < final_relevant_rank
            ):
                final = fallback
        final_rank = _target_rank(final, sample.doc_id)
        final_relevant_rank = _evidence_rank(final, sample.excerpt, searcher, source_cache)
        rows.append({
            "sample_id": sample.sample_id,
            "doc_id": sample.doc_id,
            "source_path": sample.source_path,
            "excerpt": sample.excerpt,
            "query": query,
            "planner_source": planner_source,
            "planner_calls": planner_calls,
            "first_rank": first_rank,
            "final_rank": final_rank,
            "first_relevant_rank": first_relevant_rank,
            "final_relevant_rank": final_relevant_rank,
            "fallback_query": fallback_query,
            "planner_ms": planner_ms,
            "first_search_ms": first_ms,
            "exhaustive_search_ms": exhaustive_ms,
            "fallback_search_ms": fallback_ms,
        })
        for probe in probe_counts:
            result, elapsed = _search(searcher, query, top_k, probe)
            probe_rows[probe].append({
                "rank": _target_rank(result, sample.doc_id),
                "latency_ms": elapsed,
            })
    curve = []
    for probe, values in probe_rows.items():
        ranks = [value["rank"] for value in values]
        curve.append({
            "probe_count": probe,
            "recall_at_k": sum(rank is not None for rank in ranks) / len(ranks) if ranks else 0.0,
            "precision_at_1": sum(rank == 1 for rank in ranks) / len(ranks) if ranks else 0.0,
            "median_search_ms": statistics.median(value["latency_ms"] for value in values) if values else 0.0,
        })
    return {
        "index_root": str(Path(index_root).expanduser().absolute()),
        "generator": generator.name,
        "model": getattr(generator, "model", None),
        "samples": len(rows),
        "seed": seed,
        "top_k": top_k,
        "planner_errors": planner_errors,
        "first_pass": _metrics(rows, "first_rank"),
        "final": _metrics(rows, "final_rank"),
        "first_pass_relevance": _metrics(rows, "first_relevant_rank"),
        "final_relevance": _metrics(rows, "final_relevant_rank"),
        "fallback_rate": sum(bool(row["fallback_query"]) for row in rows) / len(rows) if rows else 0.0,
        "planner_call_rate": sum(row["planner_calls"] for row in rows) / len(rows) if rows else 0.0,
        "exhaustive_escalation_rate": (
            sum(row["exhaustive_search_ms"] > 0 for row in rows) / len(rows)
            if rows else 0.0
        ),
        "deterministic_first": deterministic_first,
        "initial_probe_count": initial_probe_count,
        "latency_ms": {
            "planner": _timing_metrics(rows, "planner_ms"),
            "first_search": _timing_metrics(rows, "first_search_ms"),
            "exhaustive_search": _timing_metrics(rows, "exhaustive_search_ms"),
            "fallback_search": _timing_metrics(rows, "fallback_search_ms"),
        },
        "probe_curve": curve,
        "rows": rows,
    }


# Evaluate a real model/tool/model loop against sampled raw passages.
def evaluate_tool_agent(
    index_root: str,
    agent: Any,
    samples: int = 8,
    seed: int = 42,
    top_k: int = 10,
    max_tokens: int = 48,
    probe_count: int = 0,
    mode: str = "hybrid",
    deterministic_first: bool = False,
) -> Dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if probe_count < 0:
        raise ValueError("probe count must be non-negative")
    searcher = Searcher.from_index_root(index_root)
    sampled = sample_passages(searcher, samples, seed, max_tokens)
    source_cache: Dict[tuple[str, int, int], str] = {}
    rows = []
    errors = []
    cheap_agent = DeterministicToolAgent(mode) if deterministic_first else None
    for sample in sampled:
        started = time.perf_counter()
        model_called = bool(getattr(agent, "model", None))
        try:
            run = None
            if cheap_agent is not None:
                cheap = cheap_agent.run(sample.excerpt, searcher, top_k, probe_count)
                cheap_rank = _evidence_rank(cheap["result"], sample.excerpt, searcher, source_cache)
                if cheap_rank == 1:
                    run = cheap
                    model_called = False
            if run is None:
                run = agent.run(sample.excerpt, searcher, top_k, probe_count)
                run["search_ms"] = float(run.get("search_ms", 0.0)) + (
                    float(cheap.get("search_ms", 0.0)) if cheap_agent is not None else 0.0
                )
                run["agent_ms"] = float(run.get("agent_ms", 0.0)) + (
                    float(cheap.get("agent_ms", 0.0)) if cheap_agent is not None else 0.0
                )
                model_called = bool(getattr(agent, "model", None)) and not bool(run.get("planner_cache_hit"))
        except Exception as exc:
            run = {
                "tool_called": False,
                "model_tool_called": False,
                "recovered": False,
                "query": "",
                "result": None,
                "answer": "",
                "answer_source_path": "",
                "completion_calls": 0,
                "search_ms": 0.0,
                "agent_ms": (time.perf_counter() - started) * 1000.0,
                "fallback_calls": 0,
                "expanded_docs": 0,
            }
            errors.append({"sample_id": sample.sample_id, "error": str(exc)})
        result = run.get("result")
        target_rank = _target_rank(result, sample.doc_id) if result is not None else None
        relevant_rank = (
            _evidence_rank(result, sample.excerpt, searcher, source_cache)
            if result is not None else None
        )
        grounded_source_path = (
            result.docs[relevant_rank - 1].source_path
            if result is not None and relevant_rank else ""
        )
        answer_source_path = str(run.get("answer_source_path", ""))
        rows.append({
            "sample_id": sample.sample_id,
            "doc_id": sample.doc_id,
            "source_path": sample.source_path,
            "excerpt": sample.excerpt,
            "query": run.get("query", ""),
            "tool_called": bool(run.get("tool_called")),
            "model_called": model_called,
            "target_rank": target_rank,
            "relevant_rank": relevant_rank,
            "grounded_source_path": grounded_source_path,
            "answer_source_path": answer_source_path,
            "answer_source_match": (
                model_called and answer_source_path == sample.source_path
                if answer_source_path else False
            ),
            "grounded_source_match": grounded_source_path == sample.source_path if grounded_source_path else False,
            "planner_cache_hit": bool(run.get("planner_cache_hit")),
            "model_tool_called": bool(run.get("model_tool_called", run.get("tool_called") and model_called)),
            "recovered": bool(run.get("recovered")),
            "completion_calls": int(run.get("completion_calls", 0)),
            "fallback_calls": int(run.get("fallback_calls", 0)),
            "expanded_docs": int(run.get("expanded_docs", 0)),
            "search_ms": float(run.get("search_ms", 0.0)),
            "agent_ms": float(run.get("agent_ms", 0.0)),
        })
    return {
        "index_root": str(Path(index_root).expanduser().absolute()),
        "agent": agent.name,
        "model": getattr(agent, "model", None) or getattr(getattr(agent, "client", None), "model", None),
        "samples": len(rows),
        "seed": seed,
        "top_k": top_k,
        "probe_count": probe_count,
        "mode": mode,
        "answer_mode": "model-answer" if getattr(agent, "final_answer", False) else "tool-only",
        "deterministic_first": deterministic_first,
        "errors": errors,
        "tool_call_rate": sum(row["tool_called"] for row in rows) / len(rows) if rows else 0.0,
        "model_tool_call_rate": sum(row["model_tool_called"] for row in rows) / len(rows) if rows else 0.0,
        "recovery_rate": sum(row["recovered"] for row in rows) / len(rows) if rows else 0.0,
        "model_call_rate": sum(row["model_called"] for row in rows) / len(rows) if rows else 0.0,
        "planner_cache_hit_rate": sum(row["planner_cache_hit"] for row in rows) / len(rows) if rows else 0.0,
        "completion_calls_per_sample": sum(row["completion_calls"] for row in rows) / len(rows) if rows else 0.0,
        "fallback_rate": sum(row["fallback_calls"] > 0 for row in rows) / len(rows) if rows else 0.0,
        "expansion_rate": sum(row["expanded_docs"] > top_k for row in rows) / len(rows) if rows else 0.0,
        "target": _metrics(rows, "target_rank"),
        "evidence": _metrics(rows, "relevant_rank"),
        "answer_source_match_rate": sum(row["answer_source_match"] for row in rows) / len(rows) if rows else 0.0,
        "grounded_source_match_rate": sum(row["grounded_source_match"] for row in rows) / len(rows) if rows else 0.0,
        "latency_ms": {
            "agent": _timing_metrics(rows, "agent_ms"),
            "search": _timing_metrics(rows, "search_ms"),
        },
        "rows": rows,
    }


# Render a compact agent-quality report.
def render_report(report: Dict[str, Any]) -> str:
    first = report["first_pass"]
    final = report["final"]
    first_relevance = report["first_pass_relevance"]
    final_relevance = report["final_relevance"]
    first_label = "Initial-pass" if report["deterministic_first"] else "Model first-pass"
    latency = report["latency_ms"]
    lines = [
        "# HKM Agent Search Benchmark",
        "",
        f"- Generator: `{report['generator']}`",
        f"- Model: `{report['model'] or 'deterministic'}`",
        f"- Samples/seed/top-k: {report['samples']} / {report['seed']} / {report['top_k']}",
        f"- Planner mode: `{'deterministic-first' if report['deterministic_first'] else report['generator']}`; model calls/sample: `{report['planner_call_rate']:.3f}`; initial probe: `{report['initial_probe_count']}`",
        f"- {first_label} target-doc recall@k: `{first['recall_at_k']:.3f}`; precision@1: `{first['precision_at_1']:.3f}`; MRR: `{first['mrr']:.3f}`",
        f"- {first_label} evidence relevance@k: `{first_relevance['recall_at_k']:.3f}`; precision@1: `{first_relevance['precision_at_1']:.3f}`; MRR: `{first_relevance['mrr']:.3f}`",
        f"- Final target-doc recall@k: `{final['recall_at_k']:.3f}`; precision@1: `{final['precision_at_1']:.3f}`; MRR: `{final['mrr']:.3f}`",
        f"- Final evidence relevance@k: `{final_relevance['recall_at_k']:.3f}`; precision@1: `{final_relevance['precision_at_1']:.3f}`; MRR: `{final_relevance['mrr']:.3f}`",
        f"- Fallback rate: `{report['fallback_rate']:.3f}`",
        f"- Exhaustive escalation rate: `{report['exhaustive_escalation_rate']:.3f}`",
        f"- Latency ms (median/p95): planner `{latency['planner']['median_ms']:.2f}/{latency['planner']['p95_ms']:.2f}`; search `{latency['first_search']['median_ms']:.2f}/{latency['first_search']['p95_ms']:.2f}`; exhaustive escalation `{latency['exhaustive_search']['median_ms']:.2f}/{latency['exhaustive_search']['p95_ms']:.2f}`; fallback `{latency['fallback_search']['median_ms']:.2f}/{latency['fallback_search']['p95_ms']:.2f}`",
        "",
        "## Probe curve (model query only)",
        "",
        "| Probe count | Recall@k | Precision@1 | Median search ms |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["probe_curve"]:
        lines.append(
            f"| {row['probe_count']} | {row['recall_at_k']:.3f} | "
            f"{row['precision_at_1']:.3f} | {row['median_search_ms']:.2f} |"
        )
    lines.extend(["", "## Samples", "", "| ID | Target doc | Query | First rank | Final rank | Fallback |", "|---:|---:|---|---:|---:|---|"])
    for row in report["rows"]:
        lines.append(
            f"| {row['sample_id']} | {row['doc_id']} | {row['query'].replace('|', ' ')} | "
            f"{row['first_rank'] or '-'} | {row['final_rank'] or '-'} | "
            f"{'yes' if row['fallback_query'] else 'no'} |"
        )
    return "\n".join(lines)


# Render a compact tool-agent quality report.
def render_tool_report(report: Dict[str, Any]) -> str:
    target = report["target"]
    evidence = report["evidence"]
    latency = report["latency_ms"]
    lines = [
        "# HKM Tool-Agent Benchmark",
        "",
        f"- Agent: `{report['agent']}`",
        f"- Model: `{report['model'] or 'deterministic'}`",
        f"- Samples/seed/top-k/probe/mode: {report['samples']} / {report['seed']} / {report['top_k']} / {report['probe_count']} / `{report['mode']}`",
        f"- Answer path: `{report['answer_mode']}`",
        f"- Deterministic-first: `{report['deterministic_first']}`; tool-call rate: `{report['tool_call_rate']:.3f}`; model tool-call rate: `{report['model_tool_call_rate']:.3f}`; recovery rate: `{report['recovery_rate']:.3f}`; model-call rate: `{report['model_call_rate']:.3f}`; completion calls/sample: `{report['completion_calls_per_sample']:.2f}`",
        f"- Planner cache-hit rate: `{report['planner_cache_hit_rate']:.3f}`",
        f"- Low-confidence fallback rate: `{report['fallback_rate']:.3f}`",
        f"- Expanded result-page rate: `{report['expansion_rate']:.3f}`",
        f"- Tool target-doc recall@k: `{target['recall_at_k']:.3f}`; precision@1: `{target['precision_at_1']:.3f}`; MRR: `{target['mrr']:.3f}`",
        f"- Tool evidence relevance@k: `{evidence['recall_at_k']:.3f}`; precision@1: `{evidence['precision_at_1']:.3f}`; MRR: `{evidence['mrr']:.3f}`",
        f"- Grounded tool source-path match: `{report['grounded_source_match_rate']:.3f}`; model citation match: `{report['answer_source_match_rate']:.3f}`",
        f"- Latency ms (median/p95): agent `{latency['agent']['median_ms']:.2f}/{latency['agent']['p95_ms']:.2f}`; search `{latency['search']['median_ms']:.2f}/{latency['search']['p95_ms']:.2f}`",
        f"- Errors: `{len(report['errors'])}`",
        "",
        "## Samples",
        "",
        "| ID | Target doc | Query | Relevant rank | Grounded source | Model source | Tool | Recovery |",
        "|---:|---:|---|---:|---|---|:---:|:---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['sample_id']} | {row['doc_id']} | {row['query'].replace('|', ' ')} | "
            f"{row['relevant_rank'] or '-'} | {row['grounded_source_path'] or '-'} | "
            f"{row['answer_source_path'] or '-'} | "
            f"{'yes' if row['tool_called'] else 'no'} | {'yes' if row['recovered'] else 'no'} |"
        )
    return "\n".join(lines)


# Parse CLI arguments and write machine/human-readable evidence.
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LM Studio query agent against HKM samples.")
    parser.add_argument("index_root")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--probes", default="0,1,2,4")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=LMSTUDIO_TIMEOUT)
    parser.add_argument("--stub", action="store_true", help="Use the deterministic planner without LM Studio")
    parser.add_argument("--tool-agent", action="store_true", help="Run a model/tool/model search conversation")
    parser.add_argument("--planner-tool", action="store_true", help="Use structured model query output before the search tool")
    parser.add_argument("--native-planner-tool", action="store_true", help="Add a compact native search_index call after structured planning")
    parser.add_argument("--tool-mode", choices=["hybrid", "token", "semantic"], default="hybrid")
    parser.add_argument("--tool-only", action="store_true", help="Return the grounded tool result after one model call")
    parser.add_argument("--deterministic-first", action="store_true", help="Search cheaply before calling the model")
    parser.add_argument("--require-grounded", action="store_true", help="Exit nonzero unless final evidence metrics are all 1.0")
    parser.add_argument("--initial-probe", type=int, default=0, help="Use this probe budget before exhaustive escalation")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()
    if args.tool_agent:
        if args.stub:
            agent: Any = DeterministicToolAgent(args.tool_mode)
        else:
            client = LMStudioQueryGenerator(args.base_url, args.model, args.timeout)
            try:
                client._model_name()
                agent = (
                    LMStudioPlannerToolAgent(client, args.tool_mode, args.native_planner_tool)
                    if args.planner_tool or args.native_planner_tool
                    else LMStudioToolAgent(client, args.tool_mode, not args.tool_only)
                )
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                print(f"LM Studio unavailable; using deterministic tool agent: {exc}")
                agent = DeterministicToolAgent(args.tool_mode)
        report = evaluate_tool_agent(
            args.index_root,
            agent,
            args.samples,
            args.seed,
            args.top_k,
            args.max_tokens,
            args.initial_probe,
            args.tool_mode,
            args.deterministic_first,
        )
        markdown = render_tool_report(report)
        if args.json_output:
            Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.report_output:
            Path(args.report_output).write_text(markdown, encoding="utf-8")
        print(markdown)
        if args.require_grounded and _grounded_quality_failed(report):
            raise SystemExit("grounded evidence quality gate failed")
        return
    generator: QueryGenerator = StubQueryGenerator() if args.stub else LMStudioQueryGenerator(args.base_url, args.model, args.timeout)
    if not args.stub:
        try:
            generator._model_name()  # type: ignore[attr-defined]
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            print(f"LM Studio unavailable; using deterministic fallback: {exc}")
            generator = StubQueryGenerator()
    report = evaluate_agent(
        args.index_root,
        generator,
        args.samples,
        args.seed,
        args.top_k,
        args.max_tokens,
        [int(value) for value in args.probes.split(",") if value.strip()],
        args.deterministic_first,
        args.initial_probe,
    )
    markdown = render_report(report)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.report_output:
        Path(args.report_output).write_text(markdown, encoding="utf-8")
    print(markdown)
    if args.require_grounded and _grounded_quality_failed(report):
        raise SystemExit("grounded evidence quality gate failed")


if __name__ == "__main__":  # pragma: no cover
    main()
