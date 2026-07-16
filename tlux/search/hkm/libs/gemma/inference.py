"""Embed text through a local OpenAI-compatible EmbeddingGemma server."""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from ..drama import inference as drama


ENDPOINT = os.getenv("HKM_GEMMA_ENDPOINT", "http://127.0.0.1:4321/v1/embeddings")
MODEL = os.getenv("HKM_GEMMA_MODEL", "")
MAX_CHARS = max(1, int(os.getenv("HKM_GEMMA_MAX_CHARS", "4000")))
BATCH_SIZE = max(1, int(os.getenv("HKM_GEMMA_BATCH_SIZE", "32")))


# Tokenize with the bundled tokenizer so lexical search and previews stay stable.
#
# Arguments:
#   texts (list[str]): Input passages or queries.
#
# Returns:
#   (list[list[int]]): Stable token IDs.
def tokenize(texts: list[str]) -> list[list[int]]:
    return drama.tokenize(texts)


# Decode stored token IDs back to text for the embedding service.
#
# Arguments:
#   token_ids (list[list[int]]): Stored token IDs.
#
# Returns:
#   (list[str]): Decoded text.
def detokenize(token_ids: list[list[int]]) -> list[str]:
    return drama.detokenize(token_ids)


# Request embeddings from one OpenAI-compatible endpoint, accepting llama.cpp's legacy shape too.
#
# Arguments:
#   texts (list[str]): Prompted inputs.
#
# Returns:
#   (np.ndarray): Raw embedding matrix.
def _request(texts: list[str]) -> np.ndarray:
    payload: dict[str, object] = {"input": texts}
    if MODEL:
        payload["model"] = MODEL
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("HKM_GEMMA_TIMEOUT", "60"))) as response:
            body = json.load(response)
        rows = body.get("data", []) if isinstance(body, dict) else body
        rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
        values = [row["embedding"] for row in rows]
        values = [value[0] if isinstance(value, list) and value and isinstance(value[0], list) else value for value in values]
        if len(values) != len(texts):
            raise ValueError(f"endpoint returned {len(values)} embeddings for {len(texts)} inputs")
        return np.asarray(values, dtype=np.float32)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"HKM Gemma endpoint failed at {ENDPOINT}: {exc}") from exc


# Format a passage with its first line as a lightweight title when available.
#
# Arguments:
#   text (str): Decoded passage text.
#
# Returns:
#   (str): EmbeddingGemma document prompt.
def _document_prompt(text: str) -> str:
    title, separator, body = text.partition("\n")
    if separator and title.strip() and body.strip() and len(title.split()) <= 40:
        return f"title: {title.strip()} | text: {body.strip()}"
    return f"title: none | text: {text}"


# Compute normalized query or passage embeddings through the configured server.
#
# Arguments:
#   token_ids (list[list[int]]): Passage or query token IDs.
#   max_len (int): Maximum token IDs decoded per input.
#   role (str): Either ``doc`` or ``query``.
#
# Returns:
#   (np.ndarray): L2-normalized embedding matrix.
def embed(token_ids: list[list[int]], max_len: int = 8192, role: str = "doc") -> np.ndarray:
    if role not in {"doc", "query"}:
        raise ValueError("role must be 'doc' or 'query'")
    texts = []
    for ids in token_ids:
        text = detokenize([ids[:max_len]])[0].strip()[:MAX_CHARS]
        texts.append(f"task: search result | query: {text}" if role == "query" else _document_prompt(text))
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    batches = [_request(texts[start : start + BATCH_SIZE]) for start in range(0, len(texts), BATCH_SIZE)]
    values = np.vstack(batches)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
