"""Shared embedder interface with one default backend."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import tqdm

TokenizeFn = Callable[[list[str]], list[list[int]]]
DetokenizeFn = Callable[[list[list[int]]], list[str]]
EmbedFn = Callable[[list[list[int]], int, str], np.ndarray]


@dataclass(frozen=True)
class EmbedderBackend:
    name: str
    tokenize: TokenizeFn
    detokenize: DetokenizeFn
    embed: EmbedFn


def _load_fake_backend() -> EmbedderBackend:
    def _tok(texts: list[str]) -> list[list[int]]:
        output: list[list[int]] = []
        for text in texts:
            output.append([int(part) & 0xFFFFFFFF for part in text.split() if part.lstrip("-").isdigit()])
        return output

    def _detok(token_ids: list[list[int]]) -> list[str]:
        return [" ".join(str(token) for token in tokens) for tokens in token_ids]

    def _emb(token_ids: list[list[int]], max_len: int = 8192, role: str = "doc") -> np.ndarray:
        rows = []
        for tokens in token_ids:
            seq = (tokens[:max_len] or [0])
            mean_val = float(sum(seq)) / float(len(seq))
            span = float(max(seq) - min(seq)) if seq else 0.0
            rows.append([mean_val, float(len(seq)), span, float(seq[0])])
        return np.asarray(rows, dtype=np.float32)

    return EmbedderBackend("fake", _tok, _detok, _emb)


def _load_backend(name: str) -> EmbedderBackend:
    modules = {
        "drama": "tlux.search.hkm.libs.drama.inference",
        "contriever": "tlux.search.hkm.libs.contriever.inference",
        "e5": "tlux.search.hkm.libs.e5.inference",
        "gemma": "tlux.search.hkm.libs.gemma.inference",
    }
    if name == "fake":
        return _load_fake_backend()
    if name not in modules:
        raise ValueError(f"Unknown HKM embedder backend {name!r}")
    module = __import__(modules[name], fromlist=["tokenize", "detokenize", "embed"])
    return EmbedderBackend(name, module.tokenize, module.detokenize, module.embed)


# Return the configured embedder backend.
#
# Arguments:
#   name (str | None): Optional backend override.
#
# Returns:
#   (EmbedderBackend): Selected backend implementation.
#
def get_backend(name: str | None = None) -> EmbedderBackend:
    if os.getenv("HKM_FAKE_EMBEDDER") == "1":
        return _load_fake_backend()
    return _load_backend(name or os.getenv("HKM_EMBEDDER", "drama"))


# Return the list of supported backend names.
#
# Arguments:
#   None
#
# Returns:
#   (list[str]): Backend names available through the shared interface.
#
def available_backends() -> list[str]:
    return ["drama", "contriever", "e5", "gemma", "fake"]


DEFAULT_WINDOWS = (128, 512, 1024)
_DEFAULT_OVERLAP = 0.5


# Tokenize text using the configured backend.
#
# Arguments:
#   texts (list[str]): Input text strings.
#
# Returns:
#   (list[list[int]]): Backend token IDs.
#
def tokenize(texts: list[str]) -> list[list[int]]:
    return get_backend().tokenize(texts)


# Convert token IDs back into strings.
#
# Arguments:
#   token_ids (list[list[int]]): Token sequences to decode.
#
# Returns:
#   (list[str]): Decoded strings.
#
def detokenize(token_ids: list[list[int]]) -> list[str]:
    return get_backend().detokenize(token_ids)


# Embed token sequences with the configured backend.
#
# Arguments:
#   token_ids (list[list[int]]): Token sequences to embed.
#   max_len (int): Maximum number of tokens consumed per sequence.
#   role (str): Either "doc" or "query".
#
# Returns:
#   (np.ndarray): Batch embedding matrix.
#
def embed(token_ids: list[list[int]], max_len: int = 8192, role: str = "doc") -> np.ndarray:
    return get_backend().embed(token_ids, max_len=max_len, role=role)


# Compute embeddings for sliding windows over token sequences.
#
# Arguments:
#   token_ids_list (list[list[int]]): Token ID sequences to embed.
#   window_sizes (list[int]): Sliding-window sizes to evaluate.
#   window_overlap (float): Overlap fraction in [0, 1).
#   role (str): Either "doc" or "query".
#
# Returns:
#   (tuple[np.ndarray, list[tuple[int, int, int]]]): Embeddings and window metadata.
#
def embed_windows(
    token_ids_list: list[list[int]],
    window_sizes: list[int] = list(DEFAULT_WINDOWS),
    window_overlap: float = _DEFAULT_OVERLAP,
    role: str = "doc",
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    if role not in {"doc", "query"}:
        raise ValueError("role must be 'doc' or 'query'")
    if not (0 <= window_overlap < 1):
        raise ValueError("window_overlap must be in [0, 1)")
    if not isinstance(token_ids_list, list):
        raise TypeError("token_ids_list must be a list of lists")
    windows_meta: list[tuple[int, int, int]] = []
    by_size: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for seq_idx, ids in enumerate(token_ids_list):
        n = len(ids)
        for w in window_sizes:
            if (w > n) and (w > window_sizes[0]):
                continue
            step = max(1, int(w * (1.0 - window_overlap)))
            starts = list(range(0, max(1, n - w + 1), step))
            tail = n - w
            if tail > 0 and starts[-1] != tail:
                starts.append(tail)
            for start in starts:
                idx = len(windows_meta)
                window_ids = ids[start : start + w]
                by_size[w].append((idx, window_ids))
                windows_meta.append((start, start + len(window_ids), len(window_ids)))
    if not windows_meta:
        empty = embed([[0]], role=role)[:0]
        return empty, []
    hidden_dim = int(embed([[0]], role=role).shape[1])
    embeddings = np.zeros((len(windows_meta), hidden_dim), dtype=np.float32)
    for _, bucket in tqdm.tqdm(by_size.items(), file=sys.stdout):
        indices, windows = zip(*bucket)
        batch = embed(list(windows), role=role)
        for pos, idx in enumerate(indices):
            embeddings[idx] = batch[pos]
    return embeddings, windows_meta
