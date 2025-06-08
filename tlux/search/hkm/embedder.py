"""Simple character tokenizer and random embedding generator."""

from __future__ import annotations

import numpy as np

from .schema import WINDOW_SIZES, STRIDE_FACTOR

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> np.ndarray:
    """Convert *text* into an array of integer token ids.

    This stub tokenizer simply converts each character to its ``ord`` value
    truncated to the range ``0–255``.
    """
    if not text:
        return np.empty(0, dtype=np.uint8)
    data = np.frombuffer(text.encode("utf-8", "replace"), dtype=np.uint8)
    return data


# ---------------------------------------------------------------------------
# Random embedding table
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_CHAR_EMBED = _RNG.standard_normal((256, 1024)).astype(np.float32)


def _embed_tokens(tokens: np.ndarray) -> np.ndarray:
    """Return embeddings for *tokens* by table lookup."""
    return _CHAR_EMBED[tokens % 256]


# ---------------------------------------------------------------------------
# Public embedding helpers
# ---------------------------------------------------------------------------

def embed_windows(tokens: np.ndarray, window_size: int, stride: int | None = None) -> np.ndarray:
    """Embed sliding windows of *tokens*.

    Parameters
    ----------
    tokens:
        ``(n,)`` array of token ids.
    window_size:
        Number of tokens per window.
    stride:
        Step between consecutive windows.  Defaults to ``window_size * STRIDE_FACTOR``.
    """
    if stride is None:
        stride = max(1, int(window_size * STRIDE_FACTOR))
    if tokens.size < window_size:
        return np.empty((0, _CHAR_EMBED.shape[1]), dtype=np.float32)
    vectors = _embed_tokens(tokens)
    dim = vectors.shape[1]
    out = []
    for start in range(0, tokens.size - window_size + 1, stride):
        window = vectors[start : start + window_size]
        out.append(window.mean(axis=0, dtype=np.float32))
    if not out:
        return np.empty((0, dim), dtype=np.float32)
    return np.stack(out).astype(np.float32, copy=False)


def embed_text(text: str, window_sizes=WINDOW_SIZES) -> dict[int, np.ndarray]:
    """Tokenise *text* and return embeddings per window size."""
    tokens = tokenize(text)
    emb = {}
    for w in window_sizes:
        emb[w] = embed_windows(tokens, w)
    return emb


__all__ = ["tokenize", "embed_text", "embed_windows"]
