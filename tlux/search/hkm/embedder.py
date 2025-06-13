"""Contriever-based tokenizer and embedding generator.

This replaces the original *random* embedding stub with a production
wrapper around **facebook/contriever**.  All public function signatures
remain identical so callers need *no* changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

from .utils.contriever_embedder import ContrieverEmbedder

from .schema import STRIDE_FACTOR, WINDOW_SIZES

# ---------------------------------------------------------------------------
# Model + tokenizer: load once per process
# ---------------------------------------------------------------------------

_DEVICE = None
_EMBEDDER = ContrieverEmbedder()
_TOKENIZER = _EMBEDDER.tokenizer
_MODEL = _EMBEDDER.model
_DEVICE = _EMBEDDER.device


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def tokenize(text: str) -> np.ndarray:
    """Return 1-D NumPy array of token ids (int64)."""
    ids: List[int] = _TOKENIZER.encode(text, add_special_tokens=False)
    return np.asarray(ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Internal single-window embed util
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)  # memoise repeated windows within a doc
def _embed_window(token_tuple: tuple[int, ...]) -> np.ndarray:
    ids_tensor = (
        torch.tensor(token_tuple, dtype=torch.long, device=_DEVICE).unsqueeze(0)
    )  # (1, L)
    attn = torch.ones_like(ids_tensor)
    with torch.no_grad():
        hs = _MODEL(input_ids=ids_tensor, attention_mask=attn) # .last_hidden_state
        vec = hs.mean(dim=1).squeeze(0).cpu().numpy()
    return vec.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Public sliding-window embedder
# ---------------------------------------------------------------------------

def embed_windows(
    tokens: np.ndarray, window_size: int, stride: int | None = None
) -> np.ndarray:
    """Embed sliding windows of *tokens*.

    Parameters
    ----------
    tokens:
        Array ``(n,)`` of *Contriever* token ids.
    window_size:
        Number of tokens per window (<=512 to satisfy model maximum).
    stride:
        Defaults to ``window_size * STRIDE_FACTOR``.
    """
    if stride is None:
        stride = max(1, int(window_size * STRIDE_FACTOR))
    if tokens.size < window_size:
        return np.empty((0, _MODEL.config.hidden_size), dtype=np.float32)

    out: List[np.ndarray] = []
    for start in range(0, tokens.size - window_size + 1, stride):
        slice_ids = tuple(tokens[start : start + window_size].tolist())
        out.append(_embed_window(slice_ids))

    if not out:
        return np.empty((0, _MODEL.config.hidden_size), dtype=np.float32)
    return np.stack(out, dtype=np.float32)


def embed_text(text: str, window_sizes=WINDOW_SIZES) -> Dict[int, np.ndarray]:
    """Tokenise *text* and return dict ``{window_size: embeddings}``."""
    tokens = tokenize(text)
    emb: Dict[int, np.ndarray] = {}
    for w in window_sizes:
        emb[w] = embed_windows(tokens, w)
    return emb


__all__ = ["tokenize", "embed_text", "embed_windows"]
