"""
Text tokenizer and embedding generator.

Module provides utilities for tokenizing text, detokenizing sequences,
and generating fixed-size embedding vectors for variable-length token 
sequences. Embeddings can be computed over sliding windows of tokens 
to support long or fragmented documents.

This implementation minimizes external dependencies, supports batched
embedding by window size for efficiency, and ensures deterministic 
behavior by avoiding unseeded randomness.

Example usage:
  from text_embed import tokenize, detokenize, embed, embed_windows
  texts = ["sample doc", "another"]
  token_ids = tokenize(texts)
  embeddings = embed(token_ids)
  win_emb, meta = embed_windows(token_ids)
"""

# Module for text tokenization and embedding.

from collections import defaultdict
import numpy as np

try:
    from .utils.drama.inference import tokenize, detokenize, embed
    # from .utils.contriever.inference import tokenize, detokenize, embed
    # from .utils.e5.inference import tokenize, detokenize, embed
except ImportError:
    from tlux.search.hkm.utils.drama.inference import tokenize, detokenize, embed

_MAX_SEQ_LEN = 8192
_DEFAULT_WINDOWS = (32, 128, 512, 1024)
_DEFAULT_OVERLAP = 0.5

# Computes embeddings for sliding windows over token sequences.
#
# Description:
#   For each input token sequence, computes dense vector embeddings
#   for sliding windows of specified sizes, efficiently batching
#   by window size. Output includes both the embedding vectors and 
#   metadata describing the origin of each window.
#
# Parameters:
#   token_ids_list (list[list[int]]): List of token ID sequences.
#   window_sizes   (list[int]): Window sizes (default: [32,128,512,1024]).
#   window_overlap (float): Fractional overlap in [0, 1).
#   max_len        (int): Maximum allowed sequence length.
#   role           (str): 'doc' or 'query', selects special prefix handling.
#
# Returns:
#   (np.ndarray, list[tuple[int, int, int]]): Embeddings array (n_win, d), 
#       and metadata for each window as (seq_idx, start, win_len).
#
# Raises:
#   AssertionError: If inputs do not meet contract.
#
def embed_windows(
    token_ids_list: list[list[int]],
    window_sizes: list[int] = list(_DEFAULT_WINDOWS),
    window_overlap: float = _DEFAULT_OVERLAP,
    max_len: int = _MAX_SEQ_LEN,
    role: str = "doc",
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    # Validate arguments.
    if role not in {"doc", "query"}:
        raise ValueError("role must be 'doc' or 'query'")
    if not (0 <= window_overlap < 1):
        raise ValueError("window_overlap must be in [0, 1)")
    if not isinstance(token_ids_list, list):
        raise TypeError("token_ids_list must be a list of lists")
    # Accumulate window info and batch by window size.
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
            for s in starts:
                idx = len(windows_meta)
                window_ids = ids[s : s + w]
                by_size[w].append((idx, window_ids))
                windows_meta.append((seq_idx, s, len(window_ids)))
    # Prepare output array for embeddings.
    total = len(windows_meta)
    dummy = embed([[0]], role=role)
    if dummy.ndim != 2:
        raise RuntimeError("embed() must return (batch, dim) array")
    hidden_dim = dummy.shape[1]
    embeddings = np.zeros((total, hidden_dim), dtype=np.float32)
    # Compute embeddings in size-batched chunks.
    for w, bucket in by_size.items():
        indices, windows = zip(*bucket)
        batch_emb = embed(list(windows), role=role)
        if batch_emb.shape != (len(windows), hidden_dim):
            raise RuntimeError(
                f"embed output shape {batch_emb.shape} does not match "
                f"(batch={len(windows)}, dim={hidden_dim})"
            )
        for i, idx in enumerate(indices):
            embeddings[idx] = batch_emb[i]
    return embeddings, windows_meta


if __name__ == "__main__":
    # Demonstration / test code (no side effects on import).
    test_texts = [
        "",
        "The Eiffel Tower is in Paris.",
        "There is a tower monument in Paris that is famous.",
        "dogs around san francisco rarely wear leashes!",
        "Dogs around San Francisco rarely wear leashes.",
    ]
    # Tokenize sample texts.
    test_tokens = tokenize(test_texts)
    print("Tokens:")
    for t in test_tokens:
        print(" ", repr(detokenize([t])[0]))
        print("   ", t)
    print()
    # Compute and show embeddings for full texts.
    test_vecs = embed(test_tokens)
    print("Embedding shape:", test_vecs.shape)
    for v in test_vecs:
        norm = round(float(np.linalg.norm(v)), 2)
        head = [round(float(x), 2) for x in v[:10].tolist()]
        print(f" {norm} -- {head}")
    print()
    # Sliding window embeddings.
    win_vecs, win_meta = embed_windows(test_tokens)
    print("Windowed embedding shape:", win_vecs.shape)
    print("Window metadata:", win_meta)
