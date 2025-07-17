"""
Text tokenizer and embedding generator.
"""

# Module for generating text embeddings.

from functools import lru_cache
from collections import defaultdict
import numpy as np

from .utils.drama.inference import tokenize, detokenize, embed
# from .utils.contriever.inference import tokenize, detokenize, embed
# from .utils.e5.inference import tokenize, detokenize, embed


MAX_SEQ_LEN = 8192  # 512
DEFAULT_WINDOWS = (32, 128, 512, 1024)  # (8, 32, 128, 512, 1024) 
DEFAULT_OVERLAP = 0.5

# The following caching does not work because of list types.
# # store the original embedding function without a cache
# _embed_nocache = embed
# # memoise repeated windows within a doc
# embed = lru_cache(maxsize=1024)(embed)


# Computes embeddings for sliding windows over each token sequence,
# grouping by window size to minimize padding waste.
#
# Args:
#     token_ids_list:  List of full-token-ID sequences.
#     window_sizes:    List of window sizes (e.g. [8,32,128,512]).
#     window_overlap:  Fractional overlap between 0.0 and <1.0.
#     max_len:         Global max length (defaults to 512).
#     role:            'doc' or 'query' for prefix selection.
#
# Returns:
#     embeddings:  np.ndarray of shape (total_windows, hidden_dim)
#     windows:     List of (seq_index, start_index, window_length)
# 
def embed_windows(
    token_ids_list: list[list[int]],
    window_sizes: list[int] = DEFAULT_WINDOWS,
    window_overlap: float = DEFAULT_OVERLAP,
    max_len: int = MAX_SEQ_LEN,
    role: str = "doc",
) -> tuple[np.ndarray, list[tuple[int,int,int]]]:
    assert role in {"doc", "query"}, "role must be 'doc' or 'query'"
    assert 0 <= window_overlap < 1, "overlap must be in [0,1)"
    # 1) generate a flat list of all windows and remember where they came from
    windows_meta: list[tuple[int,int,int]] = []
    by_size: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for seq_i, ids in enumerate(token_ids_list):
        N = len(ids)
        for w in window_sizes:
            if (w > N) and (w > window_sizes[0]):
                continue
            step = max(1, int(w * (1.0 - window_overlap)))
            starts = list(range(0, max(1, N - w + 1), step))
            tail = N - w
            if tail > 0 and starts[-1] != tail:
                starts.append(tail)
            for s in starts:
                idx = len(windows_meta)
                window_ids = ids[s : s + w]
                by_size[w].append((idx, window_ids))
                windows_meta.append((seq_i, s, len(window_ids)))
    # 2) allocate output array
    total = len(windows_meta)
    # run one dummy pass to get hidden_dim
    dummy_out = embed([[0]], role=role)
    hidden_dim = dummy_out.shape[1]
    embeddings = np.zeros((total, hidden_dim), dtype=np.float32)
    # 3) embed each size-group in its own batch
    for w, bucket in by_size.items():
        indices, windows = zip(*bucket)
        # each call pads to exactly w + prefix + 2
        batch_emb = embed(
            list(windows),
            role=role,
        )
        for i, idx in enumerate(indices):
            embeddings[idx] = batch_emb[i]
    return embeddings, windows_meta



if __name__ == "__main__":
    # Testing texts.
    texts = [
        "",
        "The Eiffel Tower is in Paris.",
        "There is a tower monument in Paris that is famous.",
        "dogs around san francisco rarely wear leashes!",
        "Dogs around San Francisco rarely wear leashes.",
    ]

    # Generate tokens and print
    tokens = tokenize(texts)
    print("Tokens:")
    for t in tokens:
        print("", repr(detokenize([t])[0]))
        print("", "", t)
    print()

    # Generate embeddings and print snippet
    vecs = embed(tokens)
    print("Embedding shape:", vecs.shape)
    print()
    for v in vecs:
        norm = round(np.linalg.norm(v), 2)
        head = [round(x, 2) for x in v[:10].tolist()]
        print(f" {norm} -- {head}")
    print()

    # Use the windowed embedder.
    vecs, windows_meta = embed_windows(tokens)
    print(vecs.shape)
    print(windows_meta)

