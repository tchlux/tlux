import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


# Load once at module scope
this_dir = os.path.dirname(__file__)
tokenizer = Tokenizer.from_file(os.path.join(this_dir, "tokenizer.json"))
session = ort.InferenceSession(os.path.join(this_dir, "model_quantized.onnx"))


# Special tokens and role-specific prefixes
FIRST_TOKEN    = 0
LAST_TOKEN     = 2
PASSAGE_TOKENS = [46692, 12]
QUERY_TOKENS   = [41, 1294, 12]
MAX_SEQ_LEN    = 512


# Tokenizes a list of texts using batch encoding and strips BOS/EOS.
#
# Args:
#     texts: List of input strings.
#
# Returns:
#     List of token ID lists without special start/end tokens.
# 
def tokenize(texts: list[str]) -> list[list[int]]:
    encodings = tokenizer.encode_batch(texts)
    return [enc.ids[1:-1] for enc in encodings]


# Converts lists of token IDs back into strings.
#
# Args:
#     token_ids: List of token ID sequences.
#
# Returns:
#     List of decoded strings (special tokens skipped).
# 
def detokenize(token_ids: list[list[int]]) -> list[str]:
    return tokenizer.decode_batch(token_ids, skip_special_tokens=True)


# Computes L2-normalized embeddings via ONNX.
#
# Args:
#     token_ids: List of token ID sequences (variable-length).
#     max_len:   Maximum sequence length (including special tokens).
#     role:      'doc' for passage prefix, 'query' for query prefix.
#
# Returns:
#     NumPy array of shape (batch_size, hidden_dim) of normalized embeddings.
# 
def embed(
    token_ids: list[list[int]],
    max_len: int = MAX_SEQ_LEN,
    role: str = "doc",
) -> np.ndarray:
    assert role in {"doc", "query"}, "role must be 'doc' or 'query'"
    prefix = PASSAGE_TOKENS if role == "doc" else QUERY_TOKENS
    # Reserve space for [FIRST] + prefix + [LAST]
    body_max = max_len - 2 - len(prefix)
    batch_size = len(token_ids)
    # Build each sequence with special tokens and prefix
    sequences = []
    lengths = []
    for ids in token_ids:
        if (len(ids) > body_max): body = ids[:body_max]
        else:                     body = ids
        seq = [FIRST_TOKEN] + prefix + body + [LAST_TOKEN]
        sequences.append(seq)
        lengths.append(len(seq))
    # Pad sequences and build attention mask
    seq_len = max(lengths)
    input_ids = np.zeros((batch_size, seq_len), dtype=np.int64)
    attention_mask = np.zeros_like(input_ids)
    for i, seq in enumerate(sequences):
        L = lengths[i]
        input_ids[i, :L]      = seq
        attention_mask[i, :L] = 1
    token_type_ids = np.zeros_like(input_ids)
    # Prepare ONNX inputs; omit token_type_ids if graph doesn't require them
    ort_inputs = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    # Run inference
    last_hidden = session.run(None, ort_inputs)[0]  # (batch, seq_len, hidden_dim)
    # Mean-pool over tokens then L2-normalize
    mask_f = attention_mask.astype(np.float32)[..., None]
    pooled = (last_hidden * mask_f).sum(1) / mask_f.sum(1)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)



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
