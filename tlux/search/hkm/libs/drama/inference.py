import os
import numpy as np
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer


# Load once at module scope
this_dir = os.path.dirname(__file__)
model = SentenceTransformer("facebook/drama-base", trust_remote_code=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer = Tokenizer.from_file(os.path.join(this_dir, "tokenizer.json"))

# Special tokens
FIRST_TOKEN    = 128000
LAST_TOKEN     = 128001
MAX_SEQ_LEN    = 8192


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
#     role:      'doc' for passage, 'query' for query.
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
    sentences = detokenize(token_ids)
    if role == "doc":
        result = model.encode_document(sentences)
    else:
        result = model.encode_query(sentences)
    return result


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
