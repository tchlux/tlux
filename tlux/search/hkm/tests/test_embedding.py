if __name__ == "__main__":
    import numpy as np
    from tlux.search.hkm.embedder import tokenize, detokenize, embed, embed_windows

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
