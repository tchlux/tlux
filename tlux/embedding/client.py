# Embed the transaction names.
from tqdm import tqdm
from tlux.network import NetworkQueue
from tlux.decorators import cache
import numpy as np

# Generate a communication network queue.
embedding_queue = None
text_queue = None

# Delayed initialization for the embedding queue.
def get_embedding_queue():
    global embedding_queue
    if (embedding_queue is None):
        embedding_queue = NetworkQueue(host="Macsey.local", port=12345, listen_port=None)
    return embedding_queue

# Get embeddings for a list of strings.
@cache()
def get_embeddings(list_of_strings: list[str], batch_size: int=100):
    # Get the embedding queue for communicating with the server.
    embedding_queue = get_embedding_queue()
    # Iterate over source data and generate embeddings.
    embeddings: list[list[np.ndarray]] = []
    for batch_i in tqdm(
        range(0, len(list_of_strings), batch_size),
        total=len(list_of_strings) // batch_size,
        delay=5,
    ):
        samples = list_of_strings[batch_i : batch_i + batch_size]
        # Check remaining size of samples needed to be embedded.
        if (len(list_of_strings) >= batch_size):
            if (len(samples) == 0):
                print(f" skipping empty batch at range [{batch_i}, {batch_i+batch_size-1}]", flush=True)
                continue
            elif (len(samples) < batch_size):
                print(f" only processing {len(samples)} from batch of {batch_size}..", flush=True)
        # Generate embeddings.
        embedding_queue.put(samples)
        embeddings.append(embedding_queue.get())
    # Return the embeddings as a numpy matrix.
    embeddings = np.concat(embeddings, axis=0)
    return embeddings


# Get embedding for a single string.
@cache()
def get_embedding(string: str):
    return get_embeddings([string])[0]


# Delayed initialization for the text queue.
def get_text_queue():
    global text_queue
    if (text_queue is None):
        text_queue = NetworkQueue(host="Macsey.local", port=54321, listen_port=None)
    return text_queue


# Get strings for a list of embeddings.
@cache()
def get_texts(list_of_embeddings: list[list[float]], batch_size: int=100):
    # Get the text queue for communicating with the server.
    text_queue = get_text_queue()
    # Iterate over source data and generate embeddings.
    texts: list[str] = []
    for batch_i in tqdm(
        range(0, len(list_of_embeddings), batch_size),
        total=len(list_of_embeddings) // batch_size,
        delay=5,
    ):
        samples = list_of_embeddings[batch_i : batch_i + batch_size]
        # Check remaining size of samples needed to be embedded.
        if (len(list_of_embeddings) >= batch_size):
            if (len(samples) == 0):
                print(f" skipping empty batch at range [{batch_i}, {batch_i+batch_size-1}]", flush=True)
                continue
            elif (len(samples) < batch_size):
                print(f" only processing {len(samples)} from batch of {batch_size}..", flush=True)
        # Generate embeddings.
        text_queue.put(samples)
        texts.append(text_queue.get())
    # Return the texts.
    return texts


# Get text for a single embedding.
@cache()
def get_text(embedding: list[float]):
    return get_texts([embedding])[0]
