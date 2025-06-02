# source ~/Git/SONAR/env/bin/activate && python server.py 

import numpy as np
import torch
METAL_DEVICE = torch.device("mps")

from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
)
vec2text = EmbeddingToTextModelPipeline(
    "text_sonar_basic_decoder",
    "text_sonar_basic_encoder",
    device=METAL_DEVICE,
)

# Create a network-available Queue object to load jobs.
import logging
import socket
from tlux.network import NetworkQueue
HOST = "Macsey.local"
PORT = 54321
# Create a NetworkQueue instance.
net_queue = NetworkQueue(
    listen_host=HOST,
    listen_port=PORT,
    socket_type=socket.AF_INET,
    egress_limit_mb_sec=-1,
)

# Listen for requests to generate embeddings.
while True:
    try:
        logging.info(f"Ready and waiting for input.")
        # Get a list of text from the NetworkQueue
        inputs, client_id = net_queue.get(return_client=True)
        inputs = np.asarray(inputs)
        assert ("float" in str(inputs.dtype)), f"Expected an array of embeddings as input."
        logging.info(f" recieved list of {len(inputs)} embeddings, decoding.")
        # Either generate emgbeddings or texts.
        try:
            result = vec2text.predict(
                torch.as_tensor(np.asarray(inputs)),
                target_lang="eng_Latn"
            )
        except Exception as e:
            result = str(e)
            logging.error(f"Encountered error when embedding: {e}")
        # Send back the result.
        net_queue.put(result, client_id=client_id)
    except Exception as e:
        logging.error(f"Worker encountered error: {e}")
