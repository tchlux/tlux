# source ~/Git/SONAR/env/bin/activate && python server.py 

print("Importing libraries..", flush=True)
import numpy as np
import torch
METAL_DEVICE = torch.device("mps")

print("Loading models..", flush=True)
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
)
vec2text = EmbeddingToTextModelPipeline(
    "text_sonar_basic_decoder",
    "text_sonar_basic_encoder",
    device=METAL_DEVICE,
)
text2vec = TextToEmbeddingModelPipeline(
    "text_sonar_basic_encoder", 
    "text_sonar_basic_encoder",
    device=METAL_DEVICE,
)
print("", flush=True)


inputs = [
    "Hello! How are you?",
]
print("inputs: ", inputs, flush=True)
result = text2vec.predict(
    inputs,
    source_lang="eng_Latn"
).cpu().numpy()
print("result: ", result, flush=True)

inputs = np.asarray(result)
inputs += np.random.normal(size=inputs.shape) * 0.01


result = vec2text.predict(
    torch.as_tensor(np.asarray(inputs)),
    target_lang="eng_Latn"
)

print("result: ", result, flush=True)
