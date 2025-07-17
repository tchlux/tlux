import os
import numpy as np
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

# Load once at module scope
this_dir = os.path.dirname(__file__)
tokenizer = Tokenizer.from_file(os.path.join(this_dir, "tokenizer.json"))

queries = [
    'What percentage of the Earth\'s atmosphere is oxygen?',
    '意大利首都是哪里？',
]
documents = [
    "The amount of oxygen in the atmosphere has fluctuated over the last 600 million years, reaching a peak of 35% during the Carboniferous period, significantly higher than today's 21%.",
    "羅馬是欧洲国家意大利首都和罗马首都广域市的首府及意大利全国的政治、经济、文化和交通中心，位于意大利半島中部的台伯河下游平原地，建城初期在七座小山丘上，故又名“七丘之城”。按城市范围内的人口计算，罗马是意大利人口最多的城市，也是欧盟人口第三多的城市。",
]

model = SentenceTransformer("facebook/drama-base", trust_remote_code=True)
print()
print(type(model))
print()
query_embs = model.encode(queries, prompt_name="query")
doc_embs = model.encode(documents)

scores = model.similarity(query_embs, doc_embs)
print(scores.tolist())
# Expected output: [[0.5310, 0.0821], [0.1298, 0.6181]]
print()

exit()



#!/usr/bin/env python3
"""
Build script for DRAMA-base embeddings:

1.  Downloads tokenizer + HF model.
2.  Exports a monolithic ONNX graph (batch=1, seq=512, opset=17)
    using Optimum’s Python API, with remote code enabled and the
    safe “eager” attention path.
3.  Runs dynamic INT-8 quantisation (optional but fast).
4.  Moves the resulting files into the project root and removes
    temporary folders.

Runtime dependencies (build-time only):
  torch>=2.4.1           transformers>=4.43
  optimum>=1.30          onnx>=1.18          onnxruntime>=1.18
"""

# from __future__ import annotations
import json
import platform
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from optimum.exporters.onnx.convert import onnx_export_from_model
from onnxruntime.quantization import quantize_dynamic, QuantType


MODEL_ID = "facebook/drama-base"
OPSET    = 17
BATCH    = 1
SEQ_LEN  = 512


def export_onnx(model_id: str, out_dir: Path) -> Path:
    """
    Export `model_id` as a monolithic ONNX file into `out_dir`
    and return the file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. build config with safe attention
    cfg = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation="eager"
    )

    # 2. load model once; this blocks the wrong-library heuristic
    # model = AutoModel.from_config(cfg, trust_remote_code=True)
    model = SentenceTransformer("facebook/drama-base", trust_remote_code=True, device="cpu")
    print(type(model))
    exit(1)

    # 3. export
    onnx_path, _ = onnx_export_from_model(
        # model_id,
        output=out_dir,
        task="feature-extraction",
        opset=OPSET,
        trust_remote_code=True,
        monolith=True,
        batch_size=BATCH,
        sequence_length=SEQ_LEN,
        model=model,
    )
    return onnx_path


def quantize(src: Path, dst_dir: Path) -> Path:
    """
    Dynamic-quantize `src` ONNX model -> `dst_dir/model_quantized.onnx`
    choosing weight type by arch (AVX2 vs Apple silicon).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model_quantized.onnx"

    # MatMul + Gemm weight-only int8 is plenty for CPUs
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        optimize_model=False,          # already optimised by exporter
    )
    return dst


def main() -> None:
    root       = Path(__file__).resolve().parent
    onnx_temp  = root / "onnx-temp"
    quant_temp = root / "onnx-int8-temp"

    # 1. save tokenizer (needed at runtime)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.save_pretrained(onnx_temp)

    # 2. export ONNX
    onnx_path = export_onnx(MODEL_ID, onnx_temp)

    # 3. (optional) quantize
    quant_path = quantize(onnx_path, quant_temp)

    # 4. move artefacts into repo root
    (root / "tokenizer.json").write_bytes(
        (onnx_temp / "tokenizer.json").read_bytes()
    )
    shutil.move(str(quant_path), str(root / quant_path.name))

    # 5. clean temp dirs
    shutil.rmtree(onnx_temp)
    shutil.rmtree(quant_temp)

    print("Tokenizer  ->", root / "tokenizer.json")
    print("Quantized ONNX ->", root / quant_path.name)


if __name__ == "__main__":
    main()
