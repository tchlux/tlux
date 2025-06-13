"""Wrapper for facebook/contriever that returns a float32 embedding."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer
from .contriever.contriever import Contriever


class ContrieverEmbedder:
    """Load once, reuse many times (thread-safe for read-only)."""

    def __init__(self, model_id: str = "facebook/contriever", device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = Contriever.from_pretrained(model_id).eval()

        if (device is None):
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        hidden = self.model(**toks).last_hidden_state  # (1, L, d)
        vec = hidden.mean(dim=1)                       # mean-pool -> (1, d)
        return vec.squeeze(0).cpu().numpy().astype(np.float32)
