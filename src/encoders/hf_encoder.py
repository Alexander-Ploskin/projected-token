from __future__ import annotations
import torch
from transformers import AutoModel, AutoTokenizer

from src.encoders.base import RetrieverEncoder


class HFMeanPoolRetriever(RetrieverEncoder):
    def __init__(self, name_or_path: str, torch_dtype=torch.bfloat16):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(name_or_path, torch_dtype=torch_dtype)
        self.model.eval()

    @property
    def embed_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return x  # [B, H]
