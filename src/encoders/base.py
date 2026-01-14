from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class RetrieverEncoder(ABC, torch.nn.Module):
    @property
    @abstractmethod
    def embed_dim(self) -> int:
        pass

    @abstractmethod
    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns:
          retrieval_embeds: float tensor [batch, embed_dim] (single-vector retriever in this mock).
        """
        pass
