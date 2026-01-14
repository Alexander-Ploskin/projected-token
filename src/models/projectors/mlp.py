from __future__ import annotations
import torch


class TwoLayerMLPProjector(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, retrieval_embeds: torch.Tensor) -> torch.Tensor:
        # retrieval_embeds: [B, in_dim]
        return self.net(retrieval_embeds)  # [B, out_dim]
