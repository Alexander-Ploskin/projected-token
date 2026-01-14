from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM


@dataclass
class XRAGBatch:
    xrag_input_ids: torch.Tensor
    xrag_attention_mask: torch.Tensor
    xrag_labels: Optional[torch.Tensor] = None
    retriever_input_ids: Optional[torch.Tensor] = None
    retriever_attention_mask: Optional[torch.Tensor] = None


class XRAGForCausalLM(torch.nn.Module):
    """
    Minimal wrapper:
      - inserts ONE projected token embedding at positions where input_ids == xrag_token_id
      - forwards through underlying HF CausalLM using inputs_embeds
    """
    def __init__(self, llm: AutoModelForCausalLM, projector: torch.nn.Module, xrag_token_id: int):
        super().__init__()
        self.llm = llm
        self.projector = projector
        self.xrag_token_id = int(xrag_token_id)

    def freeze_llm(self) -> None:
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        retrieval_embeds: Optional[torch.Tensor] = None,
    ):
        # Build token embeddings then replace XRAG token positions.
        emb_layer = self.llm.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids)

        if retrieval_embeds is not None:
            projected = self.projector(retrieval_embeds)  # [B, H]
            xrag_mask = (input_ids == self.xrag_token_id)  # [B, T]
            if xrag_mask.any():
                inputs_embeds = inputs_embeds.clone()
                b_idx, t_idx = xrag_mask.nonzero(as_tuple=True)   # indices of all XRAG tokens
                inputs_embeds[b_idx, t_idx] = projected[b_idx]    # write correct row vector

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
