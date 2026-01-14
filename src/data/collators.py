from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch


def _pad_1d(seqs: List[torch.Tensor], pad_value: int, left: bool = False) -> torch.Tensor:
    if not left:
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    flipped = [torch.flip(x, dims=[0]) for x in seqs]
    padded = torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=pad_value)
    return torch.flip(padded, dims=[1])


class PretrainCollator:
    def __init__(self, llm_tokenizer, retriever_tokenizer=None, retriever_max_length: int = 180):
        self.llm_tokenizer = llm_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.retriever_max_length = retriever_max_length

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.llm_tokenizer.pad_token_id
        left = getattr(self.llm_tokenizer, "padding_side", "right") == "left"

        xrag_input_ids = _pad_1d([s["xrag_input_ids"] for s in samples], pad_id, left=left)
        xrag_labels = _pad_1d([s["xrag_labels"] for s in samples], -100, left=left)
        xrag_attention_mask = (xrag_input_ids != pad_id).long()

        batch = {
            "xrag_input_ids": xrag_input_ids,
            "xrag_labels": xrag_labels,
            "xrag_attention_mask": xrag_attention_mask,
        }

        if self.retriever_tokenizer is not None and "retriever_input_text" in samples[0]:
            texts = []
            for s in samples:
                texts.extend(s["retriever_input_text"])
            tok = self.retriever_tokenizer(
                texts,
                max_length=self.retriever_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch["retriever_input_ids"] = tok["input_ids"]
            batch["retriever_attention_mask"] = tok["attention_mask"]

        return batch
