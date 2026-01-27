from __future__ import annotations
from typing import Any

import torch


def _pad_1d(seqs: list[torch.Tensor], pad_value: int, left: bool = False) -> torch.Tensor:
    if not left:
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)  # [web:625]
    flipped = [torch.flip(x, dims=[0]) for x in seqs]
    padded = torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=pad_value)  # [web:625]
    return torch.flip(padded, dims=[1])


def _llm_pad_side(tokenizer) -> tuple[int, bool]:
    pad_id = tokenizer.pad_token_id
    left = getattr(tokenizer, "padding_side", "right") == "left"
    return pad_id, left


def _collate_xrag_stream(samples: list[dict[str, Any]], *, llm_tokenizer) -> dict[str, torch.Tensor]:
    pad_id, left = _llm_pad_side(llm_tokenizer)

    xrag_input_ids = _pad_1d([s["xrag_input_ids"] for s in samples], pad_id, left=left)
    xrag_labels = _pad_1d([s["xrag_labels"] for s in samples], -100, left=left)
    xrag_attention_mask = (xrag_input_ids != pad_id).long()

    return {
        "xrag_input_ids": xrag_input_ids,
        "xrag_labels": xrag_labels,
        "xrag_attention_mask": xrag_attention_mask,
    }


def _collate_teacher_stream(samples: list[dict[str, Any]], *, llm_tokenizer) -> dict[str, torch.Tensor]:
    pad_id, left = _llm_pad_side(llm_tokenizer)

    input_ids = _pad_1d([s["input_ids"] for s in samples], pad_id, left=left)
    labels = _pad_1d([s["labels"] for s in samples], -100, left=left)
    attention_mask = (input_ids != pad_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _tokenize_retriever_texts(
    samples: list[dict[str, Any]],
    *,
    retriever_tokenizer,
    retriever_max_length: int,
    return_mapping: bool,
) -> dict[str, torch.Tensor]:
    """
    Expects each sample has "retriever_input_text" as list[str].
    - return_mapping=False: just flatten texts (pretrain behavior).
    - return_mapping=True: also return retriever_text_to_sample (finetune behavior).
    """
    if retriever_tokenizer is None:
        return {}

    if len(samples) == 0 or "retriever_input_text" not in samples[0]:
        return {}

    texts: list[str] = []
    text_to_sample: list[int] = []

    for i, s in enumerate(samples):
        chunks = s.get("retriever_input_text", [])
        if not isinstance(chunks, list):
            chunks = [str(chunks)]
        if len(chunks) == 0:
            chunks = [""]

        for t in chunks:
            texts.append(t)
            if return_mapping:
                text_to_sample.append(i)

    tok = retriever_tokenizer(
        texts,
        max_length=retriever_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    out: dict[str, torch.Tensor] = {
        "retriever_input_ids": tok["input_ids"],
        "retriever_attention_mask": tok["attention_mask"],
    }
    if return_mapping:
        out["retriever_text_to_sample"] = torch.tensor(text_to_sample, dtype=torch.long)
    return out


class _BaseXRAGCollator:
    """
    Common collator for XRAG student stream + optional retriever text tokenization.
    Subclasses can choose whether to return a mapping from retriever rows to samples.
    """
    def __init__(self, llm_tokenizer, retriever_tokenizer=None, retriever_max_length: int = 180, *, return_retriever_mapping: bool):
        self.llm_tokenizer = llm_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.retriever_max_length = retriever_max_length
        self.return_retriever_mapping = return_retriever_mapping

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, Any] = {}
        batch.update(_collate_xrag_stream(samples, llm_tokenizer=self.llm_tokenizer))
        if "input_ids" in samples[0]:
            batch.update(_collate_teacher_stream(samples, llm_tokenizer=self.llm_tokenizer))
        batch.update(
            _tokenize_retriever_texts(
                samples,
                retriever_tokenizer=self.retriever_tokenizer,
                retriever_max_length=self.retriever_max_length,
                return_mapping=self.return_retriever_mapping,
            )
        )
        return batch


class PretrainCollator(_BaseXRAGCollator):
    def __init__(self, llm_tokenizer, retriever_tokenizer=None, retriever_max_length: int = 180):
        super().__init__(
            llm_tokenizer=llm_tokenizer,
            retriever_tokenizer=retriever_tokenizer,
            retriever_max_length=retriever_max_length,
            return_retriever_mapping=False,  # keep old behavior
        )


class FinetuneCollator(_BaseXRAGCollator):
    def __init__(self, llm_tokenizer, retriever_tokenizer=None, retriever_max_length: int = 180):
        super().__init__(
            llm_tokenizer=llm_tokenizer,
            retriever_tokenizer=retriever_tokenizer,
            retriever_max_length=retriever_max_length,
            return_retriever_mapping=True,  # needed for multi-chunk aggregation
        )
