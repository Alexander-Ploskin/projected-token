from __future__ import annotations
from typing import Any, Dict, List
import random
import torch


XRAG_TOKEN = "[XRAG]"

ParaphraseInstructions = [
    'Background: {xrag_token} means the same as',
    "Background: {xrag_token} Can you put the above sentences in your own terms?",
    "Background: {xrag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {xrag_token}\n(2)",
    "Background: {xrag_token} is a paraphrase of what?",
    "Background: {xrag_token} Could you give me a different version of the background sentences above?",
    "In other words, background: {xrag_token} is just another way of saying:",
    "You're getting across the same point whether you say background: {xrag_token} or",
    "Background: {xrag_token} After uppacking the ideas in the background information above, we got:",
    "Background: {xrag_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {xrag_token}, which also means:",
    "Strip away the mystery, and you'll find background: {xrag_token} is simply another rendition of:",
    "The essence of background: {xrag_token} is captured again in the following statement:",
]

def _pick_text(example: Dict[str, Any], prefer_summary: bool) -> str:
    if prefer_summary and isinstance(example.get("summary"), str) and example["summary"].strip():
        return example["summary"]
    if isinstance(example.get("text"), str) and example["text"].strip():
        return example["text"]
    for k in ["content", "article", "document"]:
        if isinstance(example.get(k), str) and example[k].strip():
            return example[k]
    raise KeyError(f"Can't find article text in keys={list(example.keys())}")


def _encode_chat_format(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Model-agnostic chat encoding using tokenizer.apply_chat_template().

    Returns:
      - input_ids: 1D LongTensor
      - labels:    1D LongTensor, with non-assistant tokens masked to -100
    """
    # Full conversation tokens
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_seq_length,
    )  # chat template API [web:302]

    # Compute boundary: everything before assistant content should be masked.
    # We do it by encoding the same conversation but with empty assistant content + add_generation_prompt=True,
    # so it includes the assistant prefix but no assistant text.
    prompt_only = [messages[0], {"role": "assistant", "content": ""}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_only,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_seq_length,
    )  # add_generation_prompt behavior [web:314]

    cutoff = min(len(prompt_ids), len(full_ids))

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[:cutoff] = -100
    return {"input_ids": input_ids, "labels": labels}


def encode_with_chat_format_pretrain(
    example: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
    xrag_token: str,
    retrieval_embed_length: int = 1,
    retriever_text_source: str = "text",
) -> Dict[str, Any]:
    # document for retriever
    if retriever_text_source == "summary":
        document = _pick_text(example, prefer_summary=True)
    else:
        document = _pick_text(example, prefer_summary=False)

    xrag_token_str = " ".join([xrag_token] * retrieval_embed_length)
    instruction = random.choice(ParaphraseInstructions).format_map({"xrag_token": xrag_token_str})

    messages_full = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": document},
    ]
    messages_prompt = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": ""},
    ]

    full_ids = tokenizer.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_seq_length,
    )

    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()

    cutoff = min(len(prompt_ids), len(full_ids))
    labels[:cutoff] = -100  # mask everything before assistant content

    return {
        "xrag_input_ids": input_ids,
        "xrag_labels": labels,
        "retriever_input_text": [document],  # retriever encodes the same document
    }
