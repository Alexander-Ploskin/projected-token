from __future__ import annotations
from typing import Any, Dict
import torch

from src.data.preprocessors.utils import crop_text_by_retriever_tokens, get_random

XRAG_TOKEN = "[XRAG]"

ParaphraseInstructions = [
    "Background: {xrag_token} means the same as",
    "Background: {xrag_token} Can you put the above sentences in your own terms?",
    "Background: {xrag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {xrag_token}\n(2)",
    "Background: {xrag_token} is a paraphrase of what?",
    "Background: {xrag_token} Could you give me a different version of the background sentences above?",
    "In other words, background: {xrag_token} is just another way of saying:",
    "You're getting across the same point whether you say background: {xrag_token} or",
    "Background: {xrag_token} After unpacking the ideas in the background information above, we got:",
    "Background: {xrag_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {xrag_token}, which also means:",
    "Strip away the mystery, and you'll find background: {xrag_token} is simply another rendition of:",
    "The essence of background: {xrag_token} is captured again in the following statement:",
    "Rephrase the following background: {xrag_token}",
    "Provide an alternative phrasing for: {xrag_token}",
    "How would you rewrite this background?\n{xrag_token}",
    "Background: {xrag_token}\n\nRewrite the above text while preserving its meaning.",
    "Express the following background information differently: {xrag_token}",
    "Give a synonymic version of: {xrag_token}",
    "Background: {xrag_token}\n\nIn plain English, this means:",
    "What's another way to articulate the following? {xrag_token}",
    "Synthesize the following information into a new sentence: {xrag_token}",
    "Background: {xrag_token} can be summarized or restated as:",
    "Can you provide a fresh take on this background? {xrag_token}",
    "If you had to say {xrag_token} differently, how would you do it?",
    "Here is some text: {xrag_token}. Rewrite it.",
    "{xrag_token} -> rephrased:",
    "Input: {xrag_token}. Output: A variation of the input.",
    "Reformulate the following sentence: {xrag_token}",
    "Transform this text into equivalent words: {xrag_token}",
    "Make a new version of: {xrag_token}",
    "Paraphrase the text above: {xrag_token}",
    "I need a synonym for this entire block: {xrag_token}",
    "Same meaning, different words: {xrag_token}",
    "Can you spin this text? {xrag_token}",
    "Write it in another way: {xrag_token}",
    "Original: {xrag_token}. New version:",
    "Source: {xrag_token}. Target: Paraphrase.",
    "Text to rewrite: {xrag_token}",
    "Please paraphrase the following excerpt: {xrag_token}",
    "Background information: {xrag_token}. Restate it.",
    "What is an alternative way to express {xrag_token}?",
    "Turn this into a new sentence with the same meaning: {xrag_token}",
    "Draft a rephrased version of: {xrag_token}",
    "Use your own words to describe: {xrag_token}",
    "Re-write the paragraph below:\n{xrag_token}",
    "Show me another way to write: {xrag_token}",
    "Convert this to a paraphrase: {xrag_token}",
    "Equivalently: {xrag_token}",
    "Reword: {xrag_token}",
    "Say it differently: {xrag_token}",
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


def encode_with_chat_format_pretrain(
    example: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
    max_length: int,
    min_length: int,
    crop_strategy: str,
    xrag_token: str,
    retrieval_embed_length: int = 1,
    retriever_text_source: str = "text",
    rng_seed: int = 52,
    rng_key: str = "id",   # which field to use as stable identifier
) -> Dict[str, Any]:
    # Build deterministic RNG for this example (important for num_proc>1) [web:536]
    ex_key = str(example.get(rng_key, ""))  # if missing, still deterministic but lower quality
    rng = get_random(rng_seed, ex_key)

    # document for retriever
    if retriever_text_source == "summary":
        document = _pick_text(example, prefer_summary=True)
    else:
        document = _pick_text(example, prefer_summary=False)

    # Length sampling / crop uses the per-example RNG
    document = crop_text_by_retriever_tokens(
        text=document,
        retriever_tokenizer=tokenizer,
        min_len=min_length,
        max_len=max_length,
        strategy=crop_strategy,
        rng=rng,
    )

    # Instruction sampling should also use the same RNG (avoid global random)
    xrag_token_str = " ".join([xrag_token] * retrieval_embed_length)
    instruction = rng.choice(ParaphraseInstructions).format_map({"xrag_token": xrag_token_str})

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
    labels[:cutoff] = -100

    return {
        "xrag_input_ids": input_ids,
        "xrag_labels": labels,
        "retriever_input_text": [document],
    }
