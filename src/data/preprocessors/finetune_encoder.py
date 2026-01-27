from __future__ import annotations

from typing import Any
import copy

import torch

from src.data.preprocessors.utils import (
    crop_text_by_retriever_tokens,
    get_random,
)


XRAG_TOKEN = "[XRAG]"


def _find_first_user_idx(messages: list[dict[str, str]]) -> int:
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            return i
    raise ValueError("No user message found")


def _ensure_last_assistant(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(messages) == 0 or messages[-1].get("role") != "assistant":
        return messages + [{"role": "assistant", "content": ""}]
    return messages


def _extract_context_text(messages: list[dict[str, str]], mode: str = "all_user") -> str:
    if mode == "first_user":
        i = _find_first_user_idx(messages)
        return (messages[i].get("content") or "").strip()

    if mode == "all_user":
        parts = []
        for m in messages:
            if m.get("role") == "user":
                c = (m.get("content") or "").strip()
                if c:
                    parts.append(c)
        return "\n\n".join(parts).strip()

    raise ValueError(f"Unknown context_source={mode}")


def encode_with_chat_format_finetune(
    example: dict[str, Any],
    *,
    tokenizer,
    retriever_tokenizer=None,
    max_seq_length: int,
    retrieval_embed_length: int,
    xrag_token: str = XRAG_TOKEN,
    context_source: str = "all_user",
    replace_user_with_xrag: bool = True,
    xrag_user_prefix: str = "Please answer this question: ",
    use_retriever_embed: bool = True,
    retriever_min_length: int = 32,
    retriever_max_length: int = 180,
    retriever_crop_strategy: str = "log_uniform",
    rng_seed: int = 13,
    return_teacher: bool = False,
    teacher_user_prefix: str = "",
) -> dict[str, Any]:
    """
    Input example (your format):
      - id: str
      - task_type: str (unused here, but kept in dataset)
      - messages: list[{"role":"user"/"assistant","content":str}, ...]

    Output (student-only XRAG training):
      - xrag_input_ids, xrag_labels: tensors (LLM stream)
      - retriever_input_text: [str] (text that will be encoded by the retriever into retrieval_embeds)
    """
    messages: list[dict[str, str]] = example["messages"]
    ex_id = str(example.get("id", ""))

    # deterministic RNG per example (safe under multiprocessing map)
    rng = get_random(rng_seed, ex_id)

    # 1) Build the text that will be compressed into XRAG embedding
    context_text = _extract_context_text(messages, mode=context_source)

    retriever_text = ""
    if use_retriever_embed:
        if retriever_tokenizer is None:
            raise ValueError("use_retriever_embed=True but retriever_tokenizer=None")

        retriever_text = crop_text_by_retriever_tokens(
            text=context_text,
            retriever_tokenizer=retriever_tokenizer,
            min_len=retriever_min_length,
            max_len=retriever_max_length,
            strategy=retriever_crop_strategy,
            rng=rng,
        )

    # 2) Build XRAG-token placeholder prompt and optionally replace the user content
    xrag_tokens = " ".join([xrag_token] * retrieval_embed_length)

    messages_student = copy.deepcopy(messages)
    if replace_user_with_xrag:
        user_idx = _find_first_user_idx(messages_student)
        messages_student[user_idx]["content"] = f"{xrag_user_prefix}{xrag_tokens}"

    # 3) Tokenize full conversation and prompt-only version for masking labels
    messages_student = _ensure_last_assistant(messages_student)

    full_ids = tokenizer.apply_chat_template(
        messages_student,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_seq_length,
    )

    messages_prompt = copy.deepcopy(messages_student)
    messages_prompt[-1]["content"] = ""  # blank assistant for prompt boundary
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

    # guard: if everything is ignored, CE can become NaN; keep at least one supervised token [web:592]
    if not torch.any(labels != -100):
        labels[-1] = input_ids[-1]

    ret: dict[str, Any] = {
        "xrag_input_ids": input_ids,
        "xrag_labels": labels,
    }
    if use_retriever_embed:
        ret["retriever_input_text"] = [retriever_text]

    if return_teacher:
        # Teacher = same conversation but user gets real context (no [XRAG] tokens)
        messages_teacher = copy.deepcopy(messages)
        if teacher_user_prefix or context_text:
            user_idx = _find_first_user_idx(messages_teacher)
            # Put the actual context text into prompt (teacher sees it)
            # (If you want a different phrasing, change this line.)
            messages_teacher[user_idx]["content"] = (
                f"{teacher_user_prefix}{context_text}\n\n{messages_teacher[user_idx]['content']}"
            ).strip()

        messages_teacher = _ensure_last_assistant(messages_teacher)

        full_ids_t = tokenizer.apply_chat_template(
            messages_teacher,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_seq_length,
        )

        messages_prompt_t = copy.deepcopy(messages_teacher)
        messages_prompt_t[-1]["content"] = ""
        prompt_ids_t = tokenizer.apply_chat_template(
            messages_prompt_t,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_seq_length,
        )

        input_ids_t = torch.tensor(full_ids_t, dtype=torch.long)
        labels_t = input_ids_t.clone()
        cutoff_t = min(len(prompt_ids_t), len(full_ids_t))
        labels_t[:cutoff_t] = -100
        if not torch.any(labels_t != -100):
            labels_t[-1] = input_ids_t[-1]

        ret["input_ids"] = input_ids_t
        ret["labels"] = labels_t

    return ret
