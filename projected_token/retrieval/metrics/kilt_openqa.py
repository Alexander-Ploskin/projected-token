"""KILT OpenQA retrieval metrics (in-accuracy relevance, ported from eval_kilt_openqa)."""

from __future__ import annotations

import re
import string
from typing import Any

import numpy as np


def normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def in_accuracy_match(pred: str, ref: str) -> bool:
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    if not ref_norm:
        return False
    return ref_norm in pred_norm


def hotpot_targets_from_row(row: dict[str, Any], answer_col: str = "answer") -> list[str]:
    value = row.get(answer_col)
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            token = str(item or "").strip()
            if token and token not in out:
                out.append(token)
        return out
    return []


def popqa_targets_from_row(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    obj = str(row.get("obj", "") or "").strip()
    if obj:
        out.append(obj)
    possible_answers = row.get("possible_answers")
    if possible_answers is None:
        return out
    if isinstance(possible_answers, str):
        for part in possible_answers.split("|"):
            token = str(part or "").strip()
            if token and token not in out:
                out.append(token)
    elif isinstance(possible_answers, (list, tuple)):
        for item in possible_answers:
            token = str(item or "").strip()
            if token and token not in out:
                out.append(token)
    return out


def scored_targets(targets: list[str]) -> list[str]:
    return [item for item in targets if normalize_text(item)]


def relevance_labels_for_hits(
    hits: list[dict[str, Any]],
    targets: list[str],
    *,
    passage_field: str = "document",
) -> list[int]:
    answers = scored_targets(targets)
    labels: list[int] = []
    for hit in hits:
        doc_text = str(hit.get(passage_field, "") or "")
        hit_flag = 0
        if doc_text:
            for answer in answers:
                if in_accuracy_match(doc_text, answer):
                    hit_flag = 1
                    break
        labels.append(hit_flag)
    return labels


def compute_kilt_openqa_metrics(
    relevance_flags: list[list[int]],
    top_k_values: list[int],
) -> dict[str, float]:
    results: dict[str, float] = {}
    if not relevance_flags:
        for k in top_k_values:
            results[f"recall@{k}"] = 0.0
            results[f"ndcg@{k}"] = 0.0
        results["mrr"] = 0.0
        return results

    for k in top_k_values:
        recall_values: list[float] = []
        ndcg_values: list[float] = []
        for labels in relevance_flags:
            sliced = labels[:k]
            recall_values.append(1.0 if any(sliced) else 0.0)
            dcg = 0.0
            for rank, rel in enumerate(sliced, start=1):
                if rel:
                    dcg += 1.0 / np.log2(rank + 1)
            ideal = sorted(sliced, reverse=True)
            idcg = 0.0
            for rank, rel in enumerate(ideal, start=1):
                if rel:
                    idcg += 1.0 / np.log2(rank + 1)
            ndcg_values.append(float(dcg / idcg) if idcg else 0.0)
        results[f"recall@{k}"] = float(np.mean(recall_values))
        results[f"ndcg@{k}"] = float(np.mean(ndcg_values))

    reciprocal_ranks: list[float] = []
    for labels in relevance_flags:
        rr = 0.0
        for rank, rel in enumerate(labels, start=1):
            if rel:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    results["mrr"] = float(np.mean(reciprocal_ranks))
    return results


def format_metrics_by_k(
    metrics: dict[str, float],
    top_k_values: list[int],
) -> list[dict[str, float | int]]:
    return [
        {
            "k": k,
            "recall": float(metrics[f"recall@{k}"]),
            "ndcg": float(metrics[f"ndcg@{k}"]),
        }
        for k in top_k_values
    ]
