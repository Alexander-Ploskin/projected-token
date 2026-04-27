from __future__ import annotations

from statistics import mean
from typing import Iterable

import numpy as np


def recall_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    return len(relevant & set(retrieved[:k])) / len(relevant) if relevant else 0.0


def precision_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    return len(relevant & set(retrieved[:k])) / k if k else 0.0


def mrr(relevant: set[int], retrieved: list[int]) -> float:
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    dcg = sum(1.0 / np.log2(rank + 1) for rank, doc_id in enumerate(retrieved[:k], 1) if doc_id in relevant)
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return float(dcg / ideal) if ideal else 0.0


def aggregate_rankings(cases: Iterable[tuple[set[int], list[int]]], k_values: list[int]) -> dict[str, float]:
    cases = list(cases)
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = float(mean(recall_at_k(rel, ret, k) for rel, ret in cases)) if cases else 0.0
        out[f"precision@{k}"] = float(mean(precision_at_k(rel, ret, k) for rel, ret in cases)) if cases else 0.0
        out[f"ndcg@{k}"] = float(mean(ndcg_at_k(rel, ret, k) for rel, ret in cases)) if cases else 0.0
    out["mrr"] = float(mean(mrr(rel, ret) for rel, ret in cases)) if cases else 0.0
    return out
