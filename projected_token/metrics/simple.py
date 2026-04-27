from __future__ import annotations

from difflib import SequenceMatcher
from statistics import mean


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def score_pair(reference: str, candidate: str) -> dict[str, float]:
    ref_tokens = set(tokenize(reference))
    cand_tokens = set(tokenize(candidate))
    ref_len = len(reference.split())
    cand_len = len(candidate.split())
    return {
        "jaccard": jaccard_similarity(ref_tokens, cand_tokens),
        "char_sim": SequenceMatcher(None, reference.lower(), candidate.lower()).ratio(),
        "word_overlap": len(ref_tokens & cand_tokens) / max(len(ref_tokens), 1),
        "length_ratio": cand_len / max(ref_len, 1),
    }


def aggregate_simple_metrics(references: list[str], candidates: list[str]) -> dict[str, float]:
    rows = [score_pair(ref, cand) for ref, cand in zip(references, candidates) if ref.strip() and cand.strip()]
    if not rows:
        return {}
    return {f"avg_{key}": float(mean(row[key] for row in rows)) for key in rows[0]}
