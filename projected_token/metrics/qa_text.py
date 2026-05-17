from __future__ import annotations

import re
import string
from collections.abc import Callable, Sequence


def normalize_answer(value: str) -> str:
    """Normalize QA answers following the OSCAR string matching setup."""
    text = (value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(value: str) -> list[str]:
    return normalize_answer(value).split()


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_match(prediction: str, reference: str) -> bool:
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)
    if not pred_norm or not ref_norm:
        return False
    return ref_norm in pred_norm or pred_norm in ref_norm


def answer_in_prediction(prediction: str, reference: str) -> bool:
    return contains_match(prediction, reference)


def in_accuracy_match(prediction: str, reference: str) -> bool:
    """OSCAR Appendix A.2: normalized reference is a substring of prediction."""
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)
    if not ref_norm:
        return False
    return ref_norm in pred_norm


def max_over_refs(
    prediction: str,
    references: Sequence[str],
    fn: Callable[[str, str], float | bool],
) -> float:
    if not references:
        return 0.0
    return max(float(fn(prediction, reference)) for reference in references)


def score_prediction(prediction: str, references: Sequence[str]) -> dict[str, float]:
    refs = [str(reference) for reference in references if str(reference).strip()]
    return {
        "em": max_over_refs(prediction, refs, exact_match),
        "f1": max_over_refs(prediction, refs, token_f1),
        "answer_in_prediction": max_over_refs(prediction, refs, contains_match),
        "in_accuracy": max_over_refs(prediction, refs, in_accuracy_match),
    }
