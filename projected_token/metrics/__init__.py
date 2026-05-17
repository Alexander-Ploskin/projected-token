from .metric import Metric
from .qa_text import (
    answer_in_prediction,
    contains_match,
    exact_match,
    in_accuracy_match,
    max_over_refs,
    normalize_answer,
    score_prediction,
    token_f1,
)

__all__ = [
    "Metric",
    "AlignScoreMetric",
    "GPTScoreMetric",
    "QAScoreMetric",
    "answer_in_prediction",
    "compute_in_accuracy",
    "contains_match",
    "exact_match",
    "in_accuracy_match",
    "max_over_refs",
    "normalize_answer",
    "score_prediction",
    "token_f1",
]


def __getattr__(name: str):
    if name == "AlignScoreMetric":
        from .align_score.align_score import AlignScoreMetric

        return AlignScoreMetric
    if name == "GPTScoreMetric":
        from .gpt_score.gpt_score import GPTScoreMetric

        return GPTScoreMetric
    if name in {"QAScoreMetric", "compute_in_accuracy"}:
        from .qa_score.qa_score import QAScoreMetric, compute_in_accuracy

        return {"QAScoreMetric": QAScoreMetric, "compute_in_accuracy": compute_in_accuracy}[name]
    raise AttributeError(name)
