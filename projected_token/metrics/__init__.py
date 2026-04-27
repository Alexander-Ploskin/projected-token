from .metric import Metric
from .gpt_score.gpt_score import GPTScoreMetric
from .qa_score.qa_score import QAScoreMetric, compute_in_accuracy

__all__ = ["Metric", "AlignScoreMetric", "GPTScoreMetric", "QAScoreMetric", "compute_in_accuracy"]


def __getattr__(name: str):
    if name == "AlignScoreMetric":
        from .align_score.align_score import AlignScoreMetric

        return AlignScoreMetric
    raise AttributeError(name)
