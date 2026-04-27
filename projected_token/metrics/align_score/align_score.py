from projected_token.metrics import Metric
from typing import Any

from projected_token.metrics.align_score.alignscore.alignscore import AlignScore


class AlignScoreMetric(Metric):
    def __init__(self, config: dict) -> None:
        """
        Initialize AlignScore metric.
        
        Expected config parameters:
        - model: model name (e.g., "roberta-base")
        - batch_size: batch size for scoring
        - device: device to run on (e.g., "cuda", "cpu")
        - ckpt_path: path to AlignScore checkpoint
        - evaluation_mode: evaluation mode (e.g., "nli_sp", "nli", "bin_sp", "bin")
        """
        self._scorer = AlignScore(
            model=config["model"],
            batch_size=config["batch_size"],
            device=config["device"],
            ckpt_path=config["ckpt_path"],
            evaluation_mode=config["evaluation_mode"],
        )
        
    def __call__(self, original: str, rephrased: str) -> dict[str, Any]:
        """
        Calculate AlignScore between original and rephrased text.
        
        Args:
            original: Original context text
            rephrased: Rephrased text to evaluate
            
        Returns:
            Dictionary with alignment score
        """
        # AlignScore expects lists of contexts and claims
        scores = self._scorer.score(contexts=[original], claims=[rephrased])
        
        # Return the first (and only) score
        return {"align_score": float(scores[0])}
