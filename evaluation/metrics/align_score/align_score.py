from evaluation.metrics import Metric

from typing import Any
# from alignscore import AlignScore 


class AlignScoreMetric(Metric):
    def __init__(self, config: dict) -> None:
        # self._scorer = AlignScore(
        #     model=config["model"],
        #     batch_size=config["batch_size"],
        #     device=config["device"],
        #     ckpt_path=config["ckpt"],
        #     evaluation_mode=config["eval_mode"],
        # )
        pass

    def __call__(self, original: str, rephrased: str) -> dict[str, Any]:
        # result = self._scorer.score(contexts=[original], claims=[rephrased])[0]
        # return {"score": result}
        return {"score": -1.0}
