from eval.runners.popqa import PopQAEvalRunner
from src.config.schema import ExperimentConfig


class EvalRunnerFactory:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def create(self):
        task = self.cfg.task
        if task == 'eval_popqa':
            return PopQAEvalRunner(self.cfg)
        raise ValueError(f"Unknown task: {task}")
