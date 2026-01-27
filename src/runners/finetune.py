from src.runners.base import TrainRunner
from src.train.objectives import FinetuneObjective
from src.data.datamodule import build_finetune_dataloaders
from src.train.utils import validate_pretrain_ppl


class FinetuneRunner(TrainRunner):
    def __init__(self, cfg, accelerator, tracker):
        super().__init__(cfg, accelerator, tracker,
            stage_name="finetune",
            objective=FinetuneObjective(),
            build_dataloaders_fn=build_finetune_dataloaders,
            validate_fn=validate_pretrain_ppl, # TODO: Implement validation based on the task
        )
