from __future__ import annotations

from src.runners.base import TrainRunner
from src.train.objectives import PretrainObjective
from src.data.datamodule import build_pretrain_dataloaders
from src.train.utils import validate_pretrain_ppl


class PretrainRunner(TrainRunner):
    def __init__(self, cfg, accelerator, tracker):
        super().__init__(cfg, accelerator, tracker,
            stage_name="pretrain",
            objective=PretrainObjective(),
            build_dataloaders_fn=build_pretrain_dataloaders,
            validate_fn=validate_pretrain_ppl,
        )
