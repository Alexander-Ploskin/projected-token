from __future__ import annotations
from accelerate.utils import set_seed
import torch


def seed_everything(seed: int) -> None:
    set_seed(seed)


def mean_across_processes(accelerator, x: torch.Tensor) -> torch.Tensor:
    # x: scalar tensor on each rank
    gathered = accelerator.gather(x)
    return gathered.mean()