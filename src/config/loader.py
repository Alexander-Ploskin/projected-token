from __future__ import annotations
import dataclasses
from typing import Any, Dict

import yaml

from src.config.schema import (
    ExperimentConfig,
    ModelConfig,
    RetrieverConfig,
    ProjectorConfig,
    DataConfig,
    TrainConfig,
    LoggingConfig,
    DistributedConfig,
    EvalConfig
)

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _asdict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)

def load_experiment_config(yaml_path: str, overrides: Dict[str, Any] | None = None) -> ExperimentConfig:
    base = ExperimentConfig()
    cfg_dict = _asdict(base)

    with open(yaml_path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    _deep_update(cfg_dict, user)

    if overrides:
        _deep_update(cfg_dict, overrides)

    return ExperimentConfig(
        task=cfg_dict.get("task", "pretrain"),
        model=ModelConfig(**cfg_dict["model"]),
        retriever=RetrieverConfig(**cfg_dict["retriever"]),
        projector=ProjectorConfig(**cfg_dict["projector"]),
        eval=EvalConfig(**cfg_dict.get("eval", {})),
        data=DataConfig(**cfg_dict["data"]),
        train=TrainConfig(**cfg_dict["train"]),
        objective=dict(cfg_dict["objective"]),  # dict
        logging=LoggingConfig(**cfg_dict["logging"]),
        distributed=DistributedConfig(**cfg_dict["distributed"]),
        extra=cfg_dict.get("extra", {}),
    )
