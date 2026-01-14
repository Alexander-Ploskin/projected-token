from __future__ import annotations
import dataclasses
from typing import Any, Dict

import yaml

from src.config.schema import ExperimentConfig


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

    # minimal manual reconstruction (keeps mock simple)
    return ExperimentConfig(
        task=cfg_dict.get("task", "pretrain"),
        model=type(base.model)(**cfg_dict["model"]),
        retriever=type(base.retriever)(**cfg_dict["retriever"]),
        projector=type(base.projector)(**cfg_dict["projector"]),
        data=type(base.data)(**cfg_dict["data"]),
        train=type(base.train)(**cfg_dict["train"]),
        objective=type(base.objective)(**cfg_dict["objective"]),
        logging=type(base.logging)(**cfg_dict["logging"]),
        distributed=type(base.distributed)(**cfg_dict["distributed"]),
        extra=cfg_dict.get("extra", {}),
    )
