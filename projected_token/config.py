from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ClassConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    class_: str = Field(alias="class")
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    dataset: dict[str, Any]
    model: dict[str, Any]
    generation: dict[str, Any] = Field(default_factory=dict)
    experiment: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    metrics: list[dict[str, Any]] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    recipe: str = "mlp"


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    encoder: dict[str, Any]
    index: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


LEGACY_PREFIXES = {
    "evaluation.": "projected_token.",
    "data.": "projected_token.data.",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def rewrite_class_path(class_path: str) -> str:
    for old, new in LEGACY_PREFIXES.items():
        if class_path.startswith(old):
            return new + class_path[len(old):]
    return class_path


def import_object(path: str) -> Any:
    path = rewrite_class_path(path)
    if "." not in path:
        raise ValueError(f"Expected import path 'module.Object', got: {path}")
    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def instantiate(config: Mapping[str, Any], *, kind: str = "object") -> Any:
    if "class" not in config:
        raise ValueError(f"Missing 'class' key in {kind} config")
    class_path = rewrite_class_path(str(config["class"]))
    cls = import_object(class_path)
    return cls(*config.get("args", []), **config.get("kwargs", {}))


def validate_experiment_config(config: Mapping[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(config)


def validate_training_config(config: Mapping[str, Any]) -> TrainingConfig:
    return TrainingConfig.model_validate(config)


def validate_retrieval_config(config: Mapping[str, Any]) -> RetrievalConfig:
    return RetrievalConfig.model_validate(config)
