from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from projected_token.config import load_yaml


TRAINING_RECIPES = {
    "mlp": "projected_token.training.trainer_mlp:create_mlp_trainer",
    "lora": "projected_token.training.trainer_lora:create_lora_trainer",
    "full": "projected_token.training.trainer_full:create_full_trainer",
}

ARGPARSE_RECIPES = {
    "flat": "projected_token.training.recipes.trainer_flat",
    "msmarco": "projected_token.training.recipes.trainer_msmarco",
    "msmarco_v2": "projected_token.training.recipes.trainer_msmarco_v2",
    "distill": "projected_token.training.recipes.trainer_distill",
    "hotpot_distill": "projected_token.training.recipes.trainer_hotpot_distill",
}


def _import_factory(path: str):
    module_name, name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def _config_to_argv(config: dict[str, Any]) -> list[str]:
    if "cli_args" in config:
        return [str(v) for v in config["cli_args"]]
    args: list[str] = []
    for key, value in config.items():
        if key == "recipe" or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.extend([flag, str(value)])
    return args


def run_training(config_path: str | Path, *, epochs: int | None = None) -> None:
    config = load_yaml(config_path)
    recipe = str(config.get("recipe") or Path(config_path).stem.replace("projector_", ""))
    if epochs is not None:
        config["epochs"] = epochs
    if recipe in TRAINING_RECIPES:
        factory = _import_factory(TRAINING_RECIPES[recipe])
        trainer = factory(config)
        trainer.train(num_epochs=config.get("epochs", 3), val_every_n_steps=config.get("val_every_n_steps", 100))
        return
    if recipe in ARGPARSE_RECIPES:
        module = importlib.import_module(ARGPARSE_RECIPES[recipe])
        argv = [ARGPARSE_RECIPES[recipe]] + _config_to_argv(config)
        old_argv = sys.argv
        try:
            sys.argv = argv
            module.main()
        finally:
            sys.argv = old_argv
        return
    raise ValueError(f"Unknown training recipe: {recipe}")
