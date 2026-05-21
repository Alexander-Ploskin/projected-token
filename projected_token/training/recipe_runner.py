from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from projected_token.config import load_yaml
from projected_token.artifacts import create_run_layout, write_config_lock


TRAINING_RECIPES = {
    "mlp": "projected_token.training.trainer_mlp:create_mlp_trainer",
    "lora": "projected_token.training.trainer_lora:create_lora_trainer",
    "full": "projected_token.training.trainer_full:create_full_trainer",
    "advanced": "projected_token.training.advanced_trainer:create_advanced_trainer",
}

ARGPARSE_RECIPES = {
    "flat": "projected_token.training.recipes.trainer_flat",
    "msmarco": "projected_token.training.recipes.trainer_msmarco",
    "msmarco_v2": "projected_token.training.recipes.trainer_msmarco_v2",
    "distill": "projected_token.training.recipes.trainer_distill",
    "query_distill": "projected_token.training.recipes.trainer_query_distill",
    "hotpot_distill": "projected_token.training.recipes.trainer_hotpot_distill",
}


def _import_factory(path: str):
    module_name, name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def _config_to_argv(config: dict[str, Any]) -> list[str]:
    if "cli_args" in config:
        return [str(v) for v in config["cli_args"]]
    args: list[str] = []
    skip_keys = {"recipe", "run_base_dir", "run_root", "metrics_dir", "plots_dir"}
    for key, value in config.items():
        if key in skip_keys or value is None:
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


def _prepare_run_layout(config: dict[str, Any], recipe: str, config_path: str | Path):
    config_name = Path(config_path).stem
    run_id = config.get("run_id")
    layout = create_run_layout(
        experiment_name=f"{recipe}-{config_name}",
        base_dir=config.get("run_base_dir", "artifacts/runs"),
        run_id=run_id,
    )
    config["run_root"] = str(layout.root)
    config["output_dir"] = str(layout.checkpoints_dir)
    config["log_dir"] = str(layout.logs_dir)
    config["metrics_dir"] = str(layout.metrics_dir)
    config["plots_dir"] = str(layout.plots_dir)
    write_config_lock(config, layout.root / "config.lock.yaml")
    print(f"[train] run_root={layout.root}")
    return layout


def run_training(config_path: str | Path, *, epochs: int | None = None) -> dict[str, Any]:
    config_source_path = Path(config_path)
    config = load_yaml(config_path)
    recipe = str(config.get("recipe") or Path(config_path).stem.replace("projector_", ""))
    if epochs is not None:
        config["epochs"] = epochs
    layout = _prepare_run_layout(config, recipe, config_path)
    run_config_copy_path = layout.root / config_source_path.name
    run_config_copy_path.write_text(config_source_path.read_text(encoding="utf-8"), encoding="utf-8")
    if recipe in TRAINING_RECIPES:
        factory = _import_factory(TRAINING_RECIPES[recipe])
        trainer = factory(config)
        trainer.train(num_epochs=config.get("epochs", 3), val_every_n_steps=config.get("val_every_n_steps", 100))
        return {"run_root": str(layout.root), "recipe": recipe}
    if recipe in ARGPARSE_RECIPES:
        module = importlib.import_module(ARGPARSE_RECIPES[recipe])
        argv = [ARGPARSE_RECIPES[recipe]] + _config_to_argv(config)
        old_argv = sys.argv
        try:
            sys.argv = argv
            module.main()
        finally:
            sys.argv = old_argv
        return {"run_root": str(layout.root), "recipe": recipe}
    raise ValueError(f"Unknown training recipe: {recipe}")
