from __future__ import annotations

from pathlib import Path
from typing import Any

from projected_token.config import load_yaml
from projected_token.training.experiments import run_experiment_matrix


def run_matrix(config_path: str | Path) -> list[Any]:
    cfg = load_yaml(config_path)
    config_paths = [str(path) for path in cfg.get("configs", [])]
    if not config_paths:
        raise ValueError("Matrix config must define non-empty 'configs' list.")
    summary_output = str(cfg.get("summary_output", "artifacts/results/matrix/summary.json"))
    return run_experiment_matrix(config_paths=config_paths, summary_output=summary_output)
