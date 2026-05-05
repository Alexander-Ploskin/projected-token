from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

import yaml

from projected_token.io import ensure_parent, write_csv, write_json


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return value or "run"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunLayout:
    root: Path
    checkpoints_dir: Path
    logs_dir: Path
    metrics_dir: Path
    plots_dir: Path


def create_run_layout(
    experiment_name: str,
    *,
    base_dir: str | Path = "artifacts/runs",
    run_id: str | None = None,
) -> RunLayout:
    resolved_id = run_id or f"{utc_timestamp()}_{_slugify(experiment_name)}"
    root = Path(base_dir) / resolved_id
    checkpoints_dir = root / "checkpoints"
    logs_dir = root / "logs" / "tensorboard"
    metrics_dir = root / "metrics"
    plots_dir = root / "plots"
    for path in [checkpoints_dir, logs_dir, metrics_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        root=root,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
    )


def write_config_lock(config: dict[str, Any], output_path: str | Path) -> None:
    target = ensure_parent(output_path)
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _metric_parts(metric_name: str) -> tuple[str, int | None]:
    if "@" not in metric_name:
        return metric_name, None
    name, raw_k = metric_name.split("@", 1)
    try:
        return name, int(raw_k)
    except ValueError:
        return metric_name, None


def metrics_to_rows(
    metrics: dict[str, Any],
    *,
    run_id: str,
    dataset: str,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, value in metrics.items():
        if isinstance(value, dict):
            # Nested sections are flattened recursively.
            for nested_name, nested_value in value.items():
                full_name = f"{metric_name}.{nested_name}"
                base_name, k_value = _metric_parts(full_name)
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "split": split,
                        "metric": base_name,
                        "k": k_value,
                        "value": nested_value,
                    }
                )
            continue
        base_name, k_value = _metric_parts(metric_name)
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "split": split,
                "metric": base_name,
                "k": k_value,
                "value": value,
            }
        )
    return rows


def write_metrics_bundle(
    metrics: dict[str, Any],
    *,
    run_id: str,
    dataset: str,
    split: str,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    write_json(json_path, metrics)
    write_csv(
        csv_path,
        metrics_to_rows(metrics, run_id=run_id, dataset=dataset, split=split),
    )
