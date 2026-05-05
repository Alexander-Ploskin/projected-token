from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projected_token.config import load_yaml
from projected_token.io import write_json, write_csv
from projected_token.plotting import plot_metric_comparison
from projected_token.training.recipe_runner import run_training


@dataclass(frozen=True)
class ExperimentRunResult:
    config_path: str
    run_root: str
    recipe: str
    metrics_path: str | None
    metrics: dict[str, Any] | None


def _cleanup_after_run() -> None:
    """Best-effort cleanup to avoid CUDA OOM between matrix configs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        # Cleanup is opportunistic; never fail matrix progress because of it.
        pass


def _collect_primary_metric(metrics: dict[str, Any]) -> float:
    if isinstance(metrics, list) and metrics:
        last = metrics[-1]
        if isinstance(last, dict):
            if "mrr@10" in last:
                return float(last["mrr@10"])
            if "mrr" in last:
                return float(last["mrr"])
            if "cosine_sim" in last:
                return float(last["cosine_sim"])
        return 0.0
    for key in ["val.mrr@10", "val.mrr", "mrr@10", "mrr", "cosine_sim"]:
        cursor: Any = metrics
        ok = True
        for part in key.split("."):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                ok = False
                break
        if ok and isinstance(cursor, (int, float)):
            return float(cursor)
    return 0.0


def run_experiment_matrix(config_paths: list[str], summary_output: str) -> list[ExperimentRunResult]:
    results: list[ExperimentRunResult] = []
    summary_rows: list[dict[str, Any]] = []

    for cfg in config_paths:
        resolved = run_training(cfg)
        run_root = resolved["run_root"]
        recipe = resolved["recipe"]
        metrics_path = None
        metrics = None

        checkpoint_metrics = Path(run_root) / "checkpoints" / "metrics.json"
        if checkpoint_metrics.exists():
            metrics_path = str(checkpoint_metrics)
            metrics = load_yaml(metrics_path) if checkpoint_metrics.suffix in {".yaml", ".yml"} else None
            if metrics is None:
                import json

                metrics = json.loads(checkpoint_metrics.read_text(encoding="utf-8"))

        if metrics is None:
            metrics_dir = Path(run_root) / "metrics"
            candidates = sorted(metrics_dir.glob("*history.json"))
            if candidates:
                metrics_path = str(candidates[-1])
                import json

                metrics = json.loads(candidates[-1].read_text(encoding="utf-8"))
            else:
                metrics_json = metrics_dir / "metrics.json"
                if metrics_json.exists():
                    metrics_path = str(metrics_json)
                    import json

                    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))

        primary_metric = _collect_primary_metric(metrics if isinstance(metrics, dict) else {})
        summary_rows.append(
            {
                "config_path": cfg,
                "run_root": run_root,
                "recipe": recipe,
                "primary_metric": primary_metric,
                "metrics_path": metrics_path,
            }
        )
        results.append(
            ExperimentRunResult(
                config_path=cfg,
                run_root=run_root,
                recipe=recipe,
                metrics_path=metrics_path,
                metrics=metrics if isinstance(metrics, dict) else None,
            )
        )
        _cleanup_after_run()

    summary_target = Path(summary_output)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_target, {"runs": summary_rows})
    write_csv(summary_target.with_suffix(".csv"), summary_rows)
    labels = [Path(row["config_path"]).stem for row in summary_rows]
    values = [float(row["primary_metric"]) for row in summary_rows]
    if labels and values:
        plot_metric_comparison(
            labels,
            values,
            output_path=summary_target.with_name(summary_target.stem + ".png"),
            title="Experiment Matrix Primary Metric",
            y_label="metric",
        )
    return results
