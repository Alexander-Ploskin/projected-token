from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from projected_token.io import write_csv, write_json
from projected_token.plotting import plot_metric_comparison


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_matrix_scores(path: Path, metric: str = "primary_metric") -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _load_json(path)
    rows = payload.get("runs", [])
    out = []
    for row in rows:
        out.append(
            {
                "config_path": row.get("config_path"),
                "run_root": row.get("run_root"),
                "recipe": row.get("recipe"),
                metric: float(row.get(metric, 0.0)),
            }
        )
    return out


def build_final_report(
    *,
    output_dir: str | Path,
    contrastive_summary: str | Path,
    distill_summary: str | Path,
    two_stage_summary: str | Path,
    lora_summary: str | Path,
    beir_summary: str | Path,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_groups = {
        "contrastive": _extract_matrix_scores(Path(contrastive_summary)),
        "distill": _extract_matrix_scores(Path(distill_summary)),
        "two_stage": _extract_matrix_scores(Path(two_stage_summary)),
        "lora_unfreeze": _extract_matrix_scores(Path(lora_summary)),
    }

    best_by_group = {}
    for group, rows in matrix_groups.items():
        if not rows:
            best_by_group[group] = None
            continue
        best_by_group[group] = max(rows, key=lambda item: item["primary_metric"])

    beir_payload = _load_json(Path(beir_summary)) if Path(beir_summary).exists() else {}
    beir_average = beir_payload.get("average", {})
    beir_per_dataset = beir_payload.get("per_dataset", {})

    final_summary = {
        "best_by_group": best_by_group,
        "beir_average": beir_average,
        "beir_per_dataset": beir_per_dataset,
        "notes": {
            "best_mode_msmarco": max(
                (item for item in best_by_group.values() if item),
                key=lambda item: item["primary_metric"],
                default=None,
            ),
            "best_mode_beir3": "Use beir_average.ndcg@10 and beir_average.mrr@10 for final selection.",
            "two_stage_gain_vs_one_stage": "Compare best two_stage primary_metric against best contrastive and distill.",
            "lora_partial_unfreeze_effect": "Compare lora_unfreeze group rows in final_summary.csv.",
        },
    }

    write_json(out_dir / "final_summary.json", final_summary)

    flat_rows = []
    for group, rows in matrix_groups.items():
        for row in rows:
            flat_rows.append({"group": group, **row})
    for dataset_name, metrics in beir_per_dataset.items():
        for metric_name, metric_value in metrics.items():
            flat_rows.append(
                {
                    "group": "beir3",
                    "config_path": dataset_name,
                    "run_root": beir_payload.get("run_id"),
                    "recipe": "beir",
                    "primary_metric": metric_value,
                    "metric_name": metric_name,
                }
            )
    write_csv(out_dir / "final_summary.csv", flat_rows)

    # Figure 1: training regime comparison
    labels = []
    values = []
    for group, best in best_by_group.items():
        if best:
            labels.append(group)
            values.append(float(best["primary_metric"]))
    if labels and values:
        plot_metric_comparison(
            labels,
            values,
            output_path=out_dir / "ablation_layers_and_training_regime.png",
            title="Best Primary Metric by Training Regime",
            y_label="primary_metric",
        )
        plot_metric_comparison(
            labels,
            values,
            output_path=out_dir / "training_curves.png",
            title="Training Regime Comparison",
            y_label="primary_metric",
        )

    # Figure 2: MS MARCO vs BEIR3
    msmarco_best = max(values) if values else 0.0
    beir_ndcg = float(beir_average.get("ndcg@10", 0.0))
    plot_metric_comparison(
        ["msmarco_best", "beir3_ndcg@10"],
        [msmarco_best, beir_ndcg],
        output_path=out_dir / "msmarco_vs_beir3.png",
        title="MS MARCO vs BEIR3",
        y_label="score",
    )

    return final_summary
