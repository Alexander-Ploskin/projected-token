#!/usr/bin/env python3
"""Aggregate BEIR3 summaries into reusable metric rows and comparison plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from projected_token.io import write_csv, write_json
from projected_token.plotting import plot_metric_comparison


def _metric_rows_from_summary(summary: dict[str, Any], *, source_path: Path) -> list[dict[str, Any]]:
    run_id = str(summary.get("run_id") or source_path.stem)
    split = str(summary.get("split", "test"))
    encoder_fp = str(summary.get("encoder_fingerprint", ""))
    rows: list[dict[str, Any]] = []
    per_dataset = summary.get("per_dataset", {})
    if isinstance(per_dataset, dict):
        for dataset, metrics in per_dataset.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "split": split,
                        "metric": metric,
                        "value": value,
                        "encoder_fp": encoder_fp,
                        "source_path": str(source_path),
                    }
                )
    average = summary.get("average", {})
    if isinstance(average, dict):
        for metric, value in average.items():
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": "beir3_average",
                    "split": split,
                    "metric": metric,
                    "value": value,
                    "encoder_fp": encoder_fp,
                    "source_path": str(source_path),
                }
            )
    return rows


def aggregate_beir3_results(summary_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []

    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(summary)
        rows.extend(_metric_rows_from_summary(summary, source_path=path))
        avg = summary.get("average", {})
        if isinstance(avg, dict) and "ndcg@10" in avg:
            comparison.append(
                {
                    "run_id": str(summary.get("run_id") or path.stem),
                    "ndcg@10": float(avg["ndcg@10"]),
                    "mrr@10": float(avg.get("mrr@10", 0.0)),
                    "source_path": str(path),
                }
            )

    write_json(output_dir / "beir3_metric_rows.json", rows)
    write_csv(output_dir / "beir3_metric_rows.csv", rows)
    write_json(output_dir / "beir3_comparison_summary.json", {"runs": comparison})
    write_csv(output_dir / "beir3_comparison_summary.csv", comparison)
    if comparison:
        plot_metric_comparison(
            labels=[row["run_id"] for row in comparison],
            values=[row["ndcg@10"] for row in comparison],
            output_path=output_dir / "beir3_ndcg10_comparison.png",
            title="BEIR3 Average NDCG@10 Comparison",
            y_label="ndcg@10",
        )
    return {"rows": rows, "runs": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate BEIR3 summary JSON files")
    parser.add_argument("--summary", action="append", required=True, type=Path, help="BEIR3 summary JSON; repeat for each run")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/retrieval/beir3_aggregate"))
    args = parser.parse_args()
    result = aggregate_beir3_results(args.summary, args.output_dir)
    print(json.dumps({"runs": result["runs"], "row_count": len(result["rows"])}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
