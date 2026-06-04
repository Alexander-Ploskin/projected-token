#!/usr/bin/env python3
"""Score KILT OpenQA top-k JSONL (PopQA / HotpotQA) with in-accuracy relevance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.retrieval.metrics.kilt_openqa import (
    compute_kilt_openqa_metrics,
    format_metrics_by_k,
    hotpot_targets_from_row,
    popqa_targets_from_row,
    relevance_labels_for_hits,
    scored_targets,
)

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "popqa": {
        "queries_path": Path("/data/popqa_enriched.parquet"),
        "results_jsonl": Path("artifacts/results/retrieval/kilt_e11_popqa_top100.jsonl"),
        "output_path": Path("artifacts/results/retrieval/kilt_e11_popqa_metrics.json"),
        "id_col": "id",
        "question_col": "question",
        "answer_col": None,
        "targets_fn": popqa_targets_from_row,
    },
    "hotpotqa": {
        "queries_path": Path("/data/hotpotqa/distractor/validation.parquet"),
        "results_jsonl": Path(
            "artifacts/results/retrieval/kilt_e11_hotpotqa_distractor_top100.jsonl"
        ),
        "output_path": Path(
            "artifacts/results/retrieval/kilt_e11_hotpotqa_distractor_metrics.json"
        ),
        "id_col": "id",
        "question_col": "question",
        "answer_col": "answer",
        "targets_fn": hotpot_targets_from_row,
    },
}

FALLBACK_QUERIES_PATHS: dict[str, list[Path]] = {
    "popqa": [
        REPO_ROOT / "data" / "eval" / "popqa" / "popqa_enriched.parquet",
        Path("/home/a-ploskin/repos/pt/projected-token/popqa_enriched.parquet"),
    ],
    "hotpotqa": [
        REPO_ROOT / "data" / "hotpotqa" / "distractor" / "validation.parquet",
    ],
}


def parse_top_k(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--top-k must include at least one integer.")
    if min(values) <= 0:
        raise ValueError("--top-k values must be positive.")
    return sorted(set(values))


def resolve_queries_path(dataset: str, queries_path: Path) -> Path:
    if queries_path.exists():
        return queries_path
    for candidate in FALLBACK_QUERIES_PATHS.get(dataset, []):
        if candidate.exists():
            return candidate
    return queries_path


def build_targets_fn(
    dataset: str,
    answer_col: str | None,
) -> Callable[[dict[str, Any]], list[str]]:
    if dataset == "popqa":
        return popqa_targets_from_row
    if dataset == "hotpotqa":
        col = answer_col or "answer"

        def _fn(row: dict[str, Any]) -> list[str]:
            return hotpot_targets_from_row(row, answer_col=col)

        return _fn
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score KILT OpenQA retrieval JSONL")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["popqa", "hotpotqa"],
        default="popqa",
    )
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument("--queries-path", type=Path, default=None)
    parser.add_argument("--id-col", type=str, default=None)
    parser.add_argument("--question-col", type=str, default=None)
    parser.add_argument("--answer-col", type=str, default=None)
    parser.add_argument("--passage-field", type=str, default="document")
    parser.add_argument("--top-k", type=str, default="1,5,10,20,50,100")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--save-details-path",
        type=Path,
        default=None,
        help="Optional per-query relevance flags JSON",
    )
    parser.add_argument(
        "--no-verify-ids",
        action="store_true",
        help="Skip query_id / question alignment checks against parquet",
    )
    args = parser.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    results_jsonl = args.results_jsonl or defaults["results_jsonl"]
    queries_path = resolve_queries_path(
        args.dataset,
        args.queries_path or defaults["queries_path"],
    )
    output_path = args.output_path or defaults["output_path"]
    id_col = args.id_col or defaults["id_col"]
    question_col = args.question_col or defaults["question_col"]
    answer_col = args.answer_col if args.answer_col is not None else defaults["answer_col"]
    targets_fn = build_targets_fn(args.dataset, answer_col)

    top_k_values = parse_top_k(args.top_k)

    if not queries_path.exists():
        raise FileNotFoundError(f"Queries parquet not found: {queries_path}")
    if not results_jsonl.exists():
        raise FileNotFoundError(f"Results JSONL not found: {results_jsonl}")

    df = pd.read_parquet(queries_path)
    gold_rows = [targets_fn(row) for row in df.to_dict(orient="records")]
    num_parquet = len(gold_rows)

    relevance_flags_scored: list[list[int]] = []
    details: list[dict] = []
    n_queries_scored = 0
    line_idx = 0

    with results_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if line_idx >= num_parquet:
                raise ValueError(
                    f"JSONL has more rows than parquet ({line_idx + 1} > {num_parquet})"
                )

            if not args.no_verify_ids:
                expected_id = str(df.iloc[line_idx][id_col])
                if str(record.get("query_id", "")) != expected_id:
                    raise ValueError(
                        f"query_id mismatch at line {line_idx}: "
                        f"jsonl={record.get('query_id')!r} parquet={expected_id!r}"
                    )
                expected_q = str(df.iloc[line_idx][question_col])
                if str(record.get("question", "")) != expected_q:
                    raise ValueError(
                        f"question mismatch at line {line_idx}: "
                        f"jsonl={record.get('question')!r} parquet={expected_q!r}"
                    )

            targets = gold_rows[line_idx]
            hits = record.get("top_k", [])
            labels = relevance_labels_for_hits(
                hits,
                targets,
                passage_field=args.passage_field,
            )
            answers = scored_targets(targets)
            if answers:
                relevance_flags_scored.append(labels)
                n_queries_scored += 1

            if args.save_details_path is not None:
                details.append(
                    {
                        "query_index": line_idx,
                        "query_id": record.get("query_id"),
                        "question": record.get("question"),
                        "answers": answers,
                        "retrieved_doc_ids": [h.get("doc_id", "") for h in hits],
                        "relevance": labels,
                        "is_scored_query": bool(answers),
                    }
                )

            line_idx += 1

    if line_idx != num_parquet:
        raise ValueError(
            f"Row count mismatch: JSONL={line_idx} parquet={num_parquet}"
        )

    metrics = compute_kilt_openqa_metrics(relevance_flags_scored, top_k_values)
    metrics_by_k = format_metrics_by_k(metrics, top_k_values)
    payload = {
        "dataset": args.dataset,
        "num_queries": line_idx,
        "n_queries_scored": n_queries_scored,
        "results_jsonl": str(results_jsonl),
        "queries_path": str(queries_path),
        "passage_field": args.passage_field,
        "top_k": top_k_values,
        "metrics": metrics,
        "metrics_by_k": metrics_by_k,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
        out.write("\n")

    if args.save_details_path is not None:
        args.save_details_path.parent.mkdir(parents=True, exist_ok=True)
        with args.save_details_path.open("w", encoding="utf-8") as out:
            json.dump(details, out, ensure_ascii=False, indent=2)
            out.write("\n")

    print(f"Scored {line_idx} queries ({n_queries_scored} with gold answers)")
    print(f"Wrote metrics to {output_path}")
    print(f"{'k':>6}  {'Recall':>10}  {'NDCG':>10}")
    for row in metrics_by_k:
        print(f"{row['k']:>6}  {row['recall']:10.6f}  {row['ndcg']:10.6f}")
    print(f"{'MRR':>6}  {metrics['mrr']:10.6f}")


if __name__ == "__main__":
    main()
