from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from projected_token.config import instantiate, load_yaml
from projected_token.io import load_jsonl, write_json, write_jsonl
from projected_token.metrics import GPTScoreMetric, QAScoreMetric, compute_in_accuracy
from projected_token.metrics.simple import aggregate_simple_metrics
from projected_token.logging_utils import log_step


def _reference_answers(item: dict[str, Any], answer_col: str) -> list[str]:
    value = item.get(answer_col)
    if value is None:
        value = item.get("possible_answers") or item.get("answers") or item.get("obj") or ""
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def evaluate_paraphrase(
    *,
    input_path: str | Path,
    output_path: str | Path,
    original_col: str = "s_wiki_content",
    candidate_col: str = "paraphrase",
    judge_config: dict[str, Any] | None = None,
    metric_configs: list[dict[str, Any]] | None = None,
    batch_size: int = 10,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    originals = [str(row.get(original_col, "")) for row in rows]
    candidates = [str(row.get(candidate_col, "")) for row in rows]
    summary: dict[str, Any] = {"count": len(rows), "simple": aggregate_simple_metrics(originals, candidates)}

    metrics = []
    if judge_config:
        metrics.append(GPTScoreMetric(judge_config))
    for cfg in metric_configs or []:
        metrics.append(instantiate(cfg, kind="metric"))

    if metrics:
        log_step(f"Evaluating paraphrases with {len(metrics)} metric(s)")
        for metric in metrics:
            metric_name = metric.__class__.__name__
            if isinstance(metric, GPTScoreMetric) and batch_size > 1:
                for start in tqdm(range(0, len(rows), batch_size), desc=metric_name):
                    verdicts = metric.judge_batch(candidates[start:start + batch_size], originals[start:start + batch_size])
                    for row, verdict in zip(rows[start:start + batch_size], verdicts):
                        row.setdefault("metrics", {})[metric_name] = verdict
            else:
                for row in tqdm(rows, desc=metric_name):
                    row.setdefault("metrics", {})[metric_name] = metric(str(row.get(original_col, "")), str(row.get(candidate_col, "")))
        write_jsonl(Path(output_path).with_suffix(".jsonl"), rows)
    write_json(output_path, summary)
    return summary


def evaluate_qa(
    *,
    input_path: str | Path,
    output_path: str | Path,
    judge_config: dict[str, Any] | None = None,
    question_col: str = "question",
    answer_col: str = "obj",
    prediction_col: str = "answer",
    batch_size: int = 10,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    questions = [str(row.get(question_col, "")) for row in rows]
    references = [_reference_answers(row, answer_col) for row in rows]
    predictions = [str(row.get(prediction_col, "")) for row in rows]
    in_acc = [compute_in_accuracy(pred, refs) for pred, refs in zip(predictions, references)]
    summary: dict[str, Any] = {"count": len(rows), "in_accuracy": float(mean(in_acc)) if in_acc else 0.0}

    if judge_config:
        judge = QAScoreMetric(judge_config)
        verdicts: list[dict[str, Any]] = []
        log_step("Evaluating QA answers with LLM judge")
        for start in tqdm(range(0, len(rows), batch_size), desc="qa-judge"):
            verdicts.extend(judge.judge_batch(questions[start:start + batch_size], references[start:start + batch_size], predictions[start:start + batch_size]))
        for row, verdict in zip(rows, verdicts):
            row.setdefault("metrics", {})["QAScoreMetric"] = verdict
        valid_scores = [v["qa_score"] for v in verdicts if v.get("qa_score", -1) >= 0]
        summary["qa_score"] = float(mean(valid_scores)) if valid_scores else 0.0
        summary["label_counts"] = {label: sum(1 for v in verdicts if v.get("label") == label) for label in sorted({v.get("label") for v in verdicts})}
        write_jsonl(Path(output_path).with_suffix(".jsonl"), rows)
    write_json(output_path, summary)
    return summary
