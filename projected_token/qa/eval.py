from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from projected_token.io import write_json, write_jsonl
from projected_token.metrics.qa_text import (
    contains_match,
    exact_match,
    in_accuracy_match,
    max_over_refs,
    token_f1,
)


def _norm_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _answers_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def load_predictions_jsonl(path: str | Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    predictions: dict[str, str] = {}
    inline_gold: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in tqdm(fp, desc="qa-load-pred", unit="line", dynamic_ncols=True):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid", row.get("query_index"))
            if qid is None:
                continue
            qid_str = str(qid)
            predictions[qid_str] = str(row.get("prediction", row.get("answer", "")))
            answers = _answers_from_value(row.get("answers", row.get("targets")))
            if answers:
                inline_gold[qid_str] = answers
    return predictions, inline_gold


def load_gold_jsonl(path: str | Path) -> dict[str, list[str]]:
    gold: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in tqdm(fp, desc="qa-load-gold", unit="line", dynamic_ncols=True):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid", row.get("query_index"))
            if qid is None:
                continue
            gold[str(qid)] = _answers_from_value(row.get("answers", row.get("targets")))
    return gold


def _popqa_answers(row: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    obj = _norm_text(row.get("obj"))
    if obj:
        answers.append(obj)
    possible_answers = row.get("possible_answers")
    if isinstance(possible_answers, str):
        parts = possible_answers.split("|")
    elif isinstance(possible_answers, (list, tuple, set)):
        parts = list(possible_answers)
    else:
        parts = []
    for part in parts:
        answer = _norm_text(part)
        if answer and answer not in answers:
            answers.append(answer)
    return answers


def load_gold_hf(
    *,
    dataset_key: str,
    split: str | None = None,
    max_rows: int | None = None,
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
) -> dict[str, list[str]]:
    from datasets import load_dataset

    if dataset_key == "popqa":
        dataset = load_dataset(
            "akariasai/PopQA",
            split=split or "test",
            cache_dir=hf_cache_dir,
            token=hf_token,
        )
        answer_fn = _popqa_answers
    elif dataset_key in {"hotpotqa_distractor", "hotpotqa"}:
        dataset = load_dataset(
            "hotpot_qa",
            "distractor",
            split=split or "validation",
            cache_dir=hf_cache_dir,
            token=hf_token,
        )
        answer_fn = lambda row: _answers_from_value(row.get("answer"))
    elif dataset_key == "hotpotqa_fullwiki":
        dataset = load_dataset(
            "hotpot_qa",
            "fullwiki",
            split=split or "validation",
            cache_dir=hf_cache_dir,
            token=hf_token,
        )
        answer_fn = lambda row: _answers_from_value(row.get("answer"))
    else:
        raise ValueError(f"Unsupported QA gold dataset: {dataset_key}")

    gold: dict[str, list[str]] = {}
    for idx, row_raw in enumerate(tqdm(dataset, desc=f"qa-load-gold-hf:{dataset_key}", unit="row", dynamic_ncols=True)):
        if max_rows is not None and idx >= max_rows:
            break
        row = dict(row_raw)
        gold[str(idx)] = answer_fn(row)
    return gold


def load_generation_meta(pred_path: str | Path) -> dict[str, Any] | None:
    path = Path(pred_path)
    meta_path = path.with_name(f"{path.stem}_meta.json")
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    try:
        with path.open("r", encoding="utf-8") as fp:
            for _ in range(500):
                line = fp.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model_id = row.get("model_id")
                method = row.get("method")
                if model_id is not None or method is not None:
                    return {"source": "pred_jsonl_row", "model_id": model_id, "method": method}
    except OSError:
        pass
    return None


def run_qa_eval(
    predictions: dict[str, str],
    gold: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_qids = set(predictions)
    gold_qids = set(gold)
    matched = pred_qids & gold_qids

    ems: list[float] = []
    f1s: list[float] = []
    answer_in_prediction_scores: list[float] = []
    in_accuracy_scores: list[float] = []
    errors: list[dict[str, Any]] = []

    for qid in tqdm(sorted(matched), desc="qa-score", unit="q", dynamic_ncols=True):
        prediction = predictions[qid]
        refs = [ref for ref in gold[qid] if ref]
        if not refs:
            continue
        em = max_over_refs(prediction, refs, exact_match)
        f1 = max_over_refs(prediction, refs, token_f1)
        answer_in_prediction = max_over_refs(prediction, refs, contains_match)
        in_accuracy = max_over_refs(prediction, refs, in_accuracy_match)

        ems.append(em)
        f1s.append(f1)
        answer_in_prediction_scores.append(answer_in_prediction)
        in_accuracy_scores.append(in_accuracy)
        if em < 1.0:
            errors.append(
                {
                    "qid": qid,
                    "prediction": prediction,
                    "references": refs,
                    "em": em,
                    "f1": f1,
                    "answer_in_prediction": answer_in_prediction,
                    "in_accuracy": in_accuracy,
                }
            )

    n_scored = len(ems)
    return (
        {
            "n_pred": len(pred_qids),
            "n_gold": len(gold_qids),
            "n_matched": len(matched),
            "n_missing_gold": len(pred_qids - gold_qids),
            "n_missing_pred": len(gold_qids - pred_qids),
            "n_scored": n_scored,
            "mean_em": sum(ems) / n_scored if n_scored else 0.0,
            "mean_f1": sum(f1s) / n_scored if n_scored else 0.0,
            "mean_answer_in_prediction": (
                sum(answer_in_prediction_scores) / n_scored if n_scored else 0.0
            ),
            "mean_in_accuracy": sum(in_accuracy_scores) / n_scored if n_scored else 0.0,
        },
        errors,
    )


def evaluate_predictions_jsonl(
    *,
    pred_path: str | Path,
    output_path: str | Path,
    gold_jsonl: str | Path | None = None,
    gold_hf: str | None = None,
    hf_split: str | None = None,
    max_rows: int | None = None,
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
    write_errors_path: str | Path | None = None,
) -> dict[str, Any]:
    predictions, inline_gold = load_predictions_jsonl(pred_path)
    if gold_jsonl is not None and gold_hf is not None:
        raise ValueError("Pass at most one of gold_jsonl or gold_hf.")
    if gold_jsonl is None and gold_hf is None:
        gold = inline_gold
        gold_source: dict[str, Any] = {"type": "inline", "path": str(pred_path)}
    elif gold_jsonl is not None:
        gold = load_gold_jsonl(gold_jsonl)
        gold_source = {"type": "jsonl", "path": str(Path(gold_jsonl).resolve())}
    else:
        assert gold_hf is not None
        gold = load_gold_hf(
            dataset_key=gold_hf,
            split=hf_split,
            max_rows=max_rows,
            hf_cache_dir=hf_cache_dir,
            hf_token=hf_token,
        )
        gold_source = {
            "type": "hf",
            "dataset_key": gold_hf,
            "split": hf_split,
            "max_rows": max_rows,
        }

    summary, errors = run_qa_eval(predictions, gold)
    payload: dict[str, Any] = {
        "gold_source": gold_source,
        "pred_path": str(Path(pred_path).resolve()),
        **summary,
    }
    generation_meta = load_generation_meta(pred_path)
    if generation_meta is not None:
        payload["generation_meta"] = generation_meta

    write_json(output_path, payload)
    if write_errors_path:
        write_jsonl(write_errors_path, errors)
    return payload
