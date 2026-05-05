from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from projected_token.config import load_yaml, validate_retrieval_config
from projected_token.io import load_records, write_json
from projected_token.artifacts import write_metrics_bundle
from projected_token.plotting import plot_metric_comparison
from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import create_faiss_index, l2_normalize, load_faiss_index, save_faiss_index
from projected_token.retrieval.metrics.ranking import aggregate_rankings
from projected_token.retrieval.tasks.popqa import build_popqa_cases


def _encode_batches(encoder: Any, texts: list[str], batch_size: int, *, questions: list[str] | None = None) -> np.ndarray:
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode"):
        batch_texts = texts[start:start + batch_size]
        batch_questions = questions[start:start + batch_size] if questions else None
        encoded = encoder.encode(batch_texts, batch_questions)
        if hasattr(encoded, "detach"):
            encoded = encoded.detach().cpu().numpy()
        chunks.append(np.asarray(encoded, dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def build_index(config_path: str | Path) -> dict[str, Any]:
    config = validate_retrieval_config(load_yaml(config_path)).model_dump()
    dataset_cfg = config.get("dataset", {})
    index_cfg = config.get("index", {})
    records = load_records(dataset_cfg["path"])
    text_col = dataset_cfg.get("text_col", "s_wiki_content")
    id_col = dataset_cfg.get("id_col", "id")
    valid_indices = [i for i, row in enumerate(records) if isinstance(row.get(text_col), str) and row.get(text_col)]
    texts = [records[i][text_col] for i in valid_indices]
    encoder = build_encoder(config["encoder"])
    embeddings = _encode_batches(encoder, texts, int(index_cfg.get("batch_size", 32)))
    if index_cfg.get("normalize", True):
        embeddings = l2_normalize(embeddings)
    output_dir = Path(index_cfg.get("output_dir", "artifacts/indexes/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    index = create_faiss_index(embeddings, index_cfg.get("metric", "ip"))
    save_faiss_index(index, output_dir / "index.faiss")
    metadata = {
        "schema_version": 1,
        "dataset_path": dataset_cfg["path"],
        "text_col": text_col,
        "id_col": id_col,
        "valid_indices": valid_indices,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "encoder": config["encoder"],
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def evaluate_retrieval(config_path: str | Path) -> dict[str, Any]:
    config = validate_retrieval_config(load_yaml(config_path)).model_dump()
    dataset_cfg = config.get("dataset", {})
    index_cfg = config.get("index", {})
    task_cfg = config.get("task", {})
    metric_cfg = config.get("metrics", {})
    records = load_records(dataset_cfg["path"])
    index_dir = Path(index_cfg.get("input_dir") or index_cfg.get("output_dir", "artifacts/indexes/default"))
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    index = load_faiss_index(index_dir / "index.faiss")
    task = task_cfg.get("name", "popqa")
    if task != "popqa":
        raise ValueError(f"Unified retrieval evaluator currently supports task='popqa'; use internal recipes for {task}")
    cases = build_popqa_cases(records, metadata["valid_indices"], task_cfg.get("question_col", "question"))
    encoder = build_encoder(config["encoder"])
    queries = [case["query"] for case in cases]
    top_k = [int(k) for k in metric_cfg.get("top_k", [1, 3, 5, 10, 20])]
    query_embeddings = _encode_batches(encoder, queries, int(index_cfg.get("batch_size", 32)))
    query_embeddings = l2_normalize(query_embeddings)
    _, indices = index.search(query_embeddings.astype(np.float32), max(top_k))
    ranking_cases = [(case["relevant_docs"], indices[i].tolist()) for i, case in enumerate(cases)]
    metrics = aggregate_rankings(ranking_cases, top_k)
    output_path = metric_cfg.get("output_path", "artifacts/results/retrieval_metrics.json")
    output_csv_path = metric_cfg.get(
        "output_csv_path",
        str(Path(output_path).with_suffix(".csv")),
    )
    run_id = metric_cfg.get("run_id", Path(output_path).parent.name)
    write_metrics_bundle(
        metrics,
        run_id=run_id,
        dataset=task,
        split="eval",
        json_path=output_path,
        csv_path=output_csv_path,
    )
    recall_labels = [f"R@{k}" for k in top_k if f"recall@{k}" in metrics]
    recall_values = [metrics[f"recall@{k}"] for k in top_k if f"recall@{k}" in metrics]
    if recall_labels and recall_values:
        plot_metric_comparison(
            recall_labels,
            recall_values,
            output_path=Path(output_path).with_name(Path(output_path).stem + "_recall.png"),
            title=f"{task.upper()} Recall@K",
            y_label="recall",
        )
    return metrics
