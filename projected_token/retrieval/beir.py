from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np

from projected_token.artifacts import write_metrics_bundle
from projected_token.io import write_json
from projected_token.plotting import plot_metric_comparison
from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import create_faiss_index, l2_normalize
from projected_token.retrieval.metrics.ranking import aggregate_rankings


def _encode_batches(encoder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = encoder.encode(batch_texts, None)
        if hasattr(encoded, "detach"):
            encoded = encoded.detach().cpu().numpy()
        chunks.append(np.asarray(encoded, dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with jsonlines.open(path, "r") as reader:
        return [row for row in reader]


def _load_beir_dataset(dataset_dir: Path, split: str = "test") -> tuple[list[str], list[str], list[str], dict[str, set[str]]]:
    corpus_rows = _load_jsonl(dataset_dir / "corpus.jsonl")
    query_rows = _load_jsonl(dataset_dir / "queries.jsonl")

    corpus_ids = [str(row["_id"]) for row in corpus_rows]
    corpus_texts = [
        " ".join(part for part in [str(row.get("title", "")).strip(), str(row.get("text", "")).strip()] if part).strip()
        for row in corpus_rows
    ]
    query_ids = [str(row["_id"]) for row in query_rows]
    query_texts = [str(row.get("text", "")) for row in query_rows]

    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    relevant: dict[str, set[str]] = {}
    with qrels_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(row.get("score", "1"))
            if score <= 0:
                continue
            relevant.setdefault(qid, set()).add(did)
    return corpus_ids, corpus_texts, query_ids, query_texts, relevant


def _encoder_fingerprint(encoder_cfg: dict[str, Any]) -> str:
    payload = json.dumps(encoder_cfg, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def evaluate_beir(config: dict[str, Any]) -> dict[str, Any]:
    encoder_cfg = config["encoder"]
    index_cfg = config.get("index", {})
    metric_cfg = config.get("metrics", {})
    datasets_cfg = config.get("datasets", [])
    split = str(config.get("split", "test"))
    top_k = [int(k) for k in metric_cfg.get("top_k", [1, 3, 5, 10, 20])]
    search_k = int(config.get("search_k", max(top_k)))
    batch_size = int(index_cfg.get("batch_size", 32))

    output_path = Path(metric_cfg.get("output_path", "artifacts/results/retrieval/beir3_summary.json"))
    output_csv_path = Path(metric_cfg.get("output_csv_path", output_path.with_suffix(".csv")))
    run_id = str(metric_cfg.get("run_id", output_path.parent.name))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = build_encoder(encoder_cfg)
    encoder_fp = _encoder_fingerprint(encoder_cfg)

    per_dataset: dict[str, dict[str, float]] = {}
    metric_table_rows: list[dict[str, Any]] = []

    for item in datasets_cfg:
        dataset_name = str(item["name"])
        dataset_dir = Path(item["path"])
        corpus_ids, corpus_texts, query_ids, query_texts, relevant_map = _load_beir_dataset(dataset_dir, split=split)

        embeddings = _encode_batches(encoder, corpus_texts, batch_size=batch_size)
        if index_cfg.get("normalize", True):
            embeddings = l2_normalize(embeddings)
        index = create_faiss_index(embeddings, index_cfg.get("metric", "ip"))

        query_embeddings = _encode_batches(encoder, query_texts, batch_size=batch_size)
        query_embeddings = l2_normalize(query_embeddings)
        _, indices = index.search(query_embeddings.astype(np.float32), search_k)

        ranking_cases: list[tuple[set[str], list[str]]] = []
        for idx, qid in enumerate(query_ids):
            relevant = relevant_map.get(qid, set())
            if not relevant:
                continue
            retrieved = [corpus_ids[doc_idx] for doc_idx in indices[idx].tolist() if 0 <= doc_idx < len(corpus_ids)]
            ranking_cases.append((relevant, retrieved))

        metrics = aggregate_rankings(ranking_cases, top_k)
        per_dataset[dataset_name] = metrics

        dataset_json = output_path.parent / f"{dataset_name}_metrics.json"
        dataset_csv = output_path.parent / f"{dataset_name}_metrics.csv"
        write_metrics_bundle(
            metrics,
            run_id=run_id,
            dataset=dataset_name,
            split=split,
            json_path=dataset_json,
            csv_path=dataset_csv,
        )

        for metric_name, metric_value in metrics.items():
            metric_table_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "split": split,
                    "metric": metric_name,
                    "value": metric_value,
                    "encoder_fp": encoder_fp,
                }
            )

    if per_dataset:
        avg_metrics: dict[str, float] = {}
        for name in next(iter(per_dataset.values())).keys():
            avg_metrics[name] = float(np.mean([metrics[name] for metrics in per_dataset.values()]))
    else:
        avg_metrics = {}

    summary = {
        "run_id": run_id,
        "split": split,
        "datasets": [str(item["name"]) for item in datasets_cfg],
        "top_k": top_k,
        "search_k": search_k,
        "encoder_fingerprint": encoder_fp,
        "per_dataset": per_dataset,
        "average": avg_metrics,
    }
    write_json(output_path, summary)
    write_metrics_bundle(
        avg_metrics,
        run_id=run_id,
        dataset="beir3_average",
        split=split,
        json_path=output_path.parent / "beir3_average_metrics.json",
        csv_path=output_csv_path,
    )
    write_json(output_path.parent / "beir3_metric_rows.json", metric_table_rows)

    if per_dataset and "ndcg@10" in avg_metrics:
        labels = list(per_dataset.keys())
        values = [per_dataset[name].get("ndcg@10", 0.0) for name in labels]
        plot_metric_comparison(
            labels,
            values,
            output_path=output_path.parent / "beir3_comparison.png",
            title="BEIR NDCG@10 by Dataset",
            y_label="ndcg@10",
        )

    return summary
