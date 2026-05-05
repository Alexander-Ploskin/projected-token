from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projected_token.artifacts import write_metrics_bundle
from projected_token.io import load_records, write_json
from projected_token.plotting import plot_metric_comparison
from projected_token.retrieval.beir import _load_beir_dataset
from projected_token.retrieval.metrics.ranking import aggregate_rankings
from projected_token.retrieval.tasks.popqa import build_popqa_cases


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SimpleBM25:
    tokenized_docs: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.num_docs = len(self.tokenized_docs)
        self.doc_len = np.asarray([len(doc) for doc in self.tokenized_docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.num_docs else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        doc_freq: dict[str, int] = {}

        for doc_id, tokens in enumerate(self.tokenized_docs):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, tf in counts.items():
                self.postings.setdefault(token, []).append((doc_id, tf))
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.idf = {
            token: float(np.log(1.0 + ((self.num_docs - df + 0.5) / (df + 0.5))))
            for token, df in doc_freq.items()
        }

    @classmethod
    def from_texts(cls, texts: list[str], k1: float = 1.5, b: float = 0.75) -> "SimpleBM25":
        return cls(tokenized_docs=[_tokenize(text) for text in texts], k1=k1, b=b)

    def search(self, query: str, top_k: int) -> list[int]:
        if self.num_docs == 0:
            return []
        scores = np.zeros(self.num_docs, dtype=np.float32)
        query_terms = set(_tokenize(query))
        for term in query_terms:
            posting_list = self.postings.get(term)
            if not posting_list:
                continue
            idf = self.idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for doc_id, tf in posting_list:
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_id] / self.avgdl))
                if denom > 0:
                    scores[doc_id] += idf * ((tf * (self.k1 + 1.0)) / denom)

        top_k = max(1, min(top_k, self.num_docs))
        if top_k == self.num_docs:
            ranked = np.argsort(-scores)
        else:
            candidate_ids = np.argpartition(scores, -top_k)[-top_k:]
            ranked = candidate_ids[np.argsort(-scores[candidate_ids])]
        return ranked.tolist()


def evaluate_popqa_bm25(
    *,
    dataset_path: str | Path,
    text_col: str = "s_wiki_content",
    question_col: str = "question",
    top_k: list[int] | None = None,
    output_path: str | Path = "artifacts/results/retrieval/popqa_bm25_metrics.json",
    output_csv_path: str | Path | None = None,
    run_id: str = "bm25_popqa",
) -> dict[str, float]:
    top_k = top_k or [1, 3, 5, 10, 20]
    records = load_records(dataset_path)
    valid_indices = [i for i, row in enumerate(records) if isinstance(row.get(text_col), str) and row.get(text_col)]
    texts = [records[i][text_col] for i in valid_indices]

    bm25 = SimpleBM25.from_texts(texts)
    cases = build_popqa_cases(records, valid_indices, question_col=question_col)
    max_k = max(top_k)
    ranking_cases = [(case["relevant_docs"], bm25.search(case["query"], max_k)) for case in cases]
    metrics = aggregate_rankings(ranking_cases, top_k)

    csv_path = output_csv_path or str(Path(output_path).with_suffix(".csv"))
    write_metrics_bundle(
        metrics,
        run_id=run_id,
        dataset="popqa_bm25",
        split="eval",
        json_path=output_path,
        csv_path=csv_path,
    )

    recall_labels = [f"R@{k}" for k in top_k if f"recall@{k}" in metrics]
    recall_values = [metrics[f"recall@{k}"] for k in top_k if f"recall@{k}" in metrics]
    if recall_labels and recall_values:
        plot_metric_comparison(
            recall_labels,
            recall_values,
            output_path=Path(output_path).with_name(Path(output_path).stem + "_recall.png"),
            title="POPQA BM25 Recall@K",
            y_label="recall",
        )
    return metrics


def evaluate_beir3_bm25(
    *,
    datasets: list[dict[str, str]],
    split: str = "test",
    top_k: list[int] | None = None,
    search_k: int = 100,
    output_path: str | Path = "artifacts/results/retrieval/beir3_bm25_summary.json",
    output_csv_path: str | Path | None = None,
    run_id: str = "bm25_beir3",
) -> dict[str, Any]:
    top_k = top_k or [1, 3, 5, 10, 20]
    output_path = Path(output_path)
    output_csv_path = Path(output_csv_path or output_path.with_suffix(".csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_dataset: dict[str, dict[str, float]] = {}
    for item in datasets:
        dataset_name = str(item["name"])
        dataset_dir = Path(item["path"])
        corpus_ids, corpus_texts, query_ids, query_texts, relevant_map = _load_beir_dataset(dataset_dir, split=split)

        bm25 = SimpleBM25.from_texts(corpus_texts)
        ranking_cases: list[tuple[set[str], list[str]]] = []
        for idx, qid in enumerate(query_ids):
            relevant = relevant_map.get(qid, set())
            if not relevant:
                continue
            doc_indices = bm25.search(query_texts[idx], top_k=min(search_k, len(corpus_ids)))
            retrieved = [corpus_ids[i] for i in doc_indices if 0 <= i < len(corpus_ids)]
            ranking_cases.append((relevant, retrieved))

        metrics = aggregate_rankings(ranking_cases, top_k)
        per_dataset[dataset_name] = metrics
        write_metrics_bundle(
            metrics,
            run_id=run_id,
            dataset=f"{dataset_name}_bm25",
            split=split,
            json_path=output_path.parent / f"{dataset_name}_bm25_metrics.json",
            csv_path=output_path.parent / f"{dataset_name}_bm25_metrics.csv",
        )

    if per_dataset:
        avg_metrics = {
            key: float(np.mean([dataset_metrics[key] for dataset_metrics in per_dataset.values()]))
            for key in next(iter(per_dataset.values())).keys()
        }
    else:
        avg_metrics = {}

    summary = {
        "run_id": run_id,
        "method": "bm25",
        "split": split,
        "datasets": [str(item["name"]) for item in datasets],
        "top_k": top_k,
        "search_k": search_k,
        "per_dataset": per_dataset,
        "average": avg_metrics,
    }
    write_json(output_path, summary)
    write_metrics_bundle(
        avg_metrics,
        run_id=run_id,
        dataset="beir3_bm25_average",
        split=split,
        json_path=output_path.parent / "beir3_bm25_average_metrics.json",
        csv_path=output_csv_path,
    )

    if per_dataset and "ndcg@10" in avg_metrics:
        labels = list(per_dataset.keys())
        values = [per_dataset[name].get("ndcg@10", 0.0) for name in labels]
        plot_metric_comparison(
            labels,
            values,
            output_path=output_path.parent / "beir3_bm25_comparison.png",
            title="BEIR-3 BM25 NDCG@10",
            y_label="ndcg@10",
        )
    return summary
