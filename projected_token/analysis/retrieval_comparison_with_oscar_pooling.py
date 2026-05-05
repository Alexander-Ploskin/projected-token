from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from projected_token.artifacts import write_metrics_bundle
from projected_token.io import load_records
from projected_token.plotting import plot_metric_comparison
from projected_token.retrieval.beir import evaluate_beir
from projected_token.retrieval.bm25_baseline import evaluate_beir3_bm25, evaluate_popqa_bm25
from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import create_faiss_index, l2_normalize
from projected_token.retrieval.metrics.ranking import aggregate_rankings
from projected_token.retrieval.tasks.popqa import build_popqa_cases


def _encode_batches(encoder: Any, texts: list[str], batch_size: int, questions: list[str] | None = None) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_questions = questions[start:start + batch_size] if questions else None
        encoded = encoder.encode(batch_texts, batch_questions)
        if hasattr(encoded, "detach"):
            encoded = encoded.detach().cpu().numpy()
        chunks.append(np.asarray(encoded, dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def evaluate_popqa_vector_method(
    *,
    method_name: str,
    encoder_cfg: dict[str, Any],
    dataset_path: str,
    top_k: list[int],
    output_dir: Path,
    batch_size: int,
) -> dict[str, float]:
    records = load_records(dataset_path)
    valid_indices = [i for i, row in enumerate(records) if isinstance(row.get("s_wiki_content"), str) and row.get("s_wiki_content")]
    texts = [records[i]["s_wiki_content"] for i in valid_indices]
    cases = build_popqa_cases(records, valid_indices, question_col="question")
    queries = [case["query"] for case in cases]

    encoder = build_encoder(encoder_cfg)
    doc_embeddings = l2_normalize(_encode_batches(encoder, texts, batch_size=batch_size))
    query_embeddings = l2_normalize(_encode_batches(encoder, queries, batch_size=batch_size))
    index = create_faiss_index(doc_embeddings, "ip")
    _, indices = index.search(query_embeddings.astype(np.float32), max(top_k))
    ranking_cases = [(case["relevant_docs"], indices[i].tolist()) for i, case in enumerate(cases)]
    metrics = aggregate_rankings(ranking_cases, top_k)

    write_metrics_bundle(
        metrics,
        run_id=f"popqa_{method_name}",
        dataset="popqa_enriched",
        split="eval",
        json_path=output_dir / f"{method_name}_popqa_metrics.json",
        csv_path=output_dir / f"{method_name}_popqa_metrics.csv",
    )
    return metrics


def summarize_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows}
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval comparisons with OSCAR pooling baselines.")
    parser.add_argument("--output-dir", default="artifacts/results/retrieval/comparison_with_oscar_pooling")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k = [1, 3, 5, 10, 20]

    oscar_model = "/data/huggingface/naver/oscar-qwen2-7B"
    sfr_model = "/data/huggingface/Salesforce/SFR-Embedding-Mistral"
    popqa_path = "/data/popqa_enriched.parquet"
    beir_datasets = [
        {"name": "scifact", "path": "/data/huggingface/BeIR/scifact"},
        {"name": "nfcorpus", "path": "/data/huggingface/BeIR/nfcorpus"},
        {"name": "fiqa-2018", "path": "/data/huggingface/BeIR/fiqa"},
    ]

    methods: dict[str, dict[str, Any]] = {
        "sfr_baseline": {"name": "salesforce", "kwargs": {"model_name_or_path": sfr_model, "device": args.device}},
        "oscar_baseline_first": {"name": "oscar", "kwargs": {"model_name_or_path": oscar_model, "device": args.device, "aggregation": "first"}},
        "oscar_baseline_last": {"name": "oscar", "kwargs": {"model_name_or_path": oscar_model, "device": args.device, "aggregation": "last"}},
        "oscar_baseline_flatten": {"name": "oscar", "kwargs": {"model_name_or_path": oscar_model, "device": args.device, "aggregation": "flatten"}},
        "projector_contrastive_1layer": {
            "name": "oscar_projector",
            "kwargs": {"oscar_model_name": oscar_model, "projector_path": "artifacts/runs/20260428_142514_mlp-projector-mlp-1layer-msmarco/checkpoints/best_model.pt", "device": args.device},
        },
        "projector_contrastive_2layer": {
            "name": "oscar_projector",
            "kwargs": {"oscar_model_name": oscar_model, "projector_path": "artifacts/runs/20260428_155309_mlp-projector-mlp-2layer-msmarco/checkpoints/best_model.pt", "device": args.device},
        },
        "projector_contrastive_3layer": {
            "name": "oscar_projector",
            "kwargs": {"oscar_model_name": oscar_model, "projector_path": "artifacts/runs/20260428_173204_mlp-projector-mlp-3layer-msmarco/checkpoints/best_model.pt", "device": args.device},
        },
        "projector_two_stage_1layer": {
            "name": "oscar_projector",
            "kwargs": {"oscar_model_name": oscar_model, "projector_path": "artifacts/runs/20260428_175038_mlp-two-stage-distill-to-contrastive-1layer/checkpoints/best_model.pt", "device": args.device},
        },
    }

    popqa_rows: list[dict[str, Any]] = []
    beir_rows: list[dict[str, Any]] = []

    bm25_popqa = evaluate_popqa_bm25(
        dataset_path=popqa_path,
        text_col="s_wiki_content",
        question_col="question",
        top_k=top_k,
        output_path=output_dir / "bm25_popqa_metrics.json",
        output_csv_path=output_dir / "bm25_popqa_metrics.csv",
        run_id="bm25_popqa",
    )
    popqa_rows.append({"method": "bm25_baseline", **bm25_popqa})

    bm25_beir = evaluate_beir3_bm25(
        datasets=beir_datasets,
        split="test",
        top_k=top_k,
        search_k=100,
        output_path=output_dir / "bm25_beir3_summary.json",
        output_csv_path=output_dir / "bm25_beir3_summary.csv",
        run_id="bm25_beir3",
    )
    beir_rows.append({"method": "bm25_baseline", **bm25_beir.get("average", {})})

    for method_name, encoder_cfg in methods.items():
        try:
            popqa_metrics = evaluate_popqa_vector_method(
                method_name=method_name,
                encoder_cfg=encoder_cfg,
                dataset_path=popqa_path,
                top_k=top_k,
                output_dir=output_dir,
                batch_size=args.batch_size,
            )
            popqa_rows.append({"method": method_name, **popqa_metrics})
        except Exception as exc:
            popqa_rows.append({"method": method_name, "error": str(exc)})

        try:
            beir_cfg = {
                "encoder": encoder_cfg,
                "index": {"metric": "ip", "normalize": True, "batch_size": args.batch_size},
                "split": "test",
                "search_k": 100,
                "datasets": beir_datasets,
                "metrics": {
                    "top_k": top_k,
                    "output_path": str(output_dir / f"{method_name}_beir3_summary.json"),
                    "output_csv_path": str(output_dir / f"{method_name}_beir3_summary.csv"),
                    "run_id": f"beir3_{method_name}",
                },
            }
            beir_summary = evaluate_beir(beir_cfg)
            beir_rows.append({"method": method_name, **beir_summary.get("average", {})})
        except Exception as exc:
            beir_rows.append({"method": method_name, "error": str(exc)})

    summarize_comparison(popqa_rows, output_dir / "popqa_comparison_summary.json")
    summarize_comparison(beir_rows, output_dir / "beir3_comparison_summary.json")

    valid_popqa = [r for r in popqa_rows if "mrr" in r and "ndcg@10" in r and "recall@10" in r]
    if valid_popqa:
        plot_metric_comparison(
            labels=[r["method"] for r in valid_popqa],
            values=[float(r["ndcg@10"]) for r in valid_popqa],
            output_path=output_dir / "popqa_ndcg10_comparison.png",
            title="PopQA NDCG@10 Comparison",
            y_label="ndcg@10",
        )
    valid_beir = [r for r in beir_rows if "mrr" in r and "ndcg@10" in r and "recall@10" in r]
    if valid_beir:
        plot_metric_comparison(
            labels=[r["method"] for r in valid_beir],
            values=[float(r["ndcg@10"]) for r in valid_beir],
            output_path=output_dir / "beir3_ndcg10_comparison.png",
            title="BEIR-3 NDCG@10 Comparison",
            y_label="ndcg@10",
        )

    print(json.dumps({"popqa": popqa_rows, "beir3": beir_rows}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
