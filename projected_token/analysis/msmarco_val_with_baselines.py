from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset

from projected_token.encoders.oscar import OscarEncoder
from projected_token.encoders.oscar import OscarProjectorEncoder
from projected_token.encoders.salesforce import SalesforceEncoder
from projected_token.retrieval.bm25_baseline import SimpleBM25


def _compute_batch_metrics_from_similarities(similarities: torch.Tensor) -> dict[str, float]:
    ranks = torch.argsort(torch.argsort(similarities, dim=1, descending=True), dim=1)
    batch_size = similarities.size(0)

    mrr = 0.0
    mrr_at_10 = 0.0
    ndcg_at_10 = 0.0
    recall_at_1 = 0.0
    recall_at_5 = 0.0
    recall_at_10 = 0.0

    for i in range(batch_size):
        rank = int(ranks[i, i].item()) + 1
        mrr += 1.0 / rank
        if rank <= 10:
            mrr_at_10 += 1.0 / rank
            ndcg_at_10 += 1.0 / np.log2(rank + 1.0)
        if rank == 1:
            recall_at_1 += 1.0
        if rank <= 5:
            recall_at_5 += 1.0
        if rank <= 10:
            recall_at_10 += 1.0

    return {
        "mrr": mrr / batch_size,
        "mrr@10": mrr_at_10 / batch_size,
        "ndcg@10": ndcg_at_10 / batch_size,
        "recall@1": recall_at_1 / batch_size,
        "recall@5": recall_at_5 / batch_size,
        "recall@10": recall_at_10 / batch_size,
    }


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {"mrr": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0}
    keys = items[0].keys()
    return {k: float(np.mean([m[k] for m in items])) for k in keys}


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _prepare_val_pairs(
    dataset_path: str,
    dataset_config: str,
    dataset_split: str,
    val_split: float,
    max_val_samples: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    full = load_dataset(dataset_path, name=dataset_config, split=dataset_split)
    total = len(full)
    if val_split >= 1.0:
        n_full = min(max_val_samples, total)
        queries = [str(full[i]["query"]) for i in range(n_full)]
        positives = [str(full[i]["positive"]) for i in range(n_full)]
        return queries, positives

    val_size = max(1, int(total * val_split))
    train_size = total - val_size
    if train_size < 1:
        raise ValueError(f"Not enough samples: total={total}, val_split={val_split}")

    train_subset, val_subset = torch.utils.data.random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    # Use exactly the first N items of the val subset for determinism.
    n = min(max_val_samples, len(val_subset))
    val_subset = torch.utils.data.Subset(val_subset, list(range(n)))

    queries = [str(val_subset[i]["query"]) for i in range(len(val_subset))]
    positives = [str(val_subset[i]["positive"]) for i in range(len(val_subset))]
    return queries, positives


def _batched_pairs(queries: list[str], positives: list[str], batch_size: int):
    for start in range(0, len(queries), batch_size):
        q = queries[start:start + batch_size]
        p = positives[start:start + batch_size]
        if len(q) < 2:
            continue
        yield q, p


def evaluate_projector(
    method_name: str,
    checkpoint_path: Path,
    oscar_model_name: str,
    queries: list[str],
    positives: list[str],
    batch_size: int,
    device: str,
) -> dict[str, float]:
    encoder = OscarProjectorEncoder(
        oscar_model_name=oscar_model_name,
        projector_path=str(checkpoint_path),
        device=device,
    )
    batch_metrics: list[dict[str, float]] = []
    for q_batch, p_batch in _batched_pairs(queries, positives, batch_size):
        q_emb = torch.nn.functional.normalize(_to_tensor(encoder.encode(q_batch)).float(), p=2, dim=-1)
        p_emb = torch.nn.functional.normalize(_to_tensor(encoder.encode(p_batch)).float(), p=2, dim=-1)
        sims = torch.matmul(q_emb, p_emb.T)
        batch_metrics.append(_compute_batch_metrics_from_similarities(sims))
    result = _mean_metrics(batch_metrics)
    result["method"] = method_name
    return result


def evaluate_oscar_baseline(
    *,
    method_name: str,
    aggregation: str,
    model_name_or_path: str,
    queries: list[str],
    positives: list[str],
    batch_size: int,
    device: str,
) -> dict[str, float]:
    encoder = OscarEncoder(
        model_name_or_path=model_name_or_path,
        device=device,
        aggregation=aggregation,
    )
    batch_metrics: list[dict[str, float]] = []
    for q_batch, p_batch in _batched_pairs(queries, positives, batch_size):
        q_emb = torch.nn.functional.normalize(_to_tensor(encoder.encode(q_batch)).float(), p=2, dim=-1)
        p_emb = torch.nn.functional.normalize(_to_tensor(encoder.encode(p_batch)).float(), p=2, dim=-1)
        sims = torch.matmul(q_emb, p_emb.T)
        batch_metrics.append(_compute_batch_metrics_from_similarities(sims))
    result = _mean_metrics(batch_metrics)
    result["method"] = method_name
    return result


def evaluate_sfr(
    model_name_or_path: str,
    queries: list[str],
    positives: list[str],
    batch_size: int,
    device: str,
) -> dict[str, float]:
    encoder = SalesforceEncoder(model_name_or_path=model_name_or_path, device=device)
    batch_metrics: list[dict[str, float]] = []
    for q_batch, p_batch in _batched_pairs(queries, positives, batch_size):
        q_emb = torch.nn.functional.normalize(_to_tensor(encoder.encode(documents=q_batch, questions=q_batch)).float(), p=2, dim=-1)
        p_emb = torch.nn.functional.normalize(_to_tensor(encoder.encode(documents=p_batch, questions=None)).float(), p=2, dim=-1)
        sims = torch.matmul(q_emb, p_emb.T)
        batch_metrics.append(_compute_batch_metrics_from_similarities(sims))
    result = _mean_metrics(batch_metrics)
    result["method"] = "sfr_baseline"
    return result


def evaluate_bm25(
    queries: list[str],
    positives: list[str],
    batch_size: int,
) -> dict[str, float]:
    batch_metrics: list[dict[str, float]] = []
    for q_batch, p_batch in _batched_pairs(queries, positives, batch_size):
        bm25 = SimpleBM25.from_texts(p_batch)
        sims = torch.full((len(q_batch), len(p_batch)), fill_value=-1e9, dtype=torch.float32)
        for i, q in enumerate(q_batch):
            ranked = bm25.search(q, top_k=len(p_batch))
            for pos, doc_idx in enumerate(ranked):
                if 0 <= doc_idx < len(p_batch):
                    sims[i, doc_idx] = float(len(p_batch) - pos)
        batch_metrics.append(_compute_batch_metrics_from_similarities(sims))
    result = _mean_metrics(batch_metrics)
    result["method"] = "bm25_baseline"
    return result


def _save_rows_as_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["method", "mrr", "mrr@10", "ndcg@10", "recall@1", "recall@5", "recall@10"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _plot_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [str(r["method"]) for r in rows]
    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, [float(r["mrr"]) for r in rows], width=w, label="MRR")
    ax.bar(x, [float(r["ndcg@10"]) for r in rows], width=w, label="NDCG@10")
    ax.bar(x + w, [float(r["recall@10"]) for r in rows], width=w, label="Recall@10")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title("MS MARCO Validation: Projector vs Baselines")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare projector checkpoints with BM25/SFR on MS MARCO validation split.")
    parser.add_argument("--dataset-path", default="/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3")
    parser.add_argument("--dataset-config", default="triplet")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--max-val-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oscar-model-name", default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--sfr-model-path", default="/data/huggingface/Salesforce/SFR-Embedding-Mistral")
    parser.add_argument("--output-json", default="artifacts/results/retrieval/msmarco_val_projector_vs_baselines.json")
    parser.add_argument("--output-csv", default="artifacts/results/retrieval/msmarco_val_projector_vs_baselines.csv")
    parser.add_argument("--output-plot", default="artifacts/results/retrieval/msmarco_val_projector_vs_baselines.png")
    args = parser.parse_args()

    queries, positives = _prepare_val_pairs(
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        val_split=args.val_split,
        max_val_samples=args.max_val_samples,
        seed=args.seed,
    )

    rows: list[dict[str, Any]] = []

    rows.append(
        evaluate_bm25(
            queries=queries,
            positives=positives,
            batch_size=args.batch_size,
        )
    )

    rows.append(
        evaluate_sfr(
            model_name_or_path=args.sfr_model_path,
            queries=queries,
            positives=positives,
            batch_size=args.batch_size,
            device=args.device,
        )
    )

    for aggregation in ("first", "last", "flatten"):
        rows.append(
            evaluate_oscar_baseline(
                method_name=f"oscar_baseline_{aggregation}",
                aggregation=aggregation,
                model_name_or_path=args.oscar_model_name,
                queries=queries,
                positives=positives,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    checkpoints = [
        ("projector_contrastive_1layer", Path("artifacts/runs/20260428_142514_mlp-projector-mlp-1layer-msmarco/checkpoints/best_model.pt")),
        ("projector_contrastive_2layer", Path("artifacts/runs/20260428_155309_mlp-projector-mlp-2layer-msmarco/checkpoints/best_model.pt")),
        ("projector_contrastive_3layer", Path("artifacts/runs/20260428_173204_mlp-projector-mlp-3layer-msmarco/checkpoints/best_model.pt")),
        ("projector_two_stage_1layer", Path("artifacts/runs/20260428_175038_mlp-two-stage-distill-to-contrastive-1layer/checkpoints/best_model.pt")),
        ("projector_distill_2layer", Path("artifacts/runs/20260428_201259_distill-projector-distill-2layer/checkpoints/best_model.pt")),
    ]
    for method_name, ckpt in checkpoints:
        if not ckpt.exists():
            continue
        try:
            rows.append(
                evaluate_projector(
                    method_name=method_name,
                    checkpoint_path=ckpt,
                    oscar_model_name=args.oscar_model_name,
                    queries=queries,
                    positives=positives,
                    batch_size=args.batch_size,
                    device=args.device,
                )
            )
        except Exception as exc:
            print(f"[warn] skipping {method_name}: {exc}")

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "dataset_path": args.dataset_path,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "val_split": args.val_split,
            "max_val_samples": len(queries),
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    _save_rows_as_csv(Path(args.output_csv), rows)
    _plot_comparison(Path(args.output_plot), rows)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
