from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.manifold import TSNE

from projected_token.encoders.oscar import OscarProjectorEncoder
from projected_token.io import write_csv, write_json


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return np.sum(a_n * b_n, axis=1)


def _encode(encoder: OscarProjectorEncoder, texts: list[str]) -> np.ndarray:
    emb = encoder.encode(texts)
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run embedding geometry diagnostics for OSCAR projector.")
    parser.add_argument("--oscar-model-name", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument(
        "--dataset-path",
        default="/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
    )
    parser.add_argument("--dataset-config", default="triplet")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="artifacts/analysis/embedding_diagnostics")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.dataset_path, name=args.dataset_config, split=args.dataset_split)
    n = min(args.sample_size, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    rows = [ds[int(i)] for i in idx]

    queries = [str(r["query"]) for r in rows]
    positives = [str(r["positive"]) for r in rows]
    negatives = [str(r["negative"]) for r in rows]

    encoder = OscarProjectorEncoder(
        oscar_model_name=args.oscar_model_name,
        projector_path=args.projector_path,
        device=args.device,
    )

    q_emb = _encode(encoder, queries)
    p_emb = _encode(encoder, positives)
    n_emb = _encode(encoder, negatives)

    pos_sim = _cosine(q_emb, p_emb)
    neg_sim = _cosine(q_emb, n_emb)
    margin = pos_sim - neg_sim

    # In-batch nearest-neighbor hit on positives.
    qn = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True).clip(min=1e-12)
    pn = p_emb / np.linalg.norm(p_emb, axis=1, keepdims=True).clip(min=1e-12)
    scores = qn @ pn.T
    ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
    positive_rank = np.array([ranks[i, i] + 1 for i in range(n)], dtype=np.int32)

    metrics = {
        "sample_size": n,
        "pos_sim_mean": float(np.mean(pos_sim)),
        "pos_sim_std": float(np.std(pos_sim)),
        "neg_sim_mean": float(np.mean(neg_sim)),
        "neg_sim_std": float(np.std(neg_sim)),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
        "hit@1": float(np.mean(positive_rank <= 1)),
        "hit@5": float(np.mean(positive_rank <= 5)),
        "hit@10": float(np.mean(positive_rank <= 10)),
        "mrr": float(np.mean(1.0 / positive_rank)),
    }
    write_json(out_dir / "similarity_metrics.json", metrics)

    pair_rows: list[dict[str, Any]] = []
    for i in range(n):
        pair_rows.append(
            {
                "idx": i,
                "query": queries[i],
                "positive": positives[i],
                "negative": negatives[i],
                "pos_sim": float(pos_sim[i]),
                "neg_sim": float(neg_sim[i]),
                "margin": float(margin[i]),
                "positive_rank": int(positive_rank[i]),
            }
        )
    write_csv(out_dir / "pairwise_similarity.csv", pair_rows)

    # Histograms
    plt.figure(figsize=(8, 5))
    plt.hist(pos_sim, bins=40, alpha=0.6, label="query-positive")
    plt.hist(neg_sim, bins=40, alpha=0.6, label="query-negative")
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_hist.png", dpi=180)
    plt.close()

    # t-SNE on subset to keep runtime bounded.
    tsne_n = min(120, n)
    ids = rng.choice(n, size=tsne_n, replace=False)
    emb_stack = np.concatenate([q_emb[ids], p_emb[ids], n_emb[ids]], axis=0)
    labels = (["query"] * tsne_n) + (["positive"] * tsne_n) + (["negative"] * tsne_n)
    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", perplexity=30)
    emb_2d = tsne.fit_transform(emb_stack)

    color_map = {"query": "#1f77b4", "positive": "#2ca02c", "negative": "#d62728"}
    plt.figure(figsize=(8, 6))
    for lab in ["query", "positive", "negative"]:
        mask = np.array([l == lab for l in labels], dtype=bool)
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=16, alpha=0.7, label=lab, c=color_map[lab])
    plt.title("t-SNE of Query/Positive/Negative Embeddings")
    plt.xlabel("tsne-1")
    plt.ylabel("tsne-2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "embeddings_tsne.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
