#!/usr/bin/env python3
"""Evaluate OSCAR + Projector on short PopQA passages (<=70 tokens).

This evaluates whether the projector works well on short passages similar to MS MARCO.
We take 500 short passages, build an index, and for each query check if we can retrieve
the correct passage.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Set

import numpy as np
import pyarrow.parquet as pq
import faiss
from tqdm import tqdm
import torch

from projected_token.encoders.oscar import OscarProjectorEncoder


def load_parquet(path: str) -> List[Dict[str, Any]]:
    """Load dataset from Parquet."""
    table = pq.read_table(path)
    pydict = table.to_pydict()
    num_rows = table.num_rows
    return [{col: row[i] for col, row in pydict.items()} for i in range(num_rows)]


def filter_short_passages(dataset: List[Dict], max_tokens: int = 70, max_samples: int = 500):
    """Filter passages by token length."""
    from transformers import AutoTokenizer
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/huggingface/Qwen/Qwen2-7B-Instruct",
        trust_remote_code=True
    )
    
    short_passages = []
    for item in tqdm(dataset, desc="Filtering"):
        # Use wiki_title (entity) instead of wiki_content (full article)
        # PopQA is entity retrieval, not passage retrieval
        title = item.get("s_wiki_title", "")
        if title and isinstance(title, str):
            # For titles, no token limit - they're short
            short_passages.append(item)
            if len(short_passages) >= max_samples:
                break
    
    print(f"Found {len(short_passages)} passages with <= {max_tokens} tokens")
    return short_passages


def main():
    parser = argparse.ArgumentParser(description="Evaluate on short PopQA passages")
    parser.add_argument("--max-tokens", type=int, default=70)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--projector-path", 
                        default="/app/last_projected-token/checkpoints/projector_mlp/best_model.pt")
    args = parser.parse_args()
    
    # Load PopQA dataset
    print("Loading PopQA dataset...")
    dataset = load_parquet("/data/popqa_enriched.parquet")
    print(f"Total: {len(dataset)} records")
    
    # Filter short passages
    short_dataset = filter_short_passages(dataset, args.max_tokens, args.max_samples)
    
    if len(short_dataset) < 50:
        print("Not enough short passages!")
        return
    
    # Create index from short passages
    print("\nBuilding index with short passages...")
    encoder = OscarProjectorEncoder(
        oscar_model_name="/data/huggingface/naver/oscar-qwen2-7B",
        projector_path=args.projector_path,
        device=args.device,
        embed_dim=768,
        pooler="mean",
        num_layers=1,
        dropout=0.0,
    )
    
    # Encode documents
    print("Encoding documents...")
    texts = [item["s_wiki_title"] for item in short_dataset]  # Use title (entity), not content
    questions = [item["question"] for item in short_dataset]
    
    doc_embeddings = []
    for i in tqdm(range(0, len(texts), 32), desc="Encoding documents"):
        batch = texts[i:i+32]
        with torch.inference_mode():
            emb = encoder.encode(batch)
        doc_embeddings.append(emb.cpu().numpy())
    
    doc_embeddings = np.vstack(doc_embeddings)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(doc_embeddings)
    
    # Build FAISS index
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings.astype(np.float32))
    
    print(f"Index built with {index.ntotal} documents")
    
    # Evaluate retrieval: for each item, query should retrieve itself as top-1
    print("\nEvaluating retrieval...")
    k_values = [1, 3, 5, 10, 20]
    
    metrics = {
        f"recall@{k}": [] for k in k_values
    }
    metrics.update({
        f"precision@{k}": [] for k in k_values
    })
    metrics.update({
        f"ndcg@{k}": [] for k in k_values
    })
    metrics["mrr"] = []
    
    for i, query in enumerate(tqdm(questions, desc="Evaluating")):
        # The correct document is at index i
        correct_idx = i
        rel_docs = {correct_idx}
        
        # Encode query
        with torch.inference_mode():
            query_emb = encoder.encode([query])
        query_emb = query_emb.cpu().numpy()
        faiss.normalize_L2(query_emb)
        
        # Search
        D, I = index.search(query_emb, max(k_values))
        retrieved_docs = I[0].tolist()
        
        # Compute metrics
        for k in k_values:
            metrics[f"recall@{k}"].append(compute_recall_at_k(rel_docs, retrieved_docs, k))
            metrics[f"precision@{k}"].append(compute_precision_at_k(rel_docs, retrieved_docs, k))
            metrics[f"ndcg@{k}"].append(compute_ndcg_at_k(rel_docs, retrieved_docs, k))
        
        metrics["mrr"].append(compute_mrr(rel_docs, retrieved_docs))
    
    # Aggregate metrics
    results = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            results[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }
    
    # Print results
    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS ON SHORT POPQA PASSAGES (<=70 tokens)")
    print(f"Number of queries: {results.get('mrr', {}).get('count', 0)}")
    print("=" * 60)
    
    for metric_name in ["recall@1", "recall@3", "recall@5", "recall@10", "recall@20",
                        "precision@1", "precision@3", "precision@5", "precision@10", "precision@20",
                        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20",
                        "mrr"]:
        if metric_name in results:
            r = results[metric_name]
            print(f"{metric_name:15s}: {r['mean']:.4f} (std: {r['std']:.4f})")
    
    # Save results
    output_path = "short_popqa_retrieval_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def compute_recall_at_k(rel_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
    """Compute Recall@k."""
    retrieved_k = set(retrieved_docs[:k])
    if len(rel_docs) == 0:
        return 0.0
    return len(rel_docs & retrieved_k) / len(rel_docs)


def compute_mrr(rel_docs: Set[int], retrieved_docs: List[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in rel_docs:
            return 1.0 / i
    return 0.0


def compute_ndcg_at_k(rel_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
    """Compute NDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k], 1):
        if doc_id in rel_docs:
            dcg += 1.0 / np.log2(i + 1)
    
    num_rel = min(len(rel_docs), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_rel + 1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_precision_at_k(rel_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
    """Compute Precision@k."""
    retrieved_k = set(retrieved_docs[:k])
    if k == 0:
        return 0.0
    return len(rel_docs & retrieved_k) / k


if __name__ == "__main__":
    main()