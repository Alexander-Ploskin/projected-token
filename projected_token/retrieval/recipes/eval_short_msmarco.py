#!/usr/bin/env python3
"""Evaluate OSCAR + Projector on short MS MARCO passages (<=70 tokens)."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Set

import numpy as np
import faiss
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from projected_token.encoders.oscar import OscarProjectorEncoder


def load_msmarco_short(max_tokens: int = 70, max_samples: int = 500):
    """Load short passages from MS MARCO."""
    print(f"Loading MS MARCO (max {max_tokens} tokens, max {max_samples} samples)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/huggingface/Qwen/Qwen2-7B-Instruct",
        trust_remote_code=True
    )
    
    dataset = load_dataset(
        "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
        name="triplet",
        split="train",
    )
    
    short_samples = []
    for item in tqdm(dataset, desc="Filtering by token length"):
        positive = item.get("positive", "")
        
        if not positive:
            continue
        
        tokens = tokenizer.encode(positive, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            short_samples.append({
                "query": item.get("query", ""),
                "positive": positive,
                "negative": item.get("negative", ""),
            })
            
            if len(short_samples) >= max_samples:
                break
    
    print(f"Found {len(short_samples)} samples with <= {max_tokens} tokens")
    return short_samples


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate on short MS MARCO passages")
    parser.add_argument("--max-tokens", type=int, default=70)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--projector-path", 
                        default="/app/last_projected-token/checkpoints/projector_mlp/best_model.pt")
    args = parser.parse_args()
    
    # Load short MS MARCO passages
    dataset = load_msmarco_short(args.max_tokens, args.max_samples)
    
    if len(dataset) < 50:
        print("Not enough short passages!")
        return
    
    # Create encoder
    print("\nBuilding encoder...")
    encoder = OscarProjectorEncoder(
        oscar_model_name="/data/huggingface/naver/oscar-qwen2-7B",
        projector_path=args.projector_path,
        device=args.device,
    )
    
    # Encode documents (positives)
    print("Encoding documents...")
    texts = [item["positive"] for item in dataset]
    questions = [item["query"] for item in dataset]
    
    doc_embeddings = []
    for i in tqdm(range(0, len(texts), 32), desc="Encoding documents"):
        batch = texts[i:i+32]
        with torch.inference_mode():
            emb = encoder.encode(batch)
        doc_embeddings.append(emb.cpu().numpy())
    
    doc_embeddings = np.vstack(doc_embeddings)
    faiss.normalize_L2(doc_embeddings)
    
    # Build FAISS index
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings.astype(np.float32))
    
    print(f"Index built with {index.ntotal} documents")
    
    # Evaluate retrieval
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
    
    # Track similarities
    pos_sims = []
    neg_sims = []
    
    for i, query in enumerate(tqdm(questions, desc="Evaluating")):
        correct_idx = i
        rel_docs = {correct_idx}
        
        # Encode query
        with torch.inference_mode():
            query_emb = encoder.encode([query])
        query_emb = query_emb.cpu().numpy()
        faiss.normalize_L2(query_emb)
        
        # Compute similarity to positive
        pos_sim = np.dot(query_emb, doc_embeddings[i:i+1].T)[0][0]
        pos_sims.append(float(pos_sim))
        
        # Search
        D, I = index.search(query_emb, max(k_values))
        retrieved_docs = I[0].tolist()
        
        # Compute similarity to top-10 retrieved negatives (not the correct one)
        neg_indices = [idx for idx in retrieved_docs[:10] if idx != i]
        if neg_indices:
            neg_sim = np.mean([np.dot(query_emb, doc_embeddings[idx:idx+1].T)[0][0] for idx in neg_indices])
            neg_sims.append(float(neg_sim))
        
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
    
    # Add similarity stats
    results["positive_similarity"] = {
        "mean": float(np.mean(pos_sims)),
        "std": float(np.std(pos_sims)),
    }
    if neg_sims:
        results["negative_similarity"] = {
            "mean": float(np.mean(neg_sims)),
            "std": float(np.std(neg_sims)),
        }
        results["similarity_diff"] = {
            "mean": float(np.mean(pos_sims) - np.mean(neg_sims)),
        }
    
    # Print results
    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS ON SHORT MS MARCO (<=70 tokens)")
    print(f"Number of queries: {results.get('mrr', {}).get('count', 0)}")
    print("=" * 60)
    
    for metric_name in ["recall@1", "recall@3", "recall@5", "recall@10", "recall@20",
                        "precision@1", "precision@3", "precision@5", "precision@10", "precision@20",
                        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20",
                        "mrr",
                        "positive_similarity", "negative_similarity", "similarity_diff"]:
        if metric_name in results:
            r = results[metric_name]
            if isinstance(r, dict) and "mean" in r:
                print(f"{metric_name:25s}: {r['mean']:.4f}" + (f" (std: {r['std']:.4f})" if "std" in r else ""))
    
    # Save results
    output_path = "short_msmarco_retrieval_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()