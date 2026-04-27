#!/usr/bin/env python3
"""Evaluate OSCAR + Projector on PopQA using wiki_title (entity matching).

For PopQA, the task is entity retrieval: find the Wikipedia article about 
the entity the question is asking about. Using s_wiki_title is more appropriate
than s_wiki_content which is the full article.
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
from transformers import AutoTokenizer

from projected_token.encoders.oscar import OscarProjectorEncoder


def load_popqa_by_title(max_samples: int = 500):
    """Load PopQA using wiki titles as positives (entity matching)."""
    print(f"Loading PopQA (using wiki_title, max {max_samples} samples)...")
    
    table = pq.read_table("/data/popqa_enriched.parquet")
    pydict = table.to_pydict()
    num_rows = table.num_rows
    
    samples = []
    for i in tqdm(range(num_rows), desc="Loading PopQA"):
        question = pydict["question"][i]
        wiki_title = pydict["s_wiki_title"][i]
        
        if question and wiki_title:
            samples.append({
                "query": str(question),
                "positive": str(wiki_title),  # Use title instead of content
            })
            
            if len(samples) >= max_samples:
                break
    
    print(f"Loaded {len(samples)} PopQA samples (by title)")
    return samples


def load_popqa_by_content(max_samples: int = 500):
    """Load PopQA using wiki content (original approach)."""
    print(f"Loading PopQA (using wiki_content, max {max_samples} samples)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/huggingface/Qwen/Qwen2-7B-Instruct",
        trust_remote_code=True
    )
    
    table = pq.read_table("/data/popqa_enriched.parquet")
    pydict = table.to_pydict()
    num_rows = table.num_rows
    
    samples = []
    for i in tqdm(range(num_rows), desc="Loading PopQA"):
        question = pydict["question"][i]
        wiki_content = pydict["s_wiki_content"][i]
        
        if question and wiki_content:
            tokens = tokenizer.encode(wiki_content, add_special_tokens=False)
            if len(tokens) <= 128:  # Use shorter passages
                samples.append({
                    "query": str(question),
                    "positive": str(wiki_content),
                })
                
                if len(samples) >= max_samples:
                    break
    
    print(f"Loaded {len(samples)} PopQA samples (by content, <=128 tokens)")
    return samples


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


def evaluate(dataset, encoder, device, projector_path):
    """Run evaluation on dataset."""
    if len(dataset) < 50:
        print("Not enough samples!")
        return None
    
    # Encode documents
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
    
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics["mrr"] = []
    
    pos_sims = []
    neg_sims = []
    
    for i, query in enumerate(tqdm(questions, desc="Evaluating")):
        correct_idx = i
        rel_docs = {correct_idx}
        
        with torch.inference_mode():
            query_emb = encoder.encode([query])
        query_emb = query_emb.cpu().numpy()
        faiss.normalize_L2(query_emb)
        
        # Similarity to positive
        pos_sim = np.dot(query_emb, doc_embeddings[i:i+1].T)[0][0]
        pos_sims.append(float(pos_sim))
        
        # Search
        D, I = index.search(query_emb, max(k_values))
        retrieved_docs = I[0].tolist()
        
        # Negative similarities
        neg_indices = [idx for idx in retrieved_docs[:10] if idx != i]
        if neg_indices:
            neg_sim = np.mean([np.dot(query_emb, doc_embeddings[idx:idx+1].T)[0][0] for idx in neg_indices])
            neg_sims.append(float(neg_sim))
        
        # Metrics
        for k in k_values:
            metrics[f"recall@{k}"].append(compute_recall_at_k(rel_docs, retrieved_docs, k))
            metrics[f"precision@{k}"].append(compute_precision_at_k(rel_docs, retrieved_docs, k))
            metrics[f"ndcg@{k}"].append(compute_ndcg_at_k(rel_docs, retrieved_docs, k))
        
        metrics["mrr"].append(compute_mrr(rel_docs, retrieved_docs))
    
    # Aggregate
    results = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            results[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }
    
    results["positive_similarity"] = {"mean": float(np.mean(pos_sims)), "std": float(np.std(pos_sims))}
    if neg_sims:
        results["negative_similarity"] = {"mean": float(np.mean(neg_sims)), "std": float(np.std(neg_sims))}
        results["similarity_diff"] = {"mean": float(np.mean(pos_sims) - np.mean(neg_sims))}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PopQA with wiki_title")
    parser.add_argument("--mode", choices=["title", "content", "both"], default="title")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--projector-path", 
                        default="/app/last_projected-token/checkpoints/projector_mlp/best_model.pt")
    args = parser.parse_args()
    
    # Create encoder
    print(f"\nBuilding encoder (device: {args.device})...")
    encoder = OscarProjectorEncoder(
        oscar_model_name="/data/huggingface/naver/oscar-qwen2-7B",
        projector_path=args.projector_path,
        device=args.device,
    )
    
    if args.mode in ["title", "both"]:
        print("\n" + "=" * 60)
        print("EVAL 1: PopQA with wiki_title (entity matching)")
        print("=" * 60)
        dataset_title = load_popqa_by_title(args.max_samples)
        results_title = evaluate(dataset_title, encoder, args.device, args.projector_path)
        
        if results_title:
            print("\n--- Results (wiki_title) ---")
            for m in ["mrr", "recall@1", "recall@5", "recall@10", "positive_similarity", "negative_similarity", "similarity_diff"]:
                if m in results_title:
                    r = results_title[m]
                    print(f"{m:25s}: {r['mean']:.4f}")
            
            with open("popqa_title_retrieval_metrics.json", "w") as f:
                json.dump(results_title, f, indent=2)
            print("\nSaved to popqa_title_retrieval_metrics.json")
    
    if args.mode in ["content", "both"]:
        print("\n" + "=" * 60)
        print("EVAL 2: PopQA with wiki_content (short passages)")
        print("=" * 60)
        dataset_content = load_popqa_by_content(args.max_samples)
        results_content = evaluate(dataset_content, encoder, args.device, args.projector_path)
        
        if results_content:
            print("\n--- Results (wiki_content) ---")
            for m in ["mrr", "recall@1", "recall@5", "recall@10", "positive_similarity", "negative_similarity", "similarity_diff"]:
                if m in results_content:
                    r = results_content[m]
                    print(f"{m:25s}: {r['mean']:.4f}")
            
            with open("popqa_content_retrieval_metrics.json", "w") as f:
                json.dump(results_content, f, indent=2)
            print("\nSaved to popqa_content_retrieval_metrics.json")


if __name__ == "__main__":
    main()