#!/usr/bin/env python3
"""Generate hard negatives for PopQA using dense retrieval.

For each question in PopQA, we find the most similar documents that are NOT
the correct one. These hard negatives help the model learn to distinguish
between similar entities.
"""

import argparse
import json
import numpy as np
import pyarrow.parquet as pq
import faiss
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import torch

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from projected_token.encoders.oscar import OscarEncoder


def load_popqa(path: str) -> List[Dict[str, Any]]:
    """Load PopQA dataset.
    
    Returns:
        List of dicts with: question, s_wiki_title, s_wiki_content, possible_answers
    """
    print(f"Loading PopQA from {path}...")
    table = pq.read_table(path)
    pydict = table.to_pydict()
    num_rows = table.num_rows
    
    dataset = []
    for i in range(num_rows):
        item = {col: pydict[col][i] for col in pydict.keys()}
        dataset.append({
            "question": item.get("question", [""])[0] if item.get("question") else "",
            "s_wiki_title": item.get("s_wiki_title", [""])[0] if item.get("s_wiki_title") else "",
            "s_wiki_content": item.get("s_wiki_content", [""])[0] if item.get("s_wiki_content") else "",
            "possible_answers": item.get("possible_answers", [])[0] if item.get("possible_answers") else [],
            "s_pop": item.get("s_pop", [0])[0] if item.get("s_pop") else 0,
        })
    
    print(f"Loaded {len(dataset)} PopQA samples")
    return dataset


def encode_documents_faiss(
    encoder: OscarEncoder,
    documents: List[str],
    batch_size: int = 32,
    desc: str = "Encoding",
) -> np.ndarray:
    """Encode documents and return normalized embeddings for FAISS."""
    embeddings = []
    
    for i in tqdm(range(0, len(documents), batch_size), desc=desc):
        batch = documents[i:i+batch_size]
        with torch.inference_mode():
            emb = encoder.encode(batch)
        embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    faiss.normalize_L2(embeddings)
    return embeddings.astype(np.float32)


def mine_hard_negatives_dense(
    encoder: OscarEncoder,
    queries: List[str],
    documents: List[str],
    positive_indices: List[int],
    top_k: int = 10,
    batch_size: int = 32,
) -> List[List[int]]:
    """Mine hard negatives using dense retrieval (FAISS).
    
    For each query, find top-k most similar documents that are NOT the positive.
    
    Args:
        encoder: OSCAR encoder for embeddings
        queries: List of questions
        documents: All candidate documents
        positive_indices: Index of positive document for each query
        top_k: Number of negatives to return
        batch_size: Batch size for encoding
        
    Returns:
        List of lists of negative indices for each query
    """
    print(f"\nBuilding FAISS index with {len(documents)} documents...")
    
    # Encode all documents
    doc_embeddings = encode_documents_faiss(
        encoder, documents, batch_size, desc="Encoding documents for index"
    )
    
    # Build FAISS index
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors, dim={dim}")
    
    # Mine hard negatives
    print("\nMining hard negatives...")
    all_negatives = []
    
    for i, query in enumerate(tqdm(queries, desc="Finding hard negatives")):
        # Encode query
        with torch.inference_mode():
            q_emb = encoder.encode([query])
        q_emb = q_emb.cpu().numpy()
        faiss.normalize_L2(q_emb)
        
        # Search
        D, I = index.search(q_emb, top_k + 20)  # Get more to filter
        
        # Filter out the positive document
        positive_idx = positive_indices[i]
        negatives = []
        for idx in I[0]:
            if idx != positive_idx and idx < len(documents):
                negatives.append(int(idx))
                if len(negatives) >= top_k:
                    break
        
        all_negatives.append(negatives)
    
    return all_negatives


def create_popqa_triplets_with_hard_negatives(
    popqa_data: List[Dict[str, Any]],
    encoder: OscarEncoder,
    top_k_negatives: int = 5,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Create PopQA triplets with hard negatives.
    
    Args:
        popqa_data: PopQA dataset
        encoder: OSCAR encoder
        top_k_negatives: Number of hard negatives per sample
        batch_size: Batch size for encoding
        max_samples: Limit number of samples
        
    Returns:
        List of triplets with hard negatives
    """
    if max_samples:
        popqa_data = popqa_data[:max_samples]
    
    # Use s_wiki_title as the positive document (entity retrieval)
    queries = [item["question"] for item in popqa_data]
    documents = [item["s_wiki_title"] for item in popqa_data]
    positive_indices = list(range(len(popqa_data)))
    
    # Mine hard negatives
    hard_negatives = mine_hard_negatives_dense(
        encoder=encoder,
        queries=queries,
        documents=documents,
        positive_indices=positive_indices,
        top_k=top_k_negatives,
        batch_size=batch_size,
    )
    
    # Create triplets
    triplets = []
    for i, item in enumerate(popqa_data):
        neg_indices = hard_negatives[i]
        neg_docs = [documents[idx] for idx in neg_indices]
        
        triplets.append({
            "query": item["question"],
            "positive": item["s_wiki_title"],
            "negatives": neg_docs,
            "s_wiki_content": item["s_wiki_content"],
            "possible_answers": item["possible_answers"],
            "s_pop": item["s_pop"],
            "domain": "popqa",
        })
    
    return triplets


def main():
    parser = argparse.ArgumentParser(description="Generate hard negatives for PopQA")
    parser.add_argument("--input-path", type=str, default="/data/popqa_enriched.parquet")
    parser.add_argument("--output-path", type=str, default="data/popqa_hard_negatives.json")
    parser.add_argument("--oscar-model", type=str, default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top-k-negatives", type=int, default=5, help="Number of hard negatives per sample")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()
    
    # Load encoder
    print(f"Loading OSCAR encoder: {args.oscar_model}")
    encoder = OscarEncoder(
        model_name_or_path=args.oscar_model,
        device=args.device,
    )
    
    # Load PopQA
    popqa_data = load_popqa(args.input_path)
    
    # Create triplets with hard negatives
    print(f"\nCreating triplets with {args.top_k_negatives} hard negatives per sample...")
    triplets = create_popqa_triplets_with_hard_negatives(
        popqa_data=popqa_data,
        encoder=encoder,
        top_k_negatives=args.top_k_negatives,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    
    # Save
    print(f"\nSaving to {args.output_path}...")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(triplets, f, indent=2)
    
    # Statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(triplets)}")
    print(f"Average negatives per sample: {np.mean([len(t['negatives']) for t in triplets]):.1f}")
    print(f"Min negatives: {min(len(t['negatives']) for t in triplets)}")
    print(f"Max negatives: {max(len(t['negatives']) for t in triplets)}")
    
    # Show example
    print("\n=== Example ===")
    example = triplets[0]
    print(f"Query: {example['query']}")
    print(f"Positive: {example['positive']}")
    print(f"Negatives: {example['negatives'][:2]}...")


if __name__ == "__main__":
    main()