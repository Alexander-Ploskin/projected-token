#!/usr/bin/env python3
"""Generate mixed-domain dataset for projector training with hard negatives.

This script creates a mixed dataset from:
- MS MARCO (search queries -> passages)
- PopQA (factual questions -> Wikipedia articles) with hard negatives
- Natural Questions (factual questions -> answers) with hard negatives

The dataset is used to train a projector that generalizes across domains.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from projected_token.encoders.oscar import OscarEncoder


def load_popqa_with_hard_negatives(
    hard_negatives_path: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load PopQA dataset with pre-generated hard negatives.
    
    Args:
        hard_negatives_path: Path to JSON file with hard negatives
        max_samples: Limit number of samples
        
    Returns:
        List of dicts with: query, positive, negatives, domain
    """
    print(f"Loading PopQA with hard negatives from {hard_negatives_path}...")
    
    with open(hard_negatives_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    # Convert to training format
    result = []
    for item in data:
        result.append({
            "query": item["query"],
            "positive": item["positive"],
            "negatives": item.get("negatives", []),
            "domain": "popqa",
        })
    
    print(f"Loaded {len(result)} PopQA samples with hard negatives")
    return result


def load_popqa_raw(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load PopQA dataset from local path (without hard negatives, for fallback).
    
    Returns:
        List of dicts with: query, positive, negatives, domain
    """
    print("Loading PopQA from local...")
    import pyarrow.parquet as pq
    
    # Use enriched parquet from /data
    table = pq.read_table("/data/popqa_enriched.parquet")
    pydict = table.to_pydict()
    num_rows = table.num_rows
    
    actual_max = max_samples if max_samples and max_samples < num_rows else num_rows
    
    dataset = []
    for i in tqdm(range(actual_max), desc="Loading PopQA"):
        item = {col: row[i] for col, row in pydict.items()}
        query = item.get("question") or ""
        positive = item.get("s_wiki_title") or ""
        
        dataset.append({
            "query": str(query) if query else "",
            "positive": str(positive) if positive else "",
            "negatives": [],
            "domain": "popqa",
        })
    
    print(f"Loaded {len(dataset)} PopQA samples (available: {num_rows})")
    return dataset


def load_natural_questions(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load Natural Questions dataset from local path.
    
    Returns:
        List of dicts with: query, positive, negatives, domain
    """
    print("Loading Natural Questions from local...")
    import pyarrow.parquet as pq
    
    table = pq.read_table("/data/huggingface/sentence-transformers/natural-questions/pair/train-00000-of-00001.parquet")
    pydict = table.to_pydict()
    num_rows = table.num_rows
    
    actual_max = max_samples if max_samples and max_samples < num_rows else num_rows
    
    dataset = []
    for i in tqdm(range(actual_max), desc="Loading Natural Questions"):
        query = pydict["query"][i]
        answer = pydict["answer"][i]
        
        if isinstance(query, list):
            query = query[0] if query else ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        
        dataset.append({
            "query": str(query) if query else "",
            "positive": str(answer) if answer else "",
            "negatives": [],
            "domain": "natural_questions",
        })
    
    print(f"Loaded {len(dataset)} Natural Questions samples (available: {num_rows})")
    return dataset


def load_msmarco(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MS MARCO dataset from local path.
    
    Returns:
        List of dicts with: query, positive, negatives, domain
    """
    print("Loading MS MARCO from local...")
    
    # Load from local path
    from datasets import load_dataset
    
    dataset = load_dataset(
        "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
        name="triplet",
        split="train",
    )
    
    result = []
    for i, item in enumerate(tqdm(dataset, desc="Loading MS MARCO")):
        if max_samples and i >= max_samples:
            break
        result.append({
            "query": item["query"],
            "positive": item["positive"],
            "negatives": [item["negative"]],
            "domain": "msmarco",
        })
    
    print(f"Loaded {len(result)} MS MARCO samples")
    return result


def generate_hard_negatives_for_nq(
    nq_data: List[Dict[str, Any]],
    encoder: Optional[Any] = None,
    top_k: int = 5,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Generate hard negatives for Natural Questions using random sampling.
    
    Since NQ doesn't have a clear document corpus like PopQA, we use random
    negatives from other samples as a simple approach.
    
    Args:
        nq_data: NQ dataset
        encoder: OSCAR encoder (not used for now, using random)
        top_k: Number of negatives per sample
        batch_size: Batch size
        
    Returns:
        NQ data with negatives filled
    """
    print(f"\nGenerating hard negatives for {len(nq_data)} NQ samples...")
    
    # Get all positives for random sampling
    all_positives = [item["positive"] for item in nq_data]
    
    for i, item in enumerate(tqdm(nq_data, desc="Adding NQ negatives")):
        if len(item["negatives"]) == 0:
            # Sample random negatives (different from positive)
            candidates = [p for p in all_positives if p != item["positive"]]
            if candidates:
                negs = random.sample(candidates, min(top_k, len(candidates)))
                item["negatives"] = negs
    
    return nq_data


def create_mixed_dataset(
    popqa_path: Optional[str] = None,
    msmarco_ratio: float = 0.30,
    popqa_ratio: float = 0.40,
    nq_ratio: float = 0.30,
    max_total: Optional[int] = 150000,
    output_path: str = "data/mixed_train_dataset.json",
    seed: int = 42,
    max_negatives: int = 5,
) -> Dict[str, Any]:
    """Create mixed-domain dataset with hard negatives.
    
    Args:
        popqa_path: Path to PopQA with hard negatives (JSON)
        msmarco_ratio: Target ratio for MS MARCO (default: 30%)
        popqa_ratio: Target ratio for PopQA (default: 40%)
        nq_ratio: Target ratio for Natural Questions (default: 30%)
        max_total: Maximum total samples
        output_path: Path to save the dataset
        seed: Random seed
        max_negatives: Maximum negatives per sample
        
    Returns:
        Statistics about the dataset
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Normalize ratios
    total_ratio = msmarco_ratio + popqa_ratio + nq_ratio
    msmarco_ratio = msmarco_ratio / total_ratio
    popqa_ratio = popqa_ratio / total_ratio
    nq_ratio = nq_ratio / total_ratio
    
    # Load datasets
    print("Loading datasets...")
    
    # Load PopQA with hard negatives
    if popqa_path and Path(popqa_path).exists():
        popqa_all = load_popqa_with_hard_negatives(popqa_path)
    else:
        popqa_all = load_popqa_raw()
        # Generate random negatives as fallback
        all_positives = [item["positive"] for item in popqa_all]
        for item in popqa_all:
            candidates = [p for p in all_positives if p != item["positive"]]
            if candidates:
                item["negatives"] = random.sample(candidates, min(max_negatives, len(candidates)))
    
    nq_all = load_natural_questions()
    # Generate random negatives for NQ
    all_nq_positives = [item["positive"] for item in nq_all]
    for item in nq_all:
        candidates = [p for p in all_nq_positives if p != item["positive"]]
        if candidates:
            item["negatives"] = random.sample(candidates, min(max_negatives, len(candidates)))
    
    msmarco_all = load_msmarco()
    
    # Calculate target counts based on ratios and max_total
    total_available = len(msmarco_all) + len(popqa_all) + len(nq_all)
    actual_total = min(max_total, total_available)
    
    msmarco_target = int(actual_total * msmarco_ratio)
    popqa_target = int(actual_total * popqa_ratio)
    nq_target = int(actual_total * nq_ratio)
    
    # Adjust for available data
    msmarco_target = min(msmarco_target, len(msmarco_all))
    popqa_target = min(popqa_target, len(popqa_all))
    nq_target = min(nq_target, len(nq_all))
    
    print(f"\nTarget distribution: MS MARCO={msmarco_target}, PopQA={popqa_target}, NQ={nq_target}")
    
    # Sample the data
    msmarco = msmarco_all[:msmarco_target]
    popqa = popqa_all[:popqa_target]
    nq = nq_all[:nq_target]
    
    print(f"Using: MS MARCO={len(msmarco)}, PopQA={len(popqa)}, NQ={len(nq)}")
    
    # Ensure domain is set
    for item in msmarco:
        item["domain"] = "msmarco"
    for item in popqa:
        item["domain"] = "popqa"
    for item in nq:
        item["domain"] = "natural_questions"
    
    # Combine all datasets
    all_data = msmarco + popqa + nq
    random.shuffle(all_data)
    
    # Limit negatives per sample
    for item in all_data:
        if isinstance(item["negatives"], list):
            item["negatives"] = item["negatives"][:max_negatives]
    
    # Save dataset
    print(f"\nSaving dataset to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Statistics
    domain_counts = {}
    for item in all_data:
        domain = item.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    avg_negatives = np.mean([len(item.get("negatives", [])) for item in all_data])
    
    stats = {
        "total_samples": len(all_data),
        "domain_counts": domain_counts,
        "msmarco_ratio": domain_counts.get("msmarco", 0) / len(all_data),
        "popqa_ratio": domain_counts.get("popqa", 0) / len(all_data),
        "nq_ratio": domain_counts.get("natural_questions", 0) / len(all_data),
        "avg_negatives_per_sample": avg_negatives,
    }
    
    return stats


def analyze_token_lengths(dataset_path: str, tokenizer_name: str = "/data/huggingface/Qwen/Qwen2-7B-Instruct"):
    """Analyze token lengths in the generated dataset."""
    print(f"\nAnalyzing token lengths from {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    query_lengths = []
    pos_lengths = []
    neg_lengths = []
    
    for item in tqdm(data[:1000], desc="Analyzing"):
        query = item.get("query") or ""
        positive = item.get("positive") or ""
        negative = item.get("negative") or ""
        
        if query:
            query_lengths.append(len(tokenizer.encode(str(query), add_special_tokens=False)))
        if positive:
            pos_lengths.append(len(tokenizer.encode(str(positive), add_special_tokens=False)))
        if negative:
            neg_lengths.append(len(tokenizer.encode(str(negative), add_special_tokens=False)))
    
    print("\n=== Query Lengths ===")
    print(f"Mean: {np.mean(query_lengths):.1f}, Median: {np.median(query_lengths):.1f}")
    
    print("\n=== Positive Lengths ===")
    print(f"Mean: {np.mean(pos_lengths):.1f}, Median: {np.median(pos_lengths):.1f}")
    
    print("\n=== Negative Lengths ===")
    print(f"Mean: {np.mean(neg_lengths):.1f}, Median: {np.median(neg_lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Generate mixed-domain dataset with hard negatives")
    parser.add_argument("--msmarco-ratio", type=float, default=0.30,
                        help="Target ratio for MS MARCO (default: 0.30)")
    parser.add_argument("--popqa-ratio", type=float, default=0.40,
                        help="Target ratio for PopQA (default: 0.40)")
    parser.add_argument("--nq-ratio", type=float, default=0.30,
                        help="Target ratio for Natural Questions (default: 0.30)")
    parser.add_argument("--max-total", type=int, default=150000,
                        help="Maximum total samples (default: 150000)")
    parser.add_argument("--output-path", type=str, default="data/mixed_train_dataset.json")
    parser.add_argument("--popqa-hard-negatives", type=str, default=None,
                        help="Path to PopQA with hard negatives (JSON)")
    parser.add_argument("--max-negatives", type=int, default=5,
                        help="Maximum negatives per sample (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze", action="store_true", help="Analyze token lengths after generation")
    
    args = parser.parse_args()
    
    stats = create_mixed_dataset(
        popqa_path=args.popqa_hard_negatives,
        msmarco_ratio=args.msmarco_ratio,
        popqa_ratio=args.popqa_ratio,
        nq_ratio=args.nq_ratio,
        max_total=args.max_total,
        output_path=args.output_path,
        seed=args.seed,
        max_negatives=args.max_negatives,
    )
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Domain counts: {stats['domain_counts']}")
    print(f"MS MARCO ratio: {stats['msmarco_ratio']:.1%}")
    print(f"PopQA ratio: {stats['popqa_ratio']:.1%}")
    print(f"NQ ratio: {stats['nq_ratio']:.1%}")
    print(f"Avg negatives per sample: {stats['avg_negatives_per_sample']:.1f}")
    
    if args.analyze:
        analyze_token_lengths(args.output_path)


if __name__ == "__main__":
    import torch
    main()