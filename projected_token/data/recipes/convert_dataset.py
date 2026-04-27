#!/usr/bin/env python3
"""Convert existing mixed dataset to new format with negatives list."""

import json
import random
from pathlib import Path
from tqdm import tqdm

def convert_dataset(input_path: str, output_path: str, max_negatives: int = 5, max_samples: int = None):
    """Convert dataset to new format with negatives list."""
    
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    if max_samples:
        data = data[:max_samples]
    
    # Pre-compute unique positives by domain for efficient sampling
    domain_positives = {}
    for item in data:
        domain = item.get("domain", "unknown")
        if domain not in domain_positives:
            domain_positives[domain] = []
        domain_positives[domain].append(item["positive"])
    
    # Also get all positives combined
    all_positives = [item["positive"] for item in data]
    all_positives_set = set(all_positives)
    
    converted = []
    for item in tqdm(data, desc="Converting"):
        # Convert single negative to list
        old_negative = item.get("negative")
        negatives = []
        
        if old_negative and old_negative.strip():
            negatives.append(old_negative)
        
        # Add random negatives from different domain to reach max_negatives
        domain = item.get("domain", "unknown")
        domain_negs = [p for p in domain_positives.get(domain, []) if p != item["positive"]]
        other_negs = [p for p in all_positives if p != item["positive"] and p != item.get("negative")]
        
        # First try to get negatives from same domain (harder)
        if len(negatives) < max_negatives and domain_negs:
            needed = max_negatives - len(negatives)
            # Sample from same domain
            additional = random.sample(domain_negs, min(needed, len(domain_negs)))
            negatives.extend(additional)
        
        # If still not enough, use from other domains
        if len(negatives) < max_negatives and other_negs:
            needed = max_negatives - len(negatives)
            additional = random.sample(other_negs, min(needed, len(other_negs)))
            negatives.extend(additional)
        
        converted.append({
            "query": item["query"],
            "positive": item["positive"],
            "negatives": negatives,
            "domain": domain,
        })
    
    print(f"Converted to {len(converted)} samples")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"Saved to {output_path}")
    
    # Stats
    domain_counts = {}
    for item in converted:
        domain = item.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    avg_neg = sum(len(item["negatives"]) for item in converted) / len(converted)
    
    print(f"\nDomain distribution: {domain_counts}")
    print(f"Average negatives: {avg_neg:.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/mixed_train_dataset.json")
    parser.add_argument("--output", type=str, default="data/mixed_train_hard_negatives.json")
    parser.add_argument("--max-negatives", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output, args.max_negatives, args.max_samples)