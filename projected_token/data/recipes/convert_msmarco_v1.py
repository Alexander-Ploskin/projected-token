#!/usr/bin/env python3
"""Convert MS MARCO v1 dataset from parquet to JSON for projector training.

This script:
1. Loads MS MARCO v1.1 from parquet files
2. Extracts query, positive passage (where is_selected=1), and negative passages
3. Saves in the same format as other trainers expect:
   {
       "query": str,
       "positive": str,
       "negatives": [str, str, ...],
       "domain": "msmarco",
   }
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    import subprocess
    subprocess.run(['pip', 'install', 'pandas', 'pyarrow'], check=True)
    import pandas as pd


def load_parquet_dataset(parquet_path: str) -> List[Dict]:
    """Load MS MARCO v1 dataset from parquet file."""
    print(f"Loading from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        query = row['query']
        passages = row['passages']
        
        passage_texts = passages['passage_text']
        is_selected = passages['is_selected']
        
        positives = []
        negatives = []
        
        for i, text in enumerate(passage_texts):
            if is_selected[i] == 1:
                positives.append(text)
            else:
                negatives.append(text)
        
        if not positives:
            continue
        
        sample = {
            'query': query,
            'positive': positives[0],  # Take first positive
            'negatives': negatives if negatives else [positives[0]],  # At least one negative
            'domain': 'msmarco',
        }
        samples.append(sample)
    
    return samples


def create_bm25_negatives_simple(
    samples: List[Dict],
    num_negatives: int = 5,
    random_seed: int = 42,
) -> List[Dict]:
    """Create simple negatives using random sampling from other queries' positives.
    
    This is a simpler approach than BM25 - just sample random passages from other queries.
    For more sophisticated hard negatives, use the generate_hard_negatives.py script.
    """
    random.seed(random_seed)
    
    # Collect all positives
    all_positives = [s['positive'] for s in samples]
    
    # For each sample, sample negatives
    for sample in tqdm(samples, desc="Creating negatives"):
        pos = sample['positive']
        
        # Get other positives (not the current one)
        other_positives = [p for p in all_positives if p != pos]
        
        if len(other_positives) >= num_negatives:
            sampled = random.sample(other_positives, num_negatives)
        else:
            sampled = other_positives
        
        sample['negatives'] = sampled
    
    return samples


def split_train_val(
    samples: List[Dict],
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple:
    """Split into train and validation sets."""
    random.seed(random_seed)
    random.shuffle(samples)
    
    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description="Convert MS MARCO v1.1 to JSON")
    parser.add_argument("--train-path", type=str, 
                        default="/data/huggingface/microsoft/ms_marco/v1.1/train-00000-of-00001.parquet",
                        help="Path to train parquet")
    parser.add_argument("--validation-path", type=str, 
                        default="/data/huggingface/microsoft/ms_marco/v1.1/validation-00000-of-00001.parquet",
                        help="Path to validation parquet")
    parser.add_argument("--output-dir", type=str, 
                        default="/app/last_projected-token/data",
                        help="Output directory")
    parser.add_argument("--num-negatives", type=int, default=5,
                        help="Number of negatives per sample")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Maximum number of train samples")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Maximum number of validation samples")
    
    args = parser.parse_args()
    
    # Load train
    train_samples = load_parquet_dataset(args.train_path)
    if args.max_train_samples:
        train_samples = train_samples[:args.max_train_samples]
    
    # Load validation
    val_samples = load_parquet_dataset(args.validation_path)
    if args.max_val_samples:
        val_samples = val_samples[:args.max_val_samples]
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Negatives are already in the parquet data (is_selected=0), no need to generate
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    train_path = output_dir / "msmarco_v1_train.json"
    val_path = output_dir / "msmarco_v1_val.json"
    combined_path = output_dir / "msmarco_v1_combined.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    
    # Combined: use train as training, val as validation (for mixed training)
    combined = train_samples + val_samples
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved train to {train_path}")
    print(f"Saved val to {val_path}")
    print(f"Saved combined to {combined_path}")
    
    # Stats
    avg_neg_train = sum(len(s['negatives']) for s in train_samples) / len(train_samples)
    avg_neg_val = sum(len(s['negatives']) for s in val_samples) / len(val_samples)
    
    print(f"\nTrain: {len(train_samples)} samples, avg negatives: {avg_neg_train:.1f}")
    print(f"Val: {len(val_samples)} samples, avg negatives: {avg_neg_val:.1f}")


if __name__ == "__main__":
    main()