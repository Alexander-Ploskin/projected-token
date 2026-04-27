#!/usr/bin/env python3
"""Prepare MS MARCO v1 dataset for training with BM25 hard negatives.

Uses datasets library to load MS MARCO, then creates BM25 negatives.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    import subprocess
    subprocess.run(['pip', 'install', 'datasets'], check=True)
    from datasets import load_dataset

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Installing rank_bm25...")
    import subprocess
    subprocess.run(['pip', 'install', 'rank-bm25'], check=True)
    from rank_bm25 import BM25Okapi


def load_msmarco_v1():
    """Load MS MARCO v1 from datasets."""
    print("Loading MS MARCO v1 from HuggingFace...")
    
    # Load queries
    queries_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train[:1%]")
    print(f"Loaded queries: {len(queries_dataset)}")
    
    # This is just for demo - actually MS MARCO doesn't have corpus in datasets
    # We need a different approach
    return queries_dataset


def load_corpus_from_file(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """Load corpus from jsonl file."""
    corpus = {}
    print(f"Loading corpus from {corpus_path}...")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            doc = json.loads(line.strip())
            doc_id = doc.get('_id', doc.get('id'))
            text = doc.get('title', '') + ' ' + doc.get('text', '')
            corpus[doc_id] = {
                'title': doc.get('title', ''),
                'text': text.strip(),
            }
    
    print(f"Loaded {len(corpus)} documents")
    return corpus


def load_queries_from_file(queries_path: str) -> Dict[str, str]:
    """Load queries from jsonl file."""
    queries = {}
    print(f"Loading queries from {queries_path}...")
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading queries"):
            q = json.loads(line.strip())
            queries[q['_id']] = q['text']
    
    print(f"Loaded {len(queries)} queries")
    return queries


def load_qrels_from_file(qrels_path: str) -> Dict[str, List[str]]:
    """Load qrels from tsv file."""
    qrels = {}
    
    with open(qrels_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in tqdm(f, desc="Loading qrels"):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = parts[1]
                score = int(parts[2])
                if score > 0:
                    if query_id not in qrels:
                        qrels[query_id] = []
                    qrels[query_id].append(doc_id)
    
    print(f"Loaded qrels for {len(qrels)} queries")
    return qrels


def create_bm25_negatives(
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, List[str]],
    num_negatives: int = 5,
    max_samples: Optional[int] = None,
    random_seed: int = 42,
) -> List[Dict]:
    """Create training samples with BM25 hard negatives."""
    random.seed(random_seed)
    
    all_doc_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id]['text'] for doc_id in all_doc_ids]
    
    print(f"Building BM25 index with {len(corpus_list)} documents...")
    bm25 = BM25Okapi(corpus_list)
    
    samples = []
    query_items = list(queries.items())
    
    for query_id, query_text in tqdm(query_items, desc="Creating BM25 negatives"):
        if query_id not in qrels:
            continue
        
        pos_doc_ids = qrels[query_id]
        if not pos_doc_ids:
            continue
        
        pos_doc_id = pos_doc_ids[0]
        pos_doc = corpus.get(pos_doc_id)
        
        if pos_doc is None:
            continue
        
        positive_text = pos_doc['text']
        
        query_tokens = query_text.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        scored_docs = [(doc_id, scores[i], corpus_list[i]) 
                       for i, doc_id in enumerate(all_doc_ids)]
        scored_docs.sort(key=lambda x: -x[1])
        
        negatives = []
        for neg_doc_id, neg_score, neg_text in scored_docs:
            if neg_doc_id in pos_doc_ids:
                continue
            if neg_doc_id in [n['neg_doc_id'] for n in negatives]:
                continue
            
            negatives.append({
                'neg_doc_id': neg_doc_id,
                'text': neg_text,
                'bm25_score': neg_score,
            })
            
            if len(negatives) >= num_negatives:
                break
        
        # Fill with random if not enough BM25 negatives
        if len(negatives) < num_negatives:
            neg_doc_ids_in_use = set(pos_doc_ids)
            neg_doc_ids_in_use.update([n['neg_doc_id'] for n in negatives])
            
            attempts = 0
            while len(negatives) < num_negatives and attempts < 1000:
                neg_doc_id = random.choice(all_doc_ids)
                if neg_doc_id in neg_doc_ids_in_use:
                    attempts += 1
                    continue
                
                neg_doc = corpus.get(neg_doc_id)
                if neg_doc is None:
                    attempts += 1
                    continue
                
                negatives.append({
                    'neg_doc_id': neg_doc_id,
                    'text': neg_doc['text'],
                    'bm25_score': 0.0,
                })
                neg_doc_ids_in_use.add(neg_doc_id)
                attempts += 1
        
        samples.append({
            'query': query_text,
            'positive': positive_text,
            'negatives': [n['text'] for n in negatives],
            'domain': 'msmarco',
        })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples


def create_random_negatives(
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, List[str]],
    num_negatives: int = 5,
    max_samples: Optional[int] = None,
    random_seed: int = 42,
) -> List[Dict]:
    """Create training samples with random negatives."""
    random.seed(random_seed)
    
    all_doc_ids = list(corpus.keys())
    
    samples = []
    query_items = list(queries.items())
    
    for query_id, query_text in tqdm(query_items, desc="Creating random negatives"):
        if query_id not in qrels:
            continue
        
        pos_doc_ids = qrels[query_id]
        if not pos_doc_ids:
            continue
        
        pos_doc_id = pos_doc_ids[0]
        pos_doc = corpus.get(pos_doc_id)
        
        if pos_doc is None:
            continue
        
        positive_text = pos_doc['text']
        
        negatives = []
        neg_doc_ids_in_use = set(pos_doc_ids)
        
        attempts = 0
        while len(negatives) < num_negatives and attempts < 1000:
            neg_doc_id = random.choice(all_doc_ids)
            if neg_doc_id in neg_doc_ids_in_use:
                attempts += 1
                continue
            
            neg_doc = corpus.get(neg_doc_id)
            if neg_doc is None:
                attempts += 1
                continue
            
            negatives.append(neg_doc['text'])
            neg_doc_ids_in_use.add(neg_doc_id)
            attempts += 1
        
        samples.append({
            'query': query_text,
            'positive': positive_text,
            'negatives': negatives,
            'domain': 'msmarco',
        })
        
        if max_samples and len(samples) >= max_samples:
            break
    
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
    parser = argparse.ArgumentParser(description="Prepare MS MARCO dataset")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to corpus.jsonl")
    parser.add_argument("--queries", type=str, required=True,
                        help="Path to queries.jsonl")
    parser.add_argument("--qrels", type=str, required=True,
                        help="Path to qrels.tsv")
    parser.add_argument("--output-dir", type=str, default="/app/last_projected-token/data",
                        help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples")
    parser.add_argument("--num-negatives", type=int, default=5,
                        help="Number of negatives per sample")
    parser.add_argument("--negative-type", type=str, 
                        choices=["random", "bm25"], default="bm25",
                        help="Type of negatives")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    
    args = parser.parse_args()
    
    corpus = load_corpus_from_file(args.corpus)
    queries = load_queries_from_file(args.queries)
    qrels = load_qrels_from_file(args.qrels)
    
    queries_with_qrels = {qid: queries[qid] for qid in queries if qid in qrels}
    print(f"Queries with qrels: {len(queries_with_qrels)}")
    
    print(f"Creating {args.num_negatives} negatives per sample (type: {args.negative_type})...")
    
    if args.negative_type == "bm25":
        samples = create_bm25_negatives(
            queries_with_qrels, corpus, qrels,
            num_negatives=args.num_negatives,
            max_samples=args.max_samples,
        )
    else:
        samples = create_random_negatives(
            queries_with_qrels, corpus, qrels,
            num_negatives=args.num_negatives,
            max_samples=args.max_samples,
        )
    
    print(f"Created {len(samples)} samples")
    
    train_samples, val_samples = split_train_val(samples, args.val_ratio)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{args.negative_type}" if args.negative_type != "random" else ""
    
    train_path = output_dir / f"msmarco_train{suffix}.json"
    val_path = output_dir / f"msmarco_val{suffix}.json"
    combined_path = output_dir / f"msmarco_combined{suffix}.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    
    with open(combined_path, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved train to {train_path}")
    print(f"Saved val to {val_path}")
    print(f"Saved combined to {combined_path}")
    
    print(f"\nTotal: {len(samples)}, Train: {len(train_samples)}, Val: {len(val_samples)}")


if __name__ == "__main__":
    main()