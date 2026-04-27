#!/usr/bin/env python3
"""Prepare MS MARCO v2 dataset for training with hard negatives.

This script:
1. Loads msmarco-v2 queries, corpus, and qrels
2. Creates training samples with hard negatives (BM25 or random)
3. Saves in the same format as mixed_train_large.json
"""

import json
import gzip
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import argparse

from tqdm import tqdm


def load_corpus(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """Load corpus from gzipped jsonl."""
    corpus = {}
    print(f"Loading corpus from {corpus_path}...")
    
    open_func = gzip.open if corpus_path.endswith('.gz') else open
    mode = 'rt' if corpus_path.endswith('.gz') else 'r'
    
    with open_func(corpus_path, mode, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = doc['_id']
            text = doc.get('title', '') + ' ' + doc.get('text', '')
            corpus[doc_id] = {
                'title': doc.get('title', ''),
                'text': text.strip(),
            }
    
    print(f"Loaded {len(corpus)} documents")
    return corpus


def load_queries(queries_path: str) -> Dict[str, str]:
    """Load queries from jsonl."""
    queries = {}
    print(f"Loading queries from {queries_path}...")
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line.strip())
            queries[q['_id']] = q['text']
    
    print(f"Loaded {len(queries)} queries")
    return queries


def load_qrels(qrels_path: str) -> Dict[str, List[str]]:
    """Load qrels - mapping query_id -> list of relevant doc_ids."""
    qrels = defaultdict(list)
    
    with open(qrels_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = parts[1]
                score = int(parts[2])
                if score > 0:
                    qrels[query_id].append(doc_id)
    
    qrels = dict(qrels)
    print(f"Loaded qrels for {len(qrels)} queries")
    return qrels


def create_random_negatives(
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, List[str]],
    num_negatives: int = 5,
    max_samples: Optional[int] = None,
    random_seed: int = 42,
) -> List[Dict]:
    """Create training samples with random negatives (not hard negatives)."""
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
        attempts = 0
        max_attempts = num_negatives * 10
        
        while len(negatives) < num_negatives and attempts < max_attempts:
            neg_doc_id = random.choice(all_doc_ids)
            
            if neg_doc_id in pos_doc_ids or neg_doc_id in negatives:
                attempts += 1
                continue
            
            neg_doc = corpus.get(neg_doc_id)
            if neg_doc is None:
                attempts += 1
                continue
            
            negatives.append(neg_doc['text'])
            attempts += 1
        
        samples.append({
            'query': query_text,
            'positive': positive_text,
            'negatives': negatives,
            'domain': 'msmarco-v2',
        })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples


def create_bm25_negatives(
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, List[str]],
    num_negatives: int = 5,
    max_samples: Optional[int] = None,
    random_seed: int = 42,
) -> List[Dict]:
    """Create training samples with BM25 hard negatives.
    
    Uses rank_bm25 to find documents that are relevant to the query
    but are NOT in the qrels (i.e., not the correct answer).
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not found, installing...")
        import subprocess
        subprocess.run(['pip', 'install', 'rank-bm25'], check=True)
        from rank_bm25 import BM25Okapi
    
    random.seed(random_seed)
    
    all_doc_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id]['text'] for doc_id in all_doc_ids]
    
    print("Building BM25 index...")
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
        
        # Tokenize query
        query_tokens = query_text.lower().split()
        
        # Get BM25 scores for all documents
        scores = bm25.get_scores(query_tokens)
        
        # Sort by score (descending) and get indices
        scored_docs = [(doc_id, scores[i], corpus_list[i]) 
                       for i, doc_id in enumerate(all_doc_ids)]
        
        # Sort by BM25 score descending
        scored_docs.sort(key=lambda x: -x[1])
        
        # Get top negatives: relevant to query but NOT in qrels
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
        
        # If not enough BM25 negatives, fill with random
        if len(negatives) < num_negatives:
            neg_doc_ids_in_use = set(pos_doc_ids)
            neg_doc_ids_in_use.update([n['neg_doc_id'] for n in negatives])
            
            attempts = 0
            max_attempts = num_negatives * 20
            while len(negatives) < num_negatives and attempts < max_attempts:
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
            'domain': 'msmarco-v2',
        })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples


def create_hard_negatives_mixed(
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, List[str]],
    num_negatives: int = 5,
    max_samples: Optional[int] = None,
    random_seed: int = 42,
    bm25_negatives: int = 3,
    random_negatives: int = 2,
) -> List[Dict]:
    """Create training samples with mixed BM25 + random negatives.
    
    Args:
        bm25_negatives: number of BM25 hard negatives
        random_negatives: number of random negatives
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not found, installing...")
        import subprocess
        subprocess.run(['pip', 'install', 'rank-bm25'], check=True)
        from rank_bm25 import BM25Okapi
    
    random.seed(random_seed)
    
    all_doc_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id]['text'] for doc_id in all_doc_ids]
    
    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_list)
    
    samples = []
    query_items = list(queries.items())
    
    for query_id, query_text in tqdm(query_items, desc="Creating mixed negatives"):
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
        
        # BM25 negatives
        query_tokens = query_text.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        scored_docs = [(doc_id, scores[i], corpus_list[i]) 
                       for i, doc_id in enumerate(all_doc_ids)]
        scored_docs.sort(key=lambda x: -x[1])
        
        negatives = []
        
        # Add BM25 negatives
        for neg_doc_id, neg_score, neg_text in scored_docs:
            if neg_doc_id in pos_doc_ids:
                continue
            if neg_doc_id in [n['neg_doc_id'] for n in negatives]:
                continue
            
            negatives.append({
                'neg_doc_id': neg_doc_id,
                'text': neg_text,
                'type': 'bm25',
            })
            
            if len(negatives) >= bm25_negatives:
                break
        
        # Add random negatives
        neg_doc_ids_in_use = set(pos_doc_ids)
        neg_doc_ids_in_use.update([n['neg_doc_id'] for n in negatives])
        
        attempts = 0
        max_attempts = random_negatives * 20
        while len(negatives) < bm25_negatives + random_negatives and attempts < max_attempts:
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
                'type': 'random',
            })
            neg_doc_ids_in_use.add(neg_doc_id)
            attempts += 1
        
        samples.append({
            'query': query_text,
            'positive': positive_text,
            'negatives': [n['text'] for n in negatives],
            'domain': 'msmarco-v2',
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
    parser = argparse.ArgumentParser(description="Prepare MS MARCO v2 dataset")
    parser.add_argument(
        "--corpus",
        type=str,
        default="/data/huggingface/mteb/msmarco-v2/corpus.jsonl.gz",
        help="Path to corpus.jsonl.gz",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="/data/huggingface/mteb/msmarco-v2/queries.jsonl",
        help="Path to queries.jsonl",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default="/data/huggingface/mteb/msmarco-v2/qrels/train.tsv",
        help="Path to qrels train.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/last_projected-token/data",
        help="Output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (for debugging)",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=5,
        help="Total number of negatives per sample",
    )
    parser.add_argument(
        "--negative-type",
        type=str,
        choices=["random", "bm25", "mixed"],
        default="random",
        help="Type of negatives: random, bm25, or mixed (bm25 + random)",
    )
    parser.add_argument(
        "--bm25-negatives",
        type=int,
        default=3,
        help="Number of BM25 negatives (only for --negative-type=mixed)",
    )
    parser.add_argument(
        "--random-negatives",
        type=int,
        default=2,
        help="Number of random negatives (only for --negative-type=mixed)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    
    args = parser.parse_args()
    
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    
    queries_with_qrels = {qid: queries[qid] for qid in queries if qid in qrels}
    print(f"Queries with qrels: {len(queries_with_qrels)}")
    
    print(f"Creating {args.num_negatives} negatives per sample (type: {args.negative_type})...")
    
    if args.negative_type == "random":
        samples = create_random_negatives(
            queries_with_qrels,
            corpus,
            qrels,
            num_negatives=args.num_negatives,
            max_samples=args.max_samples,
        )
    elif args.negative_type == "bm25":
        samples = create_bm25_negatives(
            queries_with_qrels,
            corpus,
            qrels,
            num_negatives=args.num_negatives,
            max_samples=args.max_samples,
        )
    elif args.negative_type == "mixed":
        samples = create_hard_negatives_mixed(
            queries_with_qrels,
            corpus,
            qrels,
            num_negatives=args.num_negatives,
            max_samples=args.max_samples,
            bm25_negatives=args.bm25_negatives,
            random_negatives=args.random_negatives,
        )
    
    print(f"Created {len(samples)} samples")
    
    train_samples, val_samples = split_train_val(samples, args.val_ratio)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{args.negative_type}" if args.negative_type != "random" else ""
    
    train_path = output_dir / f"msmarco_v2_train{suffix}.json"
    val_path = output_dir / f"msmarco_v2_val{suffix}.json"
    combined_path = output_dir / f"msmarco_v2_combined{suffix}.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    
    with open(combined_path, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved train to {train_path}")
    print(f"Saved val to {val_path}")
    print(f"Saved combined to {combined_path}")
    
    print("\nDataset statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")


if __name__ == "__main__":
    main()