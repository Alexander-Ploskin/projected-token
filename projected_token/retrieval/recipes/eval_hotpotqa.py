#!/usr/bin/env python3
"""Quick evaluation on HotpotQA - index ALL contexts from top N questions, then search.

Steps:
1. Load top N questions from HotpotQA distractor validation
2. Extract ALL contexts from ALL questions → build corpus
3. Index entire corpus once with FAISS
4. For each query: encode query → search in FAISS → calculate metrics

Usage:
    python eval_hotpotqa.py --mode distractor --method oscar_projector --projector-path checkpoints/best_model.pt --max-queries 100
    python eval_hotpotqa.py --mode distractor --method bm25 --max-queries 100
    python eval_hotpotqa.py --mode distractor --method sfr_embedding --max-queries 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

import numpy as np
import faiss
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from projected_token.encoders.oscar import OscarProjectorEncoder


def load_hotpotqa_distractor(max_queries: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """Load HotpotQA distractor validation set.
    
    Returns:
        queries: List of query dicts with question, gold_indices
        corpus: List of context dicts with id, text, question_id, original_index
    """
    import pandas as pd
    
    path = "/data/huggingface/hotpotqa/hotpot_qa/distractor/validation-00000-of-00001.parquet"
    print(f"\n[1] Loading HotpotQA distractor validation from {path}...")
    df = pd.read_parquet(path)
    
    if max_queries:
        df = df.head(max_queries)
        print(f"    Using first {max_queries} rows")
    
    queries = []
    corpus = []
    
    print(f"\n[2] Extracting contexts from {len(df)} questions...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        question_id = row["id"]
        question = row["question"]
        
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        
        gold_titles = set(row["supporting_facts"]["title"])
        
        # Build contexts for this question
        question_contexts = []
        for ctx_idx, (title, sent_list) in enumerate(zip(titles, sentences)):
            text = " ".join(sent_list) if isinstance(sent_list, list) else str(sent_list)
            
            # Global corpus index
            corpus_idx = len(corpus)
            
            # Is this a gold context for this question?
            is_gold = title in gold_titles
            
            corpus.append({
                "id": f"{question_id}_{ctx_idx}",
                "question_id": question_id,
                "ctx_index": ctx_idx,
                "title": title,
                "text": text,
                "is_gold_for_query": is_gold,
            })
            
            if is_gold:
                question_contexts.append(corpus_idx)
        
        queries.append({
            "id": question_id,
            "question": question,
            "answer": row["answer"],
            "gold_corpus_indices": question_contexts,  # Indices in global corpus
        })
    
    print(f"    Total queries: {len(queries)}")
    print(f"    Total contexts in corpus: {len(corpus)}")
    
    return queries, corpus


class RetrievalSystem:
    """Base class for retrieval methods."""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.index = None
        self.corpus_texts = None
    
    def index_corpus(self, texts: List[str]) -> None:
        """Index all documents."""
        raise NotImplementedError
    
    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for query, return scores and indices."""
        raise NotImplementedError


class OscarPoolingRetriever(RetrievalSystem):
    """OSCAR with mean pooling."""
    
    def __init__(self, oscar_model: str, device: str = "cuda:0"):
        super().__init__(device)
        print(f"\n[3] Loading OSCAR model: {oscar_model}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False
    
    def index_corpus(self, texts: List[str]) -> None:
        self.corpus_texts = texts
        print(f"\n[4] Encoding {len(texts)} documents with OSCAR pooling...")
        
        all_embeds = []
        for i in tqdm(range(0, len(texts), 32), desc="Encoding docs"):
            batch = texts[i:i+32]
            with torch.no_grad():
                mem = self.oscar_model.compress_documents(batch)
                emb = mem.mean(dim=1)
                emb = F.normalize(emb, p=2, dim=-1)
            all_embeds.append(emb.cpu().numpy())
        
        doc_embeds = np.vstack(all_embeds).astype(np.float32)
        
        print(f"\n[5] Building FAISS index...")
        dim = doc_embeds.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(doc_embeds)
        print(f"    Index built with {self.index.ntotal} vectors, dim={dim}")
    
    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            mem = self.oscar_model.compress_documents([query], questions=[query])
            emb = mem.mean(dim=1)
            emb = F.normalize(emb, p=2, dim=-1).cpu().numpy().astype(np.float32)
        
        faiss.normalize_L2(emb)
        scores, indices = self.index.search(emb, k)
        return scores[0], indices[0]


class OscarProjectorRetriever(RetrievalSystem):
    """OSCAR with trained projector."""
    
    def __init__(self, oscar_model: str, projector_path: str, device: str = "cuda:0"):
        super().__init__(device)
        print(f"\n[3] Loading OSCAR + Projector from {projector_path}")
        self.encoder = OscarProjectorEncoder(
            oscar_model_name=oscar_model,
            projector_path=projector_path,
            device=device,
        )
    
    def index_corpus(self, texts: List[str]) -> None:
        self.corpus_texts = texts
        print(f"\n[4] Encoding {len(texts)} documents with OSCAR + Projector...")
        
        all_embeds = []
        for i in tqdm(range(0, len(texts), 32), desc="Encoding docs"):
            batch = texts[i:i+32]
            with torch.no_grad():
                emb = self.encoder.encode(batch)
                emb = F.normalize(emb, p=2, dim=-1)
            all_embeds.append(emb.cpu().numpy())
        
        doc_embeds = np.vstack(all_embeds).astype(np.float32)
        
        print(f"\n[5] Building FAISS index...")
        dim = doc_embeds.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(doc_embeds)
        print(f"    Index built with {self.index.ntotal} vectors, dim={dim}")
    
    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            emb = self.encoder.encode([query])
            emb = F.normalize(emb, p=2, dim=-1).cpu().numpy().astype(np.float32)
        
        faiss.normalize_L2(emb)
        scores, indices = self.index.search(emb, k)
        return scores[0], indices[0]


class SFRRetriever(RetrievalSystem):
    """Salesforce SFR Embedding Mistral."""
    
    def __init__(self, model_path: str = "/data/huggingface/Salesforce/SFR-Embedding-Mistral", device: str = "cuda:0"):
        super().__init__(device)
        print(f"\n[3] Loading SFR model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        self.task_def = "Given a web search query, retrieve relevant passages that answer the query"
    
    def _last_token_pool(self, last_hidden_states, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size), sequence_lengths]
    
    def _encode_query(self, query: str) -> np.ndarray:
        query_text = f"Instruct: {self.task_def}\nQuery: {query}"
        inputs = self.tokenizer([query_text], padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            emb = F.normalize(emb, p=2, dim=-1).cpu().numpy().astype(np.float32)
        return emb[0]
    
    def index_corpus(self, texts: List[str]) -> None:
        self.corpus_texts = texts
        print(f"\n[4] Encoding {len(texts)} documents with SFR...")
        
        all_embeds = []
        for i in tqdm(range(0, len(texts), 16), desc="Encoding docs"):
            batch = texts[i:i+16]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                emb = F.normalize(emb, p=2, dim=-1)
            all_embeds.append(emb.cpu().numpy())
        
        doc_embeds = np.vstack(all_embeds).astype(np.float32)
        
        print(f"\n[5] Building FAISS index...")
        dim = doc_embeds.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(doc_embeds)
        print(f"    Index built with {self.index.ntotal} vectors, dim={dim}")
    
    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        emb = self._encode_query(query)
        
        faiss.normalize_L2(emb.reshape(1, -1))
        scores, indices = self.index.search(emb.reshape(1, -1), k)
        return scores[0], indices[0]


class BM25Retriever(RetrievalSystem):
    """BM25 retriever."""
    
    def __init__(self, device: str = "cuda:0"):
        super().__init__(device)
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            import subprocess
            subprocess.run(['pip', 'install', 'rank-bm25'], check=True)
            from rank_bm25 import BM25Okapi
        self.BM25Okapi = BM25Okapi
        self.bm25 = None
    
    def index_corpus(self, texts: List[str]) -> None:
        self.corpus_texts = texts
        print(f"\n[4] Building BM25 index on {len(texts)} documents...")
        
        tokenized = [t.lower().split() for t in tqdm(texts, desc="Tokenizing")]
        self.bm25 = self.BM25Okapi(tokenized)
        print(f"    BM25 index built")
    
    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_indices]
        
        return top_scores, top_indices


def evaluate(retriever: RetrievalSystem, queries: List[Dict], max_k: List[int] = [1, 2, 3, 5, 10, 20]) -> Dict:
    """Evaluate retrieval on queries."""
    results = {
        "total": len(queries),
        "recall@k": {k: 0 for k in max_k},
        "mrr": 0.0,
    }
    
    valid_queries = [q for q in queries if len(q["gold_corpus_indices"]) >= 2]
    print(f"\n[7] Valid queries (with 2+ gold contexts): {len(valid_queries)}")
    
    print(f"\n[8] Searching for each query in indexed corpus...")
    for query_data in tqdm(valid_queries, desc="Searching"):
        question = query_data["question"]
        gold_indices = set(query_data["gold_corpus_indices"])
        
        # Search in FAISS index
        scores, ranked_indices = retriever.search(question, k=max(max_k))
        
        # Calculate metrics
        for k in max_k:
            top_k = set(ranked_indices[:k].tolist() if hasattr(ranked_indices, 'tolist') else ranked_indices[:k])
            if gold_indices & top_k:  # any gold found in top_k
                results["recall@k"][k] += 1
        
        # MRR
        ranked = ranked_indices.tolist() if hasattr(ranked_indices, 'tolist') else list(ranked_indices)
        for rank, idx in enumerate(ranked, 1):
            if idx in gold_indices:
                results["mrr"] += 1.0 / rank
                break
    
    # Normalize
    n = len(valid_queries)
    for k in max_k:
        results["recall@k"][k] = results["recall@k"][k] / n if n > 0 else 0
    results["mrr"] = results["mrr"] / n if n > 0 else 0
    results["n_valid"] = n
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on HotpotQA")
    parser.add_argument("--mode", choices=["distractor", "fullwiki"], default="distractor")
    parser.add_argument("--method", required=True,
                        choices=["oscar_pooling", "oscar_projector", "bm25", "sfr_embedding"])
    parser.add_argument("--oscar-model", default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--projector-path", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    
    args = parser.parse_args()
    
    # Step 1-2: Load queries and build corpus
    queries, corpus = load_hotpotqa_distractor(args.max_queries)
    corpus_texts = [c["text"] for c in corpus]
    
    # Step 3: Create retriever
    if args.method == "oscar_pooling":
        retriever = OscarPoolingRetriever(args.oscar_model, args.device)
    elif args.method == "oscar_projector":
        if not args.projector_path:
            raise ValueError("--projector-path required for oscar_projector")
        retriever = OscarProjectorRetriever(args.oscar_model, args.projector_path, args.device)
    elif args.method == "sfr_embedding":
        retriever = SFRRetriever(device=args.device)
    elif args.method == "bm25":
        retriever = BM25Retriever(args.device)
    
    # Step 4-5: Index corpus and build FAISS
    retriever.index_corpus(corpus_texts)
    
    # Step 6-8: Evaluate
    print(f"\n=== Evaluating {args.method} on HotpotQA {args.mode} ===")
    results = evaluate(retriever, queries)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS ({results['n_valid']} valid queries)")
    print(f"{'='*50}")
    for k in [1, 2, 3, 5, 10, 20]:
        if k in results["recall@k"]:
            print(f"Recall@{k:2d}: {results['recall@k'][k]:.4f}")
    print(f"MRR:        {results['mrr']:.4f}")
    
    # Save results
    output_path = f"hotpotqa_{args.method}_{args.mode}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()