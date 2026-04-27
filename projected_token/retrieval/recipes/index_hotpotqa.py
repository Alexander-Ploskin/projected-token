#!/usr/bin/env python3
"""Index HotpotQA contexts using different methods for retrieval projected_token.

Supports:
- OSCAR with pooling (mean)
- OSCAR with projector
- BM25
- Salesforce/sfr-embedding-mistral

Usage:
    python index_hotpotqa.py --mode distractor --method oscar_pooling --max-docs 10000
    python index_hotpotqa.py --mode distractor --method oscar_projector --projector-path checkpoints/best_model.pt
    python index_hotpotqa.py --mode distractor --method bm25
    python index_hotpotqa.py --mode distractor --method sfr_embedding
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from projected_token.encoders.oscar import OscarEncoder, OscarProjectorEncoder


def load_hotpotqa(mode: str = "distractor", max_docs: Optional[int] = None) -> List[Dict]:
    """Load HotpotQA dataset and extract contexts."""
    import pandas as pd
    
    if mode == "distractor":
        path = "/data/huggingface/hotpotqa/hotpot_qa/distractor/validation-00000-of-00001.parquet"
    else:
        path = "/data/huggingface/hotpotqa/hotpot_qa/fullwiki/validation-00000-of-00001.parquet"
    
    print(f"Loading HotpotQA {mode} from {path}...")
    df = pd.read_parquet(path)
    
    if max_docs:
        df = df.head(max_docs)
    
    contexts = []
    for idx, row in df.iterrows():
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        
        for i, (title, sent_list) in enumerate(zip(titles, sentences)):
            text = " ".join(sent_list) if isinstance(sent_list, list) else str(sent_list)
            contexts.append({
                "id": f"{row['id']}_{i}",
                "question_id": row["id"],
                "title": title,
                "text": text,
                "is_gold": False,
            })
            
            # Mark gold contexts
            gold_titles = set(row["supporting_facts"]["title"]) if "supporting_facts" in row else []
            if title in gold_titles:
                contexts[-1]["is_gold"] = True
    
    print(f"Loaded {len(contexts)} contexts")
    return contexts


def load_hotpotqa_queries(mode: str = "distractor") -> List[Dict]:
    """Load HotpotQA questions with gold context indices."""
    import pandas as pd
    
    if mode == "distractor":
        path = "/data/huggingface/hotpotqa/hotpot_qa/distractor/validation-00000-of-00001.parquet"
    else:
        path = "/data/huggingface/hotpotqa/hotpot_qa/fullwiki/validation-00000-of-00001.parquet"
    
    print(f"Loading queries from {path}...")
    df = pd.read_parquet(path)
    
    queries = []
    for idx, row in df.iterrows():
        titles = row["context"]["title"]
        gold_titles = set(row["supporting_facts"]["title"]) if "supporting_facts" in row else set()
        gold_indices = [i for i, t in enumerate(titles) if t in gold_titles]
        
        queries.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "gold_indices": gold_indices,
        })
    
    print(f"Loaded {len(queries)} queries")
    return queries


class OscarPoolingIndexer:
    """Index using OSCAR with mean pooling."""
    
    def __init__(self, oscar_model_name: str, device: str = "cuda:0"):
        self.device = device
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                mem_emb = self.oscar_model.compress_documents(batch)
                # Mean pooling over MEM tokens
                emb = mem_emb.mean(dim=1)
                emb = F.normalize(emb, p=2, dim=-1)
            all_embeds.append(emb.cpu().numpy())
        
        return np.vstack(all_embeds).astype(np.float32)


class OscarProjectorIndexer:
    """Index using OSCAR with trained projector."""
    
    def __init__(self, oscar_model_name: str, projector_path: str, device: str = "cuda:0"):
        self.device = device
        print(f"Loading OSCAR + Projector from {projector_path}")
        self.encoder = OscarProjectorEncoder(
            oscar_model_name=oscar_model_name,
            projector_path=projector_path,
            device=device,
        )
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                emb = self.encoder.encode(batch)
                emb = F.normalize(emb, p=2, dim=-1)
            all_embeds.append(emb.cpu().numpy())
        
        return np.vstack(all_embeds).astype(np.float32)


class BM25Indexer:
    """Index using BM25."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("Installing rank-bm25...")
            import subprocess
            subprocess.run(['pip', 'install', 'rank-bm25'], check=True)
            from rank_bm25 import BM25Okapi
        
        self.BM25Okapi = BM25Okapi
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = []
        self.doc_texts = []
    
    def fit(self, texts: List[str], doc_ids: List[str]):
        print(f"Fitting BM25 on {len(texts)} documents...")
        self.doc_ids = doc_ids
        self.doc_texts = [t.lower().split() for t in texts]
        self.bm25 = self.BM25Okapi(self.doc_texts)
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # BM25 returns scores, not embeddings - we need a different approach
        # For fair comparison, we'll use BM25 at query time
        raise NotImplementedError("BM25 doesn't produce dense embeddings")


class SFREmbeddingIndexer:
    """Index using Salesforce SFR Mistral embedding."""
    
    def __init__(self, model_name: str = "Salesforce/sfr-embedding-mistral", device: str = "cuda:0"):
        self.device = device
        print(f"Loading SFR embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    
    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                emb = F.normalize(emb, p=2, dim=-1)
            all_embeds.append(emb.cpu().numpy())
        
        return np.vstack(all_embeds).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, method: str = "flat") -> faiss.Index:
    """Build FAISS index from embeddings."""
    dim = embeddings.shape[1]
    print(f"Building FAISS index (dim={dim}, n_vectors={len(embeddings)})")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    if method == "flat":
        index = faiss.IndexFlatIP(dim)
    elif method == "ivf":
        nlist = min(100, len(embeddings) // 10)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors")
    return index


def main():
    parser = argparse.ArgumentParser(description="Index HotpotQA contexts")
    parser.add_argument("--mode", choices=["distractor", "fullwiki"], default="distractor")
    parser.add_argument("--method", required=True,
                        choices=["oscar_pooling", "oscar_projector", "bm25", "sfr_embedding"])
    parser.add_argument("--oscar-model", default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--projector-path", type=str, default=None,
                        help="Path to projector checkpoint (required for oscar_projector)")
    parser.add_argument("--output-dir", default="./data/hotpotqa_index")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Max number of documents to index")
    parser.add_argument("--device", default="cuda:0")
    
    args = parser.parse_args()
    
    # Load data
    contexts = load_hotpotqa(args.mode, args.max_docs)
    doc_ids = [c["id"] for c in contexts]
    doc_texts = [c["text"] for c in contexts]
    
    # Create indexer
    if args.method == "oscar_pooling":
        indexer = OscarPoolingIndexer(args.oscar_model, args.device)
        embeddings = indexer.encode(doc_texts)
    elif args.method == "oscar_projector":
        if not args.projector_path:
            raise ValueError("--projector-path required for oscar_projector")
        indexer = OscarProjectorIndexer(args.oscar_model, args.projector_path, args.device)
        embeddings = indexer.encode(doc_texts)
    elif args.method == "bm25":
        indexer = BM25Indexer()
        indexer.fit(doc_texts, doc_ids)
        # BM25 doesn't produce embeddings, skip FAISS
        print("BM25 index built (not a dense index)")
        # Save BM25 indexer state
        output_dir = Path(args.output_dir) / args.method / args.mode
        output_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(output_dir / "bm25_indexer.pkl", "wb") as f:
            pickle.dump(indexer, f)
        with open(output_dir / "doc_ids.json", "w") as f:
            json.dump(doc_ids, f)
        print(f"BM25 index saved to {output_dir}")
        return
    elif args.method == "sfr_embedding":
        indexer = SFREmbeddingIndexer(device=args.device)
        embeddings = indexer.encode(doc_texts)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Save
    output_dir = Path(args.output_dir) / args.method / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "doc_ids.json", "w") as f:
        json.dump(doc_ids, f)
    
    # Save metadata
    metadata = {
        "method": args.method,
        "mode": args.mode,
        "num_docs": len(doc_ids),
        "dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Index saved to {output_dir}")


if __name__ == "__main__":
    main()