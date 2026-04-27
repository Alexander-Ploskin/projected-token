#!/usr/bin/env python3
"""Build PopQA index using DistillationProjector."""

import argparse
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from tqdm import tqdm
import torch
from transformers import AutoModel

import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from projected_token.encoders.projector import DistillationProjector


class DistillationEncoder:
    def __init__(
        self,
        oscar_model_name: str,
        projector_path: str,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()
        
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False
        
        hidden_size = self.oscar_model.compress_documents(['test']).shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")
        
        print(f"Loading projector from {projector_path}")
        checkpoint = torch.load(projector_path, weights_only=False, map_location=self.device)
        config = checkpoint.get("config", {})
        
        self.projector = DistillationProjector(
            oscar_hidden_dim=hidden_size,
            embed_dim=config.get("embed_dim", 4096),
            hidden_dim=config.get("projector_hidden_dim", 8192),
            use_normalize=config.get("use_normalize", True),
        ).to(self.device, dtype=torch.float32)
        
        state_dict = checkpoint["model_state_dict"]
        for key in state_dict:
            state_dict[key] = state_dict[key].float()
        self.projector.load_state_dict(state_dict)
        self.projector.eval()
        
        self.embed_dim = config.get("embed_dim", 4096)
        print(f"Distillation encoder ready. Embed dim: {self.embed_dim}")
    
    def encode(self, documents: List[str]) -> np.ndarray:
        valid_docs = [doc for doc in documents if doc and doc.strip()]
        
        if not valid_docs:
            return np.zeros((len(documents), self.embed_dim), dtype=np.float32)
        
        with torch.no_grad():
            mem_embeddings = self.oscar_model.compress_documents(documents=valid_docs)
            mem_embeddings = mem_embeddings.float()
            embeddings = self.projector(mem_embeddings)
            embeddings = embeddings.cpu().numpy()
        
        result = np.zeros((len(documents), self.embed_dim), dtype=np.float32)
        valid_indices = [i for i, doc in enumerate(documents) if doc and doc.strip()]
        
        for i, idx in enumerate(valid_indices):
            if i < len(embeddings):
                result[idx] = embeddings[i]
        
        return result


def load_popqa(path: str) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    pydict = table.to_pydict()
    num_rows = table.num_rows
    return [{col: row[i] for col, row in pydict.items()} for i in range(num_rows)]


def main():
    parser = argparse.ArgumentParser(description="Build PopQA index with DistillationProjector")
    
    parser.add_argument("--dataset-path", default="/data/popqa_enriched.parquet")
    parser.add_argument("--oscar-model", default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--output-dir", default="/data/popqa_index_distillation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    
    args = parser.parse_args()
    
    print(f"Loading PopQA from {args.dataset_path}...")
    dataset = load_popqa(args.dataset_path)
    print(f"Loaded {len(dataset)} records")
    
    print("\nInitializing DistillationEncoder...")
    encoder = DistillationEncoder(
        oscar_model_name=args.oscar_model,
        projector_path=args.projector_path,
        device=args.device,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    texts = []
    ids = []
    for i, item in enumerate(dataset):
        text = item.get("s_wiki_content", "")
        if text and text.strip():
            texts.append(text)
            ids.append(i)
    
    print(f"Found {len(texts)} documents with s_wiki_content")
    
    print(f"\nEncoding {len(texts)} documents...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Encoding"):
        batch = texts[i:i + args.batch_size]
        embeddings = encoder.encode(batch)
        all_embeddings.append(embeddings)
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms
    
    print("\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    print(f"Index size: {index.ntotal}")
    
    print("\nSaving index...")
    faiss.write_index(index, str(output_dir / "index"))
    
    with open(output_dir / "ids.json", "w") as f:
        json.dump(ids, f)
    
    metadata = {
        "valid_indices": ids,
        "text_col": "text",
        "aggregation": "flatten",
        "encoder": "distillation_projector",
        "embed_dim": encoder.embed_dim,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    print(f"\nSaved to {output_dir}")
    print(f"  - index")
    print(f"  - ids.json ({len(ids)} ids)")
    print(f"  - metadata.json")


if __name__ == "__main__":
    main()