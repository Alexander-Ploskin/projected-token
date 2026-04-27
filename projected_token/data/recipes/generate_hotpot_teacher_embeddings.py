#!/usr/bin/env python3
"""Generate teacher embeddings for HotpotQA distractor dataset.

This generates SFR embeddings for:
1. Questions (as queries)
2. Context documents (title + sentences combined)

The dataset is split into train (98%) and validation (2%).
"""

import argparse
import h5py
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

from transformers import AutoModel


def load_hotpotqa(path: str) -> List[Dict[str, Any]]:
    """Load HotpotQA distractor dataset."""
    dfs = []
    for p in Path(path).glob("*.parquet"):
        df = pd.read_parquet(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def create_documents_from_context(context: Dict) -> List[str]:
    """Create document texts from context.
    
    Each title + its sentences becomes one document.
    """
    documents = []
    titles = context.get('title', [])
    sentences = context.get('sentences', [])
    
    for title, sent_list in zip(titles, sentences):
        # Handle numpy arrays
        sent_list = np.array(sent_list) if not isinstance(sent_list, list) else sent_list
        if len(sent_list) > 0:
            doc = f"{title}: {' '.join(sent_list)}"
            documents.append(doc)
    
    return documents


def encode_texts_sfr(texts: List[str], model_name: str = "/data/huggingface/Salesforce/SFR-Embedding-Mistral") -> np.ndarray:
    """Encode texts using SFR-Embedding-Mistral."""
    model = SentenceTransformer(model_name, device="cuda")
    model.eval()
    
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    return embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate teacher embeddings for HotpotQA")
    
    parser.add_argument("--data-path", default="/data/huggingface/hotpotqa/hotpot_qa/distractor")
    parser.add_argument("--output-path", default="/data/teacher-embeddings/hotpotqa_teacher_embeddings.h5")
    parser.add_argument("--max-train", type=int, default=50000, help="Max train samples")
    parser.add_argument("--max-context-docs", type=int, default=5, help="Max documents per context")
    parser.add_argument("--val-split", type=float, default=0.02)
    
    args = parser.parse_args()
    
    print(f"Loading HotpotQA from {args.data_path}...")
    df = load_hotpotqa(args.data_path)
    print(f"Loaded {len(df)} records")
    
    # Take first max_train samples for training
    if args.max_train and len(df) > args.max_train:
        df = df.iloc[:args.max_train].reset_index(drop=True)
        print(f"Using first {len(df)} samples")
    
    # Split into train and validation
    val_size = int(len(df) * args.val_split)
    train_size = len(df) - val_size
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")
    
    # Prepare data
    all_data = []
    
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        print(f"\n=== Processing {split_name} ({len(split_df)} samples) ===")
        
        # Get all questions
        print("Extracting questions...")
        questions = split_df['question'].tolist()
        
        # Get all context documents
        print("Extracting context documents...")
        context_docs = []
        for i, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Preparing docs"):
            docs = create_documents_from_context(row['context'])
            # Limit to max_context_docs
            context_docs.extend(docs[:args.max_context_docs])
        
        unique_context_docs = list(set(context_docs))
        print(f"Unique context documents: {len(unique_context_docs)}")
        
        # Encode questions
        print(f"Encoding {len(questions)} questions...")
        question_embeddings = encode_texts_sfr(questions)
        
        # Encode context documents
        print(f"Encoding {len(unique_context_docs)} context documents...")
        context_embeddings = encode_texts_sfr(unique_context_docs)
        
        # Create mapping from doc text to embedding
        doc_to_embed = {doc: emb for doc, emb in zip(unique_context_docs, context_embeddings)}
        
        # Build final dataset
        split_data = {
            "split": split_name,
            "questions": questions,
            "question_embeddings": question_embeddings,
            "context_docs": unique_context_docs,
            "context_embeddings": context_embeddings,
            "doc_to_idx": {doc: i for i, doc in enumerate(unique_context_docs)},
        }
        
        all_data.append(split_data)
    
    # Save to HDF5
    print(f"\nSaving to {args.output_path}...")
    
    with h5py.File(args.output_path, 'w') as f:
        for split_data in all_data:
            split_name = split_data["split"]
            grp = f.create_group(split_name)
            
            # Questions
            grp.create_dataset("questions", data=[q.encode('utf-8') for q in split_data["questions"]])
            grp.create_dataset("question_embeddings", data=split_data["question_embeddings"])
            
            # Context documents
            grp.create_dataset("context_docs", data=[d.encode('utf-8') for d in split_data["context_docs"]])
            grp.create_dataset("context_embeddings", data=split_data["context_embeddings"])
    
    # Save metadata
    metadata_path = args.output_path.replace(".h5", "_metadata.json")
    metadata = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "num_train_questions": len(train_df),
        "num_val_questions": len(val_df),
        "num_train_docs": len(all_data[0]["context_docs"]),
        "num_val_docs": len(all_data[1]["context_docs"]),
        "embedding_dim": 4096,
        "model": "/data/huggingface/Salesforce/SFR-Embedding-Mistral",
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n=== Done ===")
    print(f"Output: {args.output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Train questions: {len(train_df)}")
    print(f"Val questions: {len(val_df)}")
    print(f"Train docs: {len(all_data[0]['context_docs'])}")
    print(f"Val docs: {len(all_data[1]['context_docs'])}")


if __name__ == "__main__":
    main()