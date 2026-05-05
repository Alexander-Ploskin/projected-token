#!/usr/bin/env python3
"""Generate teacher embeddings using SFR-Embedding-Mistral.

Saves embeddings to HDF5 file for offline distillation.
Can use:
- Tevatron/msmarco-passage-corpus (local)
- Tevatron/msmarco-doc-corpus (HuggingFace)
"""

import gzip
import os
import h5py
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


def last_token_pool(last_hidden_states, attention_mask):
    """Last token pooling as used by SFR-Embedding-Mistral."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def mean_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
    summed = (last_hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def load_local_corpus(corpus_path: str, max_samples: int = None):
    """Load corpus from local JSONL.gz file."""
    texts = []
    open_func = gzip.open if corpus_path.endswith('.gz') else open
    mode = 'rt' if corpus_path.endswith('.gz') else 'r'
    
    with open_func(corpus_path, mode, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            text = doc.get('title', '') + ' ' + doc.get('text', '')
            texts.append(text.strip())
            
            if max_samples and len(texts) >= max_samples:
                break
    
    return texts


def generate_teacher_embeddings(
    num_samples: int = 100_000,
    batch_size: int = 16,
    output_dir: str = "/data/teacher-embeddings",
    output_name: str = "teacher_embeddings_100k.h5",
    dataset_name: str = None,
    corpus_path: str = None,
    device: str = "cuda:0",
    teacher_model_name: str = "/data/huggingface/Salesforce/SFR-Embedding-Mistral",
    pooling: str = "auto",
    prompt_style: str = "auto",
):
    """Generate teacher embeddings and save to HDF5."""
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    
    # Prompt template used by SFR-Embedding-Mistral
    doc_task_def = "Given a web search query, retrieve relevant passages that answer the query"
    
    # Load texts from local corpus or HuggingFace
    if corpus_path:
        print(f"Loading corpus from local file: {corpus_path}")
        texts = load_local_corpus(corpus_path, max_samples=num_samples)
        num_samples = len(texts)
        print(f"Loaded {num_samples} documents")
    else:
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
        texts = dataset["text"]
        print(f"Dataset size: {len(texts)}")
    
    print(f"Loading teacher model: {teacher_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    model = AutoModel.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    print("Teacher model loaded")
    
    print(f"Generating embeddings for {num_samples} samples...")
    
    with h5py.File(output_path, 'w') as f:
        emb_dataset = None
        text_dataset = f.create_dataset('texts', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))

        lower_name = teacher_model_name.lower()
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_texts = texts[i:i+batch_size]

            if prompt_style == "auto":
                if "sfr-embedding-mistral" in lower_name:
                    batch_inputs = [f"Instruct: {doc_task_def}\nQuery: {text}" for text in batch_texts]
                elif "e5" in lower_name:
                    batch_inputs = [f"passage: {text}" for text in batch_texts]
                else:
                    batch_inputs = batch_texts
            elif prompt_style == "sfr":
                batch_inputs = [f"Instruct: {doc_task_def}\nQuery: {text}" for text in batch_texts]
            elif prompt_style == "e5":
                batch_inputs = [f"passage: {text}" for text in batch_texts]
            else:
                batch_inputs = batch_texts

            inputs = tokenizer(
                batch_inputs,
                max_length=4096,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.inference_mode():
                outputs = model(**inputs)
                if pooling == "auto":
                    resolved_pooling = "last_token" if "sfr-embedding-mistral" in lower_name else "mean"
                else:
                    resolved_pooling = pooling
                if resolved_pooling == "last_token":
                    embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                elif resolved_pooling == "cls":
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    embeddings = mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)

            if emb_dataset is None:
                emb_dim = int(embeddings.shape[-1])
                emb_dataset = f.create_dataset('embeddings', shape=(num_samples, emb_dim), dtype='float32')
                f.attrs["teacher_model_name"] = teacher_model_name
                f.attrs["pooling"] = resolved_pooling
                f.attrs["prompt_style"] = prompt_style

            emb_dataset[i:i+batch_size] = embeddings.cpu().float().numpy()
            text_dataset[i:i+batch_size] = batch_texts
    
    with h5py.File(output_path, 'r') as f:
        emb_shape = f["embeddings"].shape
    print(f"Saved embeddings to: {output_path}")
    print(f"Shape: {emb_shape[0]} x {emb_shape[1]}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate teacher embeddings")
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="/data/teacher-embeddings")
    parser.add_argument("--output-name", type=str, default="teacher_embeddings_100k.h5")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="HuggingFace dataset name (e.g., Tevatron/msmarco-doc-corpus)")
    parser.add_argument("--corpus-path", type=str, default=None,
                        help="Local corpus path (e.g., /data/huggingface/Tevatron/msmarco-passage-corpus/corpus.jsonl.gz)")
    parser.add_argument("--teacher-model-name", type=str, default="/data/huggingface/Salesforce/SFR-Embedding-Mistral")
    parser.add_argument("--pooling", type=str, default="auto", choices=["auto", "last_token", "mean", "cls"])
    parser.add_argument("--prompt-style", type=str, default="auto", choices=["auto", "none", "sfr", "e5"])
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    if args.corpus_path:
        print(f"Using local corpus: {args.corpus_path}")
    elif args.dataset_name:
        print(f"Using HuggingFace dataset: {args.dataset_name}")
    else:
        print("Please specify either --corpus-path or --dataset-name")
        return
    
    generate_teacher_embeddings(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        output_name=args.output_name,
        dataset_name=args.dataset_name,
        corpus_path=args.corpus_path,
        teacher_model_name=args.teacher_model_name,
        pooling=args.pooling,
        prompt_style=args.prompt_style,
        device=args.device,
    )


if __name__ == "__main__":
    main()