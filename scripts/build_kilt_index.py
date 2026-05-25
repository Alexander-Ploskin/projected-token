#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import faiss

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import l2_normalize

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for KILT using OSCAR E9")
    parser.add_argument("--output-dir", type=str, default="/mnt/raid/a-ploskin/kilt_faiss_indexes", help="Directory to save the index")
    parser.add_argument("--shard-id", type=int, required=True, help="Shard ID (0-indexed)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--compressor-device", type=str, default="cuda:0", help="Device for compressor and projector")
    parser.add_argument("--decoder-device", type=str, default="cuda:1", help="Device for decoder")
    parser.add_argument("--projector-path", type=str, default="artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt", help="Path to projector checkpoint")
    parser.add_argument("--pooler", type=str, default=None, help="Pooler override (defaults to checkpoint config)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print(f"[KILT Indexer] Output directory: {out_dir}", flush=True)
    print(f"[KILT Indexer] Processing shard {args.shard_id} of {args.num_shards} (total shards)", flush=True)

    # Set env vars for OSCAR component placement
    os.environ["OSCAR_COMPRESSOR_DEVICE"] = args.compressor_device
    os.environ["OSCAR_DECODER_DEVICE"] = args.decoder_device

    print(f"Loading OSCAR model...")
    print(f"Compressor/Projector device: {args.compressor_device}")
    print(f"Decoder device: {args.decoder_device}")
    
    encoder_cfg = {
        "name": "oscar_projector",
        "kwargs": {
            "oscar_model_name": "naver/oscar-qwen2-7B",
            "projector_path": args.projector_path,
            "device": args.compressor_device,
            "embed_dim": 768,
            "pooler": args.pooler or "flatten",
            "num_layers": 2,
            "dropout": 0.0,
        }
    }
    encoder = build_encoder(encoder_cfg)

    print(f"[KILT Indexer] Loading s-nlp/kilt dataset (shard {args.shard_id}/{args.num_shards})...", flush=True)
    ds = load_dataset("s-nlp/kilt", split="train")
    ds = ds.shard(num_shards=args.num_shards, index=args.shard_id)
    
    print(f"[KILT Indexer] Shard contains {len(ds)} documents.", flush=True)

    index = faiss.IndexFlatIP(768)
    doc_ids = []

    batch_texts = []
    batch_ids = []

    def process_batch(texts, ids):
        if not texts:
            return
        embeddings = encoder.encode(texts, None)
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = l2_normalize(embeddings)
        index.add(embeddings)
        doc_ids.extend(ids)

    for i in tqdm(range(len(ds)), desc="Encoding"):
        row = ds[i]
        title = str(row.get("wikipedia_title", "")).strip()
        text = str(row.get("text", "")).strip()
        doc_id = str(row.get("_id", ""))
        
        full_text = f"{title} {text}".strip()
        batch_texts.append(full_text)
        batch_ids.append(doc_id)

        if len(batch_texts) >= args.batch_size:
            process_batch(batch_texts, batch_ids)
            batch_texts = []
            batch_ids = []

    if batch_texts:
        process_batch(batch_texts, batch_ids)

    faiss_path = out_dir / f"kilt_shard_{args.shard_id}.faiss"
    ids_path = out_dir / f"kilt_shard_{args.shard_id}_ids.json"

    print(f"Saving FAISS index to {faiss_path}...")
    faiss.write_index(index, str(faiss_path))
    
    print(f"Saving document IDs to {ids_path}...")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    print("Done!")

if __name__ == "__main__":
    main()
