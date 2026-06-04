#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import l2_normalize
from projected_token.retrieval.beir import _load_beir_dataset, _encode_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--compressor-device", type=str, default="cuda:0")
    parser.add_argument("--decoder-device", type=str, default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="artifacts/results/retrieval/beir_nq_e9_6000_shards")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OSCAR_COMPRESSOR_DEVICE"] = args.compressor_device
    os.environ["OSCAR_DECODER_DEVICE"] = args.decoder_device

    print(f"[Shard {args.shard_id}] Loading encoder...")
    encoder_cfg = {
        "name": "oscar_projector",
        "kwargs": {
            "oscar_model_name": "naver/oscar-qwen2-7B",
            "projector_path": "artifacts/query_distill_runs/e9-flatten-kl-4096/checkpoints/checkpoint_step_6000.pt",
            "device": args.compressor_device,
            "embed_dim": 768,
            "pooler": "flatten",
            "num_layers": 2,
            "dropout": 0.0,
        }
    }
    encoder = build_encoder(encoder_cfg)

    print(f"[Shard {args.shard_id}] Loading NQ dataset...")
    corpus_ids, corpus_texts, _, _, _ = _load_beir_dataset(Path("/data/beir/nq"), split="test")

    chunk_size = (len(corpus_ids) + args.num_shards - 1) // args.num_shards
    start_idx = args.shard_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(corpus_ids))
    
    shard_corpus_ids = corpus_ids[start_idx:end_idx]
    shard_corpus_texts = corpus_texts[start_idx:end_idx]

    print(f"[Shard {args.shard_id}] Encoding {len(shard_corpus_texts)} documents (indices {start_idx} to {end_idx})...")
    embeddings = _encode_batches(encoder, shard_corpus_texts, batch_size=args.batch_size)
    embeddings = l2_normalize(embeddings)

    np.save(out_dir / f"embeddings_{args.shard_id}.npy", embeddings)
    with open(out_dir / f"ids_{args.shard_id}.json", "w") as f:
        json.dump(shard_corpus_ids, f)
        
    print(f"[Shard {args.shard_id}] Done!")

if __name__ == "__main__":
    main()