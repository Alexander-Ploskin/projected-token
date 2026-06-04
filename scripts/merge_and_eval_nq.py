#!/usr/bin/env python3
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.artifacts import write_metrics_bundle
from projected_token.io import write_json
from projected_token.oscar_runtime import (
    configure_oscar_component_devices,
    disable_resume_download_passthrough,
    disable_transformers_allocator_warmup,
)
from projected_token.retrieval.beir import _encode_query_batches, _load_beir_dataset
from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import create_faiss_index, l2_normalize
from projected_token.retrieval.metrics.ranking import aggregate_rankings


def load_oscar_split(compressor_device: str, decoder_device: str):
    """Load OSCAR on CPU first, then place compressor/decoder on separate GPUs."""
    disable_transformers_allocator_warmup()
    disable_resume_download_passthrough()

    os.environ["OSCAR_COMPRESSOR_DEVICE"] = compressor_device
    os.environ["OSCAR_DECODER_DEVICE"] = decoder_device

    print(
        f"Loading OSCAR on CPU, then placing "
        f"compressor={compressor_device}, decoder={decoder_device}..."
    )
    oscar_model = AutoModel.from_pretrained(
        "naver/oscar-qwen2-7B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    configure_oscar_component_devices(
        oscar_model,
        compressor_device=compressor_device,
        decoder_device=decoder_device,
    )
    return oscar_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--shards-dir",
        type=str,
        default="artifacts/results/retrieval/beir_nq_e9_6000_shards",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/results/retrieval/beir_nq_e9_6000",
    )
    parser.add_argument("--compressor-device", type=str, default="cuda:0")
    parser.add_argument("--decoder-device", type=str, default="cuda:1")
    parser.add_argument("--query-batch-size", type=int, default=256)
    args = parser.parse_args()

    os.environ["OSCAR_COMPRESSOR_DEVICE"] = args.compressor_device
    os.environ["OSCAR_DECODER_DEVICE"] = args.decoder_device

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading shards...")
    all_embeddings = []
    all_ids = []
    for i in range(args.num_shards):
        print(f"Loading shard {i}...")
        emb = np.load(shards_dir / f"embeddings_{i}.npy")
        with open(shards_dir / f"ids_{i}.json", "r", encoding="utf-8") as f:
            ids = json.load(f)
        all_embeddings.append(emb)
        all_ids.extend(ids)

    corpus_embeddings = np.concatenate(all_embeddings, axis=0)
    del all_embeddings
    print(f"Total corpus size: {len(all_ids)}")

    print("Building FAISS index...")
    index = create_faiss_index(corpus_embeddings, metric="ip")
    del corpus_embeddings
    gc.collect()

    print("Loading queries...")
    _, _, query_ids, query_texts, relevant_map = _load_beir_dataset(
        Path("/data/beir/nq"), split="test"
    )

    oscar_model = load_oscar_split(args.compressor_device, args.decoder_device)

    print("Loading encoder for queries...")
    encoder_cfg = {
        "name": "oscar_projector",
        "kwargs": {
            "oscar_model_name": "naver/oscar-qwen2-7B",
            "projector_path": (
                "artifacts/query_distill_runs/e9-flatten-kl-4096/checkpoints/"
                "checkpoint_step_6000.pt"
            ),
            "device": args.compressor_device,
            "embed_dim": 768,
            "pooler": "flatten",
            "num_layers": 2,
            "dropout": 0.0,
            "oscar_model_instance": oscar_model,
        },
    }
    encoder = build_encoder(encoder_cfg)

    print("Encoding queries...")
    query_embeddings = _encode_query_batches(
        encoder, query_texts, batch_size=args.query_batch_size
    )
    query_embeddings = l2_normalize(query_embeddings)

    print("Searching...")
    search_k = 100
    _, indices = index.search(query_embeddings.astype(np.float32), search_k)

    ranking_cases = []
    for idx, qid in enumerate(query_ids):
        relevant = relevant_map.get(qid, set())
        if not relevant:
            continue
        retrieved = [
            all_ids[doc_idx]
            for doc_idx in indices[idx].tolist()
            if 0 <= doc_idx < len(all_ids)
        ]
        ranking_cases.append((relevant, retrieved))

    print("Aggregating metrics...")
    top_k = [1, 3, 5, 10, 20]
    metrics = aggregate_rankings(ranking_cases, top_k)

    print(
        f"Results: NDCG@10={metrics.get('ndcg@10', 0.0):.4f}, "
        f"MRR@10={metrics.get('mrr@10', 0.0):.4f}"
    )

    summary = {
        "run_id": "beir_nq_e9_6000",
        "split": "test",
        "datasets": ["nq"],
        "top_k": top_k,
        "search_k": search_k,
        "compressor_device": args.compressor_device,
        "decoder_device": args.decoder_device,
        "per_dataset": {"nq": metrics},
        "average": metrics,
    }

    write_json(out_dir / "summary.json", summary)
    write_metrics_bundle(
        metrics,
        run_id="beir_nq_e9_6000",
        dataset="nq",
        split="test",
        json_path=out_dir / "nq_metrics.json",
        csv_path=out_dir / "nq_metrics.csv",
    )
    print("Done!")


if __name__ == "__main__":
    main()
