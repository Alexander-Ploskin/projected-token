#!/usr/bin/env python3
"""Encode query/doc pairs from a BGE teacher H5 with OSCAR+E11 projector (same schema)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from projected_token.data.recipes.generate_query_doc_teacher_embeddings import write_query_doc_teacher_h5
from projected_token.retrieval.encoders import build_encoder


def _decode_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_records_from_teacher_h5(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with h5py.File(path, "r") as h5:
        n = int(h5["queries"].shape[0])
        records: list[dict[str, Any]] = []
        for i in range(n):
            records.append(
                {
                    "query": _decode_str(h5["queries"][i]),
                    "positive_doc": _decode_str(h5["positive_docs"][i]),
                    "negative_doc": _decode_str(h5["negative_docs"][i]),
                    "sample_id": _decode_str(h5["sample_ids"][i]),
                    "source": _decode_str(h5["sources"][i]),
                    "query_id": _decode_str(h5["query_ids"][i]),
                    "positive_doc_id": _decode_str(h5["positive_doc_ids"][i]),
                    "negative_doc_id": _decode_str(h5["negative_doc_ids"][i]),
                    "negative_rank": int(h5["negative_ranks"][i]),
                    "negative_bm25_rank": int(h5["negative_bm25_ranks"][i]),
                    "negative_dense_rank": int(h5["negative_dense_ranks"][i]),
                    "negative_bm25_score": float(h5["negative_bm25_scores"][i]),
                    "negative_dense_score": float(h5["negative_dense_scores"][i]),
                    "negative_ce_score": float(h5["negative_ce_scores"][i]),
                    "positive_score": float(h5["positive_scores"][i]),
                    "negative_strategy": _decode_str(h5["negative_strategies"][i]),
                }
            )
        attrs = dict(h5.attrs)
    summary_path = path.with_suffix(".summary.json")
    mining_stats: dict[str, Any] = {"data_source": str(path)}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mining_stats = dict(summary.get("mining_stats", mining_stats))
    return records, attrs


def encode_normalized(encoder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = len(texts)
    total_batches = max(1, (total + batch_size - 1) // batch_size)
    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        batch = texts[start : start + batch_size]
        emb = encoder.encode(batch, None)
        if hasattr(emb, "detach"):
            emb = F.normalize(emb.float(), dim=-1).detach().cpu().numpy()
        else:
            emb = F.normalize(torch.tensor(emb).float(), dim=-1).numpy()
        chunks.append(np.asarray(emb, dtype=np.float32))
        processed = min(start + len(batch), total)
        print(
            f"[oscar-embed] {processed}/{total} texts ({batch_idx}/{total_batches} batches)",
            flush=True,
        )
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="OSCAR projector embeddings for BGE teacher H5 pairs")
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--projector-path",
        type=Path,
        default=Path("artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt"),
    )
    parser.add_argument("--oscar-model", default="naver/oscar-qwen2-7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--compressor-device", default=None)
    parser.add_argument("--decoder-device", default=None)
    args = parser.parse_args()

    if args.compressor_device:
        os.environ["OSCAR_COMPRESSOR_DEVICE"] = args.compressor_device
    if args.decoder_device:
        os.environ["OSCAR_DECODER_DEVICE"] = args.decoder_device

    records, source_attrs = load_records_from_teacher_h5(args.input_h5)
    print(f"[oscar-embed] loaded {len(records)} records from {args.input_h5}", flush=True)

    encoder = build_encoder(
        {
            "name": "oscar_projector",
            "kwargs": {
                "oscar_model_name": args.oscar_model,
                "projector_path": str(args.projector_path),
                "device": args.device,
                "embed_dim": 768,
                "pooler": "per_token_gated",
                "num_layers": 2,
                "dropout": 0.0,
                "projector_hidden_dim": 1024,
            },
        }
    )

    queries = [r["query"] for r in records]
    positives = [r["positive_doc"] for r in records]
    negatives = [r["negative_doc"] for r in records]

    print("[oscar-embed] encoding queries...", flush=True)
    query_embeddings = encode_normalized(encoder, queries, args.batch_size)
    print("[oscar-embed] encoding positives...", flush=True)
    positive_embeddings = encode_normalized(encoder, positives, args.batch_size)
    print("[oscar-embed] encoding negatives...", flush=True)
    negative_embeddings = encode_normalized(encoder, negatives, args.batch_size)

    model_label = f"oscar-e11:{args.projector_path.name}"
    summary = write_query_doc_teacher_h5(
        output_path=args.output_path,
        records=records,
        embeddings={
            "query_embeddings": query_embeddings,
            "positive_embeddings": positive_embeddings,
            "negative_embeddings": negative_embeddings,
        },
        model_name=model_label,
        requested_max_samples=len(records),
        requested_negative_strategy=str(source_attrs.get("negative_strategy", "unknown")),
        bm25_rank_start=int(source_attrs.get("bm25_rank_start", 10)),
        bm25_rank_end=int(source_attrs.get("bm25_rank_end", 50)),
        dense_rank_start=int(source_attrs.get("dense_rank_start", 10)),
        dense_rank_end=int(source_attrs.get("dense_rank_end", 50)),
        score_band_min_margin=None,
        score_band_max_margin=None,
        cross_encoder_model_name=None,
        cross_encoder_threshold=0.5,
        cross_encoder_batch_size=32,
        mining_stats={"source_h5": str(args.input_h5), "encoder": model_label},
    )
    with h5py.File(args.output_path, "a") as h5:
        h5.attrs["encoder_type"] = "oscar_projector"
        h5.attrs["oscar_model_name"] = args.oscar_model
        h5.attrs["projector_path"] = str(args.projector_path)
        h5.attrs["source_teacher_h5"] = str(args.input_h5)
        h5.attrs["query_prefix"] = ""
        h5.attrs["document_prefix"] = ""
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
