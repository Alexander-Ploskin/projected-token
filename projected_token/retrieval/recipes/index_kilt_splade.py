#!/usr/bin/env python3
"""Build SPLADE v3 sparse CSR index for KILT passages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from projected_token.io import write_json


def _load_config_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})
    output_cfg = config.get("output", {})
    splade_cfg = config.get("splade", {})
    return {
        "dataset": dataset_cfg.get("name", "s-nlp/kilt"),
        "dataset_config": dataset_cfg.get("config"),
        "split": dataset_cfg.get("split", "train"),
        "text_col": dataset_cfg.get("text_col", "text"),
        "max_samples": runtime_cfg.get("max_samples"),
        "hf_cache_dir": runtime_cfg.get("hf_cache_dir"),
        "hf_token": runtime_cfg.get("hf_token"),
        "model_name_or_path": model_cfg.get("name", "naver/splade-v3"),
        "max_length": int(model_cfg.get("max_length", 256)),
        "batch_size": int(runtime_cfg.get("batch_size", 8)),
        "device": runtime_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"),
        "aggregation": splade_cfg.get("aggregation", "max"),
        "top_n_terms": splade_cfg.get("top_n_terms", 128),
        "output_dir": output_cfg.get("dir", "artifacts/indexes/kilt_splade_v3"),
        "prepare_cache_path": output_cfg.get("prepare_cache_path"),
    }


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML recipe")
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = _load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="Build SPLADE CSR index for KILT", parents=[pre_parser])
    parser.add_argument("--dataset", default=defaults.get("dataset", "s-nlp/kilt"))
    parser.add_argument("--dataset-config", default=defaults.get("dataset_config"))
    parser.add_argument("--split", default=defaults.get("split", "train"))
    parser.add_argument("--text-col", default=defaults.get("text_col", "text"))
    parser.add_argument("--max-samples", type=int, default=defaults.get("max_samples"))
    parser.add_argument("--hf-cache-dir", default=defaults.get("hf_cache_dir"))
    parser.add_argument("--hf-token", default=defaults.get("hf_token"))

    parser.add_argument("--model-name-or-path", default=defaults.get("model_name_or_path", "naver/splade-v3"))
    parser.add_argument("--max-length", type=int, default=int(defaults.get("max_length", 256)))
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 8)))
    parser.add_argument("--device", default=defaults.get("device", "cpu"))
    parser.add_argument("--aggregation", choices=["max", "sum"], default=defaults.get("aggregation", "max"))
    parser.add_argument("--top-n-terms", type=int, default=defaults.get("top_n_terms", 128))

    parser.add_argument("--output-dir", default=defaults.get("output_dir", "artifacts/indexes/kilt_splade_v3"))
    parser.add_argument("--prepare-cache-path", default=defaults.get("prepare_cache_path"))
    return parser.parse_args(remaining)


def _iter_texts(args: argparse.Namespace) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        cache_dir=args.hf_cache_dir,
        token=args.hf_token,
    )
    if args.max_samples is not None:
        ds = ds.select(range(min(int(args.max_samples), len(ds))))
    if args.text_col not in ds.column_names:
        raise ValueError(f"Column '{args.text_col}' not found in dataset columns: {ds.column_names}")
    out: list[str] = []
    for row in ds:
        value = row.get(args.text_col)
        text = value if isinstance(value, str) else str(value or "")
        out.append(text)
    return out


def _aggregate_sparse(logits: torch.Tensor, attention_mask: torch.Tensor, aggregation: str) -> torch.Tensor:
    values = torch.log1p(torch.relu(logits))
    mask = attention_mask.unsqueeze(-1).to(values.dtype)
    values = values * mask
    if aggregation == "sum":
        return values.sum(dim=1)
    return values.max(dim=1).values


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = _iter_texts(args)
    if not texts:
        raise ValueError("No texts found to index.")

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, token=args.hf_token)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, token=args.hf_token).to(device).eval()
    vocab_size = int(getattr(model.config, "vocab_size", tokenizer.vocab_size))
    batch_size = int(args.batch_size)
    max_length = int(args.max_length)
    top_n_terms = int(args.top_n_terms) if args.top_n_terms is not None else None

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="encode-passages-splade"):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            dense = _aggregate_sparse(logits, inputs["attention_mask"], str(args.aggregation)).cpu().float().numpy()

        for row in dense:
            if top_n_terms is not None and top_n_terms > 0 and row.size > top_n_terms:
                top_idx = np.argpartition(-row, top_n_terms - 1)[:top_n_terms]
                mask = np.zeros_like(row, dtype=bool)
                mask[top_idx] = True
                row = np.where(mask, row, 0.0)
            nz = np.flatnonzero(row > 0)
            if nz.size:
                indices.extend(int(i) for i in nz.tolist())
                data.extend(float(row[i]) for i in nz.tolist())
            indptr.append(len(indices))

    matrix = csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(texts), vocab_size),
        dtype=np.float32,
    )
    save_npz(output_dir / "corpus_csr.npz", matrix)
    np.save(output_dir / "doc_row_ids.npy", np.arange(len(texts), dtype=np.int64), allow_pickle=False)

    run_info = {
        "retrieval_backend": "splade_csr",
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_col": args.text_col,
        "max_samples": args.max_samples,
        "model_name_or_path": args.model_name_or_path,
        "max_length": max_length,
        "batch_size": batch_size,
        "aggregation": args.aggregation,
        "top_n_terms": top_n_terms,
        "num_docs": int(matrix.shape[0]),
        "vocab_size": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "prepare_cache_path": args.prepare_cache_path,
        "artifacts": {
            "corpus_csr": "corpus_csr.npz",
            "doc_row_ids": "doc_row_ids.npy",
        },
    }
    write_json(output_dir / "metadata.json", run_info)
    return run_info


def main() -> None:
    args = parse_args()
    payload = build_index(args)
    print("SPLADE CSR index build complete")
    print(f"Output dir: {args.output_dir}")
    print(f"Docs: {payload['num_docs']}")
    print(f"NNZ: {payload['nnz']}")


if __name__ == "__main__":
    main()
