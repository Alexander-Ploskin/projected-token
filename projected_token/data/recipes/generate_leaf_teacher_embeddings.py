#!/usr/bin/env python3
"""Generate BGE teacher embeddings for LEAF JSONL shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BGE_DOCUMENT_PREFIX = "Represent this sentence: "


def default_output_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "teacher-embeddings" / "leaf-bge-base-en-v1.5"
    return Path("data/teacher-embeddings/leaf-bge-base-en-v1.5")


def read_jsonl_shard(path: Path, *, limit: int | None = None) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            text = str(row.get("text", "")).strip()
            row_id = str(row.get("id", f"{path.stem}:{len(ids)}"))
            if not text:
                continue
            ids.append(row_id)
            texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return ids, texts


def write_h5(path: Path, *, ids: list[str], texts: list[str], embeddings: np.ndarray, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("embeddings", data=embeddings.astype(np.float32))
        h5.create_dataset("ids", data=np.array(ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("sample_ids", data=np.array(ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("texts", data=np.array(texts, dtype=object), dtype=string_dtype)
        h5.attrs["teacher_model_name"] = model_name
        h5.attrs["pooling"] = "sentence_transformers"
        h5.attrs["prompt_style"] = "bge_document_prefix"
        h5.attrs["normalized"] = True


def validate_h5(path: Path, *, expected_dim: int = 768) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        required = {"embeddings", "ids", "texts"}
        missing = sorted(required.difference(h5.keys()))
        if missing:
            raise ValueError(f"{path} missing datasets: {missing}")
        embeddings_shape = tuple(h5["embeddings"].shape)
        ids_shape = tuple(h5["ids"].shape)
        texts_shape = tuple(h5["texts"].shape)
        if len(embeddings_shape) != 2:
            raise ValueError(f"{path} embeddings must be 2D, got {embeddings_shape}")
        if embeddings_shape[1] != expected_dim:
            raise ValueError(f"{path} embeddings dim must be {expected_dim}, got {embeddings_shape[1]}")
        if embeddings_shape[0] == 0:
            raise ValueError(f"{path} contains zero embeddings")
        if ids_shape[0] != embeddings_shape[0] or texts_shape[0] != embeddings_shape[0]:
            raise ValueError(f"{path} dataset lengths differ: embeddings={embeddings_shape}, ids={ids_shape}, texts={texts_shape}")
        return {"path": str(path), "embeddings": embeddings_shape, "ids": ids_shape, "texts": texts_shape}


def generate_leaf_teacher_embeddings(
    *,
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    max_seq_length: int,
    max_shards: int | None,
    shard_rank: int,
    num_shard_workers: int,
    limit_records_per_shard: int | None,
    overwrite: bool,
    validate_only: bool,
) -> list[dict[str, Any]]:
    shard_paths = sorted(input_dir.glob("shard_*.jsonl"))
    if num_shard_workers < 1:
        raise ValueError("--num-shard-workers must be >= 1")
    if shard_rank < 0 or shard_rank >= num_shard_workers:
        raise ValueError("--shard-rank must satisfy 0 <= rank < num_shard_workers")
    if num_shard_workers > 1:
        shard_paths = [path for idx, path in enumerate(shard_paths) if idx % num_shard_workers == shard_rank]
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.jsonl files found in {input_dir}")

    if validate_only:
        return [validate_h5(output_dir / f"{path.stem}.h5") for path in shard_paths]

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    summaries: list[dict[str, Any]] = []

    for shard_path in shard_paths:
        out_path = output_dir / f"{shard_path.stem}.h5"
        if out_path.exists() and not overwrite:
            print(f"[leaf-teacher] skip existing {out_path}", flush=True)
            summaries.append(validate_h5(out_path))
            continue

        ids, texts = read_jsonl_shard(shard_path, limit=limit_records_per_shard)
        if not texts:
            print(f"[leaf-teacher] skip empty shard {shard_path}", flush=True)
            continue
        prefixed = [BGE_DOCUMENT_PREFIX + text for text in texts]
        embeddings = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        write_h5(out_path, ids=ids, texts=texts, embeddings=np.asarray(embeddings), model_name=model_name)
        summary = validate_h5(out_path)
        summaries.append(summary)
        print(f"[leaf-teacher] wrote {out_path}: {summary['embeddings']}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BGE teacher embeddings for LEAF corpus shards")
    parser.add_argument("--input-dir", type=Path, default=Path("data/leaf_corpus"))
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--shard-rank", type=int, default=0, help="Worker rank for modulo sharding")
    parser.add_argument("--num-shard-workers", type=int, default=1, help="Total modulo shard workers")
    parser.add_argument("--limit-records-per-shard", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    summaries = generate_leaf_teacher_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_shards=args.max_shards,
        shard_rank=args.shard_rank,
        num_shard_workers=args.num_shard_workers,
        limit_records_per_shard=args.limit_records_per_shard,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
