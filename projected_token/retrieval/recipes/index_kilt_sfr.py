#!/usr/bin/env python3
"""Index s-nlp/kilt into FAISS using Salesforce/SFR-Embedding-Mistral.

This recipe is optimized for using up to two GPUs by sharding corpus encoding
across worker processes and then merging vectors back into the original order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from projected_token.io import write_json, write_jsonl
from projected_token.retrieval.index.vector import save_faiss_index


def parse_devices(raw_devices: str | None) -> list[str]:
    if raw_devices:
        parsed = [item.strip() for item in raw_devices.split(",") if item.strip()]
    elif torch.cuda.is_available():
        parsed = [f"cuda:{idx}" for idx in range(min(torch.cuda.device_count(), 2))]
    else:
        parsed = ["cpu"]

    if not parsed:
        return ["cpu"]

    if len(parsed) > 2:
        print(f"Received {len(parsed)} devices, using first two only: {parsed[:2]}")
        parsed = parsed[:2]

    if torch.cuda.is_available():
        max_idx = torch.cuda.device_count() - 1
        for device in parsed:
            if device.startswith("cuda:"):
                idx = int(device.split(":", maxsplit=1)[1])
                if idx > max_idx:
                    raise ValueError(
                        f"Requested device '{device}', but only {torch.cuda.device_count()} CUDA device(s) available."
                    )
    return parsed


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _encode_cache_key(args: argparse.Namespace, total_texts: int) -> str:
    payload = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_samples": args.max_samples,
        "model_name_or_path": args.model_name_or_path,
        "max_length": int(args.max_length),
        "passage_prefix": args.passage_prefix,
        "attn_implementation": args.attn_implementation,
        "allow_attn_fallback": bool(args.allow_attn_fallback),
        "total_texts": int(total_texts),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return digest[:16]


def _resolve_encode_cache_dir(args: argparse.Namespace, total_texts: int) -> Path:
    if args.encode_cache_dir:
        return Path(args.encode_cache_dir)
    key = _encode_cache_key(args, total_texts)
    return Path("artifacts/cache/kilt_encode") / key


def _encode_worker(
    shard_id: int,
    device: str,
    texts: list[str],
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    prefix: str,
    attn_implementation: str,
    allow_attn_fallback: bool,
    normalize_embeddings: bool,
    shard_cache_dir: str,
    resume_enabled: bool,
    cache_signature: str,
    queue: mp.Queue,
) -> None:
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        shard_dir = Path(shard_cache_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = shard_dir / f"shard_{shard_id}_embeddings.npy"
        progress_path = shard_dir / f"shard_{shard_id}_progress.json"
        meta_path = shard_dir / f"shard_{shard_id}_meta.json"

        processed = 0
        if resume_enabled and embeddings_path.exists() and progress_path.exists() and meta_path.exists():
            progress = _read_json(progress_path)
            meta = _read_json(meta_path)
            if (
                str(meta.get("cache_signature", "")) == cache_signature
                and int(meta.get("rows", -1)) == len(texts)
            ):
                processed = int(progress.get("processed", 0))
                if processed < 0 or processed > len(texts):
                    processed = 0
            else:
                print(f"[shard {shard_id}] Cache signature mismatch, rebuilding shard cache.")
        elif not resume_enabled and embeddings_path.exists():
            embeddings_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
        }
        requested_attn = attn_implementation
        if requested_attn != "auto":
            model_kwargs["attn_implementation"] = requested_attn

        try:
            model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs).to(device).eval()
            resolved_attn = requested_attn
        except Exception as exc:
            if not allow_attn_fallback or requested_attn in {"auto", "sdpa", "eager"}:
                raise
            fallback_attn = "sdpa"
            print(
                f"[shard {shard_id}] Failed to load with attn={requested_attn} on {device}: "
                f"{exc.__class__.__name__}: {exc}. Falling back to {fallback_attn}."
            )
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["attn_implementation"] = fallback_attn
            model = AutoModel.from_pretrained(model_name_or_path, **fallback_kwargs).to(device).eval()
            resolved_attn = fallback_attn

        embed_dim = int(model.config.hidden_size)
        if resume_enabled and embeddings_path.exists():
            shard_embeddings = np.load(embeddings_path, mmap_mode="r+")
            if shard_embeddings.shape != (len(texts), embed_dim):
                print(
                    f"[shard {shard_id}] Cached shape {shard_embeddings.shape} does not match "
                    f"expected {(len(texts), embed_dim)}. Rebuilding cache."
                )
                embeddings_path.unlink(missing_ok=True)
                progress_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                shard_embeddings = np.lib.format.open_memmap(
                    embeddings_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(texts), embed_dim),
                )
                processed = 0
        else:
            shard_embeddings = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(texts), embed_dim),
            )
            processed = 0

        _atomic_write_json(
            meta_path,
            {
                "shard_id": shard_id,
                "rows": len(texts),
                "dim": embed_dim,
                "cache_signature": cache_signature,
                "model_name_or_path": model_name_or_path,
            },
        )

        if processed > 0:
            print(f"[shard {shard_id}] Resuming from {processed}/{len(texts)}")

        iterator = range(processed, len(texts), batch_size)
        for start in tqdm(iterator, desc=f"encode-shard-{shard_id}", position=shard_id):
            batch = texts[start:start + batch_size]
            if prefix:
                batch = [f"{prefix}{text}" for text in batch]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
                emb = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                if normalize_embeddings:
                    emb = F.normalize(emb, p=2, dim=-1)
            batch_np = emb.cpu().float().numpy().astype(np.float32, copy=False)
            end = start + batch_np.shape[0]
            shard_embeddings[start:end] = batch_np
            shard_embeddings.flush()
            _atomic_write_json(
                progress_path,
                {
                    "shard_id": shard_id,
                    "processed": end,
                    "total": len(texts),
                    "completed": end >= len(texts),
                    "requested_attn": requested_attn,
                    "resolved_attn": resolved_attn,
                },
            )

        shard_embeddings.flush()
        _atomic_write_json(
            progress_path,
            {
                "shard_id": shard_id,
                "processed": len(texts),
                "total": len(texts),
                "completed": True,
                "requested_attn": requested_attn,
                "resolved_attn": resolved_attn,
            },
        )

        queue.put(
            {
                "shard_id": shard_id,
                "embeddings_path": str(embeddings_path),
                "count": int(len(texts)),
                "dim": embed_dim,
                "requested_attn": requested_attn,
                "resolved_attn": resolved_attn,
                "resumed_from": processed,
            }
        )
    except Exception:
        queue.put({"shard_id": shard_id, "error": traceback.format_exc()})


def _encode_worker_stream(
    shard_id: int,
    device: str,
    texts: list[str],
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    prefix: str,
    attn_implementation: str,
    allow_attn_fallback: bool,
    normalize_embeddings: bool,
    start_offset: int,
    queue: mp.Queue,
) -> None:
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
        }
        requested_attn = attn_implementation
        if requested_attn != "auto":
            model_kwargs["attn_implementation"] = requested_attn

        try:
            model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs).to(device).eval()
            resolved_attn = requested_attn
        except Exception as exc:
            if not allow_attn_fallback or requested_attn in {"auto", "sdpa", "eager"}:
                raise
            fallback_attn = "sdpa"
            print(
                f"[shard {shard_id}] Failed to load with attn={requested_attn} on {device}: "
                f"{exc.__class__.__name__}: {exc}. Falling back to {fallback_attn}."
            )
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["attn_implementation"] = fallback_attn
            model = AutoModel.from_pretrained(model_name_or_path, **fallback_kwargs).to(device).eval()
            resolved_attn = fallback_attn

        embed_dim = int(model.config.hidden_size)
        safe_offset = max(0, min(int(start_offset), len(texts)))
        if safe_offset > 0:
            print(f"[shard {shard_id}] Resuming streaming from {safe_offset}/{len(texts)}")
        iterator = range(safe_offset, len(texts), batch_size)
        for start in tqdm(iterator, desc=f"encode-shard-{shard_id}", position=shard_id):
            batch = texts[start:start + batch_size]
            if prefix:
                batch = [f"{prefix}{text}" for text in batch]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
                emb = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                if normalize_embeddings:
                    emb = F.normalize(emb, p=2, dim=-1)
            batch_np = emb.cpu().float().numpy().astype(np.float32, copy=False)
            queue.put(
                {
                    "type": "chunk",
                    "shard_id": shard_id,
                    "start": start,
                    "embeddings": batch_np,
                    "dim": embed_dim,
                    "resolved_attn": resolved_attn,
                }
            )

        queue.put(
            {
                "type": "done",
                "shard_id": shard_id,
                "dim": embed_dim,
                "resolved_attn": resolved_attn,
            }
        )
    except Exception:
        queue.put({"type": "error", "shard_id": shard_id, "error": traceback.format_exc()})


def _scan_flat_shard_entries(shard_dir: Path) -> list[dict[str, Any]]:
    if not shard_dir.exists():
        return []
    pattern = re.compile(r"^s(?P<shard>\d+)_(?P<start>\d+)_(?P<end>\d+)\.faiss$")
    entries: list[dict[str, Any]] = []
    for item in shard_dir.iterdir():
        if not item.is_file():
            continue
        match = pattern.match(item.name)
        if match is None:
            continue
        shard_id = int(match.group("shard"))
        start = int(match.group("start"))
        end = int(match.group("end"))
        entries.append(
            {
                "shard_id": shard_id,
                "start": start,
                "end": end,
                "count": max(0, end - start),
                "path": str(item),
            }
        )
    entries.sort(key=lambda row: (int(row["shard_id"]), int(row["start"]), int(row["end"])))
    return entries


def _compute_stream_resume_offsets(
    shard_entries: list[dict[str, Any]],
    num_shards: int,
    shard_sizes: list[int],
    max_rows_per_file: int,
) -> dict[int, int]:
    grouped: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(num_shards)}
    for entry in shard_entries:
        shard_id = int(entry["shard_id"])
        if shard_id in grouped:
            grouped[shard_id].append((int(entry["start"]), int(entry["end"])))

    offsets: dict[int, int] = {}
    for shard_id in range(num_shards):
        ranges = sorted(grouped[shard_id], key=lambda item: item[0])
        expected = 0
        for start, end in ranges:
            if start != expected:
                break
            if end <= start or (end - start) > max_rows_per_file:
                break
            expected = end
        offsets[shard_id] = min(expected, int(shard_sizes[shard_id]))
    return offsets


def _load_config_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    dataset_cfg = config.get("dataset", {})
    encoder_cfg = config.get("encoder", {})
    index_cfg = config.get("index", {})
    runtime_cfg = config.get("runtime", {})
    return {
        "dataset": dataset_cfg.get("name", config.get("dataset", "s-nlp/kilt")),
        "dataset_config": dataset_cfg.get("config"),
        "split": dataset_cfg.get("split", "train"),
        "text_col": dataset_cfg.get("text_col", "text"),
        "wikipedia_id_col": dataset_cfg.get("wikipedia_id_col", "wikipedia_id"),
        "wikipedia_title_col": dataset_cfg.get("wikipedia_title_col", "wikipedia_title"),
        "model_name_or_path": encoder_cfg.get("model_name_or_path", "Salesforce/SFR-Embedding-Mistral"),
        "max_length": int(encoder_cfg.get("max_length", 4096)),
        "passage_prefix": encoder_cfg.get("passage_prefix", ""),
        "output_dir": index_cfg.get("output_dir", "artifacts/indexes/kilt_sfr"),
        "index_type": index_cfg.get("index_type", "ivfpq"),
        "nlist": int(index_cfg.get("nlist", 16384)),
        "pq_m": int(index_cfg.get("pq_m", 64)),
        "pq_bits": int(index_cfg.get("pq_bits", 8)),
        "train_size": int(index_cfg.get("train_size", 200000)),
        "add_batch_size": int(index_cfg.get("add_batch_size", 16384)),
        "metadata_fields": index_cfg.get("metadata_fields", ["wikipedia_id", "wikipedia_title"]),
        "metric": index_cfg.get("metric", "ip"),
        "normalize": bool(index_cfg.get("normalize", True)),
        "metadata_format": index_cfg.get("metadata_format", "jsonl"),
        "devices": runtime_cfg.get("devices"),
        "batch_size": int(runtime_cfg.get("batch_size", 8)),
        "streaming_shard_batches_per_file": int(runtime_cfg.get("streaming_shard_batches_per_file", 100)),
        "attn_implementation": runtime_cfg.get("attn_implementation", "auto"),
        "allow_attn_fallback": bool(runtime_cfg.get("allow_attn_fallback", True)),
        "use_prepare_cache": bool(runtime_cfg.get("use_prepare_cache", True)),
        "refresh_prepare_cache": bool(runtime_cfg.get("refresh_prepare_cache", False)),
        "prepare_cache_path": runtime_cfg.get("prepare_cache_path"),
        "use_encode_cache": bool(runtime_cfg.get("use_encode_cache", True)),
        "refresh_encode_cache": bool(runtime_cfg.get("refresh_encode_cache", False)),
        "encode_cache_dir": runtime_cfg.get("encode_cache_dir"),
        "max_samples": runtime_cfg.get("max_samples"),
        "hf_cache_dir": runtime_cfg.get("hf_cache_dir"),
    }


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML recipe")
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = _load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Build FAISS index for s-nlp/kilt with Salesforce/SFR-Embedding-Mistral",
        parents=[pre_parser],
    )
    parser.add_argument("--dataset", default=defaults.get("dataset", "s-nlp/kilt"))
    parser.add_argument("--dataset-config", default=defaults.get("dataset_config"))
    parser.add_argument("--split", default=defaults.get("split", "train"))
    parser.add_argument("--text-col", default=defaults.get("text_col", "text"))
    parser.add_argument("--wikipedia-id-col", default=defaults.get("wikipedia_id_col", "wikipedia_id"))
    parser.add_argument("--wikipedia-title-col", default=defaults.get("wikipedia_title_col", "wikipedia_title"))
    parser.add_argument(
        "--model-name-or-path",
        default=defaults.get("model_name_or_path", "Salesforce/SFR-Embedding-Mistral"),
    )
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 8)))
    parser.add_argument(
        "--streaming-shard-batches-per-file",
        type=int,
        default=int(defaults.get("streaming_shard_batches_per_file", 100)),
        help=(
            "For flat/flat_fp16 streaming mode, number of encode batches packed "
            "into one shard .faiss file."
        ),
    )
    parser.add_argument("--max-length", type=int, default=int(defaults.get("max_length", 4096)))
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default=defaults.get("attn_implementation", "auto"),
        help="Attention backend for model loading. Use flash_attention_2 to try FlashAttention.",
    )
    parser.add_argument(
        "--allow-attn-fallback",
        dest="allow_attn_fallback",
        action="store_true",
        default=bool(defaults.get("allow_attn_fallback", True)),
        help="Fallback to sdpa if selected attention backend is unsupported.",
    )
    parser.add_argument(
        "--no-attn-fallback",
        dest="allow_attn_fallback",
        action="store_false",
        help="Disable fallback and fail immediately if selected attention backend is unsupported.",
    )
    parser.add_argument(
        "--devices",
        default=defaults.get("devices"),
        help="Comma-separated device list (e.g. cuda:0,cuda:1). Uses up to two devices.",
    )
    parser.add_argument(
        "--passage-prefix",
        default=defaults.get("passage_prefix", ""),
        help="Optional prefix added before each document text during embedding.",
    )
    parser.add_argument(
        "--use-prepare-cache",
        dest="use_prepare_cache",
        action="store_true",
        default=bool(defaults.get("use_prepare_cache", True)),
        help="Reuse cached prepare-records output if available.",
    )
    parser.add_argument(
        "--no-prepare-cache",
        dest="use_prepare_cache",
        action="store_false",
        help="Disable prepare-records cache (always recompute).",
    )
    parser.add_argument(
        "--refresh-prepare-cache",
        action="store_true",
        default=bool(defaults.get("refresh_prepare_cache", False)),
        help="Force recompute prepare-records and overwrite cache file.",
    )
    parser.add_argument(
        "--prepare-cache-path",
        default=defaults.get("prepare_cache_path"),
        help="Optional path to prepare-records parquet cache.",
    )
    parser.add_argument(
        "--use-encode-cache",
        dest="use_encode_cache",
        action="store_true",
        default=bool(defaults.get("use_encode_cache", True)),
        help="Reuse saved shard embeddings and resume encoding from progress.",
    )
    parser.add_argument(
        "--no-encode-cache",
        dest="use_encode_cache",
        action="store_false",
        help="Disable encode cache and always recompute embeddings.",
    )
    parser.add_argument(
        "--refresh-encode-cache",
        action="store_true",
        default=bool(defaults.get("refresh_encode_cache", False)),
        help="Delete existing encode cache for this run and recompute from scratch.",
    )
    parser.add_argument(
        "--encode-cache-dir",
        default=defaults.get("encode_cache_dir"),
        help="Optional directory to store shard embedding checkpoints.",
    )
    parser.add_argument("--output-dir", default=defaults.get("output_dir", "artifacts/indexes/kilt_sfr"))
    parser.add_argument(
        "--index-type",
        choices=["flat", "flat_fp16", "ivfpq"],
        default=defaults.get("index_type", "ivfpq"),
        help="FAISS index type. 'flat_fp16' avoids IVF training and stores vectors in fp16.",
    )
    parser.add_argument("--nlist", type=int, default=int(defaults.get("nlist", 16384)))
    parser.add_argument("--pq-m", type=int, default=int(defaults.get("pq_m", 64)))
    parser.add_argument("--pq-bits", type=int, default=int(defaults.get("pq_bits", 8)))
    parser.add_argument("--train-size", type=int, default=int(defaults.get("train_size", 200000)))
    parser.add_argument("--add-batch-size", type=int, default=int(defaults.get("add_batch_size", 16384)))
    parser.add_argument(
        "--metadata-fields",
        nargs="+",
        default=defaults.get("metadata_fields", ["wikipedia_id", "wikipedia_title"]),
        help="Fields to keep in final metadata rows. Supported: wikipedia_id wikipedia_title text",
    )
    parser.add_argument("--metric", choices=["ip", "l2"], default=defaults.get("metric", "ip"))
    parser.add_argument("--normalize", dest="normalize", action="store_true", default=bool(defaults.get("normalize", True)))
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument(
        "--metadata-format",
        choices=["jsonl", "parquet"],
        default=defaults.get("metadata_format", "jsonl"),
        help="Storage format for row-aligned metadata.",
    )
    parser.add_argument("--max-samples", type=int, default=defaults.get("max_samples"))
    parser.add_argument("--hf-cache-dir", default=defaults.get("hf_cache_dir"))
    parsed = parser.parse_args(remaining)
    if isinstance(parsed.metadata_fields, str):
        parsed.metadata_fields = [item.strip() for item in parsed.metadata_fields.split(",") if item.strip()]
    parsed.metadata_fields = [str(item) for item in parsed.metadata_fields]
    if int(parsed.streaming_shard_batches_per_file) < 1:
        raise ValueError("--streaming-shard-batches-per-file must be >= 1")
    return parsed


def _prepare_cache_key(args: argparse.Namespace) -> str:
    key_payload = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_col": args.text_col,
        "wikipedia_id_col": args.wikipedia_id_col,
        "wikipedia_title_col": args.wikipedia_title_col,
        "max_samples": args.max_samples,
    }
    digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return digest[:16]


def _resolve_prepare_cache_path(args: argparse.Namespace) -> Path:
    if args.prepare_cache_path:
        return Path(args.prepare_cache_path)
    key = _prepare_cache_key(args)
    return Path("artifacts/cache/kilt_prepare") / f"{key}.parquet"


def _read_prepare_cache(cache_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(cache_path, columns=["wikipedia_id", "wikipedia_title", "text"])
    pydict = table.to_pydict()
    count = table.num_rows

    texts = list(pydict["text"])
    metadata_rows = [
        {
            "wikipedia_id": pydict["wikipedia_id"][idx],
            "wikipedia_title": pydict["wikipedia_title"][idx],
            "text": pydict["text"][idx],
        }
        for idx in range(count)
    ]
    return texts, metadata_rows


def _write_prepare_cache(cache_path: Path, metadata_rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    table = pa.Table.from_pylist(metadata_rows)
    pq.write_table(table, temp_path)
    temp_path.replace(cache_path)


def load_kilt_records(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    cache_path = _resolve_prepare_cache_path(args)

    if args.use_prepare_cache and cache_path.exists() and not args.refresh_prepare_cache:
        print(f"Loading prepared records from cache: {cache_path}")
        texts, metadata_rows = _read_prepare_cache(cache_path)
        if not texts:
            raise ValueError(f"Prepare cache exists but is empty: {cache_path}")
        print(f"Loaded {len(texts)} prepared rows from cache")
        return texts, metadata_rows, {
            "prepare_cache_used": True,
            "prepare_cache_path": str(cache_path),
            "prepare_cache_refreshed": False,
        }

    from datasets import load_dataset

    print(f"Loading dataset: {args.dataset} (split={args.split})")
    ds = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        cache_dir=args.hf_cache_dir,
    )
    if args.max_samples is not None:
        sample_size = min(len(ds), int(args.max_samples))
        ds = ds.select(range(sample_size))
        print(f"Using subset: {sample_size} rows")

    required_cols = [args.text_col, args.wikipedia_id_col, args.wikipedia_title_col]
    for col in required_cols:
        if col not in ds.column_names:
            raise ValueError(f"Column '{col}' not found in dataset columns: {ds.column_names}")

    texts: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for row in tqdm(ds, desc="prepare-records"):
        text = row.get(args.text_col)
        if not isinstance(text, str) or not text.strip():
            continue
        texts.append(text)
        metadata_rows.append(
            {
                "wikipedia_id": row.get(args.wikipedia_id_col),
                "wikipedia_title": row.get(args.wikipedia_title_col),
                "text": text,
            }
        )

    if not texts:
        raise ValueError("No valid rows found after filtering by text column.")

    print(f"Prepared {len(texts)} rows with non-empty text")
    if args.use_prepare_cache:
        _write_prepare_cache(cache_path, metadata_rows)
        print(f"Saved prepare-records cache: {cache_path}")
    elif args.prepare_cache_path:
        print("prepare-cache-path was provided, but cache is disabled (--no-prepare-cache).")

    return texts, metadata_rows, {
        "prepare_cache_used": bool(args.use_prepare_cache),
        "prepare_cache_path": str(cache_path),
        "prepare_cache_refreshed": bool(args.refresh_prepare_cache),
    }


def encode_texts(texts: list[str], args: argparse.Namespace) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    devices = parse_devices(args.devices)
    print(f"Encoding with devices: {devices}")

    shard_indices: list[list[int]] = [[] for _ in range(len(devices))]
    shard_texts: list[list[str]] = [[] for _ in range(len(devices))]
    for idx, text in enumerate(texts):
        shard_id = idx % len(devices)
        shard_indices[shard_id].append(idx)
        shard_texts[shard_id].append(text)

    encode_cache_dir = _resolve_encode_cache_dir(args, len(texts))
    if args.use_encode_cache:
        if args.refresh_encode_cache and encode_cache_dir.exists():
            shutil.rmtree(encode_cache_dir, ignore_errors=True)
        encode_cache_dir.mkdir(parents=True, exist_ok=True)
        resume_enabled = not bool(args.refresh_encode_cache)
    else:
        encode_cache_dir = Path("artifacts/cache/kilt_encode/_tmp") / f"{_encode_cache_key(args, len(texts))}_{os.getpid()}"
        if encode_cache_dir.exists():
            shutil.rmtree(encode_cache_dir, ignore_errors=True)
        encode_cache_dir.mkdir(parents=True, exist_ok=True)
        resume_enabled = False

    cache_signature = _encode_cache_key(args, len(texts))
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []

    for shard_id, device in enumerate(devices):
        if not shard_texts[shard_id]:
            continue
        process = ctx.Process(
            target=_encode_worker,
            args=(
                shard_id,
                device,
                shard_texts[shard_id],
                args.model_name_or_path,
                int(args.batch_size),
                int(args.max_length),
                args.passage_prefix,
                args.attn_implementation,
                bool(args.allow_attn_fallback),
                bool(args.normalize),
                str(encode_cache_dir),
                resume_enabled,
                cache_signature,
                queue,
            ),
        )
        process.start()
        processes.append(process)

    results: list[dict[str, Any]] = []
    for _ in processes:
        results.append(queue.get())

    for process in processes:
        process.join()

    for process in processes:
        if process.exitcode not in (0, None):
            raise RuntimeError(f"Encoding process {process.pid} failed with exit code {process.exitcode}")

    errors = [result for result in results if "error" in result]
    if errors:
        raise RuntimeError("Shard encoding failed:\n" + "\n".join(err["error"] for err in errors))

    if not results:
        raise RuntimeError("No shard results were produced.")

    dim = int(results[0]["dim"])
    resolved_attn = sorted({str(item.get("resolved_attn", "auto")) for item in results})
    resumed_from = {int(item["shard_id"]): int(item.get("resumed_from", 0)) for item in results}
    shard_paths: dict[int, str] = {}
    shard_counts: dict[int, int] = {}
    for result in results:
        shard_id = int(result["shard_id"])
        shard_emb = np.load(result["embeddings_path"], mmap_mode="r")
        expected_rows = len(shard_indices[shard_id])
        if shard_emb.shape[0] != expected_rows:
            raise RuntimeError(
                f"Shard {shard_id} mismatch: {shard_emb.shape[0]} embeddings vs {expected_rows} expected rows."
            )
        shard_paths[shard_id] = str(result["embeddings_path"])
        shard_counts[shard_id] = expected_rows

    if len(resolved_attn) == 1:
        print(f"Attention backend in use: {resolved_attn[0]}")
    else:
        print(f"Attention backends in use by shard: {resolved_attn}")
    if any(count > 0 for count in resumed_from.values()):
        print(f"Resumed shard progress: {resumed_from}")

    encode_outputs = {
        "dim": dim,
        "total_rows": len(texts),
        "num_shards": len(devices),
        "shard_paths": shard_paths,
        "shard_counts": shard_counts,
    }
    encode_cache_info = {
        "encode_cache_enabled": bool(args.use_encode_cache),
        "encode_cache_refreshed": bool(args.refresh_encode_cache),
        "encode_cache_dir": str(encode_cache_dir),
        "encode_cache_resumed_from": resumed_from,
    }
    return encode_outputs, devices, encode_cache_info


def _create_faiss_index(dim: int, args: argparse.Namespace):
    import faiss

    metric = args.metric
    faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    if args.index_type == "flat":
        base_index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
        return faiss.IndexIDMap2(base_index), base_index
    if args.index_type == "flat_fp16":
        base_index = faiss.IndexScalarQuantizer(
            dim,
            faiss.ScalarQuantizer.QT_fp16,
            faiss_metric,
        )
        return faiss.IndexIDMap2(base_index), base_index

    quantizer = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    base_index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        int(args.nlist),
        int(args.pq_m),
        int(args.pq_bits),
        faiss_metric,
    )
    return faiss.IndexIDMap2(base_index), base_index


def _add_chunk_with_shard_ids(index: Any, chunk: np.ndarray, shard_id: int, start: int, num_shards: int) -> None:
    end = start + chunk.shape[0]
    local_ids = np.arange(start, end, dtype=np.int64)
    ids = local_ids * num_shards + shard_id
    index.add_with_ids(chunk.astype(np.float32, copy=False), ids)


def _build_faiss_streaming(
    texts: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Any, list[str], dict[str, Any], int]:
    devices = parse_devices(args.devices)
    print(f"Encoding with devices: {devices}")

    shard_texts: list[list[str]] = [[] for _ in range(len(devices))]
    for idx, text in enumerate(texts):
        shard_id = idx % len(devices)
        shard_texts[shard_id].append(text)

    num_shards = len(devices)
    pending_chunks: list[tuple[int, int, np.ndarray]] = []
    train_vectors: list[np.ndarray] = []
    train_collected = 0
    train_target = int(args.train_size)
    done_count = 0
    errors: list[str] = []
    resolved_attn: set[str] = set()
    index = None
    base_index = None
    embed_dim: int | None = None
    shard_dir = output_dir / "index_shards"
    resume_offsets = {idx: 0 for idx in range(num_shards)}
    shard_batches_per_file = max(1, int(args.streaming_shard_batches_per_file))
    max_rows_per_file = int(args.batch_size) * shard_batches_per_file
    if args.index_type in {"flat", "flat_fp16"}:
        shard_dir.mkdir(parents=True, exist_ok=True)
        existing_entries = _scan_flat_shard_entries(shard_dir)
        if existing_entries:
            resume_offsets = _compute_stream_resume_offsets(
                existing_entries,
                num_shards=num_shards,
                shard_sizes=[len(items) for items in shard_texts],
                max_rows_per_file=max_rows_per_file,
            )
            if any(offset > 0 for offset in resume_offsets.values()):
                print(f"Resuming flat shard streaming from offsets: {resume_offsets}")

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=max(16, len(devices) * 8))
    processes: list[mp.Process] = []
    for shard_id, device in enumerate(devices):
        if not shard_texts[shard_id]:
            continue
        if args.index_type in {"flat", "flat_fp16"} and int(resume_offsets.get(shard_id, 0)) >= len(shard_texts[shard_id]):
            continue
        process = ctx.Process(
            target=_encode_worker_stream,
            args=(
                shard_id,
                device,
                shard_texts[shard_id],
                args.model_name_or_path,
                int(args.batch_size),
                int(args.max_length),
                args.passage_prefix,
                args.attn_implementation,
                bool(args.allow_attn_fallback),
                bool(args.normalize),
                int(resume_offsets.get(shard_id, 0)),
                queue,
            ),
        )
        process.start()
        processes.append(process)

    if not processes and args.index_type not in {"flat", "flat_fp16"}:
        raise RuntimeError("No encoding workers started for streaming mode.")

    flat_group_buffers: dict[int, list[np.ndarray]] = {idx: [] for idx in range(num_shards)}
    flat_group_start: dict[int, int | None] = {idx: None for idx in range(num_shards)}
    flat_expected_start: dict[int, int] = {idx: int(resume_offsets.get(idx, 0)) for idx in range(num_shards)}
    flat_pending_by_start: dict[int, dict[int, np.ndarray]] = {idx: {} for idx in range(num_shards)}

    def _flush_flat_group(shard_id: int) -> None:
        if not flat_group_buffers[shard_id]:
            return
        start = flat_group_start[shard_id]
        if start is None:
            return
        grouped_chunk = np.concatenate(flat_group_buffers[shard_id], axis=0).astype(np.float32, copy=False)
        chunk_index, _ = _create_faiss_index(int(grouped_chunk.shape[1]), args)
        _add_chunk_with_shard_ids(chunk_index, grouped_chunk, shard_id, start, num_shards)
        end = start + int(grouped_chunk.shape[0])
        shard_file = shard_dir / f"s{shard_id}_{start}_{end}.faiss"
        save_faiss_index(chunk_index, shard_file)
        flat_group_buffers[shard_id] = []
        flat_group_start[shard_id] = None

    while done_count < len(processes):
        message = queue.get()
        msg_type = message.get("type")
        if msg_type == "error":
            errors.append(str(message.get("error", "unknown worker error")))
            done_count += 1
            continue
        if msg_type == "done":
            done_count += 1
            resolved_attn.add(str(message.get("resolved_attn", "auto")))
            if embed_dim is None:
                embed_dim = int(message["dim"])
            continue

        if msg_type != "chunk":
            errors.append(f"Unknown streaming message: {message}")
            continue

        shard_id = int(message["shard_id"])
        start = int(message["start"])
        chunk = np.asarray(message["embeddings"], dtype=np.float32)
        if embed_dim is None:
            embed_dim = int(message["dim"])
        if args.index_type in {"flat", "flat_fp16"}:
            flat_pending_by_start[shard_id][start] = chunk
            while flat_expected_start[shard_id] in flat_pending_by_start[shard_id]:
                next_start = flat_expected_start[shard_id]
                next_chunk = flat_pending_by_start[shard_id].pop(next_start)
                if flat_group_start[shard_id] is None:
                    flat_group_start[shard_id] = next_start
                flat_group_buffers[shard_id].append(next_chunk)
                flat_expected_start[shard_id] = next_start + int(next_chunk.shape[0])
                if len(flat_group_buffers[shard_id]) >= shard_batches_per_file:
                    _flush_flat_group(shard_id)
            continue

        if index is None:
            index, base_index = _create_faiss_index(embed_dim, args)

        resolved_attn.add(str(message.get("resolved_attn", "auto")))
        if args.index_type == "ivfpq" and not bool(base_index.is_trained):
            pending_chunks.append((shard_id, start, chunk))
            if train_collected < train_target:
                need = train_target - train_collected
                take = chunk[:need]
                if take.size:
                    train_vectors.append(np.asarray(take, dtype=np.float32))
                    train_collected += take.shape[0]
            if train_collected >= train_target:
                train_matrix = np.vstack(train_vectors).astype(np.float32)
                print(f"Training IVF-PQ on {train_matrix.shape[0]} vectors (streaming)...")
                base_index.train(train_matrix)
                for pending_shard_id, pending_start, pending_chunk in pending_chunks:
                    _add_chunk_with_shard_ids(index, pending_chunk, pending_shard_id, pending_start, num_shards)
                pending_chunks.clear()
                train_vectors.clear()
        else:
            _add_chunk_with_shard_ids(index, chunk, shard_id, start, num_shards)

    for process in processes:
        process.join()
    for process in processes:
        if process.exitcode not in (0, None):
            errors.append(f"Encoding process {process.pid} failed with exit code {process.exitcode}")
    if args.index_type in {"flat", "flat_fp16"}:
        for shard_id in range(num_shards):
            if flat_pending_by_start[shard_id]:
                errors.append(
                    f"Shard {shard_id} has non-contiguous pending chunks: "
                    f"{sorted(flat_pending_by_start[shard_id].keys())[:5]}"
                )
            _flush_flat_group(shard_id)

    if errors:
        raise RuntimeError("Shard encoding failed:\n" + "\n".join(errors))
    if args.index_type in {"flat", "flat_fp16"}:
        shard_manifest_entries = _scan_flat_shard_entries(shard_dir)
        if not shard_manifest_entries:
            raise RuntimeError("No flat shard files were produced.")
        import faiss

        first_chunk_path = None
        for entry in shard_manifest_entries:
            if int(entry["count"]) > 0:
                first_chunk_path = str(entry["path"])
                break
        if first_chunk_path is None:
            raise RuntimeError("Flat shard files are present, but none contain vectors.")
        probe_index = faiss.read_index(first_chunk_path)
        embed_dim = int(probe_index.d)
        manifest = {
            "index_type": args.index_type,
            "dim": embed_dim,
            "num_shards": num_shards,
            "entries": shard_manifest_entries,
        }
        manifest_path = output_dir / "index_shards_manifest.json"
        write_json(manifest_path, manifest)
        if len(resolved_attn) == 1:
            print(f"Attention backend in use: {next(iter(resolved_attn))}")
        else:
            print(f"Attention backends in use by shard: {sorted(resolved_attn)}")
        encode_cache_info = {
            "encode_cache_enabled": False,
            "encode_cache_refreshed": False,
            "encode_cache_dir": None,
            "encode_cache_resumed_from": resume_offsets,
            "encode_streaming": True,
            "index_sharded": True,
            "index_shards_dir": str(shard_dir),
            "index_shards_manifest": str(manifest_path),
            "index_shards_count": len(shard_manifest_entries),
        }
        return manifest, devices, encode_cache_info, embed_dim

    if embed_dim is None:
        raise RuntimeError("Streaming encode produced no vectors.")

    if index is None or base_index is None:
        raise RuntimeError("Streaming encode produced no IVF data.")

    if args.index_type == "ivfpq" and not bool(base_index.is_trained):
        if train_vectors:
            train_matrix = np.vstack(train_vectors).astype(np.float32)
        elif pending_chunks:
            train_matrix = np.vstack([chunk for _, _, chunk in pending_chunks]).astype(np.float32)
        else:
            raise RuntimeError("No vectors available to train IVF-PQ index.")
        print(f"Training IVF-PQ on {train_matrix.shape[0]} vectors (finalize streaming)...")
        base_index.train(train_matrix)
        for pending_shard_id, pending_start, pending_chunk in pending_chunks:
            _add_chunk_with_shard_ids(index, pending_chunk, pending_shard_id, pending_start, num_shards)

    if len(resolved_attn) == 1:
        print(f"Attention backend in use: {next(iter(resolved_attn))}")
    else:
        print(f"Attention backends in use by shard: {sorted(resolved_attn)}")

    encode_cache_info = {
        "encode_cache_enabled": False,
        "encode_cache_refreshed": False,
        "encode_cache_dir": None,
        "encode_cache_resumed_from": {},
        "encode_streaming": True,
        "index_sharded": False,
    }
    return index, devices, encode_cache_info, embed_dim


def _build_faiss_from_shards(encode_outputs: dict[str, Any], args: argparse.Namespace):
    dim = int(encode_outputs["dim"])
    num_shards = int(encode_outputs["num_shards"])
    shard_paths: dict[int, str] = {int(k): v for k, v in encode_outputs["shard_paths"].items()}

    index, base_index = _create_faiss_index(dim, args)

    if args.index_type == "ivfpq":
        train_target = int(args.train_size)
        train_vectors: list[np.ndarray] = []
        collected = 0
        for shard_id in sorted(shard_paths):
            shard_emb = np.load(shard_paths[shard_id], mmap_mode="r")
            for start in range(0, shard_emb.shape[0], int(args.add_batch_size)):
                if collected >= train_target:
                    break
                end = min(start + int(args.add_batch_size), shard_emb.shape[0], start + (train_target - collected))
                chunk = np.asarray(shard_emb[start:end], dtype=np.float32)
                if chunk.size:
                    train_vectors.append(chunk)
                    collected += chunk.shape[0]
            if collected >= train_target:
                break
        if not train_vectors:
            raise RuntimeError("Failed to collect train vectors for IVF-PQ.")
        train_matrix = np.vstack(train_vectors).astype(np.float32)
        print(f"Training IVF-PQ on {train_matrix.shape[0]} vectors...")
        base_index.train(train_matrix)

    add_bs = int(args.add_batch_size)
    for shard_id in sorted(shard_paths):
        shard_emb = np.load(shard_paths[shard_id], mmap_mode="r")
        for start in tqdm(range(0, shard_emb.shape[0], add_bs), desc=f"faiss-add-shard-{shard_id}"):
            end = min(start + add_bs, shard_emb.shape[0])
            chunk = np.asarray(shard_emb[start:end], dtype=np.float32)
            _add_chunk_with_shard_ids(index, chunk, shard_id, start, num_shards)
    return index


def _filter_metadata_fields(metadata_rows: list[dict[str, Any]], metadata_fields: list[str]) -> list[dict[str, Any]]:
    allowed = {"wikipedia_id", "wikipedia_title", "text"}
    unknown = [field for field in metadata_fields if field not in allowed]
    if unknown:
        raise ValueError(f"Unsupported metadata fields: {unknown}. Allowed: {sorted(allowed)}")
    if not metadata_fields:
        raise ValueError("metadata_fields cannot be empty.")
    return [{field: row.get(field) for field in metadata_fields} for row in metadata_rows]


def save_metadata(
    output_dir: Path,
    metadata_rows: list[dict[str, Any]],
    metadata_format: str,
    metadata_fields: list[str],
) -> str:
    filtered_rows = _filter_metadata_fields(metadata_rows, metadata_fields)
    if metadata_format == "jsonl":
        file_name = "metadata.jsonl"
        write_jsonl(output_dir / file_name, filtered_rows)
        return file_name

    file_name = "metadata.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(filtered_rows)
    pq.write_table(table, output_dir / file_name)
    return file_name


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, metadata_rows, prepare_cache_info = load_kilt_records(args)
    if args.use_encode_cache:
        encode_outputs, used_devices, encode_cache_info = encode_texts(texts, args)
        index = _build_faiss_from_shards(encode_outputs, args)
        embedding_dim = int(encode_outputs["dim"])
        index_written = True
    else:
        index, used_devices, encode_cache_info, embedding_dim = _build_faiss_streaming(texts, args, output_dir)
        encode_outputs = {"dim": embedding_dim}
        index_written = not bool(encode_cache_info.get("index_sharded", False))
    if index_written:
        save_faiss_index(index, output_dir / "index.faiss")

    metadata_file = save_metadata(
        output_dir,
        metadata_rows,
        args.metadata_format,
        args.metadata_fields,
    )
    if index_written:
        index_size = int(index.ntotal)
    else:
        index_size = len(metadata_rows)

    if len(metadata_rows) != index_size:
        raise RuntimeError(
            f"Metadata/index size mismatch: metadata={len(metadata_rows)} index={index_size}"
        )

    run_info = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_col": args.text_col,
        "wikipedia_id_col": args.wikipedia_id_col,
        "wikipedia_title_col": args.wikipedia_title_col,
        "model_name_or_path": args.model_name_or_path,
        "batch_size": int(args.batch_size),
        "streaming_shard_batches_per_file": int(args.streaming_shard_batches_per_file),
        "max_length": int(args.max_length),
        "attn_implementation": args.attn_implementation,
        "allow_attn_fallback": bool(args.allow_attn_fallback),
        "use_prepare_cache": bool(args.use_prepare_cache),
        "refresh_prepare_cache": bool(args.refresh_prepare_cache),
        "prepare_cache_path": str(_resolve_prepare_cache_path(args)),
        "use_encode_cache": bool(args.use_encode_cache),
        "refresh_encode_cache": bool(args.refresh_encode_cache),
        "encode_cache_dir": str(_resolve_encode_cache_dir(args, len(texts))) if args.use_encode_cache else None,
        "devices": used_devices,
        "metric": args.metric,
        "index_type": args.index_type,
        "nlist": int(args.nlist),
        "pq_m": int(args.pq_m),
        "pq_bits": int(args.pq_bits),
        "train_size": int(args.train_size),
        "add_batch_size": int(args.add_batch_size),
        "metadata_fields": args.metadata_fields,
        "normalize": bool(args.normalize),
        "num_vectors": int(index_size),
        "embedding_dim": int(embedding_dim),
        "metadata_file": metadata_file,
        **prepare_cache_info,
        **encode_cache_info,
    }
    write_json(output_dir / "metadata.json", run_info)

    print("Index build complete")
    print(f"Output dir: {output_dir}")
    print(f"Vectors: {index_size}")
    if index_written:
        print("Index file: index.faiss")
    else:
        print(f"Index shards manifest: {encode_cache_info.get('index_shards_manifest')}")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
