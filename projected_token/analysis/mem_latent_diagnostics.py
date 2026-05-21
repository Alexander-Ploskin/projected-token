#!/usr/bin/env python3
"""Diagnostics for raw OSCAR MEM latents before projector training."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from transformers import AutoModel

from projected_token.config import load_yaml
from projected_token.io import write_csv, write_json
from projected_token.oscar_runtime import configure_oscar_component_devices, disable_transformers_allocator_warmup
from projected_token.plotting import plot_metric_comparison
from projected_token.retrieval.beir import _load_beir_dataset
from projected_token.retrieval.index.vector import l2_normalize
from projected_token.retrieval.metrics.ranking import aggregate_rankings


DEFAULT_POOLERS = ("mean", "max", "mean_max", "flatten", "first", "last")


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _parse_poolers(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    poolers = [item for item in items if item]
    unknown = sorted(set(poolers).difference(DEFAULT_POOLERS))
    if unknown:
        raise ValueError(f"Unknown pooler(s): {unknown}; expected one of {DEFAULT_POOLERS}")
    return poolers


def _fingerprint(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _load_oscar_model(
    model_name_or_path: str,
    *,
    device: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> Any:
    disable_transformers_allocator_warmup()
    device_map = device if device != "cpu" else "cpu"
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    ).eval()
    configure_oscar_component_devices(model)
    return model


def _pool_mem(mem_hiddens: torch.Tensor, pooler: str) -> torch.Tensor:
    if pooler == "mean":
        return mem_hiddens.mean(dim=1)
    if pooler == "first":
        return mem_hiddens[:, 0, :]
    if pooler == "last":
        return mem_hiddens[:, -1, :]
    if pooler == "max":
        return mem_hiddens.max(dim=1).values
    if pooler == "mean_max":
        return torch.cat([mem_hiddens.mean(dim=1), mem_hiddens.max(dim=1).values], dim=-1)
    if pooler == "flatten":
        return mem_hiddens.reshape(mem_hiddens.shape[0], -1)
    raise ValueError(f"Unknown pooler: {pooler}")


def _encode_poolers(
    model: Any,
    texts: list[str],
    *,
    poolers: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {pooler: [] for pooler in poolers}
    total = len(texts)
    if total == 0:
        return {pooler: np.zeros((0, 0), dtype=np.float32) for pooler in poolers}

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_texts = texts[start:start + batch_size]
            valid_indices = [idx for idx, text in enumerate(batch_texts) if text.strip()]
            valid_texts = [batch_texts[idx] for idx in valid_indices]
            compressed = model.compress_documents(documents=valid_texts) if valid_texts else None
            for pooler in poolers:
                if compressed is None:
                    pooled = torch.zeros((0, 0), dtype=torch.float32)
                else:
                    pooled = _pool_mem(compressed, pooler).detach().float().cpu()
                if len(valid_indices) == len(batch_texts):
                    arr = pooled.numpy()
                else:
                    dim = int(pooled.shape[-1]) if pooled.ndim == 2 and pooled.shape[0] else 0
                    arr = np.zeros((len(batch_texts), dim), dtype=np.float32)
                    for pooled_idx, original_idx in enumerate(valid_indices):
                        arr[original_idx] = pooled[pooled_idx].numpy()
                chunks[pooler].append(np.asarray(arr, dtype=np.float32))
            processed = min(start + len(batch_texts), total)
            print(f"[mem-diagnostics] encoded {processed}/{total} texts", flush=True)

    return {pooler: np.concatenate(parts, axis=0) for pooler, parts in chunks.items()}


def _encode_poolers_to_memmaps(
    model: Any,
    texts: list[str],
    *,
    poolers: list[str],
    batch_size: int,
    cache_dir: Path,
    prefix: str,
) -> dict[str, np.memmap]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    memmaps: dict[str, np.memmap] = {}
    total = len(texts)
    if total == 0:
        return {pooler: np.lib.format.open_memmap(cache_dir / f"{prefix}_{pooler}.npy", mode="w+", dtype=np.float32, shape=(0, 0)) for pooler in poolers}

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_texts = texts[start:start + batch_size]
            valid_indices = [idx for idx, text in enumerate(batch_texts) if text.strip()]
            valid_texts = [batch_texts[idx] for idx in valid_indices]
            compressed = model.compress_documents(documents=valid_texts) if valid_texts else None
            if compressed is None and not memmaps:
                raise ValueError("Cannot infer MEM dimensions from an all-empty first batch")
            for pooler in poolers:
                if compressed is None:
                    if pooler not in memmaps:
                        raise ValueError(f"Cannot infer dimension for pooler {pooler}")
                    arr = np.zeros((len(batch_texts), memmaps[pooler].shape[1]), dtype=np.float32)
                else:
                    pooled = _pool_mem(compressed, pooler).detach().float().cpu()
                    dim = int(pooled.shape[-1])
                    if pooler not in memmaps:
                        memmaps[pooler] = np.lib.format.open_memmap(
                            cache_dir / f"{prefix}_{pooler}.npy",
                            mode="w+",
                            dtype=np.float32,
                            shape=(total, dim),
                        )
                    arr = np.zeros((len(batch_texts), dim), dtype=np.float32)
                    for pooled_idx, original_idx in enumerate(valid_indices):
                        arr[original_idx] = pooled[pooled_idx].numpy()
                memmaps[pooler][start:start + len(batch_texts)] = arr
            processed = min(start + len(batch_texts), total)
            print(f"[mem-diagnostics] encoded {processed}/{total} texts", flush=True)

    for mmap in memmaps.values():
        mmap.flush()
    return memmaps


def _cosine_stats(vectors: np.ndarray, *, sample_limit: int, rng: np.random.Generator) -> dict[str, Any]:
    if vectors.shape[0] < 2:
        return {
            "sample_count": int(vectors.shape[0]),
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "effective_rank": 0.0,
            "diagnosis": "insufficient_samples",
        }

    sample_count = min(int(sample_limit), int(vectors.shape[0]))
    sample_idx = rng.choice(vectors.shape[0], size=sample_count, replace=False) if sample_count < vectors.shape[0] else np.arange(vectors.shape[0])
    normalized = l2_normalize(vectors[sample_idx].astype(np.float32))
    cos_matrix = normalized @ normalized.T
    off_diag = cos_matrix[~np.eye(sample_count, dtype=bool)]
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    try:
        singular_values = np.linalg.svd(centered, compute_uv=False)
        energy = singular_values.astype(np.float64) ** 2
        probs = energy / energy.sum() if energy.sum() > 0 else energy
        entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        effective_rank = float(np.exp(entropy)) if probs.size else 0.0
    except np.linalg.LinAlgError:
        effective_rank = 0.0

    mean_value = float(np.mean(off_diag))
    std_value = float(np.std(off_diag))
    if mean_value > 0.95 and std_value < 0.02:
        diagnosis = "collapsed"
    elif mean_value > 0.90:
        diagnosis = "weak_variance"
    else:
        diagnosis = "separable_enough"
    return {
        "sample_count": int(sample_count),
        "mean": mean_value,
        "std": std_value,
        "p05": float(np.percentile(off_diag, 5)),
        "p50": float(np.percentile(off_diag, 50)),
        "p95": float(np.percentile(off_diag, 95)),
        "effective_rank": effective_rank,
        "diagnosis": diagnosis,
    }


def _plot_cosine_histogram(values: np.ndarray, *, output_path: Path, title: str) -> None:
    if values.size == 0:
        return
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50)
    plt.title(title)
    plt.xlabel("off-diagonal cosine")
    plt.ylabel("count")
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _cosine_off_diag_values(vectors: np.ndarray, *, sample_limit: int, rng: np.random.Generator) -> np.ndarray:
    if vectors.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    sample_count = min(int(sample_limit), int(vectors.shape[0]))
    sample_idx = rng.choice(vectors.shape[0], size=sample_count, replace=False) if sample_count < vectors.shape[0] else np.arange(vectors.shape[0])
    normalized = l2_normalize(vectors[sample_idx].astype(np.float32))
    cos_matrix = normalized @ normalized.T
    return cos_matrix[~np.eye(sample_count, dtype=bool)]


def _beir_retrieval_metrics(
    *,
    corpus_ids: list[str],
    query_ids: list[str],
    relevant_map: dict[str, set[str]],
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: list[int],
    search_k: int,
    metric: str,
    normalize: bool,
    index_batch_size: int = 512,
    query_batch_size: int = 512,
) -> dict[str, float]:
    import faiss

    dim = int(doc_vectors.shape[1])
    index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    for start in range(0, doc_vectors.shape[0], index_batch_size):
        chunk = np.asarray(doc_vectors[start:start + index_batch_size], dtype=np.float32)
        if normalize:
            chunk = l2_normalize(chunk)
        index.add(chunk)

    indices = np.empty((query_vectors.shape[0], search_k), dtype=np.int64)
    for start in range(0, query_vectors.shape[0], query_batch_size):
        queries = l2_normalize(np.asarray(query_vectors[start:start + query_batch_size], dtype=np.float32))
        _, chunk_indices = index.search(queries, search_k)
        indices[start:start + queries.shape[0]] = chunk_indices
    ranking_cases: list[tuple[set[str], list[str]]] = []
    for idx, qid in enumerate(query_ids):
        relevant = relevant_map.get(qid, set())
        if not relevant:
            continue
        retrieved = [corpus_ids[doc_idx] for doc_idx in indices[idx].tolist() if 0 <= doc_idx < len(corpus_ids)]
        ranking_cases.append((relevant, retrieved))
    return aggregate_rankings(ranking_cases, top_k)


def _resolve_h5_paths(paths_or_patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in paths_or_patterns:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(Path(path) for path in matches)
        else:
            paths.append(Path(item))
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No teacher HDF5 files matched: {paths_or_patterns}")
    return existing


def _load_teacher_probe_rows(paths: list[Path], *, max_samples: int) -> tuple[list[str], np.ndarray, list[str]]:
    texts: list[str] = []
    embeddings: list[np.ndarray] = []
    ids: list[str] = []
    remaining = max_samples
    for path in paths:
        if remaining <= 0:
            break
        with h5py.File(path, "r") as h5:
            count = min(remaining, int(h5["embeddings"].shape[0]))
            for idx in range(count):
                text = h5["texts"][idx]
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                sample_id = h5["sample_ids"][idx] if "sample_ids" in h5 else h5["ids"][idx] if "ids" in h5 else f"{path.name}:{idx}"
                if isinstance(sample_id, bytes):
                    sample_id = sample_id.decode("utf-8")
                texts.append(str(text))
                ids.append(str(sample_id))
            embeddings.append(np.asarray(h5["embeddings"][:count], dtype=np.float32))
            remaining -= count
    if not embeddings:
        raise ValueError("Teacher probe found no embeddings")
    return texts, np.concatenate(embeddings, axis=0), ids


def _linear_probe(
    *,
    mem_vectors_by_pooler: dict[str, np.ndarray],
    teacher_vectors: np.ndarray,
    train_fraction: float,
) -> dict[str, dict[str, float]]:
    n_samples = min(next(iter(mem_vectors_by_pooler.values())).shape[0], teacher_vectors.shape[0])
    if n_samples < 4:
        return {
            pooler: {"sample_count": float(n_samples), "r2": 0.0, "cosine": 0.0, "baseline_mean_teacher_cosine": 0.0}
            for pooler in mem_vectors_by_pooler
        }

    split = int(n_samples * train_fraction)
    split = min(max(split, 1), n_samples - 1)
    teacher = teacher_vectors[:n_samples].astype(np.float32)
    teacher_normed = l2_normalize(teacher)
    teacher_mean = np.mean(teacher_normed[:split], axis=0, keepdims=True)
    teacher_mean = l2_normalize(teacher_mean)
    baseline = np.repeat(teacher_mean, n_samples - split, axis=0)
    baseline_cos = np.sum(baseline * teacher_normed[split:], axis=-1)

    results: dict[str, dict[str, float]] = {}
    for pooler, vectors in mem_vectors_by_pooler.items():
        x = vectors[:n_samples].astype(np.float32)
        # LSQR avoids forming a huge feature covariance matrix for flatten pooling.
        ridge = Ridge(alpha=1.0, solver="lsqr")
        ridge.fit(x[:split], teacher[:split])
        pred = ridge.predict(x[split:]).astype(np.float32)
        pred_normed = l2_normalize(pred)
        cos = np.sum(pred_normed * teacher_normed[split:], axis=-1)
        results[pooler] = {
            "sample_count": float(n_samples),
            "train_count": float(split),
            "val_count": float(n_samples - split),
            "r2": float(r2_score(teacher[split:], pred)),
            "cosine": float(np.mean(cos)),
            "baseline_mean_teacher_cosine": float(np.mean(baseline_cos)),
        }
    return results


def _average_metrics(per_dataset: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_dataset:
        return {}
    keys = list(next(iter(per_dataset.values())).keys())
    return {key: float(np.mean([metrics[key] for metrics in per_dataset.values()])) for key in keys}


def run_mem_latent_diagnostics(
    *,
    beir_config_path: Path,
    oscar_model: str,
    output_dir: Path,
    poolers: list[str],
    device: str = "cuda:0",
    batch_size: int = 16,
    max_docs_for_cosine: int = 1000,
    teacher_h5: list[str] | None = None,
    teacher_max_samples: int = 2000,
    teacher_train_fraction: float = 0.8,
    torch_dtype_name: str = "bfloat16",
    trust_remote_code: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    poolers = _parse_poolers(poolers)
    cfg = load_yaml(beir_config_path)
    index_cfg = cfg.get("index", {})
    metric_cfg = cfg.get("metrics", {})
    datasets_cfg = cfg.get("datasets", [])
    split = str(cfg.get("split", "test"))
    top_k = [int(k) for k in metric_cfg.get("top_k", [1, 3, 5, 10, 20])]
    search_k = int(cfg.get("search_k", max(top_k)))
    max_queries_global = cfg.get("max_queries_per_dataset")
    if max_queries_global is not None:
        max_queries_global = int(max_queries_global)
    normalize = bool(index_cfg.get("normalize", True))
    metric = str(index_cfg.get("metric", "ip"))
    rng = np.random.default_rng(seed)

    run_config = {
        "beir_config": str(beir_config_path),
        "oscar_model": oscar_model,
        "poolers": poolers,
        "device": device,
        "batch_size": batch_size,
        "max_docs_for_cosine": max_docs_for_cosine,
        "teacher_h5": teacher_h5 or [],
        "teacher_max_samples": teacher_max_samples,
        "teacher_train_fraction": teacher_train_fraction,
        "torch_dtype": torch_dtype_name,
        "split": split,
        "top_k": top_k,
        "search_k": search_k,
        "seed": seed,
    }
    run_id = f"mem_latent_{_fingerprint(run_config)}"
    model = _load_oscar_model(
        oscar_model,
        device=device,
        torch_dtype=_torch_dtype(torch_dtype_name),
        trust_remote_code=trust_remote_code,
    )

    per_pooler_dataset: dict[str, dict[str, dict[str, float]]] = {pooler: {} for pooler in poolers}
    cosine_stats: dict[str, dict[str, Any]] = {}
    all_cosine_vectors: dict[str, list[np.ndarray]] = {pooler: [] for pooler in poolers}
    rows: list[dict[str, Any]] = []
    dataset_names: list[str] = []

    for item in datasets_cfg:
        dataset_name = str(item["name"])
        dataset_names.append(dataset_name)
        dataset_dir = Path(item["path"])
        dataset_max_q = item.get("max_queries", max_queries_global)
        print(f"[mem-diagnostics] dataset start: {dataset_name} ({dataset_dir})", flush=True)
        corpus_ids, corpus_texts, query_ids, query_texts, relevant_map = _load_beir_dataset(
            dataset_dir,
            split=split,
            max_queries=(int(dataset_max_q) if dataset_max_q is not None else None),
        )
        print(
            f"[mem-diagnostics] loaded {dataset_name}: corpus={len(corpus_texts)}, queries={len(query_texts)}",
            flush=True,
        )
        cache_dir = output_dir / "vector_cache" / dataset_name
        doc_vectors = _encode_poolers_to_memmaps(
            model,
            corpus_texts,
            poolers=poolers,
            batch_size=batch_size,
            cache_dir=cache_dir,
            prefix="corpus",
        )
        query_vectors = _encode_poolers_to_memmaps(
            model,
            query_texts,
            poolers=poolers,
            batch_size=batch_size,
            cache_dir=cache_dir,
            prefix="queries",
        )

        for pooler in poolers:
            stats = _cosine_stats(doc_vectors[pooler], sample_limit=max_docs_for_cosine, rng=rng)
            cosine_stats[f"{dataset_name}:{pooler}"] = stats
            off_diag = _cosine_off_diag_values(doc_vectors[pooler], sample_limit=max_docs_for_cosine, rng=rng)
            all_cosine_vectors[pooler].append(off_diag)
            _plot_cosine_histogram(
                off_diag,
                output_path=output_dir / f"cosine_hist_{dataset_name}_{pooler}.png",
                title=f"{dataset_name} raw MEM {pooler} off-diagonal cosine",
            )
            metrics = _beir_retrieval_metrics(
                corpus_ids=corpus_ids,
                query_ids=query_ids,
                relevant_map=relevant_map,
                doc_vectors=doc_vectors[pooler],
                query_vectors=query_vectors[pooler],
                top_k=top_k,
                search_k=search_k,
                metric=metric,
                normalize=normalize,
            )
            per_pooler_dataset[pooler][dataset_name] = metrics
            print(
                f"[mem-diagnostics] {dataset_name}/{pooler}: "
                f"ndcg@10={metrics.get('ndcg@10', 0.0):.6f}, "
                f"cos_mean={stats['mean']:.6f}, cos_std={stats['std']:.6f}",
                flush=True,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "pooler": pooler,
                    "diagnostic": "cosine",
                    "cosine_mean": stats["mean"],
                    "cosine_std": stats["std"],
                    "cosine_p05": stats["p05"],
                    "cosine_p50": stats["p50"],
                    "cosine_p95": stats["p95"],
                    "effective_rank": stats["effective_rank"],
                    "diagnosis": stats["diagnosis"],
                }
            )
            for metric_name, metric_value in metrics.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "pooler": pooler,
                        "diagnostic": "retrieval",
                        "metric": metric_name,
                        "value": metric_value,
                    }
                )

    linear_probe_results: dict[str, dict[str, float]] = {}
    if teacher_h5:
        teacher_paths = _resolve_h5_paths(teacher_h5)
        teacher_texts, teacher_vectors, teacher_ids = _load_teacher_probe_rows(teacher_paths, max_samples=teacher_max_samples)
        teacher_mem = _encode_poolers(model, teacher_texts, poolers=poolers, batch_size=batch_size)
        linear_probe_results = _linear_probe(
            mem_vectors_by_pooler=teacher_mem,
            teacher_vectors=teacher_vectors,
            train_fraction=teacher_train_fraction,
        )
        write_json(output_dir / "teacher_probe_ids.json", {"ids": teacher_ids, "paths": [str(path) for path in teacher_paths]})
        for pooler, metrics in linear_probe_results.items():
            rows.append({"run_id": run_id, "dataset": "teacher_h5", "pooler": pooler, "diagnostic": "linear_probe", **metrics})

    average_by_pooler = {pooler: _average_metrics(metrics) for pooler, metrics in per_pooler_dataset.items()}
    for pooler, metrics in average_by_pooler.items():
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": "beir_average",
                    "pooler": pooler,
                    "diagnostic": "retrieval",
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

    aggregate_cosine_stats: dict[str, dict[str, Any]] = {}
    for pooler, chunks in all_cosine_vectors.items():
        values = np.concatenate([chunk for chunk in chunks if chunk.size]) if any(chunk.size for chunk in chunks) else np.zeros((0,), dtype=np.float32)
        if values.size:
            aggregate_cosine_stats[pooler] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
            _plot_cosine_histogram(
                values,
                output_path=output_dir / f"cosine_hist_{pooler}.png",
                title=f"Raw MEM {pooler} off-diagonal cosine",
            )

    summary = {
        "run_id": run_id,
        "config_fingerprint": _fingerprint(run_config),
        "run_config": run_config,
        "datasets": dataset_names,
        "split": split,
        "top_k": top_k,
        "search_k": search_k,
        "per_pooler_dataset": per_pooler_dataset,
        "average_by_pooler": average_by_pooler,
        "cosine_stats": cosine_stats,
        "aggregate_cosine_stats": aggregate_cosine_stats,
        "linear_probe": linear_probe_results,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "pooling_rows.csv", rows)
    if average_by_pooler:
        plot_metric_comparison(
            labels=list(average_by_pooler.keys()),
            values=[metrics.get("ndcg@10", 0.0) for metrics in average_by_pooler.values()],
            output_path=output_dir / "retrieval_ndcg10_by_pooler.png",
            title="Raw MEM BEIR NDCG@10 by Pooler",
            y_label="ndcg@10",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw OSCAR MEM latent diagnostics on BEIR-style datasets")
    parser.add_argument("--beir-config", type=Path, required=True)
    parser.add_argument("--oscar-model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/mem_latent_diagnostics"))
    parser.add_argument("--poolers", default=",".join(DEFAULT_POOLERS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-docs-for-cosine", type=int, default=1000)
    parser.add_argument("--teacher-h5", action="append", default=[])
    parser.add_argument("--teacher-max-samples", type=int, default=2000)
    parser.add_argument("--teacher-train-fraction", type=float, default=0.8)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--no-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_mem_latent_diagnostics(
        beir_config_path=args.beir_config,
        oscar_model=args.oscar_model,
        output_dir=args.output_dir,
        poolers=_parse_poolers(args.poolers),
        device=args.device,
        batch_size=args.batch_size,
        max_docs_for_cosine=args.max_docs_for_cosine,
        teacher_h5=list(args.teacher_h5),
        teacher_max_samples=args.teacher_max_samples,
        teacher_train_fraction=args.teacher_train_fraction,
        torch_dtype_name=args.torch_dtype,
        trust_remote_code=not args.no_trust_remote_code,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
