#!/usr/bin/env python3
"""Search query sets against merged KILT FAISS shards (OSCAR E11 projector)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import l2_normalize


def log(msg: str, *, prefix: str) -> None:
    print(f"[{prefix}] {msg}", flush=True)

def load_kilt_texts(
    needed: set[str],
    arrow_dir: Path,
    text_cache_path: Path | None,
    log_prefix: str,
) -> dict[str, dict[str, str]]:
    import pyarrow.ipc as ipc

    found: dict[str, dict[str, str]] = {}
    if text_cache_path and text_cache_path.exists():
        log(f"Loading KILT text cache from {text_cache_path}", prefix=log_prefix)
        with text_cache_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                doc_id = rec["doc_id"]
                if doc_id in needed:
                    found[doc_id] = rec
        needed -= set(found.keys())
        log(f"Cache hit: {len(found)} docs, still need {len(needed)}", prefix=log_prefix)

    if not needed:
        return found

    arrow_files = sorted(arrow_dir.glob("kilt-train-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No KILT arrow shards under {arrow_dir}")

    log(
        f"Scanning {len(arrow_files)} local KILT shards for {len(needed)} doc texts...",
        prefix=log_prefix,
    )
    cache_handle = None
    if text_cache_path:
        text_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_handle = text_cache_path.open("a", encoding="utf-8")

    try:
        for arrow_file in tqdm(arrow_files, desc="KILT text lookup"):
            with arrow_file.open("rb") as handle:
                table = ipc.open_stream(handle).read_all()
            ids = table.column("_id").to_pylist()
            titles = table.column("wikipedia_title").to_pylist()
            texts = table.column("text").to_pylist()
            for doc_id, title, text in zip(ids, titles, texts):
                if doc_id not in needed:
                    continue
                title_s = str(title or "").strip()
                text_s = str(text or "").strip()
                rec = {
                    "doc_id": doc_id,
                    "title": title_s,
                    "text": text_s,
                    "document": f"{title_s} {text_s}".strip(),
                }
                found[doc_id] = rec
                needed.remove(doc_id)
                if cache_handle is not None:
                    cache_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if not needed:
                    break
            if not needed:
                break
    finally:
        if cache_handle is not None:
            cache_handle.close()

    if needed:
        sample = sorted(needed)[:5]
        raise KeyError(f"Missing {len(needed)} KILT docs after scan, e.g. {sample}")

    log(f"Resolved {len(found)} KILT document texts", prefix=log_prefix)
    return found


def load_shard_ids(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_shard_results(
    prev_scores: np.ndarray | None,
    prev_ids: np.ndarray | None,
    shard_scores: np.ndarray,
    shard_indices: np.ndarray,
    shard_doc_ids: list[str],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge top-k from a shard into running (score, global_doc_id) arrays."""
    n_queries = shard_scores.shape[0]
    doc_ids_arr = np.asarray(shard_doc_ids, dtype=object)
    safe_indices = np.clip(shard_indices, 0, len(shard_doc_ids) - 1)
    shard_global_ids = doc_ids_arr[safe_indices]
    shard_global_ids[shard_indices < 0] = ""

    if prev_scores is None:
        return shard_scores.copy(), shard_global_ids.copy()

    combined_scores = np.concatenate([prev_scores, shard_scores], axis=1)
    combined_ids = np.concatenate([prev_ids, shard_global_ids], axis=1)
    order = np.argsort(-combined_scores, axis=1)
    top_order = order[:, :top_k]
    row_idx = np.arange(n_queries)[:, None]
    merged_scores = combined_scores[row_idx, top_order]
    merged_ids = combined_ids[row_idx, top_order]
    return merged_scores, merged_ids


def search_shards(
    query_embeddings: np.ndarray,
    index_dir: Path,
    num_shards: int,
    top_k: int,
    num_threads: int,
    search_batch_size: int,
    log_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    faiss.omp_set_num_threads(num_threads)
    merged_scores: np.ndarray | None = None
    merged_ids: np.ndarray | None = None
    n_queries = query_embeddings.shape[0]

    for shard_id in range(num_shards):
        faiss_path = index_dir / f"kilt_shard_{shard_id}.faiss"
        ids_path = index_dir / f"kilt_shard_{shard_id}_ids.json"
        log(f"Loading shard {shard_id} into RAM: {faiss_path}", prefix=log_prefix)
        doc_ids = load_shard_ids(ids_path)
        index = faiss.read_index(str(faiss_path))
        if index.ntotal != len(doc_ids):
            raise ValueError(
                f"Shard {shard_id} mismatch: faiss={index.ntotal}, ids={len(doc_ids)}"
            )
        log(
            f"Searching shard {shard_id}: "
            f"{n_queries} queries x {index.ntotal} docs, top_k={top_k}, "
            f"batch_size={search_batch_size}",
            prefix=log_prefix,
        )
        t0 = __import__("time").time()
        score_chunks: list[np.ndarray] = []
        index_chunks: list[np.ndarray] = []
        for start in range(0, n_queries, search_batch_size):
            end = min(start + search_batch_size, n_queries)
            batch_scores, batch_indices = index.search(query_embeddings[start:end], top_k)
            score_chunks.append(batch_scores)
            index_chunks.append(batch_indices)
            log(
                f"Shard {shard_id} search progress: {end}/{n_queries} queries",
                prefix=log_prefix,
            )
        scores = np.concatenate(score_chunks, axis=0)
        indices = np.concatenate(index_chunks, axis=0)
        elapsed = __import__("time").time() - t0
        log(
            f"Shard {shard_id} search done in {elapsed:.1f}s "
            f"({n_queries / max(elapsed, 1e-6):.1f} queries/s)",
            prefix=log_prefix,
        )
        merged_scores, merged_ids = merge_shard_results(
            merged_scores, merged_ids, scores, indices, doc_ids, top_k
        )
        del index, scores, indices, score_chunks, index_chunks

    assert merged_scores is not None and merged_ids is not None
    return merged_scores, merged_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Query set -> top-k over KILT FAISS shards")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated"),
    )
    parser.add_argument(
        "--queries-path",
        "--popqa-path",
        type=Path,
        default=Path("/data/popqa_enriched.parquet"),
        dest="queries_path",
    )
    parser.add_argument("--question-col", type=str, default="question")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--run-name", type=str, default="KILT search")
    parser.add_argument(
        "--projector-path",
        type=str,
        default="artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt",
    )
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--search-batch-size", type=int, default=2000)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=32, help="FAISS/OpenMP threads for CPU search")
    parser.add_argument(
        "--query-embeddings-path",
        type=Path,
        default=None,
        help="Reuse precomputed query embeddings (.npy) and skip OSCAR encoding",
    )
    parser.add_argument(
        "--save-query-embeddings",
        type=Path,
        default=None,
        help="Optional path to save encoded queries as .npy",
    )
    parser.add_argument("--compressor-device", type=str, default="cuda:0")
    parser.add_argument("--decoder-device", type=str, default="cuda:1")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/results/retrieval/kilt_e11_popqa_top100.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional query limit for smoke tests")
    parser.add_argument(
        "--kilt-arrow-dir",
        type=Path,
        default=Path(
            "/mnt/raid/a-ploskin/hf_cache/datasets/s-nlp___kilt/default/0.0.0/"
            "a564cec5a4bc4bdcd9f32075a481e738bbed7067"
        ),
        help="Local KILT HF arrow shard directory",
    )
    parser.add_argument(
        "--kilt-text-cache",
        type=Path,
        default=Path("/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated/kilt_doc_texts.jsonl"),
        help="Append-only cache of resolved KILT doc_id -> text",
    )
    args = parser.parse_args()
    log_prefix = args.run_name

    os.environ["OSCAR_COMPRESSOR_DEVICE"] = args.compressor_device
    os.environ["OSCAR_DECODER_DEVICE"] = args.decoder_device

    df = pd.read_parquet(args.queries_path)
    if args.limit > 0:
        df = df.head(args.limit)
    questions = df[args.question_col].astype(str).tolist()
    qids = df[args.id_col].astype(str).tolist()
    log(f"Loaded {len(questions)} questions from {args.queries_path}", prefix=log_prefix)

    if args.query_embeddings_path is not None:
        log(f"Loading precomputed queries from {args.query_embeddings_path}", prefix=log_prefix)
        query_embeddings = np.load(args.query_embeddings_path).astype(np.float32)
        if query_embeddings.shape[0] != len(questions):
            raise ValueError(
                f"Query embedding count mismatch: {query_embeddings.shape[0]} vs {len(questions)}"
            )
    else:
        encoder_cfg = {
            "name": "oscar_projector",
            "kwargs": {
                "oscar_model_name": "naver/oscar-qwen2-7B",
                "projector_path": args.projector_path,
                "device": args.compressor_device,
                "embed_dim": 768,
                "pooler": "flatten",
                "num_layers": 2,
                "dropout": 0.0,
            },
        }
        log("Loading OSCAR + E11 projector...", prefix=log_prefix)
        encoder = build_encoder(encoder_cfg)

        query_chunks: list[np.ndarray] = []
        for start in tqdm(range(0, len(questions), args.encode_batch_size), desc="Encode queries"):
            batch = questions[start : start + args.encode_batch_size]
            encoded = encoder.encode(batch, batch)
            if hasattr(encoded, "detach"):
                encoded = encoded.detach().cpu().numpy()
            query_chunks.append(l2_normalize(np.asarray(encoded, dtype=np.float32)))
        query_embeddings = np.concatenate(query_chunks, axis=0)
        del encoder, query_chunks
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"Encoded queries: {query_embeddings.shape}", prefix=log_prefix)

    save_path = args.save_query_embeddings
    if save_path is None and args.query_embeddings_path is None:
        save_path = args.output_path.with_suffix(".queries.npy")
    if save_path is not None and args.query_embeddings_path is None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, query_embeddings)
        log(f"Saved query embeddings to {save_path}", prefix=log_prefix)

    merged_scores, merged_ids = search_shards(
        query_embeddings,
        args.index_dir,
        args.num_shards,
        args.top_k,
        args.num_threads,
        args.search_batch_size,
        log_prefix,
    )

    needed_doc_ids = {
        str(merged_ids[i, rank])
        for i in range(merged_ids.shape[0])
        for rank in range(args.top_k)
        if str(merged_ids[i, rank])
    }
    doc_texts = load_kilt_texts(
        needed_doc_ids,
        args.kilt_arrow_dir,
        args.kilt_text_cache,
        log_prefix,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as out:
        for i, question in enumerate(questions):
            hits = []
            for rank in range(args.top_k):
                doc_id = str(merged_ids[i, rank])
                if not doc_id:
                    continue
                doc = doc_texts[doc_id]
                hits.append(
                    {
                        "rank": rank + 1,
                        "doc_id": doc_id,
                        "score": float(merged_scores[i, rank]),
                        "title": doc["title"],
                        "text": doc["text"],
                        "document": doc["document"],
                    }
                )
            record = {
                "query_id": qids[i],
                "question": question,
                "top_k": hits,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"Wrote {len(questions)} results to {args.output_path}", prefix=log_prefix)


if __name__ == "__main__":
    main()
