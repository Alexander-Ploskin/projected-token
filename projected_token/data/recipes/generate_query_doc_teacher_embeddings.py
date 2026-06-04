#!/usr/bin/env python3
"""Generate BGE query/document teacher embeddings for retrieval distillation."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_DOCUMENT_PREFIX = "Represent this sentence: "
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NEGATIVE_STRATEGIES = ("random", "bm25", "bge_dense", "hybrid")


def _load_json_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    records = data[:limit] if limit else data
    out: list[dict[str, Any]] = []
    for i, row in enumerate(records):
        query = str(row.get("query", "")).strip()
        positive = str(row.get("positive_doc", row.get("positive", ""))).strip()
        negatives = row.get("negatives", [])
        negative = ""
        if isinstance(negatives, list) and negatives:
            negative = str(negatives[0]).strip()
        elif row.get("negative") is not None:
            negative = str(row.get("negative", "")).strip()
        if query and positive:
            out.append(
                {
                    "query": query,
                    "positive_doc": positive,
                    "negative_doc": negative,
                    "sample_id": str(row.get("sample_id", f"json:{i}")),
                    "source": str(row.get("source", row.get("domain", "json"))),
                }
            )
    return out


def _load_msmarco_triplets(limit: int) -> list[dict[str, Any]]:
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(ds, desc="msmarco-v1.1", total=limit)):
        if i >= limit:
            break
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        passages = row.get("passages", {}) or {}
        passage_texts = [str(x).strip() for x in passages.get("passage_text", [])]
        selected_flags = passages.get("is_selected", [])
        positive = ""
        positives: list[str] = []
        negatives: list[str] = []
        for idx, text in enumerate(passage_texts):
            if not text:
                continue
            flag = int(selected_flags[idx]) if idx < len(selected_flags) else 0
            if flag > 0:
                positives.append(text)
            else:
                negatives.append(text)
        if positives:
            positive = positives[0]
        elif passage_texts:
            positive = passage_texts[0]
        negative = negatives[0] if negatives else (passage_texts[1] if len(passage_texts) > 1 else "")
        if query and positive:
            records.append(
                {
                    "query": query,
                    "positive_doc": positive,
                    "negative_doc": negative,
                    "sample_id": f"msmarco:{i}",
                    "source": "msmarco",
                    "query_id": query_id,
                    "positive_doc_id": f"msmarco:{query_id}:pos:0",
                    "negative_doc_id": f"msmarco:{query_id}:neg:0",
                }
            )
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokenize_for_bm25(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_hard_negative_candidates(
    *,
    queries: dict[str, str],
    corpus: dict[str, str],
    relevant: dict[str, list[str]],
    rank_start: int,
    rank_end: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError("Install rank-bm25 to use --negative-strategy bm25") from exc

    corpus_ids = list(corpus.keys())
    tokenized_corpus = [_tokenize_for_bm25(corpus[cid]) for cid in tqdm(corpus_ids, desc="bm25-tokenize-corpus")]
    bm25 = BM25Okapi(tokenized_corpus)
    stop = max(rank_start + 1, rank_end)
    hard_negatives: dict[str, list[dict[str, Any]]] = {}
    positive_scores: dict[str, float] = {}

    for qid in tqdm(list(relevant.keys()), desc=f"bm25-mine-{len(relevant)}-queries"):
        query = queries.get(qid, "")
        if not query:
            continue
        excluded = set(relevant.get(qid, []))
        scores = bm25.get_scores(_tokenize_for_bm25(query))
        positive_score_candidates = [float(scores[corpus_ids.index(doc_id)]) for doc_id in excluded if doc_id in corpus]
        if positive_score_candidates:
            positive_scores[qid] = max(positive_score_candidates)
        candidate_count = min(len(corpus_ids), max(stop * 4, stop + len(excluded) + 10))
        top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        ranked = [
            {
                "doc_id": corpus_ids[int(idx)],
                "bm25_rank": rank,
                "bm25_score": float(scores[int(idx)]),
                "dense_rank": -1,
                "dense_score": float("nan"),
                "negative_source": "bm25",
            }
            for rank, idx in enumerate(top_indices, start=1)
        ]
        window = [
            item
            for item in ranked
            if rank_start <= int(item["bm25_rank"]) < rank_end
            and str(item["doc_id"]) not in excluded
            and corpus.get(str(item["doc_id"]), "")
        ]
        fallback = [item for item in ranked if str(item["doc_id"]) not in excluded and corpus.get(str(item["doc_id"]), "")]
        candidates = window or fallback
        if candidates:
            hard_negatives[qid] = candidates
    return hard_negatives, positive_scores


def _bge_dense_hard_negative_candidates(
    *,
    queries: dict[str, str],
    corpus: dict[str, str],
    relevant: dict[str, list[str]],
    model: SentenceTransformer,
    batch_size: int,
    rank_start: int,
    rank_end: int,
    query_batch_size: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    corpus_ids = [cid for cid, text in corpus.items() if text]
    corpus_id_to_index = {cid: idx for idx, cid in enumerate(corpus_ids)}
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    query_ids = [qid for qid in relevant.keys() if queries.get(qid, "")]
    stop = max(rank_start + 1, rank_end)
    if not corpus_ids or not query_ids:
        return {}

    corpus_embeddings = _encode_prefixed(model, corpus_texts, prefix=BGE_DOCUMENT_PREFIX, batch_size=batch_size)
    hard_negatives: dict[str, list[dict[str, Any]]] = {}
    positive_scores: dict[str, float] = {}
    for start in tqdm(range(0, len(query_ids), query_batch_size), desc=f"bge-dense-mine-{len(query_ids)}-queries"):
        batch_qids = query_ids[start : start + query_batch_size]
        batch_queries = [queries[qid] for qid in batch_qids]
        query_embeddings = _encode_prefixed(model, batch_queries, prefix=BGE_QUERY_PREFIX, batch_size=batch_size)
        similarities = np.matmul(query_embeddings, corpus_embeddings.T)
        for row_idx, qid in enumerate(batch_qids):
            excluded = set(relevant.get(qid, []))
            scores = similarities[row_idx]
            positive_score_candidates = [
                float(scores[corpus_id_to_index[doc_id]])
                for doc_id in excluded
                if doc_id in corpus_id_to_index
            ]
            if positive_score_candidates:
                positive_scores[qid] = max(positive_score_candidates)
            candidate_count = min(len(corpus_ids), max(stop * 4, stop + len(excluded) + 10))
            top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            ranked = [
                {
                    "doc_id": corpus_ids[int(idx)],
                    "bm25_rank": -1,
                    "bm25_score": float("nan"),
                    "dense_rank": rank,
                    "dense_score": float(scores[int(idx)]),
                    "negative_source": "bge_dense",
                }
                for rank, idx in enumerate(top_indices, start=1)
            ]
            window = [
                item
                for item in ranked
                if rank_start <= int(item["dense_rank"]) < rank_end
                and str(item["doc_id"]) not in excluded
                and corpus.get(str(item["doc_id"]), "")
            ]
            fallback = [item for item in ranked if str(item["doc_id"]) not in excluded and corpus.get(str(item["doc_id"]), "")]
            candidates = window or fallback
            if candidates:
                hard_negatives[qid] = candidates
    return hard_negatives, positive_scores


def _merge_negative_candidates(
    left: dict[str, list[dict[str, Any]]],
    right: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for source in (left, right):
        for qid, candidates in source.items():
            bucket = merged.setdefault(qid, {})
            for item in candidates:
                doc_id = str(item["doc_id"])
                if doc_id not in bucket:
                    bucket[doc_id] = dict(item)
                else:
                    existing = bucket[doc_id]
                    for key in ("bm25_rank", "dense_rank"):
                        if int(existing.get(key, -1)) < 0 and int(item.get(key, -1)) >= 0:
                            existing[key] = item[key]
                    for key in ("bm25_score", "dense_score"):
                        value = float(item.get(key, float("nan")))
                        if np.isnan(float(existing.get(key, float("nan")))) and not np.isnan(value):
                            existing[key] = value
                    existing["negative_source"] = "hybrid"
    return {qid: list(items.values()) for qid, items in merged.items()}


def _candidate_score(candidate: dict[str, Any]) -> float:
    dense = float(candidate.get("dense_score", float("nan")))
    bm25 = float(candidate.get("bm25_score", float("nan")))
    if not np.isnan(dense):
        return dense
    return bm25


def _apply_score_band_filter(
    *,
    candidates: list[dict[str, Any]],
    positive_score: float | None,
    min_margin: float | None,
    max_margin: float | None,
) -> tuple[list[dict[str, Any]], int]:
    if positive_score is None or (min_margin is None and max_margin is None):
        return candidates, 0
    kept: list[dict[str, Any]] = []
    dropped = 0
    for cand in candidates:
        neg_score = _candidate_score(cand)
        if np.isnan(neg_score):
            kept.append(cand)
            continue
        margin = float(positive_score - neg_score)
        if min_margin is not None and margin < float(min_margin):
            dropped += 1
            continue
        if max_margin is not None and margin > float(max_margin):
            dropped += 1
            continue
        kept.append(cand)
    return kept, dropped


def _load_beir_records(
    beir_dir: Path,
    split: str,
    limit: int | None,
    *,
    negative_strategy: str,
    model: SentenceTransformer | None,
    teacher_batch_size: int,
    bm25_rank_start: int,
    bm25_rank_end: int,
    bm25_seed: int,
    dense_rank_start: int,
    dense_rank_end: int,
    dense_query_batch_size: int,
    score_band_min_margin: float | None,
    score_band_max_margin: float | None,
    cross_encoder_model_name: str | None,
    cross_encoder_threshold: float,
    cross_encoder_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus_rows = _read_jsonl(beir_dir / "corpus.jsonl")
    query_rows = _read_jsonl(beir_dir / "queries.jsonl")
    qrels_path = beir_dir / "qrels" / f"{split}.tsv"
    corpus = {
        str(row["_id"]): " ".join(
            part for part in [str(row.get("title", "")).strip(), str(row.get("text", "")).strip()] if part
        ).strip()
        for row in corpus_rows
    }
    queries = {str(row["_id"]): str(row.get("text", "")).strip() for row in query_rows}
    relevant: dict[str, list[str]] = {}
    with qrels_path.open("r", encoding="utf-8") as fp:
        _ = fp.readline()
        for line in fp:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                relevant.setdefault(str(parts[0]), []).append(str(parts[1]))

    corpus_ids = list(corpus.keys())
    negative_candidates: dict[str, list[dict[str, Any]]] = {}
    positive_scores: dict[str, float] = {}
    if negative_strategy in {"bm25", "hybrid"}:
        negative_candidates, positive_scores = _bm25_hard_negative_candidates(
            queries=queries,
            corpus=corpus,
            relevant=relevant,
            rank_start=bm25_rank_start,
            rank_end=bm25_rank_end,
            seed=bm25_seed,
        )
    if negative_strategy in {"bge_dense", "hybrid"}:
        if model is None:
            raise ValueError(f"negative_strategy={negative_strategy} requires a teacher model")
        dense_candidates, dense_positive_scores = _bge_dense_hard_negative_candidates(
            queries=queries,
            corpus=corpus,
            relevant=relevant,
            model=model,
            batch_size=teacher_batch_size,
            rank_start=dense_rank_start,
            rank_end=dense_rank_end,
            query_batch_size=dense_query_batch_size,
        )
        if negative_strategy == "hybrid":
            negative_candidates = _merge_negative_candidates(negative_candidates, dense_candidates)
            positive_scores.update(dense_positive_scores)
        else:
            negative_candidates = dense_candidates
            positive_scores = dense_positive_scores

    cross_encoder = None
    if cross_encoder_model_name:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Install sentence-transformers with CrossEncoder support to use cross-encoder filtering") from exc
        cross_encoder = CrossEncoder(cross_encoder_model_name, device=str(model.device) if model is not None else None)

    rng = np.random.default_rng(bm25_seed)
    stats: dict[str, Any] = {
        "queries_total": len(relevant),
        "queries_with_candidates": 0,
        "score_band_dropped": 0,
        "cross_encoder_dropped": 0,
        "fallback_samples": 0,
    }

    records: list[dict[str, Any]] = []
    for qid, pos_ids in relevant.items():
        query = queries.get(qid, "")
        if not query:
            continue
        for pos_id in pos_ids:
            positive = corpus.get(pos_id, "")
            if not positive:
                continue
            negative = ""
            negative_id = ""
            negative_rank = -1
            negative_bm25_rank = -1
            negative_dense_rank = -1
            negative_bm25_score = float("nan")
            negative_dense_score = float("nan")
            selected_negative_strategy = negative_strategy
            candidates = negative_candidates.get(qid, [])
            if candidates:
                stats["queries_with_candidates"] += 1
                candidates, dropped_by_band = _apply_score_band_filter(
                    candidates=candidates,
                    positive_score=positive_scores.get(qid),
                    min_margin=score_band_min_margin,
                    max_margin=score_band_max_margin,
                )
                stats["score_band_dropped"] += int(dropped_by_band)
                if cross_encoder is not None and candidates:
                    pairs = [(query, corpus.get(str(item["doc_id"]), "")) for item in candidates]
                    ce_scores = cross_encoder.predict(pairs, batch_size=cross_encoder_batch_size, show_progress_bar=False)
                    kept: list[dict[str, Any]] = []
                    for item, ce_score in zip(candidates, ce_scores):
                        item = dict(item)
                        item["ce_score"] = float(ce_score)
                        if float(ce_score) < cross_encoder_threshold:
                            kept.append(item)
                    stats["cross_encoder_dropped"] += int(len(candidates) - len(kept))
                    if kept:
                        candidates = kept
            if candidates:
                candidate = candidates[int(rng.integers(0, len(candidates)))]
                negative_id = str(candidate["doc_id"])
                negative_bm25_rank = int(candidate.get("bm25_rank", -1))
                negative_dense_rank = int(candidate.get("dense_rank", -1))
                negative_rank = negative_dense_rank if negative_dense_rank >= 0 else negative_bm25_rank
                negative_bm25_score = float(candidate.get("bm25_score", float("nan")))
                negative_dense_score = float(candidate.get("dense_score", float("nan")))
                selected_negative_strategy = str(candidate.get("negative_source", negative_strategy))
                ce_score = float(candidate.get("ce_score", float("nan")))
                negative = corpus.get(negative_id, "")
            else:
                ce_score = float("nan")
            if not negative:
                for cand_id in corpus_ids:
                    if cand_id not in pos_ids and corpus.get(cand_id, ""):
                        negative_id = cand_id
                        negative = corpus[cand_id]
                        selected_negative_strategy = "fallback"
                        stats["fallback_samples"] += 1
                        break
            records.append(
                {
                    "query": query,
                    "positive_doc": positive,
                    "negative_doc": negative,
                    "sample_id": f"{beir_dir.name}:{qid}:{pos_id}",
                    "source": beir_dir.name,
                    "query_id": qid,
                    "positive_doc_id": pos_id,
                    "negative_doc_id": negative_id,
                    "negative_rank": negative_rank,
                    "negative_bm25_rank": negative_bm25_rank,
                    "negative_dense_rank": negative_dense_rank,
                    "negative_bm25_score": negative_bm25_score,
                    "negative_dense_score": negative_dense_score,
                    "negative_ce_score": ce_score,
                    "positive_score": float(positive_scores.get(qid, float("nan"))),
                    "negative_strategy": selected_negative_strategy,
                }
            )
            if limit and len(records) >= limit:
                return records, stats
    return records, stats


def _encode_prefixed(
    model: SentenceTransformer,
    texts: list[str],
    *,
    prefix: str,
    batch_size: int,
) -> np.ndarray:
    prefixed = [prefix + text for text in texts]
    return np.asarray(
        model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def prepare_query_doc_records(
    *,
    input_json: Path | None,
    beir_dir: Path | None,
    beir_split: str,
    max_samples: int,
    negative_strategy: str,
    model_name: str,
    device: str,
    max_seq_length: int,
    teacher_batch_size: int,
    bm25_rank_start: int,
    bm25_rank_end: int,
    bm25_seed: int,
    dense_rank_start: int,
    dense_rank_end: int,
    dense_query_batch_size: int,
    score_band_min_margin: float | None,
    score_band_max_margin: float | None,
    cross_encoder_model_name: str | None,
    cross_encoder_threshold: float,
    cross_encoder_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mining_stats: dict[str, Any] = {}
    if input_json:
        records = _load_json_records(input_json, max_samples)
        mining_stats["data_source"] = "json"
    elif beir_dir:
        model: SentenceTransformer | None = None
        if negative_strategy in {"bge_dense", "hybrid"}:
            model = SentenceTransformer(model_name, device=device)
            model.max_seq_length = max_seq_length
        records, mining_stats = _load_beir_records(
            beir_dir,
            beir_split,
            max_samples,
            negative_strategy=negative_strategy,
            model=model,
            teacher_batch_size=teacher_batch_size,
            bm25_rank_start=bm25_rank_start,
            bm25_rank_end=bm25_rank_end,
            bm25_seed=bm25_seed,
            dense_rank_start=dense_rank_start,
            dense_rank_end=dense_rank_end,
            dense_query_batch_size=dense_query_batch_size,
            score_band_min_margin=score_band_min_margin,
            score_band_max_margin=score_band_max_margin,
            cross_encoder_model_name=cross_encoder_model_name,
            cross_encoder_threshold=cross_encoder_threshold,
            cross_encoder_batch_size=cross_encoder_batch_size,
        )
        mining_stats["data_source"] = str(beir_dir)
    else:
        records = _load_msmarco_triplets(max_samples)
        mining_stats["data_source"] = "msmarco_triplets"
        if negative_strategy != "random":
            mining_stats["warning"] = (
                "MSMARCO triplets path does not support corpus-level mining. "
                "Using passage-provided negatives from microsoft/ms_marco v1.1."
            )
        for row in records:
            row["negative_strategy"] = "msmarco_passage_negative"
    if not records:
        raise ValueError("No valid query/document records found")
    return records, mining_stats


def compute_teacher_embeddings_for_records(
    *,
    records: list[dict[str, Any]],
    model_name: str,
    device: str,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, Any]:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    queries = [r["query"] for r in records]
    positives = [r["positive_doc"] for r in records]
    negatives = [r["negative_doc"] for r in records]
    return {
        "query_embeddings": _encode_prefixed(model, queries, prefix=BGE_QUERY_PREFIX, batch_size=batch_size),
        "positive_embeddings": _encode_prefixed(model, positives, prefix=BGE_DOCUMENT_PREFIX, batch_size=batch_size),
        "negative_embeddings": _encode_prefixed(model, negatives, prefix=BGE_DOCUMENT_PREFIX, batch_size=batch_size),
    }


def _safe_quantiles(values: np.ndarray, percentiles: tuple[int, ...]) -> dict[str, float] | None:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return None
    out: dict[str, float] = {}
    for q in percentiles:
        out[f"p{q}"] = float(np.percentile(valid, q))
    return out


def _validate_generation_summary(summary: dict[str, Any]) -> None:
    if int(summary.get("samples", 0)) <= 0:
        raise ValueError("summary validation failed: samples must be > 0")
    if int(summary.get("actual_samples", 0)) <= 0:
        raise ValueError("summary validation failed: actual_samples must be > 0")
    if int(summary.get("embedding_dim", 0)) <= 0:
        raise ValueError("summary validation failed: embedding_dim must be > 0")
    if not summary.get("negative_sources"):
        raise ValueError("summary validation failed: negative_sources must be non-empty")


def write_query_doc_teacher_h5(
    *,
    output_path: Path,
    records: list[dict[str, Any]],
    embeddings: dict[str, Any],
    model_name: str,
    requested_max_samples: int,
    requested_negative_strategy: str,
    bm25_rank_start: int,
    bm25_rank_end: int,
    dense_rank_start: int,
    dense_rank_end: int,
    score_band_min_margin: float | None,
    score_band_max_margin: float | None,
    cross_encoder_model_name: str | None,
    cross_encoder_threshold: float,
    cross_encoder_batch_size: int,
    mining_stats: dict[str, Any],
) -> dict[str, Any]:
    queries = [r["query"] for r in records]
    positives = [r["positive_doc"] for r in records]
    negatives = [r["negative_doc"] for r in records]
    sample_ids = [r["sample_id"] for r in records]
    sources = [r["source"] for r in records]
    query_ids = [r.get("query_id", "") for r in records]
    positive_doc_ids = [r.get("positive_doc_id", "") for r in records]
    negative_doc_ids = [r.get("negative_doc_id", "") for r in records]
    negative_ranks = np.array([int(r.get("negative_rank", -1)) for r in records], dtype=np.int32)
    negative_bm25_ranks = np.array([int(r.get("negative_bm25_rank", -1)) for r in records], dtype=np.int32)
    negative_dense_ranks = np.array([int(r.get("negative_dense_rank", -1)) for r in records], dtype=np.int32)
    negative_bm25_scores = np.array([float(r.get("negative_bm25_score", np.nan)) for r in records], dtype=np.float32)
    negative_dense_scores = np.array([float(r.get("negative_dense_score", np.nan)) for r in records], dtype=np.float32)
    negative_ce_scores = np.array([float(r.get("negative_ce_score", np.nan)) for r in records], dtype=np.float32)
    positive_scores = np.array([float(r.get("positive_score", np.nan)) for r in records], dtype=np.float32)
    negative_strategies = [r.get("negative_strategy", requested_negative_strategy) for r in records]
    actual_negative_scores = np.where(np.isnan(negative_dense_scores), negative_bm25_scores, negative_dense_scores)
    margins = positive_scores - actual_negative_scores
    bm25_rank_mask = negative_bm25_ranks >= 0
    dense_rank_mask = negative_dense_ranks >= 0

    query_embeddings = np.asarray(embeddings["query_embeddings"], dtype=np.float32)
    positive_embeddings = np.asarray(embeddings["positive_embeddings"], dtype=np.float32)
    negative_embeddings = np.asarray(embeddings["negative_embeddings"], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("queries", data=np.array(queries, dtype=object), dtype=string_dtype)
        h5.create_dataset("positive_docs", data=np.array(positives, dtype=object), dtype=string_dtype)
        h5.create_dataset("negative_docs", data=np.array(negatives, dtype=object), dtype=string_dtype)
        h5.create_dataset("query_embeddings", data=query_embeddings)
        h5.create_dataset("positive_embeddings", data=positive_embeddings)
        h5.create_dataset("negative_embeddings", data=negative_embeddings)
        h5.create_dataset("sample_ids", data=np.array(sample_ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("sources", data=np.array(sources, dtype=object), dtype=string_dtype)
        h5.create_dataset("query_ids", data=np.array(query_ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("positive_doc_ids", data=np.array(positive_doc_ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("negative_doc_ids", data=np.array(negative_doc_ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("negative_ranks", data=negative_ranks)
        h5.create_dataset("negative_bm25_ranks", data=negative_bm25_ranks)
        h5.create_dataset("negative_dense_ranks", data=negative_dense_ranks)
        h5.create_dataset("negative_bm25_scores", data=negative_bm25_scores)
        h5.create_dataset("negative_dense_scores", data=negative_dense_scores)
        h5.create_dataset("negative_ce_scores", data=negative_ce_scores)
        h5.create_dataset("positive_scores", data=positive_scores)
        h5.create_dataset("negative_strategies", data=np.array(negative_strategies, dtype=object), dtype=string_dtype)
        h5.attrs["teacher_model_name"] = model_name
        h5.attrs["query_prefix"] = BGE_QUERY_PREFIX
        h5.attrs["document_prefix"] = BGE_DOCUMENT_PREFIX
        h5.attrs["normalized"] = True
        h5.attrs["negative_strategy"] = requested_negative_strategy
        h5.attrs["bm25_rank_start"] = bm25_rank_start
        h5.attrs["bm25_rank_end"] = bm25_rank_end
        h5.attrs["dense_rank_start"] = dense_rank_start
        h5.attrs["dense_rank_end"] = dense_rank_end

    actual_samples = len(records)
    source_fractions = {
        source: float(count) / float(max(1, actual_samples))
        for source, count in collections.Counter(sources).items()
    }
    fallback_count = int(collections.Counter(negative_strategies).get("fallback", 0))
    summary = {
        "path": str(output_path),
        "samples": actual_samples,
        "requested_samples": int(requested_max_samples),
        "actual_samples": actual_samples,
        "shortfall": max(0, int(requested_max_samples) - actual_samples),
        "embedding_dim": int(query_embeddings.shape[-1]),
        "sources": {source: sources.count(source) for source in sorted(set(sources))},
        "source_fractions": source_fractions,
        "negative_sources": {source: negative_strategies.count(source) for source in sorted(set(negative_strategies))},
        "negative_strategy": requested_negative_strategy,
        "bm25_rank_start": bm25_rank_start,
        "bm25_rank_end": bm25_rank_end,
        "dense_rank_start": dense_rank_start,
        "dense_rank_end": dense_rank_end,
        "negative_rank_min": int(negative_ranks[negative_ranks >= 0].min()) if np.any(negative_ranks >= 0) else None,
        "negative_rank_max": int(negative_ranks[negative_ranks >= 0].max()) if np.any(negative_ranks >= 0) else None,
        "negative_bm25_rank_min": int(negative_bm25_ranks[bm25_rank_mask].min()) if np.any(bm25_rank_mask) else None,
        "negative_bm25_rank_max": int(negative_bm25_ranks[bm25_rank_mask].max()) if np.any(bm25_rank_mask) else None,
        "negative_dense_rank_min": int(negative_dense_ranks[dense_rank_mask].min()) if np.any(dense_rank_mask) else None,
        "negative_dense_rank_max": int(negative_dense_ranks[dense_rank_mask].max()) if np.any(dense_rank_mask) else None,
        "negative_ce_score_mean": float(np.nanmean(negative_ce_scores)) if np.any(~np.isnan(negative_ce_scores)) else None,
        "positive_score_mean": float(np.nanmean(positive_scores)) if np.any(~np.isnan(positive_scores)) else None,
        "positive_score_quantiles": _safe_quantiles(positive_scores, (10, 25, 50, 75, 90)),
        "negative_score_quantiles": _safe_quantiles(actual_negative_scores, (10, 25, 50, 75, 90)),
        "margin_quantiles": _safe_quantiles(margins, (10, 25, 50, 75, 90)),
        "score_band_min_margin": score_band_min_margin,
        "score_band_max_margin": score_band_max_margin,
        "cross_encoder_model_name": cross_encoder_model_name,
        "cross_encoder_threshold": cross_encoder_threshold if cross_encoder_model_name else None,
        "cross_encoder_batch_size": cross_encoder_batch_size if cross_encoder_model_name else None,
        "fallback_rate": float(fallback_count) / float(max(1, actual_samples)),
        "mining_stats": mining_stats,
    }
    _validate_generation_summary(summary)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def generate_query_doc_teacher_embeddings(
    *,
    output_path: Path,
    input_json: Path | None,
    beir_dir: Path | None,
    beir_split: str,
    max_samples: int,
    model_name: str,
    device: str,
    batch_size: int,
    max_seq_length: int,
    negative_strategy: str,
    bm25_rank_start: int,
    bm25_rank_end: int,
    bm25_seed: int,
    dense_rank_start: int,
    dense_rank_end: int,
    dense_query_batch_size: int,
    score_band_min_margin: float | None,
    score_band_max_margin: float | None,
    cross_encoder_model_name: str | None,
    cross_encoder_threshold: float,
    cross_encoder_batch_size: int,
) -> dict[str, Any]:
    records, mining_stats = prepare_query_doc_records(
        input_json=input_json,
        beir_dir=beir_dir,
        beir_split=beir_split,
        max_samples=max_samples,
        negative_strategy=negative_strategy,
        model_name=model_name,
        device=device,
        max_seq_length=max_seq_length,
        teacher_batch_size=batch_size,
        bm25_rank_start=bm25_rank_start,
        bm25_rank_end=bm25_rank_end,
        bm25_seed=bm25_seed,
        dense_rank_start=dense_rank_start,
        dense_rank_end=dense_rank_end,
        dense_query_batch_size=dense_query_batch_size,
        score_band_min_margin=score_band_min_margin,
        score_band_max_margin=score_band_max_margin,
        cross_encoder_model_name=cross_encoder_model_name,
        cross_encoder_threshold=cross_encoder_threshold,
        cross_encoder_batch_size=cross_encoder_batch_size,
    )
    embeddings = compute_teacher_embeddings_for_records(
        records=records,
        model_name=model_name,
        device=device,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )
    return write_query_doc_teacher_h5(
        output_path=output_path,
        records=records,
        embeddings=embeddings,
        model_name=model_name,
        requested_max_samples=max_samples,
        requested_negative_strategy=negative_strategy,
        bm25_rank_start=bm25_rank_start,
        bm25_rank_end=bm25_rank_end,
        dense_rank_start=dense_rank_start,
        dense_rank_end=dense_rank_end,
        score_band_min_margin=score_band_min_margin,
        score_band_max_margin=score_band_max_margin,
        cross_encoder_model_name=cross_encoder_model_name,
        cross_encoder_threshold=cross_encoder_threshold,
        cross_encoder_batch_size=cross_encoder_batch_size,
        mining_stats=mining_stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate query/document BGE teacher embeddings")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--beir-dir", type=Path, default=None)
    parser.add_argument("--beir-split", default="test")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--negative-strategy", choices=list(_NEGATIVE_STRATEGIES), default="random")
    parser.add_argument("--bm25-rank-start", type=int, default=10)
    parser.add_argument("--bm25-rank-end", type=int, default=50)
    parser.add_argument("--bm25-seed", type=int, default=42)
    parser.add_argument("--dense-rank-start", type=int, default=10)
    parser.add_argument("--dense-rank-end", type=int, default=200)
    parser.add_argument("--dense-query-batch-size", type=int, default=64)
    parser.add_argument("--score-band-min-margin", type=float, default=None)
    parser.add_argument("--score-band-max-margin", type=float, default=None)
    parser.add_argument("--cross-encoder-model-name", default=None)
    parser.add_argument("--cross-encoder-threshold", type=float, default=0.5)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=32)
    args = parser.parse_args()

    summary = generate_query_doc_teacher_embeddings(
        output_path=args.output_path,
        input_json=args.input_json,
        beir_dir=args.beir_dir,
        beir_split=args.beir_split,
        max_samples=args.max_samples,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        negative_strategy=args.negative_strategy,
        bm25_rank_start=args.bm25_rank_start,
        bm25_rank_end=args.bm25_rank_end,
        bm25_seed=args.bm25_seed,
        dense_rank_start=args.dense_rank_start,
        dense_rank_end=args.dense_rank_end,
        dense_query_batch_size=args.dense_query_batch_size,
        score_band_min_margin=args.score_band_min_margin,
        score_band_max_margin=args.score_band_max_margin,
        cross_encoder_model_name=args.cross_encoder_model_name,
        cross_encoder_threshold=args.cross_encoder_threshold,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
