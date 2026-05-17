#!/usr/bin/env python3
"""Evaluate KILT indexes on OpenQA datasets across multiple backends.

Supported retrieval backends:
- dense_faiss: single/sharded FAISS dense indexes
- bm25: bm25s lexical index
- splade_csr: sparse SPLADE CSR index
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import re
import string
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from projected_token.io import load_jsonl, load_records, write_json, write_jsonl
from projected_token.retrieval.index.vector import l2_normalize, load_faiss_index


SFR_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
SFR_DEFAULT_QUERY_PREFIX = "Instruct: {instruction}\nQuery: {query}"
SPLADE_DEFAULT_MODEL = "naver/splade-v3"


def _merge_topk(
    prev_scores: np.ndarray,
    prev_ids: np.ndarray,
    new_scores: np.ndarray,
    new_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.concatenate([prev_scores, new_scores], axis=1)
    ids = np.concatenate([prev_ids, new_ids], axis=1)
    order = np.argpartition(-scores, kth=min(k - 1, scores.shape[1] - 1), axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, order, axis=1)
    top_ids = np.take_along_axis(ids, order, axis=1)
    sorted_order = np.argsort(-top_scores, axis=1)
    return (
        np.take_along_axis(top_scores, sorted_order, axis=1),
        np.take_along_axis(top_ids, sorted_order, axis=1),
    )


def _search_shard_worker(
    worker_id: int,
    entries: list[dict[str, Any]],
    top_k: int,
    query_embeddings: np.ndarray,
    search_query_batch_size: int,
    queue: mp.Queue,
) -> None:
    import faiss

    try:
        num_queries = int(query_embeddings.shape[0])
        best_scores = np.full((num_queries, top_k), -np.inf, dtype=np.float32)
        best_ids = np.full((num_queries, top_k), -1, dtype=np.int64)

        batch_size = max(1, int(search_query_batch_size))
        for entry in entries:
            shard_path = str(entry["path"])
            shard_index = faiss.read_index(shard_path)
            for start in range(0, num_queries, batch_size):
                end = min(start + batch_size, num_queries)
                scores, ids = shard_index.search(query_embeddings[start:end], top_k)
                ids = ids.astype(np.int64, copy=False)
                merged_scores, merged_ids = _merge_topk(
                    best_scores[start:end],
                    best_ids[start:end],
                    scores,
                    ids,
                    top_k,
                )
                best_scores[start:end] = merged_scores
                best_ids[start:end] = merged_ids
                queue.put({"type": "progress", "worker_id": worker_id, "delta": 1.0})

        queue.put({"type": "result", "worker_id": worker_id, "scores": best_scores, "ids": best_ids})
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        queue.put({"type": "error", "worker_id": worker_id, "error": f"{exc.__class__.__name__}: {exc}"})


def _parse_top_k(raw_top_k: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw_top_k).split(",") if item.strip()]
    if not values:
        raise ValueError("--top-k must include at least one integer.")
    if min(values) <= 0:
        raise ValueError("--top-k values must be positive.")
    return sorted(set(values))


def _normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _in_accuracy_match(pred: str, ref: str) -> bool:
    pred_norm = _normalize_text(pred)
    ref_norm = _normalize_text(ref)
    if not ref_norm:
        return False
    return ref_norm in pred_norm


def _normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return _normalize_answers(parsed)
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple, set)):
                    return _normalize_answers(list(parsed))
            except Exception:
                pass
        normalized = _normalize_text(raw)
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized = _normalize_text(item)
                if normalized:
                    out.append(normalized)
        return out
    return []


def _popqa_targets_from_row(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    obj = str(row.get("obj", "") or "").strip()
    if obj:
        out.append(obj)
    possible_answers = row.get("possible_answers")
    if possible_answers is None:
        return out
    if isinstance(possible_answers, str):
        for part in possible_answers.split("|"):
            token = str(part or "").strip()
            if token and token not in out:
                out.append(token)
    elif isinstance(possible_answers, (list, tuple)):
        for item in possible_answers:
            token = str(item or "").strip()
            if token and token not in out:
                out.append(token)
    return out


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _pick_field(record: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    for key in candidates:
        if key in record:
            return record[key]
    return default


@dataclass
class QueryCase:
    query: str
    answers: list[str]
    metadata: dict[str, Any]


def load_popqa_cases(
    *,
    dataset_path: str | None,
    dataset_name: str,
    split: str,
    question_col: str,
    max_queries: int | None,
    hf_cache_dir: str | None,
) -> list[QueryCase]:
    if dataset_path:
        records = load_records(dataset_path)
        cases: list[QueryCase] = []
        iterator = records if max_queries is None else records[: max(0, int(max_queries))]
        for idx, row in enumerate(iterator):
            query = str(_pick_field(row, [question_col, "query", "question"], default="") or "").strip()
            if not query:
                continue
            answers = _popqa_targets_from_row(row)
            cases.append(QueryCase(query=query, answers=answers, metadata={"row_index": idx}))
        return cases

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, cache_dir=hf_cache_dir)
    if max_queries is not None:
        ds = ds.select(range(min(int(max_queries), len(ds))))
    cases: list[QueryCase] = []
    for idx, row_raw in enumerate(ds):
        row = dict(row_raw)
        query = str(_pick_field(row, [question_col, "query", "question"], default="") or "").strip()
        if not query:
            continue
        answers = _popqa_targets_from_row(row)
        cases.append(QueryCase(query=query, answers=answers, metadata={"row_index": idx}))
    return cases


def load_hotpot_cases(
    *,
    mode: str,
    split: str,
    max_queries: int | None,
    hf_cache_dir: str | None,
) -> list[QueryCase]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", mode, split=split, cache_dir=hf_cache_dir)
    if max_queries is not None:
        ds = ds.select(range(min(int(max_queries), len(ds))))

    cases: list[QueryCase] = []
    for idx, row in enumerate(ds):
        query = str(row.get("question", "") or "").strip()
        if not query:
            continue
        answers = _normalize_answers(row.get("answer"))
        cases.append(
            QueryCase(
                query=query,
                answers=answers,
                metadata={"row_index": idx, "id": row.get("id"), "mode": mode},
            )
        )
    return cases


class DenseHFQueryEncoder:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        max_length: int,
        query_prefix_template: str | None,
        instruction: str,
        pooling: str,
    ) -> None:
        self.device = device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device).eval()
        self.max_length = int(max_length)
        self.query_prefix_template = query_prefix_template
        self.instruction = instruction
        self.pooling = pooling
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0))

    @staticmethod
    def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[batch_indices, sequence_lengths]

    @staticmethod
    def _mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        summed = (last_hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _pool(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last_token":
            return self._last_token_pool(outputs.last_hidden_state, attention_mask)
        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0]
        if self.pooling == "mean":
            return self._mean_pool(outputs.last_hidden_state, attention_mask)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def _format_query(self, query: str) -> str:
        if not self.query_prefix_template:
            return query
        return self.query_prefix_template.format(instruction=self.instruction, query=query)

    def encode(self, queries: list[str], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in tqdm(range(0, len(queries), batch_size), desc="encode-queries"):
            batch = [self._format_query(text) for text in queries[start : start + batch_size]]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                emb = self._pool(outputs, inputs["attention_mask"])
                emb = F.normalize(emb, p=2, dim=-1)
            chunks.append(emb.cpu().float().numpy().astype(np.float32, copy=False))
        if not chunks:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        return np.vstack(chunks)


class SpladeV3QueryEncoder:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        max_length: int,
        query_prefix_template: str | None,
        instruction: str,
        agg: str,
        top_n_terms: int | None,
    ) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device).eval()
        self.max_length = int(max_length)
        self.query_prefix_template = query_prefix_template
        self.instruction = instruction
        self.agg = agg
        self.top_n_terms = int(top_n_terms) if top_n_terms is not None else None
        self.vocab_size = int(getattr(self.model.config, "vocab_size", self.tokenizer.vocab_size))

    def _format_query(self, query: str) -> str:
        if not self.query_prefix_template:
            return query
        return self.query_prefix_template.format(instruction=self.instruction, query=query)

    def _sparse_aggregate(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        values = torch.log1p(torch.relu(logits))
        mask = attention_mask.unsqueeze(-1).to(values.dtype)
        values = values * mask
        if self.agg == "sum":
            doc_vec = values.sum(dim=1)
        elif self.agg == "max":
            doc_vec = values.max(dim=1).values
        else:
            raise ValueError(f"Unsupported SPLADE aggregation: {self.agg}")
        return doc_vec

    def encode(self, queries: list[str], batch_size: int):
        from scipy.sparse import csr_matrix

        indptr = [0]
        indices: list[int] = []
        data: list[float] = []

        for start in tqdm(range(0, len(queries), batch_size), desc="encode-queries-splade"):
            batch = [self._format_query(text) for text in queries[start : start + batch_size]]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                dense = self._sparse_aggregate(logits, inputs["attention_mask"]).cpu().float().numpy()

            for row in dense:
                if self.top_n_terms is not None and self.top_n_terms > 0 and row.size > self.top_n_terms:
                    top_idx = np.argpartition(-row, self.top_n_terms - 1)[: self.top_n_terms]
                    mask = np.zeros_like(row, dtype=bool)
                    mask[top_idx] = True
                    row = np.where(mask, row, 0.0)
                nz = np.flatnonzero(row > 0)
                if nz.size:
                    indices.extend(int(i) for i in nz.tolist())
                    data.extend(float(row[i]) for i in nz.tolist())
                indptr.append(len(indices))

        return csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
            shape=(len(queries), self.vocab_size),
            dtype=np.float32,
        )


class DenseFaissShardSearcher:
    def __init__(
        self,
        manifest_path: str | Path,
        search_workers: int = 1,
        search_query_batch_size: int = 256,
    ) -> None:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        entries = payload.get("entries", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Manifest has no entries: {manifest_path}")
        self.entries = entries
        self.search_workers = max(1, int(search_workers))
        self.search_query_batch_size = max(1, int(search_query_batch_size))

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        import faiss

        num_queries = int(query_embeddings.shape[0])
        best_scores = np.full((num_queries, top_k), -np.inf, dtype=np.float32)
        best_ids = np.full((num_queries, top_k), -1, dtype=np.int64)

        if self.search_workers <= 1:
            for entry in tqdm(self.entries, desc="search-shards"):
                shard_index = faiss.read_index(str(entry["path"]))
                batch_size = self.search_query_batch_size
                for start in range(0, num_queries, batch_size):
                    end = min(start + batch_size, num_queries)
                    scores, ids = shard_index.search(query_embeddings[start:end], top_k)
                    ids = ids.astype(np.int64, copy=False)
                    merged_scores, merged_ids = _merge_topk(
                        best_scores[start:end],
                        best_ids[start:end],
                        scores,
                        ids,
                        top_k,
                    )
                    best_scores[start:end] = merged_scores
                    best_ids[start:end] = merged_ids
            return best_scores, best_ids

        workers = min(self.search_workers, len(self.entries))
        chunk_size = (len(self.entries) + workers - 1) // workers
        chunks = [self.entries[i : i + chunk_size] for i in range(0, len(self.entries), chunk_size)]
        query_batches = max(1, (num_queries + self.search_query_batch_size - 1) // self.search_query_batch_size)

        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        processes: list[mp.Process] = []
        worker_bars = []
        for worker_id, chunk in enumerate(chunks):
            worker_bars.append(
                tqdm(
                    total=len(chunk) * query_batches,
                    desc=f"search-worker-{worker_id + 1}",
                    position=worker_id,
                    leave=True,
                    dynamic_ncols=True,
                    mininterval=0.2,
                )
            )
            proc = ctx.Process(
                target=_search_shard_worker,
                args=(worker_id, chunk, top_k, query_embeddings, self.search_query_batch_size, queue),
            )
            proc.start()
            processes.append(proc)

        merge_bar = tqdm(
            total=len(chunks),
            desc="merge-workers",
            position=len(chunks),
            leave=True,
            dynamic_ncols=True,
            mininterval=0.2,
        )
        finished = 0
        errors: list[str] = []
        while finished < len(chunks):
            message = queue.get()
            msg_type = message.get("type")
            worker_id = int(message.get("worker_id", -1))
            if msg_type == "progress":
                if 0 <= worker_id < len(worker_bars):
                    worker_bars[worker_id].update(int(message.get("delta", 1)))
                continue
            if msg_type == "result":
                chunk_scores = np.asarray(message["scores"], dtype=np.float32)
                chunk_ids = np.asarray(message["ids"], dtype=np.int64)
                best_scores, best_ids = _merge_topk(best_scores, best_ids, chunk_scores, chunk_ids, top_k)
                finished += 1
                merge_bar.update(1)
                continue
            if msg_type == "error":
                errors.append(str(message.get("error", "unknown worker error")))
                finished += 1
                merge_bar.update(1)

        for proc in processes:
            proc.join()
        for bar in worker_bars:
            bar.close()
        merge_bar.close()

        for proc in processes:
            if proc.exitcode not in (0, None):
                errors.append(f"search worker {proc.pid} exited with code {proc.exitcode}")
        if errors:
            raise RuntimeError("Parallel shard search failed:\n" + "\n".join(errors))
        return best_scores, best_ids


class DenseFaissSingleSearcher:
    def __init__(self, index_path: str | Path) -> None:
        self.index = load_faiss_index(index_path)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(query_embeddings.astype(np.float32), top_k)


class BM25Searcher:
    def __init__(self, index_dir: Path, metadata: dict[str, Any]) -> None:
        import bm25s

        self.index_dir = index_dir
        self.metadata = metadata
        bm25s_dir = index_dir / "bm25s_index"
        if not bm25s_dir.exists():
            raise FileNotFoundError(f"BM25 index directory not found: {bm25s_dir}")
        self.retriever = bm25s.BM25.load(str(bm25s_dir), load_corpus=False, load_vocab=True)
        self.doc_order = json.loads((index_dir / "bm25s_doc_order.json").read_text(encoding="utf-8"))
        if not isinstance(self.doc_order, list):
            raise ValueError("Invalid bm25s_doc_order.json format.")

        chunk_text_by_id: dict[str, str] = {}
        chunks_path = index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"BM25 chunks file not found: {chunks_path}")
        with chunks_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                chunk_id = str(row.get("chunk_id", ""))
                if chunk_id:
                    chunk_text_by_id[chunk_id] = str(row.get("text", "") or "")

        self.doc_texts: list[str] = [str(chunk_text_by_id.get(str(chunk_id), "")) for chunk_id in self.doc_order]
        self.stopwords = metadata.get("bm25_stopwords")
        self.stemmer_name = metadata.get("bm25_stemmer")

    def _build_stemmer(self):
        if not self.stemmer_name:
            return None
        try:
            import Stemmer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("BM25 stemmer requested but PyStemmer is not installed.") from exc
        return Stemmer.Stemmer(self.stemmer_name)

    def search(self, queries: list[str], top_k: int) -> tuple[np.ndarray, np.ndarray]:
        import bm25s

        query_tokens = bm25s.tokenize(
            queries,
            stopwords=self.stopwords,
            stemmer=self._build_stemmer(),
            return_ids=False,
            show_progress=True,
            leave=False,
        )
        scores, ids = self.retriever.retrieve(
            query_tokens,
            k=top_k,
            return_as="tuple",
            show_progress=True,
            leave_progress=False,
        )
        return np.asarray(scores, dtype=np.float32), np.asarray(ids, dtype=np.int64)

    def get_doc_text(self, local_idx: int) -> str:
        if local_idx < 0 or local_idx >= len(self.doc_texts):
            return ""
        return self.doc_texts[local_idx]

    def get_doc_external_id(self, local_idx: int) -> str:
        if local_idx < 0 or local_idx >= len(self.doc_order):
            return ""
        return str(self.doc_order[local_idx])


class SpladeCSRSearcher:
    def __init__(self, index_dir: Path) -> None:
        from scipy.sparse import load_npz

        self.index_dir = index_dir
        matrix_path = index_dir / "corpus_csr.npz"
        if not matrix_path.exists():
            raise FileNotFoundError(f"SPLADE corpus matrix not found: {matrix_path}")
        self.corpus_csr = load_npz(matrix_path).tocsr().astype(np.float32)
        row_ids_path = index_dir / "doc_row_ids.npy"
        if row_ids_path.exists():
            self.doc_row_ids = np.load(row_ids_path, allow_pickle=False).astype(np.int64)
        else:
            self.doc_row_ids = np.arange(self.corpus_csr.shape[0], dtype=np.int64)
        if int(self.doc_row_ids.shape[0]) != int(self.corpus_csr.shape[0]):
            raise ValueError("SPLADE doc_row_ids.npy size does not match corpus_csr rows.")
        self.corpus_t = self.corpus_csr.transpose().tocsr()

    def search(self, query_matrix, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_matrix = query_matrix.tocsr().astype(np.float32)
        num_queries = int(query_matrix.shape[0])
        best_scores = np.full((num_queries, top_k), -np.inf, dtype=np.float32)
        best_ids = np.full((num_queries, top_k), -1, dtype=np.int64)

        for row_idx in tqdm(range(num_queries), desc="search-splade-csr"):
            row = query_matrix.getrow(row_idx)
            scores_row = row.dot(self.corpus_t)
            if scores_row.nnz == 0:
                continue
            indices = scores_row.indices.astype(np.int64, copy=False)
            values = scores_row.data.astype(np.float32, copy=False)
            if values.size <= top_k:
                order = np.argsort(-values)
            else:
                local = np.argpartition(-values, top_k - 1)[:top_k]
                order = local[np.argsort(-values[local])]
            picked_ids = indices[order]
            picked_scores = values[order]
            limit = picked_ids.size
            best_ids[row_idx, :limit] = picked_ids
            best_scores[row_idx, :limit] = picked_scores
        return best_scores, best_ids

    def local_to_row_id(self, local_idx: int) -> int:
        if local_idx < 0 or local_idx >= int(self.doc_row_ids.shape[0]):
            return -1
        return int(self.doc_row_ids[local_idx])


class ParquetTextStore:
    def __init__(self, path: str | Path, text_column: str = "text") -> None:
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.text_column = text_column
        self.file = pq.ParquetFile(self.path)
        self.row_group_offsets: list[int] = []
        current = 0
        for idx in range(self.file.num_row_groups):
            self.row_group_offsets.append(current)
            current += self.file.metadata.row_group(idx).num_rows
        self.total_rows = current
        self._cache: dict[int, list[str]] = {}

    def _row_group_for_row(self, row_id: int) -> int:
        left = 0
        right = len(self.row_group_offsets) - 1
        while left <= right:
            mid = (left + right) // 2
            start = self.row_group_offsets[mid]
            end = self.total_rows if mid + 1 >= len(self.row_group_offsets) else self.row_group_offsets[mid + 1]
            if start <= row_id < end:
                return mid
            if row_id < start:
                right = mid - 1
            else:
                left = mid + 1
        raise IndexError(f"Row id out of range: {row_id}")

    def _load_row_group(self, rg_idx: int) -> list[str]:
        if rg_idx in self._cache:
            return self._cache[rg_idx]
        table = self.file.read_row_group(rg_idx, columns=[self.text_column])
        values = table.column(0).to_pylist()
        texts = [str(value or "") for value in values]
        self._cache[rg_idx] = texts
        if len(self._cache) > 4:
            oldest = next(iter(self._cache))
            if oldest != rg_idx:
                self._cache.pop(oldest, None)
        return texts

    def prefetch_many(self, row_ids: list[int]) -> dict[int, str]:
        target_ids = sorted(set(int(item) for item in row_ids if int(item) >= 0 and int(item) < self.total_rows))
        if not target_ids:
            return {}
        out: dict[int, str] = {}
        cursor = 0
        total = len(target_ids)
        for rg_idx in tqdm(range(self.file.num_row_groups), desc="prefetch-texts"):
            if cursor >= total:
                break
            start = self.row_group_offsets[rg_idx]
            end = self.total_rows if rg_idx + 1 >= len(self.row_group_offsets) else self.row_group_offsets[rg_idx + 1]
            left = bisect_left(target_ids, start, lo=cursor)
            if left >= total:
                break
            right = bisect_left(target_ids, end, lo=left)
            if right <= left:
                continue
            table = self.file.read_row_group(rg_idx, columns=[self.text_column])
            values = table.column(0).to_pylist()
            for row_id in target_ids[left:right]:
                local_idx = row_id - start
                if 0 <= local_idx < len(values):
                    out[row_id] = str(values[local_idx] or "")
            cursor = right
        return out


def _resolve_prepare_cache_path(index_dir: Path, run_metadata: dict[str, Any], explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    raw_path = run_metadata.get("prepare_cache_path")
    if not raw_path:
        raise ValueError("Prepare cache parquet path is missing. Pass --text-cache-path explicitly.")
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    workspace_relative = (Path.cwd() / candidate).resolve()
    if workspace_relative.exists():
        return workspace_relative
    index_relative = (index_dir / candidate).resolve()
    if index_relative.exists():
        return index_relative
    raise FileNotFoundError(
        f"Cannot resolve prepare_cache_path='{raw_path}'. Pass --text-cache-path with absolute parquet path."
    )


def _compute_metrics(relevance_flags: list[list[int]], top_k_values: list[int]) -> dict[str, float]:
    results: dict[str, float] = {}
    if not relevance_flags:
        for k in top_k_values:
            results[f"recall@{k}"] = 0.0
            results[f"ndcg@{k}"] = 0.0
        results["mrr"] = 0.0
        return results

    for k in top_k_values:
        recall_values: list[float] = []
        ndcg_values: list[float] = []
        for labels in relevance_flags:
            sliced = labels[:k]
            recall_values.append(1.0 if any(sliced) else 0.0)
            dcg = 0.0
            for rank, rel in enumerate(sliced, start=1):
                if rel:
                    dcg += 1.0 / np.log2(rank + 1)
            ideal = sorted(sliced, reverse=True)
            idcg = 0.0
            for rank, rel in enumerate(ideal, start=1):
                if rel:
                    idcg += 1.0 / np.log2(rank + 1)
            ndcg_values.append(float(dcg / idcg) if idcg else 0.0)
        results[f"recall@{k}"] = float(np.mean(recall_values))
        results[f"ndcg@{k}"] = float(np.mean(ndcg_values))

    reciprocal_ranks: list[float] = []
    for labels in relevance_flags:
        rr = 0.0
        for rank, rel in enumerate(labels, start=1):
            if rel:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    results["mrr"] = float(np.mean(reciprocal_ranks))
    return results


def _build_queries_signature(cases: list[QueryCase]) -> str:
    digest = hashlib.sha1()
    for case in cases:
        digest.update(case.query.encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_cached_topk_compatible(cases: list[QueryCase], loaded_rows: list[dict[str, Any]]) -> bool:
    if len(loaded_rows) != len(cases):
        return False
    for idx, row in enumerate(loaded_rows):
        row_query = str(row.get("query", "") or "").strip()
        if row_query != cases[idx].query:
            return False
    return True


def _default_search_cache_path(output_path: str) -> Path:
    output = Path(output_path)
    stem = output.stem if output.stem else "kilt_openqa_metrics"
    return output.with_name(f"{stem}_search_ids.npy")


def _default_search_cache_meta_path(search_cache_path: Path) -> Path:
    return search_cache_path.with_suffix(".meta.json")


def _default_search_results_jsonl_path(output_path: str) -> Path:
    output = Path(output_path)
    stem = output.stem if output.stem else "kilt_openqa_metrics"
    return output.with_name(f"{stem}_topk.jsonl")


def _load_search_cache(
    search_cache_path: Path,
    search_cache_meta_path: Path,
    *,
    num_queries: int,
    max_k: int,
    queries_signature: str,
    index_dir: str,
    retrieval_backend: str,
) -> np.ndarray | None:
    if not search_cache_path.exists() or not search_cache_meta_path.exists():
        return None
    try:
        meta = json.loads(search_cache_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        int(meta.get("num_queries", -1)) != int(num_queries)
        or int(meta.get("max_k", -1)) != int(max_k)
        or str(meta.get("queries_signature", "")) != queries_signature
        or str(meta.get("index_dir", "")) != index_dir
        or str(meta.get("retrieval_backend", "")) != retrieval_backend
    ):
        return None
    array = np.load(search_cache_path, allow_pickle=False)
    if array.shape != (num_queries, max_k):
        return None
    return np.asarray(array, dtype=np.int64)


def _save_search_cache(
    search_cache_path: Path,
    search_cache_meta_path: Path,
    *,
    retrieved_ids: np.ndarray,
    num_queries: int,
    max_k: int,
    queries_signature: str,
    index_dir: str,
    retrieval_backend: str,
) -> None:
    search_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(search_cache_path, np.asarray(retrieved_ids, dtype=np.int64), allow_pickle=False)
    meta = {
        "num_queries": int(num_queries),
        "max_k": int(max_k),
        "queries_signature": queries_signature,
        "index_dir": index_dir,
        "retrieval_backend": retrieval_backend,
        "search_cache_path": str(search_cache_path),
    }
    write_json(search_cache_meta_path, meta)


def _infer_backend(index_dir: Path, cli_value: str | None, metadata: dict[str, Any]) -> str:
    if cli_value:
        return str(cli_value)
    from_meta = metadata.get("retrieval_backend")
    if from_meta:
        return str(from_meta)
    if (index_dir / "index_shards_manifest.json").exists() or (index_dir / "index.faiss").exists():
        return "dense_faiss"
    if (index_dir / "bm25s_index").exists():
        return "bm25"
    if (index_dir / "corpus_csr.npz").exists():
        return "splade_csr"
    raise ValueError("Cannot infer retrieval backend from index directory. Pass --retrieval-backend explicitly.")


def _resolve_dense_model(args: argparse.Namespace, metadata: dict[str, Any]) -> str:
    return (
        args.model_name_or_path
        or metadata.get("model_name_or_path")
        or "Salesforce/SFR-Embedding-Mistral"
    )


def _resolve_dense_pooling(args: argparse.Namespace, encoder_type: str, model_name: str) -> str:
    if args.pooling:
        return str(args.pooling)
    if encoder_type == "sfr":
        return "last_token"
    if encoder_type == "bge":
        return "cls"
    if "sfr-embedding-mistral" in model_name.lower():
        return "last_token"
    if "bge" in model_name.lower():
        return "cls"
    return "mean"


def _resolve_query_template(args: argparse.Namespace, encoder_type: str, model_name: str) -> str | None:
    if args.disable_query_prefix:
        return None
    if args.query_prefix_template is not None:
        return args.query_prefix_template
    lowered = model_name.lower()
    if encoder_type == "sfr" or "sfr-embedding-mistral" in lowered:
        return SFR_DEFAULT_QUERY_PREFIX
    return None


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    index_dir = Path(args.index_dir).resolve()
    metadata_path = index_dir / "metadata.json"
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    retrieval_backend = _infer_backend(index_dir, args.retrieval_backend, run_metadata)
    top_k_values = _parse_top_k(args.top_k)
    max_k = max(top_k_values)

    if args.dataset == "popqa":
        cases = load_popqa_cases(
            dataset_path=args.dataset_path,
            dataset_name=args.popqa_dataset_name,
            split=args.popqa_split,
            question_col=args.question_col,
            max_queries=args.max_queries,
            hf_cache_dir=args.hf_cache_dir,
        )
    elif args.dataset == "hotpotqa_distractor":
        cases = load_hotpot_cases(mode="distractor", split=args.hotpot_split, max_queries=args.max_queries, hf_cache_dir=args.hf_cache_dir)
    else:
        cases = load_hotpot_cases(mode="fullwiki", split=args.hotpot_split, max_queries=args.max_queries, hf_cache_dir=args.hf_cache_dir)

    if not cases:
        raise ValueError("No valid query cases loaded.")
    if args.dataset.startswith("hotpotqa") and not any(case.answers for case in cases):
        raise ValueError(
            "No gold answers found for HotpotQA split. "
            "Use --hotpot-split validation for metric computation (test split has no labels)."
        )
    if args.sample_ratio is not None:
        if not (0 < float(args.sample_ratio) <= 1.0):
            raise ValueError("--sample-ratio must be in (0, 1].")
        keep = max(1, int(len(cases) * float(args.sample_ratio)))
        cases = cases[:keep]

    search_results_jsonl_path = (
        Path(args.search_results_jsonl_path) if args.search_results_jsonl_path else _default_search_results_jsonl_path(args.output_path)
    )
    search_cache_path = Path(args.search_cache_path) if args.search_cache_path else _default_search_cache_path(args.output_path)
    search_cache_meta_path = _default_search_cache_meta_path(search_cache_path)

    topk_rows: list[dict[str, Any]] | None = None
    index_layout = retrieval_backend
    effective_model_name_or_path: str | None = None
    effective_query_prefix_template: str | None = None
    effective_instruction: str | None = None
    effective_encoder_type: str | None = None

    if bool(args.use_search_results_jsonl) and search_results_jsonl_path.exists():
        loaded_rows = load_jsonl(search_results_jsonl_path)
        if _is_cached_topk_compatible(cases, loaded_rows):
            patched_rows: list[dict[str, Any]] = []
            for row_idx, row in enumerate(loaded_rows):
                target_terms = list(cases[row_idx].answers)
                patched = dict(row)
                patched["answers"] = target_terms
                patched["targets"] = target_terms
                patched_rows.append(patched)
            topk_rows = patched_rows
            print(f"Loaded top-k search results: {search_results_jsonl_path}")
        else:
            print(
                "Cached top-k jsonl is incompatible with current queries "
                f"(path={search_results_jsonl_path}); rebuilding retrieval."
            )

    if topk_rows is None:
        queries_signature = _build_queries_signature(cases)
        retrieved_ids = None
        cached_backend = retrieval_backend
        if bool(args.use_search_cache):
            retrieved_ids = _load_search_cache(
                search_cache_path,
                search_cache_meta_path,
                num_queries=len(cases),
                max_k=max_k,
                queries_signature=queries_signature,
                index_dir=str(index_dir),
                retrieval_backend=cached_backend,
            )
            if retrieved_ids is not None:
                print(f"Loaded search cache: {search_cache_path}")

        doc_id_matrix: np.ndarray | None = None
        doc_text_by_id: dict[int, str] = {}
        external_doc_id_by_ranked_id: dict[int, str] = {}

        if retrieval_backend == "dense_faiss":
            model_name_or_path = _resolve_dense_model(args, run_metadata)
            encoder_type = args.encoder_type
            if encoder_type == "auto":
                encoder_type = "sfr" if "sfr-embedding-mistral" in model_name_or_path.lower() else "hf_dense"
            query_prefix_template = _resolve_query_template(args, encoder_type, model_name_or_path)
            instruction = args.instruction if args.instruction is not None else SFR_DEFAULT_INSTRUCTION
            pooling = _resolve_dense_pooling(args, encoder_type, model_name_or_path)

            encoder = DenseHFQueryEncoder(
                model_name_or_path=model_name_or_path,
                device=args.device,
                max_length=int(args.max_length),
                query_prefix_template=query_prefix_template,
                instruction=instruction,
                pooling=pooling,
            )
            query_embeddings = encoder.encode([case.query for case in cases], int(args.batch_size))
            query_embeddings = l2_normalize(query_embeddings.astype(np.float32, copy=False))

            manifest_path = index_dir / "index_shards_manifest.json"
            if manifest_path.exists():
                searcher: DenseFaissShardSearcher | DenseFaissSingleSearcher = DenseFaissShardSearcher(
                    manifest_path,
                    search_workers=int(args.search_workers),
                    search_query_batch_size=int(args.search_query_batch_size),
                )
                index_layout = "dense_faiss_sharded"
            else:
                searcher = DenseFaissSingleSearcher(index_dir / "index.faiss")
                index_layout = "dense_faiss_single"

            if retrieved_ids is None:
                _, retrieved_ids = searcher.search(query_embeddings, max_k)
                retrieved_ids = np.asarray(retrieved_ids, dtype=np.int64)
                if bool(args.save_search_cache):
                    _save_search_cache(
                        search_cache_path,
                        search_cache_meta_path,
                        retrieved_ids=retrieved_ids,
                        num_queries=len(cases),
                        max_k=max_k,
                        queries_signature=queries_signature,
                        index_dir=str(index_dir),
                        retrieval_backend=retrieval_backend,
                    )
            doc_id_matrix = np.asarray(retrieved_ids, dtype=np.int64)
            effective_model_name_or_path = model_name_or_path
            effective_query_prefix_template = query_prefix_template
            effective_instruction = instruction if query_prefix_template else None
            effective_encoder_type = encoder_type

            text_cache_path = _resolve_prepare_cache_path(index_dir, run_metadata, args.text_cache_path)
            text_store = ParquetTextStore(text_cache_path, text_column=args.text_col)
            all_row_ids = doc_id_matrix[doc_id_matrix >= 0].reshape(-1).tolist()
            doc_text_by_id = text_store.prefetch_many(all_row_ids)
            external_doc_id_by_ranked_id = {int(doc_id): str(int(doc_id)) for doc_id in set(all_row_ids)}

        elif retrieval_backend == "bm25":
            searcher = BM25Searcher(index_dir=index_dir, metadata=run_metadata)
            if retrieved_ids is None:
                _, retrieved_ids = searcher.search([case.query for case in cases], max_k)
                retrieved_ids = np.asarray(retrieved_ids, dtype=np.int64)
                if bool(args.save_search_cache):
                    _save_search_cache(
                        search_cache_path,
                        search_cache_meta_path,
                        retrieved_ids=retrieved_ids,
                        num_queries=len(cases),
                        max_k=max_k,
                        queries_signature=queries_signature,
                        index_dir=str(index_dir),
                        retrieval_backend=retrieval_backend,
                    )
            doc_id_matrix = np.asarray(retrieved_ids, dtype=np.int64)
            unique_local_ids = sorted(set(int(item) for item in doc_id_matrix.reshape(-1).tolist() if int(item) >= 0))
            doc_text_by_id = {idx: searcher.get_doc_text(idx) for idx in unique_local_ids}
            external_doc_id_by_ranked_id = {idx: searcher.get_doc_external_id(idx) for idx in unique_local_ids}
            index_layout = "bm25"
            effective_encoder_type = "bm25"

        elif retrieval_backend == "splade_csr":
            model_name_or_path = args.model_name_or_path or run_metadata.get("model_name_or_path") or SPLADE_DEFAULT_MODEL
            encoder_type = "splade_v3" if args.encoder_type == "auto" else args.encoder_type
            if encoder_type != "splade_v3":
                raise ValueError("For splade_csr backend, --encoder-type must be splade_v3 or auto.")
            query_prefix_template = _resolve_query_template(args, encoder_type, model_name_or_path)
            instruction = args.instruction or ""

            encoder = SpladeV3QueryEncoder(
                model_name_or_path=model_name_or_path,
                device=args.device,
                max_length=int(args.max_length),
                query_prefix_template=query_prefix_template,
                instruction=instruction,
                agg=args.splade_agg,
                top_n_terms=args.splade_top_n_terms,
            )
            if retrieved_ids is None:
                query_matrix = encoder.encode([case.query for case in cases], int(args.batch_size))
                searcher = SpladeCSRSearcher(index_dir=index_dir)
                _, retrieved_ids = searcher.search(query_matrix, max_k)
                retrieved_ids = np.asarray(retrieved_ids, dtype=np.int64)
                if bool(args.save_search_cache):
                    _save_search_cache(
                        search_cache_path,
                        search_cache_meta_path,
                        retrieved_ids=retrieved_ids,
                        num_queries=len(cases),
                        max_k=max_k,
                        queries_signature=queries_signature,
                        index_dir=str(index_dir),
                        retrieval_backend=retrieval_backend,
                    )
            searcher = SpladeCSRSearcher(index_dir=index_dir)
            local_ids = np.asarray(retrieved_ids, dtype=np.int64)
            doc_id_matrix = np.full_like(local_ids, -1, dtype=np.int64)
            for r in range(local_ids.shape[0]):
                for c in range(local_ids.shape[1]):
                    local_id = int(local_ids[r, c])
                    if local_id >= 0:
                        doc_id_matrix[r, c] = searcher.local_to_row_id(local_id)

            text_cache_path = _resolve_prepare_cache_path(index_dir, run_metadata, args.text_cache_path)
            text_store = ParquetTextStore(text_cache_path, text_column=args.text_col)
            all_row_ids = doc_id_matrix[doc_id_matrix >= 0].reshape(-1).tolist()
            doc_text_by_id = text_store.prefetch_many(all_row_ids)
            external_doc_id_by_ranked_id = {int(doc_id): str(int(doc_id)) for doc_id in set(all_row_ids)}
            index_layout = "splade_csr"
            effective_model_name_or_path = model_name_or_path
            effective_query_prefix_template = query_prefix_template
            effective_instruction = instruction if query_prefix_template else None
            effective_encoder_type = encoder_type

        else:
            raise ValueError(f"Unsupported retrieval backend: {retrieval_backend}")

        if doc_id_matrix is None:
            raise RuntimeError("Internal error: doc_id_matrix was not built.")

        built_rows: list[dict[str, Any]] = []
        for row_idx, case in enumerate(tqdm(cases, desc="build-topk-jsonl")):
            ids = [int(item) for item in doc_id_matrix[row_idx].tolist() if int(item) >= 0]
            docs = []
            for rank, doc_id in enumerate(ids, start=1):
                docs.append(
                    {
                        "rank": rank,
                        "doc_id": external_doc_id_by_ranked_id.get(doc_id, str(doc_id)),
                        "doc_row_id": int(doc_id) if retrieval_backend in {"dense_faiss", "splade_csr"} else None,
                        "text": str(doc_text_by_id.get(doc_id, "") or ""),
                    }
                )
            built_rows.append(
                {
                    "query_index": row_idx,
                    "query": case.query,
                    "answers": case.answers,
                    "targets": case.answers,
                    "metadata": case.metadata,
                    "docs": docs,
                }
            )
        topk_rows = built_rows
        if bool(args.save_search_results_jsonl):
            write_jsonl(search_results_jsonl_path, topk_rows)
            print(f"Saved top-k search results: {search_results_jsonl_path}")

    relevance_flags: list[list[int]] = []
    relevance_flags_scored: list[list[int]] = []
    details: list[dict[str, Any]] = []
    save_details = bool(args.save_details)
    n_queries_scored = 0

    for row_idx, row in enumerate(tqdm(topk_rows, desc="judge-relevance")):
        docs = row.get("docs", [])
        answers = [str(item) for item in row.get("targets", row.get("answers", []))]
        answers = [item for item in answers if _normalize_text(item)]
        labels: list[int] = []
        for doc in docs:
            doc_text = str(doc.get("text", "") or "")
            hit = 0
            if doc_text:
                for answer in answers:
                    if _in_accuracy_match(doc_text, answer):
                        hit = 1
                        break
            labels.append(hit)
        relevance_flags.append(labels)
        if answers:
            relevance_flags_scored.append(labels)
            n_queries_scored += 1

        if save_details:
            details.append(
                {
                    "query_index": row_idx,
                    "query": row.get("query", ""),
                    "answers": answers,
                    "retrieved_ids": [doc.get("doc_id", "") for doc in docs],
                    "relevance": labels,
                    "is_scored_query": bool(answers),
                    "metadata": row.get("metadata", {}),
                }
            )

    metric_values = _compute_metrics(relevance_flags_scored, top_k_values)
    output_payload: dict[str, Any] = {
        "dataset": args.dataset,
        "num_queries": len(cases),
        "n_queries_scored": int(n_queries_scored),
        "index_dir": str(index_dir),
        "index_layout": index_layout,
        "retrieval_backend": retrieval_backend,
        "encoder_type": effective_encoder_type,
        "model_name_or_path": effective_model_name_or_path,
        "query_prefix_template": effective_query_prefix_template,
        "instruction": effective_instruction,
        "top_k": top_k_values,
        "search_cache_path": str(search_cache_path) if bool(args.save_search_cache) or bool(args.use_search_cache) else None,
        "search_results_jsonl_path": str(search_results_jsonl_path),
        "metrics": metric_values,
    }

    output_path = Path(args.output_path)
    write_json(output_path, output_payload)
    if save_details:
        details_path = Path(args.save_details_path)
        write_json(details_path, details)
        output_payload["details_path"] = str(details_path)
    return output_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate KILT index on PopQA / HotpotQA for dense/BM25/SPLADE.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    parser.add_argument("--dataset", choices=["popqa", "hotpotqa_distractor", "hotpotqa_fullwiki"], required=False, default=None)
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional local dataset file (.parquet/.jsonl/.json).")
    parser.add_argument("--index-dir", type=str, required=False, default=None)
    parser.add_argument("--output-path", type=str, required=False, default=None)
    parser.add_argument("--sample-ratio", type=float, default=None, help="Use first X fraction of queries, e.g. 0.1.")

    parser.add_argument("--question-col", type=str, default=None)
    parser.add_argument("--text-col", type=str, default=None, help="Text column in prepare cache parquet.")
    parser.add_argument("--popqa-dataset-name", type=str, default=None)
    parser.add_argument("--popqa-split", type=str, default=None)
    parser.add_argument("--hotpot-split", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--hf-cache-dir", type=str, default=None)

    parser.add_argument("--retrieval-backend", type=str, choices=["dense_faiss", "bm25", "splade_csr"], default=None)
    parser.add_argument("--encoder-type", type=str, choices=["auto", "sfr", "bge", "hf_dense", "splade_v3", "bm25"], default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--pooling", type=str, choices=["last_token", "cls", "mean"], default=None)
    parser.add_argument("--top-k", type=str, default=None, help="Comma-separated list, e.g. 1,5,10,20")
    parser.add_argument("--search-workers", type=int, default=None, help="Number of worker processes for shard search.")
    parser.add_argument("--search-query-batch-size", type=int, default=None, help="Query batch size per shard search call.")

    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--query-prefix-template", type=str, default=None)
    parser.add_argument("--disable-query-prefix", action="store_true")

    parser.add_argument("--splade-agg", type=str, choices=["max", "sum"], default=None)
    parser.add_argument("--splade-top-n-terms", type=int, default=None)

    parser.add_argument("--text-cache-path", type=str, default=None, help="Override path to prepare cache parquet with raw text.")
    parser.add_argument("--search-cache-path", type=str, default=None, help="Path to save/load retrieved ids cache (.npy).")
    parser.add_argument("--use-search-cache", action="store_true", help="Reuse saved search cache when compatible.")
    parser.add_argument("--no-use-search-cache", dest="use_search_cache", action="store_false")
    parser.add_argument("--save-search-cache", action="store_true", help="Save search results after retrieval stage.")
    parser.add_argument("--no-save-search-cache", dest="save_search_cache", action="store_false")
    parser.add_argument("--search-results-jsonl-path", type=str, default=None, help="Path to save/load top-k docs with text (.jsonl).")
    parser.add_argument("--use-search-results-jsonl", action="store_true", help="Reuse top-k docs jsonl if compatible.")
    parser.add_argument("--no-use-search-results-jsonl", dest="use_search_results_jsonl", action="store_false")
    parser.add_argument("--save-search-results-jsonl", action="store_true", help="Save top-k docs jsonl after retrieval stage.")
    parser.add_argument("--no-save-search-results-jsonl", dest="save_search_results_jsonl", action="store_false")
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--save-details-path", type=str, default=None)
    parser.set_defaults(
        use_search_cache=True,
        save_search_cache=True,
        use_search_results_jsonl=True,
        save_search_results_jsonl=True,
    )
    return parser


def _merge_args_with_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "dataset": "popqa",
        "index_dir": "artifacts/indexes/kilt_sfr_fp16",
        "output_path": "artifacts/results/retrieval/kilt_openqa_metrics.json",
        "sample_ratio": 1.0,
        "question_col": "question",
        "text_col": "text",
        "popqa_dataset_name": "akariasai/PopQA",
        "popqa_split": "test",
        "hotpot_split": "validation",
        "retrieval_backend": "dense_faiss",
        "encoder_type": "auto",
        "device": "cuda:0",
        "batch_size": 8,
        "max_length": 4096,
        "pooling": None,
        "top_k": "1,5,10,20",
        "search_workers": 1,
        "search_query_batch_size": 256,
        "instruction": SFR_DEFAULT_INSTRUCTION,
        "query_prefix_template": SFR_DEFAULT_QUERY_PREFIX,
        "splade_agg": "max",
        "splade_top_n_terms": 128,
        "save_details_path": "artifacts/results/retrieval/kilt_openqa_details.json",
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    for key in defaults:
        if key in config and getattr(args, key) == defaults[key]:
            setattr(args, key, config[key])

    nested = config.get("dataset_cfg") or {}
    if args.dataset_path is None and "path" in nested:
        args.dataset_path = nested["path"]
    if args.question_col == defaults["question_col"] and "question_col" in nested:
        args.question_col = nested["question_col"]
    return args


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = _load_yaml(args.config)
    args = _merge_args_with_config(args, cfg)
    payload = evaluate(args)

    print("Evaluation complete")
    print(f"Dataset: {payload['dataset']}")
    print(f"Queries: {payload['num_queries']}")
    print(f"Backend: {payload['retrieval_backend']}")
    print(f"Index layout: {payload['index_layout']}")
    print(f"Output: {args.output_path}")
    print("Metrics:")
    for name, value in sorted(payload["metrics"].items()):
        print(f"  {name}: {value:.6f}")
    if "details_path" in payload:
        print(f"Details: {payload['details_path']}")


if __name__ == "__main__":
    main()
