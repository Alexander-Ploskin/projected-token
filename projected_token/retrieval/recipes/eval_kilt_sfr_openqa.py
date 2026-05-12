#!/usr/bin/env python3
"""Evaluate KILT SFR index on open-QA datasets.

This evaluator is designed for KILT indexes built by `index_kilt_sfr.py`,
including massive `flat_fp16` sharded layouts (`index_shards_manifest.json`).

Relevance rule:
- A retrieved document is relevant iff any normalized answer string is a
  substring of the normalized document text.

Reported metrics:
- recall@k (query-level hit rate)
- mrr
- ndcg@k (binary gains over judged top-k list)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from projected_token.io import load_records, write_json
from projected_token.retrieval.index.vector import l2_normalize, load_faiss_index


SFR_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def _parse_top_k(raw_top_k: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw_top_k).split(",") if item.strip()]
    if not values:
        raise ValueError("--top-k must include at least one integer.")
    if min(values) <= 0:
        raise ValueError("--top-k values must be positive.")
    return sorted(set(values))


def _normalize_text(value: str) -> str:
    return " ".join(str(value).lower().split())


def _normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = _normalize_text(value)
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


def _extract_hotpot_context_text(record: dict[str, Any]) -> str:
    context = record.get("context")
    if isinstance(context, dict):
        sentences = context.get("sentences")
        if isinstance(sentences, list):
            chunks: list[str] = []
            for item in sentences:
                if isinstance(item, list):
                    joined = " ".join(str(part) for part in item)
                    if joined.strip():
                        chunks.append(joined)
                elif isinstance(item, str) and item.strip():
                    chunks.append(item)
            merged = " ".join(chunks).strip()
            if merged:
                return merged
    return ""


@dataclass
class QueryCase:
    query: str
    answers: list[str]
    metadata: dict[str, Any]


def _load_cases_from_records(
    records: list[dict[str, Any]],
    *,
    question_col: str,
    answer_col: str,
    max_queries: int | None,
) -> list[QueryCase]:
    cases: list[QueryCase] = []
    iterator = records if max_queries is None else records[: max(0, int(max_queries))]
    for idx, row in enumerate(iterator):
        query = str(_pick_field(row, [question_col, "query", "question"], default="") or "").strip()
        if not query:
            continue
        answers = _normalize_answers(_pick_field(row, [answer_col, "possible_answers", "answers", "answer", "obj"]))
        if not answers:
            continue
        cases.append(
            QueryCase(
                query=query,
                answers=answers,
                metadata={"row_index": idx},
            )
        )
    return cases


def load_popqa_cases(
    *,
    dataset_path: str | None,
    dataset_name: str,
    split: str,
    question_col: str,
    answer_col: str,
    max_queries: int | None,
    hf_cache_dir: str | None,
) -> list[QueryCase]:
    if dataset_path:
        records = load_records(dataset_path)
        return _load_cases_from_records(
            records,
            question_col=question_col,
            answer_col=answer_col,
            max_queries=max_queries,
        )

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, cache_dir=hf_cache_dir)
    if max_queries is not None:
        ds = ds.select(range(min(int(max_queries), len(ds))))
    records = [dict(row) for row in ds]
    return _load_cases_from_records(
        records,
        question_col=question_col,
        answer_col=answer_col,
        max_queries=None,
    )


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
        if not answers:
            continue
        cases.append(
            QueryCase(
                query=query,
                answers=answers,
                metadata={"row_index": idx, "id": row.get("id"), "mode": mode},
            )
        )
    return cases


class SFRQueryEncoder:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        max_length: int,
        query_prefix_template: str | None,
        instruction: str,
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

    @staticmethod
    def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Handle both left and right padding.
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[batch_indices, sequence_lengths]

    def _format_query(self, query: str) -> str:
        if not self.query_prefix_template:
            return query
        return self.query_prefix_template.format(
            instruction=self.instruction,
            query=query,
        )

    def encode(self, queries: list[str], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in tqdm(range(0, len(queries), batch_size), desc="encode-queries"):
            batch = [self._format_query(text) for text in queries[start:start + batch_size]]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                emb = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                emb = F.normalize(emb, p=2, dim=-1)
            chunks.append(emb.cpu().float().numpy().astype(np.float32, copy=False))
        if not chunks:
            return np.zeros((0, int(self.model.config.hidden_size)), dtype=np.float32)
        return np.vstack(chunks)


class FlatShardSearcher:
    """Search over many flat shard files from index_shards_manifest.json."""

    def __init__(self, manifest_path: str | Path) -> None:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        entries = payload.get("entries", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Manifest has no entries: {manifest_path}")
        self.entries = entries

    @staticmethod
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

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        import faiss

        num_queries = int(query_embeddings.shape[0])
        best_scores = np.full((num_queries, top_k), -np.inf, dtype=np.float32)
        best_ids = np.full((num_queries, top_k), -1, dtype=np.int64)

        for entry in tqdm(self.entries, desc="search-shards"):
            shard_path = str(entry["path"])
            shard_index = faiss.read_index(shard_path)
            scores, ids = shard_index.search(query_embeddings, top_k)
            ids = ids.astype(np.int64, copy=False)
            best_scores, best_ids = self._merge_topk(best_scores, best_ids, scores, ids, top_k)
        return best_scores, best_ids


class FlatIndexSearcher:
    def __init__(self, index_path: str | Path) -> None:
        self.index = load_faiss_index(index_path)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(query_embeddings.astype(np.float32), top_k)


class ParquetTextStore:
    """Lazy row-id -> text lookup from prepare cache parquet."""

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

    def get_many(self, row_ids: list[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        for row_id in sorted(set(int(item) for item in row_ids if int(item) >= 0)):
            if row_id >= self.total_rows:
                continue
            rg_idx = self._row_group_for_row(row_id)
            rg_start = self.row_group_offsets[rg_idx]
            rg_values = self._load_row_group(rg_idx)
            local_idx = row_id - rg_start
            if 0 <= local_idx < len(rg_values):
                out[row_id] = rg_values[local_idx]
        return out


def _resolve_prepare_cache_path(index_dir: Path, run_metadata: dict[str, Any], explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    raw_path = run_metadata.get("prepare_cache_path")
    if not raw_path:
        raise ValueError(
            "Prepare cache parquet path is missing. Pass --text-cache-path explicitly."
        )

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
        f"Cannot resolve prepare_cache_path='{raw_path}'. "
        "Pass --text-cache-path with absolute parquet path."
    )


def _compute_metrics(
    relevance_flags: list[list[int]],
    top_k_values: list[int],
) -> dict[str, float]:
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    index_dir = Path(args.index_dir).resolve()
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Index metadata not found: {metadata_path}")

    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    top_k_values = _parse_top_k(args.top_k)
    max_k = max(top_k_values)

    if args.dataset == "popqa":
        cases = load_popqa_cases(
            dataset_path=args.dataset_path,
            dataset_name=args.popqa_dataset_name,
            split=args.popqa_split,
            question_col=args.question_col,
            answer_col=args.answer_col,
            max_queries=args.max_queries,
            hf_cache_dir=args.hf_cache_dir,
        )
    elif args.dataset == "hotpotqa_distractor":
        cases = load_hotpot_cases(
            mode="distractor",
            split=args.hotpot_split,
            max_queries=args.max_queries,
            hf_cache_dir=args.hf_cache_dir,
        )
    else:
        cases = load_hotpot_cases(
            mode="fullwiki",
            split=args.hotpot_split,
            max_queries=args.max_queries,
            hf_cache_dir=args.hf_cache_dir,
        )

    if not cases:
        raise ValueError("No valid query cases loaded.")

    model_name_or_path = args.model_name_or_path or run_metadata.get("model_name_or_path") or "Salesforce/SFR-Embedding-Mistral"
    query_prefix_template = args.query_prefix_template
    if args.disable_query_prefix:
        query_prefix_template = None

    encoder = SFRQueryEncoder(
        model_name_or_path=model_name_or_path,
        device=args.device,
        max_length=int(args.max_length),
        query_prefix_template=query_prefix_template,
        instruction=args.instruction,
    )

    query_embeddings = encoder.encode([case.query for case in cases], int(args.batch_size))
    query_embeddings = l2_normalize(query_embeddings.astype(np.float32, copy=False))

    manifest_path = index_dir / "index_shards_manifest.json"
    if manifest_path.exists():
        searcher: FlatShardSearcher | FlatIndexSearcher = FlatShardSearcher(manifest_path)
        index_layout = "sharded"
    else:
        searcher = FlatIndexSearcher(index_dir / "index.faiss")
        index_layout = "single"

    _, retrieved_ids = searcher.search(query_embeddings, max_k)

    text_cache_path = _resolve_prepare_cache_path(index_dir, run_metadata, args.text_cache_path)
    text_store = ParquetTextStore(text_cache_path, text_column=args.text_col)

    relevance_flags: list[list[int]] = []
    details: list[dict[str, Any]] = []
    save_details = bool(args.save_details)

    for row_idx, case in enumerate(tqdm(cases, desc="judge-relevance")):
        ids = [int(item) for item in retrieved_ids[row_idx].tolist() if int(item) >= 0]
        id_to_text = text_store.get_many(ids)
        labels: list[int] = []
        for doc_id in ids:
            doc_text = _normalize_text(id_to_text.get(doc_id, ""))
            hit = 0
            if doc_text:
                for answer in case.answers:
                    if answer and answer in doc_text:
                        hit = 1
                        break
            labels.append(hit)
        relevance_flags.append(labels)

        if save_details:
            details.append(
                {
                    "query_index": row_idx,
                    "query": case.query,
                    "answers": case.answers,
                    "retrieved_ids": ids,
                    "relevance": labels,
                    "metadata": case.metadata,
                }
            )

    metric_values = _compute_metrics(relevance_flags, top_k_values)
    output_payload: dict[str, Any] = {
        "dataset": args.dataset,
        "num_queries": len(cases),
        "index_dir": str(index_dir),
        "index_layout": index_layout,
        "model_name_or_path": model_name_or_path,
        "query_prefix_template": query_prefix_template,
        "instruction": args.instruction if query_prefix_template else None,
        "top_k": top_k_values,
        "text_cache_path": str(text_cache_path),
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
    parser = argparse.ArgumentParser(
        description="Evaluate KILT SFR index on PopQA / HotpotQA.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    parser.add_argument(
        "--dataset",
        choices=["popqa", "hotpotqa_distractor", "hotpotqa_fullwiki"],
        required=False,
        default=None,
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional local dataset file (.parquet/.jsonl/.json).")
    parser.add_argument("--index-dir", type=str, required=False, default=None)
    parser.add_argument("--output-path", type=str, required=False, default=None)

    parser.add_argument("--question-col", type=str, default=None)
    parser.add_argument("--answer-col", type=str, default=None)
    parser.add_argument("--text-col", type=str, default=None, help="Text column in prepare cache parquet.")

    parser.add_argument("--popqa-dataset-name", type=str, default=None)
    parser.add_argument("--popqa-split", type=str, default=None)
    parser.add_argument("--hotpot-split", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--hf-cache-dir", type=str, default=None)

    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--top-k", type=str, default=None, help="Comma-separated list, e.g. 1,5,10,20")

    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--query-prefix-template", type=str, default=None)
    parser.add_argument("--disable-query-prefix", action="store_true")

    parser.add_argument("--text-cache-path", type=str, default=None, help="Override path to prepare cache parquet with raw text.")
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--save-details-path", type=str, default=None)
    return parser


def _merge_args_with_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "dataset": "popqa",
        "index_dir": "artifacts/indexes/kilt_sfr_fp16",
        "output_path": "artifacts/results/retrieval/kilt_sfr_openqa_metrics.json",
        "question_col": "question",
        "answer_col": "possible_answers",
        "text_col": "text",
        "popqa_dataset_name": "akariasai/PopQA",
        "popqa_split": "test",
        "hotpot_split": "validation",
        "device": "cuda:0",
        "batch_size": 8,
        "max_length": 4096,
        "top_k": "1,5,10,20",
        "instruction": SFR_DEFAULT_INSTRUCTION,
        "query_prefix_template": "Instruct: {instruction}\nQuery: {query}",
        "save_details_path": "artifacts/results/retrieval/kilt_sfr_openqa_details.json",
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
    if args.answer_col == defaults["answer_col"] and "answer_col" in nested:
        args.answer_col = nested["answer_col"]

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
    print(f"Index layout: {payload['index_layout']}")
    print(f"Output: {args.output_path}")
    print("Metrics:")
    for name, value in sorted(payload["metrics"].items()):
        print(f"  {name}: {value:.6f}")
    if "details_path" in payload:
        print(f"Details: {payload['details_path']}")


if __name__ == "__main__":
    main()
