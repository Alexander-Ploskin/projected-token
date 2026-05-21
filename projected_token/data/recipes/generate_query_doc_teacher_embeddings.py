#!/usr/bin/env python3
"""Generate BGE query/document teacher embeddings for retrieval distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_DOCUMENT_PREFIX = "Represent this sentence: "


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
    ds = load_dataset("sentence-transformers/msmarco", "triplets", split="train", streaming=True)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(ds, desc="msmarco", total=limit)):
        if i >= limit:
            break
        query = str(row.get("query", "")).strip()
        positive = str(row.get("positive", "")).strip()
        negative = str(row.get("negative", "")).strip()
        if query and positive:
            records.append(
                {
                    "query": query,
                    "positive_doc": positive,
                    "negative_doc": negative,
                    "sample_id": f"msmarco:{i}",
                    "source": "msmarco",
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


def _load_beir_records(beir_dir: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
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
            for cand_id in corpus_ids:
                if cand_id not in pos_ids and corpus.get(cand_id, ""):
                    negative = corpus[cand_id]
                    break
            records.append(
                {
                    "query": query,
                    "positive_doc": positive,
                    "negative_doc": negative,
                    "sample_id": f"{beir_dir.name}:{qid}:{pos_id}",
                    "source": beir_dir.name,
                }
            )
            if limit and len(records) >= limit:
                return records
    return records


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
) -> dict[str, Any]:
    if input_json:
        records = _load_json_records(input_json, max_samples)
    elif beir_dir:
        records = _load_beir_records(beir_dir, beir_split, max_samples)
    else:
        records = _load_msmarco_triplets(max_samples)
    if not records:
        raise ValueError("No valid query/document records found")

    queries = [r["query"] for r in records]
    positives = [r["positive_doc"] for r in records]
    negatives = [r["negative_doc"] for r in records]
    sample_ids = [r["sample_id"] for r in records]
    sources = [r["source"] for r in records]

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length

    query_embeddings = _encode_prefixed(model, queries, prefix=BGE_QUERY_PREFIX, batch_size=batch_size)
    positive_embeddings = _encode_prefixed(model, positives, prefix=BGE_DOCUMENT_PREFIX, batch_size=batch_size)
    negative_embeddings = _encode_prefixed(model, negatives, prefix=BGE_DOCUMENT_PREFIX, batch_size=batch_size)

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
        h5.attrs["teacher_model_name"] = model_name
        h5.attrs["query_prefix"] = BGE_QUERY_PREFIX
        h5.attrs["document_prefix"] = BGE_DOCUMENT_PREFIX
        h5.attrs["normalized"] = True

    summary = {
        "path": str(output_path),
        "samples": len(records),
        "embedding_dim": int(query_embeddings.shape[-1]),
        "sources": {source: sources.count(source) for source in sorted(set(sources))},
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


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
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
