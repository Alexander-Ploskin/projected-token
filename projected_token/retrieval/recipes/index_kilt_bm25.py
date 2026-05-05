#!/usr/bin/env python3
"""Build a BM25 (bm25s) index for s-nlp/kilt passages."""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import bm25s
import torch
import transformers
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from projected_token.io import write_json


def _load_config_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    dataset_cfg = config.get("dataset", {})
    index_cfg = config.get("index", {})
    bm25_cfg = config.get("bm25", {})
    runtime_cfg = config.get("runtime", {})
    return {
        "dataset": dataset_cfg.get("name", "s-nlp/kilt"),
        "dataset_config": dataset_cfg.get("config"),
        "split": dataset_cfg.get("split", "train"),
        "text_col": dataset_cfg.get("text_col", "text"),
        "wikipedia_id_col": dataset_cfg.get("wikipedia_id_col", "wikipedia_id"),
        "wikipedia_title_col": dataset_cfg.get("wikipedia_title_col", "wikipedia_title"),
        "source_id_col": dataset_cfg.get("source_id_col", "_id"),
        "output_dir": index_cfg.get("output_dir", "artifacts/indexes/kilt_bm25"),
        "chunk_size": int(index_cfg.get("chunk_size", 128)),
        "tokenizer_model": index_cfg.get("tokenizer_model", "bert-base-uncased"),
        "tokenizer_revision": index_cfg.get("tokenizer_revision"),
        "method": bm25_cfg.get("method", "lucene"),
        "k1": float(bm25_cfg.get("k1", 1.5)),
        "b": float(bm25_cfg.get("b", 0.75)),
        "delta": float(bm25_cfg.get("delta", 0.5)),
        "stopwords": bm25_cfg.get("stopwords"),
        "stemmer": bm25_cfg.get("stemmer"),
        "max_chunks": runtime_cfg.get("max_chunks"),
        "max_samples": runtime_cfg.get("max_samples"),
        "hf_cache_dir": runtime_cfg.get("hf_cache_dir"),
        "hf_token": runtime_cfg.get("hf_token"),
        "nfc_normalize": bool(runtime_cfg.get("nfc_normalize", False)),
        "wikipedia_id_file": runtime_cfg.get("wikipedia_id_file"),
    }


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML recipe")
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = _load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Build BM25 index for s-nlp/kilt with bm25s",
        parents=[pre_parser],
    )
    parser.add_argument("--dataset", default=defaults.get("dataset", "s-nlp/kilt"))
    parser.add_argument("--dataset-config", default=defaults.get("dataset_config"))
    parser.add_argument("--split", default=defaults.get("split", "train"))
    parser.add_argument("--text-col", default=defaults.get("text_col", "text"))
    parser.add_argument("--wikipedia-id-col", default=defaults.get("wikipedia_id_col", "wikipedia_id"))
    parser.add_argument("--wikipedia-title-col", default=defaults.get("wikipedia_title_col", "wikipedia_title"))
    parser.add_argument("--source-id-col", default=defaults.get("source_id_col", "_id"))

    parser.add_argument("--output-dir", default=defaults.get("output_dir", "artifacts/indexes/kilt_bm25"))
    parser.add_argument("--chunk-size", type=int, default=int(defaults.get("chunk_size", 128)))
    parser.add_argument("--tokenizer-model", default=defaults.get("tokenizer_model", "bert-base-uncased"))
    parser.add_argument("--tokenizer-revision", default=defaults.get("tokenizer_revision"))

    parser.add_argument("--method", default=defaults.get("method", "lucene"))
    parser.add_argument("--k1", type=float, default=float(defaults.get("k1", 1.5)))
    parser.add_argument("--b", type=float, default=float(defaults.get("b", 0.75)))
    parser.add_argument("--delta", type=float, default=float(defaults.get("delta", 0.5)))
    parser.add_argument("--stopwords", default=defaults.get("stopwords"))
    parser.add_argument("--stemmer", default=defaults.get("stemmer"))

    parser.add_argument("--max-chunks", type=int, default=defaults.get("max_chunks"))
    parser.add_argument("--max-samples", type=int, default=defaults.get("max_samples"))
    parser.add_argument("--hf-cache-dir", default=defaults.get("hf_cache_dir"))
    parser.add_argument("--hf-token", default=defaults.get("hf_token"))
    parser.add_argument(
        "--nfc-normalize",
        action="store_true",
        default=bool(defaults.get("nfc_normalize", False)),
    )
    parser.add_argument("--wikipedia-id-file", default=defaults.get("wikipedia_id_file"))
    return parser.parse_args(remaining)


def _load_allowed_wikipedia_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    source = Path(path)
    allowed: set[str] = set()
    with source.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = line.strip()
            if row:
                allowed.add(row)
    return allowed


def _chunk_id(passage_row: int, chunk_index: int) -> str:
    return f"r{passage_row:012d}_c{chunk_index:04d}"


def _token_chunks(text: str, tokenizer: Any, chunk_size: int) -> list[list[int]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    return [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]


def _iter_dataset_rows(args: argparse.Namespace):
    dataset_name = str(args.dataset)
    if dataset_name.startswith("json:"):
        pattern = dataset_name[5:]
        files = sorted(glob_module.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No JSONL files matched json: glob {pattern!r}")
        return load_dataset("json", data_files=files, split=args.split, cache_dir=args.hf_cache_dir)
    return load_dataset(
        dataset_name,
        args.dataset_config,
        split=args.split,
        cache_dir=args.hf_cache_dir,
    )


def _cleanup_output(output_dir: Path) -> None:
    for file_name in ("chunks.jsonl", "chunk_offsets.jsonl", "bm25s_doc_order.json", "metadata.json"):
        target = output_dir / file_name
        if target.exists():
            target.unlink()
    bm25s_index_dir = output_dir / "bm25s_index"
    if bm25s_index_dir.exists():
        shutil.rmtree(bm25s_index_dir)


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_output(output_dir)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        revision=args.tokenizer_revision,
        token=hf_token,
    )
    stemmer_obj = None
    if args.stemmer:
        try:
            import Stemmer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("BM25 stemmer was requested, but PyStemmer is not installed.") from exc
        stemmer_obj = Stemmer.Stemmer(args.stemmer)

    rows = _iter_dataset_rows(args)
    if args.max_samples is not None:
        sample_size = min(len(rows), int(args.max_samples))
        rows = rows.select(range(sample_size))
        print(f"Using subset: {sample_size} rows")

    allowed_wikipedia_ids = _load_allowed_wikipedia_ids(args.wikipedia_id_file)
    text_col = str(args.text_col)
    wikipedia_id_col = str(args.wikipedia_id_col)
    wikipedia_title_col = str(args.wikipedia_title_col)
    source_id_col = str(args.source_id_col)
    required_cols = [text_col, wikipedia_id_col, wikipedia_title_col]
    for col in required_cols:
        if col not in rows.column_names:
            raise ValueError(f"Column '{col}' not found in dataset columns: {rows.column_names}")

    chunks_path = output_dir / "chunks.jsonl"
    offsets_path = output_dir / "chunk_offsets.jsonl"
    doc_order_path = output_dir / "bm25s_doc_order.json"

    total_chunks = 0
    total_passages = 0
    empty_passages = 0

    class _JsonListWriter:
        def __init__(self, fp):
            self._fp = fp
            self._first = True
            self._fp.write("[")

        def append(self, value: str) -> None:
            if self._first:
                self._first = False
            else:
                self._fp.write(",")
            self._fp.write(json.dumps(value, ensure_ascii=False))

        def close(self) -> None:
            self._fp.write("]")

    with chunks_path.open("wb") as chunks_fp, offsets_path.open("w", encoding="utf-8") as offsets_fp, doc_order_path.open(
        "w", encoding="utf-8"
    ) as doc_fp:
        doc_writer = _JsonListWriter(doc_fp)

        def chunk_text_iter():
            nonlocal total_chunks, total_passages, empty_passages
            for passage_row, row in enumerate(tqdm(rows, desc="prepare-chunks")):
                raw_text = row.get(text_col)
                text = raw_text if isinstance(raw_text, str) else str(raw_text or "")
                text = text.strip()
                if not text:
                    empty_passages += 1
                    continue
                if args.nfc_normalize:
                    text = unicodedata.normalize("NFC", text)

                wikipedia_id = str(row.get(wikipedia_id_col) or "")
                if allowed_wikipedia_ids is not None and wikipedia_id not in allowed_wikipedia_ids:
                    continue

                windows = _token_chunks(text, tokenizer, int(args.chunk_size))
                if not windows:
                    empty_passages += 1
                    continue

                total_passages += 1
                source_id = str(row.get(source_id_col, passage_row))
                title = row.get(wikipedia_title_col) or ""
                title = str(title)

                for chunk_index, token_ids in enumerate(windows):
                    if args.max_chunks is not None and total_chunks >= int(args.max_chunks):
                        return
                    if not token_ids:
                        continue
                    chunk_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    chunk_id = _chunk_id(passage_row, chunk_index)
                    next_passage_row = passage_row if chunk_index + 1 < len(windows) else passage_row + 1
                    next_chunk_index = chunk_index + 1 if chunk_index + 1 < len(windows) else 0
                    record = {
                        "chunk_id": chunk_id,
                        "wikipedia_id": wikipedia_id,
                        "wikipedia_title": title,
                        "source_id": source_id,
                        "passage_row": passage_row,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "next_passage_row": next_passage_row,
                        "next_chunk_index": next_chunk_index,
                    }
                    offset = chunks_fp.tell()
                    encoded_record = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                    chunks_fp.write(encoded_record)
                    offsets_fp.write(json.dumps({"chunk_id": chunk_id, "offset": offset}) + "\n")
                    doc_writer.append(chunk_id)
                    total_chunks += 1
                    yield chunk_text

        corpus_tokens = bm25s.tokenize(
            chunk_text_iter(),
            stopwords=args.stopwords,
            stemmer=stemmer_obj,
        )
        if total_chunks == 0:
            raise ValueError("BM25 build produced zero chunks; check filters, chunk size, and dataset columns.")

        retriever = bm25s.BM25(
            method=args.method,
            k1=float(args.k1),
            b=float(args.b),
            delta=float(args.delta),
        )
        retriever.index(corpus_tokens)
        retriever.save(str(output_dir / "bm25s_index"))
        doc_writer.close()

    doc_order = json.loads(doc_order_path.read_text(encoding="utf-8"))
    if not isinstance(doc_order, list) or len(doc_order) != total_chunks:
        raise RuntimeError(
            "bm25s_doc_order.json is inconsistent with built chunks "
            f"(doc_order={len(doc_order) if isinstance(doc_order, list) else 'invalid'}, chunks={total_chunks})."
        )

    return {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_col": text_col,
        "wikipedia_id_col": wikipedia_id_col,
        "wikipedia_title_col": wikipedia_title_col,
        "source_id_col": source_id_col,
        "tokenizer_model": args.tokenizer_model,
        "tokenizer_revision": args.tokenizer_revision,
        "chunk_size": int(args.chunk_size),
        "bm25_method": args.method,
        "bm25_k1": float(args.k1),
        "bm25_b": float(args.b),
        "bm25_delta": float(args.delta),
        "bm25_stopwords": args.stopwords,
        "bm25_stemmer": args.stemmer,
        "max_chunks": args.max_chunks,
        "max_samples": args.max_samples,
        "nfc_normalize": bool(args.nfc_normalize),
        "hf_cache_dir": args.hf_cache_dir,
        "num_chunks": int(total_chunks),
        "num_passages": int(total_passages),
        "num_empty_passages": int(empty_passages),
        "artifacts": {
            "chunks_manifest": "chunks.jsonl",
            "chunk_offsets_manifest": "chunk_offsets.jsonl",
            "doc_order_manifest": "bm25s_doc_order.json",
            "bm25s_index_dir": "bm25s_index",
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    run_info = build_index(args)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "metadata.json", run_info)
    print("BM25 index build complete")
    print(f"Output dir: {output_dir}")
    print(f"Chunks: {run_info['num_chunks']}")


if __name__ == "__main__":
    main()
