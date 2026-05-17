#!/usr/bin/env python3
"""Precompute BM25 top-k retrieval caches for OpenQA datasets.

This is a lightweight materialization path for large bm25s indexes: it avoids
loading all chunk texts into memory and reads only retrieved chunks via
chunk_offsets.jsonl when available.
"""

from __future__ import annotations

import argparse
import json
import os
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import bm25s
import yaml
from tqdm import tqdm

from projected_token.io import write_json


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    config: str | None
    default_split: str
    question_key: str
    id_key: str | None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "popqa": DatasetSpec(
        repo_id="akariasai/PopQA",
        config=None,
        default_split="test",
        question_key="question",
        id_key="id",
    ),
    "hotpotqa_distractor": DatasetSpec(
        repo_id="hotpotqa/hotpot_qa",
        config="distractor",
        default_split="validation",
        question_key="question",
        id_key="id",
    ),
    "hotpotqa_fullwiki": DatasetSpec(
        repo_id="hotpotqa/hotpot_qa",
        config="fullwiki",
        default_split="validation",
        question_key="question",
        id_key="id",
    ),
}


@dataclass(frozen=True)
class RunConfig:
    stage1: str
    first_stage_k: int
    final_k: int
    query_batch_size: int
    bm25_stopwords: Any
    bm25_stemmer: str | None
    n_threads: int


def _norm_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _load_config_defaults(config_path: str | None) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    runtime_cfg = cfg.get("runtime", {}) or {}
    return {
        "index_dir": cfg.get("index_dir") or cfg.get("index", {}).get("dir"),
        "out": cfg.get("out") or cfg.get("output_path"),
        "hf_dataset": cfg.get("hf_dataset") or cfg.get("dataset"),
        "hf_split": cfg.get("hf_split") or cfg.get("popqa_split") or cfg.get("hotpot_split"),
        "first_stage_k": cfg.get("first_stage_k"),
        "final_k": cfg.get("final_k"),
        "query_batch_size": cfg.get("query_batch_size") or cfg.get("batch_size"),
        "max_rows": cfg.get("max_rows") or cfg.get("max_queries") or runtime_cfg.get("max_queries"),
        "hf_cache_dir": cfg.get("hf_cache_dir") or runtime_cfg.get("hf_cache_dir"),
    }


def _iter_popqa_tsv(split: str, max_rows: int | None, hf_token: str | None) -> Iterator[tuple[str, str]]:
    if split != "test":
        raise ValueError("PopQA only supports split='test' in this implementation.")
    from huggingface_hub import hf_hub_download

    tsv_path = Path(
        hf_hub_download(
            repo_id=DATASET_SPECS["popqa"].repo_id,
            repo_type="dataset",
            filename="test.tsv",
            token=hf_token,
        )
    )
    import csv

    seen: dict[str, int] = {}
    yielded = 0
    with tsv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            question = _norm_text(row.get("question"))
            if not question:
                continue
            qid = _norm_text(row.get("id")) or f"popqa_{yielded:07d}"
            if qid in seen:
                seen[qid] += 1
                qid = f"{qid}__dup{seen[qid]}"
            else:
                seen[qid] = 0
            yield qid, question
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                break


def iter_hf_questions(
    dataset_key: str,
    split: str | None,
    max_rows: int | None,
    hf_cache_dir: str | None,
    hf_token: str | None,
) -> Iterator[tuple[str, str]]:
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unknown --hf-dataset={dataset_key!r}. Expected one of {sorted(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset_key]
    effective_split = split or spec.default_split
    if dataset_key == "popqa":
        yield from _iter_popqa_tsv(effective_split, max_rows, hf_token)
        return

    from datasets import load_dataset

    ds = load_dataset(
        spec.repo_id,
        spec.config,
        split=effective_split,
        cache_dir=hf_cache_dir,
        token=hf_token,
    )
    seen: dict[str, int] = {}
    yielded = 0
    for row_raw in ds:
        row = dict(row_raw)
        question = _norm_text(row.get(spec.question_key))
        if not question:
            continue
        qid = _norm_text(row.get(spec.id_key)) if spec.id_key else None
        qid = qid or f"{dataset_key}_{yielded:07d}"
        if qid in seen:
            seen[qid] += 1
            qid = f"{qid}__dup{seen[qid]}"
        else:
            seen[qid] = 0
        yield qid, question
        yielded += 1
        if max_rows is not None and yielded >= max_rows:
            break


def _load_chunk_offsets(index_dir: Path) -> array | None:
    path = index_dir / "chunk_offsets.jsonl"
    if not path.exists():
        return None
    offsets = array("Q")
    with path.open("r", encoding="utf-8") as fp:
        for line in tqdm(fp, desc="load-chunk-offsets", unit="chunk"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            offset = row.get("offset")
            if not isinstance(offset, int):
                raise ValueError(f"Bad chunk offset row in {path}: {line[:120]}")
            offsets.append(offset)
    return offsets


class ChunkTextStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.chunks_path = index_dir / "chunks.jsonl"
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"BM25 chunks file not found: {self.chunks_path}")
        self.offsets = _load_chunk_offsets(index_dir)
        self.fp: BinaryIO | None = None
        self.doc_order: list[str] | None = None
        self.texts: dict[str, str] | None = None
        self.cache: dict[int, tuple[str, str]] = {}
        if self.offsets is None:
            doc_order_path = index_dir / "bm25s_doc_order.json"
            if not doc_order_path.exists():
                raise FileNotFoundError(f"BM25 doc order file not found: {doc_order_path}")
            doc_order = json.loads(doc_order_path.read_text(encoding="utf-8"))
            if not isinstance(doc_order, list) or not all(isinstance(item, str) for item in doc_order):
                raise ValueError(f"Invalid bm25s_doc_order.json format: {doc_order_path}")
            self.doc_order = doc_order
            self.texts = {}
            with self.chunks_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    chunk_id = str(row.get("chunk_id", "") or "")
                    if chunk_id:
                        self.texts[chunk_id] = str(row.get("text", "") or "")
        else:
            self.fp = self.chunks_path.open("rb")

    def get_by_local_id(self, local_id: int) -> tuple[str, str]:
        if local_id in self.cache:
            return self.cache[local_id]
        if self.texts is not None:
            if self.doc_order is None or local_id < 0 or local_id >= len(self.doc_order):
                return "", ""
            chunk_id = self.doc_order[local_id]
            value = (chunk_id, self.texts.get(chunk_id, ""))
            self.cache[local_id] = value
            return value
        if self.fp is None or self.offsets is None or local_id < 0 or local_id >= len(self.offsets):
            return "", ""
        offset = int(self.offsets[local_id])
        self.fp.seek(offset)
        raw = self.fp.readline()
        if not raw:
            return "", ""
        row = json.loads(raw.decode("utf-8"))
        value = (str(row.get("chunk_id", "") or ""), str(row.get("text", "") or ""))
        self.cache[local_id] = value
        return value

    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()


def _build_stemmer(stemmer_name: str | None):
    if not stemmer_name:
        return None
    try:
        import Stemmer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("BM25 stemmer requested but PyStemmer is not installed.") from exc
    return Stemmer.Stemmer(stemmer_name)


def _load_done_qids(out_path: Path, resume: bool) -> set[str]:
    if not resume or not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid")
            if qid is not None:
                done.add(str(qid))
    return done


def _write_batch(
    *,
    retriever: Any,
    text_store: ChunkTextStore,
    batch: list[tuple[str, str]],
    out_fp: Any,
    top_k: int,
    stopwords: Any,
    stemmer: Any,
    config_hash: str,
    n_threads: int,
) -> int:
    if not batch:
        return 0
    queries = [query for _, query in batch]
    query_tokens = bm25s.tokenize(
        queries,
        stopwords=stopwords,
        stemmer=stemmer,
        return_ids=False,
        show_progress=False,
        leave=False,
    )
    result = retriever.retrieve(
        query_tokens,
        k=top_k,
        return_as="tuple",
        show_progress=False,
        leave_progress=False,
        n_threads=n_threads,
    )
    documents = result.documents
    scores = result.scores
    for row_idx, (qid, question) in enumerate(batch):
        pairs: list[tuple[float, int, str, str]] = []
        for doc_id, score in zip(documents[row_idx].tolist(), scores[row_idx].tolist()):
            local_id = int(doc_id)
            chunk_id, text = text_store.get_by_local_id(local_id)
            if chunk_id:
                pairs.append((float(score), local_id, chunk_id, text))
        pairs.sort(key=lambda item: (-item[0], item[2]))
        top_pairs = pairs[:top_k]
        rec = {
            "qid": qid,
            "question": question,
            "chunk_ids": [chunk_id for _, _, chunk_id, _ in top_pairs],
            "texts": [text for _, _, _, text in top_pairs],
            "splade_scores": [score for score, _, _, _ in top_pairs],
            "deberta_scores": [0.0 for _ in top_pairs],
            "config_hash": config_hash,
        }
        out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_fp.flush()
    return len(batch)


def _config_hash(config: RunConfig, index_dir: Path, dataset_key: str, split: str) -> str:
    import hashlib

    payload = {
        "config": asdict(config),
        "index_dir": str(index_dir.resolve()),
        "dataset_key": dataset_key,
        "split": split,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def precompute(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spec = DATASET_SPECS[str(args.hf_dataset)]
    split = args.hf_split or spec.default_split
    top_k = max(int(args.first_stage_k), int(args.final_k))

    metadata_path = index_dir / "index_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    stopwords = args.stopwords if args.stopwords is not None else metadata.get("bm25s_stopwords")
    stemmer_name = args.stemmer if args.stemmer is not None else metadata.get("bm25s_stemmer")
    run_config = RunConfig(
        stage1="bm25",
        first_stage_k=int(args.first_stage_k),
        final_k=int(args.final_k),
        query_batch_size=int(args.query_batch_size),
        bm25_stopwords=stopwords,
        bm25_stemmer=stemmer_name,
        n_threads=int(args.n_threads),
    )
    config_hash = _config_hash(run_config, index_dir, str(args.hf_dataset), split)

    bm25s_dir = index_dir / "bm25s_index"
    if not bm25s_dir.exists():
        raise FileNotFoundError(f"BM25 index directory not found: {bm25s_dir}")
    retriever = bm25s.BM25.load(str(bm25s_dir), load_corpus=False, load_vocab=True)
    text_store = ChunkTextStore(index_dir)
    done = _load_done_qids(out_path, bool(args.resume))
    mode = "a" if args.resume else "w"
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    question_iter = iter_hf_questions(
        dataset_key=str(args.hf_dataset),
        split=split,
        max_rows=args.max_rows,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=token,
    )
    stemmer = _build_stemmer(stemmer_name)

    n_written = 0
    n_seen = 0
    batch: list[tuple[str, str]] = []
    try:
        with out_path.open(mode, encoding="utf-8") as out_fp:
            for qid, question in tqdm(question_iter, desc=f"precompute-bm25({args.hf_dataset})", unit="q"):
                n_seen += 1
                if qid in done:
                    continue
                batch.append((qid, question))
                if len(batch) >= int(args.query_batch_size):
                    n_written += _write_batch(
                        retriever=retriever,
                        text_store=text_store,
                        batch=batch,
                        out_fp=out_fp,
                        top_k=top_k,
                        stopwords=stopwords,
                        stemmer=stemmer,
                        config_hash=config_hash,
                        n_threads=int(args.n_threads),
                    )
                    batch.clear()
            n_written += _write_batch(
                retriever=retriever,
                text_store=text_store,
                batch=batch,
                out_fp=out_fp,
                top_k=top_k,
                stopwords=stopwords,
                stemmer=stemmer,
                config_hash=config_hash,
                n_threads=int(args.n_threads),
            )
    finally:
        text_store.close()

    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta = {
        "stage": 2,
        "cmd": "precompute-bm25-topk",
        "index_dir": str(index_dir),
        "output": str(out_path),
        "queries_seen": n_seen,
        "queries_written": n_written,
        "queries_skipped_resume": len(done) if args.resume else 0,
        "input_source": {
            "type": "hf",
            "dataset_key": args.hf_dataset,
            "repo_id": spec.repo_id,
            "config": spec.config,
            "split": split,
            "max_rows": args.max_rows,
        },
        "config": asdict(run_config),
        "config_hash": config_hash,
        "stage1_index_meta": metadata,
    }
    write_json(meta_path, meta)
    print(f"Wrote {n_written} rows to {out_path}")
    print(f"Wrote metadata to {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = _load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Precompute BM25 top-k docs for OpenQA questions.",
        parents=[pre_parser],
    )
    parser.add_argument("--index-dir", default=defaults.get("index_dir"), required=defaults.get("index_dir") is None)
    parser.add_argument("--out", default=defaults.get("out"), required=defaults.get("out") is None)
    parser.add_argument(
        "--hf-dataset",
        choices=sorted(DATASET_SPECS),
        default=defaults.get("hf_dataset"),
        required=defaults.get("hf_dataset") is None,
    )
    parser.add_argument("--hf-split", default=defaults.get("hf_split"))
    parser.add_argument("--hf-cache-dir", default=defaults.get("hf_cache_dir"))
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--max-rows", type=int, default=defaults.get("max_rows"))
    parser.add_argument("--first-stage-k", type=int, default=int(defaults.get("first_stage_k") or 10))
    parser.add_argument("--final-k", type=int, default=int(defaults.get("final_k") or 10))
    parser.add_argument("--query-batch-size", type=int, default=int(defaults.get("query_batch_size") or 512))
    parser.add_argument("--n-threads", type=int, default=0, help="bm25s retrieval threads per batch; 0 uses bm25s default.")
    parser.add_argument("--stopwords", default=None)
    parser.add_argument("--stemmer", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(remaining)


def main() -> None:
    args = build_parser()
    precompute(args)


if __name__ == "__main__":
    main()
