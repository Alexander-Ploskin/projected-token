#!/usr/bin/env python3
"""Prepare a LEAF-style multi-domain distillation corpus.

The output is a set of JSONL shards with records:
{"id": "<stable uuid>", "text": "..."}.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from tqdm import tqdm


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_id: str
    config_name: str | None
    split: str
    text_path: str
    max_samples: int


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("fineweb", "HuggingFaceFW/fineweb", None, "train", "text", 500_000),
    SourceSpec("cc_news", "cc_news", None, "train", "text", 200_000),
    SourceSpec("ms_marco", "microsoft/ms_marco", "v2.1", "train", "passages", 200_000),
    SourceSpec("pubmed_qa", "pubmed_qa", "pqa_labeled", "train", "context", 100_000),
    SourceSpec("amazon_reviews", "fancyzhx/amazon_polarity", None, "train", "content", 100_000),
    SourceSpec("trivia_qa", "trivia_qa", "rc", "train", "search.results.search_context", 50_000),
)

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def dedup_key(text: str) -> str:
    return normalize_text(text).lower()[:100]


def stable_id(source_name: str, source_index: int, text: str) -> str:
    payload = f"{source_name}:{source_index}:{dedup_key(text)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def flatten_strings(value: Any) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = normalize_text(value)
        if text:
            yield text
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten_strings(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_strings(item)
        return
    text = normalize_text(value)
    if text:
        yield text


def get_path(value: Any, dotted_path: str) -> Any:
    parts = dotted_path.split(".")

    def descend(current: Any, idx: int) -> Any:
        if idx >= len(parts):
            return current
        part = parts[idx]
        if isinstance(current, dict):
            if part in current:
                return descend(current[part], idx + 1)
            alt = part.replace("_", "")
            matching = [key for key in current if key.replace("_", "") == alt]
            if matching:
                return descend(current[matching[0]], idx + 1)
            return None
        if isinstance(current, list):
            return [descend(item, idx) for item in current]
        return None

    return descend(value, 0)


def extract_msmarco(row: dict[str, Any]) -> list[str]:
    passages = row.get("passages")
    if isinstance(passages, dict):
        for key in ("passage_text", "text", "passages"):
            if key in passages:
                return list(flatten_strings(passages[key]))
    return list(flatten_strings(passages))


def extract_pubmed(row: dict[str, Any]) -> list[str]:
    context = row.get("context")
    if isinstance(context, dict):
        for key in ("contexts", "context", "text"):
            if key in context:
                joined = " ".join(flatten_strings(context[key]))
                return [joined] if joined else []
    return list(flatten_strings(context))


def extract_triviaqa(row: dict[str, Any]) -> list[str]:
    candidates = [
        get_path(row, "search.results.search_context"),
        get_path(row, "search_results.search_context"),
        get_path(row, "search_results.search_contexts"),
        get_path(row, "search_results"),
    ]
    for candidate in candidates:
        texts = list(flatten_strings(candidate))
        if texts:
            return texts
    return []


def extract_texts(row: dict[str, Any], source: SourceSpec) -> list[str]:
    if source.name == "ms_marco":
        return extract_msmarco(row)
    if source.name == "pubmed_qa":
        return extract_pubmed(row)
    if source.name == "trivia_qa":
        return extract_triviaqa(row)
    return list(flatten_strings(get_path(row, source.text_path)))


class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_idx = 0
        self.rows_in_shard = 0
        self.handle = None

    def __enter__(self) -> "ShardWriter":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_next()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.handle is not None:
            self.handle.close()

    def _open_next(self) -> None:
        if self.handle is not None:
            self.handle.close()
        path = self.output_dir / f"shard_{self.shard_idx:04d}.jsonl"
        self.handle = path.open("w", encoding="utf-8")
        self.rows_in_shard = 0
        self.shard_idx += 1

    def write(self, record: dict[str, str]) -> None:
        if self.handle is None:
            self._open_next()
        if self.rows_in_shard >= self.shard_size:
            self._open_next()
        assert self.handle is not None
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.rows_in_shard += 1


def load_source_dataset(source: SourceSpec, *, streaming: bool):
    args: list[str] = [source.dataset_id]
    if source.config_name:
        args.append(source.config_name)
    try:
        return load_dataset(*args, split=source.split, streaming=streaming)
    except Exception:
        if source.name == "fineweb" and source.config_name is None:
            return load_dataset(source.dataset_id, "sample-10BT", split=source.split, streaming=streaming)
        raise


def scaled_sources(limit_scale: float | None, max_samples_per_source: int | None) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for source in DEFAULT_SOURCES:
        max_samples = source.max_samples
        if limit_scale is not None:
            max_samples = max(1, int(max_samples * limit_scale))
        if max_samples_per_source is not None:
            max_samples = min(max_samples, max_samples_per_source)
        sources.append(replace(source, max_samples=max_samples))
    return sources


def prepare_leaf_corpus(
    *,
    output_dir: Path,
    shard_size: int,
    min_chars: int,
    max_chars: int,
    limit_scale: float | None,
    max_samples_per_source: int | None,
    streaming: bool,
    overwrite: bool,
) -> dict[str, Any]:
    existing = sorted(output_dir.glob("shard_*.jsonl"))
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already contains shard_*.jsonl; pass --overwrite to replace them")
    if overwrite:
        for path in existing:
            path.unlink()

    seen: set[str] = set()
    total_written = 0
    per_source: dict[str, int] = {}

    with ShardWriter(output_dir, shard_size) as writer:
        for source in scaled_sources(limit_scale, max_samples_per_source):
            accepted = 0
            raw_rows = 0
            dataset = load_source_dataset(source, streaming=streaming)
            progress = tqdm(dataset, desc=f"leaf:{source.name}", unit="row")
            for row in progress:
                raw_rows += 1
                texts = extract_texts(dict(row), source)
                for text in texts:
                    text = normalize_text(text)
                    if not (min_chars <= len(text) <= max_chars):
                        continue
                    key = dedup_key(text)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    writer.write({"id": stable_id(source.name, accepted, text), "text": text})
                    accepted += 1
                    total_written += 1
                    progress.set_postfix({"accepted": accepted, "total": total_written})
                    if accepted >= source.max_samples:
                        break
                if accepted >= source.max_samples:
                    break
            per_source[source.name] = accepted
            print(f"[leaf-corpus] {source.name}: accepted={accepted} raw_rows={raw_rows}", flush=True)

    summary = {
        "output_dir": str(output_dir),
        "total_documents": total_written,
        "shard_size": shard_size,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "sources": per_source,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LEAF-style multi-domain distillation corpus")
    parser.add_argument("--output-dir", type=Path, default=Path("data/leaf_corpus"))
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=8192)
    parser.add_argument("--limit-scale", type=float, default=None, help="Scale every source max_samples, e.g. 0.001")
    parser.add_argument("--max-samples-per-source", type=int, default=None, help="Smoke-test cap applied to every source")
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepare_leaf_corpus(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        limit_scale=args.limit_scale,
        max_samples_per_source=args.max_samples_per_source,
        streaming=args.streaming,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
