#!/usr/bin/env python3
"""Generate BM25-mined hard negatives for mixed retrieval datasets."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from projected_token.retrieval.bm25_baseline import SimpleBM25


def _load_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mine_bm25_negatives(records: list[dict[str, Any]], top_k: int, rank_min: int, rank_max: int) -> list[dict[str, Any]]:
    docs = [str(r.get("positive_doc", r.get("positive", ""))) for r in records]
    queries = [str(r.get("query", "")) for r in records]
    bm25 = SimpleBM25.from_texts(docs)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(records, desc="mine_bm25")):
        ranked = bm25.search(queries[i], top_k=max(rank_max, top_k))
        negatives: list[str] = []
        for pos, doc_idx in enumerate(ranked, start=1):
            if doc_idx == i:
                continue
            if pos < rank_min or pos > rank_max:
                continue
            cand = docs[doc_idx]
            if cand and cand != docs[i]:
                negatives.append(cand)
            if len(negatives) >= top_k:
                break
        row_out = dict(row)
        row_out["negatives"] = negatives
        row_out["hard_negative_source"] = "bm25_mined"
        out.append(row_out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate BM25 mined hard negatives")
    parser.add_argument("--input-path", type=str, required=True, help="Path to mixed dataset JSON")
    parser.add_argument("--output-path", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--top-k-negatives", type=int, default=5, help="Number of hard negatives per sample")
    parser.add_argument("--rank-min", type=int, default=2, help="Minimum BM25 rank to consider")
    parser.add_argument("--rank-max", type=int, default=30, help="Maximum BM25 rank to consider")
    args = parser.parse_args()

    records = _load_records(args.input_path)
    mined = _mine_bm25_negatives(
        records=records,
        top_k=args.top_k_negatives,
        rank_min=args.rank_min,
        rank_max=args.rank_max,
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(mined, f, indent=2, ensure_ascii=True)

    avg_negs = sum(len(r.get("negatives", [])) for r in mined) / max(1, len(mined))
    print(f"Saved hard negatives: {args.output_path}")
    print(f"Samples: {len(mined)}, avg_negatives={avg_negs:.2f}")


if __name__ == "__main__":
    main()