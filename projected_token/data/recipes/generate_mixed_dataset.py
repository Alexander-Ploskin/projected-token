#!/usr/bin/env python3
"""Generate mixed-domain retrieval dataset with explicit source metadata.

Default Stage-B mixture targets:
- MSMARCO
- HotpotQA
- FEVER
- FiQA
- SciQ
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from tqdm import tqdm


def _normalize_sample(query: str, positive: str, negatives: list[str], source: str, sample_id: str) -> dict[str, Any]:
    return {
        "query": str(query or "").strip(),
        "positive_doc": str(positive or "").strip(),
        "negatives": [str(n).strip() for n in negatives if str(n).strip()],
        "source": source,
        "sample_id": sample_id,
    }


def _load_msmarco(max_samples: int) -> list[dict[str, Any]]:
    ds = load_dataset("sentence-transformers/msmarco", "triplets", split="train", streaming=True)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(ds, desc="msmarco", total=max_samples)):
        if i >= max_samples:
            break
        out.append(
            _normalize_sample(
                query=row.get("query", ""),
                positive=row.get("positive", ""),
                negatives=[row.get("negative", "")],
                source="msmarco",
                sample_id=f"msmarco:{i}",
            )
        )
    return out


def _load_hotpotqa(max_samples: int) -> list[dict[str, Any]]:
    ds = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
    qds = load_dataset("BeIR/hotpotqa", "queries", split="queries")
    qrels = load_dataset("BeIR/hotpotqa-qrels", split="test")
    corpus = {str(r["_id"]): f"{str(r.get('title', '')).strip()} {str(r.get('text', '')).strip()}".strip() for r in ds}
    queries = {str(r["_id"]): str(r.get("text", "")) for r in qds}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(qrels, desc="hotpotqa")):
        if i >= max_samples:
            break
        if int(row.get("score", 1)) <= 0:
            continue
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        out.append(
            _normalize_sample(
                query=queries.get(qid, ""),
                positive=corpus.get(did, ""),
                negatives=[],
                source="hotpotqa",
                sample_id=f"hotpotqa:{qid}:{did}",
            )
        )
    return out


def _load_fever(max_samples: int) -> list[dict[str, Any]]:
    ds = load_dataset("BeIR/fever", "corpus", split="corpus")
    qds = load_dataset("BeIR/fever", "queries", split="queries")
    qrels = load_dataset("BeIR/fever-qrels", split="test")
    corpus = {str(r["_id"]): f"{str(r.get('title', '')).strip()} {str(r.get('text', '')).strip()}".strip() for r in ds}
    queries = {str(r["_id"]): str(r.get("text", "")) for r in qds}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(qrels, desc="fever")):
        if i >= max_samples:
            break
        if int(row.get("score", 1)) <= 0:
            continue
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        out.append(
            _normalize_sample(
                query=queries.get(qid, ""),
                positive=corpus.get(did, ""),
                negatives=[],
                source="fever",
                sample_id=f"fever:{qid}:{did}",
            )
        )
    return out


def _load_fiqa(max_samples: int) -> list[dict[str, Any]]:
    ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    qds = load_dataset("BeIR/fiqa", "queries", split="queries")
    qrels = load_dataset("BeIR/fiqa-qrels", split="test")
    corpus = {str(r["_id"]): f"{str(r.get('title', '')).strip()} {str(r.get('text', '')).strip()}".strip() for r in ds}
    queries = {str(r["_id"]): str(r.get("text", "")) for r in qds}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(qrels, desc="fiqa")):
        if i >= max_samples:
            break
        if int(row.get("score", 1)) <= 0:
            continue
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        out.append(
            _normalize_sample(
                query=queries.get(qid, ""),
                positive=corpus.get(did, ""),
                negatives=[],
                source="fiqa",
                sample_id=f"fiqa:{qid}:{did}",
            )
        )
    return out


def _load_sciq(max_samples: int) -> list[dict[str, Any]]:
    ds = load_dataset("allenai/sciq", split="train")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(ds, desc="sciq")):
        if i >= max_samples:
            break
        out.append(
            _normalize_sample(
                query=row.get("question", ""),
                positive=row.get("support", ""),
                negatives=[row.get("distractor1", ""), row.get("distractor2", ""), row.get("distractor3", "")],
                source="sciq",
                sample_id=f"sciq:{i}",
            )
        )
    return out


def _inject_random_negatives(samples: list[dict[str, Any]], max_negatives: int, seed: int) -> None:
    rng = random.Random(seed)
    positives = [s["positive_doc"] for s in samples if s["positive_doc"]]
    for sample in samples:
        if sample["negatives"]:
            sample["negatives"] = sample["negatives"][:max_negatives]
            continue
        candidates = [p for p in positives if p != sample["positive_doc"]]
        if candidates:
            sample["negatives"] = rng.sample(candidates, min(max_negatives, len(candidates)))


def create_mixed_dataset(
    *,
    max_total: int = 150_000,
    output_path: str = "data/mixed_train_dataset.json",
    seed: int = 42,
    max_negatives: int = 5,
    msmarco_weight: float = 0.40,
    hotpotqa_weight: float = 0.20,
    fever_weight: float = 0.15,
    fiqa_weight: float = 0.15,
    sciq_weight: float = 0.10,
) -> Dict[str, Any]:
    random.seed(seed)
    weights = {
        "msmarco": msmarco_weight,
        "hotpotqa": hotpotqa_weight,
        "fever": fever_weight,
        "fiqa": fiqa_weight,
        "sciq": sciq_weight,
    }
    total_w = sum(weights.values()) or 1.0
    weights = {k: v / total_w for k, v in weights.items()}
    targets = {k: int(max_total * w) for k, w in weights.items()}

    print("Loading mixed domains with targets:", targets)
    loaded = {
        "msmarco": _load_msmarco(targets["msmarco"]),
        "hotpotqa": _load_hotpotqa(targets["hotpotqa"]),
        "fever": _load_fever(targets["fever"]),
        "fiqa": _load_fiqa(targets["fiqa"]),
        "sciq": _load_sciq(targets["sciq"]),
    }
    all_data = [item for domain_items in loaded.values() for item in domain_items]
    _inject_random_negatives(all_data, max_negatives=max_negatives, seed=seed)
    random.shuffle(all_data)
    all_data = all_data[:max_total]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=True)

    source_counts: dict[str, int] = {}
    for item in all_data:
        source = item.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    avg_negatives = sum(len(item.get("negatives", [])) for item in all_data) / max(1, len(all_data))
    stats = {
        "total_samples": len(all_data),
        "source_counts": source_counts,
        "avg_negatives_per_sample": avg_negatives,
        "weights": weights,
        "targets": targets,
    }
    stats_path = str(Path(output_path).with_suffix(".stats.json"))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=True)
    print(f"Saved mixed dataset: {output_path}")
    print(f"Saved stats: {stats_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate Stage-B mixed-domain dataset")
    parser.add_argument("--max-total", type=int, default=150000, help="Maximum total samples")
    parser.add_argument("--output-path", type=str, default="data/mixed_train_dataset.json")
    parser.add_argument("--max-negatives", type=int, default=5,
                        help="Maximum negatives per sample (default: 5)")
    parser.add_argument("--msmarco-weight", type=float, default=0.40)
    parser.add_argument("--hotpotqa-weight", type=float, default=0.20)
    parser.add_argument("--fever-weight", type=float, default=0.15)
    parser.add_argument("--fiqa-weight", type=float, default=0.15)
    parser.add_argument("--sciq-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    stats = create_mixed_dataset(
        max_total=args.max_total,
        output_path=args.output_path,
        seed=args.seed,
        max_negatives=args.max_negatives,
        msmarco_weight=args.msmarco_weight,
        hotpotqa_weight=args.hotpotqa_weight,
        fever_weight=args.fever_weight,
        fiqa_weight=args.fiqa_weight,
        sciq_weight=args.sciq_weight,
    )

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Source counts: {stats['source_counts']}")
    print(f"Average negatives: {stats['avg_negatives_per_sample']:.2f}")
    print(f"Weights: {stats['weights']}")
    print(f"Targets: {stats['targets']}")


if __name__ == "__main__":
    main()