#!/usr/bin/env python3
"""Merge E11 corpus-eval summaries into one BeIR-6 summary."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts/results/retrieval"
ORDER = ["scifact", "nfcorpus", "fiqa-2018", "arguana", "trec-covid", "quora"]


def main() -> None:
    a = json.loads((ROOT / "e11_best_fiqa_quora_trec/summary.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "e11_best_scifact_nfcorpus_arguana/summary.json").read_text(encoding="utf-8"))
    per = {**a["per_dataset"], **b["per_dataset"]}
    per = {k: per[k] for k in ORDER if k in per}
    keys = next(iter(per.values())).keys()
    avg = {k: sum(per[d][k] for d in per) / len(per) for k in keys}
    out = {
        "run_id": "e11_best_beir6_full_corpus_eval",
        "split": "test",
        "search_k": 100,
        "checkpoint": "artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt",
        "pooler": "per_token_gated",
        "datasets": ORDER,
        "sources": {
            "scifact": "e11_best_scifact_nfcorpus_arguana",
            "nfcorpus": "e11_best_scifact_nfcorpus_arguana",
            "arguana": "e11_best_scifact_nfcorpus_arguana",
            "fiqa-2018": "e11_best_fiqa_quora_trec",
            "trec-covid": "e11_best_fiqa_quora_trec",
            "quora": "e11_best_fiqa_quora_trec",
        },
        "per_dataset": per,
        "average": avg,
    }
    out_dir = ROOT / "e11_best_beir6_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.json'}")
    for d in ORDER:
        m = per[d]
        print(f"  {d:12} ndcg@10={m['ndcg@10']:.4f}  mrr@10={m['mrr@10']:.4f}")
    print(f"  {'macro avg':12} ndcg@10={avg['ndcg@10']:.4f}  mrr@10={avg['mrr@10']:.4f}")


if __name__ == "__main__":
    main()
