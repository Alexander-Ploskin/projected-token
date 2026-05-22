#!/usr/bin/env python3
"""Run a single BeIR-6 evaluation job (bm25, bge, or oscar checkpoint)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.io import load_json, write_json
from projected_token.retrieval.beir import evaluate_beir
from projected_token.retrieval.bm25_baseline import evaluate_beir3_bm25


def _datasets() -> list[dict[str, str]]:
    beir_root = Path("/data/beir")
    if not beir_root.exists():
        beir_root = REPO_ROOT / "data" / "beir"
    return [
        {"name": "scifact", "path": str(beir_root / "scifact")},
        {"name": "nfcorpus", "path": str(beir_root / "nfcorpus")},
        {"name": "fiqa-2018", "path": str(beir_root / "fiqa")},
        {"name": "arguana", "path": str(beir_root / "arguana")},
        {"name": "trec-covid", "path": str(REPO_ROOT.parent / "trec-covid")},
        {"name": "quora", "path": str(beir_root / "quora")},
    ]


def _check_datasets() -> None:
    missing: list[Path] = []
    for item in _datasets():
        root = Path(item["path"])
        for rel in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            path = root / rel
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing BeIR files:\n" + "\n".join(str(path) for path in missing))


def _build_dense_config(*, encoder_name: str, encoder_kwargs: dict[str, Any], output_dir: Path, run_id: str) -> dict[str, Any]:
    return {
        "encoder": {"name": encoder_name, "kwargs": encoder_kwargs},
        "index": {"metric": "ip", "normalize": True, "batch_size": 32},
        "split": "test",
        "search_k": 100,
        "datasets": _datasets(),
        "metrics": {
            "top_k": [1, 3, 5, 10, 20],
            "output_path": str(output_dir / "summary.json"),
            "output_csv_path": str(output_dir / "summary.csv"),
            "run_id": run_id,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bm25", "bge", "oscar"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "artifacts" / "query_distill_runs_5ep_resume2750_noes" / "stage-b-query-distill-dataablate-msmarco-bm25-30-100-mse-only-5ep-resume2750-noes"))
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()

    _check_datasets()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "bm25":
        result = evaluate_beir3_bm25(
            datasets=_datasets(),
            split="test",
            top_k=[1, 3, 5, 10, 20],
            search_k=100,
            output_path=out_dir / "summary.json",
            output_csv_path=out_dir / "summary.csv",
            run_id="beir6_bm25",
        )
        write_json(out_dir / "job_summary.json", result)
        return

    if args.mode == "bge":
        cfg = _build_dense_config(
            encoder_name="bge_base_en_v15",
            encoder_kwargs={
                "model_name_or_path": "BAAI/bge-base-en-v1.5",
                "device": args.device,
                "trust_remote_code": True,
            },
            output_dir=out_dir,
            run_id="beir6_bge_base_en_v15",
        )
        result = evaluate_beir(cfg)
        write_json(out_dir / "job_summary.json", result)
        return

    if args.step is None:
        raise ValueError("--step is required for --mode oscar")
    run_root = Path(args.run_root)
    ckpt = run_root / "checkpoints" / f"checkpoint_step_{args.step}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    train_cfg = load_json(run_root / "checkpoints" / "config.json")
    cfg = _build_dense_config(
        encoder_name="oscar_projector",
        encoder_kwargs={
            "oscar_model_name": "naver/oscar-qwen2-7B",
            "projector_path": str(ckpt),
            "device": args.device,
            "embed_dim": int(train_cfg.get("embed_dim", 768)),
            "pooler": str(train_cfg.get("pooler", "mean")),
            "num_layers": int(train_cfg.get("num_layers", 1)),
            "dropout": float(train_cfg.get("dropout", 0.0)),
        },
        output_dir=out_dir,
        run_id=f"beir6_oscar_projector_step_{args.step}",
    )
    result = evaluate_beir(cfg)
    write_json(out_dir / "job_summary.json", result)


if __name__ == "__main__":
    main()
