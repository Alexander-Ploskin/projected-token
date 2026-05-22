#!/usr/bin/env python3
"""Run BeIR-6 for BM25, BGE, and selected OSCAR projector checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.io import load_json, write_json
from projected_token.retrieval.beir import evaluate_beir
from projected_token.retrieval.bm25_baseline import evaluate_beir3_bm25

DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "query_distill_runs_5ep_resume2750_noes" / (
    "stage-b-query-distill-dataablate-msmarco-bm25-30-100-mse-only-5ep-resume2750-noes"
)
DEFAULT_STEPS = (10750, 9500, 10000, 10250, 5250, 8500)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _beir6_datasets() -> list[dict[str, str]]:
    return [
        {"name": "scifact", "path": str(REPO_ROOT / "data" / "beir" / "scifact")},
        {"name": "nfcorpus", "path": str(REPO_ROOT / "data" / "beir" / "nfcorpus")},
        {"name": "fiqa-2018", "path": str(REPO_ROOT / "data" / "beir" / "fiqa")},
        {"name": "arguana", "path": str(REPO_ROOT / "data" / "beir" / "arguana")},
        {"name": "trec-covid", "path": str(REPO_ROOT.parent / "trec-covid")},
        {"name": "quora", "path": str(REPO_ROOT / "data" / "beir" / "quora")},
    ]


def _build_beir_config(*, encoder_name: str, encoder_kwargs: dict[str, Any], out_dir: Path, run_id: str) -> dict[str, Any]:
    return {
        "encoder": {"name": encoder_name, "kwargs": encoder_kwargs},
        "index": {"metric": "ip", "normalize": True, "batch_size": 32},
        "split": "test",
        "search_k": 100,
        "datasets": _beir6_datasets(),
        "metrics": {
            "top_k": [1, 3, 5, 10, 20],
            "output_path": str(out_dir / "summary.json"),
            "output_csv_path": str(out_dir / "summary.csv"),
            "run_id": run_id,
        },
    }


def _assert_datasets_exist() -> None:
    missing: list[Path] = []
    for item in _beir6_datasets():
        dataset_path = Path(item["path"])
        required = [dataset_path / "corpus.jsonl", dataset_path / "queries.jsonl", dataset_path / "qrels" / "test.tsv"]
        for path in required:
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(str(path) for path in missing))


def _assert_checkpoints_exist(run_root: Path, steps: tuple[int, ...]) -> None:
    ckpt_dir = run_root / "checkpoints"
    missing = [ckpt_dir / f"checkpoint_step_{step}.pt" for step in steps if not (ckpt_dir / f"checkpoint_step_{step}.pt").exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(str(path) for path in missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "results" / "retrieval" / "beir6_batch_20260522"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    steps = tuple(args.steps)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    _assert_datasets_exist()
    _assert_checkpoints_exist(run_root, steps)

    train_cfg_path = run_root / "config.lock.yaml"
    train_cfg = load_json(run_root / "checkpoints" / "config.json")
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {train_cfg_path}")

    # BM25 baseline
    bm25_dir = output_root / "bm25"
    bm25_summary = evaluate_beir3_bm25(
        datasets=_beir6_datasets(),
        split="test",
        top_k=[1, 3, 5, 10, 20],
        search_k=100,
        output_path=bm25_dir / "summary.json",
        output_csv_path=bm25_dir / "summary.csv",
        run_id="beir6_bm25",
    )

    # BGE baseline
    bge_dir = output_root / "bge_base_en_v15"
    bge_cfg = _build_beir_config(
        encoder_name="bge_base_en_v15",
        encoder_kwargs={
            "model_name_or_path": "BAAI/bge-base-en-v1.5",
            "device": args.device,
            "trust_remote_code": True,
        },
        out_dir=bge_dir,
        run_id="beir6_bge_base_en_v15",
    )
    _write_yaml(output_root / "generated_configs" / "beir6_bge_base_en_v15.yaml", bge_cfg)
    bge_summary = evaluate_beir(bge_cfg)

    # OSCAR projector checkpoints
    oscar_results: dict[str, Any] = {}
    for step in steps:
        run_id = f"beir6_oscar_projector_step_{step}"
        step_dir = output_root / f"oscar_projector_step_{step}"
        ckpt_path = run_root / "checkpoints" / f"checkpoint_step_{step}.pt"
        oscar_cfg = _build_beir_config(
            encoder_name="oscar_projector",
            encoder_kwargs={
                "oscar_model_name": "naver/oscar-qwen2-7B",
                "projector_path": str(ckpt_path),
                "device": args.device,
                "embed_dim": int(train_cfg.get("embed_dim", 768)),
                "pooler": str(train_cfg.get("pooler", "mean")),
                "num_layers": int(train_cfg.get("num_layers", 1)),
                "dropout": float(train_cfg.get("dropout", 0.0)),
            },
            out_dir=step_dir,
            run_id=run_id,
        )
        _write_yaml(output_root / "generated_configs" / f"{run_id}.yaml", oscar_cfg)
        oscar_results[str(step)] = evaluate_beir(oscar_cfg)

    ranking_rows: list[dict[str, Any]] = []
    ranking_rows.append({"model_or_step": "bm25", **bm25_summary.get("average", {})})
    ranking_rows.append({"model_or_step": "bge_base_en_v15", **bge_summary.get("average", {})})
    for step in steps:
        ranking_rows.append({"model_or_step": f"oscar_projector_step_{step}", **oscar_results[str(step)].get("average", {})})
    ranking_rows.sort(key=lambda row: float(row.get("ndcg@10", 0.0)), reverse=True)

    final_summary = {
        "run_root": str(run_root),
        "steps": list(steps),
        "datasets": _beir6_datasets(),
        "results": {
            "bm25": bm25_summary,
            "bge_base_en_v15": bge_summary,
            "oscar_projector": oscar_results,
        },
        "ranking_by_ndcg10": ranking_rows,
    }
    write_json(output_root / "beir6_batch_summary.json", final_summary)


if __name__ == "__main__":
    main()
