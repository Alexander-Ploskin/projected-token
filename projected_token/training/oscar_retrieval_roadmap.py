from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from projected_token.config import load_yaml
from projected_token.io import write_json
from projected_token.retrieval.beir import evaluate_beir
from projected_token.retrieval.bm25_baseline import evaluate_beir3_bm25, evaluate_popqa_bm25
from projected_token.retrieval.pipeline import build_index, evaluate_retrieval
from projected_token.training.matrix_runner import run_matrix


@dataclass(frozen=True)
class EvalProtocol:
    popqa_dataset_path: str
    popqa_text_col: str
    popqa_id_col: str
    popqa_question_col: str
    beir_datasets: list[dict[str, str]]
    split: str
    top_k: list[int]
    search_k: int
    batch_size: int
    index_metric: str
    normalize: bool
    device: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


def _load_protocol(config_path: str | Path) -> EvalProtocol:
    cfg = load_yaml(_resolve_path(config_path))
    popqa = cfg.get("popqa", {})
    beir = cfg.get("beir", {})
    retrieval = cfg.get("retrieval", {})
    return EvalProtocol(
        popqa_dataset_path=str(popqa.get("dataset_path", "data/eval/popqa/popqa_enriched.parquet")),
        popqa_text_col=str(popqa.get("text_col", "s_wiki_content")),
        popqa_id_col=str(popqa.get("id_col", "id")),
        popqa_question_col=str(popqa.get("question_col", "question")),
        beir_datasets=list(beir.get("datasets", [])),
        split=str(beir.get("split", "test")),
        top_k=[int(k) for k in retrieval.get("top_k", [1, 3, 5, 10, 20])],
        search_k=int(retrieval.get("search_k", 100)),
        batch_size=int(retrieval.get("batch_size", 32)),
        index_metric=str(retrieval.get("index_metric", "ip")),
        normalize=bool(retrieval.get("normalize", True)),
        device=str(retrieval.get("device", "cuda:0")),
    )


def _pick_best_row(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = data.get("runs", [])
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("primary_metric", 0.0)))


def _encoder_kwargs_from_train_cfg(train_cfg: dict[str, Any], checkpoint_path: str, protocol: EvalProtocol) -> dict[str, Any]:
    num_layers = int(train_cfg.get("num_layers", train_cfg.get("projector_num_layers", 1)))
    dropout = float(train_cfg.get("dropout", train_cfg.get("projector_dropout", 0.0)))
    pooler = str(train_cfg.get("pooler", "mean"))
    embed_dim = int(train_cfg.get("embed_dim", 768))
    oscar_model_name = str(
        train_cfg.get("oscar_model_name")
        or train_cfg.get("oscar_model")
        or "naver/oscar-qwen2-7B"
    )
    return {
        "oscar_model_name": oscar_model_name,
        "projector_path": checkpoint_path,
        "device": protocol.device,
        "embed_dim": embed_dim,
        "pooler": pooler,
        "num_layers": num_layers,
        "dropout": dropout,
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _evaluate_best_projector(
    *,
    stage_name: str,
    run_root: str,
    train_config_path: str,
    protocol: EvalProtocol,
    output_dir: Path,
) -> dict[str, Any]:
    train_cfg = load_yaml(_resolve_path(train_config_path))
    checkpoint_path = f"{run_root}/checkpoints/best_model.pt"
    encoder_kwargs = _encoder_kwargs_from_train_cfg(train_cfg, checkpoint_path, protocol)

    generated_dir = output_dir / "generated_configs"
    popqa_cfg_path = generated_dir / f"{stage_name}_popqa.yaml"
    beir_cfg_path = generated_dir / f"{stage_name}_beir.yaml"

    popqa_cfg = {
        "dataset": {
            "path": protocol.popqa_dataset_path,
            "text_col": protocol.popqa_text_col,
            "id_col": protocol.popqa_id_col,
        },
        "encoder": {"name": "oscar_projector", "kwargs": encoder_kwargs},
        "index": {
            "output_dir": str(output_dir / "indexes" / f"{stage_name}_popqa"),
            "metric": protocol.index_metric,
            "normalize": protocol.normalize,
            "batch_size": protocol.batch_size,
        },
        "task": {"name": "popqa", "question_col": protocol.popqa_question_col},
        "metrics": {
            "top_k": protocol.top_k,
            "output_path": str(output_dir / f"{stage_name}_popqa_metrics.json"),
            "output_csv_path": str(output_dir / f"{stage_name}_popqa_metrics.csv"),
            "run_id": f"{stage_name}_popqa",
        },
    }
    beir_cfg = {
        "encoder": {"name": "oscar_projector", "kwargs": encoder_kwargs},
        "index": {
            "metric": protocol.index_metric,
            "normalize": protocol.normalize,
            "batch_size": protocol.batch_size,
        },
        "split": protocol.split,
        "search_k": protocol.search_k,
        "datasets": protocol.beir_datasets,
        "metrics": {
            "top_k": protocol.top_k,
            "output_path": str(output_dir / f"{stage_name}_beir_summary.json"),
            "output_csv_path": str(output_dir / f"{stage_name}_beir_summary.csv"),
            "run_id": f"{stage_name}_beir",
        },
    }
    _write_yaml(popqa_cfg_path, popqa_cfg)
    _write_yaml(beir_cfg_path, beir_cfg)

    build_index(str(popqa_cfg_path))
    popqa_metrics = evaluate_retrieval(str(popqa_cfg_path))
    beir_summary = evaluate_beir(beir_cfg)

    summary = {
        "stage": stage_name,
        "run_root": run_root,
        "train_config_path": train_config_path,
        "checkpoint_path": checkpoint_path,
        "popqa": popqa_metrics,
        "beir_average": beir_summary.get("average", {}),
        "beir_per_dataset": beir_summary.get("per_dataset", {}),
    }
    write_json(output_dir / f"{stage_name}_best_eval.json", summary)
    return summary


def _run_bm25_baselines(protocol: EvalProtocol, output_dir: Path) -> dict[str, Any]:
    bm25_popqa = evaluate_popqa_bm25(
        dataset_path=protocol.popqa_dataset_path,
        text_col=protocol.popqa_text_col,
        question_col=protocol.popqa_question_col,
        top_k=protocol.top_k,
        output_path=output_dir / "bm25_popqa_metrics.json",
        output_csv_path=output_dir / "bm25_popqa_metrics.csv",
        run_id="roadmap_bm25_popqa",
    )
    bm25_beir = evaluate_beir3_bm25(
        datasets=protocol.beir_datasets,
        split=protocol.split,
        top_k=protocol.top_k,
        search_k=protocol.search_k,
        output_path=output_dir / "bm25_beir_summary.json",
        output_csv_path=output_dir / "bm25_beir_summary.csv",
        run_id="roadmap_bm25_beir3",
    )
    payload = {"popqa": bm25_popqa, "beir": bm25_beir}
    write_json(output_dir / "bm25_baselines.json", payload)
    return payload


def _run_bge_teacher_baseline(protocol: EvalProtocol, output_dir: Path) -> dict[str, Any]:
    beir_cfg = {
        "encoder": {
            "name": "bge",
            "kwargs": {
                "model_name_or_path": "BAAI/bge-base-en-v1.5",
                "device": protocol.device,
                "normalize_embeddings": True,
                "query_prefix": "Represent this sentence for searching relevant passages: ",
                "document_prefix": "",
            },
        },
        "index": {
            "metric": protocol.index_metric,
            "normalize": protocol.normalize,
            "batch_size": protocol.batch_size,
        },
        "split": protocol.split,
        "search_k": protocol.search_k,
        "datasets": protocol.beir_datasets,
        "metrics": {
            "top_k": protocol.top_k,
            "output_path": str(output_dir / "bge_teacher_beir_summary.json"),
            "output_csv_path": str(output_dir / "bge_teacher_beir_summary.csv"),
            "run_id": "roadmap_bge_teacher_beir",
        },
    }
    summary = evaluate_beir(beir_cfg)
    write_json(output_dir / "bge_teacher_baseline.json", summary)
    return summary


def _run_matrix_stage(stage_name: str, matrix_path: Path, protocol: EvalProtocol, output_dir: Path) -> dict[str, Any]:
    run_matrix(matrix_path)
    matrix_cfg = load_yaml(matrix_path)
    summary_path = _resolve_path(matrix_cfg.get("summary_output", ""))
    best_row = _pick_best_row(summary_path)
    if not best_row:
        raise RuntimeError(f"No completed runs found in summary: {summary_path}")
    return _evaluate_best_projector(
        stage_name=stage_name,
        run_root=str(best_row["run_root"]),
        train_config_path=str(best_row["config_path"]),
        protocol=protocol,
        output_dir=output_dir,
    )


def run_roadmap(stage: str, protocol_config: str, run_baselines: bool) -> dict[str, Any]:
    protocol = _load_protocol(protocol_config)
    output_override = os.environ.get("PROJECTED_TOKEN_ROADMAP_OUTPUT_DIR")
    output_dir = _resolve_path(output_override) if output_override else _resolve_path("artifacts/results/retrieval/roadmap")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = _resolve_path(".artifacts_local/results/retrieval/roadmap")
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"[roadmap] cannot write to {output_dir}; using fallback {fallback}",
            flush=True,
        )
        output_dir = fallback
    matrix_map = {
        "a": _resolve_path("configs/training/matrix_stage_a.yaml"),
        "b": _resolve_path("configs/training/matrix_stage_b_distill.yaml"),
        "c": _resolve_path("configs/training/matrix_stage_c_two_stage.yaml"),
        "d": _resolve_path("configs/training/matrix_stage_d_joint_loss.yaml"),
        "e": _resolve_path("configs/training/matrix_stage_e_full_ft.yaml"),
    }

    results: dict[str, Any] = {"stage": stage, "protocol_config": str(protocol_config)}
    if stage in {"freeze-eval", "all"} and run_baselines:
        results["bm25_baselines"] = _run_bm25_baselines(protocol, output_dir)
        results["bge_teacher_baseline"] = _run_bge_teacher_baseline(protocol, output_dir)

    ordered_stages = ["a", "b", "c", "d", "e"] if stage == "all" else [stage]
    for stage_key in ordered_stages:
        if stage_key not in matrix_map:
            if stage_key != "freeze-eval":
                raise ValueError(f"Unknown stage: {stage_key}")
            continue
        results[f"stage_{stage_key}"] = _run_matrix_stage(
            stage_name=f"stage_{stage_key}",
            matrix_path=matrix_map[stage_key],
            protocol=protocol,
            output_dir=output_dir,
        )

    write_json(output_dir / "roadmap_summary.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OSCAR retrieval roadmap stages.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "freeze-eval", "a", "b", "c", "d", "e"],
    )
    parser.add_argument(
        "--protocol-config",
        type=str,
        default="configs/retrieval/eval_protocol_oscar.yaml",
    )
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args()
    run_roadmap(
        stage=args.stage,
        protocol_config=args.protocol_config,
        run_baselines=not args.no_baselines,
    )


if __name__ == "__main__":
    main()
