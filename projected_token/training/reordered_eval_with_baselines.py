from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from projected_token.retrieval.bm25_baseline import evaluate_beir3_bm25, evaluate_popqa_bm25


TARGET_CONTR = "/app/.venv/bin/python -m projected_token train-matrix --config configs/training/matrix_contrastive.yaml"
TARGET_TWO = "/app/.venv/bin/python -m projected_token train-matrix --config configs/training/matrix_two_stage.yaml"

SUMMARY_CONTR = Path("artifacts/results/matrix/contrastive_summary.json")
SUMMARY_DISTILL = Path("artifacts/results/matrix/distill_summary.json")
SUMMARY_TWO = Path("artifacts/results/matrix/two_stage_summary.json")

GEN_DIR = Path("artifacts/results/retrieval/generated")
GEN_DIR.mkdir(parents=True, exist_ok=True)


def is_running(target: str) -> bool:
    out = subprocess.check_output(["ps", "-eo", "cmd"], text=True)
    return any(line.strip() == target for line in out.splitlines())


def run_cmd(cmd: list[str]) -> None:
    print("[reorder+baselines] RUN:", " ".join(cmd))
    sys.stdout.flush()
    subprocess.check_call(cmd)


def wait_for_not_running(target: str, tag: str, sleep_s: int = 120) -> None:
    while is_running(target):
        print(f"[reorder+baselines] waiting {tag} to finish...")
        sys.stdout.flush()
        time.sleep(sleep_s)


def wait_for_paths(paths: list[Path], sleep_s: int = 60) -> None:
    while True:
        missing = [str(p) for p in paths if not p.exists()]
        if not missing:
            return
        print(f"[reorder+baselines] waiting for summaries: {missing}")
        sys.stdout.flush()
        time.sleep(sleep_s)


def scalar_candidates(obj: Any):
    if isinstance(obj, dict):
        for key in ("mrr@10", "mrr", "cosine_sim", "ndcg@10", "recall@10"):
            if key in obj and isinstance(obj[key], (int, float)):
                yield float(obj[key])
        for value in obj.values():
            yield from scalar_candidates(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from scalar_candidates(item)


def score_from_metrics(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    vals = list(scalar_candidates(data))
    return max(vals) if vals else 0.0


def pick_best(summary_path: Path) -> tuple[dict[str, Any] | None, float]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    best_row: dict[str, Any] | None = None
    best_score = -1.0
    for row in runs:
        mp = row.get("metrics_path")
        score = score_from_metrics(Path(mp)) if mp else float(row.get("primary_metric", 0.0))
        if score > best_score:
            best_score = score
            best_row = row
    return best_row, best_score


def load_train_cfg(cfg_path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}


def make_popqa_cfg(train_cfg: dict[str, Any], run_root: str, out_stem: str) -> Path:
    num_layers = int(train_cfg.get("num_layers", train_cfg.get("projector_num_layers", 1)))
    dropout = float(train_cfg.get("dropout", train_cfg.get("projector_dropout", 0.0)))
    oscar_model = train_cfg.get("oscar_model_name") or train_cfg.get("oscar_model") or "/data/huggingface/naver/oscar-qwen2-7B"
    embed_dim = int(train_cfg.get("embed_dim", 768))
    pooler = train_cfg.get("pooler", "mean")
    cfg = {
        "dataset": {
            "path": "data/eval/popqa/popqa_enriched.parquet",
            "text_col": "s_wiki_content",
            "id_col": "id",
        },
        "encoder": {
            "name": "oscar_projector",
            "kwargs": {
                "oscar_model_name": oscar_model,
                "projector_path": f"{run_root}/checkpoints/best_model.pt",
                "device": "cuda:0",
                "embed_dim": embed_dim,
                "pooler": pooler,
                "num_layers": num_layers,
                "dropout": dropout,
            },
        },
        "index": {
            "output_dir": f"artifacts/indexes/{out_stem}",
            "metric": "ip",
            "normalize": True,
            "batch_size": 32,
        },
        "task": {"name": "popqa", "question_col": "question"},
        "metrics": {
            "top_k": [1, 3, 5, 10, 20],
            "output_path": f"artifacts/results/retrieval/{out_stem}_metrics.json",
        },
    }
    path = GEN_DIR / f"{out_stem}.yaml"
    path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def make_beir_cfg(train_cfg: dict[str, Any], run_root: str, out_stem: str) -> Path:
    num_layers = int(train_cfg.get("num_layers", train_cfg.get("projector_num_layers", 1)))
    dropout = float(train_cfg.get("dropout", train_cfg.get("projector_dropout", 0.0)))
    oscar_model = train_cfg.get("oscar_model_name") or train_cfg.get("oscar_model") or "/data/huggingface/naver/oscar-qwen2-7B"
    embed_dim = int(train_cfg.get("embed_dim", 768))
    pooler = train_cfg.get("pooler", "mean")
    cfg = {
        "encoder": {
            "name": "oscar_projector",
            "kwargs": {
                "oscar_model_name": oscar_model,
                "projector_path": f"{run_root}/checkpoints/best_model.pt",
                "device": "cuda:0",
                "embed_dim": embed_dim,
                "pooler": pooler,
                "num_layers": num_layers,
                "dropout": dropout,
            },
        },
        "index": {"metric": "ip", "normalize": True, "batch_size": 32},
        "split": "test",
        "search_k": 100,
        "datasets": [
            {"name": "scifact", "path": "data/beir/scifact"},
            {"name": "nfcorpus", "path": "data/beir/nfcorpus"},
            {"name": "fiqa-2018", "path": "data/beir/fiqa"},
        ],
        "metrics": {
            "top_k": [1, 3, 5, 10, 20],
            "output_path": f"artifacts/results/retrieval/{out_stem}_beir3_summary.json",
            "output_csv_path": f"artifacts/results/retrieval/{out_stem}_beir3_summary.csv",
        },
    }
    path = GEN_DIR / f"{out_stem}_beir.yaml"
    path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def make_sfr_popqa_cfg() -> Path:
    cfg = {
        "dataset": {
            "path": "data/eval/popqa/popqa_enriched.parquet",
            "text_col": "s_wiki_content",
            "id_col": "id",
        },
        "encoder": {
            "name": "salesforce",
            "kwargs": {
                "model_name_or_path": "/data/huggingface/Salesforce/SFR-Embedding-Mistral",
                "device": "cuda:0",
            },
        },
        "index": {
            "output_dir": "artifacts/indexes/popqa_sfr_baseline",
            "metric": "ip",
            "normalize": True,
            "batch_size": 32,
        },
        "task": {"name": "popqa", "question_col": "question"},
        "metrics": {
            "top_k": [1, 3, 5, 10, 20],
            "output_path": "artifacts/results/retrieval/popqa_sfr_baseline_metrics.json",
            "output_csv_path": "artifacts/results/retrieval/popqa_sfr_baseline_metrics.csv",
            "run_id": "sfr_popqa_baseline",
        },
    }
    path = GEN_DIR / "popqa_sfr_baseline.yaml"
    path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def make_sfr_beir_cfg() -> Path:
    cfg = {
        "encoder": {
            "name": "salesforce",
            "kwargs": {
                "model_name_or_path": "/data/huggingface/Salesforce/SFR-Embedding-Mistral",
                "device": "cuda:0",
            },
        },
        "index": {"metric": "ip", "normalize": True, "batch_size": 32},
        "split": "test",
        "search_k": 100,
        "datasets": [
            {"name": "scifact", "path": "data/beir/scifact"},
            {"name": "nfcorpus", "path": "data/beir/nfcorpus"},
            {"name": "fiqa-2018", "path": "data/beir/fiqa"},
        ],
        "metrics": {
            "top_k": [1, 3, 5, 10, 20],
            "output_path": "artifacts/results/retrieval/beir3_sfr_baseline_summary.json",
            "output_csv_path": "artifacts/results/retrieval/beir3_sfr_baseline_summary.csv",
            "run_id": "sfr_beir3_baseline",
        },
    }
    path = GEN_DIR / "beir3_sfr_baseline.yaml"
    path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def main() -> None:
    print("[reorder+baselines] waiting for contrastive matrix before distill rerun")
    wait_for_not_running(TARGET_CONTR, "contrastive matrix")

    print("[reorder+baselines] starting distill matrix rerun on GPU0")
    run_cmd(
        [
            "bash",
            "-lc",
            "cd /app && export PYTHONUNBUFFERED=1 && CUDA_VISIBLE_DEVICES=0 poetry run python -m projected_token train-matrix --config configs/training/matrix_distill.yaml",
        ]
    )

    print("[reorder+baselines] waiting for two-stage matrix completion before consolidated eval")
    wait_for_not_running(TARGET_TWO, "two-stage matrix")
    wait_for_paths([SUMMARY_CONTR, SUMMARY_DISTILL, SUMMARY_TWO])

    tracks = {
        "contrastive": SUMMARY_CONTR,
        "distill": SUMMARY_DISTILL,
        "two_stage": SUMMARY_TWO,
    }

    best_payload: dict[str, dict[str, Any]] = {}
    for track, summary_path in tracks.items():
        row, score = pick_best(summary_path)
        best_payload[track] = {
            "score": score,
            "run_root": row.get("run_root") if row else None,
            "config_path": row.get("config_path") if row else None,
        }
        print(
            f"[reorder+baselines] best {track}: score={score:.6f} "
            f"run_root={best_payload[track]['run_root']}"
        )

    # Projector tracks: PopQA + BEIR
    for track, summary_path in tracks.items():
        row, _ = pick_best(summary_path)
        if not row:
            print(f"[reorder+baselines] skip {track}: no best row")
            continue
        cfg_path = row.get("config_path")
        run_root = row.get("run_root")
        if not cfg_path or not run_root:
            print(f"[reorder+baselines] skip {track}: missing config_path/run_root")
            continue
        train_cfg = load_train_cfg(cfg_path)
        popqa_cfg = make_popqa_cfg(train_cfg, run_root, f"popqa_{track}_best")
        run_cmd(["poetry", "run", "python", "-m", "projected_token", "retrieval", "build-index", "--config", str(popqa_cfg)])
        run_cmd(["poetry", "run", "python", "-m", "projected_token", "retrieval", "evaluate", "--config", str(popqa_cfg)])

    for track, summary_path in tracks.items():
        row, _ = pick_best(summary_path)
        if not row:
            continue
        cfg_path = row.get("config_path")
        run_root = row.get("run_root")
        if not cfg_path or not run_root:
            continue
        train_cfg = load_train_cfg(cfg_path)
        beir_cfg = make_beir_cfg(train_cfg, run_root, f"beir3_{track}_best")
        run_cmd(["poetry", "run", "python", "-m", "projected_token", "retrieval", "evaluate-beir", "--config", str(beir_cfg)])

    # SFR baselines
    sfr_popqa_cfg = make_sfr_popqa_cfg()
    run_cmd(["poetry", "run", "python", "-m", "projected_token", "retrieval", "build-index", "--config", str(sfr_popqa_cfg)])
    run_cmd(["poetry", "run", "python", "-m", "projected_token", "retrieval", "evaluate", "--config", str(sfr_popqa_cfg)])

    sfr_beir_cfg = make_sfr_beir_cfg()
    run_cmd(["poetry", "run", "python", "-m", "projected_token", "retrieval", "evaluate-beir", "--config", str(sfr_beir_cfg)])

    # BM25 baselines
    bm25_popqa = evaluate_popqa_bm25(
        dataset_path="data/eval/popqa/popqa_enriched.parquet",
        text_col="s_wiki_content",
        question_col="question",
        top_k=[1, 3, 5, 10, 20],
        output_path="artifacts/results/retrieval/popqa_bm25_baseline_metrics.json",
        output_csv_path="artifacts/results/retrieval/popqa_bm25_baseline_metrics.csv",
        run_id="bm25_popqa_baseline",
    )
    bm25_beir = evaluate_beir3_bm25(
        datasets=[
            {"name": "scifact", "path": "data/beir/scifact"},
            {"name": "nfcorpus", "path": "data/beir/nfcorpus"},
            {"name": "fiqa-2018", "path": "data/beir/fiqa"},
        ],
        split="test",
        top_k=[1, 3, 5, 10, 20],
        search_k=100,
        output_path="artifacts/results/retrieval/beir3_bm25_baseline_summary.json",
        output_csv_path="artifacts/results/retrieval/beir3_bm25_baseline_summary.csv",
        run_id="bm25_beir3_baseline",
    )

    baseline_summary = {
        "best_projector_tracks": best_payload,
        "bm25": {
            "popqa": bm25_popqa,
            "beir3": bm25_beir.get("average", {}),
        },
        "sfr_outputs": {
            "popqa": "artifacts/results/retrieval/popqa_sfr_baseline_metrics.json",
            "beir3": "artifacts/results/retrieval/beir3_sfr_baseline_summary.json",
        },
    }
    Path("artifacts/results/retrieval/baseline_summary.json").write_text(
        json.dumps(baseline_summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print("[reorder+baselines] done: distill rerun + projector evals + BM25/SFR baselines")


if __name__ == "__main__":
    main()
