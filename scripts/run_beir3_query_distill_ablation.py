#!/usr/bin/env python3
"""Run the BeIR3 query-distill ablation once teacher H5 files are present."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIGS = (
    "configs/training/stage_b_query_distill_bge_base_flatten_beir_ablation_current_1k.yaml",
    "configs/training/stage_b_query_distill_bge_base_flatten_beir_ablation_contrastive_1k.yaml",
)


def _teacher_embeddings_from_config(config_path: Path) -> list[Path]:
    paths: list[Path] = []
    in_teacher_list = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "teacher_embeddings:":
            in_teacher_list = True
            continue
        if in_teacher_list:
            if line.startswith("- "):
                paths.append(Path(line[2:].strip().strip("'\"")))
                continue
            if not raw_line.startswith((" ", "\t")):
                break
    return paths


def _missing_teacher_files(config_paths: list[Path]) -> list[Path]:
    missing: list[Path] = []
    for config_path in config_paths:
        for path in _teacher_embeddings_from_config(config_path):
            if not path.exists():
                missing.append(path)
    return sorted(set(missing))


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument(
        "--train-command",
        nargs="+",
        default=["poetry", "run", "projected-token", "train", "--config"],
        help="Command prefix used before each config path.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    config_paths = [Path(path) for path in args.configs]
    missing = _missing_teacher_files(config_paths)
    if missing:
        print("[ablation] missing teacher H5 files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        raise SystemExit(2)
    print("[ablation] teacher H5 preflight passed", flush=True)

    if args.preflight_only:
        return

    if not args.skip_train:
        for config_path in config_paths:
            _run([*args.train_command, str(config_path)])

    if args.skip_eval:
        return

    print(
        "[ablation] training completed. Run full BeIR3 eval for the best checkpoints "
        "by pointing a retrieval config at each run's checkpoints/best_model.pt.",
        flush=True,
    )


if __name__ == "__main__":
    main()
