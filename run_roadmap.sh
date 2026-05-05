#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/a-ploskin/repos/ms-thesis/projected-token"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/roadmap_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[START] $(date -Is)"
cd "$REPO_DIR"

poetry install

# 1) Freeze eval protocol + BM25 baselines
poetry run python -m projected_token train-roadmap --stage freeze-eval

# 2) Stage A
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_a.yaml

# 3) Stage B
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_b_distill.yaml

# 4) Stage C
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_c_two_stage.yaml

# 5) Stage D
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_d_joint_loss.yaml

# 6) Stage E
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_e_full_ft.yaml

# Optional consolidated pass (re-evaluates best per stage via roadmap orchestrator)
poetry run python -m projected_token train-roadmap --stage all

echo "[DONE] $(date -Is)"
echo "Log saved to: $LOG_FILE"
