#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="${CONTAINER:-pt-exp-mgpu}"
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/run_e9_flatten_kl_4096_${TS}.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[E9][START] $(date -Is) container=$CONTAINER"

if ! docker ps --format '{{.Names}}' | awk -v c="$CONTAINER" '$0==c{found=1} END{exit(found?0:1)}'; then
  echo "[E9][ERROR] container $CONTAINER not running"
  exit 1
fi

TOKEN_FILE="$HOME/.cache/huggingface/token"
HF_TOKEN_VALUE=""
if [[ -s "$TOKEN_FILE" ]]; then
  HF_TOKEN_VALUE="$(tr -d '\n' < "$TOKEN_FILE")"
fi
if [[ -z "$HF_TOKEN_VALUE" ]]; then
  echo "[E9][WARN] HF token missing; gated models may fail"
fi

SESSION="stage_b_e9_flatten_kl"
CFG="configs/training/e9_flatten_kl_4096.yaml"

docker exec "$CONTAINER" bash -lc "tmux kill-session -t $SESSION >/dev/null 2>&1 || true"

HF_TOKEN="$HF_TOKEN_VALUE" \
CONTAINER="$CONTAINER" \
SESSION_NAME="$SESSION" \
TRAIN_CONFIG_PATH="$CFG" \
TEACHER_SCRIPT="scripts/build_query_distill_candidates_h5.sh" \
TEACHER_OUT_DIR="artifacts/teacher-embeddings/query-doc-bge-base-candidates" \
SKIP_TEACHER_IF_EXISTS=1 \
CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3" \
OSCAR_COMPRESSOR_DEVICE_VALUE="cuda:0" \
OSCAR_DECODER_DEVICE_VALUE="cuda:1" \
BEIR_PIPELINE_A_DEVICE_VALUE="cuda:2" \
BEIR_PIPELINE_B_DEVICE_VALUE="cuda:3" \
bash run_docker_tmux_query_distill_full_design.sh

echo "[E9][OK] launch requested"
echo "[E9][NEXT] tail -f artifacts/logs/${SESSION}.log"
