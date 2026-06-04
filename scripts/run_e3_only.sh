#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="${CONTAINER:-pt-exp-mgpu}"
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/run_e3_only_${TS}.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[SEQ][START] $(date -Is) container=$CONTAINER"

if ! docker ps --format '{{.Names}}' | awk -v c="$CONTAINER" '$0==c{found=1} END{exit(found?0:1)}'; then
  echo "[SEQ][ERROR] container $CONTAINER not running"
  exit 1
fi

TOKEN_FILE="$HOME/.cache/huggingface/token"
HF_TOKEN_VALUE=""
if [[ -s "$TOKEN_FILE" ]]; then
  HF_TOKEN_VALUE="$(tr -d '\n' < "$TOKEN_FILE")"
fi
if [[ -z "$HF_TOKEN_VALUE" ]]; then
  echo "[SEQ][WARN] HF token missing; gated models may fail"
fi

# Ensure no stale session
docker exec "$CONTAINER" bash -lc "tmux kill-session -t stage_b_e3_5ep_beir5 >/dev/null 2>&1 || true"

run_one() {
  local name="$1"
  local cfg="$2"
  local session="$3"
  local gpus="$4"
  local comp="$5"
  local dec="$6"
  local pipea="$7"
  local pipeb="$8"

  echo "[SEQ][RUN] $name cfg=$cfg session=$session"
  HF_TOKEN="$HF_TOKEN_VALUE" \
  CONTAINER="$CONTAINER" \
  SESSION_NAME="$session" \
  TRAIN_CONFIG_PATH="$cfg" \
  TEACHER_SCRIPT="scripts/build_query_distill_candidates_h5.sh" \
  TEACHER_OUT_DIR="artifacts/teacher-embeddings/query-doc-bge-base-candidates" \
  SKIP_TEACHER_IF_EXISTS=1 \
  CUDA_VISIBLE_DEVICES_VALUE="$gpus" \
  OSCAR_COMPRESSOR_DEVICE_VALUE="$comp" \
  OSCAR_DECODER_DEVICE_VALUE="$dec" \
  BEIR_PIPELINE_A_DEVICE_VALUE="$pipea" \
  BEIR_PIPELINE_B_DEVICE_VALUE="$pipeb" \
  bash run_docker_tmux_query_distill_full_design.sh

  local exit_file="artifacts/logs/${session}.exit"
  echo "[SEQ][WAIT] $name waiting for $exit_file"
  while [[ ! -f "$exit_file" ]]; do
    sleep 20
  done
  local ec
  ec="$(tr -d '[:space:]' < "$exit_file")"
  echo "[SEQ][DONE] $name exit=$ec"
  if [[ "$ec" != "0" ]]; then
    echo "[SEQ][ERROR] $name failed"
    return 1
  fi
  return 0
}

run_one "E3" "configs/training/e3_5ep_beir5.yaml" "stage_b_e3_5ep_beir5" "0,1,2,3" "cuda:0" "cuda:1" "cuda:2" "cuda:3"

echo "[SEQ][SUCCESS] all runs finished"
