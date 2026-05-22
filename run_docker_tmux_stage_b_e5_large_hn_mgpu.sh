#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"

CONTAINER="${CONTAINER:-pt-exp-mgpu}"
SESSION_NAME="${SESSION_NAME:-stage_b_e5_large_hn_mgpu}"
CONFIG_PATH="${CONFIG_PATH:-configs/training/stage_b_distill_e5_large_hn.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,3,4}"
OSCAR_COMPRESSOR_DEVICE_VALUE="${OSCAR_COMPRESSOR_DEVICE_VALUE:-cuda:0}"
OSCAR_DECODER_DEVICE_VALUE="${OSCAR_DECODER_DEVICE_VALUE:-cuda:1}"
ASYNC_VALIDATION_DEVICE_VALUE="${ASYNC_VALIDATION_DEVICE_VALUE:-cuda:2}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="$LOG_DIR/docker_tmux_stage_b_e5_large_hn_mgpu_${TIMESTAMP}.log"
STAGE_LOG_CONT="/workspace/projected-token/artifacts/logs/${SESSION_NAME}_${TIMESTAMP}.log"
EXIT_FILE_CONT="/workspace/projected-token/artifacts/logs/${SESSION_NAME}.exit"
EXIT_FILE_HOST="$REPO_DIR/artifacts/logs/${SESSION_NAME}.exit"

exec > >(tee -a "$ORCH_LOG") 2>&1

echo "[ORCH][START] $(date -Is)"
echo "[ORCH][CONTAINER] $CONTAINER"
echo "[ORCH][SESSION] $SESSION_NAME"
echo "[ORCH][CONFIG] $CONFIG_PATH"
echo "[ORCH][INSTALL_DEPS] $INSTALL_DEPS"
echo "[ORCH][PROFILE] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE OSCAR_COMPRESSOR_DEVICE=$OSCAR_COMPRESSOR_DEVICE_VALUE OSCAR_DECODER_DEVICE=$OSCAR_DECODER_DEVICE_VALUE ASYNC_VALIDATION_DEVICE=$ASYNC_VALIDATION_DEVICE_VALUE"
if [[ -n "$HF_TOKEN_VALUE" ]]; then
  echo "[ORCH][HF_TOKEN] provided"
else
  echo "[ORCH][HF_TOKEN] not provided"
fi

if ! docker ps --format '{{.Names}}' | awk -v c="$CONTAINER" '$0==c{found=1} END{exit(found?0:1)}'; then
  echo "[ORCH][ERROR] Container '$CONTAINER' is not running."
  exit 1
fi

docker exec "$CONTAINER" bash -lc "test -f /workspace/projected-token/$CONFIG_PATH"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[ORCH][SETUP] Installing project dependencies inside container (no venv)"
  docker exec "$CONTAINER" bash -lc "cd /workspace/projected-token && python -m pip install --upgrade pip && python -m pip install -e ."
fi

rm -f "$EXIT_FILE_HOST"

docker exec \
  -e SESSION_NAME="$SESSION_NAME" \
  -e STAGE_LOG_CONT="$STAGE_LOG_CONT" \
  -e EXIT_FILE_CONT="$EXIT_FILE_CONT" \
  -e CONFIG_PATH="$CONFIG_PATH" \
  -e CUDA_VISIBLE_DEVICES_VALUE="$CUDA_VISIBLE_DEVICES_VALUE" \
  -e OSCAR_COMPRESSOR_DEVICE_VALUE="$OSCAR_COMPRESSOR_DEVICE_VALUE" \
  -e OSCAR_DECODER_DEVICE_VALUE="$OSCAR_DECODER_DEVICE_VALUE" \
  -e ASYNC_VALIDATION_DEVICE_VALUE="$ASYNC_VALIDATION_DEVICE_VALUE" \
  -e PYTORCH_CUDA_ALLOC_CONF_VALUE="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
  -e HF_TOKEN_VALUE="$HF_TOKEN_VALUE" \
  "$CONTAINER" \
  bash -lc 'cd /workspace/projected-token && mkdir -p artifacts/logs && rm -f "$EXIT_FILE_CONT" && tmux kill-session -t "$SESSION_NAME" >/dev/null 2>&1 || true && tmux new-session -d -s "$SESSION_NAME" "cd /workspace/projected-token && export PYTHONUNBUFFERED=1 && export HF_HOME=/data/huggingface && export PYTHONPATH=/workspace/projected-token:\${PYTHONPATH:-} && export HF_TOKEN=$HF_TOKEN_VALUE && export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN_VALUE && export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE && export OSCAR_DISABLE_ADAPTER_WARMUP=1 && export OSCAR_COMPRESSOR_DEVICE=$OSCAR_COMPRESSOR_DEVICE_VALUE && export OSCAR_DECODER_DEVICE=$OSCAR_DECODER_DEVICE_VALUE && export ASYNC_VALIDATION_DEVICE=$ASYNC_VALIDATION_DEVICE_VALUE && export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF_VALUE && echo [START] \$(date -Is) config=$CONFIG_PATH cuda_visible=\$CUDA_VISIBLE_DEVICES oscar_compressor=\$OSCAR_COMPRESSOR_DEVICE oscar_decoder=\$OSCAR_DECODER_DEVICE async_device=\$ASYNC_VALIDATION_DEVICE hf_token_set=\$( [ -n \"\$HF_TOKEN\" ] && echo yes || echo no ) | tee -a $STAGE_LOG_CONT && python -m projected_token train --config $CONFIG_PATH \${TRAIN_EXTRA_ARGS:-} 2>&1 | tee -a $STAGE_LOG_CONT; ec=\${PIPESTATUS[0]}; echo [DONE] \$(date -Is) exit=\$ec | tee -a $STAGE_LOG_CONT; echo \$ec > $EXIT_FILE_CONT"'

docker exec "$CONTAINER" bash -lc "tmux has-session -t \"$SESSION_NAME\""
echo "[ORCH][OK] tmux session started"
echo "[ORCH][STAGE_LOG] $REPO_DIR/artifacts/logs/$(basename "$STAGE_LOG_CONT")"
echo "[ORCH][EXIT_FILE] $EXIT_FILE_HOST"
echo "[ORCH][NEXT] docker exec -it $CONTAINER tmux attach -t $SESSION_NAME"
