#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"

CONTAINER="${CONTAINER:-pt-exp-mgpu}"
SESSION_NAME="${SESSION_NAME:-stage_b_hn_with_bge_teacher}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/training/stage_b_distill_bge_base_hn.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,3,4}"
OSCAR_COMPRESSOR_DEVICE_VALUE="${OSCAR_COMPRESSOR_DEVICE_VALUE:-cuda:0}"
OSCAR_DECODER_DEVICE_VALUE="${OSCAR_DECODER_DEVICE_VALUE:-cuda:1}"
ASYNC_VALIDATION_DEVICE_VALUE="${ASYNC_VALIDATION_DEVICE_VALUE:-cuda:2}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"

TEACHER_CORPUS_PATH="${TEACHER_CORPUS_PATH:-}"
TEACHER_NUM_SAMPLES="${TEACHER_NUM_SAMPLES:-100000}"
TEACHER_OUTPUT_DIR="${TEACHER_OUTPUT_DIR:-/data/teacher-embeddings}"
TEACHER_OUTPUT_NAME="${TEACHER_OUTPUT_NAME:-bge-base-en-v1.5_teacher_embeddings_msmarco_100k.h5}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-BAAI/bge-base-en-v1.5}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda:0}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-32}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="$LOG_DIR/docker_tmux_bge_teacher_then_stage_b_hn_${TIMESTAMP}.log"
PIPELINE_LOG_CONT="/workspace/projected-token/artifacts/logs/${SESSION_NAME}_${TIMESTAMP}.log"
EXIT_FILE_CONT="/workspace/projected-token/artifacts/logs/${SESSION_NAME}.exit"
EXIT_FILE_HOST="$REPO_DIR/artifacts/logs/${SESSION_NAME}.exit"

exec > >(tee -a "$ORCH_LOG") 2>&1

echo "[ORCH][START] $(date -Is)"
echo "[ORCH][CONTAINER] $CONTAINER"
echo "[ORCH][SESSION] $SESSION_NAME"
echo "[ORCH][TRAIN_CONFIG] $TRAIN_CONFIG_PATH"
echo "[ORCH][TEACHER_OUTPUT] $TEACHER_OUTPUT_DIR/$TEACHER_OUTPUT_NAME"
echo "[ORCH][TEACHER_NUM_SAMPLES] $TEACHER_NUM_SAMPLES"
if [[ -n "$TEACHER_CORPUS_PATH" ]]; then
  echo "[ORCH][TEACHER_SOURCE] corpus_path=$TEACHER_CORPUS_PATH"
else
  echo "[ORCH][TEACHER_SOURCE] dataset_name=Tevatron/msmarco-passage-corpus"
fi
echo "[ORCH][INSTALL_DEPS] $INSTALL_DEPS"
if [[ -n "$HF_TOKEN_VALUE" ]]; then
  echo "[ORCH][HF_TOKEN] provided"
else
  echo "[ORCH][HF_TOKEN] not provided"
fi

if ! docker ps --format '{{.Names}}' | awk -v c="$CONTAINER" '$0==c{found=1} END{exit(found?0:1)}'; then
  echo "[ORCH][ERROR] Container '$CONTAINER' is not running."
  exit 1
fi

docker exec "$CONTAINER" bash -lc "test -f /workspace/projected-token/$TRAIN_CONFIG_PATH"
if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[ORCH][SETUP] Installing project dependencies inside container (no venv)"
  docker exec "$CONTAINER" bash -lc "cd /workspace/projected-token && python -m pip install --upgrade pip && python -m pip install -e ."
fi

rm -f "$EXIT_FILE_HOST"

docker exec \
  -e SESSION_NAME="$SESSION_NAME" \
  -e PIPELINE_LOG_CONT="$PIPELINE_LOG_CONT" \
  -e EXIT_FILE_CONT="$EXIT_FILE_CONT" \
  -e TRAIN_CONFIG_PATH="$TRAIN_CONFIG_PATH" \
  -e CUDA_VISIBLE_DEVICES_VALUE="$CUDA_VISIBLE_DEVICES_VALUE" \
  -e OSCAR_COMPRESSOR_DEVICE_VALUE="$OSCAR_COMPRESSOR_DEVICE_VALUE" \
  -e OSCAR_DECODER_DEVICE_VALUE="$OSCAR_DECODER_DEVICE_VALUE" \
  -e ASYNC_VALIDATION_DEVICE_VALUE="$ASYNC_VALIDATION_DEVICE_VALUE" \
  -e PYTORCH_CUDA_ALLOC_CONF_VALUE="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
  -e HF_TOKEN_VALUE="$HF_TOKEN_VALUE" \
  -e TEACHER_CORPUS_PATH="$TEACHER_CORPUS_PATH" \
  -e TEACHER_NUM_SAMPLES="$TEACHER_NUM_SAMPLES" \
  -e TEACHER_OUTPUT_DIR="$TEACHER_OUTPUT_DIR" \
  -e TEACHER_OUTPUT_NAME="$TEACHER_OUTPUT_NAME" \
  -e TEACHER_MODEL_NAME="$TEACHER_MODEL_NAME" \
  -e TEACHER_DEVICE="$TEACHER_DEVICE" \
  -e TEACHER_BATCH_SIZE="$TEACHER_BATCH_SIZE" \
  "$CONTAINER" \
  bash -lc 'cd /workspace/projected-token && mkdir -p artifacts/logs && rm -f "$EXIT_FILE_CONT" && tmux kill-session -t "$SESSION_NAME" >/dev/null 2>&1 || true && tmux new-session -d -s "$SESSION_NAME" "cd /workspace/projected-token && export PYTHONUNBUFFERED=1 && export HF_HOME=/data/huggingface && export PYTHONPATH=/workspace/projected-token:\${PYTHONPATH:-} && export HF_TOKEN=$HF_TOKEN_VALUE && export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN_VALUE && export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE && export OSCAR_DISABLE_ADAPTER_WARMUP=1 && export OSCAR_COMPRESSOR_DEVICE=$OSCAR_COMPRESSOR_DEVICE_VALUE && export OSCAR_DECODER_DEVICE=$OSCAR_DECODER_DEVICE_VALUE && export ASYNC_VALIDATION_DEVICE=$ASYNC_VALIDATION_DEVICE_VALUE && export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF_VALUE && echo [START] \$(date -Is) teacher_output=$TEACHER_OUTPUT_DIR/$TEACHER_OUTPUT_NAME train_config=$TRAIN_CONFIG_PATH hf_token_set=\$( [ -n \"\$HF_TOKEN\" ] && echo yes || echo no ) num_samples=$TEACHER_NUM_SAMPLES | tee -a $PIPELINE_LOG_CONT && if [ -n \"$TEACHER_CORPUS_PATH\" ]; then python -m projected_token data teacher-embeddings --corpus-path $TEACHER_CORPUS_PATH --num-samples $TEACHER_NUM_SAMPLES --output-dir $TEACHER_OUTPUT_DIR --output-name $TEACHER_OUTPUT_NAME --teacher-model-name $TEACHER_MODEL_NAME --pooling mean --prompt-style none --device $TEACHER_DEVICE --batch-size $TEACHER_BATCH_SIZE 2>&1 | tee -a $PIPELINE_LOG_CONT; else python -m projected_token data teacher-embeddings --dataset-name Tevatron/msmarco-passage-corpus --num-samples $TEACHER_NUM_SAMPLES --output-dir $TEACHER_OUTPUT_DIR --output-name $TEACHER_OUTPUT_NAME --teacher-model-name $TEACHER_MODEL_NAME --pooling mean --prompt-style none --device $TEACHER_DEVICE --batch-size $TEACHER_BATCH_SIZE 2>&1 | tee -a $PIPELINE_LOG_CONT; fi && ls -lh $TEACHER_OUTPUT_DIR/$TEACHER_OUTPUT_NAME | tee -a $PIPELINE_LOG_CONT && python -m projected_token train --config $TRAIN_CONFIG_PATH \${TRAIN_EXTRA_ARGS:-} 2>&1 | tee -a $PIPELINE_LOG_CONT; ec=\${PIPESTATUS[0]}; echo [DONE] \$(date -Is) exit=\$ec | tee -a $PIPELINE_LOG_CONT; echo \$ec > $EXIT_FILE_CONT"'

docker exec "$CONTAINER" bash -lc "tmux has-session -t \"$SESSION_NAME\""
echo "[ORCH][OK] tmux session started"
echo "[ORCH][PIPELINE_LOG] $REPO_DIR/artifacts/logs/$(basename "$PIPELINE_LOG_CONT")"
echo "[ORCH][EXIT_FILE] $EXIT_FILE_HOST"
echo "[ORCH][NEXT] docker exec -it $CONTAINER tmux attach -t $SESSION_NAME"
