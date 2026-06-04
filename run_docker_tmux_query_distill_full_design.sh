#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"

CONTAINER="${CONTAINER:-pt-exp-mgpu}"
SESSION_NAME="${SESSION_NAME:-stage_b_query_distill_full_design}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/training/stage_b_query_distill_bge_base_flatten_full_design.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,3,4}"
OSCAR_COMPRESSOR_DEVICE_VALUE="${OSCAR_COMPRESSOR_DEVICE_VALUE:-cuda:0}"
OSCAR_DECODER_DEVICE_VALUE="${OSCAR_DECODER_DEVICE_VALUE:-cuda:1}"
MSMARCO_DEVICE_VALUE="${MSMARCO_DEVICE_VALUE:-cuda:0}"
BEIR_PIPELINE_A_DEVICE_VALUE="${BEIR_PIPELINE_A_DEVICE_VALUE:-cuda:1}"
BEIR_PIPELINE_B_DEVICE_VALUE="${BEIR_PIPELINE_B_DEVICE_VALUE:-cuda:2}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"

TEACHER_SCRIPT="${TEACHER_SCRIPT:-scripts/build_query_distill_full_h5.sh}"
TEACHER_OUT_DIR="${TEACHER_OUT_DIR:-artifacts/teacher-embeddings/query-doc-bge-base-full}"
SKIP_TEACHER_IF_EXISTS="${SKIP_TEACHER_IF_EXISTS:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="$LOG_DIR/docker_tmux_query_distill_full_design_${TIMESTAMP}.log"
PIPELINE_LOG_CONT="/workspace/projected-token/artifacts/logs/${SESSION_NAME}_${TIMESTAMP}.log"
EXIT_FILE_CONT="/workspace/projected-token/artifacts/logs/${SESSION_NAME}.exit"
EXIT_FILE_HOST="$REPO_DIR/artifacts/logs/${SESSION_NAME}.exit"

exec > >(tee -a "$ORCH_LOG") 2>&1

echo "[ORCH][START] $(date -Is)"
echo "[ORCH][CONTAINER] $CONTAINER"
echo "[ORCH][SESSION] $SESSION_NAME"
echo "[ORCH][TRAIN_CONFIG] $TRAIN_CONFIG_PATH"
echo "[ORCH][TEACHER_SCRIPT] $TEACHER_SCRIPT"
echo "[ORCH][SKIP_TEACHER_IF_EXISTS] $SKIP_TEACHER_IF_EXISTS"
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
docker exec "$CONTAINER" bash -lc "test -f /workspace/projected-token/$TEACHER_SCRIPT"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[ORCH][SETUP] Installing project dependencies inside container"
  docker exec "$CONTAINER" bash -lc "cd /workspace/projected-token && python -m pip install --upgrade pip && python -m pip install -e ."
fi

rm -f "$EXIT_FILE_HOST"

docker exec \
  -e SESSION_NAME="$SESSION_NAME" \
  -e PIPELINE_LOG_CONT="$PIPELINE_LOG_CONT" \
  -e EXIT_FILE_CONT="$EXIT_FILE_CONT" \
  -e TRAIN_CONFIG_PATH="$TRAIN_CONFIG_PATH" \
  -e TEACHER_SCRIPT="$TEACHER_SCRIPT" \
  -e TEACHER_OUT_DIR="$TEACHER_OUT_DIR" \
  -e SKIP_TEACHER_IF_EXISTS="$SKIP_TEACHER_IF_EXISTS" \
  -e HF_TOKEN_VALUE="$HF_TOKEN_VALUE" \
  -e CUDA_VISIBLE_DEVICES_VALUE="$CUDA_VISIBLE_DEVICES_VALUE" \
  -e OSCAR_COMPRESSOR_DEVICE_VALUE="$OSCAR_COMPRESSOR_DEVICE_VALUE" \
  -e OSCAR_DECODER_DEVICE_VALUE="$OSCAR_DECODER_DEVICE_VALUE" \
  -e MSMARCO_DEVICE_VALUE="$MSMARCO_DEVICE_VALUE" \
  -e BEIR_PIPELINE_A_DEVICE_VALUE="$BEIR_PIPELINE_A_DEVICE_VALUE" \
  -e BEIR_PIPELINE_B_DEVICE_VALUE="$BEIR_PIPELINE_B_DEVICE_VALUE" \
  -e PYTORCH_CUDA_ALLOC_CONF_VALUE="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
  "$CONTAINER" \
  bash -lc 'cd /workspace/projected-token && mkdir -p artifacts/logs && rm -f "$EXIT_FILE_CONT" && tmux kill-session -t "$SESSION_NAME" >/dev/null 2>&1 || true && cat > /tmp/query_distill_full_design_run.sh <<'"'"'EOS'"'"'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/projected-token
export PYTHONUNBUFFERED=1
export HF_HOME=/data/huggingface
export PYTHONPATH=/workspace/projected-token:${PYTHONPATH:-}
export HF_TOKEN="${HF_TOKEN_VALUE:-}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN_VALUE:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0,3,4}"
export OSCAR_DISABLE_ADAPTER_WARMUP=1
export OSCAR_COMPRESSOR_DEVICE="${OSCAR_COMPRESSOR_DEVICE_VALUE:-cuda:0}"
export OSCAR_DECODER_DEVICE="${OSCAR_DECODER_DEVICE_VALUE:-cuda:1}"
export MSMARCO_DEVICE="${MSMARCO_DEVICE_VALUE:-cuda:0}"
export BEIR_PIPELINE_A_DEVICE="${BEIR_PIPELINE_A_DEVICE_VALUE:-cuda:1}"
export BEIR_PIPELINE_B_DEVICE="${BEIR_PIPELINE_B_DEVICE_VALUE:-cuda:2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"
echo "[START] $(date -Is) train_config=${TRAIN_CONFIG_PATH} hf_token_set=$( [ -n "${HF_TOKEN:-}" ] && echo yes || echo no )" | tee -a "${PIPELINE_LOG_CONT}"
missing=0
if [[ ! -d "${TEACHER_OUT_DIR}" ]] || [[ -z "$(ls -A "${TEACHER_OUT_DIR}"/*.h5 2>/dev/null)" ]]; then
  missing=1
fi
if [[ "${SKIP_TEACHER_IF_EXISTS:-1}" == "1" && "${missing}" == "0" ]]; then
  echo "[cache] teacher embeddings found; skipping generation" | tee -a "${PIPELINE_LOG_CONT}"
else
  echo "[cache] teacher embeddings missing or skip disabled; running generation" | tee -a "${PIPELINE_LOG_CONT}"
  bash "${TEACHER_SCRIPT}" 2>&1 | tee -a "${PIPELINE_LOG_CONT}"
fi
python - <<'PY'
import glob
import h5py
import os
paths = sorted(glob.glob(os.path.join(os.environ["TEACHER_OUT_DIR"], "*.h5")))
print(f"[health] h5_count {len(paths)}")
for p in paths:
    with h5py.File(p, "r") as h:
        n = int(h["query_embeddings"].shape[0])
        d = int(h["query_embeddings"].shape[1])
    print(f"[health] {p} samples={n} dim={d}")
PY
python -m projected_token train --config "${TRAIN_CONFIG_PATH}" ${TRAIN_EXTRA_ARGS:-} 2>&1 | tee -a "${PIPELINE_LOG_CONT}"
ec=${PIPESTATUS[0]}
echo "[DONE] $(date -Is) exit=${ec}" | tee -a "${PIPELINE_LOG_CONT}"
echo "${ec}" > "${EXIT_FILE_CONT}"
exit "${ec}"
EOS
chmod +x /tmp/query_distill_full_design_run.sh && tmux new-session -d -s "$SESSION_NAME" \
  -e SESSION_NAME="$SESSION_NAME" \
  -e PIPELINE_LOG_CONT="$PIPELINE_LOG_CONT" \
  -e EXIT_FILE_CONT="$EXIT_FILE_CONT" \
  -e TRAIN_CONFIG_PATH="$TRAIN_CONFIG_PATH" \
  -e TEACHER_SCRIPT="$TEACHER_SCRIPT" \
  -e TEACHER_OUT_DIR="$TEACHER_OUT_DIR" \
  -e SKIP_TEACHER_IF_EXISTS="$SKIP_TEACHER_IF_EXISTS" \
  -e HF_TOKEN_VALUE="$HF_TOKEN_VALUE" \
  -e CUDA_VISIBLE_DEVICES_VALUE="$CUDA_VISIBLE_DEVICES_VALUE" \
  -e OSCAR_COMPRESSOR_DEVICE_VALUE="$OSCAR_COMPRESSOR_DEVICE_VALUE" \
  -e OSCAR_DECODER_DEVICE_VALUE="$OSCAR_DECODER_DEVICE_VALUE" \
  -e MSMARCO_DEVICE_VALUE="$MSMARCO_DEVICE_VALUE" \
  -e BEIR_PIPELINE_A_DEVICE_VALUE="$BEIR_PIPELINE_A_DEVICE_VALUE" \
  -e BEIR_PIPELINE_B_DEVICE_VALUE="$BEIR_PIPELINE_B_DEVICE_VALUE" \
  -e PYTORCH_CUDA_ALLOC_CONF_VALUE="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
  "bash /tmp/query_distill_full_design_run.sh"'

docker exec "$CONTAINER" bash -lc "tmux has-session -t \"$SESSION_NAME\""
echo "[ORCH][OK] tmux session started"
echo "[ORCH][PIPELINE_LOG] $REPO_DIR/artifacts/logs/$(basename "$PIPELINE_LOG_CONT")"
echo "[ORCH][EXIT_FILE] $EXIT_FILE_HOST"
echo "[ORCH][NEXT] docker exec -it $CONTAINER tmux attach -t $SESSION_NAME"
