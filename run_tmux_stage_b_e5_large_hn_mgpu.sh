#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/venv}"
CONFIG_PATH="${CONFIG_PATH:-configs/training/stage_b_distill_e5_large_hn.yaml}"
SESSION_NAME="${SESSION_NAME:-stage_b_e5_large_hn_mgpu}"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
EXIT_FILE="$LOG_DIR/${SESSION_NAME}.exit"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,3,4}"
OSCAR_COMPRESSOR_DEVICE_VALUE="${OSCAR_COMPRESSOR_DEVICE_VALUE:-cuda:0}"
OSCAR_DECODER_DEVICE_VALUE="${OSCAR_DECODER_DEVICE_VALUE:-cuda:1}"
ASYNC_VALIDATION_DEVICE_VALUE="${ASYNC_VALIDATION_DEVICE_VALUE:-cuda:2}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"

PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[ERROR] venv not found at $VENV_DIR"
  echo "[HINT] create it first and install deps into it"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python not found in venv: $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$REPO_DIR/$CONFIG_PATH" ]]; then
  echo "[ERROR] config not found: $REPO_DIR/$CONFIG_PATH"
  exit 1
fi

tmux kill-session -t "$SESSION_NAME" >/dev/null 2>&1 || true
rm -f "$EXIT_FILE"

RUN_CMD=$(cat <<EOF
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export OSCAR_DISABLE_ADAPTER_WARMUP=1
export OSCAR_COMPRESSOR_DEVICE="$OSCAR_COMPRESSOR_DEVICE_VALUE"
export OSCAR_DECODER_DEVICE="$OSCAR_DECODER_DEVICE_VALUE"
export ASYNC_VALIDATION_DEVICE="$ASYNC_VALIDATION_DEVICE_VALUE"
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE"
export PYTHONPATH="$REPO_DIR:\${PYTHONPATH:-}"
echo "[START] \$(date -Is) config=$CONFIG_PATH cuda_visible=\$CUDA_VISIBLE_DEVICES oscar_compressor=\$OSCAR_COMPRESSOR_DEVICE oscar_decoder=\$OSCAR_DECODER_DEVICE async_device=\$ASYNC_VALIDATION_DEVICE" | tee -a "$LOG_FILE"
"$PYTHON_BIN" -m projected_token train --config "$CONFIG_PATH" \${TRAIN_EXTRA_ARGS:-} 2>&1 | tee -a "$LOG_FILE"
ec=\${PIPESTATUS[0]}
echo "[DONE] \$(date -Is) exit=\$ec" | tee -a "$LOG_FILE"
echo "\$ec" > "$EXIT_FILE"
exit "\$ec"
EOF
)

tmux new-session -d -s "$SESSION_NAME" "bash -lc $(printf '%q' "$RUN_CMD")"
echo "[OK] Started tmux session: $SESSION_NAME"
echo "[OK] Log file: $LOG_FILE"
echo "[OK] Exit marker: $EXIT_FILE"
echo "[NEXT] tmux attach -t $SESSION_NAME"
