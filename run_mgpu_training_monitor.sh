#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"
MONITOR_LOG="$LOG_DIR/mgpu_training_monitor_$(date +%Y%m%d_%H%M%S).log"

INTERVAL_SEC="${INTERVAL_SEC:-120}"
CONTAINER="${CONTAINER:-pt-exp-mgpu}"

echo "[MONITOR][START] $(date -Is)"
echo "[MONITOR][LOG] $MONITOR_LOG"
echo "[MONITOR][INTERVAL_SEC] $INTERVAL_SEC"
echo "[MONITOR][CONTAINER] $CONTAINER"

while true; do
  {
    echo
    echo "===== $(date -Is) ====="
    echo "[MONITOR] tmux sessions:"
    docker exec "$CONTAINER" bash -lc "tmux ls || true"

    latest_log="$(ls -1t "$LOG_DIR"/stage_*.log 2>/dev/null | awk 'NR==1{print; exit}')"
    if [[ -z "${latest_log:-}" ]]; then
      echo "[MONITOR] no stage logs yet"
    else
      echo "[MONITOR] latest_stage_log=$latest_log"
      python3 - <<PY
import re
from pathlib import Path

p = Path(r"$latest_log")
raw = p.read_text(encoding="utf-8", errors="replace")
text = raw.replace("\r", "\n")

epoch_matches = list(re.finditer(r"Epoch\\s+(\\d+)/(\\d+)", text))
loss_step_matches = list(re.finditer(r"loss=([0-9.]+),\\s*step=(\\d+)", text))
val_loss_matches = list(re.finditer(r"Val - Loss:\\s*([0-9.]+),\\s*MRR:\\s*([0-9.]+)", text))
val_recall_matches = list(re.finditer(r"Val - R@1:\\s*([0-9.]+),\\s*R@5:\\s*([0-9.]+),\\s*R@10:\\s*([0-9.]+)", text))
done_matches = list(re.finditer(r"\\[DONE\\].*", text))
err_matches = list(re.finditer(r"(OutOfMemoryError|\\[ERROR\\]|Traceback.*)", text))

if epoch_matches:
    e = epoch_matches[-1]
    print(f"[MONITOR] epoch={e.group(1)}/{e.group(2)}")
if loss_step_matches:
    ls = loss_step_matches[-1]
    print(f"[MONITOR] train_step={ls.group(2)} train_loss={ls.group(1)}")
if val_loss_matches:
    vl = val_loss_matches[-1]
    print(f"[MONITOR] val_loss={vl.group(1)} val_mrr={vl.group(2)}")
if val_recall_matches:
    vr = val_recall_matches[-1]
    print(f"[MONITOR] val_r@1={vr.group(1)} val_r@5={vr.group(2)} val_r@10={vr.group(3)}")
if done_matches:
    print(f"[MONITOR] stage_done_line={done_matches[-1].group(0)}")
if err_matches:
    print(f"[MONITOR] last_error_like={err_matches[-1].group(0)[:200]}")
PY
    fi

    echo "[MONITOR] gpu snapshot:"
    docker exec "$CONTAINER" bash -lc "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
    echo "[MONITOR] gpu processes:"
    docker exec "$CONTAINER" bash -lc "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader"
  } | tee -a "$MONITOR_LOG"

  sleep "$INTERVAL_SEC"
done
