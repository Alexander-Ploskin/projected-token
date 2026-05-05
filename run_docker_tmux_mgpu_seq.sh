#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/docker_tmux_mgpu_orchestrator_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$ORCH_LOG") 2>&1

CONTAINER="pt-exp-mgpu"
START_STAGE="a"
INCLUDE_FREEZE_EVAL=0

# Safe runtime profile (override via environment if needed).
CUDA_VISIBLE_DEVICES_SAFE="${CUDA_VISIBLE_DEVICES_SAFE:-0,1,2}"
OSCAR_DECODER_DEVICE_SAFE="${OSCAR_DECODER_DEVICE_SAFE:-cuda:1}"
OSCAR_COMPRESSOR_DEVICE_SAFE="${OSCAR_COMPRESSOR_DEVICE_SAFE:-cuda:0}"
PYTORCH_CUDA_ALLOC_CONF_SAFE="${PYTORCH_CUDA_ALLOC_CONF_SAFE:-expandable_segments:True}"

usage() {
  cat <<'EOF'
Usage: run_docker_tmux_mgpu_seq.sh [--from-stage <stage>] [--include-freeze-eval]

Options:
  --from-stage <stage>      Start from one of: a, b, c, d, e. Default: a
  --include-freeze-eval     Run freeze-eval before training stages
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-stage)
      START_STAGE="${2:-}"
      shift 2
      ;;
    --include-freeze-eval)
      INCLUDE_FREEZE_EVAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ORCH][ERROR] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

case "$START_STAGE" in
  a|b|c|d|e) ;;
  *)
    echo "[ORCH][ERROR] Invalid --from-stage value: $START_STAGE"
    usage
    exit 2
    ;;
esac

echo "[ORCH][START] $(date -Is)"
echo "[ORCH][LOG] $ORCH_LOG"
echo "[ORCH][CONTAINER] $CONTAINER"
echo "[ORCH][PROFILE] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_SAFE OSCAR_COMPRESSOR_DEVICE=$OSCAR_COMPRESSOR_DEVICE_SAFE OSCAR_DECODER_DEVICE=$OSCAR_DECODER_DEVICE_SAFE"
echo "[ORCH][FROM_STAGE] $START_STAGE"
echo "[ORCH][INCLUDE_FREEZE_EVAL] $INCLUDE_FREEZE_EVAL"

if ! docker ps --format '{{.Names}}' | awk -v c="$CONTAINER" '$0==c{found=1} END{exit(found?0:1)}'; then
  echo "[ORCH][ERROR] Container '$CONTAINER' is not running."
  exit 1
fi

stage_to_session() {
  local stage="$1"
  case "$stage" in
    freeze-eval) echo "stage_freeze_eval" ;;
    a) echo "stage_a" ;;
    b) echo "stage_b" ;;
    c) echo "stage_c" ;;
    d) echo "stage_d" ;;
    e) echo "stage_e" ;;
    *) echo "stage_${stage}" ;;
  esac
}

run_stage() {
  local stage="$1"
  local session
  session="$(stage_to_session "$stage")"
  local stage_ts
  stage_ts="$(date +%Y%m%d_%H%M%S)"
  local stage_log_cont="/workspace/projected-token/artifacts/logs/${session}_${stage_ts}.log"
  local exit_file_cont="/workspace/projected-token/artifacts/logs/${session}.exit"
  local exit_file_host="$REPO_DIR/artifacts/logs/${session}.exit"

  echo "[ORCH][STAGE] stage=${stage} session=${session}"
  docker exec \
    -e STAGE="$stage" \
    -e SESSION="$session" \
    -e STAGE_LOG="$stage_log_cont" \
    -e EXIT_FILE="$exit_file_cont" \
    -e CUDA_VISIBLE_DEVICES_SAFE="$CUDA_VISIBLE_DEVICES_SAFE" \
    -e OSCAR_DECODER_DEVICE_SAFE="$OSCAR_DECODER_DEVICE_SAFE" \
    -e OSCAR_COMPRESSOR_DEVICE_SAFE="$OSCAR_COMPRESSOR_DEVICE_SAFE" \
    -e PYTORCH_CUDA_ALLOC_CONF_SAFE="$PYTORCH_CUDA_ALLOC_CONF_SAFE" \
    "$CONTAINER" \
    bash -lc 'cd /workspace/projected-token && mkdir -p artifacts/logs && rm -f "$EXIT_FILE" && tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true && tmux new-session -d -s "$SESSION" "cd /workspace/projected-token && source .venv/bin/activate && export PYTHONUNBUFFERED=1 && export HF_HOME=/data/huggingface && export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_SAFE && export OSCAR_DISABLE_ADAPTER_WARMUP=1 && export OSCAR_DECODER_DEVICE=$OSCAR_DECODER_DEVICE_SAFE && export OSCAR_COMPRESSOR_DEVICE=$OSCAR_COMPRESSOR_DEVICE_SAFE && export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF_SAFE && echo [START] \$(date -Is) stage=$STAGE | tee -a $STAGE_LOG && python -m projected_token train-roadmap --stage $STAGE 2>&1 | tee -a $STAGE_LOG; ec=\${PIPESTATUS[0]}; echo [DONE] \$(date -Is) stage=$STAGE exit=\$ec | tee -a $STAGE_LOG; echo \$ec > $EXIT_FILE"'

  docker exec "$CONTAINER" bash -lc "tmux has-session -t \"$session\""
  echo "[ORCH][WAIT] stage=${stage} is running..."
  while docker exec "$CONTAINER" bash -lc "tmux has-session -t \"$session\"" >/dev/null 2>&1; do
    sleep 60
    echo "[ORCH][WAIT] stage=${stage} still running..."
  done

  if [[ ! -f "$exit_file_host" ]]; then
    echo "[ORCH][ERROR] exit marker not found for stage=${stage}: $exit_file_host"
    return 1
  fi
  local exit_code
  exit_code="$(awk 'NR==1{print; exit}' "$exit_file_host")"
  if [[ "$exit_code" != "0" ]]; then
    echo "[ORCH][ERROR] stage=${stage} failed with exit_code=${exit_code}"
    return 1
  fi
  echo "[ORCH][OK] stage=${stage} completed successfully"
}

stages=()
if [[ "$INCLUDE_FREEZE_EVAL" -eq 1 ]]; then
  stages+=("freeze-eval")
fi
for st in a b c d e; do
  if [[ "$st" < "$START_STAGE" ]]; then
    continue
  fi
  stages+=("$st")
done

for st in "${stages[@]}"; do
  run_stage "$st"
done

echo "[ORCH][DONE] $(date -Is)"
