#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/docker_tmux_orchestrator_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$ORCH_LOG") 2>&1

echo "[ORCH][START] $(date -Is)"
echo "[ORCH][LOG] $ORCH_LOG"

run_stage() {
  local container="$1"
  local session="$2"
  local stage="$3"
  local stage_ts
  stage_ts="$(date +%Y%m%d_%H%M%S)"
  local stage_log_cont="/workspace/projected-token/artifacts/logs/${session}_${stage_ts}.log"
  local exit_file_cont="/workspace/projected-token/artifacts/logs/${session}.exit"
  local exit_file_host="$REPO_DIR/artifacts/logs/${session}.exit"

  echo "[ORCH][STAGE] stage=${stage} container=${container} session=${session}"
  docker exec \
    -e STAGE="$stage" \
    -e SESSION="$session" \
    -e STAGE_LOG="$stage_log_cont" \
    -e EXIT_FILE="$exit_file_cont" \
    "$container" \
    bash -lc 'cd /workspace/projected-token && mkdir -p artifacts/logs && rm -f "$EXIT_FILE" && tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true && tmux new-session -d -s "$SESSION" "cd /workspace/projected-token && source .venv/bin/activate && export PYTHONUNBUFFERED=1 && export HF_HOME=/data/huggingface && export CUDA_VISIBLE_DEVICES=0 && echo [START] \$(date -Is) stage=$STAGE | tee -a $STAGE_LOG && python -m projected_token train-roadmap --stage $STAGE 2>&1 | tee -a $STAGE_LOG; ec=\${PIPESTATUS[0]}; echo [DONE] \$(date -Is) stage=$STAGE exit=\$ec | tee -a $STAGE_LOG; echo \$ec > $EXIT_FILE"'

  docker exec "$container" bash -lc "tmux has-session -t \"$session\""
  echo "[ORCH][WAIT] stage=${stage} is running..."
  while docker exec "$container" bash -lc "tmux has-session -t \"$session\"" >/dev/null 2>&1; do
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

run_stage "pt-exp-gpu0" "stage_freeze_eval" "freeze-eval"
run_stage "pt-exp-gpu0" "stage_a" "a"
run_stage "pt-exp-gpu1" "stage_b" "b"
run_stage "pt-exp-gpu2" "stage_c" "c"
run_stage "pt-exp-gpu0" "stage_d" "d"
run_stage "pt-exp-gpu1" "stage_e" "e"

echo "[ORCH][DONE] $(date -Is)"
