#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/docker_tmux_orchestrator_parallel_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$ORCH_LOG") 2>&1

echo "[ORCH][START] $(date -Is)"
echo "[ORCH][LOG] $ORCH_LOG"

start_stage() {
  local container="$1"
  local session="$2"
  local stage="$3"
  local stage_ts
  stage_ts="$(date +%Y%m%d_%H%M%S)"
  local stage_log_cont="/workspace/projected-token/artifacts/logs/${session}_${stage_ts}.log"
  local exit_file_cont="/workspace/projected-token/artifacts/logs/${session}.exit"

  echo "[ORCH][LAUNCH] stage=${stage} container=${container} session=${session}"
  docker exec \
    -e STAGE="$stage" \
    -e SESSION="$session" \
    -e STAGE_LOG="$stage_log_cont" \
    -e EXIT_FILE="$exit_file_cont" \
    "$container" \
    bash -lc 'cd /workspace/projected-token && mkdir -p artifacts/logs && rm -f "$EXIT_FILE" && tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true && tmux new-session -d -s "$SESSION" "cd /workspace/projected-token && source .venv/bin/activate && export PYTHONUNBUFFERED=1 && export HF_HOME=/data/huggingface && export CUDA_VISIBLE_DEVICES=0 && echo [START] \$(date -Is) stage=$STAGE | tee -a $STAGE_LOG && python -m projected_token train-roadmap --stage $STAGE 2>&1 | tee -a $STAGE_LOG; ec=\${PIPESTATUS[0]}; echo [DONE] \$(date -Is) stage=$STAGE exit=\$ec | tee -a $STAGE_LOG; echo \$ec > $EXIT_FILE"'
}

stage_running() {
  local container="$1"
  local session="$2"
  docker exec "$container" bash -lc "tmux has-session -t \"$session\"" >/dev/null 2>&1
}

stage_exit_code() {
  local session="$1"
  local exit_file_host="$REPO_DIR/artifacts/logs/${session}.exit"
  if [[ ! -f "$exit_file_host" ]]; then
    echo ""
    return 0
  fi
  awk 'NR==1{print; exit}' "$exit_file_host"
}

wait_stage() {
  local container="$1"
  local session="$2"
  local stage="$3"
  echo "[ORCH][WAIT] stage=${stage} session=${session} waiting..."
  while stage_running "$container" "$session"; do
    sleep 60
    echo "[ORCH][WAIT] stage=${stage} still running..."
  done

  local code
  code="$(stage_exit_code "$session")"
  if [[ -z "$code" ]]; then
    echo "[ORCH][ERROR] stage=${stage} exit marker missing"
    return 1
  fi
  if [[ "$code" != "0" ]]; then
    echo "[ORCH][ERROR] stage=${stage} failed with exit_code=${code}"
    return 1
  fi
  echo "[ORCH][OK] stage=${stage} completed successfully"
}

rm -f "$REPO_DIR"/artifacts/logs/stage_{a,b,c,d,e}.exit || true

# Start 3 parallel branches immediately.
start_stage "pt-exp-gpu0" "stage_a" "a"
start_stage "pt-exp-gpu1" "stage_b" "b"
start_stage "pt-exp-gpu2" "stage_d" "d"

# Stage C depends on stage B.
(
  wait_stage "pt-exp-gpu1" "stage_b" "b"
  start_stage "pt-exp-gpu1" "stage_c" "c"
  wait_stage "pt-exp-gpu1" "stage_c" "c"
) &
pid_c_chain=$!

# Stage E starts as soon as either stage A or stage D frees a container.
e_started=0
while [[ "$e_started" -eq 0 ]]; do
  if ! stage_running "pt-exp-gpu2" "stage_d"; then
    code_d="$(stage_exit_code "stage_d")"
    if [[ "$code_d" == "0" ]]; then
      echo "[ORCH][CHAIN] launching stage=e on pt-exp-gpu2 (after stage=d)"
      start_stage "pt-exp-gpu2" "stage_e" "e"
      e_started=1
      break
    elif [[ -n "$code_d" ]]; then
      echo "[ORCH][ERROR] stage=d finished with exit_code=${code_d}; cannot launch stage=e on that branch"
      break
    fi
  fi

  if ! stage_running "pt-exp-gpu0" "stage_a"; then
    code_a="$(stage_exit_code "stage_a")"
    if [[ "$code_a" == "0" ]]; then
      echo "[ORCH][CHAIN] launching stage=e on pt-exp-gpu0 (after stage=a)"
      start_stage "pt-exp-gpu0" "stage_e" "e"
      e_started=1
      break
    elif [[ -n "$code_a" ]]; then
      echo "[ORCH][WARN] stage=a finished with exit_code=${code_a}; waiting for stage=d branch to launch stage=e"
    fi
  fi

  sleep 30
done

# Wait for stage A completion regardless of whether it was used for stage E launch.
wait_stage "pt-exp-gpu0" "stage_a" "a" || true

if [[ "$e_started" -eq 1 ]]; then
  # Detect where stage E was started.
  if stage_running "pt-exp-gpu2" "stage_e" || [[ "$(stage_exit_code "stage_e")" != "" ]]; then
    wait_stage "pt-exp-gpu2" "stage_e" "e" || wait_stage "pt-exp-gpu0" "stage_e" "e"
  else
    wait_stage "pt-exp-gpu0" "stage_e" "e"
  fi
fi

wait "$pid_c_chain"

echo "[ORCH][DONE] $(date -Is)"
