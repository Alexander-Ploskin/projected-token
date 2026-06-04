#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="pt-exp-mgpu"
OUT_DIR="/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated"
MASTER_LOG="${OUT_DIR}/top100_with_texts.log"
HF_TOKEN="${HF_TOKEN:-}"
HF_ENV=""
if [[ -n "${HF_TOKEN}" ]]; then
  HF_ENV="HF_TOKEN=${HF_TOKEN} "
fi

docker exec "$CONTAINER" bash -lc "pkill -9 -f '[s]earch_kilt_popqa.py' || true"
sleep 2

docker exec -d -w /workspace/projected-token "$CONTAINER" bash -lc \
  "CUDA_VISIBLE_DEVICES=0,1 ${HF_ENV}HF_HOME=/mnt/raid/a-ploskin/hf_cache \
   nohup bash scripts/_run_kilt_top100_with_texts_inner.sh \
     > ${MASTER_LOG} 2>&1 &"

echo "Started sequential PopQA + HotpotQA top-100 re-run with document texts."
echo "Master log: ${MASTER_LOG}"
