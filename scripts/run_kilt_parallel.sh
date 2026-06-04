#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="pt-exp-mgpu"
OUT_DIR="/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated"
PROJECTOR_PATH="artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt"
NUM_SHARDS=3
HF_TOKEN="${HF_TOKEN:-}"
HF_ENV=""
if [[ -n "${HF_TOKEN}" ]]; then
  HF_ENV="HF_TOKEN=${HF_TOKEN} "
fi

start_shard() {
  local SHARD_ID=$1
  local GPUS=$2
  docker exec -d -w /workspace/projected-token "$CONTAINER" bash -lc \
    "CUDA_VISIBLE_DEVICES=${GPUS} ${HF_ENV}HF_HOME=/mnt/raid/a-ploskin/hf_cache \
     nohup python3 -u scripts/build_kilt_index.py \
       --shard-id ${SHARD_ID} --num-shards ${NUM_SHARDS} \
       --output-dir ${OUT_DIR} --projector-path ${PROJECTOR_PATH} \
       --compressor-device cuda:0 --decoder-device cuda:1 \
       --batch-size 256 --checkpoint-every 500000 \
       >> ${OUT_DIR}/shard_${SHARD_ID}.log 2>&1 &"
  echo "  shard ${SHARD_ID} -> CUDA_VISIBLE_DEVICES=${GPUS} (compressor=cuda:0, decoder=cuda:1)"
}

echo "Killing old KILT processes..."
docker exec "$CONTAINER" bash -lc "pkill -9 -f '[b]uild_kilt_index.py' || true"
sleep 3
docker exec "$CONTAINER" mkdir -p "$OUT_DIR"

echo "Starting KILT index (2 parallel + shard 2 queued)..."
echo "PyTorch CUDA 0-3 = 3090 Ti | CUDA 4-5 = 2080 Ti (unused)"

# OSCAR needs 2x 3090 per worker; 4x 3090 available -> max 2 parallel
start_shard 0 "0,1"
start_shard 1 "2,3"

# Shard 2 starts when shard 0 finishes
nohup bash -c "
  while docker exec ${CONTAINER} pgrep -f 'build_kilt_index.py --shard-id 0' >/dev/null 2>&1; do sleep 60; done
  docker exec -d -w /workspace/projected-token ${CONTAINER} bash -lc \
    \"CUDA_VISIBLE_DEVICES=0,1 ${HF_ENV}HF_HOME=/mnt/raid/a-ploskin/hf_cache \
     nohup python3 -u scripts/build_kilt_index.py \
       --shard-id 2 --num-shards ${NUM_SHARDS} \
       --output-dir ${OUT_DIR} --projector-path ${PROJECTOR_PATH} \
       --compressor-device cuda:0 --decoder-device cuda:1 \
       --batch-size 256 --checkpoint-every 500000 \
       >> ${OUT_DIR}/shard_2.log 2>&1 &\"
" >/dev/null 2>&1 &

echo ""
echo "Shards 0,1 running. Shard 2 auto-starts when shard 0 finishes."
echo "Logs: ${OUT_DIR}/shard_*.log"
