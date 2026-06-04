#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="pt-exp-mgpu"
OUT_DIR="/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated"
PROJECTOR_PATH="artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt"
SHARD_ID=2
NUM_SHARDS=3
NUM_PARTS=2
HF_TOKEN="${HF_TOKEN:-}"
HF_ENV=""
if [[ -n "${HF_TOKEN}" ]]; then
  HF_ENV="HF_TOKEN=${HF_TOKEN} "
fi

start_part() {
  local PART_ID=$1
  local GPUS=$2
  docker exec -d -w /workspace/projected-token "$CONTAINER" bash -lc \
    "CUDA_VISIBLE_DEVICES=${GPUS} ${HF_ENV}HF_HOME=/mnt/raid/a-ploskin/hf_cache \
     nohup python3 -u scripts/build_kilt_index.py \
       --shard-id ${SHARD_ID} --num-shards ${NUM_SHARDS} \
       --part-id ${PART_ID} --num-parts ${NUM_PARTS} \
       --no-resume \
       --output-dir ${OUT_DIR} --projector-path ${PROJECTOR_PATH} \
       --compressor-device cuda:0 --decoder-device cuda:1 \
       --batch-size 256 --checkpoint-every 500000 \
       >> ${OUT_DIR}/shard_${SHARD_ID}_part${PART_ID}.log 2>&1 &"
  echo "  shard ${SHARD_ID} part ${PART_ID} -> CUDA_VISIBLE_DEVICES=${GPUS}"
}

echo "Killing old KILT processes..."
docker exec "$CONTAINER" bash -lc "pkill -9 -f '[b]uild_kilt_index.py' || true"
sleep 2
docker exec "$CONTAINER" mkdir -p "$OUT_DIR"

echo "Starting shard ${SHARD_ID} split into ${NUM_PARTS} parallel workers..."
echo "PyTorch CUDA 0-3 = 3090 Ti | CUDA 4-5 = 2080 Ti (unused)"
start_part 0 "0,1"
start_part 1 "2,3"

echo ""
echo "After both parts finish, merge with:"
echo "  docker exec -w /workspace/projected-token ${CONTAINER} python3 scripts/merge_kilt_shard_parts.py \\"
echo "    --output-dir ${OUT_DIR} --shard-id ${SHARD_ID} --num-parts ${NUM_PARTS}"
echo ""
echo "Logs: ${OUT_DIR}/shard_${SHARD_ID}_part_*.log"
