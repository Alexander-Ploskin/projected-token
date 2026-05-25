#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="pt-exp-mgpu"
OUT_DIR="/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated"
PROJECTOR_PATH="artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt"
NUM_SHARDS=4

echo "Stopping E11 training if running..."
tmux kill-session -t e11_train 2>/dev/null || true
docker exec "$CONTAINER" bash -lc 'kill $(pgrep -f "projected_token train --config configs/training/e11_pertoken_gated.yaml") 2>/dev/null || true'
sleep 2

echo "Creating output directory..."
docker exec "$CONTAINER" mkdir -p "$OUT_DIR"

echo "Killing old KILT shard sessions if any..."
for i in 0 1 2 3; do
  tmux kill-session -t "kilt_shard_${i}" 2>/dev/null || true
done
docker exec "$CONTAINER" bash -lc "pkill -f '[b]uild_kilt_index.py' || true"
sleep 2

echo "Starting ${NUM_SHARDS} KILT index shards (1x 3090 Ti each)..."
echo "Output: ${OUT_DIR}"
echo "Projector: ${PROJECTOR_PATH}"

GPUS=(0 3 4 5)
for SHARD_ID in "${!GPUS[@]}"; do
  GPU="${GPUS[$SHARD_ID]}"
  tmux kill-session -t "kilt_shard_${SHARD_ID}" 2>/dev/null || true
  tmux new-session -d -s "kilt_shard_${SHARD_ID}" \
    "docker exec -w /workspace/projected-token ${CONTAINER} bash -lc 'CUDA_VISIBLE_DEVICES=${GPU} HF_TOKEN=?? HF_HOME=/mnt/raid/a-ploskin/hf_cache python3 scripts/build_kilt_index.py --shard-id ${SHARD_ID} --num-shards ${NUM_SHARDS} --output-dir ${OUT_DIR} --projector-path ${PROJECTOR_PATH} --compressor-device cuda:0 --decoder-device cuda:0 --batch-size 256' 2>&1 | tee ${OUT_DIR}/shard_${SHARD_ID}.log"
  echo "  shard ${SHARD_ID} -> GPU ${GPU} (host tmux: kilt_shard_${SHARD_ID})"
done

echo ""
echo "All ${NUM_SHARDS} KILT shards started."
echo "Logs: ${OUT_DIR}/shard_*.log"
echo "Monitor: tmux ls | grep kilt_shard"
