#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="pt-exp-mgpu"
OUT_DIR="artifacts/results/retrieval/beir_nq_e9_6000_shards"

echo "Creating output directory inside container..."
docker exec "$CONTAINER" mkdir -p "/workspace/projected-token/$OUT_DIR"

echo "Killing old tmux sessions if any..."
docker exec "$CONTAINER" bash -lc "tmux kill-session -t nq_shard_0 >/dev/null 2>&1 || true"
docker exec "$CONTAINER" bash -lc "tmux kill-session -t nq_shard_1 >/dev/null 2>&1 || true"
docker exec "$CONTAINER" bash -lc "tmux kill-session -t nq_shard_2 >/dev/null 2>&1 || true"
docker exec "$CONTAINER" bash -lc "tmux kill-session -t nq_shard_3 >/dev/null 2>&1 || true"

echo "Killing stray python processes..."
docker exec "$CONTAINER" bash -lc "pkill -f '[e]ncode_nq_shard.py' || true"

echo "Starting 4 shards in tmux with 1x 3090 Ti per shard..."
# PyTorch GPU indices:
# 0, 1, 2, 3 are RTX 3090 Ti (24GB)
# 4, 5 are RTX 2080 Ti (11GB)

# Shard 0: PyTorch 0 (Physical 0)
docker exec -d -w /workspace/projected-token "$CONTAINER" tmux new-session -d -s nq_shard_0 "CUDA_VISIBLE_DEVICES=0 HF_TOKEN=?? HF_HOME=/data/huggingface python scripts/encode_nq_shard.py --shard-id 0 --num-shards 4 --compressor-device cuda:0 --decoder-device cuda:0 | tee $OUT_DIR/shard_0.log"

# Shard 1: PyTorch 1 (Physical 3)
docker exec -d -w /workspace/projected-token "$CONTAINER" tmux new-session -d -s nq_shard_1 "CUDA_VISIBLE_DEVICES=1 HF_TOKEN=?? HF_HOME=/data/huggingface python scripts/encode_nq_shard.py --shard-id 1 --num-shards 4 --compressor-device cuda:0 --decoder-device cuda:0 | tee $OUT_DIR/shard_1.log"

# Shard 2: PyTorch 2 (Physical 4)
docker exec -d -w /workspace/projected-token "$CONTAINER" tmux new-session -d -s nq_shard_2 "CUDA_VISIBLE_DEVICES=2 HF_TOKEN=?? HF_HOME=/data/huggingface python scripts/encode_nq_shard.py --shard-id 2 --num-shards 4 --compressor-device cuda:0 --decoder-device cuda:0 | tee $OUT_DIR/shard_2.log"

# Shard 3: PyTorch 3 (Physical 5)
docker exec -d -w /workspace/projected-token "$CONTAINER" tmux new-session -d -s nq_shard_3 "CUDA_VISIBLE_DEVICES=3 HF_TOKEN=?? HF_HOME=/data/huggingface python scripts/encode_nq_shard.py --shard-id 3 --num-shards 4 --compressor-device cuda:0 --decoder-device cuda:0 | tee $OUT_DIR/shard_3.log"

echo "All 4 shards started!"
echo "Check logs in $OUT_DIR/shard_X.log"
echo "When they finish, run: docker exec -w /workspace/projected-token pt-exp-mgpu HF_TOKEN=?? HF_HOME=/data/huggingface python scripts/merge_and_eval_nq.py --num-shards 4"