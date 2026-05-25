#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_PATH="configs/training/e11_pertoken_gated.yaml"

export CUDA_VISIBLE_DEVICES="0,3,4,5"
export OSCAR_COMPRESSOR_DEVICE="cuda:0"
export OSCAR_DECODER_DEVICE="cuda:1"
export ASYNC_VALIDATION_DEVICE="cuda:2"

export HF_HOME=/data/huggingface
export PYTHONUNBUFFERED=1
export OSCAR_DISABLE_ADAPTER_WARMUP=1

tmux kill-session -t e11_train 2>/dev/null || true
tmux new-session -d -s e11_train "docker exec -w /workspace/projected-token pt-exp-mgpu bash -lc 'pip install click transformers datasets h5py pyyaml tqdm faiss-cpu accelerate && pip install -e . && export PYTHONPATH=/workspace/projected-token:\${PYTHONPATH:-} && HF_TOKEN=?? HF_HOME=/data/huggingface CUDA_VISIBLE_DEVICES=0,3,4,5 OSCAR_COMPRESSOR_DEVICE=cuda:0 OSCAR_DECODER_DEVICE=cuda:1 ASYNC_VALIDATION_DEVICE=cuda:2 OSCAR_DISABLE_ADAPTER_WARMUP=1 python3 -m projected_token train --config $CONFIG_PATH' 2>&1 | tee artifacts/logs/stage_b_e11_pertoken_gated_$(date +%Y%m%d_%H%M%S).log"

echo "Started E11 training in tmux session 'e11_train'"
