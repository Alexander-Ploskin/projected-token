#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="pt-exp-mgpu"
OUT_DIR="/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated"
HOTPOT_PARQUET="/data/hotpotqa/distractor/validation.parquet"
RESULT="artifacts/results/retrieval/kilt_e11_hotpotqa_distractor_top100.jsonl"
LOG="${OUT_DIR}/hotpotqa_search.log"
HF_TOKEN="${HF_TOKEN:-}"
HF_ENV=""
if [[ -n "${HF_TOKEN}" ]]; then
  HF_ENV="HF_TOKEN=${HF_TOKEN} "
fi

echo "Ensuring HotpotQA distractor validation parquet..."
docker exec "$CONTAINER" bash -lc \
  "mkdir -p /data/hotpotqa/distractor && \
   HF_HOME=/mnt/raid/a-ploskin/hf_cache python3 - <<'PY'
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
path = Path('${HOTPOT_PARQUET}')
if not path.exists() or path.stat().st_size < 1_000_000:
    downloaded = hf_hub_download(
        repo_id='hotpot_qa',
        repo_type='dataset',
        filename='distractor/validation/0000.parquet',
        revision='refs/convert/parquet',
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, path)
    print(f'Downloaded HotpotQA validation to {path}')
else:
    print(f'Using existing HotpotQA validation at {path}')
PY"

QUERY_NPY="artifacts/results/retrieval/kilt_e11_hotpotqa_distractor_top100.queries.npy"
EXTRA_ARGS=""
if docker exec "$CONTAINER" test -f "/workspace/projected-token/${QUERY_NPY}"; then
  EXTRA_ARGS="--query-embeddings-path ${QUERY_NPY}"
  echo "Reusing cached query embeddings: ${QUERY_NPY}"
fi

docker exec "$CONTAINER" bash -lc "pkill -9 -f '[s]earch_kilt_popqa.py' || true"
sleep 2
docker exec "$CONTAINER" mkdir -p "$(dirname "${RESULT}")"

echo "Starting HotpotQA distractor top-100 search over KILT shards 0,1,2..."
docker exec -d -w /workspace/projected-token "$CONTAINER" bash -lc \
  "CUDA_VISIBLE_DEVICES=0,1 ${HF_ENV}HF_HOME=/mnt/raid/a-ploskin/hf_cache \
   nohup python3 -u scripts/search_kilt_popqa.py \
     --index-dir ${OUT_DIR} \
     --queries-path ${HOTPOT_PARQUET} \
     --question-col question --id-col id \
     --run-name 'HotpotQA/KILT' \
     --projector-path artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt \
     --compressor-device cuda:0 --decoder-device cuda:1 \
     --top-k 100 --num-threads 32 \
     --output-path ${RESULT} \
     ${EXTRA_ARGS} \
     > ${LOG} 2>&1 &"

echo "Log: ${LOG}"
echo "Results: ${RESULT}"
