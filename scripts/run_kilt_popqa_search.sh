#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="pt-exp-mgpu"
OUT="/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated"
RESULT="artifacts/results/retrieval/kilt_e11_popqa_top100.jsonl"
LOG="/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated/popqa_search.log"
HF_TOKEN="${HF_TOKEN:-}"
HF_ENV=""
if [[ -n "${HF_TOKEN}" ]]; then
  HF_ENV="HF_TOKEN=${HF_TOKEN} "
fi

docker exec "$CONTAINER" bash -lc "pkill -9 -f '[s]earch_kilt_popqa.py' || true"
sleep 2
docker exec "$CONTAINER" mkdir -p "$(dirname "${RESULT}")"

QUERY_NPY="artifacts/results/retrieval/kilt_e11_popqa_top100.queries.npy"
EXTRA_ARGS=""
if docker exec "$CONTAINER" test -f "/workspace/projected-token/${QUERY_NPY}"; then
  EXTRA_ARGS="--query-embeddings-path ${QUERY_NPY}"
  echo "Reusing cached query embeddings: ${QUERY_NPY}"
fi

echo "Starting PopQA top-100 search over KILT shards 0,1,2 (in-RAM, batched)..."
docker exec -d -w /workspace/projected-token "$CONTAINER" bash -lc \
  "CUDA_VISIBLE_DEVICES=0,1 ${HF_ENV}HF_HOME=/mnt/raid/a-ploskin/hf_cache \
   nohup python3 -u scripts/search_kilt_popqa.py \
     --index-dir ${OUT} \
     --popqa-path /data/popqa_enriched.parquet \
     --run-name 'PopQA/KILT' \
     --projector-path artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt \
     --compressor-device cuda:0 --decoder-device cuda:1 \
     --top-k 100 --num-threads 32 \
     --output-path ${RESULT} \
     ${EXTRA_ARGS} \
     > ${LOG} 2>&1 &"

echo "Log: ${LOG}"
echo "Results: ${RESULT}"
