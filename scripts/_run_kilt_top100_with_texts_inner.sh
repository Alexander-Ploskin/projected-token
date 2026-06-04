#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="/mnt/raid/a-ploskin/kilt_faiss_indexes/e11-pertoken-gated"
PROJECTOR="artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt"
TEXT_CACHE="${OUT_DIR}/kilt_doc_texts.jsonl"
COMMON=(
  python3 -u scripts/search_kilt_popqa.py
  --index-dir "${OUT_DIR}"
  --projector-path "${PROJECTOR}"
  --compressor-device cuda:0
  --decoder-device cuda:1
  --top-k 100
  --num-threads 32
  --kilt-text-cache "${TEXT_CACHE}"
)

run_popqa() {
  local result="artifacts/results/retrieval/kilt_e11_popqa_top100.jsonl"
  local log="${OUT_DIR}/popqa_search.log"
  local query_npy="artifacts/results/retrieval/kilt_e11_popqa_top100.queries.npy"
  local extra=()
  if [[ -f "${query_npy}" ]]; then
    extra=(--query-embeddings-path "${query_npy}")
    echo "[top100-texts] Reusing PopQA query embeddings: ${query_npy}"
  fi
  echo "[top100-texts] PopQA search -> ${result}"
  "${COMMON[@]}" \
    --popqa-path /data/popqa_enriched.parquet \
    --run-name "PopQA/KILT" \
    --output-path "${result}" \
    "${extra[@]}" \
    > "${log}" 2>&1
  echo "[top100-texts] PopQA done. Log: ${log}"
}

run_hotpotqa() {
  local parquet="/data/hotpotqa/distractor/validation.parquet"
  local result="artifacts/results/retrieval/kilt_e11_hotpotqa_distractor_top100.jsonl"
  local log="${OUT_DIR}/hotpotqa_search.log"
  local query_npy="artifacts/results/retrieval/kilt_e11_hotpotqa_distractor_top100.queries.npy"
  mkdir -p /data/hotpotqa/distractor
  HF_HOME=/mnt/raid/a-ploskin/hf_cache python3 - <<'PY'
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
path = Path("/data/hotpotqa/distractor/validation.parquet")
if not path.exists() or path.stat().st_size < 1_000_000:
    downloaded = hf_hub_download(
        repo_id="hotpot_qa",
        repo_type="dataset",
        filename="distractor/validation/0000.parquet",
        revision="refs/convert/parquet",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, path)
    print(f"Downloaded HotpotQA validation to {path}")
else:
    print(f"Using existing HotpotQA validation at {path}")
PY
  local extra=()
  if [[ -f "${query_npy}" ]]; then
    extra=(--query-embeddings-path "${query_npy}")
    echo "[top100-texts] Reusing HotpotQA query embeddings: ${query_npy}"
  fi
  echo "[top100-texts] HotpotQA search -> ${result}"
  "${COMMON[@]}" \
    --queries-path "${parquet}" \
    --question-col question \
    --id-col id \
    --run-name "HotpotQA/KILT" \
    --output-path "${result}" \
    "${extra[@]}" \
    > "${log}" 2>&1
  echo "[top100-texts] HotpotQA done. Log: ${log}"
}

run_popqa
run_hotpotqa
echo "[top100-texts] All done."
