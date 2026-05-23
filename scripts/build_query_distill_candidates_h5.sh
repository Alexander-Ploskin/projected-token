#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_DIR/artifacts/teacher-embeddings/query-doc-bge-base-candidates}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-base-en-v1.5}"
MSMARCO_DEVICE="${MSMARCO_DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BEIR_ROOT="${BEIR_ROOT:-/data/beir}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] beir root: $BEIR_ROOT"
echo "[INFO] msmarco device: $MSMARCO_DEVICE"

force_flag=""
if [[ "$FORCE_REBUILD" == "1" ]]; then
  force_flag="--force"
fi

python -m projected_token data query-distill-teacher-bundle \
  --out-dir "$OUT_DIR" \
  --beir-root "$BEIR_ROOT" \
  --model-name "$MODEL_NAME" \
  --device "$MSMARCO_DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --msmarco-negative-strategy hybrid \
  --msmarco-bm25-rank-start 10 \
  --msmarco-bm25-rank-end 100 \
  --msmarco-dense-rank-start 10 \
  --msmarco-dense-rank-end 100 \
  $force_flag

echo "[DONE] Full query_distill candidates H5 generation completed"
ls -lh "$OUT_DIR"/*.h5
