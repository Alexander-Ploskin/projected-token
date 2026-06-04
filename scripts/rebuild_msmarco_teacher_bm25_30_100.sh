#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_DIR/artifacts/teacher-embeddings/query-doc-bge-base-full}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-base-en-v1.5}"
MSMARCO_DEVICE="${MSMARCO_DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MSMARCO_SAMPLES="${MSMARCO_SAMPLES:-500000}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] msmarco device: $MSMARCO_DEVICE"
echo "[INFO] rebuilding msmarco-hard.h5 with bm25 rank window 30..100"

python -m projected_token data query-doc-teacher-embeddings \
  --output-path "$OUT_DIR/msmarco-hard.h5" \
  --max-samples "$MSMARCO_SAMPLES" \
  --model-name "$MODEL_NAME" \
  --device "$MSMARCO_DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --negative-strategy bm25 \
  --bm25-rank-start 30 \
  --bm25-rank-end 100

echo "[DONE] rebuilt $OUT_DIR/msmarco-hard.h5"
