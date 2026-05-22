#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_DIR/artifacts/teacher-embeddings/query-doc-bge-base-full}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-base-en-v1.5}"
MSMARCO_DEVICE="${MSMARCO_DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
CROSS_ENCODER_MODEL_NAME="${CROSS_ENCODER_MODEL_NAME:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
CROSS_ENCODER_THRESHOLD="${CROSS_ENCODER_THRESHOLD:-0.5}"
SCORE_BAND_MIN_MARGIN="${SCORE_BAND_MIN_MARGIN:-0.05}"
SCORE_BAND_MAX_MARGIN="${SCORE_BAND_MAX_MARGIN:-0.30}"
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
  --score-band-min-margin "$SCORE_BAND_MIN_MARGIN" \
  --score-band-max-margin "$SCORE_BAND_MAX_MARGIN" \
  --cross-encoder-model-name "$CROSS_ENCODER_MODEL_NAME" \
  --cross-encoder-threshold "$CROSS_ENCODER_THRESHOLD" \
  $force_flag

echo "[DONE] Full query_distill H5 generation completed"
ls -lh "$OUT_DIR"/*.h5
