#!/bin/bash
# Variant B: LoRA проектор - MLP с LoRA адаптером, OSCAR заморожен

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Use local HuggingFace mirror
export HF_ENDPOINT=https://huggingface.artifactory.s.o3.ru/artifactory/api/huggingfaceml/huggingface-remote
export HF_HUB_ETAG_TIMEOUT=86400
export HF_HUB_DOWNLOAD_TIMEOUT=86400

poetry run python scripts/train_projector.py lora configs/projector_lora.yaml \
    --epochs 3