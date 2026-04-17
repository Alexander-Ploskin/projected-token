#!/bin/bash
# Быстрый тест обучения с ограниченными данными

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Use local HuggingFace mirror
export HF_ENDPOINT=https://huggingface.artifactory.s.o3.ru/artifactory/api/huggingfaceml/huggingface-remote
export HF_HUB_ETAG_TIMEOUT=86400
export HF_HUB_DOWNLOAD_TIMEOUT=86400

VARIANT=${1:-mlp}
EPOCHS=${2:-1}

echo "=========================================="
echo "Quick test: $VARIANT variant, $EPOCHS epoch(s)"
echo "=========================================="

case "$VARIANT" in
    mlp)
        CONFIG="configs/projector_mlp.yaml"
        ;;
    lora)
        CONFIG="configs/projector_lora.yaml"
        ;;
    full)
        CONFIG="configs/projector_full.yaml"
        ;;
    *)
        echo "Unknown variant: $VARIANT"
        echo "Usage: $0 [mlp|lora|full] [epochs]"
        exit 1
        ;;
esac

# Создаем временный конфиг с ограниченными данными
TEMP_CONFIG="/tmp/test_projector_${VARIANT}.yaml"

# Читаем оригинальный конфиг и добавляем лимиты
python3 -c "
import yaml
with open('$CONFIG') as f:
    config = yaml.safe_load(f)

config['max_train_samples'] = 500
config['max_val_samples'] = 10
config['batch_size'] = 32
config['output_dir'] = './checkpoints/test_${VARIANT}'
config['log_dir'] = './logs/test_${VARIANT}'

with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f)
"

echo "Config: $TEMP_CONFIG"
echo "Training samples: 500"
echo "Validation samples: 100"
echo ""

poetry run python scripts/train_projector.py "$VARIANT" "$TEMP_CONFIG" --epochs "$EPOCHS"

echo ""
echo "=========================================="
echo "Test completed!"
echo "Logs: tensorboard --logdir=./logs/test_${VARIANT}"
echo "=========================================="